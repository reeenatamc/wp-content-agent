import asyncio
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from tests.fakes import fake_client
from wp_content_agent.errors import LLMError, RunNotFoundError, RunStateError, ToolError
from wp_content_agent.graph import AgentContext, AgentState, build_graph, next_revision_count
from wp_content_agent.nodes import NODES
from wp_content_agent.runner import AgentRunner
from wp_content_agent.schemas import (
    ApprovalDecision,
    Brief,
    Draft,
    NodeUsage,
    Review,
    RunStatus,
)
from wp_content_agent.settings import Settings

BRIEF = Brief(topic="Caching in Django", audience="Backend developers")
DRAFT = Draft(
    title="Caching Django querysets",
    slug="caching-django-querysets",
    excerpt="",
    content_html="<p>x</p>",
    meta_description="m",
    tag_names=["django"],
)


def _usage(node: str) -> NodeUsage:
    return NodeUsage(node=node, model="m", input_tokens=10, output_tokens=5, cost_usd=0.25)


def _nodes(**overrides: Any) -> dict[str, Any]:
    """Fake model and WordPress nodes around the real human_approval node."""

    def plan(state: AgentState) -> dict[str, Any]:
        return {"status": RunStatus.GATHERING, "usage": [_usage("plan")]}

    def gather_context(state: AgentState) -> dict[str, Any]:
        return {"status": RunStatus.WRITING}

    def write(state: AgentState) -> dict[str, Any]:
        return {"draft": DRAFT, "revision_count": next_revision_count(state)}

    def review(state: AgentState) -> dict[str, Any]:
        return {
            "review": Review(passed=True, checks=[], fixes=[]),
            "status": RunStatus.AWAITING_APPROVAL,
        }

    def publish_draft(state: AgentState) -> dict[str, Any]:
        return {
            "status": RunStatus.DRAFT_CREATED,
            "wp_post_id": 8,
            "edit_url": "https://wp.test/wp-admin/post.php?post=8&action=edit",
        }

    nodes: dict[str, Any] = {
        "plan": plan,
        "gather_context": gather_context,
        "write": write,
        "review": review,
        "human_approval": NODES["human_approval"],
        "publish_draft": publish_draft,
    }
    return {**nodes, **overrides}


def _runner(**overrides: Any) -> AgentRunner:
    context = AgentContext(
        settings=Settings(_env_file=None),  # type: ignore[call-arg]
        writer=fake_client(),
        reviewer=fake_client(),
        wordpress=object(),  # type: ignore[arg-type]
    )
    graph = build_graph(_nodes(**overrides), checkpointer=InMemorySaver())  # type: ignore[arg-type]
    return AgentRunner(graph, context)


async def test_start_pauses_for_approval_with_the_draft() -> None:
    runner = _runner()

    snapshot = await runner.start(BRIEF, run_id="r1")

    assert snapshot.status == RunStatus.AWAITING_APPROVAL
    assert snapshot.approval is not None
    assert snapshot.approval.title == DRAFT.title
    assert snapshot.approval.tag_names == ["django"]
    assert snapshot.review_passed is True


async def test_start_reports_progress_per_node() -> None:
    runner = _runner()
    seen: list[str] = []

    await runner.start(BRIEF, run_id="r1", on_progress=lambda node, update: seen.append(node))

    assert seen == ["plan", "gather_context", "write", "review"]


async def test_start_generates_a_run_id() -> None:
    runner = _runner()

    snapshot = await runner.start(BRIEF)

    assert len(snapshot.run_id) == 32


async def test_start_rejects_an_existing_run_id() -> None:
    runner = _runner()
    await runner.start(BRIEF, run_id="r1")

    with pytest.raises(RunStateError, match="already exists"):
        await runner.start(BRIEF, run_id="r1")


async def test_approve_creates_the_draft() -> None:
    runner = _runner()
    await runner.start(BRIEF, run_id="r1")

    snapshot = await runner.resume("r1", ApprovalDecision(approved=True))

    assert snapshot.status == RunStatus.DRAFT_CREATED
    assert snapshot.wp_post_id == 8
    assert snapshot.approval is None


async def test_reject_ends_without_a_post() -> None:
    runner = _runner()
    await runner.start(BRIEF, run_id="r1")

    snapshot = await runner.resume("r1", ApprovalDecision(approved=False, note="off topic"))

    assert snapshot.status == RunStatus.REJECTED
    assert snapshot.wp_post_id is None


async def test_resume_twice_raises_state_error() -> None:
    runner = _runner()
    await runner.start(BRIEF, run_id="r1")
    await runner.resume("r1", ApprovalDecision(approved=True))

    with pytest.raises(RunStateError, match="not awaiting approval"):
        await runner.resume("r1", ApprovalDecision(approved=True))


async def test_get_unknown_run_raises_not_found() -> None:
    runner = _runner()

    with pytest.raises(RunNotFoundError):
        await runner.get("missing")


async def test_resume_unknown_run_raises_not_found() -> None:
    runner = _runner()

    with pytest.raises(RunNotFoundError):
        await runner.resume("missing", ApprovalDecision(approved=True))


async def test_node_failure_marks_run_failed_and_keeps_usage() -> None:
    def write(state: AgentState) -> dict[str, Any]:
        raise LLMError("write: output did not match Draft", _usage("write"))

    runner = _runner(write=write)

    snapshot = await runner.start(BRIEF, run_id="r1")

    assert snapshot.status == RunStatus.FAILED
    assert snapshot.error == "write: output did not match Draft"
    assert [entry.node for entry in snapshot.usage] == ["plan", "write"]
    assert snapshot.total_cost_usd == 0.5


async def test_failure_in_first_node_marks_run_failed() -> None:
    def plan(state: AgentState) -> dict[str, Any]:
        raise RuntimeError("boom")

    runner = _runner(plan=plan)

    snapshot = await runner.start(BRIEF, run_id="r1")

    assert snapshot.status == RunStatus.FAILED
    assert snapshot.error == "internal error (RuntimeError); see the logs"


async def test_failed_run_cannot_be_resumed() -> None:
    def review(state: AgentState) -> dict[str, Any]:
        raise LLMError("review failed")

    runner = _runner(review=review)
    await runner.start(BRIEF, run_id="r1")

    with pytest.raises(RunStateError):
        await runner.resume("r1", ApprovalDecision(approved=True))


async def test_snapshot_totals_tokens() -> None:
    runner = _runner()

    snapshot = await runner.start(BRIEF, run_id="r1")

    assert snapshot.total_tokens == 15


async def test_publish_failure_after_approval_leaves_nothing_pending() -> None:
    def publish_draft(state: AgentState) -> dict[str, Any]:
        raise ToolError("create_draft_post failed: timeout")

    runner = _runner(publish_draft=publish_draft)
    await runner.start(BRIEF, run_id="r1")

    snapshot = await runner.resume("r1", ApprovalDecision(approved=True))

    assert snapshot.status == RunStatus.FAILED
    assert snapshot.approval is None
    assert (await runner._state("r1")).next == ()


async def test_failure_before_approval_leaves_nothing_pending() -> None:
    def write(state: AgentState) -> dict[str, Any]:
        raise LLMError("write failed")

    runner = _runner(write=write)

    await runner.start(BRIEF, run_id="r1")

    assert (await runner._state("r1")).next == ()


async def test_raising_progress_callback_does_not_fail_the_run() -> None:
    def broken(node: str, update: Any) -> None:
        raise BrokenPipeError

    runner = _runner()

    snapshot = await runner.start(BRIEF, run_id="r1", on_progress=broken)

    assert snapshot.status == RunStatus.AWAITING_APPROVAL


async def test_internal_error_message_hides_details() -> None:
    def plan(state: AgentState) -> dict[str, Any]:
        raise RuntimeError("https://user:secret@example.com")

    runner = _runner(plan=plan)

    snapshot = await runner.start(BRIEF, run_id="r1")

    assert snapshot.error is not None
    assert "secret" not in snapshot.error


async def test_cancelled_run_is_marked_failed() -> None:
    started = asyncio.Event()

    async def write(state: AgentState) -> dict[str, Any]:
        started.set()
        await asyncio.Event().wait()
        return {}

    runner = _runner(write=write)
    task = asyncio.create_task(runner.start(BRIEF, run_id="r1"))
    await started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert (await runner.get("r1")).status == RunStatus.FAILED


async def test_concurrent_approvals_publish_once() -> None:
    published = 0

    async def publish_draft(state: AgentState) -> dict[str, Any]:
        nonlocal published
        published += 1
        await asyncio.sleep(0)
        return {"status": RunStatus.DRAFT_CREATED, "wp_post_id": 8}

    runner = _runner(publish_draft=publish_draft)
    await runner.start(BRIEF, run_id="r1")

    results = await asyncio.gather(
        runner.resume("r1", ApprovalDecision(approved=True)),
        runner.resume("r1", ApprovalDecision(approved=True)),
        return_exceptions=True,
    )

    assert published == 1
    assert sum(isinstance(result, RunStateError) for result in results) == 1
