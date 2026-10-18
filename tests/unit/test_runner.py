import asyncio
import logging
from typing import Any

import pytest

from tests.fake_graph import BRIEF, DRAFT, fake_runner, usage
from wp_content_agent.errors import LLMError, RunNotFoundError, RunStateError, ToolError
from wp_content_agent.graph import AgentState
from wp_content_agent.runner import checkpoint_serializer
from wp_content_agent.schemas import ApprovalDecision, RunStatus


async def test_start_pauses_for_approval_with_the_draft() -> None:
    runner = fake_runner()

    snapshot = await runner.start(BRIEF, run_id="r1")

    assert snapshot.status == RunStatus.AWAITING_APPROVAL
    assert snapshot.approval is not None
    assert snapshot.approval.title == DRAFT.title
    assert snapshot.approval.tag_names == ["django"]
    assert snapshot.review_passed is True


async def test_start_reports_progress_per_node() -> None:
    runner = fake_runner()
    seen: list[str] = []

    await runner.start(BRIEF, run_id="r1", on_progress=lambda node, update: seen.append(node))

    assert seen == ["plan", "gather_context", "write", "review"]


async def test_start_generates_a_run_id() -> None:
    runner = fake_runner()

    snapshot = await runner.start(BRIEF)

    assert len(snapshot.run_id) == 32


async def test_start_rejects_an_existing_run_id() -> None:
    runner = fake_runner()
    await runner.start(BRIEF, run_id="r1")

    with pytest.raises(RunStateError, match="already exists"):
        await runner.start(BRIEF, run_id="r1")


async def test_approve_creates_the_draft() -> None:
    runner = fake_runner()
    await runner.start(BRIEF, run_id="r1")

    snapshot = await runner.resume("r1", ApprovalDecision(approved=True))

    assert snapshot.status == RunStatus.DRAFT_CREATED
    assert snapshot.wp_post_id == 8
    assert snapshot.approval is None


async def test_reject_ends_without_a_post() -> None:
    runner = fake_runner()
    await runner.start(BRIEF, run_id="r1")

    snapshot = await runner.resume("r1", ApprovalDecision(approved=False, note="off topic"))

    assert snapshot.status == RunStatus.REJECTED
    assert snapshot.wp_post_id is None


async def test_resume_twice_raises_state_error() -> None:
    runner = fake_runner()
    await runner.start(BRIEF, run_id="r1")
    await runner.resume("r1", ApprovalDecision(approved=True))

    with pytest.raises(RunStateError, match="not awaiting approval"):
        await runner.resume("r1", ApprovalDecision(approved=True))


async def test_get_unknown_run_raises_not_found() -> None:
    runner = fake_runner()

    with pytest.raises(RunNotFoundError):
        await runner.get("missing")


async def test_resume_unknown_run_raises_not_found() -> None:
    runner = fake_runner()

    with pytest.raises(RunNotFoundError):
        await runner.resume("missing", ApprovalDecision(approved=True))


async def test_node_failure_marks_run_failed_and_keeps_usage() -> None:
    def write(state: AgentState) -> dict[str, Any]:
        raise LLMError("write: output did not match Draft", usage("write"))

    runner = fake_runner(write=write)

    snapshot = await runner.start(BRIEF, run_id="r1")

    assert snapshot.status == RunStatus.FAILED
    assert snapshot.error == "write: output did not match Draft"
    assert [entry.node for entry in snapshot.usage] == ["plan", "write"]
    assert snapshot.total_cost_usd == 0.5


async def test_failure_in_first_node_marks_run_failed() -> None:
    def plan(state: AgentState) -> dict[str, Any]:
        raise RuntimeError("boom")

    runner = fake_runner(plan=plan)

    snapshot = await runner.start(BRIEF, run_id="r1")

    assert snapshot.status == RunStatus.FAILED
    assert snapshot.error == "internal error (RuntimeError); see the logs"


async def test_failed_run_cannot_be_resumed() -> None:
    def review(state: AgentState) -> dict[str, Any]:
        raise LLMError("review failed")

    runner = fake_runner(review=review)
    await runner.start(BRIEF, run_id="r1")

    with pytest.raises(RunStateError):
        await runner.resume("r1", ApprovalDecision(approved=True))


async def test_snapshot_totals_tokens() -> None:
    runner = fake_runner()

    snapshot = await runner.start(BRIEF, run_id="r1")

    assert snapshot.total_tokens == 15


async def test_publish_failure_after_approval_leaves_nothing_pending() -> None:
    def publish_draft(state: AgentState) -> dict[str, Any]:
        raise ToolError("create_draft_post failed: timeout")

    runner = fake_runner(publish_draft=publish_draft)
    await runner.start(BRIEF, run_id="r1")

    snapshot = await runner.resume("r1", ApprovalDecision(approved=True))

    assert snapshot.status == RunStatus.FAILED
    assert snapshot.approval is None
    assert (await runner._state("r1")).next == ()


async def test_failure_before_approval_leaves_nothing_pending() -> None:
    def write(state: AgentState) -> dict[str, Any]:
        raise LLMError("write failed")

    runner = fake_runner(write=write)

    await runner.start(BRIEF, run_id="r1")

    assert (await runner._state("r1")).next == ()


async def test_raising_progress_callback_does_not_fail_the_run() -> None:
    def broken(node: str, update: Any) -> None:
        raise BrokenPipeError

    runner = fake_runner()

    snapshot = await runner.start(BRIEF, run_id="r1", on_progress=broken)

    assert snapshot.status == RunStatus.AWAITING_APPROVAL


async def test_internal_error_message_hides_details() -> None:
    def plan(state: AgentState) -> dict[str, Any]:
        raise RuntimeError("https://user:secret@example.com")

    runner = fake_runner(plan=plan)

    snapshot = await runner.start(BRIEF, run_id="r1")

    assert snapshot.error is not None
    assert "secret" not in snapshot.error


async def test_cancelled_run_is_marked_failed() -> None:
    started = asyncio.Event()

    async def write(state: AgentState) -> dict[str, Any]:
        started.set()
        await asyncio.Event().wait()
        return {}

    runner = fake_runner(write=write)
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

    runner = fake_runner(publish_draft=publish_draft)
    await runner.start(BRIEF, run_id="r1")

    results = await asyncio.gather(
        runner.resume("r1", ApprovalDecision(approved=True)),
        runner.resume("r1", ApprovalDecision(approved=True)),
        return_exceptions=True,
    )

    assert published == 1
    assert sum(isinstance(result, RunStateError) for result in results) == 1


def test_checkpoint_serializer_loads_state_models_without_warnings(
    caplog: pytest.LogCaptureFixture,
) -> None:
    serde = checkpoint_serializer()
    values = {"brief": BRIEF, "draft": DRAFT, "usage": [usage("plan")], "status": RunStatus.FAILED}

    with caplog.at_level(logging.WARNING):
        loaded = serde.loads_typed(serde.dumps_typed(values))

    assert loaded == values
    assert caplog.records == []
