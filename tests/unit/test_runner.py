import asyncio
from typing import Any

import pytest
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.types import Command

from tests.fake_graph import BRIEF, DRAFT, fake_runner, flaky_publish, usage
from wp_content_agent.errors import LLMError, RunNotFoundError, RunStateError, ToolError
from wp_content_agent.graph import AgentState, next_revision_count
from wp_content_agent.runner import AgentRunner, _config, _snapshot, checkpoint_serializer
from wp_content_agent.schemas import (
    ApprovalDecision,
    Brief,
    CheckResult,
    Outline,
    OutlineSection,
    RelatedPost,
    ResearchNotes,
    Review,
    RunSnapshot,
    RunStatus,
    SourceNote,
    Term,
)


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


async def _failed_at_publish(failures: int = 1) -> tuple[AgentRunner, RunSnapshot]:
    runner = fake_runner(publish_draft=flaky_publish(failures))
    await runner.start(BRIEF, run_id="r1")
    return runner, await runner.resume("r1", ApprovalDecision(approved=True))


async def test_publish_failure_can_be_retried() -> None:
    runner, failed = await _failed_at_publish()

    snapshot = await runner.retry("r1")

    assert failed.can_retry is True
    assert snapshot.status == RunStatus.DRAFT_CREATED
    assert snapshot.wp_post_id == 8
    assert snapshot.error is None
    assert snapshot.can_retry is False


async def test_retry_that_died_mid_way_can_be_retried_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, _ = await _failed_at_publish(failures=2)

    async def died(*args: Any) -> None:
        # The process is gone, so nothing marks the run failed.
        raise KeyboardInterrupt

    monkeypatch.setattr(runner, "_mark_failed", died)
    with pytest.raises(KeyboardInterrupt):
        await runner.retry("r1")
    monkeypatch.undo()

    stuck = await runner.get("r1")
    snapshot = await runner.retry("r1")

    assert (stuck.status, stuck.can_retry) == (RunStatus.PUBLISHING, True)
    assert snapshot.status == RunStatus.DRAFT_CREATED


async def test_retry_runs_only_publish_and_spends_no_model_tokens() -> None:
    runner, failed = await _failed_at_publish()
    progress: list[str] = []

    snapshot = await runner.retry("r1", on_progress=lambda node, update: progress.append(node))

    assert progress == ["publish_draft"]
    assert snapshot.usage == failed.usage


async def test_retry_can_fail_again_and_be_retried_again() -> None:
    runner, _ = await _failed_at_publish(failures=2)

    second = await runner.retry("r1")
    third = await runner.retry("r1")

    assert (second.status, second.can_retry) == (RunStatus.FAILED, True)
    assert third.status == RunStatus.DRAFT_CREATED


async def test_a_later_pass_through_the_retried_node_is_not_retryable() -> None:
    calls = 0

    def review(state: AgentState) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise LLMError("review failed", usage("review"))
        passed = calls > 2
        return {
            "review": Review(passed=passed, checks=[], fixes=[]),
            "status": RunStatus.AWAITING_APPROVAL if passed else RunStatus.WRITING,
            "usage": [usage("review")],
        }

    runner = fake_runner(review=review)
    await runner.start(BRIEF, run_id="r1")
    await runner.retry("r1")

    # Every checkpoint after the retried review ran, including the second pass through it.
    after_fork = []
    async for state in runner._graph.aget_state_history(_config("r1")):
        if state.metadata and state.metadata.get("source") == "fork":
            break
        after_fork.append(_snapshot("r1", state))

    assert any(state.status == RunStatus.WRITING for state in after_fork)
    assert not any(state.can_retry for state in after_fork)


async def test_retry_that_fails_before_forking_stays_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, _ = await _failed_at_publish()

    def broken(*args: Any, **kwargs: Any) -> Any:
        raise OSError("checkpoint file is gone")

    monkeypatch.setattr(runner._graph, "astream", broken)
    snapshot = await runner.retry("r1")

    assert (snapshot.status, snapshot.failed_node, snapshot.can_retry) == (
        RunStatus.FAILED,
        "publish_draft",
        True,
    )


def _failing_once(node: str, error: Exception, success: dict[str, Any] | None = None) -> Any:
    """A node that raises `error` on its first call, then bills one entry and returns `success`."""
    calls = 0

    def run(state: AgentState) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise error
        return {**(success or {}), "usage": [usage(node)]}

    return run


async def _retry_with_progress(runner: AgentRunner) -> tuple[RunSnapshot, list[str]]:
    progress: list[str] = []
    snapshot = await runner.retry("r1", on_progress=lambda node, update: progress.append(node))
    return snapshot, progress


async def test_retry_of_the_first_node_runs_the_whole_graph_once() -> None:
    plan = _failing_once("plan", LLMError("plan failed", usage("plan")))
    runner = fake_runner(plan=plan)
    failed = await runner.start(BRIEF, run_id="r1")

    snapshot, progress = await _retry_with_progress(runner)

    assert (failed.status, failed.can_retry, failed.failed_node) == (RunStatus.FAILED, True, "plan")
    assert progress == ["plan", "gather_context", "write", "review"]
    assert (snapshot.status, snapshot.error, snapshot.can_retry) == (
        RunStatus.AWAITING_APPROVAL,
        None,
        False,
    )
    assert [entry.node for entry in snapshot.usage] == ["plan", "plan"]


async def test_retry_of_a_node_without_usage_adds_nothing_to_the_cost() -> None:
    runner = fake_runner(
        gather_context=_failing_once(
            "gather_context", ToolError("x"), {"status": RunStatus.WRITING}
        )
    )
    await runner.start(BRIEF, run_id="r1")

    snapshot, progress = await _retry_with_progress(runner)

    assert progress == ["gather_context", "write", "review"]
    assert [entry.node for entry in snapshot.usage] == ["plan", "gather_context"]


async def test_retry_of_review_does_not_write_again() -> None:
    review = _failing_once(
        "review",
        LLMError("review failed", usage("review")),
        {"review": Review(passed=True, checks=[], fixes=[]), "status": RunStatus.AWAITING_APPROVAL},
    )
    runner = fake_runner(review=review)
    await runner.start(BRIEF, run_id="r1")

    snapshot, progress = await _retry_with_progress(runner)

    assert progress == ["review"]
    assert snapshot.status == RunStatus.AWAITING_APPROVAL
    assert [entry.node for entry in snapshot.usage] == ["plan", "review", "review"]


async def test_retry_of_write_inside_the_review_loop_keeps_the_earlier_revisions() -> None:
    calls = 0

    def write(state: AgentState) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise LLMError("write failed", usage("write"))
        return {
            "draft": DRAFT,
            "revision_count": next_revision_count(state),
            "usage": [usage("write")],
        }

    def review(state: AgentState) -> dict[str, Any]:
        passed = state["revision_count"] > 0
        return {
            "review": Review(passed=passed, checks=[], fixes=[]),
            "status": RunStatus.AWAITING_APPROVAL,
            "usage": [usage("review")],
        }

    runner = fake_runner(write=write, review=review)
    failed = await runner.start(BRIEF, run_id="r1")

    snapshot, progress = await _retry_with_progress(runner)

    assert failed.failed_node == "write"
    assert progress == ["write", "review"]
    assert (snapshot.status, snapshot.revision_count) == (RunStatus.AWAITING_APPROVAL, 1)
    nodes = [entry.node for entry in snapshot.usage]
    assert nodes == ["plan", "write", "review", "write", "write", "review"]


async def test_usage_of_a_retry_that_fails_again_is_counted_once_per_attempt() -> None:
    attempts = iter([LLMError("one", usage("review")), LLMError("two", usage("review"))])

    def review(state: AgentState) -> dict[str, Any]:
        for error in attempts:
            raise error
        return {
            "review": Review(passed=True, checks=[], fixes=[]),
            "status": RunStatus.AWAITING_APPROVAL,
            "usage": [usage("review")],
        }

    runner = fake_runner(review=review)
    await runner.start(BRIEF, run_id="r1")

    second = await runner.retry("r1")
    third = await runner.retry("r1")

    assert (second.status, second.can_retry) == (RunStatus.FAILED, True)
    assert [entry.node for entry in second.usage] == ["plan", "review", "review"]
    assert [entry.node for entry in third.usage] == ["plan", "review", "review", "review"]
    assert third.status == RunStatus.AWAITING_APPROVAL


async def test_invalid_approval_decision_can_be_retried_into_a_new_approval() -> None:
    runner = fake_runner()
    await runner.start(BRIEF, run_id="r1")
    await runner._drive(Command(resume={"approved": "perhaps"}), "r1", None)
    failed = await runner.get("r1")

    waiting = await runner.retry("r1")
    done = await runner.resume("r1", ApprovalDecision(approved=True))

    assert (failed.status, failed.failed_node, failed.can_retry) == (
        RunStatus.FAILED,
        "human_approval",
        True,
    )
    assert (waiting.status, waiting.can_retry) == (RunStatus.AWAITING_APPROVAL, False)
    assert waiting.approval is not None
    assert (done.status, done.wp_post_id) == (RunStatus.DRAFT_CREATED, 8)


async def test_retry_refuses_a_run_that_was_never_marked_failed() -> None:
    runner = fake_runner()
    await runner.start(BRIEF, run_id="r1")
    before = await runner._checkpoint_before("r1", "human_approval")
    assert before is not None
    # A failed_node left over while the run waits for approval must not offer a retry.
    await runner._drive(
        Command(update={"failed_node": "human_approval"}), "r1", None, start_from=before.config
    )

    snapshot = await runner.get("r1")

    assert (snapshot.status, snapshot.can_retry) == (RunStatus.AWAITING_APPROVAL, False)


async def test_retry_refuses_a_post_that_came_back_non_draft() -> None:
    def publish_draft(state: AgentState) -> dict[str, Any]:
        return {"status": RunStatus.FAILED, "error": "not a draft", "wp_post_id": 8}

    runner = fake_runner(publish_draft=publish_draft)
    await runner.start(BRIEF, run_id="r1")
    failed = await runner.resume("r1", ApprovalDecision(approved=True))

    with pytest.raises(RunStateError, match="cannot be retried"):
        await runner.retry("r1")

    assert failed.can_retry is False


async def test_retry_refuses_a_failed_publish_that_already_has_a_post() -> None:
    runner, _ = await _failed_at_publish()
    await runner._graph.aupdate_state(_config("r1"), {"wp_post_id": 8}, as_node="publish_draft")

    with pytest.raises(RunStateError, match="cannot be retried"):
        await runner.retry("r1")


async def test_retry_refuses_runs_that_are_waiting_or_done() -> None:
    runner = fake_runner()
    waiting = await runner.start(BRIEF, run_id="r1")

    with pytest.raises(RunStateError, match="cannot be retried"):
        await runner.retry("r1")
    done = await runner.resume("r1", ApprovalDecision(approved=True))
    with pytest.raises(RunStateError, match="cannot be retried"):
        await runner.retry("r1")

    assert (waiting.can_retry, done.can_retry) == (False, False)


async def test_retry_unknown_run_raises_not_found() -> None:
    runner = fake_runner()

    with pytest.raises(RunNotFoundError):
        await runner.retry("missing")


async def test_failure_records_the_node_that_raised() -> None:
    def write(state: AgentState) -> dict[str, Any]:
        raise LLMError("write failed")

    runner = fake_runner(write=write)

    await runner.start(BRIEF, run_id="r1")

    assert (await runner._state("r1")).values["failed_node"] == "write"


def _full_state() -> dict[str, Any]:
    brief = Brief.model_validate(
        {**BRIEF.model_dump(), "source_urls": ["https://docs.example.com/cache/"]}
    )
    return {
        "brief": brief,
        "outline": Outline(
            working_title="t",
            angle="a",
            sections=[
                OutlineSection(heading="h", level=2, key_points=["p"]),
                OutlineSection(heading="i", level=3, key_points=["q"]),
            ],
        ),
        "context": ResearchNotes(
            sources=[SourceNote(url="https://docs.example.com/cache/", title="t", facts=["f"])],
            related_posts=[RelatedPost(id=1, title="t", link="https://x.test/t")],
            categories=[Term(id=1, name="Backend", slug="backend")],
        ),
        "draft": DRAFT,
        "review": Review(
            passed=False, checks=[CheckResult(name="readability", passed=False)], fixes=["f"]
        ),
        "usage": [usage("plan")],
        "status": RunStatus.AWAITING_APPROVAL,
    }


def test_checkpoint_serializer_round_trips_every_state_type() -> None:
    serde = checkpoint_serializer()
    values = _full_state()

    loaded = serde.loads_typed(serde.dumps_typed(values))

    assert loaded == values


def test_checkpoint_serializer_blocks_types_outside_the_state() -> None:
    # Guards the test above: with an allowlist, unknown types do not load back as models.
    serde = JsonPlusSerializer(allowed_msgpack_modules=[RunStatus])

    loaded = serde.loads_typed(serde.dumps_typed({"draft": DRAFT}))

    assert loaded != {"draft": DRAFT}


def _revise(note: str = "shorter intro") -> ApprovalDecision:
    return ApprovalDecision(approved=False, revise=True, note=note)


async def test_revise_writes_again_and_stops_at_approval() -> None:
    writes: list[str] = []

    def write(state: AgentState) -> dict[str, Any]:
        writes.append(state.get("human_feedback", ""))
        return {"draft": DRAFT, "revision_count": next_revision_count(state)}

    runner = fake_runner(write=write)
    await runner.start(BRIEF, run_id="r1")

    snapshot = await runner.resume("r1", _revise())

    assert writes == ["", "shorter intro"]
    assert snapshot.status == RunStatus.AWAITING_APPROVAL
    assert snapshot.approval is not None
    assert snapshot.approval.human_revisions_left == 1


async def test_revise_with_no_revisions_left_is_refused_before_anything_runs() -> None:
    writes = 0

    def write(state: AgentState) -> dict[str, Any]:
        nonlocal writes
        writes += 1
        return {"draft": DRAFT, "revision_count": next_revision_count(state)}

    runner = fake_runner(write=write)
    await runner.start(BRIEF, run_id="r1")
    await runner.resume("r1", _revise())
    await runner.resume("r1", _revise())

    with pytest.raises(RunStateError, match="no human revisions left"):
        await runner.resume("r1", _revise())

    assert writes == 3
    assert (await runner.get("r1")).status == RunStatus.AWAITING_APPROVAL
