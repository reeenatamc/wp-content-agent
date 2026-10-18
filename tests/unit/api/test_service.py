import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from tests.fake_graph import BRIEF, fake_runner, streaming_runner
from wp_content_agent.api import service as service_module
from wp_content_agent.api.service import RunEvent, RunService
from wp_content_agent.errors import RunNotFoundError, RunStateError
from wp_content_agent.graph import AgentState
from wp_content_agent.schemas import ApprovalDecision, RunSnapshot


async def test_events_of_a_live_run_go_snapshot_nodes_tokens_end() -> None:
    service = RunService(streaming_runner(), max_runs=1)
    await service.start(BRIEF, run_id="r1")

    events = [event async for event in service.events("r1")]

    kinds = [event.type for event in events]
    assert (kinds[0], kinds[-1]) == ("snapshot", "end")
    assert set(kinds[1:-1]) == {"node", "token"}
    assert kinds.index("node") < kinds.index("token")
    assert events[-1].data["status"] == "awaiting_approval"
    assert next(e.data["node"] for e in events if e.type == "token") == "write"
    assert not service._subscribers


async def test_events_of_a_finished_run_are_a_snapshot_and_the_end() -> None:
    service = RunService(fake_runner(), max_runs=1)
    await service.start(BRIEF, run_id="r1")
    await service.wait("r1")

    events = [event async for event in service.events("r1")]

    assert [event.type for event in events] == ["snapshot", "end"]
    assert events[1].data["run_id"] == "r1"


async def test_events_of_an_unknown_run_raise_not_found() -> None:
    service = RunService(fake_runner(), max_runs=1)

    with pytest.raises(RunNotFoundError):
        await anext(service.events("nope"))

    assert not service._subscribers


async def test_a_slow_subscriber_is_dropped_and_its_stream_ends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(service_module, "SUBSCRIBER_QUEUE_SIZE", 1)
    service = RunService(streaming_runner(), max_runs=1)
    await service.start(BRIEF, run_id="r1")
    events = service.events("r1")
    first = await anext(events)

    await service.wait("r1")
    rest = [event async for event in events]

    assert first.type == "snapshot"
    assert rest == []
    assert not service._subscribers


async def test_a_resume_refused_after_its_claim_ends_the_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = fake_runner()
    service = RunService(runner, max_runs=1)
    await service.start(BRIEF, run_id="r1")
    await service.wait("r1")
    await service.resume("r1", ApprovalDecision(approved=False))
    await service.wait("r1")
    real_get, gate = runner.get, asyncio.Event()

    async def slow_get(run_id: str) -> RunSnapshot:
        snapshot = await real_get(run_id)
        await gate.wait()
        return snapshot

    # The resume claims the run, then waits on its state check; a client subscribes then.
    monkeypatch.setattr(runner, "get", slow_get)
    refused = asyncio.create_task(service.resume("r1", ApprovalDecision(approved=True)))
    await asyncio.sleep(0)
    events = service.events("r1")
    first = asyncio.create_task(anext(events))
    await asyncio.sleep(0)
    gate.set()

    with pytest.raises(RunStateError):
        await refused
    # Bounded, so a stream left open fails the test instead of hanging it.
    rest = await asyncio.wait_for(_collect(events), timeout=5)

    assert (await first).type == "snapshot"
    assert [event.type for event in rest] == ["end"]
    assert not service._subscribers


async def test_a_stream_open_at_shutdown_ends_with_the_failed_run() -> None:
    async def stuck(state: AgentState) -> dict[str, Any]:
        await asyncio.Event().wait()
        return {}

    service = RunService(fake_runner(plan=stuck), max_runs=1)
    await service.start(BRIEF, run_id="r1")
    events = service.events("r1")
    await anext(events)
    reader = asyncio.create_task(anext(events))
    await asyncio.sleep(0)

    await service.aclose()
    end = await reader

    assert end.type == "end"
    assert end.data["status"] == "failed"


async def _collect(events: AsyncIterator[RunEvent]) -> list[RunEvent]:
    return [event async for event in events]
