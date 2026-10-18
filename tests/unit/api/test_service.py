import pytest

from tests.fake_graph import BRIEF, fake_runner, streaming_runner
from wp_content_agent.api import service as service_module
from wp_content_agent.api.service import RunService
from wp_content_agent.errors import RunNotFoundError


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
