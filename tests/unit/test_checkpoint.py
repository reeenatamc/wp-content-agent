from pathlib import Path

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import ValidationError

from tests.fake_graph import fake_nodes, flaky_publish
from tests.fakes import fake_client
from wp_content_agent.graph import AgentContext, build_graph
from wp_content_agent.runner import AgentRunner, checkpoint_serializer
from wp_content_agent.schemas import ApprovalDecision, Brief, RunStatus
from wp_content_agent.settings import Settings


async def test_run_with_source_urls_survives_a_sqlite_checkpoint(tmp_path: Path) -> None:
    # InMemorySaver in the other runner tests does not serialize, so it cannot catch
    # a state value that the real checkpointer fails to write or read back.
    brief = Brief(topic="Caching in Django", audience="devs", source_urls=["https://a.test/x"])
    context = AgentContext(
        settings=Settings(_env_file=None),  # type: ignore[call-arg]
        writer=fake_client(),
        reviewer=fake_client(),
        wordpress=object(),  # type: ignore[arg-type]
    )

    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "c.sqlite")) as saver:
        saver.serde = checkpoint_serializer()
        graph = build_graph(fake_nodes(), checkpointer=saver)  # type: ignore[arg-type]
        runner = AgentRunner(graph, context)
        pending = await runner.start(brief, run_id="r")
        done = await runner.resume("r", ApprovalDecision(approved=True))

    assert (pending.status, pending.error) == (RunStatus.AWAITING_APPROVAL, None)
    assert done.status == RunStatus.DRAFT_CREATED


async def test_retry_runs_only_publish_with_a_sqlite_checkpoint(tmp_path: Path) -> None:
    brief = Brief(topic="Caching in Django", audience="devs")
    context = AgentContext(
        settings=Settings(_env_file=None),  # type: ignore[call-arg]
        writer=fake_client(),
        reviewer=fake_client(),
        wordpress=object(),  # type: ignore[arg-type]
    )
    visited: list[str] = []

    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "c.sqlite")) as saver:
        saver.serde = checkpoint_serializer()
        nodes = fake_nodes(publish_draft=flaky_publish())
        graph = build_graph(nodes, checkpointer=saver)  # type: ignore[arg-type]
        runner = AgentRunner(graph, context)
        await runner.start(brief, run_id="r")
        failed = await runner.resume("r", ApprovalDecision(approved=True))
        done = await runner.retry("r", on_progress=lambda node, update: visited.append(node))

    assert (failed.status, failed.can_retry) == (RunStatus.FAILED, True)
    assert visited == ["publish_draft"]
    assert (done.status, done.wp_post_id, done.error) == (RunStatus.DRAFT_CREATED, 8, None)
    assert done.usage == failed.usage


def test_brief_source_urls_are_validated_and_normalized() -> None:
    brief = Brief(topic="abc", audience="devs", source_urls=["https://A.test"])

    assert brief.source_urls == ["https://a.test/"]


def test_brief_rejects_a_non_http_url() -> None:
    with pytest.raises(ValidationError):
        Brief(topic="abc", audience="devs", source_urls=["ftp://a.test/x"])
