from collections import Counter
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt

from tests.fakes import fake_client
from wp_content_agent.graph import AgentContext, AgentState, build_graph
from wp_content_agent.schemas import Brief, Review, RunStatus
from wp_content_agent.settings import Settings


class FakeNodes:
    """Minimal nodes that record calls and let each test choose the review outcome."""

    def __init__(self, review_passes: list[bool]) -> None:
        self.calls: Counter[str] = Counter()
        self._review_passes = review_passes

    def mapping(self) -> dict[str, Any]:
        return {
            "plan": self._record("plan", {"status": RunStatus.GATHERING}),
            "gather_context": self._record("gather_context", {"status": RunStatus.WRITING}),
            "write": self.write,
            "review": self.review,
            "human_approval": self.human_approval,
            "publish_draft": self._record(
                "publish_draft", {"status": RunStatus.DRAFT_CREATED, "wp_post_id": 7}
            ),
        }

    def _record(self, name: str, update: dict[str, Any]) -> Any:
        def node(state: AgentState) -> dict[str, Any]:
            self.calls[name] += 1
            return update

        return node

    def write(self, state: AgentState) -> dict[str, Any]:
        self.calls["write"] += 1
        revision = state.get("revision_count", 0) + (1 if "review" in state else 0)
        return {"revision_count": revision}

    def review(self, state: AgentState) -> dict[str, Any]:
        self.calls["review"] += 1
        passed = self._review_passes.pop(0)
        return {"review": Review(passed=passed, checks=[], fixes=[])}

    def human_approval(self, state: AgentState) -> dict[str, Any]:
        self.calls["human_approval"] += 1
        decision = interrupt({"revision_count": state.get("revision_count", 0)})
        status = RunStatus.PUBLISHING if decision["approved"] else RunStatus.REJECTED
        return {"status": status}


@pytest.fixture
def context() -> AgentContext:
    return AgentContext(
        settings=Settings(_env_file=None, max_revisions=2),
        writer=fake_client(),
        reviewer=fake_client(),
        wordpress=object(),  # type: ignore[arg-type]
    )


def _start(nodes: FakeNodes) -> tuple[Any, dict[str, Any]]:
    graph = build_graph(nodes.mapping(), checkpointer=InMemorySaver())  # type: ignore[arg-type]
    config = {"configurable": {"thread_id": "t1"}}
    return graph, config


BRIEF = {"brief": Brief(topic="Caching in Django", audience="Backend developers")}


async def test_passing_review_pauses_for_approval_then_publishes(context: AgentContext) -> None:
    nodes = FakeNodes(review_passes=[True])
    graph, config = _start(nodes)

    paused = await graph.ainvoke(BRIEF, config, context=context)
    done = await graph.ainvoke(Command(resume={"approved": True}), config, context=context)

    assert "__interrupt__" in paused
    assert nodes.calls["write"] == 1
    assert done["status"] == RunStatus.DRAFT_CREATED
    assert nodes.calls["publish_draft"] == 1


async def test_failing_review_stops_after_max_revisions(context: AgentContext) -> None:
    nodes = FakeNodes(review_passes=[False, False, False])
    graph, config = _start(nodes)

    paused = await graph.ainvoke(BRIEF, config, context=context)

    assert "__interrupt__" in paused
    assert nodes.calls["write"] == 3
    assert paused["revision_count"] == 2


async def test_review_loop_exits_as_soon_as_review_passes(context: AgentContext) -> None:
    nodes = FakeNodes(review_passes=[False, True])
    graph, config = _start(nodes)

    await graph.ainvoke(BRIEF, config, context=context)

    assert nodes.calls["write"] == 2
    assert nodes.calls["review"] == 2


async def test_rejection_ends_without_publishing(context: AgentContext) -> None:
    nodes = FakeNodes(review_passes=[True])
    graph, config = _start(nodes)

    await graph.ainvoke(BRIEF, config, context=context)
    done = await graph.ainvoke(Command(resume={"approved": False}), config, context=context)

    assert done["status"] == RunStatus.REJECTED
    assert nodes.calls["publish_draft"] == 0


def test_build_graph_rejects_missing_nodes() -> None:
    with pytest.raises(ValueError, match="publish_draft"):
        build_graph({"plan": lambda state: {}})  # type: ignore[dict-item]
