from collections import Counter
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt

from tests.fakes import fake_client
from wp_content_agent.graph import AgentContext, AgentState, build_graph, next_revision_count
from wp_content_agent.graph.builder import route_after_approval
from wp_content_agent.schemas import Brief, Draft, Review, RunStatus
from wp_content_agent.settings import Settings

DRAFT = Draft(
    title="A draft title",
    slug="a-draft",
    excerpt="",
    content_html="<p>x</p>",
    meta_description="m",
)


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
        return {"draft": DRAFT, "revision_count": next_revision_count(state)}

    def review(self, state: AgentState) -> dict[str, Any]:
        self.calls["review"] += 1
        passed = self._review_passes.pop(0)
        return {"review": Review(passed=passed, checks=[], fixes=[])}

    def human_approval(self, state: AgentState) -> dict[str, Any]:
        self.calls["human_approval"] += 1
        decision = interrupt({"revision_count": state.get("revision_count", 0)})
        if decision.get("revise"):
            return {"status": RunStatus.WRITING, "human_feedback": decision["note"]}
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


async def test_revise_goes_back_to_write_and_review_then_approves(
    context: AgentContext,
) -> None:
    nodes = FakeNodes(review_passes=[True, True])
    graph, config = _start(nodes)
    revise = {"approved": False, "revise": True, "note": "shorter intro"}

    await graph.ainvoke(BRIEF, config, context=context)
    paused = await graph.ainvoke(Command(resume=revise), config, context=context)
    done = await graph.ainvoke(Command(resume={"approved": True}), config, context=context)

    assert "__interrupt__" in paused
    assert nodes.calls["write"] == 2
    assert nodes.calls["review"] == 2
    assert nodes.calls["human_approval"] == 4  # each resume runs the node again from the top
    assert nodes.calls["publish_draft"] == 1
    assert done["status"] == RunStatus.DRAFT_CREATED


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (RunStatus.WRITING, "write"),
        (RunStatus.PUBLISHING, "publish_draft"),
        (RunStatus.REJECTED, "__end__"),
    ],
)
def test_route_after_approval(status: RunStatus, expected: str) -> None:
    assert route_after_approval({"status": status}) == expected


def test_build_graph_rejects_missing_nodes() -> None:
    with pytest.raises(ValueError, match="publish_draft"):
        build_graph({"plan": lambda state: {}})  # type: ignore[dict-item]


def test_build_graph_rejects_unknown_nodes() -> None:
    nodes = FakeNodes(review_passes=[]).mapping()
    nodes["publish"] = nodes.pop("publish_draft")

    with pytest.raises(ValueError, match="unknown nodes: \\['publish'\\]"):
        build_graph(nodes)  # type: ignore[arg-type]


def test_next_revision_count_starts_at_zero_and_counts_rewrites() -> None:
    first = next_revision_count({})
    second = next_revision_count({"draft": DRAFT, "revision_count": 0})

    assert (first, second) == (0, 1)
