from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from tests.unit.nodes.helpers import runtime
from tests.unit.nodes.test_review import ALL_PASS, GOOD
from wp_content_agent.graph import AgentContext, AgentState
from wp_content_agent.nodes.human_approval import approval_request, human_approval
from wp_content_agent.nodes.review import combine
from wp_content_agent.schemas import ResearchNotes, RunStatus, Term

NOTES = ResearchNotes(categories=[Term(id=1, name="Backend", slug="backend")])


def _graph() -> Any:
    graph = StateGraph(AgentState, context_schema=AgentContext)
    graph.add_node("human_approval", human_approval)
    graph.add_edge(START, "human_approval")
    graph.add_edge("human_approval", END)
    return graph.compile(checkpointer=InMemorySaver())


def _state() -> dict[str, Any]:
    draft = GOOD.model_copy(update={"category_ids": [1, 9]})
    return {"draft": draft, "review": combine(ALL_PASS, draft, []), "context": NOTES}


def test_approval_request_names_categories_and_failed_checks() -> None:
    state = _state()
    state["review"] = combine(ALL_PASS, GOOD.model_copy(update={"meta_description": "x"}), [])

    request = approval_request(state)  # type: ignore[arg-type]

    assert request.category_names == ["Backend", "9"]
    assert request.failed_checks == ["meta_description_length: 1 characters; needs 140 to 160"]
    assert not request.review_passed


@pytest.mark.parametrize(
    ("decision", "status"),
    [({"approved": True}, RunStatus.PUBLISHING), ({"approved": False}, RunStatus.REJECTED)],
)
async def test_resume_sets_status_and_note(decision: dict[str, Any], status: RunStatus) -> None:
    graph, config, ctx = _graph(), {"configurable": {"thread_id": "t"}}, runtime().context

    paused = await graph.ainvoke(_state(), config, context=ctx)
    result = await graph.ainvoke(
        Command(resume={**decision, "note": "looks good"}), config, context=ctx
    )

    assert paused["__interrupt__"][0].value["title"] == GOOD.title
    assert result["status"] == status
    assert result["approval_note"] == "looks good"


async def test_invalid_resume_value_raises_agent_error() -> None:
    graph, config, ctx = _graph(), {"configurable": {"thread_id": "t"}}, runtime().context
    await graph.ainvoke(_state(), config, context=ctx)

    with pytest.raises(Exception, match="invalid approval decision"):
        await graph.ainvoke(Command(resume={"approved": "maybe"}), config, context=ctx)
