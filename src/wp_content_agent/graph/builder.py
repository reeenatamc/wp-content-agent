from collections.abc import Mapping
from typing import Any, Literal, get_args

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from wp_content_agent.graph.context import AgentContext
from wp_content_agent.graph.state import AgentState, should_revise
from wp_content_agent.schemas import RunStatus

NodeName = Literal["plan", "gather_context", "write", "review", "human_approval", "publish_draft"]
NODE_ORDER: tuple[NodeName, ...] = get_args(NodeName)


def route_after_review(
    state: AgentState, runtime: Runtime[AgentContext]
) -> Literal["write", "human_approval"]:
    if should_revise(state, runtime.context.settings.max_revisions):
        return "write"
    return "human_approval"


def route_after_approval(state: AgentState) -> Literal["publish_draft", "__end__"]:
    if state.get("status") == RunStatus.PUBLISHING:
        return "publish_draft"
    return "__end__"


def build_graph(
    nodes: Mapping[NodeName, Any], checkpointer: BaseCheckpointSaver[Any] | None = None
) -> CompiledStateGraph[AgentState, AgentContext, AgentState, AgentState]:
    """Wire the nodes into the agent graph.

    Nodes are injected so tests can run the real wiring (review loop, interrupt,
    resume) with fakes. The production nodes live in `wp_content_agent.nodes`.

    Contract for nodes: on failure they raise `AgentError`. The graph does not route
    on error states; the caller catches the exception and marks the run as failed.
    The write node sets `revision_count` with `next_revision_count`, which is what
    bounds the review loop.
    """
    missing, unknown = set(NODE_ORDER) - set(nodes), set(nodes) - set(NODE_ORDER)
    if missing or unknown:
        raise ValueError(f"missing nodes: {sorted(missing)}; unknown nodes: {sorted(unknown)}")

    graph = StateGraph(AgentState, context_schema=AgentContext)
    for name in NODE_ORDER:
        graph.add_node(name, nodes[name])

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "gather_context")
    graph.add_edge("gather_context", "write")
    graph.add_edge("write", "review")
    graph.add_conditional_edges("review", route_after_review)
    graph.add_conditional_edges("human_approval", route_after_approval)
    graph.add_edge("publish_draft", END)
    return graph.compile(checkpointer=checkpointer)
