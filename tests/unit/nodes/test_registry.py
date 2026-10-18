from langgraph.checkpoint.memory import InMemorySaver

from wp_content_agent.graph import build_graph
from wp_content_agent.graph.builder import NODE_ORDER
from wp_content_agent.nodes import NODES


def test_registry_covers_every_node_and_compiles() -> None:
    graph = build_graph(NODES, InMemorySaver())

    assert set(NODES) == set(NODE_ORDER)
    assert set(NODE_ORDER) <= set(graph.get_graph().nodes)
