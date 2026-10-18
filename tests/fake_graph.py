"""A real graph with fake model and WordPress nodes, for runner, CLI and API tests."""

from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from tests.fakes import fake_client
from wp_content_agent.errors import ToolError
from wp_content_agent.graph import AgentContext, AgentState, build_graph, next_revision_count
from wp_content_agent.nodes import NODES
from wp_content_agent.runner import AgentRunner
from wp_content_agent.schemas import Brief, Draft, NodeUsage, Review, RunStatus
from wp_content_agent.settings import Settings

BRIEF = Brief(topic="Caching in Django", audience="Backend developers")
DRAFT = Draft(
    title="Caching Django querysets",
    slug="caching-django-querysets",
    excerpt="",
    content_html="<p>x</p>",
    meta_description="m",
    tag_names=["django"],
)


def usage(node: str) -> NodeUsage:
    return NodeUsage(node=node, model="m", input_tokens=10, output_tokens=5, cost_usd=0.25)


def fake_nodes(**overrides: Any) -> dict[str, Any]:
    """Fake model and WordPress nodes around the real human_approval node."""

    def plan(state: AgentState) -> dict[str, Any]:
        return {"status": RunStatus.GATHERING, "usage": [usage("plan")]}

    def gather_context(state: AgentState) -> dict[str, Any]:
        return {"status": RunStatus.WRITING}

    def write(state: AgentState) -> dict[str, Any]:
        return {"draft": DRAFT, "revision_count": next_revision_count(state)}

    def review(state: AgentState) -> dict[str, Any]:
        return {
            "review": Review(passed=True, checks=[], fixes=[]),
            "status": RunStatus.AWAITING_APPROVAL,
        }

    def publish_draft(state: AgentState) -> dict[str, Any]:
        return {
            "status": RunStatus.DRAFT_CREATED,
            "wp_post_id": 8,
            "edit_url": "https://wp.test/wp-admin/post.php?post=8&action=edit",
        }

    nodes: dict[str, Any] = {
        "plan": plan,
        "gather_context": gather_context,
        "write": write,
        "review": review,
        "human_approval": NODES["human_approval"],
        "publish_draft": publish_draft,
    }
    return {**nodes, **overrides}


def flaky_publish(failures: int = 1) -> Any:
    """A publish_draft node that raises ToolError the first `failures` times, then succeeds."""
    calls = 0

    def publish_draft(state: AgentState) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls <= failures:
            raise ToolError("create_draft_post failed: timeout")
        return {
            "status": RunStatus.DRAFT_CREATED,
            "wp_post_id": 8,
            "edit_url": "https://wp.test/wp-admin/post.php?post=8&action=edit",
        }

    return publish_draft


def fake_runner(settings: Settings | None = None, **overrides: Any) -> AgentRunner:
    context = AgentContext(
        settings=settings or Settings(_env_file=None),  # type: ignore[call-arg]
        writer=fake_client(),
        reviewer=fake_client(),
        wordpress=object(),  # type: ignore[arg-type]
    )
    graph = build_graph(fake_nodes(**overrides), checkpointer=InMemorySaver())  # type: ignore[arg-type]
    return AgentRunner(graph, context)
