from typing import Any

from wp_content_agent.graph import NodeName
from wp_content_agent.nodes.gather_context import gather_context
from wp_content_agent.nodes.human_approval import human_approval
from wp_content_agent.nodes.plan import plan
from wp_content_agent.nodes.publish_draft import publish_draft
from wp_content_agent.nodes.review import review
from wp_content_agent.nodes.write import write

NODES: dict[NodeName, Any] = {
    "plan": plan,
    "gather_context": gather_context,
    "write": write,
    "review": review,
    "human_approval": human_approval,
    "publish_draft": publish_draft,
}

__all__ = ["NODES"]
