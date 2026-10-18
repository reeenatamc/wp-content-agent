import operator
from typing import Annotated, TypedDict

from wp_content_agent.schemas import (
    Brief,
    Draft,
    NodeUsage,
    Outline,
    ResearchNotes,
    Review,
    RunStatus,
)


class AgentState(TypedDict, total=False):
    brief: Brief
    outline: Outline
    context: ResearchNotes
    draft: Draft
    review: Review
    revision_count: int
    approval_note: str
    wp_post_id: int
    edit_url: str
    status: RunStatus
    # Each node appends its own entry, so the list is the per-node cost log for the run.
    usage: Annotated[list[NodeUsage], operator.add]


def next_revision_count(state: AgentState) -> int:
    """Revisions so far, counting the draft about to be written. The first draft is 0."""
    if "draft" not in state:
        return 0
    return state.get("revision_count", 0) + 1
