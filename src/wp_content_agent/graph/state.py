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


def should_revise(state: AgentState, max_revisions: int) -> bool:
    """True when the latest review failed and the revision budget is not used up."""
    review = state.get("review")
    if review is None:
        raise ValueError("review node finished without writing a review")
    return not review.passed and state.get("revision_count", 0) < max_revisions
