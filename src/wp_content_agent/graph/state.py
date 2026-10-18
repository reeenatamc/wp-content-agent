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
    error: str
