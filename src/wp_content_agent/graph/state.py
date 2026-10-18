import operator
from enum import Enum
from typing import Annotated, Any, TypedDict, get_args, get_type_hints

from pydantic import BaseModel

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
    # The editor's note to apply on the next write. The write node clears it.
    human_feedback: str
    human_revision_count: int
    wp_post_id: int
    edit_url: str
    status: RunStatus
    # Each node appends its own entry, so the list is the per-node cost log for the run.
    usage: Annotated[list[NodeUsage], operator.add]
    # Problems that did not stop the run but the person should hear about.
    warnings: Annotated[list[str], operator.add]
    error: str
    # The node that raised, set when a run is marked failed.
    failed_node: str


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


def checkpoint_types() -> list[type]:
    """Every model and enum the state can hold, nested ones included.

    The checkpointer only loads types on this list, so it is derived from AgentState
    rather than kept by hand: a new field cannot be forgotten.
    """
    found: set[type] = set()

    def visit(annotation: Any) -> None:
        for arg in get_args(annotation):
            visit(arg)
        if not isinstance(annotation, type) or annotation in found:
            return
        if issubclass(annotation, BaseModel):
            found.add(annotation)
            for info in annotation.model_fields.values():
                visit(info.annotation)
        elif issubclass(annotation, Enum):
            found.add(annotation)

    for annotation in get_type_hints(AgentState, include_extras=True).values():
        visit(annotation)
    return sorted(found, key=lambda t: f"{t.__module__}.{t.__qualname__}")
