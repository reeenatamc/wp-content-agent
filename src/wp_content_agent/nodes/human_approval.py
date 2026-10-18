from typing import Any

from langgraph.runtime import Runtime
from langgraph.types import interrupt
from pydantic import ValidationError

from wp_content_agent.errors import AgentError
from wp_content_agent.graph.context import AgentContext
from wp_content_agent.graph.state import AgentState
from wp_content_agent.schemas import ApprovalDecision, ApprovalRequest, ResearchNotes, RunStatus


def approval_request(state: AgentState, max_human_revisions: int) -> ApprovalRequest:
    draft = state["draft"]
    review = state.get("review")
    notes = state.get("context") or ResearchNotes()
    names = {category.id: category.name for category in notes.categories}
    return ApprovalRequest(
        title=draft.title,
        slug=draft.slug,
        excerpt=draft.excerpt,
        meta_description=draft.meta_description,
        content_html=draft.content_html,
        category_names=[names.get(cid, str(cid)) for cid in draft.category_ids],
        tag_names=draft.tag_names,
        review_passed=review.passed if review else False,
        failed_checks=[f"{c.name}: {c.note}" for c in review.failed_checks] if review else [],
        revision_count=state.get("revision_count", 0),
        human_revisions_left=max(0, max_human_revisions - state.get("human_revision_count", 0)),
    )


async def human_approval(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    # interrupt() pauses the run here; on resume the node runs again from the top and
    # interrupt() returns the value passed in Command(resume=...).
    request = approval_request(state, runtime.context.settings.max_human_revisions)
    raw = interrupt(request.model_dump())
    try:
        decision = ApprovalDecision.model_validate(raw)
    except ValidationError as exc:
        raise AgentError(f"invalid approval decision: {exc}") from exc
    if decision.revise:
        if request.human_revisions_left == 0:
            raise AgentError("no human revisions left; approve or reject the draft")
        return {
            "status": RunStatus.WRITING,
            "approval_note": decision.note,
            "human_feedback": decision.note,
            "human_revision_count": state.get("human_revision_count", 0) + 1,
        }
    status = RunStatus.PUBLISHING if decision.approved else RunStatus.REJECTED
    return {"status": status, "approval_note": decision.note}
