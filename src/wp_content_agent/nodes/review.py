from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime

from wp_content_agent.graph.context import AgentContext
from wp_content_agent.graph.state import AgentState, should_revise
from wp_content_agent.nodes.checks import (
    allowed_html_check,
    meta_description_check,
    unsupported_numbers,
)
from wp_content_agent.nodes.render import render_brief, render_context, render_draft, render_outline
from wp_content_agent.schemas import (
    CheckName,
    CheckResult,
    Draft,
    ResearchNotes,
    Review,
    ReviewVerdict,
    RunStatus,
)

MODEL_CHECKS: tuple[CheckName, ...] = (
    "matches_brief",
    "structure",
    "keyword_usage",
    "readability",
    "no_unsupported_claims",
)

SYSTEM_PROMPT = f"""You review a blog post draft against its brief before a human sees it.

Return exactly one result for each of these checks: {", ".join(MODEL_CHECKS)}.
- matches_brief: covers the topic for the stated audience and tone, near the target length.
- structure: follows the outline; headings are in order, with H3 only under an H2.
- keyword_usage: every keyword appears naturally; none is stuffed.
- readability: short paragraphs, plain words, no filler or repetition.
- no_unsupported_claims: every factual claim, number, study or quote is backed by the \
research notes. Anything else fails this check, even if it sounds plausible.

Be strict but fair: fail a check only for a problem a reader or editor would notice. For \
each failure, write a note saying where the problem is, and add a concrete fix (what to \
change, and where) to fixes. When everything passes, fixes is empty."""


async def review(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    ctx = runtime.context
    draft = state["draft"]
    notes = state.get("context") or ResearchNotes()
    reviewer = ctx.reviewer
    task = "\n\n".join(
        [
            render_brief(state["brief"]),
            f"Outline:\n{_outline(state)}",
            f"Research notes:\n{render_context(notes)}",
        ]
    )
    messages = [
        reviewer.system(SYSTEM_PROMPT),
        reviewer.human(task, cache=True),
        HumanMessage(f"Draft to review:\n{render_draft(draft)}"),
    ]
    verdict, usage = await reviewer.structured(ReviewVerdict, messages, node="review")

    result = combine(verdict, draft, _allowed_texts(state, notes))
    status = (
        RunStatus.WRITING
        if should_revise({**state, "review": result}, ctx.settings.max_revisions)
        else RunStatus.AWAITING_APPROVAL
    )
    return {"review": result, "status": status, "usage": [usage]}


def combine(verdict: ReviewVerdict, draft: Draft, allowed_texts: list[str]) -> Review:
    """Merge the model's checks with the mechanical ones; code wins where it can decide."""
    by_name = {check.name: check for check in verdict.checks if check.name in MODEL_CHECKS}
    checks = [
        by_name.get(name) or CheckResult(name=name, passed=False, note="not evaluated")
        for name in MODEL_CHECKS
    ]
    fixes = list(verdict.fixes)

    stray = unsupported_numbers(draft, allowed_texts)
    if stray:
        index = MODEL_CHECKS.index("no_unsupported_claims")
        listed = ", ".join(stray)
        checks[index] = CheckResult(
            name="no_unsupported_claims",
            passed=False,
            note=f"numbers not found in the sources: {listed}",
        )
        fixes.append(f"Remove or rephrase without numbers: {listed}.")

    for check, fix in (meta_description_check(draft), allowed_html_check(draft)):
        checks.append(check)
        if fix:
            fixes.append(fix)
    return Review(passed=all(check.passed for check in checks), checks=checks, fixes=fixes)


def _outline(state: AgentState) -> str:
    outline = state.get("outline")
    return render_outline(outline) if outline else "none"


def _allowed_texts(state: AgentState, notes: ResearchNotes) -> list[str]:
    texts = [render_brief(state["brief"]), *(p.title for p in notes.related_posts)]
    texts.extend(fact for source in notes.sources for fact in source.facts)
    if outline := state.get("outline"):
        texts.append(render_outline(outline))
    return texts
