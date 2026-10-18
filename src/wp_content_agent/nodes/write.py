import logging
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.runtime import Runtime

from wp_content_agent.graph.context import AgentContext
from wp_content_agent.graph.state import AgentState, next_revision_count
from wp_content_agent.nodes.render import (
    render_brief,
    render_context,
    render_feedback,
    render_outline,
)
from wp_content_agent.schemas import Draft, ResearchNotes, RunStatus

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You write blog posts for a WordPress site.

Write the post described by the brief, following the outline's sections and order. Match \
the audience and tone, and land within 10 percent of the target length.

Facts: state as fact only what appears in the research notes' sources, and attribute it \
("according to <source title>"). Never invent statistics, percentages, studies, quotes or \
dates. If a section needs evidence the notes do not have, make the point without numbers.

Body HTML: use only <p>, <h2>, <h3>, <ul>, <ol>, <li>, <a href>, <strong>, <em>, \
<blockquote> and <code>. Each top-level element on its own line. No <h1> (the title is \
separate), no inline styles, classes, scripts or block comments. Link to existing posts \
from the notes where they genuinely help, and to sources you cite.

Readability: short paragraphs (at most four sentences), plain words, active voice, and a \
heading every 200 to 300 words. Use each keyword naturally at least once, one of them in \
the first paragraph; never stuff them.

Fields:
- title: specific, under 70 characters when possible.
- slug: a few lowercase words.
- excerpt: one or two sentences.
- meta_description: between 140 and 160 characters, counted with spaces. Count before \
answering.
- category_ids: ids from the notes' existing categories only.
- tag_names: three to six short tags, reusing existing tag names when they fit."""

REVISION_INSTRUCTIONS = """Revise your draft above to fix every point in this review. \
Keep what already works; do not rewrite passing sections from scratch."""


async def write(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    writer = runtime.context.writer
    notes = state.get("context") or ResearchNotes()
    task = "\n\n".join(
        [
            render_brief(state["brief"]),
            f"Outline:\n{render_outline(state['outline'])}",
            f"Research notes:\n{render_context(notes)}",
        ]
    )
    # The system prompt and the task stay identical across revisions, so they are cached.
    messages: list[BaseMessage] = [writer.system(SYSTEM_PROMPT), writer.human(task, cache=True)]
    previous, review = state.get("draft"), state.get("review")
    if previous is not None and review is not None:
        # The previous draft is replayed as the model's own answer, then the review follows.
        messages.append(AIMessage(previous.model_dump_json()))
        messages.append(HumanMessage(f"{REVISION_INSTRUCTIONS}\n\n{render_feedback(review)}"))

    draft, usage = await writer.structured(Draft, messages, node="write")
    return {
        "draft": _known_categories(draft, notes),
        "revision_count": next_revision_count(state),
        "status": RunStatus.REVIEWING,
        "usage": [usage],
    }


def _known_categories(draft: Draft, notes: ResearchNotes) -> Draft:
    known = {category.id for category in notes.categories}
    kept = [cid for cid in draft.category_ids if cid in known]
    if len(kept) < len(draft.category_ids):
        logger.warning("dropped unknown category ids: %s", set(draft.category_ids) - known)
    return draft.model_copy(update={"category_ids": kept})
