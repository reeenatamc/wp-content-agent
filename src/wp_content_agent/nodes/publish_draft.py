import logging
from typing import Any

from langgraph.runtime import Runtime

from wp_content_agent.errors import ToolError
from wp_content_agent.graph.context import AgentContext, WordPressGateway
from wp_content_agent.graph.state import AgentState
from wp_content_agent.nodes.gutenberg import to_blocks
from wp_content_agent.schemas import ResearchNotes, RunStatus

logger = logging.getLogger(__name__)


async def publish_draft(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    if state.get("wp_post_id"):
        # A resumed run that already created the post must not create a second one.
        logger.info("draft %s already exists; skipping creation", state["wp_post_id"])
        return {"status": RunStatus.DRAFT_CREATED}

    draft = state["draft"]
    wordpress = runtime.context.wordpress
    notes = state.get("context") or ResearchNotes()
    # Tags first and the post last, so a failure leaves no post with missing tags.
    tag_ids, warnings = await _resolve_tags(wordpress, draft.tag_names, notes)
    created = await wordpress.create_draft_post(
        title=draft.title,
        slug=draft.slug,
        excerpt=draft.excerpt,
        content_html=to_blocks(draft.content_html),
        categories=draft.category_ids,
        tags=tag_ids,
        meta_description=draft.meta_description,
    )
    if created.status != "draft":
        # Keep the id so the post can be found, but do not report success.
        message = f"WordPress returned status {created.status!r} instead of draft"
        logger.error("post %s: %s", created.id, message)
        return {
            "wp_post_id": created.id,
            "edit_url": created.edit_url,
            "status": RunStatus.FAILED,
            "error": message,
        }
    if draft.meta_description and not created.meta_description_saved:
        warnings.append(
            "WordPress did not store the meta description; install the wpagent-meta "
            "mu-plugin or set WP_META_DESCRIPTION_KEY to your SEO plugin's field"
        )
    logger.info("created draft %s: %s", created.id, created.edit_url)
    return {
        "wp_post_id": created.id,
        "edit_url": created.edit_url,
        "status": RunStatus.DRAFT_CREATED,
        "warnings": warnings,
    }


async def _resolve_tags(
    wordpress: WordPressGateway, names: list[str], notes: ResearchNotes
) -> tuple[list[int], list[str]]:
    known = {tag.name.casefold(): tag.id for tag in notes.tags}
    ids: list[int] = []
    warnings: list[str] = []
    unique = {n.strip().casefold(): n.strip() for n in reversed(names) if n.strip()}
    for name in reversed(unique.values()):
        tag_id = known.get(name.casefold())
        if tag_id is None:
            try:
                tag_id = (await wordpress.create_tag(name)).id
            except ToolError as exc:
                logger.warning("could not create tag %r: %s", name, exc)
                warnings.append(f"tag {name!r} was not added: {exc}")
                continue
        if tag_id not in ids:
            ids.append(tag_id)
    return ids, warnings
