"""Plain-text renderings of state objects for prompts. Pure functions, easy to test."""

from wp_content_agent.schemas import Brief, Draft, Outline, ResearchNotes, Review


def render_brief(brief: Brief) -> str:
    lines = [
        f"Topic: {brief.topic}",
        f"Audience: {brief.audience}",
        f"Tone: {brief.tone}",
        f"Target length: about {brief.target_words} words",
        f"Keywords: {', '.join(brief.keywords) or 'none'}",
    ]
    if brief.source_urls:
        lines.append("Source URLs:")
        lines.extend(f"- {url}" for url in brief.source_urls)
    return "\n".join(lines)


def render_outline(outline: Outline) -> str:
    lines = [f"Working title: {outline.working_title}", f"Angle: {outline.angle}", ""]
    for section in outline.sections:
        indent = "  " if section.level == 3 else ""
        lines.append(f"{indent}- H{section.level}: {section.heading}")
        lines.extend(f"{indent}    * {point}" for point in section.key_points)
    return "\n".join(lines)


def render_context(notes: ResearchNotes) -> str:
    parts: list[str] = []
    if notes.sources:
        parts.append("Facts from sources (the only claims you may state as fact):")
        for source in notes.sources:
            parts.append(f"[{source.title}]({source.url})")
            parts.extend(f"- {fact}" for fact in source.facts)
    else:
        parts.append("No sources were provided. Do not state statistics or cite studies.")
    if notes.related_posts:
        parts.append("\nExisting posts on the site (do not duplicate; link if relevant):")
        parts.extend(f"- {post.title}: {post.link}" for post in notes.related_posts)
    if notes.categories:
        parts.append("\nExisting categories (choose by id):")
        parts.extend(f"- {c.id}: {c.name}" for c in notes.categories)
    if notes.tags:
        parts.append("\nExisting tags (prefer these names):")
        parts.append(", ".join(t.name for t in notes.tags))
    return "\n".join(parts)


def render_draft(draft: Draft) -> str:
    return "\n".join(
        [
            f"Title: {draft.title}",
            f"Slug: {draft.slug}",
            f"Excerpt: {draft.excerpt}",
            f"Meta description ({len(draft.meta_description)} chars): {draft.meta_description}",
            f"Category ids: {draft.category_ids}",
            f"Tags: {', '.join(draft.tag_names)}",
            "Body HTML:",
            draft.content_html,
        ]
    )


def render_feedback(review: Review) -> str:
    lines = ["The previous draft failed review."]
    lines.extend(f"- {check.name}: {check.note}" for check in review.failed_checks)
    if review.fixes:
        lines.append("Fixes to apply:")
        lines.extend(f"- {fix}" for fix in review.fixes)
    return "\n".join(lines)
