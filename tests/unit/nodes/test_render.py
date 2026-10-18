from tests.unit.nodes.helpers import BRIEF
from wp_content_agent.nodes.render import (
    render_brief,
    render_context,
    render_draft,
    render_feedback,
    render_outline,
)
from wp_content_agent.schemas import (
    CheckResult,
    Draft,
    Outline,
    OutlineSection,
    RelatedPost,
    ResearchNotes,
    Review,
    SourceNote,
    Term,
)


def test_render_brief_lists_sources_and_keywords() -> None:
    text = render_brief(BRIEF)

    assert "Keywords: django, caching" in text
    assert "- https://example.com/django-cache" in text


def test_render_brief_without_keywords_says_none() -> None:
    brief = BRIEF.model_copy(update={"keywords": [], "source_urls": []})

    text = render_brief(brief)

    assert "Keywords: none" in text
    assert "Source URLs" not in text


def test_render_outline_indents_h3_under_h2() -> None:
    outline = Outline(
        working_title="T",
        angle="A",
        sections=[
            OutlineSection(heading="Why", level=2, key_points=["cost"]),
            OutlineSection(heading="Details", level=3, key_points=[]),
        ],
    )

    text = render_outline(outline)

    assert "- H2: Why\n    * cost" in text
    assert "  - H3: Details" in text


def test_render_context_without_sources_forbids_statistics() -> None:
    text = render_context(ResearchNotes())

    assert "Do not state statistics" in text


def test_render_context_includes_facts_posts_and_terms() -> None:
    notes = ResearchNotes(
        sources=[SourceNote(url="https://s.test", title="Src", facts=["Fact one"])],
        related_posts=[RelatedPost(id=1, title="Old post", link="https://wp.test/old")],
        categories=[Term(id=3, name="Backend", slug="backend")],
        tags=[Term(id=4, name="django", slug="django")],
    )

    text = render_context(notes)

    assert "- Fact one" in text
    assert "Old post: https://wp.test/old" in text
    assert "- 3: Backend" in text
    assert "django" in text


def test_render_draft_shows_meta_description_length() -> None:
    draft = Draft(
        title="A title", slug="a", excerpt="", content_html="<p>x</p>", meta_description="abc"
    )

    assert "Meta description (3 chars): abc" in render_draft(draft)


def test_render_feedback_lists_only_failed_checks_and_fixes() -> None:
    review = Review(
        passed=False,
        checks=[
            CheckResult(name="structure", passed=True, note="ok"),
            CheckResult(name="readability", passed=False, note="long sentences"),
        ],
        fixes=["Split paragraph 2"],
    )

    text = render_feedback(review)

    assert "readability: long sentences" in text
    assert "structure" not in text
    assert "- Split paragraph 2" in text
