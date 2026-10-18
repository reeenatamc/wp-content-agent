import pytest
from pydantic import ValidationError

from wp_content_agent.schemas import (
    ApprovalDecision,
    ApprovalRequest,
    Brief,
    Draft,
    Outline,
    OutlineSection,
    Review,
)
from wp_content_agent.schemas.draft import slugify


def _draft(**overrides: object) -> Draft:
    fields: dict[str, object] = {
        "title": "A practical guide",
        "slug": "a-practical-guide",
        "excerpt": "Short summary.",
        "content_html": "<p>Body</p>",
        "meta_description": "x" * 150,
    }
    fields.update(overrides)
    return Draft.model_validate(fields)


class TestBrief:
    def test_applies_defaults(self) -> None:
        brief = Brief(topic="Caching in Django", audience="Backend developers")

        assert brief.target_words == 900
        assert brief.keywords == []

    def test_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            Brief.model_validate({"topic": "abc", "audience": "devs", "publish": True})

    @pytest.mark.parametrize("words", [100, 5000])
    def test_rejects_target_words_out_of_range(self, words: int) -> None:
        with pytest.raises(ValidationError):
            Brief(topic="abc", audience="devs", target_words=words)

    def test_rejects_invalid_source_url(self) -> None:
        with pytest.raises(ValidationError):
            Brief.model_validate({"topic": "abc", "audience": "devs", "source_urls": ["nope"]})


class TestDraft:
    def test_normalizes_slug(self) -> None:
        draft = _draft(slug="  Caché en Django: Guía Rápida! ")

        assert draft.slug == "cache-en-django-guia-rapida"

    def test_rejects_empty_slug(self) -> None:
        with pytest.raises(ValidationError):
            _draft(slug="!!!")

    def test_rejects_too_many_tags(self) -> None:
        with pytest.raises(ValidationError):
            _draft(tag_names=[f"t{i}" for i in range(9)])


def test_slugify_collapses_separators() -> None:
    assert slugify("a  --  b__c") == "a-b-c"


def test_review_lists_failed_checks() -> None:
    review = Review.model_validate(
        {
            "passed": False,
            "checks": [
                {"name": "structure", "passed": True},
                {"name": "readability", "passed": False, "note": "Long paragraphs"},
            ],
            "fixes": ["Split the second section"],
        }
    )

    assert [check.name for check in review.failed_checks] == ["readability"]


def test_outline_must_start_with_h2() -> None:
    sections = [
        OutlineSection(heading="Sub", level=3, key_points=[]),
        OutlineSection(heading="Main", level=2, key_points=[]),
    ]

    with pytest.raises(ValidationError, match="first section must be an H2"):
        Outline(working_title="T", angle="A", sections=sections)


class TestApprovalDecision:
    def test_revise_with_a_note_is_valid(self) -> None:
        decision = ApprovalDecision(approved=False, revise=True, note="shorter intro")

        assert decision.revise
        assert decision.note == "shorter intro"

    def test_revise_defaults_to_false(self) -> None:
        assert not ApprovalDecision(approved=True).revise

    @pytest.mark.parametrize("note", ["", "   \n"])
    def test_revise_needs_a_note_with_text(self, note: str) -> None:
        with pytest.raises(ValidationError, match="needs a note"):
            ApprovalDecision(approved=False, revise=True, note=note)

    def test_revise_cannot_also_approve(self) -> None:
        with pytest.raises(ValidationError, match="both approve and revise"):
            ApprovalDecision(approved=True, revise=True, note="x")

    def test_revise_must_be_a_real_bool(self) -> None:
        with pytest.raises(ValidationError):
            ApprovalDecision.model_validate({"approved": False, "revise": "yes", "note": "x"})


class TestApprovalRequest:
    def test_request_saved_before_revisions_loads_with_none_left(self) -> None:
        saved = {
            "title": "T",
            "slug": "t",
            "excerpt": "E",
            "meta_description": "M",
            "content_html": "<p>x</p>",
            "category_names": [],
            "tag_names": [],
            "review_passed": True,
            "failed_checks": [],
            "revision_count": 0,
        }

        request = ApprovalRequest.model_validate(saved)

        assert request.human_revisions_left == 0
