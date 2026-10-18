import pytest
from pydantic import ValidationError

from wp_content_agent.schemas import Brief, Draft, Review
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
