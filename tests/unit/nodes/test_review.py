import pytest

from tests.fakes import ai, fake_client
from tests.unit.nodes.helpers import BRIEF, runtime
from tests.unit.nodes.test_plan import OUTLINE
from tests.unit.nodes.test_write import DRAFT
from wp_content_agent.errors import LLMError
from wp_content_agent.nodes.review import MODEL_CHECKS, combine, review
from wp_content_agent.schemas import (
    CheckResult,
    ResearchNotes,
    ReviewVerdict,
    RunStatus,
    SourceNote,
)

GOOD = DRAFT.model_copy(
    update={"content_html": "<p>Cache views.</p>", "meta_description": "m" * 150}
)
ALL_PASS = ReviewVerdict(
    checks=[CheckResult(name=name, passed=True) for name in MODEL_CHECKS], fixes=[]
)


def _state(draft: object = GOOD, **extra: object) -> dict[str, object]:
    return {"brief": BRIEF, "outline": OUTLINE, "draft": draft, "revision_count": 0, **extra}


async def test_passing_review_moves_to_approval() -> None:
    reviewer = fake_client(ALL_PASS)

    update = await review(_state(), runtime(reviewer=reviewer))  # type: ignore[arg-type]

    assert update["review"].passed
    assert update["status"] == RunStatus.AWAITING_APPROVAL
    assert update["usage"][0].node == "review"
    assert [c.name for c in update["review"].checks][-2:] == [
        "meta_description_length",
        "allowed_html",
    ]


async def test_short_meta_description_fails_even_if_model_passes_everything() -> None:
    draft = GOOD.model_copy(update={"meta_description": "too short"})
    reviewer = fake_client(ALL_PASS)

    update = await review(_state(draft), runtime(reviewer=reviewer))  # type: ignore[arg-type]

    assert not update["review"].passed
    assert update["status"] == RunStatus.WRITING
    assert any("meta description" in fix for fix in update["review"].fixes)


async def test_failed_review_at_revision_limit_goes_to_approval() -> None:
    draft = GOOD.model_copy(update={"meta_description": "too short"})
    reviewer = fake_client(ALL_PASS)

    update = await review(
        _state(draft, revision_count=2),  # type: ignore[arg-type]
        runtime(reviewer=reviewer, max_revisions=2),
    )

    assert not update["review"].passed
    assert update["status"] == RunStatus.AWAITING_APPROVAL


async def test_review_raises_llm_error_on_bad_output() -> None:
    reviewer = fake_client(ai("looks fine to me"))

    with pytest.raises(LLMError):
        await review(_state(), runtime(reviewer=reviewer))  # type: ignore[arg-type]


def test_combine_marks_missing_model_checks_as_failed() -> None:
    verdict = ReviewVerdict(checks=[CheckResult(name="structure", passed=True)], fixes=[])

    result = combine(verdict, GOOD, [])

    missing = {c.name for c in result.failed_checks}
    assert missing == set(MODEL_CHECKS) - {"structure"}
    assert not result.passed


def test_combine_ignores_model_verdicts_on_mechanical_checks() -> None:
    verdict = ALL_PASS.model_copy(
        update={
            "checks": [
                *ALL_PASS.checks,
                CheckResult(name="meta_description_length", passed=True),
            ]
        }
    )
    draft = GOOD.model_copy(update={"meta_description": "short"})

    result = combine(verdict, draft, [])

    meta = [c for c in result.checks if c.name == "meta_description_length"]
    assert len(meta) == 1 and not meta[0].passed


def test_combine_fails_claims_check_on_numbers_not_in_sources() -> None:
    draft = GOOD.model_copy(update={"content_html": "<p>Caching cut costs by 63%.</p>"})
    notes = ResearchNotes(sources=[SourceNote(url="u", title="t", facts=["Load fell 40%."])])
    facts = [fact for s in notes.sources for fact in s.facts]

    result = combine(ALL_PASS, draft, facts)

    claims = next(c for c in result.checks if c.name == "no_unsupported_claims")
    assert not claims.passed
    assert "63%" in claims.note


def test_combine_keeps_model_note_when_numbers_also_fail() -> None:
    checks = [
        CheckResult(name=name, passed=name != "no_unsupported_claims", note="invented quote")
        for name in MODEL_CHECKS
    ]
    draft = GOOD.model_copy(update={"content_html": "<p>Costs fell 63%.</p>"})

    result = combine(ReviewVerdict(checks=checks, fixes=[]), draft, [])

    claims = next(c for c in result.checks if c.name == "no_unsupported_claims")
    assert claims.note == "invented quote; numbers not found in the sources: 63%"


async def test_numbers_from_source_titles_are_allowed_but_target_length_is_not() -> None:
    notes = ResearchNotes(
        sources=[SourceNote(url="u", title="2025 State of Django", facts=["Caching helps."])]
    )
    draft = GOOD.model_copy(
        update={"content_html": "<p>Per the 2025 State of Django report, 600 teams agree.</p>"}
    )

    update = await review(
        _state(draft, context=notes),  # type: ignore[arg-type]
        runtime(reviewer=fake_client(ALL_PASS)),
    )

    claims = next(c for c in update["review"].checks if c.name == "no_unsupported_claims")
    assert claims.note == "numbers not found in the sources: 600"
