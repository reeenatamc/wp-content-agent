from datetime import UTC, datetime
from pathlib import Path

from evals.run import EvalResult, _words, render_markdown, result_from
from wp_content_agent.briefs import load_brief
from wp_content_agent.schemas import ApprovalRequest, NodeUsage, RunSnapshot, RunStatus
from wp_content_agent.settings import Settings

BRIEFS = Path(__file__).parents[2] / "evals" / "briefs"


def _approval(**overrides: object) -> ApprovalRequest:
    values: dict[str, object] = {
        "title": "t",
        "slug": "t",
        "excerpt": "e",
        "meta_description": "m" * 150,
        "content_html": "<p>One &amp; two</p><p>three</p>",
        "category_names": [],
        "tag_names": [],
        "review_passed": False,
        "failed_checks": ["meta_length: 120 chars", "headings: none"],
        "revision_count": 2,
        "human_revisions_left": 2,
    }
    return ApprovalRequest.model_validate(values | overrides)


def _usage(node: str, cache_read: int = 0) -> NodeUsage:
    return NodeUsage(
        node=node,
        model="m",
        input_tokens=100,
        output_tokens=20,
        cache_read_tokens=cache_read,
        cost_usd=0.01,
    )


def test_every_eval_brief_loads() -> None:
    paths = sorted(BRIEFS.glob("*.yaml"))

    briefs = [load_brief(path) for path in paths]

    assert len(briefs) >= 5
    assert len({brief.topic for brief in briefs}) == len(briefs)


def test_result_from_reads_the_approval_and_sums_usage() -> None:
    snapshot = RunSnapshot(
        run_id="r",
        status=RunStatus.AWAITING_APPROVAL,
        approval=_approval(),
        review_passed=False,
        revision_count=2,
        usage=[_usage("plan"), _usage("write", cache_read=60)],
    )

    result = result_from("01-x", snapshot, 12.34)

    assert result.status == "awaiting_approval"
    assert result.failed_checks == ["meta_length", "headings"]
    assert (result.words, result.meta_chars) == (4, 150)
    assert (result.input_tokens, result.output_tokens, result.cache_read_tokens) == (200, 40, 60)
    assert result.cost_usd == 0.02
    assert result.seconds == 12.3


def test_result_from_a_failed_run_has_no_draft_numbers() -> None:
    snapshot = RunSnapshot(run_id="r", status=RunStatus.FAILED, error="plan: model call failed")

    result = result_from("02-y", snapshot, 1.0)

    assert (result.review_passed, result.words, result.failed_checks) == (False, 0, [])
    assert result.error == "plan: model call failed"


def test_render_markdown_reports_pass_rate_cost_and_cache_share() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    results = [
        EvalResult("a", "awaiting_approval", True, 0, input_tokens=300, cache_read_tokens=150),
        EvalResult("b", "awaiting_approval", False, 2, ["meta_length"], input_tokens=100),
        EvalResult("c", "failed", False, 0, cost_usd=0.5, error="boom"),
    ]

    text = render_markdown(results, settings, datetime(2026, 10, 18, 9, 30, tzinfo=UTC))

    assert "Run on 2026-10-18 09:30 UTC" in text
    assert "| b | fail | 2 | meta_length |" in text
    assert "| c | error |" in text
    assert "Pass rate: 1/2 completed runs (1 errors). Average revisions: 1.0." in text
    assert "Total cost: $0.5000." in text
    assert "Cache reads: 150 of 400 input tokens (38%)." in text
    assert "- c: boom" in text


def test_render_markdown_without_results_does_not_divide_by_zero() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    text = render_markdown([], settings, datetime(2026, 10, 18, tzinfo=UTC))

    assert "Pass rate: 0/0 completed runs (0 errors)." in text


def test_render_markdown_escapes_table_cells() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    results = [EvalResult("a|b", "awaiting_approval", False, 1, ["x|y"])]

    text = render_markdown(results, settings, datetime(2026, 10, 18, tzinfo=UTC))

    assert r"| a\|b | fail | 1 | x\|y |" in text


def test_words_ignores_inline_tags_and_scripts() -> None:
    content = "<p>re<em>use</em>d foo<code>bar</code></p><p>two</p><script>var x = 1</script>"

    assert _words(content) == 3
