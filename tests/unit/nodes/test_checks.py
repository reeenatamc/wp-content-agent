import pytest

from tests.unit.nodes.test_write import DRAFT
from wp_content_agent.nodes.checks import (
    allowed_html_check,
    data_numbers,
    meta_description_check,
    unsupported_numbers,
)


@pytest.mark.parametrize(
    ("length", "passed"), [(139, False), (140, True), (160, True), (161, False)]
)
def test_meta_description_length_bounds(length: int, passed: bool) -> None:
    draft = DRAFT.model_copy(update={"meta_description": "x" * length})

    check, fix = meta_description_check(draft)

    assert check.passed is passed
    assert (fix is None) is passed


def test_short_meta_description_fix_says_lengthen_with_count() -> None:
    draft = DRAFT.model_copy(update={"meta_description": "x" * 98})

    check, fix = meta_description_check(draft)

    assert check.note == "98 characters; needs 140 to 160"
    assert fix is not None and fix.startswith("Lengthen")


def test_allowed_html_passes_semantic_markup() -> None:
    html = (
        '<h2>A</h2>\n<p>See <a href="https://x.test">x</a>, <code>qs</code>.</p>\n'
        "<ul><li>i</li></ul>"
    )
    draft = DRAFT.model_copy(update={"content_html": html})

    check, fix = allowed_html_check(draft)

    assert check.passed
    assert fix is None


def test_allowed_html_flags_tags_attributes_and_comments() -> None:
    html = '<h1>T</h1><p style="color:red">x</p><!-- wp:paragraph --><script>1</script>'
    draft = DRAFT.model_copy(update={"content_html": html})

    check, fix = allowed_html_check(draft)

    assert not check.passed
    for problem in ("<h1>", "<script>", 'style="" on <p>', "HTML comments"):
        assert problem in check.note
    assert fix is not None


def test_data_numbers_ignores_single_digits_and_tags() -> None:
    numbers = data_numbers('<h2 data-x="12">3 ways</h2><p>40% of 1,200 users in 2024, v5.2</p>')

    assert numbers == {"40%", "1200", "2024", "5.2"}


def test_unsupported_numbers_allows_numbers_from_sources() -> None:
    draft = DRAFT.model_copy(
        update={"content_html": "<p>Load fell 40%, and 73% of teams agree.</p>"}
    )

    stray = unsupported_numbers(draft, ["Per-view caching cut load by 40%."])

    assert stray == ["73%"]


def test_allowed_html_flags_unsafe_link_schemes() -> None:
    html = '<p><a href="javascript:alert(1)">x</a> <a href="/ok">y</a></p>'
    draft = DRAFT.model_copy(update={"content_html": html})

    check, _ = allowed_html_check(draft)

    assert not check.passed
    assert "javascript:alert(1)" in check.note
    assert "/ok" not in check.note


def test_data_numbers_ignores_entities_and_splits_lists() -> None:
    numbers = data_numbers("<p>It&#8217;s fine &#8212; values 10,20,30</p>")

    assert numbers == {"10", "20", "30"}


def test_percent_word_matches_percent_sign() -> None:
    assert data_numbers("forty: 40 percent") == data_numbers("40%")


def test_percentage_is_not_excused_by_a_plain_number() -> None:
    html = "<p>A 1200% increase.</p>"
    draft = DRAFT.model_copy(update={"content_html": html})

    assert unsupported_numbers(draft, ["about 1200 words"]) == ["1200%"]


def test_meta_description_numbers_are_checked() -> None:
    draft = DRAFT.model_copy(update={"meta_description": "Cut load by 63% today."})

    assert unsupported_numbers(draft, []) == ["63%"]
