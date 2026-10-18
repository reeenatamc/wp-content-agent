from wp_content_agent.nodes.gutenberg import to_blocks


def test_paragraph_and_headings_become_blocks() -> None:
    html = '<h2>Why</h2>\n<p>Text with <a href="https://x.test">a link</a>.</p>\n<h3>Detail</h3>'

    blocks = to_blocks(html).split("\n\n")

    assert blocks == [
        '<!-- wp:heading -->\n<h2 class="wp-block-heading">Why</h2>\n<!-- /wp:heading -->',
        '<!-- wp:paragraph -->\n<p>Text with <a href="https://x.test">a link</a>.</p>\n'
        "<!-- /wp:paragraph -->",
        '<!-- wp:heading {"level":3} -->\n<h3 class="wp-block-heading">Detail</h3>\n'
        "<!-- /wp:heading -->",
    ]


def test_lists_wrap_each_item_and_nested_lists() -> None:
    html = "<ol><li>One<ul><li>Sub</li></ul></li><li>Two</li></ol>"

    result = to_blocks(html)

    assert result.startswith('<!-- wp:list {"ordered":true} -->\n<ol class="wp-block-list">')
    assert result.count("<!-- wp:list-item -->") == 3
    assert '<!-- wp:list -->\n<ul class="wp-block-list">' in result
    assert result.endswith("</ol>\n<!-- /wp:list -->")


def test_quote_text_is_wrapped_in_a_paragraph() -> None:
    result = to_blocks("<blockquote>Said it.</blockquote>")

    assert result == (
        '<!-- wp:quote -->\n<blockquote class="wp-block-quote">'
        "<!-- wp:paragraph -->\n<p>Said it.</p>\n<!-- /wp:paragraph -->"
        "</blockquote>\n<!-- /wp:quote -->"
    )


def test_unknown_elements_become_custom_html_blocks() -> None:
    result = to_blocks("<table><tr><td>x</td></tr></table>")

    assert result == "<!-- wp:html -->\n<table><tr><td>x</td></tr></table>\n<!-- /wp:html -->"


def test_loose_text_and_entities_are_kept() -> None:
    result = to_blocks("Intro &amp; more\n<p>A &#8212; B</p>")

    assert "<p>Intro &amp; more</p>" in result
    assert "<p>A &#8212; B</p>" in result


def test_unclosed_and_stray_tags_do_not_lose_text() -> None:
    result = to_blocks("<p>Open paragraph</em>")

    assert result == "<!-- wp:paragraph -->\n<p>Open paragraph</p>\n<!-- /wp:paragraph -->"


def test_lists_inside_custom_html_stay_raw() -> None:
    result = to_blocks("<div><ul><li>a</li></ul></div>")

    assert result == "<!-- wp:html -->\n<div><ul><li>a</li></ul></div>\n<!-- /wp:html -->"


def test_quote_with_inline_markup_and_entities_is_one_paragraph() -> None:
    result = to_blocks("<blockquote>Fish &amp; chips, <em>said</em> well.</blockquote>")

    assert result.count("<!-- wp:paragraph -->") == 1
    assert "<p>Fish &amp; chips, <em>said</em> well.</p>" in result


def test_quote_children_lists_become_list_blocks() -> None:
    result = to_blocks("<blockquote><p>A</p><ul><li>b</li></ul></blockquote>")

    assert '<blockquote class="wp-block-quote"><!-- wp:paragraph -->' in result
    assert "<!-- wp:list-item -->" in result


def test_top_level_inline_run_becomes_one_paragraph() -> None:
    result = to_blocks("Intro <strong>bold</strong> and <a href='/x'>more</a>.<h2>Next</h2>")

    blocks = result.split("\n\n")
    assert blocks[0] == (
        "<!-- wp:paragraph -->\n<p>Intro <strong>bold</strong> and <a href='/x'>more</a>.</p>\n"
        "<!-- /wp:paragraph -->"
    )
    assert blocks[1].startswith("<!-- wp:heading -->")


def test_attributes_are_kept_and_all_heading_levels_map() -> None:
    result = to_blocks('<h2 id="faq">FAQ</h2><h5>Small</h5><p class="lead">x</p>')

    assert '<h2 class="wp-block-heading" id="faq">' in result
    assert '<!-- wp:heading {"level":5} -->\n<h5 class="wp-block-heading">' in result
    assert '<p class="lead">x</p>' in result


def test_void_elements_do_not_swallow_following_blocks() -> None:
    result = to_blocks("<hr><p>After</p><table><tr><td>a<wbr>b</td></tr></table><p>End</p>")

    blocks = result.split("\n\n")
    assert blocks[0] == "<!-- wp:html -->\n<hr>\n<!-- /wp:html -->"
    assert blocks[1] == "<!-- wp:paragraph -->\n<p>After</p>\n<!-- /wp:paragraph -->"
    assert blocks[3] == "<!-- wp:paragraph -->\n<p>End</p>\n<!-- /wp:paragraph -->"


def test_more_marker_is_kept_and_other_top_level_comments_dropped() -> None:
    result = to_blocks("<p>A</p><!--more--><!-- note --><p>B</p>")

    assert result.split("\n\n")[1] == "<!-- wp:more -->\n<!--more-->\n<!-- /wp:more -->"
    assert "note" not in result
