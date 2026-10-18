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
