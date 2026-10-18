"""Turn the writer's semantic HTML into serialized Gutenberg blocks.

The model writes plain HTML, which is easier for it and for the review checks. WordPress
stores block editor content as HTML with block comments, so the conversion happens here,
in code, just before the draft is created. Anything without a block mapping becomes a
Custom HTML block, which the editor always accepts.
"""

from html.parser import HTMLParser

_LISTS = frozenset({"ul", "ol"})
_VOID = frozenset({"br", "img", "hr"})


class _BlockWriter(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.blocks: list[str] = []
        self._out: list[str] = []
        # Each open element with the text that closes it (end tag plus block comment).
        self._stack: list[tuple[str, str]] = []
        self._loose: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        parent = self._stack[-1][0] if self._stack else None
        raw = self.get_starttag_text() or f"<{tag}>"
        if parent is None:
            opening, closing = _top_level(tag, raw)
        elif tag == "li" and parent in _LISTS:
            opening, closing = "<!-- wp:list-item -->\n<li>", "</li>\n<!-- /wp:list-item -->"
        elif tag in _LISTS and parent == "li":
            opening, closing = _list(tag)
        elif tag == "p" and parent == "blockquote" and len(self._stack) == 1:
            opening, closing = "<!-- wp:paragraph -->\n<p>", "</p>\n<!-- /wp:paragraph -->"
        else:
            opening, closing = raw, f"</{tag}>"
        if parent is None:
            self._flush_loose()
            self._out = []
        self._out.append(opening)
        if tag in _VOID:
            if parent is None:
                self._out.append("\n<!-- /wp:html -->")
            self._finish_if_top_level()
            return
        self._stack.append((tag, closing))

    def handle_endtag(self, tag: str) -> None:
        if not any(open_tag == tag for open_tag, _ in self._stack):
            return  # stray end tag
        while self._stack:
            open_tag, closing = self._stack.pop()
            self._out.append(closing)
            if open_tag == tag:
                break
        self._finish_if_top_level()

    def handle_data(self, data: str) -> None:
        if not self._stack:
            self._loose.append(data)
        elif [tag for tag, _ in self._stack] == ["blockquote"] and data.strip():
            self._out.append(_paragraph(data.strip()))
        else:
            self._out.append(data)

    def handle_entityref(self, name: str) -> None:
        self.handle_data(f"&{name};")

    def handle_charref(self, name: str) -> None:
        self.handle_data(f"&#{name};")

    def close(self) -> None:
        super().close()
        while self._stack:  # unclosed elements at the end of the input
            self._out.append(self._stack.pop()[1])
        self._finish_if_top_level()
        self._flush_loose()

    def _flush_loose(self) -> None:
        # Loose text between top-level elements becomes its own paragraph.
        text = " ".join("".join(self._loose).split())
        if text:
            self.blocks.append(_paragraph(text))
        self._loose = []

    def _finish_if_top_level(self) -> None:
        if not self._stack and self._out:
            self.blocks.append("".join(self._out))
            self._out = []


def _paragraph(text: str) -> str:
    return f"<!-- wp:paragraph -->\n<p>{text}</p>\n<!-- /wp:paragraph -->"


def _top_level(tag: str, raw: str) -> tuple[str, str]:
    if tag == "p":
        return "<!-- wp:paragraph -->\n<p>", "</p>\n<!-- /wp:paragraph -->"
    if tag in {"h2", "h3", "h4"}:
        level = int(tag[1])
        attrs = "" if level == 2 else f' {{"level":{level}}}'
        return (
            f'<!-- wp:heading{attrs} -->\n<{tag} class="wp-block-heading">',
            f"</{tag}>\n<!-- /wp:heading -->",
        )
    if tag in _LISTS:
        return _list(tag)
    if tag == "blockquote":
        return (
            '<!-- wp:quote -->\n<blockquote class="wp-block-quote">',
            "</blockquote>\n<!-- /wp:quote -->",
        )
    return f"<!-- wp:html -->\n{raw}", f"</{tag}>\n<!-- /wp:html -->"


def _list(tag: str) -> tuple[str, str]:
    attrs = ' {"ordered":true}' if tag == "ol" else ""
    return (
        f'<!-- wp:list{attrs} -->\n<{tag} class="wp-block-list">',
        f"</{tag}>\n<!-- /wp:list -->",
    )


def to_blocks(html: str) -> str:
    writer = _BlockWriter()
    writer.feed(html)
    writer.close()
    return "\n\n".join(writer.blocks)
