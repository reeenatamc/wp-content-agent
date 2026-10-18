"""Turn the writer's semantic HTML into serialized Gutenberg blocks.

The model writes plain HTML, which is easier for it and for the review checks. WordPress
stores block editor content as HTML with block comments, so the conversion happens here,
in code, just before the draft is created. Paragraphs, headings, lists and quotes map to
core blocks; other block-level elements become Custom HTML blocks, whose content is kept
as written. Loose text and inline elements between blocks are grouped into paragraphs.
"""

from dataclasses import dataclass, field
from html import escape
from html.parser import HTMLParser

_LISTS = frozenset({"ul", "ol"})
_HEADINGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
_VOID = frozenset(
    {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source"}
    | {"track", "wbr"}
)
_BLOCK_LEVEL = (
    _LISTS
    | _HEADINGS
    | {"p", "blockquote", "div", "table", "pre", "figure", "section", "article", "aside"}
    | {"header", "footer", "nav", "hr", "form", "details", "dl", "iframe", "video", "audio"}
    | {"script", "style", "main", "address", "fieldset"}
)
_MORE = "<!-- wp:more -->\n<!--more-->\n<!-- /wp:more -->"


@dataclass
class _Frame:
    tag: str
    closing: str
    # "run": inline element in a paragraph being grouped; "quote": quote block whose
    # children are blocks; "list"/"item": core list markup; "raw": everything else.
    kind: str
    run: list[str] = field(default_factory=list)


class _BlockWriter(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.blocks: list[str] = []
        self._out: list[str] = []
        self._stack: list[_Frame] = []
        self._top_run: list[str] = []

    # A container is the top level or a quote: places where children are blocks.
    def _container(self) -> int | None:
        """Index of the container that receives text now (-1 = top level), or None."""
        index = len(self._stack) - 1
        while index >= 0 and self._stack[index].kind == "run":
            index -= 1
        if index == -1 or self._stack[index].kind == "quote":
            return index
        return None

    def _run(self, container: int) -> list[str]:
        return self._top_run if container == -1 else self._stack[container].run

    def _in_run(self, container: int) -> bool:
        return len(self._stack) - 1 > container

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        raw = self.get_starttag_text() or f"<{tag}>"
        container = self._container()
        if container is not None and (self._in_run(container) or tag not in _BLOCK_LEVEL):
            self._run(container).append(raw)
            if tag not in _VOID:
                self._stack.append(_Frame(tag, f"</{tag}>", "run"))
            return
        if container is not None:
            self._flush_run(container)
            opening, closing, kind = _block(tag, attrs, raw)
            if container == -1:
                self._out = []
        else:
            parent = self._stack[-1].kind
            if tag == "li" and parent == "list":
                opening, closing, kind = (
                    "<!-- wp:list-item -->\n" + _open("li", attrs),
                    "</li>\n<!-- /wp:list-item -->",
                    "item",
                )
            elif tag in _LISTS and parent == "item":
                opening, closing, kind = _list(tag, attrs)
            else:
                opening, closing, kind = raw, f"</{tag}>", "raw"
        self._out.append(opening)
        if tag in _VOID:
            if container is not None:  # a void element as a block, such as <hr>
                self._out.append("\n<!-- /wp:html -->")
            self._finish_if_top_level()
            return
        self._stack.append(_Frame(tag, closing, kind))

    def handle_endtag(self, tag: str) -> None:
        if tag in _VOID or not any(frame.tag == tag for frame in self._stack):
            return
        while self._stack:
            frame = self._stack.pop()
            if frame.kind == "run":
                container = self._container()
                self._run(-1 if container is None else container).append(frame.closing)
            else:
                if frame.kind == "quote":
                    self._flush_run(len(self._stack), frame)
                self._out.append(frame.closing)
            if frame.tag == tag:
                break
        self._finish_if_top_level()

    def handle_data(self, data: str) -> None:
        container = self._container()
        if container is None:
            self._out.append(data)
        else:
            self._run(container).append(data)

    def handle_entityref(self, name: str) -> None:
        self.handle_data(f"&{name};")

    def handle_charref(self, name: str) -> None:
        self.handle_data(f"&#{name};")

    def handle_comment(self, data: str) -> None:
        container = self._container()
        if container is None:
            self._out.append(f"<!--{data}-->")
        elif container == -1 and data.strip() == "more":
            self._flush_run(-1)
            self.blocks.append(_MORE)
        # Other comments between blocks would read as broken block markup; drop them.

    def close(self) -> None:
        super().close()
        while self._stack:  # close whatever the input left open
            self.handle_endtag(self._stack[-1].tag)
        self._flush_run(-1)

    def _flush_run(self, container: int, frame: _Frame | None = None) -> None:
        run = frame.run if frame is not None else self._run(container)
        text = " ".join("".join(run).split())
        run.clear()
        if not text:
            return
        if container == -1:
            self.blocks.append(_paragraph(text))
        else:
            self._out.append(_paragraph(text))

    def _finish_if_top_level(self) -> None:
        if not self._stack and self._out:
            self.blocks.append("".join(self._out))
            self._out = []


def _open(tag: str, attrs: list[tuple[str, str | None]], css_class: str = "") -> str:
    classes = " ".join(filter(None, [css_class, *(v or "" for k, v in attrs if k == "class")]))
    parts = [f'class="{escape(classes)}"'] if classes else []
    parts += [f'{k}="{escape(v or "")}"' for k, v in attrs if k != "class"]
    return f"<{tag}{''.join(' ' + p for p in parts)}>"


def _paragraph(text: str) -> str:
    return f"<!-- wp:paragraph -->\n<p>{text}</p>\n<!-- /wp:paragraph -->"


def _block(tag: str, attrs: list[tuple[str, str | None]], raw: str) -> tuple[str, str, str]:
    if tag == "p":
        opening = f"<!-- wp:paragraph -->\n{_open('p', attrs)}"
        return opening, "</p>\n<!-- /wp:paragraph -->", "raw"
    if tag in _HEADINGS:
        level = int(tag[1])
        settings = "" if level == 2 else f' {{"level":{level}}}'
        return (
            f"<!-- wp:heading{settings} -->\n{_open(tag, attrs, 'wp-block-heading')}",
            f"</{tag}>\n<!-- /wp:heading -->",
            "raw",
        )
    if tag in _LISTS:
        return _list(tag, attrs)
    if tag == "blockquote":
        return (
            f"<!-- wp:quote -->\n{_open('blockquote', attrs, 'wp-block-quote')}",
            "</blockquote>\n<!-- /wp:quote -->",
            "quote",
        )
    return f"<!-- wp:html -->\n{raw}", f"</{tag}>\n<!-- /wp:html -->", "raw"


def _list(tag: str, attrs: list[tuple[str, str | None]]) -> tuple[str, str, str]:
    settings = ' {"ordered":true}' if tag == "ol" else ""
    return (
        f"<!-- wp:list{settings} -->\n{_open(tag, attrs, 'wp-block-list')}",
        f"</{tag}>\n<!-- /wp:list -->",
        "list",
    )


def to_blocks(html: str) -> str:
    writer = _BlockWriter()
    writer.feed(html)
    writer.close()
    return "\n\n".join(writer.blocks)
