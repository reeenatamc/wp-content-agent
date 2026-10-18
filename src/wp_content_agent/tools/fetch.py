import logging
import re
from collections.abc import Iterable
from html.parser import HTMLParser
from urllib.parse import urlsplit, urlunsplit

import httpx
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel

from wp_content_agent.errors import FetchError

logger = logging.getLogger(__name__)

_SKIPPED_TAGS = frozenset(
    {"script", "style", "noscript", "nav", "header", "footer", "aside", "form", "svg"}
)
_BLOCK_TAGS = frozenset(
    {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "br", "div", "section", "article", "tr"}
)
_WHITESPACE = re.compile(r"[ \t\r\f\v\xa0]+")


class PageText(BaseModel):
    url: str
    title: str
    text: str
    truncated: bool


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._in_title = False
        self._in_h1 = False
        self.title_parts: list[str] = []
        self.h1_parts: list[str] = []
        self.chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIPPED_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = True
        elif tag == "h1":
            self._in_h1 = True
        if tag in _BLOCK_TAGS:
            self.chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIPPED_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = False
        elif tag == "h1":
            self._in_h1 = False
        if tag in _BLOCK_TAGS:
            self.chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self.title_parts.append(data)
            return
        if self._in_h1:
            self.h1_parts.append(data)
        self.chunks.append(data)


def _clean(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def extract_readable_text(html: str) -> tuple[str, str]:
    """Return (title, text) with boilerplate tags dropped and block boundaries as newlines."""
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    title = _clean(" ".join(parser.title_parts)) or _clean(" ".join(parser.h1_parts))
    lines = (_clean(line) for line in "".join(parser.chunks).split("\n"))
    return title, "\n".join(line for line in lines if line)


async def fetch_page(url: str, http: httpx.AsyncClient, max_chars: int) -> PageText:
    try:
        response = await http.get(url, follow_redirects=True)
    except httpx.TimeoutException as exc:
        raise FetchError(f"timeout fetching {url}") from exc
    except httpx.HTTPError as exc:
        raise FetchError(f"could not fetch {url}: {type(exc).__name__}") from exc

    final_url = response.url
    if final_url.scheme not in ("http", "https"):
        raise FetchError(f"redirected to unsupported scheme {final_url.scheme!r}: {url}")
    if not response.is_success:
        raise FetchError(f"HTTP {response.status_code} fetching {url}")
    content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
    if not content_type.startswith("text/"):
        raise FetchError(f"unsupported content type {content_type or 'unknown'!r}: {url}")

    title, text = extract_readable_text(response.text)
    if not text:
        raise FetchError(f"no readable text at {url}")
    truncated = len(text) > max_chars
    logger.info("fetched %s (%d chars, truncated=%s)", final_url, len(text), truncated)
    return PageText(url=url, title=title, text=text[:max_chars], truncated=truncated)


def _normalize(url: str) -> str:
    parts = urlsplit(url.strip())
    path = parts.path.rstrip("/")
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, parts.query, ""))


def make_fetch_tool(
    allowed_urls: Iterable[str], http: httpx.AsyncClient, max_chars: int
) -> BaseTool:
    allowed = {_normalize(u) for u in allowed_urls}

    async def fetch_url(url: str) -> str:
        if _normalize(url) not in allowed:
            logger.warning("refused non-allowlisted url: %s", url)
            return f"error: url not in the brief's source_urls: {url}"
        try:
            page = await fetch_page(url, http, max_chars)
        except FetchError as exc:
            logger.warning("fetch failed: %s", exc)
            return f"error: {exc}"
        return page.model_dump_json()

    return StructuredTool.from_function(
        coroutine=fetch_url,
        name="fetch_url",
        description=(
            "Fetch the readable text of a web page. Only the source URLs listed in the "
            "content brief can be fetched; any other URL is refused. Returns JSON with "
            "url, title, text and truncated, or an error message."
        ),
    )
