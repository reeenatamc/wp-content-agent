import logging
import re
from collections.abc import Collection, Iterable
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel

from wp_content_agent.errors import FetchError

logger = logging.getLogger(__name__)

_SKIPPED_TAGS = frozenset(
    {"script", "style", "noscript", "template", "svg", "nav", "footer", "iframe"}
)
_BLOCK_TAGS = frozenset(
    {
        "p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "ul", "ol", "br", "div", "section",
        "article", "tr", "td", "th", "table", "blockquote", "pre", "dd", "dt", "figcaption",
        "header", "main",
    }
)  # fmt: skip
_HTML_TYPES = frozenset({"text/html", "application/xhtml+xml"})
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_DEFAULT_PORTS = {"http": 80, "https": 443}
_MAX_REDIRECTS = 5
_MAX_TITLE_CHARS = 200
_WHITESPACE = re.compile(r"\s+")


class PageText(BaseModel):
    url: str
    requested_url: str
    title: str
    text: str
    truncated: bool


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skipped: list[str] = []
        self._in_title = False
        self._in_h1 = False
        self._h1_done = False
        self.title_parts: list[str] = []
        self.h1_parts: list[str] = []
        self.chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIPPED_TAGS:
            self._skipped.append(tag)
            return
        if self._skipped:
            return
        if self._in_title and (tag == "body" or tag in _BLOCK_TAGS):
            self._in_title = False  # </title> is missing
        if tag == "title":
            self._in_title = True
        elif tag == "h1" and not self._h1_done:
            self._in_h1 = True
        if tag in _BLOCK_TAGS:
            self.chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self._skipped:
            if tag in self._skipped:
                while self._skipped.pop() != tag:
                    pass
            return
        if tag in _SKIPPED_TAGS:
            return  # unmatched end tag
        if tag == "title":
            self._in_title = False
        elif tag == "h1" and self._in_h1:
            self._in_h1 = False
            self._h1_done = True
        if tag in _BLOCK_TAGS:
            self.chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skipped:
            return
        if self._in_title:
            self.title_parts.append(data)
            return
        data = data.replace("\n", " ")
        if self._in_h1:
            self.h1_parts.append(data)
        self.chunks.append(data)


def _clean(text: str) -> str:
    return _WHITESPACE.sub(" ", text.replace("\xa0", " ")).strip()


def extract_readable_text(html: str) -> tuple[str, str]:
    """Return (title, text) with boilerplate tags dropped and block boundaries as newlines."""
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    title = _clean(" ".join(parser.title_parts)) or _clean(" ".join(parser.h1_parts))
    lines = (_clean(line) for line in "".join(parser.chunks).split("\n"))
    return title[:_MAX_TITLE_CHARS], "\n".join(line for line in lines if line)


def _host_key(url: str) -> str | None:
    """Normalized host (with port only when not the scheme default), or None if unusable."""
    try:
        parts = urlsplit(url.strip())
        host = parts.hostname
        port = parts.port
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    if scheme not in _DEFAULT_PORTS or not host:
        return None
    if port is None or port == _DEFAULT_PORTS[scheme]:
        return host.lower()
    return f"{host.lower()}:{port}"


def _normalize(url: str) -> str:
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower()
    userinfo, at, hostport = parts.netloc.rpartition("@")
    host, colon, port = hostport.rpartition(":")
    if not colon or not port.isdigit():
        host, port = hostport, ""
    elif int(port) == _DEFAULT_PORTS.get(scheme):
        port = ""
    netloc = f"{userinfo}{at}{host.lower()}" + (f":{port}" if port else "")
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((scheme, netloc, path, parts.query, ""))


def _decode(raw: bytes, encoding: str | None) -> str:
    try:
        return raw.decode(encoding or "utf-8", errors="replace")
    except LookupError:
        return raw.decode("utf-8", errors="replace")


async def fetch_page(
    url: str,
    http: httpx.AsyncClient,
    max_chars: int,
    allowed_urls: Collection[str],
    max_bytes: int = 2_000_000,
) -> PageText:
    """Fetch url, following redirects by hand and only toward hosts of allowed_urls."""
    hosts = {h for u in allowed_urls if (h := _host_key(u))}
    current = url
    for _ in range(_MAX_REDIRECTS + 1):
        host = _host_key(current)
        if host is None or host not in hosts:
            raise FetchError(f"redirect to a host outside the brief's sources: {current}")
        try:
            async with http.stream("GET", current, follow_redirects=False) as response:
                if response.status_code in _REDIRECT_STATUSES and "location" in response.headers:
                    current = urljoin(current, response.headers["location"])
                    continue
                return await _read_page(response, url, current, max_chars, max_bytes)
        except httpx.TimeoutException as exc:
            raise FetchError(f"timeout fetching {current}") from exc
        except httpx.InvalidURL as exc:
            raise FetchError(f"invalid url {current}") from exc
        except httpx.HTTPError as exc:
            raise FetchError(f"could not fetch {current}: {type(exc).__name__}") from exc
    raise FetchError(f"too many redirects (max {_MAX_REDIRECTS}): {url}")


async def _read_page(
    response: httpx.Response, requested: str, final: str, max_chars: int, max_bytes: int
) -> PageText:
    if not response.is_success:
        raise FetchError(f"HTTP {response.status_code} fetching {final}")
    content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
    is_plain = content_type == "text/plain"
    if content_type and content_type not in _HTML_TYPES and not is_plain:
        raise FetchError(f"unsupported content type {content_type!r}: {final}")

    buffer = bytearray()
    byte_truncated = False
    async for chunk in response.aiter_bytes():
        buffer.extend(chunk)
        if len(buffer) > max_bytes:
            del buffer[max_bytes:]
            byte_truncated = True
            break

    body = _decode(bytes(buffer), response.encoding)
    if is_plain:
        title, text = "", _clean(body)
    else:
        title, text = extract_readable_text(body)
    if not text:
        raise FetchError(f"no readable text at {final}")
    truncated = byte_truncated or len(text) > max_chars
    logger.info("fetched %s (%d chars, truncated=%s)", final, len(text), truncated)
    return PageText(
        url=final, requested_url=requested, title=title, text=text[:max_chars], truncated=truncated
    )


def make_fetch_tool(
    allowed_urls: Iterable[str], http: httpx.AsyncClient, max_chars: int
) -> BaseTool:
    originals = {_normalize(u): u.strip() for u in allowed_urls}

    async def fetch_url(url: str) -> str:
        original = originals.get(_normalize(url))
        if original is None:
            logger.warning("refused non-allowlisted url: %s", url)
            return f"error: url not in the brief's source_urls: {url}"
        try:
            page = await fetch_page(original, http, max_chars, list(originals.values()))
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
            "url, requested_url, title, text and truncated, or an error message."
        ),
    )
