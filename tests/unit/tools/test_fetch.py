import json

import httpx
import pytest

from wp_content_agent.errors import FetchError
from wp_content_agent.tools import extract_readable_text, fetch_page, make_fetch_tool

PAGE = """<html><head><title> My  Page </title><style>p{}</style></head><body>
<header>Site header</header><nav>Menu</nav>
<h1>Heading</h1><p>First   paragraph.</p><script>var x = 1;</script>
<ul><li>One</li><li>Two</li></ul><footer>Copyright</footer></body></html>"""


def _client(transport: httpx.MockTransport) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=transport)


def _html_response(body: str = PAGE) -> httpx.Response:
    return httpx.Response(200, text=body, headers={"content-type": "text/html; charset=utf-8"})


def test_extract_skips_boilerplate_and_keeps_breaks() -> None:
    title, text = extract_readable_text(PAGE)

    assert title == "My Page"
    assert text == "Heading\nFirst paragraph.\nOne\nTwo"


def test_title_falls_back_to_h1() -> None:
    title, text = extract_readable_text("<body><h1>Only h1</h1><p>Body</p></body>")

    assert title == "Only h1"
    assert text == "Only h1\nBody"


async def test_fetch_page_truncates() -> None:
    transport = httpx.MockTransport(lambda request: _html_response())
    async with _client(transport) as http:
        page = await fetch_page("https://example.com/a", http, max_chars=10)

    assert page.truncated is True
    assert page.text == "Heading\nFi"


async def test_fetch_page_not_truncated_when_short() -> None:
    transport = httpx.MockTransport(lambda request: _html_response())
    async with _client(transport) as http:
        page = await fetch_page("https://example.com/a", http, max_chars=1000)

    assert page.truncated is False
    assert page.title == "My Page"


async def test_non_2xx_raises() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(404, text="nope"))
    async with _client(transport) as http:
        with pytest.raises(FetchError, match="HTTP 404"):
            await fetch_page("https://example.com/a", http, 100)


async def test_non_html_content_type_raises() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, content=b"%PDF", headers={"content-type": "application/pdf"}
        )
    )
    async with _client(transport) as http:
        with pytest.raises(FetchError, match="content type"):
            await fetch_page("https://example.com/a", http, 100)


async def test_timeout_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    async with _client(httpx.MockTransport(handler)) as http:
        with pytest.raises(FetchError, match="timeout"):
            await fetch_page("https://example.com/a", http, 100)


async def test_empty_text_raises() -> None:
    transport = httpx.MockTransport(
        lambda request: _html_response("<body><script>x</script></body>")
    )
    async with _client(transport) as http:
        with pytest.raises(FetchError, match="no readable text"):
            await fetch_page("https://example.com/a", http, 100)


async def test_tool_refuses_non_allowlisted_url_without_request() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return _html_response()

    async with _client(httpx.MockTransport(handler)) as http:
        tool = make_fetch_tool(["https://example.com/a"], http, 100)
        result = await tool.ainvoke({"url": "http://169.254.169.254/latest/meta-data"})

    assert "not in the brief's source_urls" in result
    assert calls == []


async def test_tool_allows_url_differing_by_slash_fragment_and_host_case() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return _html_response()

    async with _client(httpx.MockTransport(handler)) as http:
        tool = make_fetch_tool(["https://example.com/a"], http, 1000)
        result = await tool.ainvoke({"url": "HTTPS://Example.COM/a/#section"})

    assert len(calls) == 1
    assert json.loads(result)["title"] == "My Page"


async def test_tool_returns_error_text_on_fetch_failure() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(500))
    async with _client(transport) as http:
        tool = make_fetch_tool(["https://example.com/a"], http, 100)
        result = await tool.ainvoke({"url": "https://example.com/a"})

    assert result.startswith("error:")
    assert "HTTP 500" in result
