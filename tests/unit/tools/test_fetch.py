import json

import httpx
import pytest

from wp_content_agent.errors import FetchError
from wp_content_agent.tools import extract_readable_text, fetch_page, make_fetch_tool
from wp_content_agent.tools.fetch import _normalize

PAGE = """<html><head><title> My  Page </title><style>p{}</style></head><body>
<header>Site header</header><nav>Menu</nav>
<h1>Heading</h1><p>First   paragraph.</p><script>var x = 1;</script>
<ul><li>One</li><li>Two</li></ul><footer>Copyright</footer></body></html>"""


ALLOWED = ["https://example.com/a"]


def _client(transport: httpx.MockTransport) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=transport)


def _html_response(body: str = PAGE) -> httpx.Response:
    return httpx.Response(200, text=body, headers={"content-type": "text/html; charset=utf-8"})


def test_extract_skips_boilerplate_and_keeps_breaks() -> None:
    title, text = extract_readable_text(PAGE)

    assert title == "My Page"
    assert text == "Site header\nHeading\nFirst paragraph.\nOne\nTwo"


def test_title_falls_back_to_h1() -> None:
    title, text = extract_readable_text("<body><h1>Only h1</h1><p>Body</p></body>")

    assert title == "Only h1"
    assert text == "Only h1\nBody"


async def test_fetch_page_truncates() -> None:
    transport = httpx.MockTransport(lambda request: _html_response())
    async with _client(transport) as http:
        page = await fetch_page("https://example.com/a", http, 10, ALLOWED)

    assert page.truncated is True
    assert page.text == "Site heade"


async def test_fetch_page_not_truncated_when_short() -> None:
    transport = httpx.MockTransport(lambda request: _html_response())
    async with _client(transport) as http:
        page = await fetch_page("https://example.com/a", http, 1000, ALLOWED)

    assert page.truncated is False
    assert page.title == "My Page"


async def test_non_2xx_raises() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(404, text="nope"))
    async with _client(transport) as http:
        with pytest.raises(FetchError, match="HTTP 404"):
            await fetch_page("https://example.com/a", http, 100, ALLOWED)


async def test_non_html_content_type_raises() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, content=b"%PDF", headers={"content-type": "application/pdf"}
        )
    )
    async with _client(transport) as http:
        with pytest.raises(FetchError, match="content type"):
            await fetch_page("https://example.com/a", http, 100, ALLOWED)


async def test_timeout_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    async with _client(httpx.MockTransport(handler)) as http:
        with pytest.raises(FetchError, match="timeout"):
            await fetch_page("https://example.com/a", http, 100, ALLOWED)


async def test_empty_text_raises() -> None:
    transport = httpx.MockTransport(
        lambda request: _html_response("<body><script>x</script></body>")
    )
    async with _client(transport) as http:
        with pytest.raises(FetchError, match="no readable text"):
            await fetch_page("https://example.com/a", http, 100, ALLOWED)


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


def _redirect(location: str, status: int = 302) -> httpx.Response:
    return httpx.Response(status, headers={"location": location})


async def test_redirect_to_other_host_is_refused_and_never_requested() -> None:
    evil_calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "evil.test":
            evil_calls.append(request)
            return _html_response()
        return _redirect("http://evil.test/secret")

    async with _client(httpx.MockTransport(handler)) as http:
        with pytest.raises(FetchError, match="redirect to a host outside the brief's sources"):
            await fetch_page("https://example.com/a", http, 100, ALLOWED)

    assert evil_calls == []


async def test_redirect_to_unsupported_scheme_is_refused() -> None:
    transport = httpx.MockTransport(lambda request: _redirect("file:///etc/passwd"))
    async with _client(transport) as http:
        with pytest.raises(FetchError, match="outside the brief's sources"):
            await fetch_page("https://example.com/a", http, 100, ALLOWED)


async def test_redirect_within_same_host_is_followed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/a":
            return _redirect("/b")
        return _html_response()

    async with _client(httpx.MockTransport(handler)) as http:
        page = await fetch_page("https://example.com/a", http, 1000, ALLOWED)

    assert page.url == "https://example.com/b"
    assert page.requested_url == "https://example.com/a"


async def test_redirect_loop_fails() -> None:
    transport = httpx.MockTransport(lambda request: _redirect("/a"))
    async with _client(transport) as http:
        with pytest.raises(FetchError, match="too many redirects"):
            await fetch_page("https://example.com/a", http, 100, ALLOWED)


async def test_body_over_max_bytes_is_truncated() -> None:
    body = "<p>" + "x" * 5000 + "</p>"
    transport = httpx.MockTransport(lambda request: _html_response(body))
    async with _client(transport) as http:
        page = await fetch_page("https://example.com/a", http, 10_000, ALLOWED, max_bytes=100)

    assert page.truncated is True
    assert 0 < len(page.text) <= 100


async def test_content_type_is_checked_before_reading_body() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, content=b"x" * 5_000_000, headers={"content-type": "text/css"}
        )
    )
    async with _client(transport) as http:
        with pytest.raises(FetchError, match="content type"):
            await fetch_page("https://example.com/a", http, 100, ALLOWED)


async def test_plain_text_is_not_parsed_as_html() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, text="a <b>  tag\nand more", headers={"content-type": "text/plain"}
        )
    )
    async with _client(transport) as http:
        page = await fetch_page("https://example.com/a", http, 100, ALLOWED)

    assert page.text == "a <b> tag and more"


async def test_xhtml_is_parsed_as_html() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, text="<p>Hi</p>", headers={"content-type": "application/xhtml+xml"}
        )
    )
    async with _client(transport) as http:
        page = await fetch_page("https://example.com/a", http, 100, ALLOWED)

    assert page.text == "Hi"


async def test_missing_content_type_is_tried_as_html() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=b"<p>Hi</p>"))
    async with _client(transport) as http:
        page = await fetch_page("https://example.com/a", http, 100, ALLOWED)

    assert page.text == "Hi"


def test_header_inside_article_is_kept() -> None:
    title, text = extract_readable_text("<article><header><h1>Title</h1></header><p>Body</p>")

    assert title == "Title"
    assert text == "Title\nBody"


def test_form_wrapped_page_keeps_text() -> None:
    _, text = extract_readable_text("<form><div>Legacy body</div></form>")

    assert text == "Legacy body"


def test_unmatched_nav_end_tag_and_unclosed_script_are_harmless() -> None:
    _, text = extract_readable_text("<p>Before</p></nav><p>After</p><script>var x = 1;")

    assert text == "Before\nAfter"


def test_table_cells_are_separate_words() -> None:
    _, text = extract_readable_text("<table><tr><td>A</td><td>B</td></tr></table>")

    assert text == "A\nB"


def test_source_newlines_are_spaces() -> None:
    _, text = extract_readable_text("<p>one\ntwo</p>")

    assert text == "one two"


def test_title_stops_at_body_when_unclosed_and_is_capped() -> None:
    title, _ = extract_readable_text("<title>" + "t" * 300 + "<body><p>x</p>")

    assert title == "t" * 200


def test_only_first_h1_is_title_fallback() -> None:
    title, _ = extract_readable_text("<h1>First</h1><h1>Second</h1>")

    assert title == "First"


async def test_tool_fetches_allowlisted_original_not_model_string() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return _html_response()

    async with _client(httpx.MockTransport(handler)) as http:
        tool = make_fetch_tool(["https://example.com/a"], http, 1000)
        await tool.ainvoke({"url": "  HTTPS://EXAMPLE.com:443/a/  "})

    assert [str(c.url) for c in calls] == ["https://example.com/a"]


def test_normalize_drops_default_ports_and_keeps_others() -> None:
    assert _normalize("http://Example.com:80") == "http://example.com/"
    assert _normalize("https://example.com:443/a/") == "https://example.com/a"
    assert _normalize("https://example.com:8443/a?q=1#f") == "https://example.com:8443/a?q=1"
    assert _normalize("https://User@Example.com/a") == "https://User@example.com/a"
