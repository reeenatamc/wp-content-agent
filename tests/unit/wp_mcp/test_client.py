import base64
import json
from collections.abc import Callable

import httpx
import pytest

from wp_mcp.client import WordPressClient
from wp_mcp.errors import WordPressError
from wp_mcp.models import CreateDraftInput
from wp_mcp.settings import WPSettings

Handler = Callable[[httpx.Request], httpx.Response]


@pytest.fixture
def settings() -> WPSettings:
    return WPSettings(
        base_url="http://wp.test",
        username="agent",
        app_password="abcd efgh",  # type: ignore[arg-type]
        max_retries=2,
        backoff_s=0,
    )


@pytest.fixture
def make_client(settings: WPSettings) -> Callable[[Handler], WordPressClient]:
    def _make(handler: Handler) -> WordPressClient:
        http = httpx.AsyncClient(
            base_url="http://wp.test/wp-json/wp/v2",
            auth=httpx.BasicAuth("agent", "abcd efgh"),
            transport=httpx.MockTransport(handler),
        )
        return WordPressClient(http, settings)

    return _make


def _draft(**overrides: object) -> CreateDraftInput:
    base: dict[str, object] = {"title": "T", "content_html": "<p>x</p>"}
    return CreateDraftInput.model_validate({**base, **overrides})


async def test_basic_auth_header_is_sent(make_client: Callable[[Handler], WordPressClient]) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=[])

    await make_client(handler).list_categories()

    expected = base64.b64encode(b"agent:abcd efgh").decode()
    assert seen[0].headers["Authorization"] == f"Basic {expected}"


async def test_from_settings_builds_base_url_and_auth(settings: WPSettings) -> None:
    client = WordPressClient.from_settings(settings)

    assert str(client._http.base_url) == "http://wp.test/wp-json/wp/v2/"
    assert client._http.timeout.read == settings.timeout_s
    await client.aclose()


async def test_draft_status_is_forced_and_meta_uses_configured_key(
    make_client: Callable[[Handler], WordPressClient],
) -> None:
    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(201, json={"id": 7, "status": "draft", "link": "http://wp.test/?p=7"})

    draft = CreateDraftInput.model_construct(
        title="T",
        slug="t",
        excerpt="",
        content_html="<p>x</p>",
        categories=[1],
        tags=[2],
        meta_description="Short",
        status="publish",
    )
    result = await make_client(handler).create_draft_post(draft)

    assert bodies[0]["status"] == "draft"
    assert bodies[0]["meta"] == {"meta_description": "Short"}
    assert bodies[0]["categories"] == [1]
    assert result.edit_url == "http://wp.test/wp-admin/post.php?post=7&action=edit"


async def test_draft_without_meta_description_omits_meta(
    make_client: Callable[[Handler], WordPressClient],
) -> None:
    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(201, json={"id": 1, "status": "draft", "link": "l"})

    await make_client(handler).create_draft_post(_draft())

    assert "meta" not in bodies[0]


async def test_retries_on_503_then_succeeds(
    make_client: Callable[[Handler], WordPressClient],
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503)
        return httpx.Response(200, json=[{"id": 1, "name": "News", "slug": "news"}])

    categories = await make_client(handler).list_categories()

    assert calls == 2
    assert categories[0].name == "News"


async def test_no_retry_on_400(make_client: Callable[[Handler], WordPressClient]) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, json={"code": "rest_invalid_param", "message": "Bad"})

    with pytest.raises(WordPressError) as info:
        await make_client(handler).list_categories()

    assert calls == 1
    assert info.value.status_code == 400
    assert info.value.code == "rest_invalid_param"


async def test_retries_exhausted_raises(
    make_client: Callable[[Handler], WordPressClient],
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503)

    with pytest.raises(WordPressError) as info:
        await make_client(handler).list_categories()

    assert calls == 3
    assert info.value.status_code == 503


async def test_timeout_raises_wordpress_error(
    make_client: Callable[[Handler], WordPressClient],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(WordPressError) as info:
        await make_client(handler).list_categories()

    assert info.value.code == "timeout"
    assert info.value.status_code is None


async def test_create_tag_returns_existing_on_term_exists(
    make_client: Callable[[Handler], WordPressClient],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(
                400,
                json={"code": "term_exists", "message": "exists", "data": {"term_id": 9}},
            )
        assert request.url.path.endswith("/tags/9")
        return httpx.Response(200, json={"id": 9, "name": "Python", "slug": "python"})

    tag = await make_client(handler).create_tag("Python")

    assert (tag.id, tag.slug) == (9, "python")


async def test_html_in_titles_is_unwrapped(
    make_client: Callable[[Handler], WordPressClient],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[
                {
                    "id": 3,
                    "title": {"rendered": "Tom &amp; <em>Jerry</em> &#233;"},
                    "link": "l",
                    "status": "publish",
                }
            ],
        )

    posts = await make_client(handler).list_posts(search="tom", per_page=5)

    assert posts[0].title == "Tom & Jerry é"


async def test_get_post_requests_edit_context_and_cleans_excerpt(
    make_client: Callable[[Handler], WordPressClient],
) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "id": 4,
                "title": {"raw": "T", "rendered": "T"},
                "status": "draft",
                "link": "l",
                "excerpt": {"raw": "x", "rendered": "<p>Summary</p>\n"},
            },
        )

    post = await make_client(handler).get_post(4)

    assert seen[0].url.params["context"] == "edit"
    assert post.excerpt == "Summary"
    assert post.edit_url.endswith("post=4&action=edit")


async def test_draft_post_is_not_retried_after_server_error(
    make_client: Callable[[Handler], WordPressClient],
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(502)

    with pytest.raises(WordPressError):
        await make_client(handler).create_draft_post(_draft())

    assert calls == 1


async def test_draft_post_is_not_retried_after_read_timeout(
    make_client: Callable[[Handler], WordPressClient],
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(WordPressError):
        await make_client(handler).create_draft_post(_draft())

    assert calls == 1


async def test_draft_post_is_retried_when_connection_failed(
    make_client: Callable[[Handler], WordPressClient],
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(201, json={"id": 9, "status": "draft", "link": "http://wp.test/?p=9"})

    created = await make_client(handler).create_draft_post(_draft())

    assert calls == 2
    assert created.id == 9
