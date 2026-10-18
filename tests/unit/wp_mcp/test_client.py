import base64
import json
from collections.abc import Callable

import httpx
import pytest
from pydantic import ValidationError

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


async def test_non_json_success_body_raises_invalid_response(
    make_client: Callable[[Handler], WordPressClient],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>maintenance</html>")

    with pytest.raises(WordPressError) as info:
        await make_client(handler).list_categories()

    assert info.value.code == "invalid_response"


async def test_draft_response_missing_keys_raises_invalid_response(
    make_client: Callable[[Handler], WordPressClient],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(201, json={"id": 1})

    with pytest.raises(WordPressError) as info:
        await make_client(handler).create_draft_post(_draft())

    assert info.value.code == "invalid_response"


async def test_get_post_with_missing_keys_raises_invalid_response(
    make_client: Callable[[Handler], WordPressClient],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": 4})

    with pytest.raises(WordPressError) as info:
        await make_client(handler).get_post(4)

    assert info.value.code == "invalid_response"


async def test_draft_post_is_retried_after_pool_timeout(
    make_client: Callable[[Handler], WordPressClient],
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.PoolTimeout("no free connection", request=request)
        return httpx.Response(201, json={"id": 9, "status": "draft", "link": "l"})

    created = await make_client(handler).create_draft_post(_draft())

    assert calls == 2
    assert created.id == 9


async def test_connection_error_code_for_transport_failure(
    make_client: Callable[[Handler], WordPressClient],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(WordPressError) as info:
        await make_client(handler).list_categories()

    assert info.value.code == "connection_error"
    assert isinstance(info.value.__cause__, httpx.ConnectError)


async def test_meta_description_saved_when_echoed_in_response(
    make_client: Callable[[Handler], WordPressClient],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            201,
            json={"id": 1, "status": "draft", "link": "l", "meta": {"meta_description": "Short"}},
        )

    created = await make_client(handler).create_draft_post(_draft(meta_description="Short"))

    assert created.meta_description_saved is True


async def test_meta_description_not_saved_logs_warning(
    make_client: Callable[[Handler], WordPressClient], caplog: pytest.LogCaptureFixture
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(201, json={"id": 1, "status": "draft", "link": "l", "meta": []})

    with caplog.at_level("WARNING"):
        created = await make_client(handler).create_draft_post(_draft(meta_description="Short"))

    assert created.meta_description_saved is False
    assert "meta_description" in caplog.text
    assert "wpagent-meta.php" in caplog.text


async def test_meta_description_saved_is_false_when_not_sent(
    make_client: Callable[[Handler], WordPressClient],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(201, json={"id": 1, "status": "draft", "link": "l"})

    created = await make_client(handler).create_draft_post(_draft())

    assert created.meta_description_saved is False


async def test_terms_follow_total_pages_header(
    make_client: Callable[[Handler], WordPressClient],
) -> None:
    pages: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        page = request.url.params["page"]
        pages.append(page)
        term = {"id": int(page), "name": f"T{page}", "slug": f"t{page}"}
        return httpx.Response(200, json=[term], headers={"X-WP-TotalPages": "3"})

    tags = await make_client(handler).list_tags()

    assert pages == ["1", "2", "3"]
    assert [tag.id for tag in tags] == [1, 2, 3]


async def test_terms_stop_at_max_pages_with_warning(
    settings: WPSettings, caplog: pytest.LogCaptureFixture
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=[], headers={"X-WP-TotalPages": "50"})

    http = httpx.AsyncClient(base_url="http://wp.test", transport=httpx.MockTransport(handler))
    client = WordPressClient(http, settings.model_copy(update={"max_term_pages": 2}))

    with caplog.at_level("WARNING"):
        await client.list_categories()

    assert calls == 2
    assert "WP_MAX_TERM_PAGES" in caplog.text


async def test_terms_without_total_pages_header_read_one_page(
    make_client: Callable[[Handler], WordPressClient],
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=[{"id": 1, "name": "News", "slug": "news"}])

    await make_client(handler).list_categories()

    assert calls == 1


def _rate_limited(retry_after: str) -> Handler:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": retry_after})
        return httpx.Response(200, json=[])

    return handler


async def test_retry_after_seconds_is_used_and_capped(
    settings: WPSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("wp_mcp.client.asyncio.sleep", fake_sleep)
    capped = settings.model_copy(update={"max_retry_after_s": 5})
    for value, client_settings in (("2", settings), ("120", capped)):
        http = httpx.AsyncClient(
            base_url="http://wp.test", transport=httpx.MockTransport(_rate_limited(value))
        )
        await WordPressClient(http, client_settings).list_tags()

    assert sleeps == [2.0, 5.0]


async def test_retry_after_in_the_past_waits_for_backoff(
    settings: WPSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("wp_mcp.client.asyncio.sleep", fake_sleep)
    handler = _rate_limited("Wed, 21 Oct 2015 07:28:00 GMT")
    http = httpx.AsyncClient(base_url="http://wp.test", transport=httpx.MockTransport(handler))

    await WordPressClient(http, settings.model_copy(update={"backoff_s": 0.25})).list_tags()

    assert sleeps == [0.25]


@pytest.mark.parametrize("value", ["soon", "nan", "inf"])
async def test_invalid_retry_after_falls_back_to_backoff(
    value: str, settings: WPSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("wp_mcp.client.asyncio.sleep", fake_sleep)
    http = httpx.AsyncClient(
        base_url="http://wp.test", transport=httpx.MockTransport(_rate_limited(value))
    )

    await WordPressClient(http, settings.model_copy(update={"backoff_s": 0.25})).list_tags()

    assert sleeps == [0.25]


async def test_terms_repeated_across_pages_are_deduped(
    make_client: Callable[[Handler], WordPressClient],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params["page"])
        items = [
            {"id": page, "name": f"T{page}", "slug": f"t{page}"},
            {"id": 1, "name": "T1", "slug": "t1"},
        ]
        return httpx.Response(200, json=items, headers={"X-WP-TotalPages": "2"})

    tags = await make_client(handler).list_tags()

    assert [tag.id for tag in tags] == [1, 2]


def test_term_pages_must_be_positive() -> None:
    with pytest.raises(ValueError):
        WPSettings(username="u", app_password="p", max_term_pages=0)  # type: ignore[arg-type]


def test_empty_credentials_in_env_are_reported_as_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WP_USERNAME", "")
    monkeypatch.setenv("WP_APP_PASSWORD", "")

    with pytest.raises(ValidationError, match="username\n  Field required"):
        WPSettings(_env_file=None)  # type: ignore[call-arg]
