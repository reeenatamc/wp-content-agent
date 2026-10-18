from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from mcp.client.session import ClientSession
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session

from wp_mcp.client import WordPressClient
from wp_mcp.errors import WordPressError
from wp_mcp.models import CreateDraftInput, DraftCreated, PostSummary, Term
from wp_mcp.server import build_server
from wp_mcp.settings import WPSettings


class FakeClient(WordPressClient):
    def __init__(self, http: httpx.AsyncClient) -> None:
        settings = WPSettings(
            base_url="http://wp.test",
            username="agent",
            app_password="x",  # type: ignore[arg-type]
        )
        super().__init__(http, settings)
        self.drafts: list[CreateDraftInput] = []

    async def list_posts(self, search: str = "", per_page: int = 10) -> list[PostSummary]:
        return [PostSummary(id=1, title=f"{search}:{per_page}", link="l", status="draft")]

    async def create_tag(self, name: str) -> Term:
        if name == "boom":
            raise WordPressError("denied", status_code=401, code="rest_cannot_create")
        return Term(id=5, name=name, slug=name.lower())

    async def create_draft_post(self, draft: CreateDraftInput) -> DraftCreated:
        self.drafts.append(draft)
        return DraftCreated(
            id=8,
            status="draft",
            link="l",
            edit_url=self.edit_url(8),
            meta_description_saved=False,
        )


@pytest.fixture
async def fake() -> AsyncIterator[FakeClient]:
    http = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    yield FakeClient(http)
    await http.aclose()


@pytest.fixture
def server(fake: FakeClient) -> FastMCP:
    return build_server(fake)


async def _call(server: FastMCP, name: str, args: dict[str, Any]) -> Any:
    async with create_connected_server_and_client_session(server) as session:
        assert isinstance(session, ClientSession)
        return await session.call_tool(name, args)


async def test_lists_the_six_tools(server: FastMCP) -> None:
    async with create_connected_server_and_client_session(server) as session:
        tools = await session.list_tools()

    assert {t.name for t in tools.tools} == {
        "list_posts",
        "list_categories",
        "list_tags",
        "create_tag",
        "create_draft_post",
        "get_post",
    }


async def test_list_posts_returns_structured_content(server: FastMCP) -> None:
    result = await _call(server, "list_posts", {"search": "ai", "per_page": 3})

    assert not result.isError
    assert result.structuredContent["result"][0]["title"] == "ai:3"


async def test_create_draft_post_passes_validated_input(server: FastMCP, fake: FakeClient) -> None:
    result = await _call(
        server,
        "create_draft_post",
        {"title": "T", "content_html": "<p>x</p>", "tags": [1], "meta_description": "m"},
    )

    assert not result.isError
    assert result.structuredContent["status"] == "draft"
    assert fake.drafts[0].tags == [1]
    assert fake.drafts[0].meta_description == "m"


async def test_wordpress_error_becomes_tool_error(server: FastMCP) -> None:
    result = await _call(server, "create_tag", {"name": "boom"})

    assert result.isError
    assert "rest_cannot_create" in result.content[0].text
    assert "\n" not in result.content[0].text


async def test_invalid_input_becomes_tool_error(server: FastMCP) -> None:
    result = await _call(server, "list_posts", {"per_page": 99})

    assert result.isError
    assert "per_page" in result.content[0].text


async def test_blank_tag_name_is_rejected(server: FastMCP) -> None:
    result = await _call(server, "create_tag", {"name": "   "})

    assert result.isError


async def test_per_page_limit_is_in_the_tool_schema(server: FastMCP) -> None:
    async with create_connected_server_and_client_session(server) as session:
        tools = await session.list_tools()

    list_posts = next(t for t in tools.tools if t.name == "list_posts")
    assert list_posts.inputSchema["properties"]["per_page"]["maximum"] == 20
