from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import pytest
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp.shared.memory import create_connected_server_and_client_session

from tests.fakes import FakeWordPressClient
from wp_content_agent.errors import ToolError
from wp_content_agent.tools.mcp_gateway import McpWordPress, _server_env
from wp_mcp.server import build_server


@asynccontextmanager
async def connected_gateway() -> AsyncIterator[McpWordPress]:
    # A context manager rather than a yield fixture: the MCP session's task group must
    # be entered and exited in the same task.
    http = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    server = build_server(FakeWordPressClient(http))
    async with create_connected_server_and_client_session(server) as session:
        tools = await load_mcp_tools(session)
        try:
            yield McpWordPress(session, tools, call_timeout_s=5)
        finally:
            await http.aclose()


async def test_model_only_gets_read_only_tools() -> None:
    async with connected_gateway() as gateway:
        names = {tool.name for tool in gateway.research_tools}

    assert names == {"list_posts", "list_tags", "get_post"}


async def test_list_results_are_parsed_into_term_lists() -> None:
    async with connected_gateway() as gateway:
        categories = await gateway.list_categories()
        tags = await gateway.list_tags("dj")

    assert [c.slug for c in categories.terms] == ["backend"]
    assert [t.name for t in tags.terms] == ["django"]
    assert not categories.truncated
    assert not tags.truncated


async def test_truncated_flag_reaches_the_gateway() -> None:
    http = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    fake = FakeWordPressClient(http)
    fake.truncated_terms = True
    async with create_connected_server_and_client_session(build_server(fake)) as session:
        gateway = McpWordPress(session, await load_mcp_tools(session), call_timeout_s=5)
        tags = await gateway.list_tags()
    await http.aclose()

    assert tags.truncated


async def test_create_draft_post_returns_structured_result() -> None:
    async with connected_gateway() as gateway:
        created = await gateway.create_draft_post(title="T", content_html="<p>x</p>", tags=[4])

    assert created.id == 8
    assert created.status == "draft"


async def test_tool_errors_become_tool_error() -> None:
    async with connected_gateway() as gateway:
        with pytest.raises(ToolError, match=r"create_tag: .*denied"):
            await gateway.create_tag("boom")


def test_server_env_forwards_only_wordpress_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WP_USERNAME", "agent")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")

    env = _server_env()

    assert env["WP_USERNAME"] == "agent"
    assert "ANTHROPIC_API_KEY" not in env


async def test_research_tool_returns_json_for_the_model() -> None:
    async with connected_gateway() as gateway:
        text = await gateway.run_research_tool("list_posts", {"search": "cache"})

    assert '"cache:10"' in text


async def test_research_tool_refuses_write_tools() -> None:
    async with connected_gateway() as gateway:
        text = await gateway.run_research_tool("create_draft_post", {"title": "x"})

    assert text.startswith("error: unknown tool")
