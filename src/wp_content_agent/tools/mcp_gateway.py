"""WordPress access for the graph, through the wp_mcp server over stdio."""

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession
from mcp.shared.exceptions import McpError
from mcp.types import TextContent

from wp_content_agent.errors import ToolError
from wp_content_agent.graph.context import WordPressGateway
from wp_content_agent.schemas import Term
from wp_content_agent.settings import Settings

logger = logging.getLogger(__name__)

# The model only gets read-only tools. Writes happen in publish_draft, in code.
RESEARCH_TOOL_NAMES = frozenset({"list_posts", "list_tags", "get_post"})


class McpWordPress:
    def __init__(
        self, session: ClientSession, tools: list[BaseTool], call_timeout_s: float
    ) -> None:
        self._session = session
        self._tools = [tool for tool in tools if tool.name in RESEARCH_TOOL_NAMES]
        self._timeout = timedelta(seconds=call_timeout_s)

    @property
    def research_tools(self) -> list[Any]:
        return list(self._tools)

    async def list_categories(self) -> list[Term]:
        return _terms(await self._call("list_categories", {}))

    async def list_tags(self, search: str = "") -> list[Term]:
        return _terms(await self._call("list_tags", {"search": search}))

    async def create_tag(self, name: str) -> Term:
        return Term.model_validate(await self._call("create_tag", {"name": name}))

    async def create_draft_post(self, **fields: Any) -> dict[str, Any]:
        return await self._call("create_draft_post", fields)

    async def _call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            result = await self._session.call_tool(name, arguments, self._timeout)
        except McpError as exc:
            raise ToolError(f"{name}: {exc}") from exc
        if result.isError:
            message = " ".join(c.text for c in result.content if isinstance(c, TextContent))
            raise ToolError(f"{name}: {message or 'tool failed'}")
        if result.structuredContent is None:
            raise ToolError(f"{name}: no structured result")
        return result.structuredContent


def _terms(data: dict[str, Any]) -> list[Term]:
    # FastMCP wraps list results as {"result": [...]}.
    return [Term.model_validate(item) for item in data.get("result", [])]


def _server_env() -> dict[str, str]:
    # The MCP SDK passes only a minimal environment to the subprocess, so forward
    # the WordPress settings explicitly. Values in .env are read by the server itself.
    return {key: value for key, value in os.environ.items() if key.startswith("WP_")}


@asynccontextmanager
async def connect_wordpress(settings: Settings) -> AsyncIterator[McpWordPress]:
    """Start the wp_mcp server and keep one session open for the whole run."""
    client = MultiServerMCPClient(
        {
            "wordpress": {
                "transport": "stdio",
                "command": settings.mcp_command,
                "args": settings.mcp_args,
                "env": _server_env(),
            }
        }
    )
    async with client.session("wordpress") as session:
        tools = await load_mcp_tools(session)
        logger.info("connected to wp_mcp with %d tools", len(tools))
        yield McpWordPress(session, tools, settings.mcp_call_timeout_s)


_: type[WordPressGateway] = McpWordPress
