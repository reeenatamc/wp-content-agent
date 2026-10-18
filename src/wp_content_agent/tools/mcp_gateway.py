"""WordPress access for the graph, through the wp_mcp server over stdio."""

import json
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any

import anyio
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession
from mcp.shared.exceptions import McpError
from mcp.types import TextContent
from pydantic import ValidationError

from wp_content_agent.errors import ToolError
from wp_content_agent.graph.context import WordPressGateway
from wp_content_agent.schemas import CreatedDraft, Term, TermList
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

    async def list_categories(self) -> TermList:
        return _term_list("list_categories", await self._call("list_categories", {}))

    async def list_tags(self, search: str = "") -> TermList:
        return _term_list("list_tags", await self._call("list_tags", {"search": search}))

    async def create_tag(self, name: str) -> Term:
        return Term.model_validate(await self._call("create_tag", {"name": name}))

    async def create_draft_post(self, **fields: Any) -> CreatedDraft:
        data = await self._call("create_draft_post", fields)
        try:
            return CreatedDraft.model_validate(data)
        except ValidationError as exc:
            raise ToolError(f"create_draft_post: unexpected result: {exc}") from exc

    async def run_research_tool(self, name: str, arguments: dict[str, Any]) -> str:
        if name not in RESEARCH_TOOL_NAMES:
            return f"error: unknown tool {name}"
        try:
            return json.dumps(await self._call(name, arguments))
        except ToolError as exc:
            return f"error: {exc}"

    async def _call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            result = await self._session.call_tool(name, arguments, self._timeout)
        except McpError as exc:
            raise ToolError(f"{name}: {exc}") from exc
        except (anyio.ClosedResourceError, anyio.BrokenResourceError, OSError) as exc:
            raise ToolError(f"{name}: lost connection to wp_mcp: {exc!r}") from exc
        if result.isError:
            message = " ".join(c.text for c in result.content if isinstance(c, TextContent))
            raise ToolError(f"{name}: {message or result.model_dump_json(exclude_none=True)}")
        if not isinstance(result.structuredContent, dict):
            raise ToolError(
                f"{name}: expected a structured object, got {result.structuredContent!r}"
            )
        return result.structuredContent


def _term_list(name: str, data: dict[str, Any]) -> TermList:
    try:
        return TermList.model_validate(data)
    except ValidationError as exc:
        raise ToolError(f"{name}: unexpected result: {exc}") from exc


def _server_env() -> dict[str, str]:
    # The MCP SDK passes only a minimal environment to the subprocess, so forward
    # the WordPress settings explicitly. Values in .env are read by the server itself.
    return {key: value for key, value in os.environ.items() if key.startswith("WP_")}


@asynccontextmanager
async def connect_wordpress(settings: Settings) -> AsyncIterator[WordPressGateway]:
    """Start the wp_mcp server and keep one session open for the whole run.

    The server reads WP_* from the environment or from `.env` in the working directory.
    """
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
    try:
        session_cm = client.session("wordpress")
        session = await session_cm.__aenter__()
    except Exception as exc:
        raise ToolError(
            "could not start wp_mcp; check WP_BASE_URL, WP_USERNAME and WP_APP_PASSWORD "
            f"in the environment or .env: {exc!r}"
        ) from exc
    try:
        tools = await load_mcp_tools(session)
        logger.info("connected to wp_mcp with %d tools", len(tools))
        yield McpWordPress(session, tools, settings.mcp_call_timeout_s)
    finally:
        await session_cm.__aexit__(None, None, None)
