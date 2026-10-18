"""MCP server (stdio) exposing WordPress operations as tools."""

import logging
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager
from typing import Annotated

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.server.session import ServerSession
from pydantic import Field, StringConstraints

from wp_mcp.client import WordPressClient
from wp_mcp.errors import WordPressError
from wp_mcp.models import CreateDraftInput, DraftCreated, PostDetail, PostSummary, Term
from wp_mcp.settings import WPSettings

logger = logging.getLogger(__name__)

Ctx = Context[ServerSession, WordPressClient]


def _one_line(text: str) -> str:
    return " ".join(text.split())


async def _guard[T](call: Awaitable[T]) -> T:
    try:
        return await call
    except WordPressError as exc:
        logger.warning("WordPress call failed: %s", exc)
        raise ToolError(_one_line(str(exc))) from exc


def build_server(client: WordPressClient | None = None) -> FastMCP:
    """Build the server. An injected client is used as is and not closed on shutdown."""

    @asynccontextmanager
    async def lifespan(_: FastMCP) -> AsyncIterator[WordPressClient]:
        if client is not None:
            yield client
            return
        own = WordPressClient.from_settings(WPSettings())
        try:
            yield own
        finally:
            await own.aclose()

    mcp = FastMCP("wordpress", lifespan=lifespan)

    def wp(ctx: Ctx) -> WordPressClient:
        return ctx.request_context.lifespan_context

    @mcp.tool()
    async def list_posts(
        ctx: Ctx,
        search: Annotated[str, Field(max_length=100)] = "",
        per_page: Annotated[int, Field(ge=1, le=20)] = 10,
    ) -> list[PostSummary]:
        """List published and draft posts, optionally filtered by a search term (max 20)."""
        return await _guard(wp(ctx).list_posts(search, per_page))

    @mcp.tool()
    async def list_categories(ctx: Ctx) -> list[Term]:
        """List all categories with their ids."""
        return await _guard(wp(ctx).list_categories())

    @mcp.tool()
    async def list_tags(ctx: Ctx, search: Annotated[str, Field(max_length=100)] = "") -> list[Term]:
        """List tags with their ids, optionally filtered by a search term."""
        return await _guard(wp(ctx).list_tags(search))

    @mcp.tool()
    async def create_tag(
        ctx: Ctx,
        name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=60)],
    ) -> Term:
        """Create a tag, or return the existing one if the name is taken."""
        return await _guard(wp(ctx).create_tag(name))

    @mcp.tool()
    async def create_draft_post(
        ctx: Ctx,
        title: Annotated[
            str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)
        ],
        content_html: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)],
        slug: Annotated[str, Field(max_length=200)] = "",
        excerpt: str = "",
        categories: list[int] | None = None,
        tags: list[int] | None = None,
        meta_description: Annotated[str, Field(max_length=320)] = "",
    ) -> DraftCreated:
        """Create a post with status draft. Never publishes. Categories and tags are ids."""
        draft = CreateDraftInput(
            title=title,
            content_html=content_html,
            slug=slug,
            excerpt=excerpt,
            categories=categories or [],
            tags=tags or [],
            meta_description=meta_description,
        )
        return await _guard(wp(ctx).create_draft_post(draft))

    @mcp.tool()
    async def get_post(ctx: Ctx, post_id: int) -> PostDetail:
        """Get a post by id, including its edit URL."""
        return await _guard(wp(ctx).get_post(post_id))

    return mcp


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    build_server().run(transport="stdio")


if __name__ == "__main__":
    main()
