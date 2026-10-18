from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from wp_content_agent.llm import ModelClient
from wp_content_agent.schemas import CreatedDraft, Term
from wp_content_agent.settings import Settings


class WordPressGateway(Protocol):
    """The WordPress operations the graph needs, served by the MCP server."""

    @property
    def research_tools(self) -> list[Any]:
        """Read-only LangChain tools whose schemas are bound to the gather_context model."""
        ...

    async def run_research_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Run one research tool call; failures come back as text for the model."""
        ...

    async def list_categories(self) -> list[Term]: ...

    async def list_tags(self, search: str = "") -> list[Term]: ...

    async def create_tag(self, name: str) -> Term: ...

    async def create_draft_post(self, **fields: Any) -> CreatedDraft: ...


@dataclass(frozen=True)
class AgentContext:
    """Run-scoped dependencies. Passed as LangGraph runtime context, never checkpointed."""

    settings: Settings
    writer: ModelClient
    reviewer: ModelClient
    wordpress: WordPressGateway
    extra_tools: tuple[Any, ...] = ()
    http: httpx.AsyncClient | None = None
