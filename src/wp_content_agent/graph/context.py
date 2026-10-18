from dataclasses import dataclass
from typing import Any, Protocol

from wp_content_agent.llm import ModelClient
from wp_content_agent.schemas import Term
from wp_content_agent.settings import Settings


class WordPressGateway(Protocol):
    """The WordPress operations the graph needs, served by the MCP server."""

    @property
    def research_tools(self) -> list[Any]:
        """Read-only LangChain tools the gather_context model may call."""
        ...

    async def list_categories(self) -> list[Term]: ...

    async def list_tags(self, search: str = "") -> list[Term]: ...

    async def create_tag(self, name: str) -> Term: ...

    async def create_draft_post(self, **fields: Any) -> dict[str, Any]: ...


@dataclass(frozen=True)
class AgentContext:
    """Run-scoped dependencies. Passed as LangGraph runtime context, never checkpointed."""

    settings: Settings
    writer: ModelClient
    reviewer: ModelClient
    wordpress: WordPressGateway
    extra_tools: tuple[Any, ...] = ()
