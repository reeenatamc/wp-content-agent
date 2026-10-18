"""Exception types raised by the agent. Each maps to a clear failure the caller can act on."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wp_content_agent.schemas import NodeUsage


class AgentError(Exception):
    """Base class for agent failures."""


class LLMError(AgentError):
    """The model call failed or returned output that does not fit the expected schema.

    `usage` is set when the call was billed before failing, so totals stay accurate.
    """

    def __init__(self, message: str, usage: "NodeUsage | None" = None) -> None:
        super().__init__(message)
        self.usage = usage


class ToolError(AgentError):
    """An MCP tool or the URL fetcher returned an error."""


class FetchError(ToolError):
    """A source URL could not be fetched or had no readable text."""
