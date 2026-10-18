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


class RunNotFoundError(AgentError):
    """No run exists with the given id."""


class RunStateError(AgentError):
    """The run is not in a state that allows the requested action."""


class RunStoppedError(AgentError):
    """A run was stopped from outside before it finished."""


class RunLimitError(AgentError):
    """Too many runs are in progress to start another one."""


class BriefError(AgentError):
    """A brief file is missing, is not valid YAML or JSON, or does not match the schema."""
