"""Exception types raised by the agent. Each maps to a clear failure the caller can act on."""


class AgentError(Exception):
    """Base class for agent failures."""


class LLMError(AgentError):
    """The model call failed or returned output that does not fit the expected schema."""


class ToolError(AgentError):
    """An MCP tool or the URL fetcher returned an error."""


class FetchError(ToolError):
    """A source URL could not be fetched or had no readable text."""
