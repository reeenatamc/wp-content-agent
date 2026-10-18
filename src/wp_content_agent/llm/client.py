from collections.abc import Mapping, Sequence
from typing import Any, cast

import anthropic
import openai
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel

from wp_content_agent.errors import LLMError
from wp_content_agent.llm.usage import usage_from_messages
from wp_content_agent.schemas import NodeUsage
from wp_content_agent.settings import ModelPrice

_PROVIDER_ERRORS = (anthropic.APIError, openai.APIError)


class ModelClient:
    """A chat model plus the bookkeeping every node needs: typed output and usage."""

    def __init__(
        self, chat_model: BaseChatModel, model_name: str, prices: Mapping[str, ModelPrice]
    ) -> None:
        self.chat_model = chat_model
        self.model_name = model_name
        self._prices = prices

    async def structured[T: BaseModel](
        self, schema: type[T], messages: Sequence[BaseMessage], *, node: str
    ) -> tuple[T, NodeUsage]:
        # json_schema uses the provider's native structured output. The default
        # (forced tool call) is rejected by current Claude models.
        runnable = self.chat_model.with_structured_output(
            schema, method="json_schema", include_raw=True
        )
        try:
            result = cast(dict[str, Any], await runnable.ainvoke(list(messages)))
        except _PROVIDER_ERRORS as exc:
            raise LLMError(f"{node}: model call failed: {exc}") from exc

        raw = cast(AIMessage, result["raw"])
        usage = self.usage(node, [raw])
        parsed = result.get("parsed")
        if result.get("parsing_error") is not None or not isinstance(parsed, schema):
            raise LLMError(f"{node}: output did not match {schema.__name__}")
        return parsed, usage

    async def chat(self, messages: Sequence[BaseMessage], tools: Sequence[Any] = ()) -> AIMessage:
        model = self.chat_model.bind_tools(list(tools)) if tools else self.chat_model
        try:
            return await model.ainvoke(list(messages))
        except _PROVIDER_ERRORS as exc:
            raise LLMError(f"model call failed: {exc}") from exc

    def usage(self, node: str, messages: Sequence[BaseMessage]) -> NodeUsage:
        return usage_from_messages(node, self.model_name, messages, self._prices)
