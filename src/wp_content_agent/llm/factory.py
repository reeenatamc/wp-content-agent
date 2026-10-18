from collections.abc import Sequence
from typing import Any, Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from wp_content_agent.errors import LLMError
from wp_content_agent.llm.client import ModelClient
from wp_content_agent.schemas import NodeUsage
from wp_content_agent.settings import Settings

Role = Literal["writer", "reviewer"]


def build_chat_model(settings: Settings, model: str) -> BaseChatModel:
    if settings.llm_provider == "anthropic":
        if settings.anthropic_api_key is None:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        return ChatAnthropic(
            model=model,
            api_key=settings.anthropic_api_key,
            timeout=settings.llm_timeout_s,
            max_retries=settings.llm_max_retries,
            max_tokens=settings.llm_max_tokens,
        )
    if settings.openai_api_key is None:
        raise LLMError("OPENAI_API_KEY is not set")
    return ChatOpenAI(
        model=model,
        api_key=settings.openai_api_key,
        timeout=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
        max_tokens=settings.llm_max_tokens,
        # OpenAI leaves usage out of streamed replies unless asked.
        stream_usage=True,
    )


class UnavailableModelClient(ModelClient):
    """Stands in for a model that could not be built, such as one without an API key.

    Commands that never call a model use it, so they work without the key; a call
    still fails with the reason the model was unavailable.
    """

    def __init__(self, model_name: str, reason: str) -> None:
        self.model_name = model_name
        self.cache_prompts = False
        self._reason = reason

    # Read-only on purpose: there is no model, and touching it should say why.
    @property
    def chat_model(self) -> BaseChatModel:  # type: ignore[override]
        raise LLMError(self._reason)

    async def structured[T: BaseModel](
        self, schema: type[T], messages: Sequence[BaseMessage], *, node: str
    ) -> tuple[T, NodeUsage]:
        raise LLMError(f"{node}: {self._reason}")

    async def chat(
        self, messages: Sequence[BaseMessage], tools: Sequence[Any] = (), *, node: str
    ) -> AIMessage:
        raise LLMError(f"{node}: {self._reason}")


def build_model_client(settings: Settings, role: Role, *, required: bool = True) -> ModelClient:
    """With required=False, a model that cannot be built fails on first use instead."""
    model = settings.writer if role == "writer" else settings.reviewer
    try:
        chat_model = build_chat_model(settings, model)
    except LLMError as exc:
        if required:
            raise
        return UnavailableModelClient(model, str(exc))
    return ModelClient(
        chat_model,
        model,
        settings.model_prices,
        cache_prompts=settings.llm_provider == "anthropic",
    )
