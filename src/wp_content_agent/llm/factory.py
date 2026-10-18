from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from wp_content_agent.errors import LLMError
from wp_content_agent.llm.client import ModelClient
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
    )


def build_model_client(settings: Settings, role: Role) -> ModelClient:
    model = settings.writer if role == "writer" else settings.reviewer
    return ModelClient(build_chat_model(settings, model), model, settings.model_prices)
