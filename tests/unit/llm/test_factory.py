import pytest
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from wp_content_agent.errors import LLMError
from wp_content_agent.llm import build_model_client
from wp_content_agent.settings import Settings


def test_anthropic_reviewer_uses_reviewer_model() -> None:
    settings = Settings(_env_file=None, anthropic_api_key="test")

    client = build_model_client(settings, "reviewer")

    assert isinstance(client.chat_model, ChatAnthropic)
    assert client.model_name == "claude-haiku-5-5"


def test_openai_provider_builds_openai_model() -> None:
    settings = Settings(
        _env_file=None,
        llm_provider="openai",
        openai_api_key="test",
        writer_model="w",
        reviewer_model="r",
    )

    client = build_model_client(settings, "writer")

    assert isinstance(client.chat_model, ChatOpenAI)
    assert client.model_name == "w"


def test_missing_api_key_fails_at_build_time(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    settings = Settings(_env_file=None)

    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        build_model_client(settings, "writer")


def test_empty_api_key_counts_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    settings = Settings(_env_file=None)

    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        build_model_client(settings, "writer")


def test_cache_prompts_enabled_only_for_anthropic() -> None:
    anthropic_settings = Settings(_env_file=None, anthropic_api_key="test")
    openai_settings = Settings(
        _env_file=None,
        llm_provider="openai",
        openai_api_key="test",
        writer_model="w",
        reviewer_model="r",
    )

    assert build_model_client(anthropic_settings, "writer").cache_prompts is True
    assert build_model_client(openai_settings, "writer").cache_prompts is False
