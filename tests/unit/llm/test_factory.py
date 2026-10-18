from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

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
