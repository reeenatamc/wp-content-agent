import pytest
from pydantic import ValidationError

from wp_content_agent.settings import Settings


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("LLM_PROVIDER", "WRITER_MODEL", "REVIEWER_MODEL", "MODEL_PRICES"):
        monkeypatch.delenv(name, raising=False)


def test_anthropic_defaults_fill_models() -> None:
    settings = Settings(_env_file=None)

    assert settings.writer == "claude-sonnet-5-5"
    assert settings.reviewer == "claude-haiku-5-5"


def test_openai_requires_explicit_models() -> None:
    with pytest.raises(ValidationError, match="WRITER_MODEL"):
        Settings(_env_file=None, llm_provider="openai")


def test_openai_accepts_explicit_models() -> None:
    settings = Settings(_env_file=None, llm_provider="openai", writer_model="w", reviewer_model="r")

    assert (settings.writer, settings.reviewer) == ("w", "r")


def test_model_prices_can_be_overridden_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_PRICES", '{"my-model": {"input": 1, "output": 2}}')

    settings = Settings(_env_file=None)

    assert settings.model_prices["my-model"].output == 2


@pytest.mark.parametrize("name", ["API_TOKEN", "WEBHOOK_SECRET", "OPENAI_API_KEY"])
def test_blank_secrets_are_unset(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    monkeypatch.setenv(name, "  ")

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert getattr(settings, name.lower()) is None
