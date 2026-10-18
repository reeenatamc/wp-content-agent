"""Runtime configuration for the agent, loaded from the environment or `.env`."""

import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["anthropic", "openai"]

_ANTHROPIC_DEFAULTS = {"writer": "claude-sonnet-5-5", "reviewer": "claude-haiku-5-5"}


class ModelPrice(BaseModel):
    """USD per million tokens."""

    input: float = Field(ge=0)
    output: float = Field(ge=0)


def _default_prices() -> dict[str, ModelPrice]:
    return {
        "claude-sonnet-5-5": ModelPrice(input=2.0, output=10.0),
        "claude-haiku-5-5": ModelPrice(input=0.10, output=0.50),
        "claude-opus-5-5": ModelPrice(input=4.0, output=20.0),
    }


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: LLMProvider = "anthropic"
    writer_model: str | None = None
    reviewer_model: str | None = None
    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    llm_timeout_s: float = Field(default=120.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0)
    llm_max_tokens: int = Field(default=16_000, ge=1024)
    model_prices: dict[str, ModelPrice] = Field(default_factory=_default_prices)

    max_revisions: int = Field(default=2, ge=0)
    max_tool_iterations: int = Field(default=6, ge=1)

    fetch_timeout_s: float = Field(default=15.0, gt=0)
    fetch_max_chars: int = Field(default=8000, ge=500)

    mcp_command: str = Field(default=sys.executable)
    mcp_args: list[str] = Field(default_factory=lambda: ["-m", "wp_mcp.server"])
    mcp_call_timeout_s: float = Field(default=60.0, gt=0)

    checkpoint_db: Path = Path("wpagent.sqlite")
    webhook_secret: SecretStr | None = None
    log_level: str = "INFO"

    @model_validator(mode="after")
    def _resolve_models(self) -> Self:
        if self.llm_provider == "anthropic":
            self.writer_model = self.writer_model or _ANTHROPIC_DEFAULTS["writer"]
            self.reviewer_model = self.reviewer_model or _ANTHROPIC_DEFAULTS["reviewer"]
        elif not (self.writer_model and self.reviewer_model):
            # No OpenAI defaults on purpose: model names change too often to guess.
            raise ValueError(
                "WRITER_MODEL and REVIEWER_MODEL are required when LLM_PROVIDER=openai"
            )
        return self

    @property
    def writer(self) -> str:
        assert self.writer_model is not None
        return self.writer_model

    @property
    def reviewer(self) -> str:
        assert self.reviewer_model is not None
        return self.reviewer_model


@lru_cache
def get_settings() -> Settings:
    return Settings()
