"""Runtime configuration for the agent, loaded from the environment or `.env`."""

import logging
import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["anthropic", "openai"]

_ANTHROPIC_DEFAULTS = {"writer": "claude-sonnet-5-5", "reviewer": "claude-haiku-5-5"}


class ModelPrice(BaseModel):
    """USD per million tokens. Unset cache prices fall back to the plain input price."""

    input: float = Field(ge=0)
    output: float = Field(ge=0)
    cache_read: float | None = Field(default=None, ge=0)
    cache_write: float | None = Field(default=None, ge=0)

    @property
    def cache_read_price(self) -> float:
        return self.cache_read if self.cache_read is not None else self.input

    @property
    def cache_write_price(self) -> float:
        return self.cache_write if self.cache_write is not None else self.input


def _default_prices() -> dict[str, ModelPrice]:
    return {
        # Anthropic 5-minute cache: reads at 0.1x input, writes at 1.25x.
        "claude-sonnet-5-5": ModelPrice(input=2.0, output=10.0, cache_read=0.2, cache_write=2.5),
        "claude-haiku-5-5": ModelPrice(input=0.10, output=0.50, cache_read=0.01, cache_write=0.125),
        "claude-opus-5-5": ModelPrice(input=4.0, output=20.0, cache_read=0.4, cache_write=5.0),
    }


class Settings(BaseSettings):
    # .env.example ships lines such as `ANTHROPIC_API_KEY=`. An empty value means unset,
    # so it falls back to the default instead of becoming an empty key or token.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_ignore_empty=True)

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
    # When set, every /runs endpoint needs "Authorization: Bearer <token>". Without it the
    # API only accepts a loopback host.
    api_token: SecretStr | None = None
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=0, le=65535)
    max_concurrent_runs: int = Field(default=3, ge=1)
    log_level: str = "INFO"

    @field_validator("log_level")
    @classmethod
    def _known_log_level(cls, value: str) -> str:
        level = value.upper()
        if level not in logging.getLevelNamesMapping():
            raise ValueError(f"unknown log level {value!r}")
        return level

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
