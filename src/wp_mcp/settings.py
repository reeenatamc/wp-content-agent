"""WordPress connection settings, loaded from `WP_*` environment variables or `.env`."""

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class WPSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="WP_", env_file=".env", extra="ignore", env_ignore_empty=True
    )

    base_url: str = "http://localhost:8080"
    username: str = Field(min_length=1)
    app_password: SecretStr = Field(min_length=1)
    timeout_s: float = 15
    max_retries: int = 3
    backoff_s: float = 0.5
    # Upper bound for a Retry-After header, so a rate limit cannot stall a run for long.
    max_retry_after_s: float = Field(default=30, ge=0)
    # Categories and tags are read 100 per page, up to this many pages.
    max_term_pages: int = Field(default=10, ge=1)
    meta_description_key: str = "meta_description"
