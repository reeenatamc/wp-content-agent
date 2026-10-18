"""WordPress connection settings, loaded from `WP_*` environment variables or `.env`."""

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class WPSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="WP_", env_file=".env", extra="ignore")

    base_url: str = "http://localhost:8080"
    username: str = ""
    app_password: SecretStr = SecretStr("")
    timeout_s: float = 15
    max_retries: int = 3
    backoff_s: float = 0.5
    meta_description_key: str = "meta_description"
