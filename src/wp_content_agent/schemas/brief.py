from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, HttpUrl, TypeAdapter

_HTTP_URL = TypeAdapter(HttpUrl)


def _http_url(value: str) -> str:
    # Validated as a URL but kept as str: the checkpointer cannot serialize HttpUrl.
    return str(_HTTP_URL.validate_python(value))


HttpUrlStr = Annotated[str, AfterValidator(_http_url)]


class Brief(BaseModel):
    """What the user asks for. Everything the graph writes must trace back to this."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    topic: str = Field(min_length=3, max_length=200)
    audience: str = Field(min_length=3, max_length=200)
    tone: str = Field(default="practical", max_length=60)
    target_words: int = Field(default=900, ge=300, le=3000)
    keywords: list[str] = Field(default_factory=list, max_length=10)
    source_urls: list[HttpUrlStr] = Field(default_factory=list, max_length=5)
