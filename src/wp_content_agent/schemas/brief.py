from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class Brief(BaseModel):
    """What the user asks for. Everything the graph writes must trace back to this."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    topic: str = Field(min_length=3, max_length=200)
    audience: str = Field(min_length=3, max_length=200)
    tone: str = Field(default="practical", max_length=60)
    target_words: int = Field(default=900, ge=300, le=3000)
    keywords: list[str] = Field(default_factory=list, max_length=10)
    source_urls: list[HttpUrl] = Field(default_factory=list, max_length=5)
