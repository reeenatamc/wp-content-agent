"""Tool input models and the compact outputs returned to the LLM."""

import html
import re
from typing import Annotated, Any

from pydantic import BaseModel, Field, StringConstraints, field_validator

_TAG_RE = re.compile(r"<[^>]*>")


def clean_text(value: Any) -> str:
    """Unwrap WP's `{"rendered": ...}`, strip HTML tags and unescape entities."""
    if isinstance(value, dict):
        value = value.get("rendered", value.get("raw", ""))
    return html.unescape(_TAG_RE.sub("", str(value or ""))).strip()


class ListPostsInput(BaseModel):
    search: str = Field(default="", max_length=100)
    per_page: int = Field(default=10, ge=1, le=20)


class ListTagsInput(BaseModel):
    search: str = Field(default="", max_length=100)


class CreateTagInput(BaseModel):
    name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=60)]


class CreateDraftInput(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    slug: str = Field(default="", max_length=200)
    excerpt: str = ""
    content_html: str = Field(min_length=1)
    categories: list[int] = Field(default_factory=list)
    tags: list[int] = Field(default_factory=list)
    meta_description: str = Field(default="", max_length=320)


class PostSummary(BaseModel):
    id: int
    title: str
    link: str
    status: str

    _clean_title = field_validator("title", mode="before")(clean_text)


class Term(BaseModel):
    id: int
    name: str
    slug: str


class PostDetail(BaseModel):
    id: int
    title: str
    status: str
    link: str
    edit_url: str
    excerpt: str = ""

    _clean = field_validator("title", "excerpt", mode="before")(clean_text)


class DraftCreated(BaseModel):
    id: int
    status: str
    link: str
    edit_url: str
