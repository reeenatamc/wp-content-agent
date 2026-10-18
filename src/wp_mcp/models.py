"""Draft input and the compact outputs returned to the LLM."""

import html
import re
from typing import Annotated, Any

from pydantic import BaseModel, Field, field_validator

_TAG_RE = re.compile(r"<[^>]*>")

# Rejects blank values without stripping them, so HTML whitespace reaches WordPress as sent.
NonBlank = Annotated[str, Field(min_length=1, pattern=r"\S")]


def clean_text(value: Any) -> str:
    """Unwrap WP's `{"rendered": ...}`, strip HTML tags and unescape entities."""
    if isinstance(value, dict):
        value = value.get("rendered", value.get("raw", ""))
    return html.unescape(_TAG_RE.sub("", str(value or ""))).strip()


class CreateDraftInput(BaseModel):
    title: Annotated[NonBlank, Field(max_length=200)]
    slug: str = Field(default="", max_length=200)
    excerpt: str = ""
    content_html: NonBlank
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


class TermList(BaseModel):
    terms: list[Term]
    # True when the taxonomy has more pages than WP_MAX_TERM_PAGES allows.
    truncated: bool = False


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
    meta_description_saved: bool
    reused: bool = False
