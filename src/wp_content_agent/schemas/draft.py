import re
import unicodedata

from pydantic import BaseModel, Field, field_validator

_NON_SLUG = re.compile(r"[^a-z0-9]+")

# Shared by the writer prompt, the schema descriptions and the review checks.
META_MIN, META_MAX = 140, 160
ALLOWED_TAGS = ("p", "h2", "h3", "ul", "ol", "li", "a", "strong", "em", "blockquote", "code")
ALLOWED_TAGS_TEXT = ", ".join(f"<{tag}>" for tag in ALLOWED_TAGS)


def slugify(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return _NON_SLUG.sub("-", ascii_value.lower()).strip("-")


class Draft(BaseModel):
    title: str = Field(min_length=5, max_length=120)
    slug: str = Field(description="URL slug, lowercase words joined by hyphens")
    excerpt: str = Field(max_length=400, description="One or two sentences for listings")
    content_html: str = Field(
        description=f"Post body as semantic HTML using only {ALLOWED_TAGS_TEXT}. "
        "No <h1>, no inline styles, no scripts."
    )
    category_ids: list[int] = Field(
        default_factory=list, description="IDs of existing categories only"
    )
    tag_names: list[str] = Field(default_factory=list, max_length=8)
    # Length is checked by the review node rather than here, so a miss triggers a
    # revision instead of a hard parse failure.
    meta_description: str = Field(
        description=f"SEO meta description, {META_MIN} to {META_MAX} characters"
    )

    @field_validator("slug")
    @classmethod
    def _normalize_slug(cls, value: str) -> str:
        slug = slugify(value)
        if not slug:
            raise ValueError("slug has no usable characters")
        return slug
