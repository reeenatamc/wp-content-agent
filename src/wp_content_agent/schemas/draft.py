import re
import unicodedata

from pydantic import BaseModel, Field, field_validator

_NON_SLUG = re.compile(r"[^a-z0-9]+")


def slugify(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return _NON_SLUG.sub("-", ascii_value.lower()).strip("-")


class Draft(BaseModel):
    title: str = Field(min_length=5, max_length=120)
    slug: str = Field(description="URL slug, lowercase words joined by hyphens")
    excerpt: str = Field(max_length=400, description="One or two sentences for listings")
    content_html: str = Field(
        description="Post body as semantic HTML: <p>, <h2>, <h3>, <ul>, <ol>, <li>, <a>, "
        "<strong>, <em>, <blockquote>. No <h1>, no inline styles, no scripts."
    )
    category_ids: list[int] = Field(
        default_factory=list, description="IDs of existing categories only"
    )
    tag_names: list[str] = Field(default_factory=list, max_length=8)
    # Length is checked by the review node rather than here, so a miss triggers a
    # revision instead of a hard parse failure.
    meta_description: str = Field(description="SEO meta description, 140 to 160 characters")

    @field_validator("slug")
    @classmethod
    def _normalize_slug(cls, value: str) -> str:
        slug = slugify(value)
        if not slug:
            raise ValueError("slug has no usable characters")
        return slug
