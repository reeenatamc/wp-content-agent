from pydantic import BaseModel, Field


class Term(BaseModel):
    """A WordPress category or tag."""

    id: int
    name: str
    slug: str


class SourceNote(BaseModel):
    url: str
    title: str
    facts: list[str] = Field(
        description="Claims stated in the source, close to verbatim, that the post may cite"
    )


class RelatedPost(BaseModel):
    id: int
    title: str
    link: str


class ResearchNotes(BaseModel):
    sources: list[SourceNote] = Field(default_factory=list)
    related_posts: list[RelatedPost] = Field(
        default_factory=list, description="Existing posts that overlap with the topic"
    )
    categories: list[Term] = Field(default_factory=list)
    tags: list[Term] = Field(default_factory=list)
