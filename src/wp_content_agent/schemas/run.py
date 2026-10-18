from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class RunStatus(StrEnum):
    PLANNING = "planning"
    GATHERING = "gathering"
    WRITING = "writing"
    REVIEWING = "reviewing"
    AWAITING_APPROVAL = "awaiting_approval"
    PUBLISHING = "publishing"
    DRAFT_CREATED = "draft_created"
    REJECTED = "rejected"
    FAILED = "failed"


class NodeUsage(BaseModel):
    node: str
    model: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cache_read_tokens: int = Field(default=0, ge=0)
    cache_write_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(ge=0)


class ApprovalRequest(BaseModel):
    """What a human sees before the draft is sent to WordPress."""

    title: str
    slug: str
    excerpt: str
    meta_description: str
    content_html: str
    category_names: list[str]
    tag_names: list[str]
    review_passed: bool
    failed_checks: list[str]
    revision_count: int


class ApprovalDecision(BaseModel):
    """The value a run is resumed with after human_approval."""

    # Strict, so a string such as "yes" can never count as an approval.
    model_config = ConfigDict(strict=True, extra="forbid")

    approved: bool
    note: str = Field(default="", max_length=2000)
