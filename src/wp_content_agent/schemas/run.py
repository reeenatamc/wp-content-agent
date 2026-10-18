from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, computed_field


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


class RunSnapshot(BaseModel):
    """Where a run stands, as the CLI and the API report it."""

    run_id: str
    status: RunStatus
    approval: ApprovalRequest | None = None
    wp_post_id: int | None = None
    edit_url: str | None = None
    review_passed: bool | None = None
    revision_count: int = 0
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
    usage: list[NodeUsage] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_cost_usd(self) -> float:
        return round(sum(entry.cost_usd for entry in self.usage), 6)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_tokens(self) -> int:
        return sum(entry.input_tokens + entry.output_tokens for entry in self.usage)
