from enum import StrEnum

from pydantic import BaseModel, Field


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
    cost_usd: float = Field(ge=0)
