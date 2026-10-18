from typing import Literal

from pydantic import BaseModel, Field

CheckName = Literal[
    "matches_brief",
    "structure",
    "keyword_usage",
    "readability",
    "no_unsupported_claims",
    "meta_description_length",
    "allowed_html",
]


class CheckResult(BaseModel):
    name: CheckName
    passed: bool
    note: str = Field(default="", description="Why it failed, or empty when it passed")


class ReviewVerdict(BaseModel):
    """What the reviewer model returns. Mechanical checks are added in code afterwards."""

    checks: list[CheckResult]
    fixes: list[str] = Field(description="Concrete edits the writer should make")


class Review(BaseModel):
    passed: bool
    checks: list[CheckResult]
    fixes: list[str]

    @property
    def failed_checks(self) -> list[CheckResult]:
        return [check for check in self.checks if not check.passed]
