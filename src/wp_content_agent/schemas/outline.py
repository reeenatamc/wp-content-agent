from typing import Literal

from pydantic import BaseModel, Field, field_validator


class OutlineSection(BaseModel):
    heading: str = Field(description="Section heading, written as it should appear in the post")
    level: Literal[2, 3] = Field(description="2 for a main section, 3 for a subsection")
    key_points: list[str] = Field(description="What this section must cover, 1 to 5 points")


class Outline(BaseModel):
    working_title: str
    angle: str = Field(description="One sentence on the take that makes this post useful")
    sections: list[OutlineSection] = Field(min_length=2)

    @field_validator("sections")
    @classmethod
    def _starts_with_h2(cls, sections: list[OutlineSection]) -> list[OutlineSection]:
        if sections[0].level != 2:
            raise ValueError("the first section must be an H2")
        return sections
