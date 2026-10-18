from typing import Literal

from pydantic import BaseModel, Field


class OutlineSection(BaseModel):
    heading: str = Field(description="Section heading, written as it should appear in the post")
    level: Literal[2, 3] = Field(description="2 for a main section, 3 for a subsection")
    key_points: list[str] = Field(description="What this section must cover, 1 to 5 points")


class Outline(BaseModel):
    working_title: str
    angle: str = Field(description="One sentence on the take that makes this post useful")
    sections: list[OutlineSection] = Field(min_length=2)
