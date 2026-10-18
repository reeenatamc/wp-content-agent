import json
from typing import Any

from langchain_core.tools import StructuredTool

from wp_content_agent.errors import ToolError
from wp_content_agent.schemas import CreatedDraft, Term, TermList

POSTS = [
    {
        "id": 3,
        "title": "Django caching basics",
        "link": "https://wp.test/caching",
        "status": "publish",
    }
]


def _list_posts(search: str = "", per_page: int = 10) -> str:
    return ""


class FakeGateway:
    """In-memory WordPressGateway that records every call."""

    def __init__(
        self,
        *,
        tags: list[Term] | None = None,
        fail_tags: set[str] | None = None,
        truncated_categories: bool = False,
        truncated_tags: bool = False,
    ) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.tags = [Term(id=4, name="django", slug="django")] if tags is None else tags
        self.fail_tags = fail_tags or set()
        self.truncated_categories = truncated_categories
        self.truncated_tags = truncated_tags
        self.created: list[dict[str, Any]] = []

    @property
    def research_tools(self) -> list[Any]:
        return [
            StructuredTool.from_function(_list_posts, name="list_posts", description="Search posts")
        ]

    async def run_research_tool(self, name: str, arguments: dict[str, Any]) -> str:
        self.calls.append((name, arguments))
        if name == "list_posts":
            return json.dumps({"result": POSTS})
        return f"error: unknown tool {name}"

    async def list_categories(self) -> TermList:
        self.calls.append(("list_categories", None))
        return TermList(
            terms=[Term(id=1, name="Backend", slug="backend")],
            truncated=self.truncated_categories,
        )

    async def list_tags(self, search: str = "") -> TermList:
        self.calls.append(("list_tags", search))
        return TermList(terms=list(self.tags), truncated=self.truncated_tags)

    async def create_tag(self, name: str) -> Term:
        self.calls.append(("create_tag", name))
        if name in self.fail_tags:
            raise ToolError(f"create_tag: denied {name}")
        term = Term(id=100 + len(self.tags), name=name, slug=name.lower())
        self.tags.append(term)
        return term

    async def create_draft_post(self, **fields: Any) -> CreatedDraft:
        self.calls.append(("create_draft_post", fields))
        self.created.append(fields)
        return CreatedDraft(
            id=8,
            status="draft",
            link="https://wp.test/?p=8",
            edit_url="https://wp.test/wp-admin/post.php?post=8&action=edit",
            meta_description_saved=True,
        )
