from typing import Any

import pytest

from tests.unit.nodes.fake_wordpress import FakeGateway
from tests.unit.nodes.helpers import runtime
from tests.unit.nodes.test_review import GOOD
from wp_content_agent.errors import ToolError
from wp_content_agent.nodes.publish_draft import publish_draft
from wp_content_agent.schemas import CreatedDraft, ResearchNotes, RunStatus, Term

NOTES = ResearchNotes(tags=[Term(id=4, name="Django", slug="django")])


def _state(**draft_updates: Any) -> dict[str, Any]:
    draft = GOOD.model_copy(update={"tag_names": ["django", "caching"], **draft_updates})
    return {"draft": draft, "context": NOTES, "status": RunStatus.PUBLISHING}


async def test_creates_missing_tags_then_the_draft() -> None:
    gateway = FakeGateway(tags=[])

    update = await publish_draft(_state(), runtime(wordpress=gateway))  # type: ignore[arg-type]

    names = [name for name, _ in gateway.calls]
    assert names == ["create_tag", "create_draft_post"]
    fields = gateway.created[0]
    assert fields["tags"] == [4, 100]
    assert fields["content_html"].startswith("<!-- wp:paragraph -->")
    assert update["wp_post_id"] == 8
    assert update["edit_url"].endswith("post=8&action=edit")
    assert update["status"] == RunStatus.DRAFT_CREATED
    assert update["warnings"] == []


async def test_existing_post_id_skips_creation() -> None:
    gateway = FakeGateway()
    state = {**_state(), "wp_post_id": 8}

    update = await publish_draft(state, runtime(wordpress=gateway))  # type: ignore[arg-type]

    assert gateway.calls == []
    assert update == {"status": RunStatus.DRAFT_CREATED}


async def test_failed_tag_is_skipped_with_a_warning() -> None:
    gateway = FakeGateway(tags=[], fail_tags={"caching"})

    update = await publish_draft(_state(), runtime(wordpress=gateway))  # type: ignore[arg-type]

    assert gateway.created[0]["tags"] == [4]
    assert "'caching' was not added" in update["warnings"][0]


async def test_duplicate_tag_names_are_sent_once() -> None:
    gateway = FakeGateway(tags=[])

    await publish_draft(
        _state(tag_names=["Django", "django ", "Caching", "caching"]),  # type: ignore[arg-type]
        runtime(wordpress=gateway),
    )

    assert gateway.created[0]["tags"] == [4, 100]
    assert [args for name, args in gateway.calls if name == "create_tag"] == ["Caching"]


async def test_unsaved_meta_description_adds_warning() -> None:
    class NoMeta(FakeGateway):
        async def create_draft_post(self, **fields: Any) -> CreatedDraft:
            created = await super().create_draft_post(**fields)
            return created.model_copy(update={"meta_description_saved": False})

    update = await publish_draft(_state(), runtime(wordpress=NoMeta()))  # type: ignore[arg-type]

    assert "meta description" in update["warnings"][0]


async def test_reused_draft_adds_warning() -> None:
    class Reused(FakeGateway):
        async def create_draft_post(self, **fields: Any) -> CreatedDraft:
            created = await super().create_draft_post(**fields)
            return created.model_copy(update={"reused": True})

    update = await publish_draft(_state(), runtime(wordpress=Reused()))  # type: ignore[arg-type]

    assert update["status"] == RunStatus.DRAFT_CREATED
    assert update["warnings"] == ["reused existing draft 8 instead of creating a second one"]


async def test_non_draft_status_from_wordpress_raises() -> None:
    class Published(FakeGateway):
        async def create_draft_post(self, **fields: Any) -> CreatedDraft:
            created = await super().create_draft_post(**fields)
            return created.model_copy(update={"status": "publish"})

    update = await publish_draft(_state(), runtime(wordpress=Published()))  # type: ignore[arg-type]

    assert update["status"] == RunStatus.FAILED
    assert update["wp_post_id"] == 8
    assert "instead of draft" in update["error"]


async def test_create_draft_failure_propagates() -> None:
    class Down(FakeGateway):
        async def create_draft_post(self, **fields: Any) -> CreatedDraft:
            raise ToolError("create_draft_post: connection refused")

    with pytest.raises(ToolError):
        await publish_draft(_state(), runtime(wordpress=Down()))  # type: ignore[arg-type]
