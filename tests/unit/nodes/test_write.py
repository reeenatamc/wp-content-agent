import pytest
from langchain_core.messages import AIMessage, HumanMessage

from tests.fakes import PRICES, FakeChatModel, ai, fake_client
from tests.unit.nodes.helpers import BRIEF, runtime
from tests.unit.nodes.test_plan import OUTLINE
from wp_content_agent.errors import LLMError
from wp_content_agent.llm import ModelClient
from wp_content_agent.nodes.write import write
from wp_content_agent.schemas import CheckResult, Draft, ResearchNotes, Review, RunStatus, Term

DRAFT = Draft(
    title="Cache Django querysets",
    slug="cache-django-querysets",
    excerpt="When and how to cache.",
    content_html="<p>Django caching.</p>",
    category_ids=[1, 42],
    tag_names=["django"],
    meta_description="m" * 150,
)
NOTES = ResearchNotes(categories=[Term(id=1, name="Backend", slug="backend")])
REVIEW = Review(
    passed=False,
    checks=[CheckResult(name="meta_description_length", passed=False, note="98 chars")],
    fixes=["Lengthen the meta description"],
)


def _state(**extra: object) -> dict[str, object]:
    return {"brief": BRIEF, "outline": OUTLINE, "context": NOTES, **extra}


async def test_first_draft_has_revision_zero_and_moves_to_review() -> None:
    writer = fake_client(DRAFT)

    update = await write(_state(), runtime(writer=writer))  # type: ignore[arg-type]

    assert update["revision_count"] == 0
    assert update["status"] == RunStatus.REVIEWING
    assert update["usage"][0].node == "write"


async def test_unknown_category_ids_are_dropped() -> None:
    writer = fake_client(DRAFT)

    update = await write(_state(), runtime(writer=writer))  # type: ignore[arg-type]

    assert update["draft"].category_ids == [1]


async def test_first_draft_sends_no_revision_turn() -> None:
    writer = fake_client(DRAFT)

    await write(_state(), runtime(writer=writer))  # type: ignore[arg-type]

    assert isinstance(writer.chat_model, FakeChatModel)
    sent = writer.chat_model.calls[0]
    assert len(sent) == 2
    assert "Research notes:" in str(sent[1].content)


async def test_revision_includes_feedback_and_previous_draft() -> None:
    writer = fake_client(DRAFT)
    state = _state(draft=DRAFT, review=REVIEW, revision_count=0)

    update = await write(state, runtime(writer=writer))  # type: ignore[arg-type]

    assert isinstance(writer.chat_model, FakeChatModel)
    *_, previous, last = writer.chat_model.calls[0]
    assert isinstance(previous, AIMessage)
    assert Draft.model_validate_json(str(previous.content)) == DRAFT
    assert isinstance(last, HumanMessage)
    assert "meta_description_length: 98 chars" in str(last.content)
    assert update["revision_count"] == 1


async def test_system_and_task_are_marked_for_caching() -> None:
    model = FakeChatModel.scripted(DRAFT)
    writer = ModelClient(model, "fake-model", PRICES, cache_prompts=True)

    await write(_state(), runtime(writer=writer))  # type: ignore[arg-type]

    system, task = model.calls[0]
    for message in (system, task):
        assert isinstance(message.content, list)
        assert message.content[0]["cache_control"] == {"type": "ephemeral"}


async def test_write_without_context_still_forbids_statistics() -> None:
    writer = fake_client(DRAFT)
    state = {"brief": BRIEF, "outline": OUTLINE}

    update = await write(state, runtime(writer=writer))  # type: ignore[arg-type]

    assert isinstance(writer.chat_model, FakeChatModel)
    assert "Do not state statistics" in str(writer.chat_model.calls[0][1].content)
    assert update["draft"].category_ids == []


async def test_write_raises_llm_error_on_unparseable_output() -> None:
    writer = fake_client(ai("prose instead of json"))

    with pytest.raises(LLMError):
        await write(_state(), runtime(writer=writer))  # type: ignore[arg-type]
