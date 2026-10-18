import pytest

from tests.fakes import FakeChatModel, ai, fake_client
from tests.unit.nodes.helpers import BRIEF, runtime
from wp_content_agent.errors import LLMError
from wp_content_agent.nodes.plan import plan
from wp_content_agent.schemas import Outline, OutlineSection, RunStatus

OUTLINE = Outline(
    working_title="Cache querysets without stale data",
    angle="Cache where it pays off",
    sections=[
        OutlineSection(heading="When caching helps", level=2, key_points=["read-heavy views"]),
        OutlineSection(heading="Invalidation", level=2, key_points=["signals"]),
    ],
)


async def test_plan_returns_outline_usage_and_next_status() -> None:
    writer = fake_client(OUTLINE)

    update = await plan({"brief": BRIEF}, runtime(writer=writer))

    assert update["outline"] == OUTLINE
    assert update["status"] == RunStatus.GATHERING
    assert update["usage"][0].node == "plan"


async def test_plan_sends_the_brief_to_the_writer() -> None:
    writer = fake_client(OUTLINE)

    await plan({"brief": BRIEF}, runtime(writer=writer))

    assert isinstance(writer.chat_model, FakeChatModel)
    sent = writer.chat_model.calls[0]
    assert "Caching Django querysets" in str(sent[-1].content)


async def test_plan_raises_llm_error_when_output_does_not_parse() -> None:
    writer = fake_client(ai("not json"))

    with pytest.raises(LLMError):
        await plan({"brief": BRIEF}, runtime(writer=writer))
