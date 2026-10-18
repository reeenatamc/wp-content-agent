import anthropic
import httpx
import pytest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from tests.fakes import ai, fake_client
from wp_content_agent.errors import LLMError
from wp_content_agent.llm import estimate_cost
from wp_content_agent.settings import ModelPrice


class Answer(BaseModel):
    value: str


async def test_structured_returns_parsed_output_and_usage() -> None:
    client = fake_client(Answer(value="ok"))

    answer, usage = await client.structured(Answer, [HumanMessage("q")], node="plan")

    assert answer.value == "ok"
    assert (usage.node, usage.input_tokens, usage.output_tokens) == ("plan", 10, 5)
    assert usage.cost_usd == pytest.approx((10 * 1.0 + 5 * 2.0) / 1_000_000)


async def test_structured_raises_when_output_does_not_parse() -> None:
    client = fake_client(ai("not json"))

    with pytest.raises(LLMError, match="did not match Answer"):
        await client.structured(Answer, [HumanMessage("q")], node="plan")


async def test_provider_errors_become_llm_errors() -> None:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    client = fake_client(anthropic.APIConnectionError(request=request))

    with pytest.raises(LLMError, match="model call failed"):
        await client.chat([HumanMessage("q")])


def test_usage_sums_tokens_across_messages() -> None:
    client = fake_client()

    usage = client.usage("gather_context", [ai(tokens=(100, 20)), ai(tokens=(50, 10))])

    assert (usage.input_tokens, usage.output_tokens) == (150, 30)


def test_unknown_model_costs_zero() -> None:
    assert estimate_cost("unknown", 1000, 1000, {"m": ModelPrice(input=1, output=1)}) == 0.0
