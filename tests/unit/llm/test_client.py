import anthropic
import httpx
import pytest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from tests.fakes import PRICES, FakeChatModel, ai, fake_client
from wp_content_agent.errors import LLMError
from wp_content_agent.llm import ModelClient, estimate_cost
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

    with pytest.raises(LLMError, match="did not match Answer") as info:
        await client.structured(Answer, [HumanMessage("q")], node="plan")

    assert info.value.usage is not None
    assert info.value.usage.input_tokens == 10


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


def test_dated_model_name_uses_base_price() -> None:
    prices = {"claude-sonnet-5-5": ModelPrice(input=2, output=10)}

    cost = estimate_cost("claude-sonnet-5-5-20261001", 1_000_000, 0, prices)

    assert cost == 2.0


def test_cost_prices_cached_tokens_with_explicit_prices() -> None:
    prices = {"m": ModelPrice(input=2.0, output=10.0, cache_read=0.5, cache_write=3.0)}

    cost = estimate_cost("m", 1000, 100, prices, cache_read_tokens=600, cache_write_tokens=200)

    assert cost == pytest.approx((200 * 2.0 + 600 * 0.5 + 200 * 3.0 + 100 * 10.0) / 1_000_000)


def test_cost_cache_prices_default_to_input_multiples() -> None:
    prices = {"m": ModelPrice(input=2.0, output=10.0)}

    cost = estimate_cost("m", 1000, 0, prices, cache_read_tokens=600, cache_write_tokens=200)

    assert cost == pytest.approx((200 * 2.0 + 600 * 0.2 + 200 * 2.5) / 1_000_000)


def test_cost_uncached_input_floors_at_zero() -> None:
    prices = {"m": ModelPrice(input=1.0, output=1.0)}

    cost = estimate_cost("m", 10, 0, prices, cache_read_tokens=20)

    assert cost == pytest.approx(20 * 0.1 / 1_000_000)


def test_usage_sums_cache_tokens_from_input_token_details() -> None:
    client = fake_client()

    usage = client.usage("n", [ai(cache=(4, 2)), ai(cache=(1, 0))])

    assert (usage.cache_read_tokens, usage.cache_write_tokens) == (5, 2)
    assert usage.input_tokens == 20


def test_usage_without_token_details_has_no_cache_tokens() -> None:
    client = fake_client()

    usage = client.usage("n", [ai()])

    assert (usage.cache_read_tokens, usage.cache_write_tokens) == (0, 0)


def test_system_marks_prompt_for_caching_when_enabled() -> None:
    client = ModelClient(FakeChatModel.scripted(), "fake-model", PRICES, cache_prompts=True)

    message = client.system("rules")

    assert message.content == [
        {"type": "text", "text": "rules", "cache_control": {"type": "ephemeral"}}
    ]


def test_system_is_plain_text_without_caching() -> None:
    message = fake_client().system("rules")

    assert message.content == "rules"
