"""Token accounting. Costs are estimates from the configured price table, not billing data."""

import logging
from collections.abc import Iterable, Mapping

from langchain_core.messages import AIMessage, BaseMessage

from wp_content_agent.schemas import NodeUsage
from wp_content_agent.settings import ModelPrice

logger = logging.getLogger(__name__)


def estimate_cost(
    model: str, input_tokens: int, output_tokens: int, prices: Mapping[str, ModelPrice]
) -> float:
    price = _find_price(model, prices)
    if price is None:
        logger.warning("no price configured for model %s; cost reported as 0", model)
        return 0.0
    return (input_tokens * price.input + output_tokens * price.output) / 1_000_000


def _find_price(model: str, prices: Mapping[str, ModelPrice]) -> ModelPrice | None:
    if model in prices:
        return prices[model]
    # Dated snapshots such as "claude-sonnet-5-5-20261001" share the base model's price.
    matches = [name for name in prices if model.startswith(f"{name}-")]
    return prices[max(matches, key=len)] if matches else None


def usage_from_messages(
    node: str, model: str, messages: Iterable[BaseMessage], prices: Mapping[str, ModelPrice]
) -> NodeUsage:
    input_tokens = output_tokens = 0
    for message in messages:
        if isinstance(message, AIMessage) and message.usage_metadata:
            input_tokens += message.usage_metadata["input_tokens"]
            output_tokens += message.usage_metadata["output_tokens"]
    usage = NodeUsage(
        node=node,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=estimate_cost(model, input_tokens, output_tokens, prices),
    )
    logger.info(
        "node=%s model=%s input_tokens=%d output_tokens=%d cost_usd=%.5f",
        node,
        model,
        input_tokens,
        output_tokens,
        usage.cost_usd,
    )
    return usage
