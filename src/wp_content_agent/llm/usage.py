"""Token accounting. Costs are estimates from the configured price table, not billing data."""

import logging
from collections.abc import Iterable, Mapping

from langchain_core.messages import AIMessage, BaseMessage

from wp_content_agent.schemas import NodeUsage
from wp_content_agent.settings import ModelPrice

logger = logging.getLogger(__name__)

_warned_models: set[str] = set()


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    prices: Mapping[str, ModelPrice],
    *,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    price = _find_price(model, prices)
    if price is None:
        if model not in _warned_models:
            _warned_models.add(model)
            logger.warning("no price configured for model %s; cost reported as 0", model)
        return 0.0
    # langchain reports input_tokens including cache reads and writes, so the uncached
    # part is what remains after subtracting them.
    uncached = max(input_tokens - cache_read_tokens - cache_write_tokens, 0)
    total = (
        uncached * price.input
        + cache_read_tokens * price.cache_read_price
        + cache_write_tokens * price.cache_write_price
        + output_tokens * price.output
    )
    return total / 1_000_000


def _find_price(model: str, prices: Mapping[str, ModelPrice]) -> ModelPrice | None:
    if model in prices:
        return prices[model]
    # Dated snapshots such as "claude-sonnet-5-5-20261001" share the base model's price.
    matches = [name for name in prices if model.startswith(f"{name}-")]
    return prices[max(matches, key=len)] if matches else None


def usage_from_messages(
    node: str, model: str, messages: Iterable[BaseMessage], prices: Mapping[str, ModelPrice]
) -> NodeUsage:
    input_tokens = output_tokens = cache_read = cache_write = 0
    for message in messages:
        if isinstance(message, AIMessage) and message.usage_metadata:
            input_tokens += message.usage_metadata["input_tokens"]
            output_tokens += message.usage_metadata["output_tokens"]
            details = message.usage_metadata.get("input_token_details") or {}
            cache_read += details.get("cache_read") or 0
            cache_write += details.get("cache_creation") or 0
    usage = NodeUsage(
        node=node,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        cost_usd=estimate_cost(
            model,
            input_tokens,
            output_tokens,
            prices,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
        ),
    )
    logger.info(
        "node=%s model=%s input_tokens=%d output_tokens=%d "
        "cache_read_tokens=%d cache_write_tokens=%d cost_usd=%.5f",
        node,
        model,
        input_tokens,
        output_tokens,
        cache_read,
        cache_write,
        usage.cost_usd,
    )
    return usage


def merge_usage(first: NodeUsage, second: NodeUsage) -> NodeUsage:
    """Sum two entries for the same node and model, such as a tool loop and its final call."""
    if (first.node, first.model) != (second.node, second.model):
        raise ValueError(
            f"cannot merge usage for {first.node}/{first.model} and {second.node}/{second.model}"
        )
    return first.model_copy(
        update={
            "input_tokens": first.input_tokens + second.input_tokens,
            "output_tokens": first.output_tokens + second.output_tokens,
            "cache_read_tokens": first.cache_read_tokens + second.cache_read_tokens,
            "cache_write_tokens": first.cache_write_tokens + second.cache_write_tokens,
            "cost_usd": first.cost_usd + second.cost_usd,
        }
    )
