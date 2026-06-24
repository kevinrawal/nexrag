"""
Token-based cost computation for LLM calls.

NexRAG does not maintain a live pricing table — AI provider pricing changes
frequently. Users supply their own pricing in nexrag.yaml under
observability.metrics.pricing. The observer uses this to compute
llm.cost_per_query_usd from the token counts already in the LLM event.

If no price is configured for a model, cost is not reported (the metric is
simply not emitted for that run). This is intentional — silent omission is
better than a wrong cost estimate.
"""

from __future__ import annotations


def cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    pricing: dict[str, tuple[float, float]],
) -> float | None:
    """
    Compute USD cost for one LLM call.

    Args:
        model:        Model name exactly as returned by the LLM adapter.
        input_tokens: Prompt token count.
        output_tokens: Completion token count.
        pricing:      {model_name: (input_price_per_1k, output_price_per_1k)}.
                      Built by _factory.py from ObservabilityConfig.metrics.pricing.

    Returns:
        USD cost as a float, or None if the model has no configured pricing.
    """
    prices = pricing.get(model)
    if prices is None:
        return None
    input_price, output_price = prices
    return (input_tokens * input_price + output_tokens * output_price) / 1000.0


def build_pricing_table(
    raw: dict[str, object],
) -> dict[str, tuple[float, float]]:
    """
    Build the (input, output) tuple map from schema PricingConfig objects.

    Args:
        raw: {model_name: PricingConfig} from ObservabilityConfig.metrics.pricing.
             PricingConfig has .input and .output fields (float, USD per 1K tokens).
    """
    result: dict[str, tuple[float, float]] = {}
    for model_name, cfg in raw.items():
        result[model_name] = (float(getattr(cfg, "input", 0.0)), float(getattr(cfg, "output", 0.0)))
    return result
