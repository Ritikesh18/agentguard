"""Static pricing table for LLM models."""

from typing import Optional

# Pricing structure: {model_id: {'input': price_per_1k_tokens, 'output': price_per_1k_tokens}}
# All prices in USD as of 2026
PRICING = {
    # OpenAI — GPT-4o family
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    # OpenAI — GPT-4.1 family (released 2025)
    "gpt-4.1": {"input": 0.002, "output": 0.008},
    "gpt-4.1-mini": {"input": 0.0004, "output": 0.0016},
    "gpt-4.1-nano": {"input": 0.0001, "output": 0.0004},
    # OpenAI — reasoning models
    "o1": {"input": 0.015, "output": 0.06},
    "o1-mini": {"input": 0.003, "output": 0.012},
    "o3": {"input": 0.01, "output": 0.04},
    "o3-mini": {"input": 0.0011, "output": 0.0044},
    "o4-mini": {"input": 0.0011, "output": 0.0044},
    # OpenAI — legacy
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    # Anthropic — Claude 4 family (2025/2026)
    "claude-opus-4": {"input": 0.015, "output": 0.075},
    "claude-sonnet-4": {"input": 0.003, "output": 0.015},
    # Anthropic — Claude 3.7 / 3.5 family
    "claude-3-7-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3-5-haiku": {"input": 0.0008, "output": 0.004},
    "claude-3-opus": {"input": 0.015, "output": 0.075},
    # Azure (same as OpenAI)
    "azure/gpt-4o": {"input": 0.0025, "output": 0.01},
    "azure/gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "azure/gpt-4.1": {"input": 0.002, "output": 0.008},
    # AWS Bedrock
    "bedrock/claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
    "bedrock/claude-3-7-sonnet": {"input": 0.003, "output": 0.015},
    "bedrock/llama3-70b": {"input": 0.00099, "output": 0.00099},
    "bedrock/llama3-8b": {"input": 0.0003, "output": 0.0006},
    # Fallback for unknown models
    "__default__": {"input": 0.003, "output": 0.015},
}


def get_price(model: str) -> dict[str, float]:
    """
    Get pricing for a model with fuzzy matching.

    Handles version variants like 'gpt-4o-2024-11-20' -> 'gpt-4o'

    Args:
        model: Model identifier (e.g., 'gpt-4o', 'gpt-4o-2024-11-20', 'claude-3-5-sonnet')

    Returns:
        Dict with 'input' and 'output' keys (prices per 1k tokens)
    """
    # Exact match
    if model in PRICING:
        return PRICING[model]

    # Fuzzy match: strip date/version suffixes for OpenAI models
    # e.g., 'gpt-4o-2024-11-20' -> 'gpt-4o'
    for key in PRICING.keys():
        if key != "__default__" and model.startswith(key):
            return PRICING[key]

    # Fallback to default if nothing matches
    return PRICING["__default__"]


def estimate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int = 0,
) -> float:
    """
    Estimate cost in USD for a given token count.

    Args:
        model: Model identifier
        prompt_tokens: Number of input tokens
        completion_tokens: Number of output tokens (optional, for post-call accounting)

    Returns:
        Cost in USD
    """
    pricing = get_price(model)
    input_cost = (prompt_tokens / 1000) * pricing["input"]
    output_cost = (completion_tokens / 1000) * pricing["output"]
    return input_cost + output_cost
