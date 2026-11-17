"""
OpenAI model pricing configuration.

Prices are USD per 1,000,000 tokens. Update this file when OpenAI
changes published rates. Cost is always:

    input_cost  = actual_input_tokens  * (input_per_million / 1_000_000)
    output_cost = actual_output_tokens * (output_per_million / 1_000_000)
    total_cost  = input_cost + output_cost

Never estimate tokens. Only apply these rates to actual usage
returned by the API.
"""

# USD per 1,000,000 tokens
MODEL_PRICING = {
    "gpt-4o-mini": {
        "input_per_million": 0.150,
        "output_per_million": 0.600,
    },
    "gpt-4o": {
        "input_per_million": 2.50,
        "output_per_million": 10.00,
    },
    "gpt-4.1-mini": {
        "input_per_million": 0.40,
        "output_per_million": 1.60,
    },
    "gpt-4.1": {
        "input_per_million": 2.00,
        "output_per_million": 8.00,
    },
    "text-embedding-3-small": {
        "input_per_million": 0.020,
        "output_per_million": 0.0,
    },
    "text-embedding-3-large": {
        "input_per_million": 0.130,
        "output_per_million": 0.0,
    },
}

DEFAULT_PRICING = {
    "input_per_million": 0.150,
    "output_per_million": 0.600,
}


def normalize_model_name(model: str | None) -> str:
    if not model:
        return "unknown"

    name = str(model).strip()

    # OpenAI sometimes returns snapshot ids like gpt-4o-mini-2024-07-18
    for known in MODEL_PRICING:
        if name == known or name.startswith(known + "-"):
            return known

    return name


def get_model_pricing(model: str | None) -> dict:
    known = normalize_model_name(model)

    if known in MODEL_PRICING:
        return MODEL_PRICING[known]

    return dict(DEFAULT_PRICING)


def calculate_cost(
    model: str | None,
    input_tokens: int | None,
    output_tokens: int | None,
) -> dict:
    """
    Calculate USD cost from actual token counts.

    If token counts are missing, costs are None (never invented).
    """

    if input_tokens is None and output_tokens is None:
        return {
            "input_cost": None,
            "output_cost": None,
            "total_cost": None,
        }

    pricing = get_model_pricing(model)
    input_count = 0 if input_tokens is None else int(input_tokens)
    output_count = 0 if output_tokens is None else int(output_tokens)

    input_cost = input_count * (
        pricing["input_per_million"] / 1_000_000
    )
    output_cost = output_count * (
        pricing["output_per_million"] / 1_000_000
    )

    if input_tokens is None:
        input_cost = None

    if output_tokens is None:
        output_cost = None

    if input_cost is None or output_cost is None:
        total_cost = None
    else:
        total_cost = input_cost + output_cost

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
    }
