import math

from options_pricer.black_scholes import validate_inputs


def binomial_tree_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    steps: int,
    option_type: str = "call",
) -> float:
    """Price a European option with a Cox-Ross-Rubinstein binomial tree."""
    validate_inputs(S, K, T, sigma)

    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps must be a positive integer")

    option_type = option_type.lower()
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be either 'call' or 'put'")

    dt = T / steps
    up = math.exp(sigma * math.sqrt(dt))
    down = 1 / up
    growth = math.exp(r * dt)
    probability = (growth - down) / (up - down)

    if not 0 <= probability <= 1:
        raise ValueError(
            "steps are too few for a valid risk-neutral probability; increase steps"
        )

    discount = math.exp(-r * dt)
    payoff_sign = 1 if option_type == "call" else -1

    # Begin at the all-down terminal node and move upward through the tree.
    stock_price = S * down**steps
    stock_ratio = up / down
    option_values = []
    for _ in range(steps + 1):
        option_values.append(max(payoff_sign * (stock_price - K), 0.0))
        stock_price *= stock_ratio

    for nodes_remaining in range(steps, 0, -1):
        for node in range(nodes_remaining):
            option_values[node] = discount * (
                probability * option_values[node + 1]
                + (1 - probability) * option_values[node]
            )

    return option_values[0]
