import numpy as np


def monte_carlo_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    simulations: int = 100_000,
    seed: int | None = 42,
) -> float:
    """
    Price a European option using Monte Carlo simulation under geometric Brownian motion.
    """
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        raise ValueError("S, K, T, and sigma must be positive.")
    if simulations <= 0:
        raise ValueError("simulations must be positive.")

    rng = np.random.default_rng(seed)

    z = rng.standard_normal(simulations)
    terminal_prices = S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * z)

    option_type = option_type.lower()

    if option_type == "call":
        payoffs = np.maximum(terminal_prices - K, 0)
    elif option_type == "put":
        payoffs = np.maximum(K - terminal_prices, 0)
    else:
        raise ValueError("option_type must be either 'call' or 'put'")

    return float(np.exp(-r * T) * np.mean(payoffs))