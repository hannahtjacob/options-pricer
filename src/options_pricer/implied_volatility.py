from scipy.optimize import brentq
from options_pricer.black_scholes import black_scholes_price


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    lower_bound: float = 1e-6,
    upper_bound: float = 5.0,
) -> float:
    """
    Estimate implied volatility by solving for sigma where
    Black-Scholes price equals observed market price.
    """
    if market_price <= 0:
        raise ValueError("market_price must be positive.")

    def objective(sigma: float) -> float:
        return black_scholes_price(S, K, T, r, sigma, option_type) - market_price

    try:
        return float(brentq(objective, lower_bound, upper_bound))
    except ValueError as exc:
        raise ValueError(
            "Could not solve for implied volatility. Check whether market_price is reasonable."
        ) from exc