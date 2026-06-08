import math
from scipy.stats import norm


def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    validate_inputs(S, K, T, sigma)
    return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))


def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    return d1(S, K, T, r, sigma) - sigma * math.sqrt(T)


def black_scholes_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> float:
    """
    Price a European option using the Black-Scholes formula.

    S: current stock price
    K: strike price
    T: time to expiration in years
    r: risk-free interest rate as decimal
    sigma: volatility as decimal
    option_type: "call" or "put"
    """
    validate_inputs(S, K, T, sigma)

    option_type = option_type.lower()
    d_1 = d1(S, K, T, r, sigma)
    d_2 = d2(S, K, T, r, sigma)

    if option_type == "call":
        return S * norm.cdf(d_1) - K * math.exp(-r * T) * norm.cdf(d_2)

    if option_type == "put":
        return K * math.exp(-r * T) * norm.cdf(-d_2) - S * norm.cdf(-d_1)

    raise ValueError("option_type must be either 'call' or 'put'")


def validate_inputs(S: float, K: float, T: float, sigma: float) -> None:
    if S <= 0:
        raise ValueError("Stock price S must be positive.")
    if K <= 0:
        raise ValueError("Strike price K must be positive.")
    if T <= 0:
        raise ValueError("Time to expiration T must be positive.")
    if sigma <= 0:
        raise ValueError("Volatility sigma must be positive.")