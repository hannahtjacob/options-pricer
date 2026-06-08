import math
from scipy.stats import norm
from options_pricer.black_scholes import d1, d2, validate_inputs


def delta(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    validate_inputs(S, K, T, sigma)
    d_1 = d1(S, K, T, r, sigma)

    if option_type.lower() == "call":
        return norm.cdf(d_1)
    if option_type.lower() == "put":
        return norm.cdf(d_1) - 1

    raise ValueError("option_type must be either 'call' or 'put'")


def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    validate_inputs(S, K, T, sigma)
    d_1 = d1(S, K, T, r, sigma)
    return norm.pdf(d_1) / (S * sigma * math.sqrt(T))


def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    validate_inputs(S, K, T, sigma)
    d_1 = d1(S, K, T, r, sigma)
    return S * norm.pdf(d_1) * math.sqrt(T) / 100


def theta(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    validate_inputs(S, K, T, sigma)
    d_1 = d1(S, K, T, r, sigma)
    d_2 = d2(S, K, T, r, sigma)

    first_term = -(S * norm.pdf(d_1) * sigma) / (2 * math.sqrt(T))

    if option_type.lower() == "call":
        second_term = -r * K * math.exp(-r * T) * norm.cdf(d_2)
        return (first_term + second_term) / 365

    if option_type.lower() == "put":
        second_term = r * K * math.exp(-r * T) * norm.cdf(-d_2)
        return (first_term + second_term) / 365

    raise ValueError("option_type must be either 'call' or 'put'")


def rho(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    validate_inputs(S, K, T, sigma)
    d_2 = d2(S, K, T, r, sigma)

    if option_type.lower() == "call":
        return K * T * math.exp(-r * T) * norm.cdf(d_2) / 100

    if option_type.lower() == "put":
        return -K * T * math.exp(-r * T) * norm.cdf(-d_2) / 100

    raise ValueError("option_type must be either 'call' or 'put'")