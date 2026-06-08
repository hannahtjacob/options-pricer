import pytest
from options_pricer.black_scholes import black_scholes_price
from options_pricer.implied_volatility import implied_volatility


def test_implied_volatility_recovers_original_sigma_call():
    true_sigma = 0.2
    market_price = black_scholes_price(100, 100, 1, 0.05, true_sigma, "call")

    estimated_sigma = implied_volatility(market_price, 100, 100, 1, 0.05, "call")

    assert estimated_sigma == pytest.approx(true_sigma, rel=1e-4)


def test_implied_volatility_recovers_original_sigma_put():
    true_sigma = 0.3
    market_price = black_scholes_price(100, 90, 0.5, 0.03, true_sigma, "put")

    estimated_sigma = implied_volatility(market_price, 100, 90, 0.5, 0.03, "put")

    assert estimated_sigma == pytest.approx(true_sigma, rel=1e-4)