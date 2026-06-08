import pytest
from options_pricer.black_scholes import black_scholes_price
from options_pricer.monte_carlo import monte_carlo_price


def test_monte_carlo_call_close_to_black_scholes():
    bs_price = black_scholes_price(100, 100, 1, 0.05, 0.2, "call")
    mc_price = monte_carlo_price(100, 100, 1, 0.05, 0.2, "call", simulations=200_000, seed=42)

    assert mc_price == pytest.approx(bs_price, rel=0.03)


def test_monte_carlo_put_close_to_black_scholes():
    bs_price = black_scholes_price(100, 100, 1, 0.05, 0.2, "put")
    mc_price = monte_carlo_price(100, 100, 1, 0.05, 0.2, "put", simulations=200_000, seed=42)

    assert mc_price == pytest.approx(bs_price, rel=0.05)