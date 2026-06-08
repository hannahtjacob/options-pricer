import pytest

from options_pricer.binomial_tree import binomial_tree_price
from options_pricer.black_scholes import black_scholes_price


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_binomial_tree_converges_to_black_scholes(option_type):
    parameters = dict(S=100, K=100, T=1, r=0.05, sigma=0.2)
    black_scholes = black_scholes_price(**parameters, option_type=option_type)

    price_with_10_steps = binomial_tree_price(
        **parameters, steps=10, option_type=option_type
    )
    price_with_1000_steps = binomial_tree_price(
        **parameters, steps=1000, option_type=option_type
    )

    assert abs(price_with_1000_steps - black_scholes) < abs(
        price_with_10_steps - black_scholes
    )
    assert price_with_1000_steps == pytest.approx(black_scholes, abs=0.01)


def test_invalid_steps():
    with pytest.raises(ValueError, match="positive integer"):
        binomial_tree_price(100, 100, 1, 0.05, 0.2, steps=0)


def test_invalid_option_type():
    with pytest.raises(ValueError, match="option_type"):
        binomial_tree_price(100, 100, 1, 0.05, 0.2, steps=100, option_type="invalid")
