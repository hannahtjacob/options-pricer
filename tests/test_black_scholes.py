import pytest
from options_pricer.black_scholes import black_scholes_price


def test_black_scholes_call_price():
    price = black_scholes_price(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="call")
    assert price == pytest.approx(10.4506, rel=1e-4)


def test_black_scholes_put_price():
    price = black_scholes_price(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="put")
    assert price == pytest.approx(5.5735, rel=1e-4)


def test_invalid_option_type():
    with pytest.raises(ValueError):
        black_scholes_price(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="invalid")


def test_invalid_inputs():
    with pytest.raises(ValueError):
        black_scholes_price(S=-100, K=100, T=1, r=0.05, sigma=0.2)