import pytest
from options_pricer.greeks import delta, gamma, vega, theta, rho


def test_delta_call():
    value = delta(100, 100, 1, 0.05, 0.2, "call")
    assert value == pytest.approx(0.6368, rel=1e-3)


def test_delta_put():
    value = delta(100, 100, 1, 0.05, 0.2, "put")
    assert value == pytest.approx(-0.3632, rel=1e-3)


def test_gamma():
    value = gamma(100, 100, 1, 0.05, 0.2)
    assert value == pytest.approx(0.0188, abs=5e-5)


def test_vega():
    value = vega(100, 100, 1, 0.05, 0.2)
    assert value == pytest.approx(0.3752, rel=1e-3)


def test_theta_call():
    value = theta(100, 100, 1, 0.05, 0.2, "call")
    assert value < 0


def test_rho_call():
    value = rho(100, 100, 1, 0.05, 0.2, "call")
    assert value > 0
