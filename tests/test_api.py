import pytest
from pydantic import ValidationError

from api.main import PriceRequest, price_option


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_price_option_returns_prices_and_greeks(option_type):
    request = PriceRequest(
        S=100,
        K=100,
        T=1,
        r=0.05,
        sigma=0.2,
        option_type=option_type,
    )

    response = price_option(request)

    assert response.black_scholes_price > 0
    assert response.monte_carlo_price > 0
    assert response.monte_carlo_price == pytest.approx(
        response.black_scholes_price, rel=0.05
    )
    assert response.greeks.gamma > 0
    assert response.greeks.vega > 0


def test_price_request_rejects_invalid_inputs():
    with pytest.raises(ValidationError):
        PriceRequest(
            S=-100,
            K=100,
            T=1,
            r=0.05,
            sigma=0.2,
            option_type="call",
        )


def test_price_request_rejects_invalid_option_type():
    with pytest.raises(ValidationError):
        PriceRequest(
            S=100,
            K=100,
            T=1,
            r=0.05,
            sigma=0.2,
            option_type="invalid",
        )
