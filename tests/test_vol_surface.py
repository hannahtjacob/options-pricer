import matplotlib
import pandas as pd
import pytest

from options_pricer.black_scholes import black_scholes_price
from options_pricer.vol_surface import (
    calculate_volatility_surface,
    plot_volatility_surface,
)

matplotlib.use("Agg")


def test_calculate_volatility_surface_recovers_volatility():
    valuation_date = pd.Timestamp("2026-01-01")
    rows = []

    for days, strike, option_type in [
        (90, 90, "call"),
        (90, 100, "call"),
        (90, 110, "call"),
        (180, 90, "put"),
        (180, 100, "put"),
        (180, 110, "put"),
    ]:
        time_to_expiration = days / 365
        rows.append(
            {
                "strike": strike,
                "expiration": valuation_date + pd.Timedelta(days=days),
                "lastPrice": black_scholes_price(
                    100, strike, time_to_expiration, 0.05, 0.25, option_type
                ),
                "option_type": option_type,
            }
        )

    surface = calculate_volatility_surface(
        pd.DataFrame(rows), S=100, r=0.05, valuation_date=valuation_date
    )

    assert list(surface.columns) == [
        "strike",
        "expiration",
        "time_to_expiration",
        "market_price",
        "implied_volatility",
    ]
    assert len(surface) == 6
    assert surface["implied_volatility"].tolist() == pytest.approx(
        [0.25] * 6, rel=1e-4
    )


def test_calculate_volatility_surface_prefers_mid_and_cleans_invalid_rows():
    chain = pd.DataFrame(
        [
            {
                "strike": 100,
                "expiration": "2026-07-01",
                "mid": 10.0,
                "lastPrice": 11.0,
                "option_type": "call",
            },
            {
                "strike": 100,
                "expiration": "2025-12-01",
                "mid": 10.0,
                "lastPrice": 11.0,
                "option_type": "call",
            },
            {
                "strike": 100,
                "expiration": "2026-07-01",
                "mid": -1.0,
                "lastPrice": 10.0,
                "option_type": "call",
            },
        ]
    )

    surface = calculate_volatility_surface(
        chain, S=100, r=0.05, valuation_date="2026-01-01"
    )

    assert len(surface) == 2
    assert surface["market_price"].tolist() == [10.0, 10.0]


def test_plot_volatility_surface_returns_3d_axes():
    surface = pd.DataFrame(
        {
            "strike": [90, 100, 110, 90],
            "time_to_expiration": [0.25, 0.25, 0.25, 0.5],
            "implied_volatility": [0.22, 0.2, 0.21, 0.24],
        }
    )

    figure, axes = plot_volatility_surface(surface)

    assert axes.name == "3d"
    assert axes.get_xlabel() == "Strike"
    figure.clear()
