from datetime import date, datetime

import pandas as pd

from options_pricer.implied_volatility import implied_volatility


SURFACE_COLUMNS = [
    "strike",
    "expiration",
    "time_to_expiration",
    "market_price",
    "implied_volatility",
]


def calculate_volatility_surface(
    options_chain: pd.DataFrame,
    S: float,
    r: float,
    valuation_date: str | date | datetime | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Calculate implied volatility for each valid row in an options chain."""
    required_columns = {"strike", "expiration", "option_type"}
    missing_columns = required_columns - set(options_chain.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"options_chain is missing required columns: {missing}")
    if S <= 0:
        raise ValueError("S must be positive")

    market_prices = _get_market_prices(options_chain)
    expirations = pd.to_datetime(
        options_chain["expiration"], errors="coerce", utc=True
    ).dt.tz_convert(None)
    valuation = _normalize_valuation_date(valuation_date)

    working = pd.DataFrame(
        {
            "strike": pd.to_numeric(options_chain["strike"], errors="coerce"),
            "expiration": expirations,
            "market_price": market_prices,
            "option_type": options_chain["option_type"].astype("string").str.lower(),
        }
    )
    working["time_to_expiration"] = (
        working["expiration"] - valuation
    ).dt.total_seconds() / (365 * 24 * 60 * 60)

    valid = (
        working["strike"].gt(0)
        & working["market_price"].gt(0)
        & working["time_to_expiration"].gt(0)
        & working["option_type"].isin(["call", "put"])
    )
    working = working.loc[valid].copy()

    implied_volatilities = []
    for row in working.itertuples(index=False):
        try:
            volatility = implied_volatility(
                market_price=row.market_price,
                S=S,
                K=row.strike,
                T=row.time_to_expiration,
                r=r,
                option_type=row.option_type,
            )
        except ValueError:
            volatility = float("nan")
        implied_volatilities.append(volatility)

    working["implied_volatility"] = implied_volatilities
    working = working.dropna(subset=["implied_volatility"])

    working["expiration"] = working["expiration"].dt.normalize()
    return working[SURFACE_COLUMNS].sort_values(
        ["expiration", "strike"], ignore_index=True
    )


def plot_volatility_surface(
    surface: pd.DataFrame,
    *,
    title: str = "Implied Volatility Surface",
) -> "Figure":
    """Create a 3D matplotlib figure from calculated volatility surface data."""
    missing_columns = {
        "strike",
        "time_to_expiration",
        "implied_volatility",
    } - set(surface.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"surface is missing required columns: {missing}")

    plot_data = surface[
        ["strike", "time_to_expiration", "implied_volatility"]
    ].dropna()
    if len(plot_data.drop_duplicates(["strike", "time_to_expiration"])) < 3:
        raise ValueError("surface must contain at least three distinct data points")
    if plot_data["strike"].nunique() < 2 or plot_data["time_to_expiration"].nunique() < 2:
        raise ValueError(
            "surface must contain multiple strikes and expiration dates"
        )

    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    figure = plt.figure()
    axes = figure.add_subplot(111, projection="3d")
    axes.plot_trisurf(
        plot_data["strike"],
        plot_data["time_to_expiration"],
        plot_data["implied_volatility"],
        cmap="viridis",
        edgecolor="none",
    )
    axes.set_xlabel("Strike")
    axes.set_ylabel("Time to Expiration (Years)")
    axes.set_zlabel("Implied Volatility")
    axes.set_title(title)

    return figure


def _get_market_prices(options_chain: pd.DataFrame) -> pd.Series:
    prices = pd.Series(float("nan"), index=options_chain.index, dtype=float)

    for column in ("mid", "midPrice"):
        if column in options_chain.columns:
            candidate = pd.to_numeric(options_chain[column], errors="coerce")
            prices = prices.fillna(candidate.where(candidate.gt(0)))

    if {"bid", "ask"}.issubset(options_chain.columns):
        bid = pd.to_numeric(options_chain["bid"], errors="coerce")
        ask = pd.to_numeric(options_chain["ask"], errors="coerce")
        midpoint = (bid + ask) / 2
        prices = prices.fillna(midpoint.where(bid.ge(0) & ask.gt(0) & ask.ge(bid)))

    if "lastPrice" in options_chain.columns:
        last_price = pd.to_numeric(options_chain["lastPrice"], errors="coerce")
        prices = prices.fillna(last_price.where(last_price.gt(0)))

    if prices.isna().all():
        raise ValueError(
            "options_chain must include lastPrice, mid, midPrice, or bid and ask"
        )

    return prices


def _normalize_valuation_date(
    valuation_date: str | date | datetime | pd.Timestamp | None,
) -> pd.Timestamp:
    valuation = pd.Timestamp(valuation_date or date.today())
    if valuation.tzinfo is not None:
        valuation = valuation.tz_convert(None)
    return valuation.normalize()
