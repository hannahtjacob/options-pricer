import numpy as np
import pandas as pd


def option_payoff(
    stock_prices: np.ndarray,
    strike: float,
    premium: float,
    option_type: str = "call",
    position: str = "long",
) -> pd.DataFrame:
    option_type = option_type.lower()
    position = position.lower()

    if option_type == "call":
        intrinsic = np.maximum(stock_prices - strike, 0)
    elif option_type == "put":
        intrinsic = np.maximum(strike - stock_prices, 0)
    else:
        raise ValueError("option_type must be either 'call' or 'put'")

    if position == "long":
        profit = intrinsic - premium
    elif position == "short":
        profit = premium - intrinsic
    else:
        raise ValueError("position must be either 'long' or 'short'")

    return pd.DataFrame(
        {
            "stock_price": stock_prices,
            "intrinsic_value": intrinsic,
            "profit": profit,
        }
    )