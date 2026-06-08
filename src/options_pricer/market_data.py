import pandas as pd
import yfinance as yf


def get_spot_price(ticker: str) -> float:
    stock = yf.Ticker(ticker)
    history = stock.history(period="1d")

    if history.empty:
        raise ValueError(f"No price data found for ticker: {ticker}")

    return float(history["Close"].iloc[-1])


def get_option_expirations(ticker: str) -> list[str]:
    stock = yf.Ticker(ticker)
    return list(stock.options)


def get_option_chain(ticker: str, expiration: str, option_type: str = "call") -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    chain = stock.option_chain(expiration)

    option_type = option_type.lower()

    if option_type == "call":
        return chain.calls
    if option_type == "put":
        return chain.puts

    raise ValueError("option_type must be either 'call' or 'put'")