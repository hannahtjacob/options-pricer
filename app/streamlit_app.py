import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.append(str(SRC_DIR))

from options_pricer.black_scholes import black_scholes_price
from options_pricer.greeks import delta, gamma, vega, theta, rho
from options_pricer.implied_volatility import implied_volatility
from options_pricer.monte_carlo import monte_carlo_price
from options_pricer.payoff import option_payoff


st.set_page_config(page_title="Options Pricer", layout="wide")

st.title("Options Pricer")
st.write(
    "Black-Scholes pricing, Greeks, Monte Carlo simulation, implied volatility, "
    "and payoff visualization for European options."
)

with st.sidebar:
    st.header("Option Inputs")

    S = st.number_input("Stock Price (S)", min_value=0.01, value=100.0, step=1.0)
    K = st.number_input("Strike Price (K)", min_value=0.01, value=100.0, step=1.0)
    T = st.number_input("Time to Expiration (Years)", min_value=0.001, value=1.0, step=0.1)
    r = st.number_input("Risk-Free Rate", min_value=0.0, value=0.05, step=0.01)
    sigma = st.number_input("Volatility", min_value=0.001, value=0.20, step=0.01)
    option_type = st.selectbox("Option Type", ["call", "put"])
    simulations = st.slider("Monte Carlo Simulations", 1_000, 200_000, 50_000, step=1_000)

bs_price = black_scholes_price(S, K, T, r, sigma, option_type)
mc_price = monte_carlo_price(S, K, T, r, sigma, option_type, simulations=simulations)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Black-Scholes Price", f"${bs_price:.4f}")

with col2:
    st.metric("Monte Carlo Price", f"${mc_price:.4f}")

with col3:
    st.metric("Absolute Difference", f"${abs(bs_price - mc_price):.4f}")

st.subheader("Greeks")

greeks = {
    "Delta": delta(S, K, T, r, sigma, option_type),
    "Gamma": gamma(S, K, T, r, sigma),
    "Vega": vega(S, K, T, r, sigma),
    "Theta": theta(S, K, T, r, sigma, option_type),
    "Rho": rho(S, K, T, r, sigma, option_type),
}

st.dataframe(greeks, use_container_width=True)

st.subheader("Implied Volatility")

market_price = st.number_input("Market Option Price", min_value=0.01, value=float(bs_price), step=0.1)

try:
    iv = implied_volatility(market_price, S, K, T, r, option_type)
    st.metric("Estimated Implied Volatility", f"{iv:.2%}")
except ValueError as exc:
    st.warning(str(exc))

st.subheader("Payoff Diagram")

stock_prices = np.linspace(S * 0.5, S * 1.5, 100)
payoff_df = option_payoff(stock_prices, K, bs_price, option_type, "long")

fig, ax = plt.subplots()
ax.plot(payoff_df["stock_price"], payoff_df["profit"])
ax.axhline(0, linestyle="--")
ax.axvline(K, linestyle="--")
ax.set_xlabel("Stock Price at Expiration")
ax.set_ylabel("Profit")
ax.set_title(f"Long {option_type.capitalize()} Payoff")
st.pyplot(fig)