import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.append(str(SRC_DIR))

from options_pricer.black_scholes import black_scholes_price
from options_pricer.greeks import delta, gamma, vega, theta, rho
from options_pricer.implied_volatility import implied_volatility
from options_pricer.market_data import (
    get_option_chain,
    get_option_expirations,
    get_spot_price,
)
from options_pricer.monte_carlo import monte_carlo_price
from options_pricer.payoff import option_payoff
from options_pricer.vol_surface import (
    calculate_volatility_surface,
    plot_volatility_surface,
)


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

fig, ax = plt.subplots(figsize=(6.5, 4))
fig.patch.set_facecolor("#0e1117")
ax.set_facecolor("#0e1117")
ax.plot(
    payoff_df["stock_price"],
    payoff_df["profit"],
    color="#22d3ee",
    linewidth=2.5,
)
ax.axhline(0, color="#facc15", linestyle="--", linewidth=1.3, alpha=0.9)
ax.axvline(K, color="#f472b6", linestyle="--", linewidth=1.3, alpha=0.9)
ax.set_xlabel("Stock Price at Expiration")
ax.set_ylabel("Profit")
ax.set_title(f"Long {option_type.capitalize()} Payoff")
ax.tick_params(colors="#f8fafc")
ax.xaxis.label.set_color("#f8fafc")
ax.yaxis.label.set_color("#f8fafc")
ax.title.set_color("#f8fafc")
for spine in ax.spines.values():
    spine.set_color("#475569")
ax.grid(color="#334155", linestyle="-", linewidth=0.6, alpha=0.45)
st.pyplot(fig, use_container_width=False)

st.subheader("Volatility Surface")

surface_col1, surface_col2 = st.columns(2)
with surface_col1:
    surface_ticker = st.text_input("Ticker", value="AAPL").strip().upper()
    surface_option_type = st.selectbox(
        "Surface Option Type",
        ["call", "put"],
    )
with surface_col2:
    surface_expiration_count = st.slider(
        "Number of Expirations",
        min_value=1,
        max_value=10,
        value=4,
    )
    st.caption(f"Risk-free rate: {r:.2%}")

if st.button("Generate volatility surface"):
    if not surface_ticker:
        st.warning("Enter a ticker symbol.")
    else:
        with st.spinner(f"Loading {surface_ticker} option chains..."):
            try:
                surface_spot_price = get_spot_price(surface_ticker)
                available_expirations = get_option_expirations(surface_ticker)
            except Exception as exc:
                st.error(
                    f"Could not load market data for {surface_ticker}: {exc}"
                )
            else:
                if not available_expirations:
                    st.warning(
                        f"No option expirations were found for {surface_ticker}."
                    )
                else:
                    selected_expirations = available_expirations[
                        :surface_expiration_count
                    ]
                    chains = []
                    skipped_expirations = []

                    for expiration in selected_expirations:
                        try:
                            chain = get_option_chain(
                                surface_ticker,
                                expiration,
                                surface_option_type,
                            )
                        except Exception:
                            skipped_expirations.append(expiration)
                            continue

                        if chain.empty:
                            skipped_expirations.append(expiration)
                            continue

                        chain = chain.copy()
                        chain["expiration"] = expiration
                        chain["option_type"] = surface_option_type
                        chains.append(chain)

                    if skipped_expirations:
                        st.warning(
                            "Skipped expirations with unavailable option data: "
                            + ", ".join(skipped_expirations)
                        )

                    if not chains:
                        st.warning(
                            "No usable option chains were returned. Try another "
                            "ticker or option type."
                        )
                    else:
                        try:
                            combined_chain = pd.concat(chains, ignore_index=True)
                            surface_df = calculate_volatility_surface(
                                combined_chain,
                                S=surface_spot_price,
                                r=r,
                            )
                        except (TypeError, ValueError) as exc:
                            st.error(f"Could not calculate the surface: {exc}")
                        else:
                            if surface_df.empty:
                                st.warning(
                                    "No valid implied volatilities could be "
                                    "calculated from the returned quotes."
                                )
                            else:
                                st.write(
                                    f"Spot price for {surface_ticker}: "
                                    f"${surface_spot_price:.2f}"
                                )
                                st.dataframe(
                                    surface_df.head(25),
                                    use_container_width=True,
                                )

                                try:
                                    surface_figure = plot_volatility_surface(
                                        surface_df,
                                        title=(
                                            f"{surface_ticker} "
                                            f"{surface_option_type.capitalize()} "
                                            "Implied Volatility Surface"
                                        ),
                                    )
                                except (RuntimeError, ValueError) as exc:
                                    st.warning(
                                        f"Not enough usable data to plot the "
                                        f"surface: {exc}"
                                    )
                                else:
                                    st.pyplot(surface_figure)
                                    plt.close(surface_figure)
