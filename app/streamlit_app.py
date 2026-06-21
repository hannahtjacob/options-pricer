import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

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


st.set_page_config(
    page_title="OptionScope",
    page_icon="Δ",
    layout="wide",
    initial_sidebar_state="expanded",
)


BRAND_PURPLE = "#8b7cf6"
PANEL_BG = "#242421"
CARD_BG = "#1d1d1a"
TEXT_MUTED = "#aaa79f"
TEXT_MAIN = "#f4f2ec"
CORAL = "#ff7043"


def inject_styles() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@500;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

        :root {{
            --brand-purple: {BRAND_PURPLE};
            --panel-bg: {PANEL_BG};
            --card-bg: {CARD_BG};
            --text-main: {TEXT_MAIN};
            --text-muted: {TEXT_MUTED};
            --border: #44433f;
        }}

        .stApp {{
            background: var(--panel-bg);
            color: var(--text-main);
            font-family: "Inter", sans-serif;
        }}

        [data-testid="stSidebar"] {{
            background: #20201d;
            border-right: 1px solid var(--border);
        }}

        [data-testid="stSidebar"] * {{
            color: var(--text-main);
        }}

        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] label {{
            color: #d5d2ca;
        }}

        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] [data-baseweb="select"] > div {{
            background: #292a27;
            color: var(--text-main);
            border-color: #565550;
            font-family: "JetBrains Mono", monospace;
            font-size: 1.05rem;
        }}

        [data-testid="stSidebar"] [data-testid="stNumberInput"] button {{
            background: #2d2e2a;
            border-color: #565550;
        }}

        [data-testid="stSidebar"] [data-testid="stNumberInput"] button svg,
        [data-testid="stSidebar"] [data-baseweb="select"] svg {{
            fill: var(--text-main);
        }}

        .block-container {{
            padding-top: 1.2rem;
            padding-bottom: 5rem;
            max-width: 1280px;
        }}

        div[data-testid="stTabs"] button {{
            color: #c6c2ba;
            font-size: 1.05rem;
            font-weight: 800;
            padding: 0.8rem 1.6rem;
        }}

        div[data-testid="stTabs"] button[aria-selected="true"] {{
            color: var(--brand-purple);
        }}

        div[data-testid="stTabs"] div[data-baseweb="tab-highlight"] {{
            background-color: var(--brand-purple);
        }}

        .sidebar-brand {{
            display: flex;
            align-items: center;
            gap: 0.85rem;
            padding: 0.25rem 0 1.35rem;
            border-bottom: 1px solid var(--border);
            margin-bottom: 1.4rem;
        }}

        .delta-mark {{
            width: 3.2rem;
            height: 3.2rem;
            border-radius: 0.65rem;
            display: grid;
            place-items: center;
            background: var(--brand-purple);
            color: white;
            font-weight: 800;
            font-size: 1.55rem;
        }}

        .brand-title {{
            color: var(--text-main);
            font-size: 1.35rem;
            font-weight: 800;
            line-height: 1;
        }}

        .brand-subtitle {{
            color: #c7c3bb;
            font-size: 0.95rem;
            font-weight: 700;
            margin-top: 0.25rem;
        }}

        .section-kicker {{
            color: var(--text-muted);
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.1rem;
            text-transform: uppercase;
            margin: 1.3rem 0 0.7rem;
        }}

        .metric-card,
        .greek-card,
        .chart-panel,
        .iv-panel {{
            background: var(--card-bg);
            border: 1px solid transparent;
            border-radius: 0.8rem;
            padding: 1.25rem 1.35rem;
        }}

        .metric-card,
        .greek-card {{
            margin-bottom: 1rem;
        }}

        .chart-panel,
        .iv-panel {{
            margin-top: 1rem;
        }}

        .metric-label,
        .greek-label {{
            color: var(--text-muted);
            font-weight: 800;
            letter-spacing: 0.08rem;
            text-transform: uppercase;
        }}

        .metric-value {{
            color: var(--text-main);
            font-family: "JetBrains Mono", monospace;
            font-size: 2rem;
            font-weight: 700;
            margin-top: 0.35rem;
            white-space: nowrap;
        }}

        .metric-value.accent {{
            color: var(--brand-purple);
        }}

        .metric-subtitle {{
            color: var(--text-muted);
            font-weight: 700;
            margin-top: 0.15rem;
        }}

        .delta-pill {{
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            margin-top: 0.45rem;
            padding: 0.18rem 0.55rem;
            border-radius: 999px;
            background: #e4f6d8;
            color: #356b16;
            font-family: "JetBrains Mono", monospace;
            font-weight: 800;
        }}

        .greek-card {{
            min-height: 8rem;
        }}

        .greek-value {{
            color: var(--text-main);
            font-family: "JetBrains Mono", monospace;
            font-size: 1.75rem;
            font-weight: 700;
            margin: 0.55rem 0 0.75rem;
        }}

        .greek-value.negative {{
            color: {CORAL};
        }}

        .bar-track {{
            width: 100%;
            height: 0.32rem;
            background: #33332f;
            border-radius: 999px;
            overflow: hidden;
        }}

        .bar-fill {{
            height: 100%;
            min-width: 0.35rem;
            border-radius: 999px;
            background: #b9a7ff;
        }}

        .bar-fill.negative {{
            background: #ffa38c;
        }}

        .panel-title {{
            color: #d9d6ce;
            font-weight: 800;
            font-size: 1.15rem;
            margin-bottom: 0.8rem;
        }}

        .inline-note {{
            color: var(--text-muted);
            font-weight: 700;
            margin-top: -0.3rem;
            margin-bottom: 0.9rem;
        }}

        .terminal-footer {{
            position: fixed;
            left: 0;
            right: 0;
            bottom: 0;
            z-index: 999;
            min-height: 3.2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
            padding: 0.65rem 2rem;
            background: #20201d;
            border-top: 1px solid var(--border);
            color: var(--text-muted);
            font-weight: 800;
        }}

        .status-left {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .status-dot {{
            width: 0.65rem;
            height: 0.65rem;
            border-radius: 50%;
            background: #70b72a;
        }}

        .status-right {{
            font-family: "JetBrains Mono", monospace;
            text-align: right;
        }}

        .mono {{
            font-family: "JetBrains Mono", monospace;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(
    label: str,
    value: str,
    subtitle: str,
    accent: bool = False,
    delta_value: float | None = None,
) -> None:
    accent_class = " accent" if accent else ""
    delta_html = ""
    if delta_value is not None:
        delta_html = f'<div class="delta-pill">□ {delta_value:.2%}</div>'

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value{accent_class}">{value}</div>
            <div class="metric-subtitle">{subtitle}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_greek_card(name: str, value: float, max_magnitude: float) -> None:
    width = 0 if max_magnitude == 0 else min(abs(value) / max_magnitude * 100, 100)
    negative_class = " negative" if value < 0 else ""
    st.markdown(
        f"""
        <div class="greek-card">
            <div class="greek-label">{name}</div>
            <div class="greek-value{negative_class}">{value:.4f}</div>
            <div class="bar-track">
                <div class="bar-fill{negative_class}" style="width: {width:.1f}%"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def payoff_svg(payoff_df: pd.DataFrame, strike: float, premium: float, option_type: str) -> str:
    width = 760
    height = 230
    left = 58
    right = 18
    top = 18
    bottom = 38
    plot_width = width - left - right
    plot_height = height - top - bottom

    x_values = payoff_df["stock_price"].to_numpy()
    y_values = payoff_df["profit"].to_numpy()
    x_min = float(np.min(x_values))
    x_max = float(np.max(x_values))
    y_min = float(min(np.min(y_values), 0))
    y_max = float(max(np.max(y_values), 0))
    y_padding = max((y_max - y_min) * 0.15, premium * 0.25, 1.0)
    y_min -= y_padding
    y_max += y_padding

    def x_coord(value: float) -> float:
        return left + ((value - x_min) / (x_max - x_min)) * plot_width

    def y_coord(value: float) -> float:
        return top + ((y_max - value) / (y_max - y_min)) * plot_height

    curve_points = " ".join(
        f"{x_coord(float(x)):.2f},{y_coord(float(y)):.2f}"
        for x, y in zip(x_values, y_values)
    )
    zero_y = y_coord(0)
    area_points = (
        f"{x_coord(float(x_values[0])):.2f},{zero_y:.2f} "
        + curve_points
        + f" {x_coord(float(x_values[-1])):.2f},{zero_y:.2f}"
    )
    strike_x = x_coord(strike)
    x_ticks = np.linspace(x_min, x_max, 5)
    y_ticks = np.linspace(y_min + y_padding, y_max - y_padding, 3)

    x_tick_svg = "\n".join(
        f"""
        <line x1="{x_coord(float(tick)):.2f}" y1="{height - bottom}" x2="{x_coord(float(tick)):.2f}" y2="{height - bottom + 4}" stroke="#5a5953"/>
        <text x="{x_coord(float(tick)):.2f}" y="{height - 12}" text-anchor="middle" fill="#aaa79f" font-size="11">{tick:.0f}</text>
        """
        for tick in x_ticks
    )
    y_tick_svg = "\n".join(
        f"""
        <line x1="{left - 4}" y1="{y_coord(float(tick)):.2f}" x2="{left}" y2="{y_coord(float(tick)):.2f}" stroke="#5a5953"/>
        <text x="{left - 10}" y="{y_coord(float(tick)) + 4:.2f}" text-anchor="end" fill="#aaa79f" font-size="11">{tick:.0f}</text>
        """
        for tick in y_ticks
    )

    return f"""
    <svg viewBox="0 0 {width} {height}" role="img" aria-label="Payoff at expiration chart" style="width:100%; height:auto;">
        <rect x="0" y="0" width="{width}" height="{height}" rx="10" fill="#22231f" stroke="#44433f"/>
        <text x="{left + 14}" y="{top + 22}" fill="#aaa79f" font-size="12">Long {option_type} · BS premium ${premium:.2f}</text>
        <line x1="{left}" y1="{zero_y:.2f}" x2="{width - right}" y2="{zero_y:.2f}" stroke="#3a3a36" stroke-dasharray="4 5"/>
        <line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#393934"/>
        <line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#393934"/>
        {x_tick_svg}
        {y_tick_svg}
        <polygon points="{area_points}" fill="#8b7cf6" opacity="0.18"/>
        <polyline points="{curve_points}" fill="none" stroke="#8b7cf6" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/>
        <line x1="{strike_x:.2f}" y1="{top + 16}" x2="{strike_x:.2f}" y2="{height - bottom - 8}" stroke="#ff9b73" stroke-width="1.2" stroke-dasharray="5 5"/>
        <circle cx="{strike_x:.2f}" cy="{y_coord(float(payoff_df.iloc[np.abs(x_values - strike).argmin()]['profit'])):.2f}" r="4" fill="#8b7cf6"/>
        <text x="{strike_x + 8:.2f}" y="{y_coord(0) - 8:.2f}" fill="#b9a7ff" font-size="11">strike</text>
        <text x="{width / 2}" y="{height - 4}" text-anchor="middle" fill="#d5d2ca" font-size="12">Stock price at expiration</text>
        <text x="16" y="{height / 2}" transform="rotate(-90 16 {height / 2})" text-anchor="middle" fill="#d5d2ca" font-size="12">Profit</text>
    </svg>
    """


def render_payoff_panel(payoff_df: pd.DataFrame, strike: float, premium: float, option_type: str) -> None:
    svg = payoff_svg(payoff_df, strike, premium, option_type)
    components.html(
        f"""
        <style>
            html,
            body {{
                margin: 0;
                background: transparent;
                color: #f4f2ec;
                font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            }}

            .chart-panel {{
                box-sizing: border-box;
                width: 100%;
                background: #1d1d1a;
                border-radius: 0.8rem;
                padding: 1.25rem 1.35rem;
            }}

            .panel-title {{
                color: #d9d6ce;
                font-weight: 800;
                font-size: 1.15rem;
                margin-bottom: 0.8rem;
            }}
        </style>
        <div class="chart-panel">
            <div class="panel-title">&#9633;&nbsp;&nbsp;Payoff at expiration</div>
            {svg}
        </div>
        """,
        height=340,
        scrolling=False,
    )


inject_styles()

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="delta-mark">Δ</div>
            <div>
                <div class="brand-title">OptionScope</div>
                <div class="brand-subtitle">European Options Pricer</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-kicker">Instrument</div>', unsafe_allow_html=True)

    S = st.number_input("Stock Price (S)", min_value=0.01, value=100.0, step=1.0)
    K = st.number_input("Strike Price (K)", min_value=0.01, value=100.0, step=1.0)
    T = st.number_input("Time to Expiration (Years)", min_value=0.001, value=1.0, step=0.1)
    r = st.number_input("Risk-Free Rate", min_value=0.0, value=0.05, step=0.01)
    sigma = st.number_input("Volatility", min_value=0.001, value=0.20, step=0.01)
    st.markdown('<div class="section-kicker">Option Type</div>', unsafe_allow_html=True)
    option_type = st.selectbox("Option Type", ["call", "put"])
    st.markdown('<div class="section-kicker">Monte Carlo</div>', unsafe_allow_html=True)
    simulations = st.slider("Monte Carlo Simulations", 1_000, 200_000, 50_000, step=1_000)

bs_price = black_scholes_price(S, K, T, r, sigma, option_type)
mc_price = monte_carlo_price(S, K, T, r, sigma, option_type, simulations=simulations)

greeks = {
    "Delta": delta(S, K, T, r, sigma, option_type),
    "Gamma": gamma(S, K, T, r, sigma),
    "Vega": vega(S, K, T, r, sigma),
    "Theta": theta(S, K, T, r, sigma, option_type),
    "Rho": rho(S, K, T, r, sigma, option_type),
}
max_greek_magnitude = max(abs(value) for value in greeks.values())

stock_prices = np.linspace(S * 0.5, S * 1.5, 100)
payoff_df = option_payoff(stock_prices, K, bs_price, option_type, "long")

pricing_tab, greeks_tab, payoff_tab, surface_tab = st.tabs(
    ["Pricing", "Greeks", "Payoff", "Vol surface"]
)

with pricing_tab:
    card_col1, card_col2, card_col3 = st.columns(3)
    difference = abs(bs_price - mc_price)
    difference_ratio = difference / bs_price if bs_price else 0
    with card_col1:
        render_metric_card("Black-Scholes", f"${bs_price:.4f}", "Analytical")
    with card_col2:
        render_metric_card(
            "Monte Carlo",
            f"${mc_price:.4f}",
            f"{simulations // 1000}k paths",
            accent=True,
        )
    with card_col3:
        render_metric_card(
            "Difference",
            f"${difference:.4f}",
            "model spread",
            delta_value=difference_ratio,
        )

    st.markdown('<div class="panel-title">Greeks</div>', unsafe_allow_html=True)
    greek_cols = st.columns(5)
    for greek_col, (name, value) in zip(greek_cols, greeks.items()):
        with greek_col:
            render_greek_card(name, value, max_greek_magnitude)

    render_payoff_panel(payoff_df, K, bs_price, option_type)

    st.markdown(
        '<div class="iv-panel"><div class="panel-title">□&nbsp;&nbsp;Implied volatility</div>',
        unsafe_allow_html=True,
    )
    market_price = st.number_input(
        "Market Option Price",
        min_value=0.01,
        value=float(bs_price),
        step=0.1,
    )

    try:
        iv = implied_volatility(market_price, S, K, T, r, option_type)
        st.markdown(
            f'<div class="inline-note">Estimated IV <span class="mono" style="color:{BRAND_PURPLE}; font-size:1.8rem;">{iv:.2%}</span></div>',
            unsafe_allow_html=True,
        )
    except ValueError as exc:
        st.warning(str(exc))
    st.markdown("</div>", unsafe_allow_html=True)

with greeks_tab:
    st.markdown('<div class="panel-title">Greek sensitivities</div>', unsafe_allow_html=True)
    greek_cols = st.columns(5)
    for greek_col, (name, value) in zip(greek_cols, greeks.items()):
        with greek_col:
            render_greek_card(name, value, max_greek_magnitude)

with payoff_tab:
    render_payoff_panel(payoff_df, K, bs_price, option_type)

with surface_tab:
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

st.markdown(
    f"""
    <div class="terminal-footer">
        <div class="status-left">
            <span class="status-dot"></span>
            <span>Black-Scholes · Monte Carlo ready</span>
        </div>
        <div class="status-right">
            {option_type.upper()} · S={S:.2f} · K={K:.2f} · T={T:.2f}yr · σ={sigma:.2%}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
