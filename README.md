# Options Pricer

A Python options-pricing engine for European options, implementing Black-Scholes pricing, Greeks, Monte Carlo simulation, implied volatility estimation, binomial tree pricing, volatility surface visualization, market option-chain analysis, and a FastAPI pricing endpoint.

## Features

* Price European call and put options using the Black-Scholes model
* Compute option Greeks: Delta, Gamma, Vega, Theta, and Rho
* Price options using Monte Carlo simulation under geometric Brownian motion
* Estimate implied volatility using numerical root finding
* Generate payoff diagrams for long and short calls and puts
* Implement Cox-Ross-Rubinstein binomial tree pricing
* Compare binomial tree prices against Black-Scholes prices to study convergence
* Compute finite-difference Greeks and compare them with analytical Greeks
* Retrieve optional market data and option-chain data using `yfinance`
* Compare theoretical prices, market prices, and implied volatility
* Build implied volatility surfaces across strikes and expirations
* Visualize pricing results and risk metrics in a Streamlit dashboard
* Expose pricing functionality through a FastAPI endpoint
* Validate code with a pytest test suite
* Run automated tests with GitHub Actions CI

## Tech Stack

* Python
* NumPy
* SciPy
* pandas
* matplotlib
* Streamlit
* FastAPI
* Pydantic
* pytest
* yfinance

## Project Structure

```text
options-pricer/
│
├── README.md
├── requirements.txt
├── pyproject.toml
│
├── src/
│   └── options_pricer/
│       ├── __init__.py
│       ├── black_scholes.py
│       ├── greeks.py
│       ├── finite_difference_greeks.py
│       ├── monte_carlo.py
│       ├── binomial_tree.py
│       ├── implied_volatility.py
│       ├── payoff.py
│       ├── market_data.py
│       ├── vol_surface.py
│       └── utils.py
│
├── app/
│   └── streamlit_app.py
│
├── api/
│   └── main.py
│
├── tests/
│   ├── test_black_scholes.py
│   ├── test_greeks.py
│   ├── test_finite_difference_greeks.py
│   ├── test_monte_carlo.py
│   ├── test_binomial_tree.py
│   ├── test_implied_volatility.py
│   └── test_api.py
│
├── notebooks/
│   └── option_pricing_demo.ipynb
│
├── images/
│   ├── dashboard.png
│   ├── payoff.png
│   └── volatility_surface.png
│
└── .github/
    └── workflows/
        └── tests.yml
```

## Methodology

This project implements multiple approaches to pricing European options.

The Black-Scholes model provides a closed-form theoretical price for European call and put options. The project also computes the standard Greeks, which measure sensitivity to underlying price, volatility, time decay, and interest rates.

Monte Carlo simulation estimates option value by simulating terminal stock prices under geometric Brownian motion and discounting the expected payoff back to present value.

The Cox-Ross-Rubinstein binomial tree model prices options by constructing a recombining tree of possible stock prices. As the number of time steps increases, the binomial price can be compared against the Black-Scholes price to demonstrate convergence.

Implied volatility is estimated by solving for the volatility that makes the Black-Scholes theoretical price equal to an observed market price. This is used to analyze how market expectations vary across strikes and expirations.

The volatility surface module computes implied volatility across option chains and visualizes the relationship between strike price, expiration, and implied volatility.

## Dashboard

The Streamlit dashboard allows users to interactively change option assumptions and view:

* Black-Scholes price
* Monte Carlo price
* Binomial tree price
* Greeks
* Finite-difference Greek comparison
* Implied volatility
* Payoff diagram
* Volatility surface visualization

To run the dashboard:

```bash
streamlit run app/streamlit_app.py
```

## API

The project includes a FastAPI endpoint for programmatic option pricing.

Example endpoint:

```text
POST /price
```

Example request body:

```json
{
  "S": 100,
  "K": 100,
  "T": 1,
  "r": 0.05,
  "sigma": 0.2,
  "option_type": "call"
}
```

Example response:

```json
{
  "black_scholes_price": 10.4506,
  "monte_carlo_price": 10.42,
  "binomial_tree_price": 10.44,
  "greeks": {
    "delta": 0.6368,
    "gamma": 0.0188,
    "vega": 0.3752,
    "theta": -0.0175,
    "rho": 0.5323
  }
}
```

To run the API:

```bash
uvicorn api.main:app --reload
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/options-pricer.git
cd options-pricer
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running Tests

Run the full test suite:

```bash
pytest
```

## Disclaimer

This project is for educational and research purposes only. It does not provide investment advice and should not be used for live trading decisions. Market data, when used, is pulled from public data sources through third-party libraries and may be delayed, incomplete, or inaccurate.
