# Options Pricer

A Python options-pricing engine that implements Black-Scholes pricing, Greeks, Monte Carlo simulation, implied volatility estimation, payoff diagrams, and an interactive Streamlit dashboard. This project is for educational and research purposes only. Market data is pulled through yfinance for educational/research purposes only. The project does not provide investment advice and should not be used for live trading decisions.

## Features

- European call and put pricing using the Black-Scholes model
- Greeks: Delta, Gamma, Vega, Theta, and Rho
- Monte Carlo option pricing under geometric Brownian motion
- Implied volatility solver using numerical root finding
- Payoff diagram generator for long calls and puts
- Streamlit dashboard for interactive pricing and visualization
- Optional market data utilities using yfinance
- Pytest test suite and GitHub Actions CI

## Tech Stack

- Python
- NumPy
- SciPy
- pandas
- matplotlib
- Streamlit
- pytest
- yfinance

## Project Structure

```text
src/options_pricer/
  black_scholes.py
  greeks.py
  monte_carlo.py
  implied_volatility.py
  payoff.py
  market_data.py
app/
  streamlit_app.py
tests/
  test_black_scholes.py
  test_greeks.py
  test_monte_carlo.py
  test_implied_volatility.py

## How to Run

git clone https://github.com/YOUR_USERNAME/options-pricer.git
cd options-pricer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py

## How to Run Tests

pytest