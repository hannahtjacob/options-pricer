from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from options_pricer.black_scholes import black_scholes_price
from options_pricer.greeks import delta, gamma, rho, theta, vega
from options_pricer.monte_carlo import monte_carlo_price


class PriceRequest(BaseModel):
    S: float = Field(gt=0, description="Current stock price")
    K: float = Field(gt=0, description="Option strike price")
    T: float = Field(gt=0, description="Time to expiration in years")
    r: float = Field(description="Risk-free interest rate as a decimal")
    sigma: float = Field(gt=0, description="Volatility as a decimal")
    option_type: Literal["call", "put"]


class GreeksResponse(BaseModel):
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


class PriceResponse(BaseModel):
    black_scholes_price: float
    monte_carlo_price: float
    greeks: GreeksResponse


app = FastAPI(
    title="Options Pricer API",
    description="Price European call and put options.",
    version="0.1.0",
)


@app.post("/price", response_model=PriceResponse)
def price_option(request: PriceRequest) -> PriceResponse:
    parameters = request.model_dump()

    return PriceResponse(
        black_scholes_price=black_scholes_price(**parameters),
        monte_carlo_price=monte_carlo_price(**parameters),
        greeks=GreeksResponse(
            delta=float(delta(**parameters)),
            gamma=float(
                gamma(request.S, request.K, request.T, request.r, request.sigma)
            ),
            vega=float(
                vega(request.S, request.K, request.T, request.r, request.sigma)
            ),
            theta=float(theta(**parameters)),
            rho=float(rho(**parameters)),
        ),
    )
