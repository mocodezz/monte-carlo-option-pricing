"""Closed-form option prices and Greeks.

These serve two purposes in this project:

1. As a *benchmark* for the Monte Carlo engine on European options, where an
   exact answer exists and any discrepancy is therefore measurable error.
2. As *control variates* for the Monte Carlo engine on payoffs where no closed
   form exists.  In particular ``geometric_asian_call`` is exact, and the
   arithmetic Asian option -- which has no closed form -- is strongly
   correlated with it.  See ``mcopt.engine.price_asian``.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

__all__ = [
    "black_scholes_greeks",
    "black_scholes_price",
    "geometric_asian_call",
]


def _d1_d2(S0: float, K: float, r: float, q: float, sigma: float, T: float):
    """Standard Black-Scholes d1/d2 terms."""
    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def _degenerate_price(S0, K, r, q, T, kind):
    """Price when there is no randomness left: sigma == 0 or T == 0.

    The original implementation divided by ``sigma * sqrt(T)`` unconditionally
    and raised on both of these perfectly reasonable inputs.
    """
    forward = S0 * np.exp((r - q) * T)
    intrinsic = max(forward - K, 0.0) if kind == "call" else max(K - forward, 0.0)
    return float(np.exp(-r * T) * intrinsic)


def black_scholes_price(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    q: float = 0.0,
    kind: str = "call",
) -> float:
    """Exact price of a European option under Black-Scholes.

    Parameters
    ----------
    S0 : spot price
    K : strike
    r : continuously compounded risk-free rate
    sigma : volatility (annualised, e.g. 0.2 for 20%)
    T : time to maturity in years
    q : continuous dividend yield
    kind : ``"call"`` or ``"put"``
    """
    kind = _check_kind(kind)
    if T <= 0 or sigma <= 0:
        return _degenerate_price(S0, K, r, q, T, kind)

    d1, d2 = _d1_d2(S0, K, r, q, sigma, T)
    disc, div = np.exp(-r * T), np.exp(-q * T)
    if kind == "call":
        return float(S0 * div * norm.cdf(d1) - K * disc * norm.cdf(d2))
    return float(K * disc * norm.cdf(-d2) - S0 * div * norm.cdf(-d1))


def black_scholes_greeks(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    q: float = 0.0,
    kind: str = "call",
) -> dict[str, float]:
    """Analytic Greeks, used to validate the Monte Carlo finite-difference ones.

    ``vega`` is per 1.0 of volatility (divide by 100 for "per vol point"),
    ``theta`` is per year (divide by 365 for per-day), ``rho`` is per 1.0 of
    rate.
    """
    kind = _check_kind(kind)
    if T <= 0 or sigma <= 0:
        raise ValueError("Greeks are undefined at T=0 or sigma=0")

    d1, d2 = _d1_d2(S0, K, r, q, sigma, T)
    disc, div = np.exp(-r * T), np.exp(-q * T)
    pdf_d1 = norm.pdf(d1)

    gamma = div * pdf_d1 / (S0 * sigma * np.sqrt(T))
    vega = S0 * div * pdf_d1 * np.sqrt(T)
    theta_common = -S0 * div * pdf_d1 * sigma / (2 * np.sqrt(T))

    if kind == "call":
        delta = div * norm.cdf(d1)
        theta = theta_common + q * S0 * div * norm.cdf(d1) - r * K * disc * norm.cdf(d2)
        rho = K * T * disc * norm.cdf(d2)
    else:
        delta = -div * norm.cdf(-d1)
        theta = theta_common - q * S0 * div * norm.cdf(-d1) + r * K * disc * norm.cdf(-d2)
        rho = -K * T * disc * norm.cdf(-d2)

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(theta),
        "rho": float(rho),
    }


def geometric_asian_call(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_fixings: int,
    q: float = 0.0,
) -> float:
    """Exact price of a discretely-monitored *geometric* average-price call.

    The geometric average of lognormals is itself lognormal, so this has a
    closed form even though the arithmetic average does not.  That is precisely
    what makes it a near-perfect control variate for the arithmetic Asian.

    Fixings are assumed equally spaced at t_i = i*T/n for i = 1..n.
    """
    if n_fixings < 1:
        raise ValueError("n_fixings must be >= 1")
    if T <= 0 or sigma <= 0:
        forward = S0 * np.exp((r - q) * T)
        return float(np.exp(-r * T) * max(forward - K, 0.0))

    t = np.linspace(T / n_fixings, T, n_fixings)

    # log G = log S0 + (r - q - sigma^2/2) * mean(t) + (sigma/n) * sum(W_{t_i})
    mu = np.log(S0) + (r - q - 0.5 * sigma**2) * t.mean()
    # Cov(W_ti, W_tj) = min(ti, tj); computed directly to avoid a fiddly
    # closed-form sum that is easy to get subtly wrong.
    var = (sigma**2) * np.minimum.outer(t, t).sum() / n_fixings**2
    sd = np.sqrt(var)

    d1 = (mu - np.log(K) + var) / sd
    d2 = d1 - sd
    expected_G = np.exp(mu + 0.5 * var)
    return float(np.exp(-r * T) * (expected_G * norm.cdf(d1) - K * norm.cdf(d2)))


def _check_kind(kind: str) -> str:
    kind = kind.lower()
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    return kind
