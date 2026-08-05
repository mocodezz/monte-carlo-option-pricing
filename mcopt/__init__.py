"""Monte Carlo option pricing with honest error bars.

Quick start
-----------
>>> from mcopt import price_european, black_scholes_price
>>> est = price_european(S0=100, K=100, r=0.05, sigma=0.2, T=1.0,
...                      n_paths=100_000, seed=42)
>>> print(est)                                    # doctest: +SKIP
10.4409 +/- 0.0088 (95% CI [10.4237, 10.4581], antithetic + control variate)
>>> black_scholes_price(100, 100, 0.05, 0.2, 1.0) # doctest: +SKIP
10.450583572185565
"""

from .analytic import black_scholes_greeks, black_scholes_price, geometric_asian_call
from .convergence import convergence_study, fitted_slope
from .engine import (
    PriceEstimate,
    mc_greeks_fd,
    price_asian,
    price_barrier_up_and_out_call,
    price_european,
    sample_terminal,
    simulate_paths,
)

__version__ = "1.0.0"

__all__ = [
    "PriceEstimate",
    "black_scholes_greeks",
    "black_scholes_price",
    "convergence_study",
    "fitted_slope",
    "geometric_asian_call",
    "mc_greeks_fd",
    "price_asian",
    "price_barrier_up_and_out_call",
    "price_european",
    "sample_terminal",
    "simulate_paths",
]
