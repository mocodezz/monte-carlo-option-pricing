"""Convergence diagnostics.

The defining property of Monte Carlo is that its error shrinks like
1/sqrt(N): to get one more decimal place you need a hundred times the work.
This module produces the evidence for that claim rather than asserting it.
"""

from __future__ import annotations

import numpy as np

from .analytic import black_scholes_price
from .engine import price_european

__all__ = ["convergence_study", "fitted_slope"]


def convergence_study(S0, K, r, sigma, T, q=0.0, kind="call",
                      path_counts=None, antithetic=False,
                      control_variate=False, seed=7):
    """Price a European option at increasing path counts.

    Returns a dict of arrays: ``n_paths``, ``price``, ``std_error``,
    ``abs_error`` (against the exact Black-Scholes value) and the scalar
    ``exact``.

    Defaults to *plain* Monte Carlo, because the point of the plot is to show
    the raw 1/sqrt(N) rate.  Switch the variance reduction on to show that it
    shifts the line down without changing its slope -- a smaller constant, not
    a better rate.
    """
    if path_counts is None:
        path_counts = np.unique(np.logspace(2, 6, 20).astype(int) // 2 * 2)

    exact = black_scholes_price(S0, K, r, sigma, T, q, kind)
    prices, errors = [], []

    for i, n in enumerate(path_counts):
        est = price_european(S0, K, r, sigma, T, int(n), q, kind,
                             antithetic=antithetic,
                             control_variate=control_variate,
                             seed=seed + i)
        prices.append(est.value)
        errors.append(est.std_error)

    prices = np.asarray(prices)
    return {
        "n_paths": np.asarray(path_counts, dtype=int),
        "price": prices,
        "std_error": np.asarray(errors),
        "abs_error": np.abs(prices - exact),
        "exact": exact,
    }


def fitted_slope(n_paths, std_error) -> float:
    """Least-squares slope of log(std_error) against log(n_paths).

    Monte Carlo theory predicts -0.5.  Recovering that number from your own
    output is a much stronger claim than quoting it from a textbook.
    """
    mask = np.asarray(std_error) > 0
    log_n = np.log(np.asarray(n_paths, dtype=float)[mask])
    log_e = np.log(np.asarray(std_error, dtype=float)[mask])
    return float(np.polyfit(log_n, log_e, 1)[0])
