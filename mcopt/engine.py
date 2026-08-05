"""Monte Carlo pricing engine.

Design notes, in the order they matter:

* **Every price is returned with a standard error.**  A Monte Carlo estimate
  without an error bar is not a result, it is a rumour.  ``PriceEstimate``
  makes the error impossible to forget.
* **European payoffs never build a path grid.**  Only the terminal value is
  needed, so only the terminal value is drawn -- one normal per path instead of
  ``n_steps`` of them.
* **Antithetic standard errors are computed on pair averages**, not on the raw
  sample.  Antithetic draws are negatively correlated by construction, so
  ``std(all_samples) / sqrt(n)`` would misstate the true error.
* **Everything takes a seed.**  Two runs with the same seed are bit-identical.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .analytic import geometric_asian_call

__all__ = [
    "PriceEstimate",
    "make_rng",
    "mc_greeks_fd",
    "price_asian",
    "price_barrier_up_and_out_call",
    "price_european",
    "sample_terminal",
    "simulate_paths",
]

_Z95 = 1.959963984540054  # two-sided 95% normal quantile


@dataclass(frozen=True)
class PriceEstimate:
    """A Monte Carlo price together with its sampling error."""

    value: float
    std_error: float
    n_effective: int
    method: str = "plain"

    @property
    def ci95(self) -> tuple[float, float]:
        half = _Z95 * self.std_error
        return (self.value - half, self.value + half)

    def sigmas_from(self, reference: float) -> float:
        """How many standard errors away a benchmark price sits.

        Under 2 means the difference is indistinguishable from sampling noise.
        """
        if self.std_error == 0:
            return 0.0 if np.isclose(self.value, reference) else float("inf")
        return abs(self.value - reference) / self.std_error

    def __str__(self) -> str:
        lo, hi = self.ci95
        return f"{self.value:.4f} +/- {self.std_error:.4f} (95% CI [{lo:.4f}, {hi:.4f}], {self.method})"


def make_rng(seed: int | None = None, rng: np.random.Generator | None = None):
    """Return a NumPy Generator.  Prefer this over the legacy global np.random."""
    if rng is not None:
        return rng
    return np.random.default_rng(seed)


def _normals(rng, n_paths: int, n_steps: int | None, antithetic: bool):
    """Draw standard normals, mirrored in pairs when antithetic sampling is on."""
    shape = (n_paths,) if n_steps is None else (n_paths, n_steps)
    if not antithetic:
        return rng.standard_normal(shape)
    if n_paths % 2:
        raise ValueError("antithetic sampling requires an even n_paths")
    half = (n_paths // 2,) if n_steps is None else (n_paths // 2, n_steps)
    z = rng.standard_normal(half)
    return np.concatenate([z, -z], axis=0)


def _summarise(payoffs, antithetic, controls=None, control_mean=None, method="plain"):
    """Collapse a sample of discounted payoffs into a PriceEstimate.

    Antithetic pairs are averaged *first* so the reported error accounts for
    their negative correlation.  The control-variate coefficient is estimated
    from the same sample, which introduces an O(1/n) bias -- negligible at the
    path counts used here, and the standard practice.
    """
    payoffs = np.asarray(payoffs, dtype=float)
    if antithetic:
        m = payoffs.shape[0] // 2
        payoffs = 0.5 * (payoffs[:m] + payoffs[m:])
        if controls is not None:
            controls = 0.5 * (controls[:m] + controls[m:])

    if controls is not None:
        controls = np.asarray(controls, dtype=float)
        cov = np.cov(payoffs, controls, ddof=1)
        beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0.0
        payoffs = payoffs - beta * (controls - control_mean)

    n = payoffs.shape[0]
    return PriceEstimate(
        value=float(payoffs.mean()),
        std_error=float(payoffs.std(ddof=1) / np.sqrt(n)),
        n_effective=int(n),
        method=method,
    )


def _label(antithetic: bool, control_variate: bool) -> str:
    parts = [n for n, on in (("antithetic", antithetic), ("control variate", control_variate)) if on]
    return " + ".join(parts) if parts else "plain"


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------

def sample_terminal(S0, r, sigma, T, n_paths, q=0.0, antithetic=False, rng=None, seed=None):
    """Draw terminal prices S_T directly.  One normal per path, no path grid.

    This is exact for geometric Brownian motion -- there is no discretisation
    error to trade off, because the GBM solution is known in closed form.
    """
    rng = make_rng(seed, rng)
    z = _normals(rng, n_paths, None, antithetic)
    drift = (r - q - 0.5 * sigma**2) * T
    return S0 * np.exp(drift + sigma * np.sqrt(T) * z)


def simulate_paths(S0, r, sigma, T, n_steps, n_paths, q=0.0, antithetic=False,
                   rng=None, seed=None, max_gb=2.0):
    """Simulate full GBM paths.  Only needed for path-dependent payoffs.

    Returns ``(t, paths)`` where ``t`` has length ``n_steps + 1`` and ``paths``
    has shape ``(n_paths, n_steps + 1)`` with ``paths[:, 0] == S0``.
    """
    gb = n_paths * (n_steps + 1) * 8 / 1024**3
    if gb > max_gb:
        raise MemoryError(
            f"path grid would need ~{gb:.1f} GB; reduce n_paths/n_steps "
            f"or raise max_gb (currently {max_gb} GB)"
        )

    rng = make_rng(seed, rng)
    dt = T / n_steps
    z = _normals(rng, n_paths, n_steps, antithetic)
    w = np.cumsum(z, axis=1) * np.sqrt(dt)
    t = np.linspace(0.0, T, n_steps + 1)
    log_increment = (r - q - 0.5 * sigma**2) * t[1:] + sigma * w

    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = S0
    paths[:, 1:] = S0 * np.exp(log_increment)
    return t, paths


# --------------------------------------------------------------------------
# Pricers
# --------------------------------------------------------------------------

def price_european(S0, K, r, sigma, T, n_paths=100_000, q=0.0, kind="call",
                   antithetic=True, control_variate=True, rng=None, seed=None):
    """Monte Carlo price of a European call or put.

    The control variate is the terminal price itself, whose expectation
    ``S0 * exp((r - q) T)`` is known exactly.  It is strongly correlated with
    the payoff, so subtracting its sampling error removes most of the noise.
    """
    kind = kind.lower()
    if kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")

    s_t = sample_terminal(S0, r, sigma, T, n_paths, q, antithetic, rng, seed)
    intrinsic = s_t - K if kind == "call" else K - s_t
    payoffs = np.exp(-r * T) * np.maximum(intrinsic, 0.0)

    controls, control_mean = None, None
    if control_variate:
        controls = s_t
        control_mean = S0 * np.exp((r - q) * T)

    return _summarise(payoffs, antithetic, controls, control_mean,
                      _label(antithetic, control_variate))


def price_asian(S0, K, r, sigma, T, n_paths=50_000, n_fixings=52, q=0.0,
                kind="call", antithetic=True, control_variate=True,
                rng=None, seed=None):
    """Arithmetic average-price Asian option.

    This payoff has **no closed form**, which is the entire justification for
    reaching for Monte Carlo in the first place.  The control variate is the
    *geometric* Asian, which does have a closed form and is almost perfectly
    correlated with the arithmetic one -- typically cutting the standard error
    by more than an order of magnitude.
    """
    kind = kind.lower()
    if kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")

    _, paths = simulate_paths(S0, r, sigma, T, n_fixings, n_paths, q,
                              antithetic, rng, seed)
    fixings = paths[:, 1:]  # exclude t=0 from the average
    arithmetic = fixings.mean(axis=1)
    intrinsic = arithmetic - K if kind == "call" else K - arithmetic
    payoffs = np.exp(-r * T) * np.maximum(intrinsic, 0.0)

    controls, control_mean = None, None
    if control_variate and kind == "call":
        geometric = np.exp(np.log(fixings).mean(axis=1))
        controls = np.exp(-r * T) * np.maximum(geometric - K, 0.0)
        control_mean = geometric_asian_call(S0, K, r, sigma, T, n_fixings, q)

    return _summarise(payoffs, antithetic, controls, control_mean,
                      _label(antithetic, control_variate and kind == "call"))


def price_barrier_up_and_out_call(S0, K, barrier, r, sigma, T, n_paths=50_000,
                                  n_steps=252, q=0.0, antithetic=True,
                                  rng=None, seed=None):
    """Up-and-out call, monitored discretely on the simulation grid.

    Note the bias: discrete monitoring can miss a barrier breach that happened
    *between* two grid points, so this systematically **overprices** the
    knock-out relative to continuous monitoring.  Increasing ``n_steps``
    reduces the bias but never removes it -- a real limitation worth stating
    rather than hiding.
    """
    _, paths = simulate_paths(S0, r, sigma, T, n_steps, n_paths, q,
                              antithetic, rng, seed)
    survived = paths.max(axis=1) < barrier
    payoffs = np.exp(-r * T) * np.maximum(paths[:, -1] - K, 0.0) * survived
    return _summarise(payoffs, antithetic, method=_label(antithetic, False))


# --------------------------------------------------------------------------
# Greeks
# --------------------------------------------------------------------------

def mc_greeks_fd(S0, K, r, sigma, T, n_paths=200_000, q=0.0, kind="call",
                 seed=12345, h_rel=0.01, h_vol=0.01, h_rate=1e-4):
    """Delta, gamma, vega and rho by finite differences with common random numbers.

    The seed is deliberately reused across every bumped valuation.  With
    independent draws the difference of two noisy prices is dominated by noise
    -- the estimate of delta would be worse than useless at any practical path
    count.  Reusing the draws makes the noise cancel almost entirely.  This is
    the single most important trick in finite-difference Monte Carlo Greeks.
    """
    def price(spot=S0, vol=sigma, rate=r):
        return price_european(spot, K, rate, vol, T, n_paths, q, kind,
                              antithetic=True, control_variate=True, seed=seed).value

    h_s = S0 * h_rel
    base = price()
    up, down = price(spot=S0 + h_s), price(spot=S0 - h_s)

    return {
        "delta": (up - down) / (2 * h_s),
        "gamma": (up - 2 * base + down) / h_s**2,
        "vega": (price(vol=sigma + h_vol) - price(vol=sigma - h_vol)) / (2 * h_vol),
        "rho": (price(rate=r + h_rate) - price(rate=r - h_rate)) / (2 * h_rate),
    }
