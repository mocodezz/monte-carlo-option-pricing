"""Reproduce every number quoted in the README.

    python benchmark.py

Nothing in the README is asserted without being printed here first.
"""

from __future__ import annotations

import time

import numpy as np

from mcopt import (
    black_scholes_greeks,
    black_scholes_price,
    convergence_study,
    fitted_slope,
    geometric_asian_call,
    mc_greeks_fd,
    price_asian,
    price_barrier_up_and_out_call,
    price_european,
    sample_terminal,
    simulate_paths,
)

MARKET = dict(S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0)
RULE = "-" * 78


def header(text: str) -> None:
    print(f"\n{RULE}\n{text}\n{RULE}")


def european_table() -> None:
    header("EUROPEAN CALL -- 100,000 paths -- effect of variance reduction")
    exact = black_scholes_price(**MARKET)
    print(f"Exact Black-Scholes price: {exact:.6f}\n")
    print(f"{'method':<30}{'price':>10}{'std error':>12}{'error cut':>12}{'path saving':>14}")
    baseline = None
    for anti, cv in [(False, False), (True, False), (False, True), (True, True)]:
        est = price_european(**MARKET, n_paths=100_000, antithetic=anti,
                             control_variate=cv, seed=42)
        baseline = baseline or est.std_error
        ratio = baseline / est.std_error
        print(f"{est.method:<30}{est.value:>10.4f}{est.std_error:>12.5f}"
              f"{ratio:>11.1f}x{ratio ** 2:>13.0f}x")


def sampling_cost() -> None:
    header("COST OF THE PATH GRID -- 100,000 paths")
    t0 = time.perf_counter()
    simulate_paths(S0=100, r=0.05, sigma=0.2, T=1.0, n_steps=252,
                   n_paths=100_000, seed=1)
    grid = time.perf_counter() - t0

    t0 = time.perf_counter()
    sample_terminal(S0=100, r=0.05, sigma=0.2, T=1.0, n_paths=100_000, seed=1)
    terminal = time.perf_counter() - t0

    print(f"full 252-step path grid : {grid:.4f}s")
    print(f"terminal values only    : {terminal:.4f}s")
    print(f"speedup for a European payoff, which needs only the terminal value: "
          f"{grid / terminal:.0f}x")


def asian_table() -> None:
    header("ARITHMETIC ASIAN CALL (52 fixings) -- no closed form exists")
    baseline = None
    for anti, cv in [(False, False), (True, True)]:
        est = price_asian(**MARKET, n_paths=50_000, n_fixings=52,
                          antithetic=anti, control_variate=cv, seed=31)
        baseline = baseline or est.std_error
        print(f"{est.method:<30}{est.value:>10.4f}{est.std_error:>12.5f}"
              f"{baseline / est.std_error:>11.1f}x")
    print(f"{'geometric Asian (closed form)':<30}"
          f"{geometric_asian_call(**MARKET, n_fixings=52):>10.4f}")
    print(f"{'European call (reference)':<30}{black_scholes_price(**MARKET):>10.4f}")


def convergence_rate() -> None:
    header("CONVERGENCE RATE")
    study = convergence_study(**MARKET)
    slope = fitted_slope(study["n_paths"], study["std_error"])
    print(f"fitted log-log slope of standard error vs paths : {slope:.4f}")
    print("Monte Carlo theory predicts                     : -0.5000")


def greeks_table() -> None:
    header("GREEKS -- finite difference with common random numbers")
    mc = mc_greeks_fd(**MARKET, n_paths=200_000, seed=4242)
    exact = black_scholes_greeks(**MARKET)
    print(f"{'greek':<10}{'monte carlo':>14}{'analytic':>14}{'abs diff':>14}")
    for name in ("delta", "gamma", "vega", "rho"):
        print(f"{name:<10}{mc[name]:>14.5f}{exact[name]:>14.5f}"
              f"{abs(mc[name] - exact[name]):>14.2e}")


def barrier_example() -> None:
    header("UP-AND-OUT BARRIER CALL, H = 130")
    est = price_barrier_up_and_out_call(**MARKET, barrier=130.0,
                                        n_paths=100_000, n_steps=252, seed=5)
    print(est)
    print(f"vanilla call for reference: {black_scholes_price(**MARKET):.4f}")


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    european_table()
    sampling_cost()
    asian_table()
    convergence_rate()
    greeks_table()
    barrier_example()
    print()
