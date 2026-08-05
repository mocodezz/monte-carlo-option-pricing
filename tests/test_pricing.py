"""Tests that actually assert something.

The interesting ones are not "does it run" but:
  * does Monte Carlo land within its own stated error of the exact answer,
  * does each variance-reduction technique measurably reduce that error,
  * is the closed-form geometric Asian correct (checked against a Monte Carlo
    of the same payoff, which is an independent derivation),
  * are the finite-difference Greeks close to the analytic ones.
"""

import numpy as np
import pytest

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

BASE = dict(S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0)
# sample_terminal / simulate_paths describe the *process*, not the contract,
# so they take no strike.
PROCESS = {k: v for k, v in BASE.items() if k != "K"}


# ---------------------------------------------------------------- analytic

@pytest.mark.parametrize("kind", ["call", "put"])
def test_put_call_parity(kind):
    c = black_scholes_price(**BASE, kind="call")
    p = black_scholes_price(**BASE, kind="put")
    lhs = c - p
    rhs = BASE["S0"] - BASE["K"] * np.exp(-BASE["r"] * BASE["T"])
    assert lhs == pytest.approx(rhs, abs=1e-10)


def test_known_black_scholes_value():
    # Standard textbook case: S=K=100, r=5%, vol=20%, T=1 -> ~10.4506
    assert black_scholes_price(**BASE) == pytest.approx(10.450583572, abs=1e-6)


@pytest.mark.parametrize("degenerate", [dict(T=0.0), dict(sigma=0.0)])
def test_degenerate_inputs_do_not_raise(degenerate):
    """The original code divided by sigma*sqrt(T) and blew up on both."""
    args = {**BASE, **degenerate}
    itm = black_scholes_price(**{**args, "K": 80.0}, kind="call")
    otm = black_scholes_price(**{**args, "K": 120.0}, kind="call")
    assert itm > 0 and otm == pytest.approx(0.0, abs=1e-12)


def test_call_price_monotone_in_volatility():
    lo = black_scholes_price(**{**BASE, "sigma": 0.10})
    hi = black_scholes_price(**{**BASE, "sigma": 0.40})
    assert hi > lo


# ---------------------------------------------------------------- sampling

def test_terminal_sampling_matches_forward():
    s_t = sample_terminal(**PROCESS, n_paths=400_000, seed=1)
    expected = BASE["S0"] * np.exp(BASE["r"] * BASE["T"])
    se = s_t.std(ddof=1) / np.sqrt(s_t.size)
    assert abs(s_t.mean() - expected) < 4 * se


def test_paths_start_at_spot_and_have_right_shape():
    t, paths = simulate_paths(**PROCESS, n_steps=50, n_paths=1_000, seed=3)
    assert paths.shape == (1_000, 51)
    assert t.shape == (51,)
    assert np.allclose(paths[:, 0], BASE["S0"])


def test_antithetic_draws_are_mirrored():
    s_t = sample_terminal(**PROCESS, n_paths=1_000, antithetic=True, seed=5)
    # log returns of paired paths must sum to twice the drift
    log_ret = np.log(s_t / BASE["S0"])
    a, b = log_ret[:500], log_ret[500:]
    drift = (BASE["r"] - 0.5 * BASE["sigma"] ** 2) * BASE["T"]
    assert np.allclose(a + b, 2 * drift)


def test_path_grid_memory_guard():
    with pytest.raises(MemoryError):
        simulate_paths(**PROCESS, n_steps=10_000, n_paths=10_000_000, seed=1)


# ---------------------------------------------------------------- european MC

@pytest.mark.parametrize("kind", ["call", "put"])
def test_mc_agrees_with_black_scholes_within_stated_error(kind):
    exact = black_scholes_price(**BASE, kind=kind)
    est = price_european(**BASE, n_paths=200_000, kind=kind, seed=42)
    assert est.sigmas_from(exact) < 4.0, f"{est} vs exact {exact:.4f}"


def test_confidence_interval_brackets_the_truth():
    exact = black_scholes_price(**BASE)
    lo, hi = price_european(**BASE, n_paths=200_000, seed=11).ci95
    assert lo < exact < hi


def test_antithetic_reduces_standard_error():
    plain = price_european(**BASE, n_paths=100_000, antithetic=False,
                           control_variate=False, seed=99)
    anti = price_european(**BASE, n_paths=100_000, antithetic=True,
                          control_variate=False, seed=99)
    assert anti.std_error < plain.std_error


def test_control_variate_reduces_standard_error_a_lot():
    plain = price_european(**BASE, n_paths=100_000, antithetic=False,
                           control_variate=False, seed=7)
    cv = price_european(**BASE, n_paths=100_000, antithetic=False,
                        control_variate=True, seed=7)
    assert cv.std_error < plain.std_error / 2


def test_seed_is_reproducible():
    a = price_european(**BASE, n_paths=20_000, seed=2024)
    b = price_european(**BASE, n_paths=20_000, seed=2024)
    c = price_european(**BASE, n_paths=20_000, seed=2025)
    assert a.value == b.value
    assert a.value != c.value


def test_antithetic_requires_even_paths():
    with pytest.raises(ValueError):
        price_european(**BASE, n_paths=1_001, antithetic=True, seed=1)


def test_deep_out_of_the_money_call_is_near_zero():
    est = price_european(**{**BASE, "K": 400.0}, n_paths=100_000, seed=8)
    assert 0.0 <= est.value < 0.01


# ---------------------------------------------------------------- asian

def test_geometric_asian_closed_form_matches_its_own_monte_carlo():
    """Independent check of the formula used as a control variate."""
    n_fix = 12
    exact = geometric_asian_call(**BASE, n_fixings=n_fix)
    _, paths = simulate_paths(**PROCESS, n_steps=n_fix, n_paths=400_000, seed=17)
    geo = np.exp(np.log(paths[:, 1:]).mean(axis=1))
    payoff = np.exp(-BASE["r"] * BASE["T"]) * np.maximum(geo - BASE["K"], 0.0)
    se = payoff.std(ddof=1) / np.sqrt(payoff.size)
    assert abs(payoff.mean() - exact) < 4 * se


def test_arithmetic_asian_exceeds_geometric_asian():
    """AM >= GM pointwise, so the arithmetic option must be worth more."""
    n_fix = 12
    geo = geometric_asian_call(**BASE, n_fixings=n_fix)
    ari = price_asian(**BASE, n_paths=100_000, n_fixings=n_fix, seed=21)
    assert ari.value > geo


def test_asian_is_cheaper_than_european():
    """Averaging damps volatility, so the Asian call is worth less."""
    euro = black_scholes_price(**BASE)
    asian = price_asian(**BASE, n_paths=100_000, n_fixings=52, seed=23)
    assert asian.value < euro


def test_geometric_control_variate_slashes_asian_error():
    plain = price_asian(**BASE, n_paths=50_000, n_fixings=52,
                        antithetic=False, control_variate=False, seed=31)
    cv = price_asian(**BASE, n_paths=50_000, n_fixings=52,
                     antithetic=False, control_variate=True, seed=31)
    assert cv.std_error < plain.std_error / 10


# ---------------------------------------------------------------- barrier

def test_knocked_out_call_is_worth_less_than_vanilla():
    vanilla = black_scholes_price(**BASE)
    barrier = price_barrier_up_and_out_call(**BASE, barrier=130.0,
                                            n_paths=50_000, n_steps=252, seed=5)
    assert 0 < barrier.value < vanilla


def test_unreachable_barrier_recovers_vanilla_price():
    vanilla = black_scholes_price(**BASE)
    est = price_barrier_up_and_out_call(**BASE, barrier=1e6, n_paths=50_000,
                                        n_steps=64, seed=6)
    assert est.sigmas_from(vanilla) < 4.0


# ---------------------------------------------------------------- greeks

def test_finite_difference_greeks_match_analytic():
    exact = black_scholes_greeks(**BASE)
    mc = mc_greeks_fd(**BASE, n_paths=200_000, seed=4242)
    assert mc["delta"] == pytest.approx(exact["delta"], abs=5e-3)
    assert mc["vega"] == pytest.approx(exact["vega"], rel=5e-2)
    assert mc["rho"] == pytest.approx(exact["rho"], rel=5e-2)
    assert mc["gamma"] == pytest.approx(exact["gamma"], abs=5e-3)


def test_call_delta_is_between_zero_and_one():
    mc = mc_greeks_fd(**BASE, n_paths=100_000, seed=1)
    assert 0.0 < mc["delta"] < 1.0


# ---------------------------------------------------------------- convergence

def test_error_decays_at_the_theoretical_rate():
    study = convergence_study(**BASE, path_counts=np.logspace(2, 5.5, 12).astype(int))
    slope = fitted_slope(study["n_paths"], study["std_error"])
    assert slope == pytest.approx(-0.5, abs=0.06), f"slope was {slope:.3f}"
