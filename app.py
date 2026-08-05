"""Streamlit front end for the Monte Carlo option pricing engine.

Run locally:   streamlit run app.py
Deploy:        push to GitHub, then share.streamlit.io -> "New app"
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

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
    simulate_paths,
)

st.set_page_config(page_title="Monte Carlo Option Pricer", page_icon="~", layout="wide")

ACCENT = "#4c8bf5"
GRID = {"alpha": 0.25, "linewidth": 0.6}


def styled_axes(figsize=(7, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(True, **GRID)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return fig, ax


# ---------------------------------------------------------------- sidebar

st.sidebar.title("Contract & market")
S0 = st.sidebar.number_input("Spot price S0", 1.0, 10_000.0, 100.0, step=1.0)
K = st.sidebar.number_input("Strike K", 1.0, 10_000.0, 100.0, step=1.0)
T = st.sidebar.number_input("Maturity T (years)", 0.01, 30.0, 1.0, step=0.25)
r = st.sidebar.slider("Risk-free rate r", -0.02, 0.20, 0.05, 0.005, format="%.3f")
q = st.sidebar.slider("Dividend yield q", 0.00, 0.15, 0.00, 0.005, format="%.3f")
sigma = st.sidebar.slider("Volatility sigma", 0.01, 1.50, 0.20, 0.01)

st.sidebar.title("Simulation")
n_paths = st.sidebar.select_slider(
    "Paths", [10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000], value=100_000
)
antithetic = st.sidebar.checkbox("Antithetic variates", True)
control_variate = st.sidebar.checkbox("Control variate", True)
seed = st.sidebar.number_input("Seed", 0, 10**6, 42, step=1)

MARKET = dict(S0=S0, K=K, r=r, sigma=sigma, T=T, q=q)


@st.cache_data(show_spinner=False)
def cached_european(market, n, anti, cv, sd, kind):
    return price_european(**market, n_paths=n, kind=kind, antithetic=anti,
                          control_variate=cv, seed=sd)


@st.cache_data(show_spinner=False)
def cached_convergence(market, sd, anti, cv):
    return convergence_study(**market, antithetic=anti, control_variate=cv, seed=sd)


# ---------------------------------------------------------------- header

st.title("Monte Carlo Option Pricing")
st.caption(
    "Every price below carries a standard error. A Monte Carlo number without "
    "one tells you nothing about how much of it is signal."
)

tab_price, tab_conv, tab_paths, tab_greeks, tab_exotic = st.tabs(
    ["Pricing", "Convergence", "Paths", "Greeks", "Exotics"]
)

# ---------------------------------------------------------------- pricing

with tab_price:
    st.subheader("European options: Monte Carlo against the exact answer")
    st.write(
        "European payoffs have a closed form, so this comparison is not a "
        "pricing exercise — it is a **correctness test** for the engine. "
        "If the exact price does not sit inside the confidence interval, "
        "something is wrong."
    )

    cols = st.columns(2)
    for col, kind in zip(cols, ("call", "put"), strict=True):
        est = cached_european(MARKET, n_paths, antithetic, control_variate, seed, kind)
        exact = black_scholes_price(**MARKET, kind=kind)
        lo, hi = est.ci95
        with col:
            st.metric(f"{kind.title()} — Monte Carlo", f"{est.value:.4f}",
                      f"{est.value - exact:+.4f} vs exact")
            st.write(f"**Black-Scholes (exact):** {exact:.4f}")
            st.write(f"**Standard error:** {est.std_error:.4f}")
            st.write(f"**95% CI:** [{lo:.4f}, {hi:.4f}]")
            sig = est.sigmas_from(exact)
            st.write(f"**Distance from exact:** {sig:.2f} standard errors")
            if sig < 2:
                st.success("Within noise — consistent with the exact price.")
            elif sig < 4:
                st.warning("Marginal. Rerun with a different seed before concluding.")
            else:
                st.error("Outside plausible sampling error — investigate.")
            st.caption(f"Method: {est.method}")

    st.divider()
    st.subheader("What the variance reduction buys")
    rows, baseline = [], None
    for anti, cv in [(False, False), (True, False), (False, True), (True, True)]:
        e = cached_european(MARKET, n_paths, anti, cv, seed, "call")
        baseline = baseline or e.std_error
        ratio = baseline / e.std_error if e.std_error else float("inf")
        rows.append({
            "Method": e.method,
            "Price": round(e.value, 4),
            "Std error": round(e.std_error, 5),
            "Error reduction": f"{ratio:.1f}x",
            "Equivalent path saving": f"{ratio ** 2:.0f}x",
        })
    st.dataframe(rows, width="stretch", hide_index=True)
    st.caption(
        "Error falls as 1/sqrt(N), so halving the standard error is worth "
        "quadrupling the path count. The last column is that trade expressed "
        "as the paths you no longer have to simulate."
    )

# ---------------------------------------------------------------- convergence

with tab_conv:
    st.subheader("Does the error really decay like 1/sqrt(N)?")
    study = cached_convergence(MARKET, seed, antithetic, control_variate)
    slope = fitted_slope(study["n_paths"], study["std_error"])

    fig, ax = styled_axes((7, 4.2))
    ax.loglog(study["n_paths"], study["std_error"], "o-", color=ACCENT,
              label="measured standard error")
    ref = study["std_error"][0] * np.sqrt(study["n_paths"][0] / study["n_paths"])
    ax.loglog(study["n_paths"], ref, "--", color="grey", label=r"theoretical $1/\sqrt{N}$")
    ax.set_xlabel("paths")
    ax.set_ylabel("standard error")
    ax.legend(frameon=False)
    st.pyplot(fig, width="content")

    c1, c2 = st.columns(2)
    c1.metric("Fitted log-log slope", f"{slope:.4f}")
    c2.metric("Theoretical slope", "-0.5000")
    st.write(
        f"The fitted slope is **{slope:.4f}** against a theoretical **-0.5**. "
        "This is the central limitation of Monte Carlo: one extra decimal place "
        "of accuracy costs a hundred times the computation. Variance reduction "
        "shifts this line downward; it does not change its slope."
    )

    fig2, ax2 = styled_axes((7, 3.6))
    ax2.semilogx(study["n_paths"], study["price"], "o-", color=ACCENT, label="MC price")
    ax2.fill_between(study["n_paths"],
                     study["price"] - 1.96 * study["std_error"],
                     study["price"] + 1.96 * study["std_error"],
                     alpha=0.2, color=ACCENT, label="95% CI")
    ax2.axhline(study["exact"], color="crimson", lw=1.2, label="exact")
    ax2.set_xlabel("paths")
    ax2.set_ylabel("call price")
    ax2.legend(frameon=False)
    st.pyplot(fig2, width="content")

# ---------------------------------------------------------------- paths

with tab_paths:
    st.subheader("Simulated price paths")
    n_show = st.slider("Paths to display", 5, 200, 40)
    n_steps = st.slider("Time steps", 12, 504, 252)

    _, paths = simulate_paths(S0=S0, r=r, sigma=sigma, T=T, n_steps=n_steps,
                              n_paths=n_show, q=q, seed=seed)
    t = np.linspace(0, T, n_steps + 1)

    fig, ax = styled_axes((8, 4.5))
    ax.plot(t, paths.T, lw=0.8, alpha=0.55)
    ax.axhline(K, color="crimson", ls="--", lw=1.2, label=f"strike {K:g}")
    # Scale to what is actually drawn, not to a larger hidden sample.
    ax.set_ylim(paths.min() * 0.97, paths.max() * 1.03)
    ax.set_xlim(0, T)
    ax.set_xlabel("time (years)")
    ax.set_ylabel("price")
    ax.legend(frameon=False)
    st.pyplot(fig, width="content")

    terminal = paths[:, -1]
    st.write(
        f"Of the {n_show} displayed paths, "
        f"**{(terminal > K).sum()}** finish in the money."
    )

# ---------------------------------------------------------------- greeks

with tab_greeks:
    st.subheader("Greeks: finite differences vs closed form")
    st.write(
        "Bumped valuations reuse the **same random draws** (common random "
        "numbers). Without that, the difference of two noisy prices is mostly "
        "noise and the estimates are unusable."
    )
    try:
        mc = mc_greeks_fd(**MARKET, n_paths=min(n_paths, 200_000), seed=seed)
        exact_g = black_scholes_greeks(**MARKET)
        st.dataframe(
            [{"Greek": g, "Monte Carlo": round(mc[g], 5),
              "Analytic": round(exact_g[g], 5),
              "Abs difference": f"{abs(mc[g] - exact_g[g]):.2e}"}
             for g in ("delta", "gamma", "vega", "rho")],
            width="stretch", hide_index=True,
        )
        st.caption("vega is per 1.0 of volatility; rho is per 1.0 of rate.")
    except ValueError as exc:
        st.error(str(exc))

# ---------------------------------------------------------------- exotics

with tab_exotic:
    st.subheader("Where Monte Carlo actually earns its keep")
    st.write(
        "Neither payoff below has a closed-form solution, which is the real "
        "reason to simulate at all."
    )

    left, right = st.columns(2)

    with left:
        st.markdown("**Arithmetic average-price Asian call**")
        n_fix = st.slider("Fixings", 4, 252, 52)
        asian = price_asian(**MARKET, n_paths=min(n_paths, 200_000), n_fixings=n_fix,
                            antithetic=antithetic, control_variate=control_variate,
                            seed=seed)
        geo = geometric_asian_call(**MARKET, n_fixings=n_fix)
        st.metric("Asian call", f"{asian.value:.4f}", f"SE {asian.std_error:.5f}")
        st.write(f"Geometric Asian (closed form, used as control): **{geo:.4f}**")
        st.write(f"European call for reference: **{black_scholes_price(**MARKET):.4f}**")
        st.caption(
            "Averaging damps the volatility of the payoff, so the Asian is "
            "always cheaper than the European. The geometric version is exactly "
            "solvable and nearly perfectly correlated with the arithmetic one, "
            "which is what makes it such an effective control variate."
        )

    with right:
        st.markdown("**Up-and-out barrier call**")
        barrier = st.number_input("Barrier H", float(K), 10_000.0,
                                  float(max(K * 1.3, K + 1)), step=1.0)
        n_mon = st.slider("Monitoring dates", 12, 504, 252)
        bar = price_barrier_up_and_out_call(**MARKET, barrier=barrier,
                                            n_paths=min(n_paths, 200_000),
                                            n_steps=n_mon, antithetic=antithetic,
                                            seed=seed)
        st.metric("Barrier call", f"{bar.value:.4f}", f"SE {bar.std_error:.5f}")
        st.write(f"Vanilla call for reference: **{black_scholes_price(**MARKET):.4f}**")
        st.caption(
            "Monitored discretely, so a breach between two dates is missed and "
            "the knock-out is systematically overpriced relative to continuous "
            "monitoring. Raising the monitoring frequency shrinks the bias but "
            "never eliminates it."
        )
