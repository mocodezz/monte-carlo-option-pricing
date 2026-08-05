"""Regenerate the figures embedded in the README.

    python scripts/make_figures.py

Writes PNGs to docs/figures/. Deterministic: same seeds, same output.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcopt import (
    convergence_study,
    fitted_slope,
    price_european,
    simulate_paths,
)

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
MARKET = dict(S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0)

INK = "#1f2328"
ACCENT = "#2f6fed"
MUTED = "#8b949e"
WARN = "#d1242f"


def style(ax) -> None:
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def save(fig, name: str, dpi: int = 160) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path}")


def fig_convergence() -> None:
    study = convergence_study(**MARKET)
    slope = fitted_slope(study["n_paths"], study["std_error"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.1))

    ax1.loglog(study["n_paths"], study["std_error"], "o-", color=ACCENT,
               markersize=4, linewidth=1.6, label="measured standard error")
    ref = study["std_error"][0] * np.sqrt(study["n_paths"][0] / study["n_paths"])
    ax1.loglog(study["n_paths"], ref, "--", color=MUTED, linewidth=1.4,
               label=r"theoretical $1/\sqrt{N}$")
    ax1.set_xlabel("paths (N)")
    ax1.set_ylabel("standard error")
    ax1.set_title(f"Error decay: fitted slope {slope:.4f} vs theory -0.5", fontsize=11)
    ax1.legend(frameon=False, fontsize=9)
    style(ax1)

    ax2.semilogx(study["n_paths"], study["price"], "o-", color=ACCENT,
                 markersize=4, linewidth=1.6, label="Monte Carlo estimate")
    ax2.fill_between(study["n_paths"],
                     study["price"] - 1.96 * study["std_error"],
                     study["price"] + 1.96 * study["std_error"],
                     alpha=0.18, color=ACCENT, label="95% confidence interval")
    ax2.axhline(study["exact"], color=WARN, linewidth=1.3, label="exact Black-Scholes")
    ax2.set_xlabel("paths (N)")
    ax2.set_ylabel("call price")
    ax2.set_title("The interval narrows onto the exact price", fontsize=11)
    ax2.legend(frameon=False, fontsize=9)
    style(ax2)

    fig.tight_layout()
    save(fig, "convergence.png")


def fig_variance_reduction() -> None:
    labels, errors = [], []
    for anti, cv in [(False, False), (True, False), (False, True), (True, True)]:
        est = price_european(**MARKET, n_paths=100_000, antithetic=anti,
                             control_variate=cv, seed=42)
        labels.append(est.method.replace(" + ", "\n+ "))
        errors.append(est.std_error)

    baseline = errors[0]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    bars = ax.bar(labels, errors, color=[MUTED, MUTED, MUTED, ACCENT], width=0.62)
    for bar, err in zip(bars, errors, strict=True):
        ax.annotate(f"{err:.5f}\n({baseline / err:.1f}x less error)",
                    (bar.get_x() + bar.get_width() / 2, err),
                    textcoords="offset points", xytext=(0, 5),
                    ha="center", fontsize=9, color=INK)
    ax.set_ylabel("standard error")
    ax.set_ylim(0, max(errors) * 1.35)
    ax.set_title("Variance reduction, European call, 100,000 paths", fontsize=11)
    style(ax)
    fig.tight_layout()
    save(fig, "variance_reduction.png")


def fig_paths() -> None:
    n_show, n_steps = 60, 252
    t, paths = simulate_paths(S0=100.0, r=0.05, sigma=0.2, T=1.0,
                              n_steps=n_steps, n_paths=n_show, seed=11)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11, 4.1), gridspec_kw={"width_ratios": [2.4, 1]}
    )

    terminal = paths[:, -1]
    for path, s_t in zip(paths, terminal, strict=True):
        ax1.plot(t, path, linewidth=0.8, alpha=0.55,
                 color=ACCENT if s_t > MARKET["K"] else MUTED)
    ax1.axhline(MARKET["K"], color=WARN, linestyle="--", linewidth=1.3,
                label=f"strike {MARKET['K']:g}")
    # Axis limits follow what is actually drawn.
    ax1.set_ylim(paths.min() * 0.97, paths.max() * 1.03)
    ax1.set_xlim(0, MARKET["T"])
    ax1.set_xlabel("time (years)")
    ax1.set_ylabel("price")
    ax1.set_title(f"{n_show} GBM paths — blue finish in the money", fontsize=11)
    ax1.legend(frameon=False, fontsize=9)
    style(ax1)

    big = simulate_paths(S0=100.0, r=0.05, sigma=0.2, T=1.0, n_steps=1,
                         n_paths=200_000, seed=12)[1][:, -1]
    ax2.hist(big, bins=90, color=ACCENT, alpha=0.75, edgecolor="none")
    ax2.axvline(MARKET["K"], color=WARN, linestyle="--", linewidth=1.3)
    ax2.set_xlabel(r"terminal price $S_T$")
    ax2.set_ylabel("frequency")
    ax2.set_title("Lognormal terminal distribution", fontsize=11)
    ax2.set_xlim(0, np.percentile(big, 99.5))
    style(ax2)

    fig.tight_layout()
    save(fig, "paths.png", dpi=120)


if __name__ == "__main__":
    fig_convergence()
    fig_variance_reduction()
    fig_paths()
