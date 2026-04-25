"""Produce publication-grade figures for the QF submission.

Outputs (PDF) to qf/figures/:
    fig_mahalanobis.pdf       - analytical d^2 vs rho0 (+ best-singleton baseline)
    fig_amp_surface.pdf       - empirical AUC amplification over (rho0, sigma_eps)
    fig_misspecification.pdf  - Fisher-optimal vs fixed-weight AUC wedge
    fig_deterrence.pdf        - equilibrium r*, c* vs detection sensitivity
    fig_welfare_wedge.pdf     - private vs social optimal threshold across F/H ratios
"""
from __future__ import annotations

import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from equilibrium import (
    GameParams,
    best_response,
    deterrence_sweep,
    optimise_threshold,
    private_welfare,
    social_welfare,
)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 160,
    "savefig.bbox": "tight",
})

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Figure 1: Analytical Mahalanobis deflection vs rho0
# ---------------------------------------------------------------------------
def fig_mahalanobis(mu_r: float = 1.5, mu_c: float = 0.6,
                    sigma_r: float = 0.6, sigma_c: float = 0.5) -> None:
    rho = np.linspace(-0.95, 0.95, 401)
    a = (mu_r ** 2) / sigma_r ** 2 + (mu_c ** 2) / sigma_c ** 2
    b = 2 * mu_r * mu_c / (sigma_r * sigma_c)
    d2 = (a - rho * b) / (1 - rho ** 2)
    d2_singleton_r = (mu_r ** 2) / sigma_r ** 2
    d2_singleton_c = (mu_c ** 2) / sigma_c ** 2
    d2_best_single = max(d2_singleton_r, d2_singleton_c)

    disc = max(a ** 2 - b ** 2, 0.0)
    rho_min = b / (a + np.sqrt(disc)) if disc > 0 else b / (2 * a)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(rho, d2, lw=2.0, color="#1f4e79",
            label=r"Fisher-optimal $d^{\star 2}(\rho_0)$")
    ax.axhline(d2_best_single, ls="--", color="#b35900",
               label=rf"Best singleton ($d^2={d2_best_single:.2f}$)")
    ax.axvline(rho_min, ls=":", color="#7f7f7f",
               label=rf"Minimiser $\rho_0^\star \approx {rho_min:.2f}$")
    ax.scatter([rho_min], [(a - rho_min * b) / (1 - rho_min ** 2)],
               color="#7f7f7f", zorder=5)
    ax.set_xlabel(r"Benign-flow correlation $\rho_0 = \mathrm{Corr}(r,c\mid H_0)$")
    ax.set_ylabel(r"Deflection $d^{\star 2} = \mu^\top \Sigma_0^{-1}\mu$")
    ax.set_title("Fisher-Mahalanobis amplification geometry")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper center")
    ax.set_ylim(0, min(d2.max() * 1.05, 60))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_mahalanobis.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: Empirical amplification surface
# ---------------------------------------------------------------------------
def fig_amp_surface(csv_path: str | None = None) -> None:
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(__file__), "results", "grid_primary.csv")
    df = pd.read_csv(csv_path)
    # Focus on rho1=0.3 (calibrated manipulation complementarity)
    df = df[df["rho1"] == 0.3].copy()
    agg = df.groupby(["rho0", "sigma_eps"]).agg(
        amp_opt_mean=("amp_opt", "mean"),
        amp_opt_sd=("amp_opt", "std"),
        amp_fix_mean=("amp_fix", "mean"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    markers = {0.0: "o", 0.25: "s", 0.5: "^"}
    colors = {0.0: "#1f4e79", 0.25: "#b35900", 0.5: "#4c7a34"}
    for se, grp in agg.groupby("sigma_eps"):
        g = grp.sort_values("rho0")
        ax.errorbar(g["rho0"], g["amp_opt_mean"], yerr=g["amp_opt_sd"],
                    marker=markers.get(se, "o"), color=colors.get(se, "k"),
                    lw=1.6, capsize=3,
                    label=rf"$\sigma_\varepsilon={se:.2f}$")
    ax.set_xlabel(r"$\rho_0$")
    ax.set_ylabel(r"AUC amplification: $\mathrm{AUC}(w^\star) - \max_k \mathrm{AUC}(e_k)$")
    ax.set_title("Empirical amplification (Fisher-optimal weights)")
    ax.axhline(0, color="k", lw=0.5)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_amp_surface.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: Misspecification wedge
# ---------------------------------------------------------------------------
def fig_misspecification(csv_path: str | None = None) -> None:
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(__file__), "results", "grid_primary.csv")
    df = pd.read_csv(csv_path)
    df = df[(df["rho1"] == 0.3) & (df["sigma_eps"] == 0.25)].copy()
    agg = df.groupby("rho0").agg(
        opt=("auc_opt", "mean"),
        fix=("auc_fix", "mean"),
    ).reset_index().sort_values("rho0")

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(agg["rho0"], agg["opt"], "o-", color="#1f4e79", lw=2.0,
            label=r"Fisher-optimal $w^\star=\Sigma_0^{-1}\mu$")
    ax.plot(agg["rho0"], agg["fix"], "s--", color="#b35900", lw=2.0,
            label=r"Fixed weights $w=(1,0.8)$")
    ax.set_xlabel(r"$\rho_0$")
    ax.set_ylabel("Composite AUC")
    ax.set_title(r"Efficiency loss from weight misspecification ($\sigma_\varepsilon=0.25$)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_misspecification.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: Deterrence
# ---------------------------------------------------------------------------
def fig_deterrence(p: GameParams | None = None) -> None:
    p = p or GameParams()
    grid_a = np.linspace(0.3, 2.2, 20)
    grid_b = np.linspace(0.3, 2.2, 20)
    sweep_a = deterrence_sweep(p, grid_a, keep="alpha")
    sweep_b = deterrence_sweep(p, grid_b, keep="beta")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=False)
    ax = axes[0]
    ax.plot(sweep_a["alpha"], sweep_a["r"], "o-", color="#1f4e79", lw=2.0,
            label=r"$r^\star(\alpha)$, $\beta$ fixed")
    ax.plot(sweep_b["alpha"], sweep_b["r"], "s--", color="#b35900", lw=2.0,
            label=r"$r^\star(\beta)$, $\alpha$ fixed")
    ax.set_xlabel(r"Detection sensitivity")
    ax.set_ylabel(r"Equilibrium rush-order intensity $r^\star$")
    ax.set_title(r"Strategic deterrence: rush orders")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(sweep_a["alpha"], sweep_a["c"], "o-", color="#1f4e79", lw=2.0,
            label=r"$c^\star(\alpha)$, $\beta$ fixed")
    ax.plot(sweep_b["alpha"], sweep_b["c"], "s--", color="#b35900", lw=2.0,
            label=r"$c^\star(\beta)$, $\alpha$ fixed")
    ax.set_xlabel(r"Detection sensitivity")
    ax.set_ylabel(r"Equilibrium cancellation ratio $c^\star$")
    ax.set_title(r"Strategic deterrence: cancellations")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_deterrence.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: Private vs social threshold wedge
# ---------------------------------------------------------------------------
def fig_welfare_wedge(p: GameParams | None = None) -> None:
    # Use a smaller private penalty L so the wedge is clearly visible across H.
    base = p or GameParams(L=40.0, F=15.0, xi=50.0, zeta=35.0, cD=1.5)
    H_grid = np.linspace(50, 500, 19)
    tau_priv, tau_soc = [], []
    tau_grid = np.linspace(-1.0, 3.0, 161)
    for Hv in H_grid:
        pars = GameParams(**{**base.__dict__, "H": float(Hv)})
        tp, _ = optimise_threshold(private_welfare, pars, tau_grid)
        ts, _ = optimise_threshold(social_welfare, pars, tau_grid)
        tau_priv.append(tp); tau_soc.append(ts)
    tau_priv = np.array(tau_priv); tau_soc = np.array(tau_soc)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(H_grid, tau_priv, "o-", color="#1f4e79", lw=2.0, label=r"Private optimum $\tau^\star_P$")
    ax.plot(H_grid, tau_soc, "s--", color="#b35900", lw=2.0, label=r"Social optimum $\tau^\star_{SO}$")
    ax.fill_between(H_grid, tau_soc, tau_priv,
                    where=(tau_priv > tau_soc), alpha=0.18, color="#b35900",
                    label="Underinvestment wedge")
    ax.set_xlabel(r"Social benefit of detection $H$ (private penalty $L=40$)")
    ax.set_ylabel(r"Optimal detection threshold $\tau^\star$")
    ax.set_title(r"Private vs social thresholds (lower $\tau$ = more intense monitoring)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_welfare_wedge.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
def main() -> None:
    fig_mahalanobis()
    fig_amp_surface()
    fig_misspecification()
    fig_deterrence()
    fig_welfare_wedge()
    print(f"Wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
