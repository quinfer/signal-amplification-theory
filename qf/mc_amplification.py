"""Monte Carlo verification of the Fisher-Mahalanobis Signal Amplification Theorem.

This module implements the corrected Signal Amplification Theorem for market
manipulation detection. It distinguishes H0-covariance (benign order-flow
co-movement) from H1-covariance (manipulation complementarity) and compares:

    * Fisher-optimal composite detector   w* = Sigma_0^{-1} mu
    * Fixed-weight (suboptimal) composite  w = (w1, w2)
    * Singleton detectors                 (1,0) and (0,1)

on both the analytical Mahalanobis deflection d^2 = mu' Sigma_0^{-1} mu and
the empirical ROC AUC.

All simulations are vectorised NumPy and run comfortably on Apple Silicon.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Fast ROC / AUC utilities (shared with legacy scripts but self-contained)
# ---------------------------------------------------------------------------
def fast_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int8)
    n = y_true.size
    n_pos = int(y_true.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, n + 1)
    sum_ranks_pos = ranks[y_true == 1].sum(dtype=np.float64)
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, fpr_target: float) -> float:
    y_true = y_true.astype(np.int8)
    n_pos = int(y_true.sum())
    n_neg = y_true.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    cum_pos = np.cumsum(y_sorted)
    cum_neg = np.cumsum(1 - y_sorted)
    fpr = cum_neg / max(1, n_neg)
    idx = int(np.searchsorted(fpr, fpr_target, side="left"))
    idx = min(idx, y_true.size - 1)
    return float(cum_pos[idx] / max(1, n_pos))


# ---------------------------------------------------------------------------
# Analytical quantities
# ---------------------------------------------------------------------------
def mahalanobis_deflection(mu: np.ndarray, sigma0: np.ndarray) -> float:
    """d^2 = mu' Sigma0^{-1} mu (Fisher-optimal deflection)."""
    return float(mu @ np.linalg.solve(sigma0, mu))


def singleton_deflection(mu: np.ndarray, sigma0: np.ndarray) -> float:
    """Best-singleton SNR: max_i mu_i^2 / Sigma0_{ii}."""
    diag = np.diag(sigma0)
    return float(np.max(mu ** 2 / diag))


def fisher_weights(mu: np.ndarray, sigma0: np.ndarray) -> np.ndarray:
    """Fisher-optimal linear detector weights w* = Sigma0^{-1} mu (unnormalised)."""
    return np.linalg.solve(sigma0, mu)


def deflection_under_weights(w: np.ndarray, mu: np.ndarray, sigma0: np.ndarray) -> float:
    num = float(w @ mu) ** 2
    den = float(w @ sigma0 @ w)
    return num / den if den > 0 else 0.0


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
@dataclass
class SimConfig:
    n_pos: int = 500_000
    n_neg: int = 500_000
    mu_r: float = 1.5
    mu_c: float = 1.0
    sigma_r0: float = 0.6     # H0 standard deviation (rush-order noise)
    sigma_c0: float = 0.4     # H0 standard deviation (cancellation noise)
    sigma_r1: float = 0.6     # H1 standard deviation (rush order during manipulation)
    sigma_c1: float = 0.4     # H1 standard deviation (cancellation during manipulation)
    rho0: float = 0.0          # H0 correlation (benign co-movement)
    rho1: float = 0.3          # H1 correlation (manipulation complementarity)
    sigma_eps: float = 0.0     # additional measurement noise on both features
    fixed_w: Tuple[float, float] = (1.0, 0.8)
    fpr_target: float = 0.05
    seed: int = 42


def _cov(sigma_r: float, sigma_c: float, rho: float) -> np.ndarray:
    return np.array(
        [[sigma_r ** 2, rho * sigma_r * sigma_c],
         [rho * sigma_r * sigma_c, sigma_c ** 2]],
        dtype=np.float64,
    )


def simulate(cfg: SimConfig, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    mu0 = np.array([0.0, 0.0])
    mu1 = np.array([cfg.mu_r, cfg.mu_c])
    sigma0 = _cov(cfg.sigma_r0, cfg.sigma_c0, cfg.rho0)
    sigma1 = _cov(cfg.sigma_r1, cfg.sigma_c1, cfg.rho1)

    # Add independent measurement noise to both features if requested.
    if cfg.sigma_eps > 0:
        sigma0 = sigma0 + cfg.sigma_eps ** 2 * np.eye(2)
        sigma1 = sigma1 + cfg.sigma_eps ** 2 * np.eye(2)

    x_neg = rng.multivariate_normal(mu0, sigma0, size=cfg.n_neg)
    x_pos = rng.multivariate_normal(mu1, sigma1, size=cfg.n_pos)

    x = np.concatenate([x_pos, x_neg], axis=0).astype(np.float64)
    y = np.concatenate(
        [np.ones(cfg.n_pos, dtype=np.int8), np.zeros(cfg.n_neg, dtype=np.int8)]
    )
    return {
        "x": x,
        "y": y,
        "mu1": mu1,
        "sigma0": sigma0,
        "sigma1": sigma1,
    }


# ---------------------------------------------------------------------------
# One-shot experiment
# ---------------------------------------------------------------------------
def run_experiment(cfg: SimConfig) -> Dict:
    rng = np.random.default_rng(cfg.seed)
    data = simulate(cfg, rng)
    x, y = data["x"], data["y"]
    mu1, sigma0 = data["mu1"], data["sigma0"]

    # Analytical
    d2_fisher = mahalanobis_deflection(mu1, sigma0)
    d2_best_single = singleton_deflection(mu1, sigma0)
    amp_analytical = d2_fisher - d2_best_single

    # Weights
    w_opt = fisher_weights(mu1, sigma0)
    w_fixed = np.array(cfg.fixed_w, dtype=np.float64)

    # Scores
    s_r = x[:, 0]
    s_c = x[:, 1]
    s_opt = x @ w_opt
    s_fix = x @ w_fixed

    # Empirical
    auc_r = fast_auc(y, s_r)
    auc_c = fast_auc(y, s_c)
    auc_opt = fast_auc(y, s_opt)
    auc_fix = fast_auc(y, s_fix)
    max_single = max(auc_r, auc_c)
    amp_opt = auc_opt - max_single
    amp_fix = auc_fix - max_single

    return {
        "config": asdict(cfg),
        # Analytical
        "d2_fisher": d2_fisher,
        "d2_best_single": d2_best_single,
        "amplification_analytical": amp_analytical,
        "w_opt": w_opt.tolist(),
        # Empirical
        "auc_r": auc_r,
        "auc_c": auc_c,
        "auc_opt": auc_opt,
        "auc_fix": auc_fix,
        "amp_opt": amp_opt,
        "amp_fix": amp_fix,
        "tpr_at_fpr_opt": tpr_at_fpr(y, s_opt, cfg.fpr_target),
        "tpr_at_fpr_fix": tpr_at_fpr(y, s_fix, cfg.fpr_target),
    }


# ---------------------------------------------------------------------------
# Grid runner
# ---------------------------------------------------------------------------
def run_grid(
    rho0_grid: Sequence[float],
    rho1_grid: Sequence[float],
    sigma_eps_grid: Sequence[float],
    n_pos: int = 200_000,
    n_neg: int = 200_000,
    seeds: Sequence[int] = (42,),
    mu_r: float = 1.5,
    mu_c: float = 1.0,
    sigma_r0: float = 0.6,
    sigma_c0: float = 0.4,
    sigma_r1: float = 0.6,
    sigma_c1: float = 0.4,
    fixed_w: Tuple[float, float] = (1.0, 0.8),
    fpr_target: float = 0.05,
) -> List[Dict]:
    rows = []
    for rho0 in rho0_grid:
        for rho1 in rho1_grid:
            for se in sigma_eps_grid:
                for seed in seeds:
                    cfg = SimConfig(
                        n_pos=n_pos,
                        n_neg=n_neg,
                        mu_r=mu_r,
                        mu_c=mu_c,
                        sigma_r0=sigma_r0,
                        sigma_c0=sigma_c0,
                        sigma_r1=sigma_r1,
                        sigma_c1=sigma_c1,
                        rho0=rho0,
                        rho1=rho1,
                        sigma_eps=se,
                        fixed_w=fixed_w,
                        fpr_target=fpr_target,
                        seed=seed,
                    )
                    res = run_experiment(cfg)
                    rows.append(res)
    return rows


def _flatten_row(res: Dict) -> Dict:
    cfg = res["config"]
    return {
        "rho0": cfg["rho0"],
        "rho1": cfg["rho1"],
        "sigma_eps": cfg["sigma_eps"],
        "n_pos": cfg["n_pos"],
        "n_neg": cfg["n_neg"],
        "seed": cfg["seed"],
        "d2_fisher": res["d2_fisher"],
        "d2_best_single": res["d2_best_single"],
        "amp_analytical": res["amplification_analytical"],
        "auc_r": res["auc_r"],
        "auc_c": res["auc_c"],
        "auc_opt": res["auc_opt"],
        "auc_fix": res["auc_fix"],
        "amp_opt": res["amp_opt"],
        "amp_fix": res["amp_fix"],
        "tpr_fpr_opt": res["tpr_at_fpr_opt"],
        "tpr_fpr_fix": res["tpr_at_fpr_fix"],
    }


def write_grid_csv(rows: List[Dict], path: str) -> None:
    import csv
    flat = [_flatten_row(r) for r in rows]
    fieldnames = list(flat[0].keys())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in flat:
            w.writerow(r)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["single", "grid"], default="grid")
    ap.add_argument("--out-json", default="qf/results/mc_qf_single.json")
    ap.add_argument("--out-csv", default="qf/results/mc_qf_grid.csv")
    ap.add_argument("--n-pos", type=int, default=200_000)
    ap.add_argument("--n-neg", type=int, default=200_000)
    ap.add_argument("--mu-r", type=float, default=1.5)
    ap.add_argument("--mu-c", type=float, default=1.0)
    ap.add_argument("--sigma-r0", type=float, default=0.6)
    ap.add_argument("--sigma-c0", type=float, default=0.4)
    ap.add_argument("--sigma-r1", type=float, default=0.6)
    ap.add_argument("--sigma-c1", type=float, default=0.4)
    ap.add_argument("--rho0", type=float, default=0.0)
    ap.add_argument("--rho1", type=float, default=0.3)
    ap.add_argument("--sigma-eps", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--rho0-grid", type=float, nargs="+",
        default=[-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5, 0.7],
    )
    ap.add_argument(
        "--rho1-grid", type=float, nargs="+",
        default=[-0.2, 0.0, 0.2, 0.4, 0.6],
    )
    ap.add_argument(
        "--sigma-eps-grid", type=float, nargs="+",
        default=[0.0, 0.25, 0.5],
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024])
    args = ap.parse_args()

    t0 = time.time()
    if args.mode == "single":
        cfg = SimConfig(
            n_pos=args.n_pos, n_neg=args.n_neg,
            mu_r=args.mu_r, mu_c=args.mu_c,
            sigma_r0=args.sigma_r0, sigma_c0=args.sigma_c0,
            sigma_r1=args.sigma_r1, sigma_c1=args.sigma_c1,
            rho0=args.rho0, rho1=args.rho1,
            sigma_eps=args.sigma_eps, seed=args.seed,
        )
        res = run_experiment(cfg)
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(json.dumps(res, indent=2))
    else:
        rows = run_grid(
            rho0_grid=args.rho0_grid,
            rho1_grid=args.rho1_grid,
            sigma_eps_grid=args.sigma_eps_grid,
            n_pos=args.n_pos, n_neg=args.n_neg,
            seeds=args.seeds,
            mu_r=args.mu_r, mu_c=args.mu_c,
            sigma_r0=args.sigma_r0, sigma_c0=args.sigma_c0,
            sigma_r1=args.sigma_r1, sigma_c1=args.sigma_c1,
        )
        write_grid_csv(rows, args.out_csv)
        print(f"Wrote {len(rows)} grid rows to {args.out_csv}")
    print(f"Elapsed {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
