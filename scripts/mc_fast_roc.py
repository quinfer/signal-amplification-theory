import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import numpy as np


def _fast_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUC via rank formula without sklearn.

    Assumes y_true in {0,1}. Handles ties approximately (average rank not computed),
    which is fine when noise is continuous; ties should be rare.
    """
    y_true = y_true.astype(np.int8)
    n = y_true.size
    n_pos = int(y_true.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    # Ranks from 1..n (stable mergesort to be conservative with ties)
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, n + 1)
    sum_ranks_pos = ranks[y_true == 1].sum(dtype=np.float64)
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, fpr_target: float) -> float:
    """Compute TPR at a fixed FPR by thresholding on score.

    Sorts scores descending and walks the cumulative counts to the target FPR.
    """
    y_true = y_true.astype(np.int8)
    n_pos = int(y_true.sum())
    n_neg = y_true.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0
    order = np.argsort(-y_score)  # descending
    y_sorted = y_true[order]
    # Cumulative positives/negatives as we lower threshold
    cum_pos = np.cumsum(y_sorted)
    cum_neg = np.cumsum(1 - y_sorted)
    fpr = cum_neg / max(1, n_neg)
    idx = np.searchsorted(fpr, fpr_target, side="left")
    if idx >= y_true.size:
        idx = y_true.size - 1
    tpr = cum_pos[idx] / max(1, n_pos)
    return float(tpr)


def _auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute Average Precision (area under PR curve) efficiently.

    Uses the identity AP = mean of precision at all ranks where a positive is found.
    Scores are sorted descending; ties handled stably via mergesort in argsort.
    """
    y_true = y_true.astype(np.int8)
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return 0.0
    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    cum_tp = np.cumsum(y_sorted, dtype=np.int64)
    ranks = np.arange(1, y_sorted.size + 1, dtype=np.int64)
    precision = cum_tp / ranks
    ap = precision[y_sorted == 1].sum(dtype=np.float64) / max(1, n_pos)
    return float(ap)


@dataclass
class MCConfig:
    n_pos: int = 500_000
    n_neg: int = 500_000
    mu_r: float = 1.5
    mu_c: float = 1.0
    sigma_r: float = 0.6
    sigma_c: float = 0.4
    rho: float = 0.3  # Corr(r,c) during manipulation
    bg_c_max: float = 0.2  # background cancellation upper bound (no manipulation)
    w1: float = 1.0
    w2: float = 0.8
    sigma_eps: float = 0.5
    fpr_target: float = 0.05
    bootstrap: int = 0
    seed: int = 42
    out: Optional[str] = None
    plot: Optional[str] = None


def simulate(config: MCConfig, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_pos, n_neg = config.n_pos, config.n_neg
    mu = np.array([config.mu_r, config.mu_c], dtype=np.float32)
    cov = np.array(
        [
            [config.sigma_r ** 2, config.rho * config.sigma_r * config.sigma_c],
            [config.rho * config.sigma_r * config.sigma_c, config.sigma_c ** 2],
        ],
        dtype=np.float64,
    )
    # Manipulation class: correlated r,c
    rc_pos = rng.multivariate_normal(mu, cov, size=n_pos).astype(np.float32)
    r_pos = np.clip(rc_pos[:, 0], 0.0, None)
    c_pos = np.clip(rc_pos[:, 1], 0.0, 1.0)
    # Normal class: mostly near zero rush, small cancellation background
    r_neg = np.zeros(n_neg, dtype=np.float32)
    c_neg = rng.uniform(0.0, config.bg_c_max, size=n_neg).astype(np.float32)
    # Stack
    r = np.concatenate([r_pos, r_neg])
    c = np.concatenate([c_pos, c_neg])
    y = np.concatenate([np.ones(n_pos, dtype=np.int8), np.zeros(n_neg, dtype=np.int8)])
    # Noise and scores
    eps = rng.normal(0.0, config.sigma_eps, size=r.size).astype(np.float32)
    s_r = r + eps
    s_c = c + eps
    s_comp = config.w1 * r + config.w2 * c + eps
    return y, s_r, s_c, s_comp, r


def bootstrap_auc(y: np.ndarray, scores: Tuple[np.ndarray, ...], B: int, rng: np.random.Generator) -> Tuple[np.ndarray, ...]:
    n = y.size
    aucs = [np.empty(B, dtype=np.float64) for _ in scores]
    for b in range(B):
        idx = rng.integers(0, n, size=n, dtype=np.int64)
        y_b = y[idx]
        for j, s in enumerate(scores):
            aucs[j][b] = _fast_auc(y_b, s[idx])
    return tuple(aucs)


def main():
    ap = argparse.ArgumentParser(description="Fast Monte Carlo ROC/AUC simulation for composite vs single features.")
    ap.add_argument("--n-pos", type=int, default=500_000)
    ap.add_argument("--n-neg", type=int, default=500_000)
    ap.add_argument("--mu-r", type=float, default=1.5)
    ap.add_argument("--mu-c", type=float, default=1.0)
    ap.add_argument("--sigma-r", type=float, default=0.6)
    ap.add_argument("--sigma-c", type=float, default=0.4)
    ap.add_argument("--rho", type=float, default=0.3)
    ap.add_argument("--bg-c-max", type=float, default=0.2)
    ap.add_argument("--w1", type=float, default=1.0)
    ap.add_argument("--w2", type=float, default=0.8)
    ap.add_argument("--sigma-eps", type=float, default=0.5)
    ap.add_argument("--fpr", type=float, default=0.05, help="TPR reported at this FPR")
    ap.add_argument("--bootstrap", type=int, default=0, help="Number of bootstrap replicates for AUC CIs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="results/mc_summary.json")
    ap.add_argument("--plot", type=str, default=None, help="Path to save ROC figure (optional)")
    args = ap.parse_args()

    cfg = MCConfig(
        n_pos=args.n_pos,
        n_neg=args.n_neg,
        mu_r=args.mu_r,
        mu_c=args.mu_c,
        sigma_r=args.sigma_r,
        sigma_c=args.sigma_c,
        rho=args.rho,
        bg_c_max=args.bg_c_max,
        w1=args.w1,
        w2=args.w2,
        sigma_eps=args.sigma_eps,
        fpr_target=args.fpr,
        bootstrap=args.bootstrap,
        seed=args.seed,
        out=args.out,
        plot=args.plot,
    )

    os.makedirs(os.path.dirname(cfg.out), exist_ok=True)
    rng = np.random.default_rng(cfg.seed)

    t0 = time.time()
    y, s_r, s_c, s_comp, r = simulate(cfg, rng)
    t_gen = time.time()

    auc_r = _fast_auc(y, s_r)
    auc_c = _fast_auc(y, s_c)
    auc_comp = _fast_auc(y, s_comp)
    auprc_r = _auprc(y, s_r)
    auprc_c = _auprc(y, s_c)
    auprc_comp = _auprc(y, s_comp)
    max_ind = max(auc_r, auc_c)
    amp = auc_comp - max_ind
    tpr_at_fpr = _tpr_at_fpr(y, s_comp, cfg.fpr_target)
    t_auc = time.time()

    out = {
        "config": asdict(cfg),
        "n_total": int(y.size),
        "auc_r": auc_r,
        "auc_c": auc_c,
        "auc_comp": auc_comp,
        "auprc_r": auprc_r,
        "auprc_c": auprc_c,
        "auprc_comp": auprc_comp,
        "amplification": amp,
        "tpr_at_fpr": {str(cfg.fpr_target): tpr_at_fpr},
        "timing_sec": {"generate": t_gen - t0, "metrics": t_auc - t_gen, "total": t_auc - t0},
    }

    if cfg.bootstrap and cfg.bootstrap > 0:
        b_rng = np.random.default_rng(cfg.seed + 1)
        auc_boot = bootstrap_auc(y, (s_r, s_c, s_comp), cfg.bootstrap, b_rng)
        def _ci(a: np.ndarray) -> Tuple[float, float]:
            return (float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975)))
        out["auc_ci95"] = {
            "r": _ci(auc_boot[0]),
            "c": _ci(auc_boot[1]),
            "comp": _ci(auc_boot[2]),
        }

    # Optional plot
    if cfg.plot:
        try:
            import matplotlib.pyplot as plt
        except Exception:
            plt = None
        if plt is not None:
            # Simple ROC for composite
            order = np.argsort(-s_comp)
            y_sorted = y[order]
            cum_pos = np.cumsum(y_sorted)
            cum_neg = np.cumsum(1 - y_sorted)
            n_pos = int(y.sum())
            n_neg = y.size - n_pos
            fpr = cum_neg / max(1, n_neg)
            tpr = cum_pos / max(1, n_pos)
            plt.figure(figsize=(5, 5))
            plt.plot(fpr, tpr, label=f"Composite (AUC={auc_comp:.3f})", lw=2)
            plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC (Composite)")
            plt.legend()
            plt.tight_layout()
            os.makedirs(os.path.dirname(cfg.plot), exist_ok=True)
            plt.savefig(cfg.plot, dpi=160)
            plt.close()

    with open(cfg.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # Console summary
    print("=== Monte Carlo Summary ===")
    print(f"Samples: pos={cfg.n_pos:,}, neg={cfg.n_neg:,} | total={cfg.n_pos + cfg.n_neg:,}")
    print(f"AUC r={auc_r:.4f} | c={auc_c:.4f} | comp={auc_comp:.4f} | amplification={amp:.4f}")
    print(f"PR AUC r={auprc_r:.4f} | c={auprc_c:.4f} | comp={auprc_comp:.4f}")
    print(f"TPR at FPR={cfg.fpr_target:.3f}: {tpr_at_fpr:.4f}")
    if cfg.bootstrap and cfg.bootstrap > 0:
        ci = out["auc_ci95"]
        print(f"95% CI AUC r={ci['r']} | c={ci['c']} | comp={ci['comp']}")
    print(f"Timing (s): generate={out['timing_sec']['generate']:.3f}, metrics={out['timing_sec']['metrics']:.3f}, total={out['timing_sec']['total']:.3f}")


if __name__ == "__main__":
    main()
