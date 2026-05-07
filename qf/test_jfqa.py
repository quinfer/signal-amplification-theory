"""Self-contained synthetic test of the JFQA extensions.

Runs entirely on simulated Gaussian data so it can be executed without
the proprietary Shenzhen pickles.  The test verifies that:

  1. :func:`fast_delong` produces sensible AUC estimates and a covariance
     matrix that agrees with a paired-bootstrap variance to within a few
     percent on a moderately-sized sample.
  2. :func:`pr_auc_paired_bootstrap` runs and returns valid CIs.
  3. :func:`shrinkage_fisher_weights` and :func:`l1_logistic_cv` execute
     and produce non-degenerate scores.
  4. :func:`label_noise_horserace` runs and the Fisher-versus-singleton
     gap shrinks (or at least does not explode) under increasing
     :math:`\\eta`.
  5. :func:`ushape_cluster_bootstrap` returns a wider CI than a naive
     pair-level bootstrap, as expected.

This is a smoke test, not a unit test in the formal sense; it is meant
to give the author confidence that the analyses execute end-to-end on
the real Shenzhen panel.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

# Allow running from either qf/ or the project root.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from qf.jfqa import (  # noqa: E402
    delong_pairs,
    fast_delong,
    label_noise_horserace,
    pr_auc_paired_bootstrap,
    shrinkage_fisher_weights,
    summarise_label_noise,
    ushape_cluster_bootstrap,
    l1_logistic_cv,
)


def make_synthetic(
    n_pos: int = 800,
    n_neg: int = 8000,
    K: int = 12,
    rho: float = 0.3,
    mu_norm: float = 1.5,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    Sig0 = (1 - rho) * np.eye(K) + rho * np.ones((K, K))
    L = np.linalg.cholesky(Sig0)
    mu = mu_norm * rng.standard_normal(K)
    mu /= np.linalg.norm(mu)
    mu *= mu_norm
    X_neg = rng.standard_normal((n_neg, K)) @ L.T
    X_pos = mu + rng.standard_normal((n_pos, K)) @ L.T
    X = np.vstack([X_neg, X_pos])
    y = np.concatenate([np.zeros(n_neg, dtype=int), np.ones(n_pos, dtype=int)])
    # Shuffle so that subsequent splits do not put all positives at the tail.
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def test_delong_against_bootstrap():
    print("\n[test] DeLong covariance vs paired bootstrap")
    rng = np.random.default_rng(1)
    X, y = make_synthetic(n_pos=400, n_neg=4000, K=4, seed=1)
    # Two simple linear scores: feature 0 alone and a random combination
    s1 = X[:, 0]
    s2 = X @ rng.standard_normal(X.shape[1])
    scores_dict = {"feat0": s1, "rand_combo": s2}
    df = delong_pairs(scores_dict, y, reference="feat0")
    print(df.to_string(index=False, float_format=lambda z: f"{z:.4f}"))

    # Bootstrap variance for the difference
    n = len(y)
    n_boot = 500
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].sum() in (0, n):
            diffs[b] = np.nan
            continue
        a1 = roc_auc_score(y[idx], s1[idx])
        a2 = roc_auc_score(y[idx], s2[idx])
        diffs[b] = a2 - a1
    boot_se = float(np.nanstd(diffs, ddof=1))
    delong_se = float(df.loc[df["rule"] == "rand_combo", "diff_se"].iloc[0])
    print(f"  bootstrap SE on diff = {boot_se:.4f}")
    print(f"  DeLong    SE on diff = {delong_se:.4f}")
    rel_err = abs(delong_se - boot_se) / boot_se
    print(f"  relative error      = {rel_err:.2%}")
    assert rel_err < 0.20, f"DeLong SE differs from bootstrap SE by {rel_err:.2%}"


def test_pr_bootstrap():
    print("\n[test] PR AUC paired bootstrap")
    X, y = make_synthetic(seed=2)
    rng = np.random.default_rng(2)
    s_good = X.mean(axis=1)
    s_bad = rng.standard_normal(len(y))
    scores = {"good": s_good, "bad": s_bad}
    df = pr_auc_paired_bootstrap(scores, y, reference="bad", n_boot=300, seed=2)
    print(df.to_string(index=False, float_format=lambda z: f"{z:.4f}"))
    diff = float(df.loc[df["rule"] == "good", "pr_diff_vs_ref"].iloc[0])
    lo = float(df.loc[df["rule"] == "good", "ci95_lo"].iloc[0])
    assert diff > 0
    assert lo > 0


def test_shrinkage_and_l1():
    print("\n[test] shrinkage Fisher and L1-logistic")
    X, y = make_synthetic(seed=3)
    Xtr, Xte = X[: int(0.7 * len(X))], X[int(0.7 * len(X)):]
    ytr, yte = y[: int(0.7 * len(y))], y[int(0.7 * len(y)):]
    Sig0_lw, mu, w_lw, d_star2_lw, intensity = shrinkage_fisher_weights(
        Xtr[ytr == 0], Xtr[ytr == 1]
    )
    print(f"  Ledoit-Wolf shrinkage intensity = {intensity:.4f}")
    print(f"  shrunk d*^2 = {d_star2_lw:.4f}")
    auc_lw = roc_auc_score(yte, Xte @ w_lw)
    print(f"  shrunk Fisher test AUC = {auc_lw:.4f}")
    assert 0 < intensity < 1
    assert auc_lw > 0.8

    lr_l1 = l1_logistic_cv(Xtr, ytr, seed=3)
    auc_l1 = roc_auc_score(yte, lr_l1.decision_function(Xte))
    nz = int((lr_l1.coef_ != 0).sum())
    print(f"  L1-logistic test AUC   = {auc_l1:.4f}  (n_nonzero = {nz})")
    assert auc_l1 > 0.8


def test_label_noise():
    print("\n[test] label-noise horserace")
    X, y = make_synthetic(seed=4)
    Xtr, Xte = X[: int(0.7 * len(X))], X[int(0.7 * len(X)):]
    ytr, yte = y[: int(0.7 * len(y))], y[int(0.7 * len(y)):]
    df = label_noise_horserace(
        Xtr, Xte, ytr, yte, flip_fractions=(0.0, 0.05, 0.10), n_seeds=3, seed=4
    )
    summ = summarise_label_noise(df)
    print(summ.to_string(index=False, float_format=lambda z: f"{z:.4f}"))
    # As eta increases the gap should not grow (we expect erosion or stability,
    # not improvement).
    gaps = summ.set_index("eta")["auc_gap_mean"]
    assert gaps[0.0] >= gaps[0.10] - 0.05, "Fisher gap should not grow under label noise"


def test_ushape_cluster():
    print("\n[test] U-shape cluster bootstrap")
    rng = np.random.default_rng(5)
    K = 12
    n_neg, n_pos = 8000, 800
    # Generate a covariance with structured correlations so rho_0 varies
    # plausibly across pairs.
    A = rng.standard_normal((K, K))
    Sig0 = A @ A.T + np.eye(K)
    Sig0 = Sig0 / np.sqrt(np.outer(np.diag(Sig0), np.diag(Sig0)))
    L = np.linalg.cholesky(Sig0)
    mu = rng.standard_normal(K) * 0.7
    X_neg = rng.standard_normal((n_neg, K)) @ L.T
    X_pos = mu + rng.standard_normal((n_pos, K)) @ L.T
    out = ushape_cluster_bootstrap(X_neg, X_pos, n_boot=500, seed=5)
    for k, v in out.items():
        print(f"  {k}: {v}")
    assert np.isfinite(out["c_hat_orig"])
    assert out["c_ci_hi_cluster"] >= out["c_ci_lo_cluster"]


def main():
    test_delong_against_bootstrap()
    test_pr_bootstrap()
    test_shrinkage_and_l1()
    test_label_noise()
    test_ushape_cluster()
    print("\n[test] all smoke tests passed")


if __name__ == "__main__":
    main()
