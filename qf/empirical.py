"""Empirical validation of the Fisher-Mahalanobis amplification theorem
on Chinese Level-2 order-book data (Dai et al., SD-FMM dataset).

This script produces the figures and tables for the empirical section of
the QF manuscript:

  1. Out-of-sample horserace: singletons, fixed-weight, Fisher, logistic,
     random forest, XGBoost.  ROC-AUC, PR-AUC, TPR@5%-FPR.
  2. Misspecification-bound validation: empirical d^2(wh)/d*^2 vs. the
     theoretical cos^2(theta) in the Sigma_0 inner product.
  3. Cross-sectional rho_0 heterogeneity: for every feature pair, compute
     benign-flow correlation and empirical two-feature amplification;
     compare with the closed form of Corollary 1.
  4. Feature importance: analytical d^2(e_k) ranks tied to microstructure
     interpretation.

Writes CSV tables to qf/results/ and PDFs to qf/figures/.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError, pinv, solve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


HERE = Path(__file__).resolve().parent
DATA_ROOT = Path(
    "/Users/quinference/Library/CloudStorage/Dropbox/Documents/Research/"
    "market_manipulation_anomaly_detection/market_manipulation_detection"
)
X_PKL = DATA_ROOT / "data" / "x_data.pkl"
Y_PKL = DATA_ROOT / "data" / "y_dex.pkl"

RESULTS = HERE / "results"
FIGURES = HERE / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

FEATURE_NAMES = [
    "rel_spread_mean", "rel_spread_std",
    "imbalance_mean", "imbalance_std",
    "bid_depth_mean", "ask_depth_mean",
    "log_total_volume", "log_max_step_volume",
    "realised_vol_bps", "up_move_freq",
    "abs_return_mean_bps", "range_bps",
]


def window_features(win: np.ndarray) -> np.ndarray:
    """Reduce a (T, 26) LOB window to a 12-dim microstructure vector."""
    bid1 = win[:, 6]
    ask1 = win[:, 16]
    valid = (bid1 > 0) & (ask1 > 0)
    if valid.sum() < 5:
        return np.full(12, np.nan)

    w = win[valid]
    mid = 0.5 * (w[:, 6] + w[:, 16])
    spread = w[:, 16] - w[:, 6]
    rel_spread = spread / np.clip(mid, 1e-6, None) * 1e4
    bid_depth = w[:, 11:16].sum(axis=1)
    ask_depth = w[:, 21:26].sum(axis=1)
    total_depth = bid_depth + ask_depth
    imbalance = (bid_depth - ask_depth) / np.clip(total_depth, 1.0, None)
    ret = np.diff(np.log(np.clip(mid, 1e-6, None)))
    cum_vol = w[:, 3]
    vol_step = np.clip(np.diff(cum_vol), 0, None)
    hi = w[:, 0]
    lo = w[:, 1]
    hi_nz = hi[hi > 0]
    lo_nz = lo[lo > 0]
    range_bps = (hi_nz.max() - lo_nz.min()) if (hi_nz.size and lo_nz.size) else 0.0

    return np.array([
        np.mean(rel_spread),
        np.std(rel_spread),
        np.mean(imbalance),
        np.std(imbalance),
        np.mean(bid_depth),
        np.mean(ask_depth),
        np.log1p(vol_step.sum()),
        np.log1p(vol_step.max()),
        np.std(ret) * 1e4 if ret.size else 0.0,
        (ret > 0).mean() if ret.size else 0.0,
        np.mean(np.abs(ret)) * 1e4 if ret.size else 0.0,
        range_bps / np.clip(mid.mean(), 1e-6, None) * 1e4,
    ])


def extract_features(x_list, y):
    X = np.empty((len(x_list), 12), dtype=np.float64)
    t0 = time.time()
    for i, w in enumerate(x_list):
        X[i] = window_features(w)
        if i and i % 10000 == 0:
            print(f"  {i}/{len(x_list)} ({time.time() - t0:.1f}s)")
    mask = np.isfinite(X).all(axis=1)
    return X[mask], y[mask]


# ---------- metrics ----------
def tpr_at_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float = 0.05) -> float:
    neg = scores[~y_true]
    thr = np.quantile(neg, 1 - target_fpr)
    return float((scores[y_true] >= thr).mean())


def fisher_weights(X_neg: np.ndarray, X_pos: np.ndarray):
    Sig0 = np.cov(X_neg.T)
    mu = X_pos.mean(axis=0) - X_neg.mean(axis=0)
    try:
        w = solve(Sig0, mu)
    except LinAlgError:
        w = pinv(Sig0) @ mu
    d_star2 = float(mu @ w)
    return Sig0, mu, w, d_star2


def deflection(w: np.ndarray, mu: np.ndarray, Sig0: np.ndarray) -> float:
    num = float((w @ mu) ** 2)
    den = float(w @ Sig0 @ w)
    return num / max(den, 1e-12)


def sigma0_cos2(w_hat: np.ndarray, w_star: np.ndarray, Sig0: np.ndarray) -> float:
    """cos^2 of the angle between w_hat and w_star in the Sigma_0 inner product."""
    num = float(w_hat @ Sig0 @ w_star) ** 2
    den = float((w_hat @ Sig0 @ w_hat) * (w_star @ Sig0 @ w_star))
    return num / max(den, 1e-12)


def summarise(name: str, y: np.ndarray, s: np.ndarray) -> dict:
    return {
        "rule": name,
        "auc": roc_auc_score(y, s),
        "pr_auc": average_precision_score(y, s),
        "tpr@5fpr": tpr_at_fpr(y, s, 0.05),
    }


# ---------- main pipeline ----------
def run_horserace(Xtr: np.ndarray, Xte: np.ndarray, ytr: np.ndarray, yte: np.ndarray,
                  mu: np.ndarray, w_star: np.ndarray):
    results = []

    # 1. Singletons, with sign aligned to training mu
    for k, name in enumerate(FEATURE_NAMES):
        s = Xte[:, k] * np.sign(mu[k])
        results.append({"family": "singleton", **summarise(name, yte, s)})

    # 2. Fixed-weight composites
    w_eq = np.sign(mu)
    results.append({"family": "fixed", **summarise("equal (sign(mu))", yte, Xte @ w_eq)})
    # L2-normalised random-ish (1,0.8,...) style fixed rule flagged in Prop. 3 discussion:
    w_ptw = np.ones(Xte.shape[1]) * 0.8
    w_ptw[mu.argmax()] = 1.0
    w_ptw *= np.sign(mu)
    results.append({"family": "fixed", **summarise("expert (1,0.8,...)", yte, Xte @ w_ptw)})

    # 3. Fisher-optimal
    results.append({"family": "fisher", **summarise("Fisher w*", yte, Xte @ w_star)})

    # 4. Logistic regression (class-balanced)
    lr = LogisticRegression(max_iter=5000, class_weight="balanced").fit(Xtr, ytr)
    results.append({"family": "ml", **summarise("logistic", yte, lr.decision_function(Xte))})

    # 5. Random forest
    print("[ml] training random forest...")
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=20, n_jobs=-1,
        class_weight="balanced_subsample", random_state=42,
    ).fit(Xtr, ytr)
    results.append({"family": "ml", **summarise("random forest", yte, rf.predict_proba(Xte)[:, 1])})

    # 6. XGBoost
    print("[ml] training xgboost...")
    scale = (ytr == 0).sum() / max(1, (ytr == 1).sum())
    xgb = XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale,
        eval_metric="aucpr", n_jobs=-1, random_state=42, tree_method="hist",
    ).fit(Xtr, ytr)
    results.append({"family": "ml", **summarise("xgboost", yte, xgb.predict_proba(Xte)[:, 1])})

    df = pd.DataFrame(results)
    df.to_csv(RESULTS / "horserace.csv", index=False)
    return df, {"lr": lr, "rf": rf, "xgb": xgb, "w_star": w_star, "w_eq": w_eq, "w_ptw": w_ptw}


def misspec_check(X_neg_tr: np.ndarray, mu: np.ndarray, Sig0: np.ndarray,
                  w_star: np.ndarray, X_pos_tr: np.ndarray, X_neg_te: np.ndarray,
                  X_pos_te: np.ndarray, yte: np.ndarray, Xte: np.ndarray, rng):
    """Sample random weight vectors; for each, compute cos^2 in the training
    Sigma_0-metric and the empirical d^2(wh)/d*^2 on held-out data.

    Proposition 3 asserts d^2(wh)/d*^2 <= cos^2 theta. We therefore report:

      - the fraction of draws that *violate* the bound in the held-out sample
        (should be small and driven entirely by finite-sample covariance
        estimation error),
      - the conditional mean and standard deviation of the empirical ratio
        within quartile bins of cos^2 theta,
      - the slope of the best-fit y = beta * x through the origin (consistent
        with a proportional-tightness interpretation of the bound).
    """
    K = Sig0.shape[0]
    m = 1000
    cos2_vals = np.empty(m)
    ratio_empirical = np.empty(m)

    mu_te = X_pos_te.mean(axis=0) - X_neg_te.mean(axis=0)
    Sig0_te = np.cov(X_neg_te.T)
    d_star2_te = float(mu_te @ (pinv(Sig0_te) @ mu_te))

    for i in range(m):
        wh = rng.standard_normal(K)
        cos2_vals[i] = sigma0_cos2(wh, w_star, Sig0)
        ratio_empirical[i] = deflection(wh, mu_te, Sig0_te) / max(d_star2_te, 1e-12)

    out = pd.DataFrame({
        "cos2_sigma0": cos2_vals,
        "emp_ratio_d2": ratio_empirical,
    })
    out.to_csv(RESULTS / "misspec.csv", index=False)
    return out


def misspec_diagnostics(df_miss: pd.DataFrame):
    """Violation rate, conditional means within cos^2 quartiles."""
    x = df_miss["cos2_sigma0"].values
    y = df_miss["emp_ratio_d2"].values

    # violations = ratio > cos^2 (should be ~0 if the bound is tight)
    violations = (y > x).mean()

    # conditional mean of ratio in cos^2 quartile bins
    qs = np.quantile(x, [0, 0.25, 0.5, 0.75, 1.0])
    bin_id = np.clip(np.digitize(x, qs[1:-1]), 0, 3)
    rows = []
    for b in range(4):
        msk = bin_id == b
        rows.append({
            "cos2_bin": f"Q{b+1}",
            "cos2_lower": qs[b],
            "cos2_upper": qs[b + 1],
            "cos2_mean": x[msk].mean(),
            "emp_ratio_mean": y[msk].mean(),
            "emp_ratio_std": y[msk].std(),
            "n": int(msk.sum()),
        })
    bins_df = pd.DataFrame(rows)
    bins_df.to_csv(RESULTS / "misspec_bins.csv", index=False)

    slope = float(np.dot(x, y) / np.dot(x, x))
    r2 = float(1 - np.var(y - slope * x) / np.var(y))

    return {
        "violation_rate": float(violations),
        "slope_through_origin": slope,
        "r2": r2,
    }, bins_df


def rho0_cross_section(X_neg_tr: np.ndarray, X_pos_tr: np.ndarray):
    """For every feature pair (i,j), compute benign-flow correlation rho_0,
    analytical d^{*2}(i,j) and the best-singleton d^2 among {i,j}; compare
    with Corollary 1 prediction.
    """
    K = X_neg_tr.shape[1]
    mu = X_pos_tr.mean(axis=0) - X_neg_tr.mean(axis=0)
    rows = []
    for i in range(K):
        for j in range(i + 1, K):
            Sig_ij = np.cov(X_neg_tr[:, [i, j]].T)
            mu_ij = mu[[i, j]]
            try:
                w = solve(Sig_ij, mu_ij)
            except LinAlgError:
                w = pinv(Sig_ij) @ mu_ij
            d_star2 = float(mu_ij @ w)
            d2_i = mu_ij[0] ** 2 / Sig_ij[0, 0]
            d2_j = mu_ij[1] ** 2 / Sig_ij[1, 1]
            d2_best = max(d2_i, d2_j)
            rho0 = Sig_ij[0, 1] / np.sqrt(Sig_ij[0, 0] * Sig_ij[1, 1])
            rows.append({
                "i": i, "j": j,
                "feat_i": FEATURE_NAMES[i], "feat_j": FEATURE_NAMES[j],
                "rho0": rho0,
                "d_star2": d_star2,
                "d2_best": d2_best,
                "amplification": d_star2 / max(d2_best, 1e-12),
            })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "rho0_cross_section.csv", index=False)
    return df


def ushape_test(df_rho: pd.DataFrame, rng, n_boot: int = 2000):
    """Test the U-shape prediction of Corollary 1 on the 66 feature pairs.

    We fit a quadratic a + b*rho + c*rho^2 with c > 0 implying U-shape.
    Bootstrap over feature pairs to get a confidence interval on c and on
    the argmin of the quadratic (the interior minimum). We also run a
    non-parametric monotonicity test (Spearman on rho in each half).
    """
    from scipy.stats import spearmanr
    rho = df_rho["rho0"].values
    amp = df_rho["amplification"].values

    # Fit quadratic
    X = np.column_stack([np.ones_like(rho), rho, rho ** 2])
    beta = np.linalg.lstsq(X, amp, rcond=None)[0]

    # Bootstrap
    n = len(rho)
    betas = np.empty((n_boot, 3))
    rho_stars = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        Xb = np.column_stack([np.ones(n), rho[idx], rho[idx] ** 2])
        bb = np.linalg.lstsq(Xb, amp[idx], rcond=None)[0]
        betas[b] = bb
        rho_stars[b] = -bb[1] / (2 * bb[2]) if bb[2] > 1e-8 else np.nan

    c_ci = np.percentile(betas[:, 2], [2.5, 97.5])
    c_hat = float(beta[2])
    p_c_positive = float((betas[:, 2] > 0).mean())
    rho_star_hat = float(-beta[1] / (2 * beta[2])) if beta[2] > 1e-8 else np.nan
    rho_star_ci = np.percentile(rho_stars[np.isfinite(rho_stars)], [2.5, 97.5])

    # Non-parametric monotonicity tests in each half
    # (left half of rho axis should have negative Spearman, right half positive)
    left = rho < rho_star_hat
    right = ~left
    rho_left_sp, rho_left_p = spearmanr(rho[left], amp[left]) if left.sum() > 3 else (np.nan, np.nan)
    rho_right_sp, rho_right_p = spearmanr(rho[right], amp[right]) if right.sum() > 3 else (np.nan, np.nan)

    stats = {
        "n_pairs": int(n),
        "c_hat": c_hat,
        "c_ci_lo": float(c_ci[0]),
        "c_ci_hi": float(c_ci[1]),
        "p_c_positive": p_c_positive,
        "rho_star_hat": rho_star_hat,
        "rho_star_ci_lo": float(rho_star_ci[0]) if np.isfinite(rho_star_ci[0]) else np.nan,
        "rho_star_ci_hi": float(rho_star_ci[1]) if np.isfinite(rho_star_ci[1]) else np.nan,
        "spearman_left": float(rho_left_sp) if np.isfinite(rho_left_sp) else np.nan,
        "spearman_left_p": float(rho_left_p) if np.isfinite(rho_left_p) else np.nan,
        "spearman_right": float(rho_right_sp) if np.isfinite(rho_right_sp) else np.nan,
        "spearman_right_p": float(rho_right_p) if np.isfinite(rho_right_p) else np.nan,
    }
    pd.Series(stats).to_csv(RESULTS / "ushape_test.csv")
    return stats, beta, betas


def ml_vs_fisher_analysis(Xtr: np.ndarray, Xte: np.ndarray,
                           ytr: np.ndarray, yte: np.ndarray,
                           xgb, w_star: np.ndarray, rng):
    """Decompose the XGBoost-over-Fisher gap.

    (a) Permutation importance: how much does shuffling each feature hurt
        XGBoost's test AUC?
    (b) Residual non-linearity: train XGBoost on the Fisher score only
        (one-dim) vs. the full feature set; the AUC gap is the non-linear
        interaction residual unreachable by linear rules.
    (c) Per-quartile stratification by Fisher score: where in the Fisher
        distribution does XGBoost add the most value?
    """
    from sklearn.metrics import roc_auc_score

    # (a) permutation importance
    base_auc = roc_auc_score(yte, xgb.predict_proba(Xte)[:, 1])
    imp_rows = []
    Xp = Xte.copy()
    for k, name in enumerate(FEATURE_NAMES):
        col = Xp[:, k].copy()
        Xp[:, k] = rng.permutation(Xp[:, k])
        a = roc_auc_score(yte, xgb.predict_proba(Xp)[:, 1])
        Xp[:, k] = col
        imp_rows.append({"feature": name, "auc_drop": base_auc - a})
    imp = pd.DataFrame(imp_rows).sort_values("auc_drop", ascending=False)
    imp.to_csv(RESULTS / "xgb_permutation_importance.csv", index=False)

    # (b) Fisher-only XGBoost
    ftr_tr = (Xtr @ w_star).reshape(-1, 1)
    ftr_te = (Xte @ w_star).reshape(-1, 1)
    scale = (ytr == 0).sum() / max(1, (ytr == 1).sum())
    xgb_linear = XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale,
        eval_metric="aucpr", n_jobs=-1, random_state=42, tree_method="hist",
    ).fit(ftr_tr, ytr)
    auc_fisher_monotone = roc_auc_score(yte, xgb_linear.predict_proba(ftr_te)[:, 1])

    # (c) per-quartile gain
    s_fisher = Xte @ w_star
    s_xgb = xgb.predict_proba(Xte)[:, 1]
    qs = np.quantile(s_fisher, [0, 0.25, 0.5, 0.75, 1.0])
    bid = np.clip(np.digitize(s_fisher, qs[1:-1]), 0, 3)
    strat = []
    for b in range(4):
        msk = bid == b
        if msk.sum() > 20 and yte[msk].mean() > 0 and yte[msk].mean() < 1:
            auc_f = roc_auc_score(yte[msk], s_fisher[msk])
            auc_x = roc_auc_score(yte[msk], s_xgb[msk])
            strat.append({
                "fisher_quartile": f"Q{b+1}",
                "n": int(msk.sum()),
                "n_pos": int(yte[msk].sum()),
                "auc_fisher_within": auc_f,
                "auc_xgb_within": auc_x,
                "gap": auc_x - auc_f,
            })
    strat_df = pd.DataFrame(strat)
    strat_df.to_csv(RESULTS / "fisher_quartile_strat.csv", index=False)

    return {
        "base_xgb_auc": float(base_auc),
        "xgb_auc_on_fisher_score_only": float(auc_fisher_monotone),
        "importance": imp,
        "quartile": strat_df,
    }


def economic_magnitude(yte: np.ndarray, scores_fisher: np.ndarray,
                        scores_best_singleton: np.ndarray,
                        scores_xgb: np.ndarray,
                        avg_fine_rmb: float = 3_000_000.0):
    """Back-of-envelope economic magnitude of the Fisher gain.

    We fix an FPR budget (1% and 5%), compute TPR at each, and report:
      - extra true-positive cases detected in the test panel (sample figure),
      - the implied RMB value assuming an average CSRC penalty of 3m RMB
        per market-manipulation administrative decision (median of the
        165-case dataset used by Dai et al. 2026).

    These numbers are sample-level, not structural estimates. Scaling to
    a full trading year requires the venue-wide episode count, which we
    do not have, and sector/size weights that we also do not have.
    """
    n_pos = int(yte.sum())
    rows = []
    for fpr_target in [0.01, 0.05]:
        tpr_fisher = tpr_at_fpr(yte, scores_fisher, fpr_target)
        tpr_best = tpr_at_fpr(yte, scores_best_singleton, fpr_target)
        tpr_xgb = tpr_at_fpr(yte, scores_xgb, fpr_target)

        extra_fisher = (tpr_fisher - tpr_best) * n_pos
        extra_xgb = (tpr_xgb - tpr_best) * n_pos

        rows.append({
            "fpr_target": fpr_target,
            "tpr_best_singleton": tpr_best,
            "tpr_fisher": tpr_fisher,
            "tpr_xgb": tpr_xgb,
            "extra_cases_fisher_sample": extra_fisher,
            "extra_cases_xgb_over_fisher_sample": (tpr_xgb - tpr_fisher) * n_pos,
            "rmb_gain_fisher_sample": extra_fisher * avg_fine_rmb,
            "rmb_gain_xgb_over_fisher_sample": (tpr_xgb - tpr_fisher) * n_pos * avg_fine_rmb,
        })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "economic_magnitude.csv", index=False)
    return df


# ---------- figures ----------
def fig_horserace(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    order = df.sort_values("auc", ascending=True).reset_index(drop=True)
    colors = {"singleton": "#bdbdbd", "fixed": "#f59e0b",
              "fisher": "#065f46", "ml": "#1e3a8a"}
    for i, row in order.iterrows():
        ax.barh(i, row["auc"] - 0.5, left=0.5, height=0.65,
                color=colors[row["family"]], alpha=0.9,
                edgecolor="black", linewidth=0.4)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order["rule"], fontsize=8)
    ax.set_xlabel("ROC AUC")
    ax.set_xlim(0.5, 0.82)
    ax.axvline(0.5, color="#666", linestyle="--", linewidth=0.7)
    ax.set_title("Out-of-sample discrimination: Chinese Level-2 manipulation episodes",
                 fontsize=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, ec="black", lw=0.4)
               for c in colors.values()]
    ax.legend(handles, ["singleton", "fixed-weight", "Fisher $w^\\star$", "ML"],
              loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_horserace.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_misspec(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(df["cos2_sigma0"], df["emp_ratio_d2"], s=8, alpha=0.35, color="#1e3a8a")
    ax.plot([0, 1], [0, 1], "--", color="#c53030", label="$y=x$ (Prop. 3 prediction)")
    ax.set_xlabel("$\\cos^2\\theta$ (training $\\Sigma_0$-angle to $w^\\star$)")
    ax.set_ylabel("Empirical $d^2(\\widehat{w})/d^{\\star 2}$ (test)")
    ax.set_title(f"Misspecification-identity check ($n={len(df):,}$ random weight vectors)",
                 fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, linestyle=":", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_misspec_empirical.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_rho0_cross(df: pd.DataFrame, beta=None, betas=None):
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    sc = ax.scatter(df["rho0"], df["amplification"], s=22, c=df["d_star2"],
                    cmap="viridis", alpha=0.8, edgecolor="black", linewidth=0.2)
    cb = plt.colorbar(sc, ax=ax, label="$d^{\\star 2}(\\rho_0)$")
    ax.set_xlabel("Benign-flow correlation $\\rho_0$")
    ax.set_ylabel("Two-feature amplification $d^{\\star 2}/d^2_{\\rm best}$")
    ax.set_title("Cross-sectional amplification over 66 microstructure feature pairs",
                 fontsize=10)
    ax.axhline(1.0, color="#666", linestyle="--", linewidth=0.7)
    if beta is not None:
        xg = np.linspace(df["rho0"].min(), df["rho0"].max(), 120)
        yhat = beta[0] + beta[1] * xg + beta[2] * xg ** 2
        ax.plot(xg, yhat, color="#c53030", linewidth=1.8,
                label="quadratic fit ($\\hat c = {:.2f}$)".format(beta[2]))
        if betas is not None:
            Yb = betas[:, 0:1] + betas[:, 1:2] * xg + betas[:, 2:3] * xg ** 2
            lo, hi = np.percentile(Yb, [2.5, 97.5], axis=0)
            ax.fill_between(xg, lo, hi, color="#c53030", alpha=0.15,
                            label="95% bootstrap band")
        ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, linestyle=":", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_rho0_empirical.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_xgb_importance(imp: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    order = imp.sort_values("auc_drop")
    ax.barh(order["feature"], order["auc_drop"], color="#1e3a8a",
            edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Drop in test AUC when feature is permuted")
    ax.set_title("XGBoost permutation importance on held-out Shenzhen panel",
                 fontsize=10)
    ax.grid(True, linestyle=":", linewidth=0.5, axis="x")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_xgb_importance.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_roc_pr(Xte, yte, models, mu):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    series = [
        ("best singleton", Xte[:, np.argmax(np.abs(mu))] * np.sign(mu[np.argmax(np.abs(mu))]),
         "#bdbdbd", "-"),
        ("fixed (sign($\\mu$))", Xte @ models["w_eq"], "#f59e0b", "--"),
        ("Fisher $w^\\star$", Xte @ models["w_star"], "#065f46", "-"),
        ("logistic", models["lr"].decision_function(Xte), "#1e3a8a", "--"),
        ("random forest", models["rf"].predict_proba(Xte)[:, 1], "#6b21a8", "-."),
        ("xgboost", models["xgb"].predict_proba(Xte)[:, 1], "#be185d", ":"),
    ]
    for name, s, c, ls in series:
        fpr, tpr, _ = roc_curve(yte, s)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(yte, s):.3f})",
                     color=c, linestyle=ls, linewidth=1.3)
        pr, rc, _ = precision_recall_curve(yte, s)
        axes[1].plot(rc, pr, label=f"{name} (AP={average_precision_score(yte, s):.3f})",
                     color=c, linestyle=ls, linewidth=1.3)
    axes[0].plot([0, 1], [0, 1], "--", color="#888", linewidth=0.7)
    axes[0].set_xlabel("False-positive rate")
    axes[0].set_ylabel("True-positive rate")
    axes[0].set_title("ROC")
    axes[0].legend(fontsize=7.5, loc="lower right")
    axes[0].grid(True, linestyle=":", linewidth=0.5)
    axes[1].axhline(float(yte.mean()), color="#888", linestyle="--", linewidth=0.7,
                    label=f"base rate $\\pi_1={yte.mean():.3f}$")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall")
    axes[1].legend(fontsize=7.5, loc="upper right")
    axes[1].grid(True, linestyle=":", linewidth=0.5)
    fig.suptitle("Chinese Level-2 manipulation episodes: out-of-sample performance (n={}, pos={})"
                 .format(len(yte), int(yte.sum())), fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_roc_pr.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    rng = np.random.default_rng(42)

    print("[load] reading pickles...")
    x_list = joblib.load(X_PKL)
    y_all = joblib.load(Y_PKL)
    print(f"[load] n_episodes = {len(x_list)}, pi1 = {y_all.mean():.4f}")

    print("[features] extracting 12-dim summary per window...")
    X, y = extract_features(x_list, y_all)
    print(f"[features] X shape {X.shape}, pi1 = {y.mean():.4f}")

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y,
    )
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xte_s = scaler.transform(Xte)

    # Fisher on training benign
    X_neg_tr, X_pos_tr = Xtr_s[~ytr], Xtr_s[ytr]
    Sig0, mu, w_star, d_star2 = fisher_weights(X_neg_tr, X_pos_tr)
    print(f"[fisher] d*^2 = {d_star2:.4f}")

    # 1. Horserace
    df_race, models = run_horserace(Xtr_s, Xte_s, ytr, yte, mu, w_star)
    print("\n=== horserace ===")
    print(df_race.to_string(index=False, float_format=lambda z: f"{z:.4f}"))

    # 2. Misspecification check
    print("\n[misspec] sampling 1000 random weights...")
    X_neg_te, X_pos_te = Xte_s[~yte], Xte_s[yte]
    df_miss = misspec_check(X_neg_tr, mu, Sig0, w_star, X_pos_tr,
                            X_neg_te, X_pos_te, yte, Xte_s, rng)
    miss_stats, miss_bins = misspec_diagnostics(df_miss)
    print(f"  violation rate       = {miss_stats['violation_rate']:.4f}")
    print(f"  slope (through 0)    = {miss_stats['slope_through_origin']:.4f}")
    print(f"  R^2                  = {miss_stats['r2']:.4f}")
    print(miss_bins.to_string(index=False, float_format=lambda z: f"{z:.3f}"))

    # 3. rho_0 cross-section over feature pairs + U-shape test
    print("\n[cross] computing amplification for all feature pairs...")
    df_rho = rho0_cross_section(X_neg_tr, X_pos_tr)
    print(df_rho.sort_values("amplification", ascending=False).head(10).to_string(
        index=False, float_format=lambda z: f"{z:.3f}"))
    ush, beta, betas = ushape_test(df_rho, rng)
    print(f"\n[U-shape] c_hat = {ush['c_hat']:.3f}  CI ({ush['c_ci_lo']:.3f},{ush['c_ci_hi']:.3f})"
          f"  p(c>0) = {ush['p_c_positive']:.3f}")
    print(f"[U-shape] rho_star = {ush['rho_star_hat']:.3f}  CI "
          f"({ush['rho_star_ci_lo']:.3f},{ush['rho_star_ci_hi']:.3f})")
    print(f"[U-shape] Spearman left ={ush['spearman_left']:.3f} (p={ush['spearman_left_p']:.3f})"
          f"  right ={ush['spearman_right']:.3f} (p={ush['spearman_right_p']:.3f})")

    # 4. Fisher-vs-ML analysis
    print("\n[ml-analysis] permutation importance and Fisher-score-only XGBoost...")
    ml_diag = ml_vs_fisher_analysis(Xtr_s, Xte_s, ytr, yte, models["xgb"], w_star, rng)
    print(ml_diag["importance"].to_string(index=False, float_format=lambda z: f"{z:.4f}"))
    print("\nxgb on Fisher score only: AUC =", round(ml_diag["xgb_auc_on_fisher_score_only"], 4),
          "(vs full =", round(ml_diag["base_xgb_auc"], 4), ")")
    print(ml_diag["quartile"].to_string(index=False, float_format=lambda z: f"{z:.3f}"))

    # 5. Economic magnitude
    print("\n[econ] back-of-envelope RMB magnitudes...")
    sc_best = Xte_s[:, int(np.argmax(np.abs(mu)))] * np.sign(mu[int(np.argmax(np.abs(mu)))])
    sc_fisher = Xte_s @ w_star
    sc_xgb = models["xgb"].predict_proba(Xte_s)[:, 1]
    econ = economic_magnitude(yte, sc_fisher, sc_best, sc_xgb)
    print(econ.to_string(index=False, float_format=lambda z: f"{z:.3f}"))

    # 6. Figures
    print("\n[figs] rendering...")
    fig_horserace(df_race)
    fig_misspec(df_miss)
    fig_rho0_cross(df_rho, beta=beta, betas=betas)
    fig_roc_pr(Xte_s, yte, models, mu)
    fig_xgb_importance(ml_diag["importance"])

    # 5. Tables (LaTeX) -- written manually to avoid jinja2 dependency
    # Keep the 3 best singletons + fixed + Fisher + all ML for the paper.
    top_singletons = (df_race[df_race["family"] == "singleton"]
                      .sort_values("auc", ascending=False).head(3))
    rest = df_race[df_race["family"] != "singleton"]
    race_subset = pd.concat([top_singletons, rest], ignore_index=True)

    def _latex_row(r):
        name = r["rule"].replace("_", r"\_").replace("%", r"\%")
        return f"{name} & {r['auc']:.4f} & {r['pr_auc']:.4f} & {r['tpr@5fpr']:.4f} \\\\"

    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Rule & ROC AUC & PR AUC & TPR@5\%FPR \\",
        r"\midrule",
    ]
    prev_family = None
    for _, r in race_subset.iterrows():
        if prev_family is not None and prev_family != r["family"]:
            lines.append(r"\addlinespace[2pt]")
        lines.append(_latex_row(r))
        prev_family = r["family"]
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(RESULTS / "horserace_tab.tex", "w") as f:
        f.write("\n".join(lines))

    # Save key scalars for LaTeX \newcommand injection
    d2_best_training = max(
        ((X_pos_tr[:, k].mean() - X_neg_tr[:, k].mean()) ** 2) /
        (X_neg_tr[:, k].var() + 1e-12)
        for k in range(X.shape[1]))
    scalars = {
        "n_train": int(ytr.size),
        "n_test": int(yte.size),
        "n_pos_test": int(yte.sum()),
        "pi1_overall": float(y.mean()),
        "d_star2_train": float(d_star2),
        "d2_best_train": float(d2_best_training),
        "amp_factor": float(d_star2 / d2_best_training),
        "misspec_slope": float(miss_stats["slope_through_origin"]),
        "misspec_r2": float(miss_stats["r2"]),
        "misspec_violation_rate": float(miss_stats["violation_rate"]),
        "ushape_c_hat": float(ush["c_hat"]),
        "ushape_c_ci_lo": float(ush["c_ci_lo"]),
        "ushape_c_ci_hi": float(ush["c_ci_hi"]),
        "ushape_p_c_positive": float(ush["p_c_positive"]),
        "ushape_rho_star": float(ush["rho_star_hat"]),
        "xgb_auc_on_fisher_only": float(ml_diag["xgb_auc_on_fisher_score_only"]),
        "auc_best_singleton": float(df_race[df_race["family"] == "singleton"]["auc"].max()),
        "auc_fisher": float(df_race[df_race["rule"] == "Fisher w*"]["auc"].iloc[0]),
        "auc_logistic": float(df_race[df_race["rule"] == "logistic"]["auc"].iloc[0]),
        "auc_rf": float(df_race[df_race["rule"] == "random forest"]["auc"].iloc[0]),
        "auc_xgb": float(df_race[df_race["rule"] == "xgboost"]["auc"].iloc[0]),
        "pr_best_singleton": float(df_race[df_race["family"] == "singleton"]["pr_auc"].max()),
        "pr_fisher": float(df_race[df_race["rule"] == "Fisher w*"]["pr_auc"].iloc[0]),
        "pr_xgb": float(df_race[df_race["rule"] == "xgboost"]["pr_auc"].iloc[0]),
        "tpr_best_singleton": float(df_race[df_race["family"] == "singleton"]["tpr@5fpr"].max()),
        "tpr_fisher": float(df_race[df_race["rule"] == "Fisher w*"]["tpr@5fpr"].iloc[0]),
        "tpr_xgb": float(df_race[df_race["rule"] == "xgboost"]["tpr@5fpr"].iloc[0]),
        "econ_extra_fisher_5fpr": float(econ[econ["fpr_target"] == 0.05]["extra_cases_fisher_sample"].iloc[0]),
        "econ_rmb_fisher_5fpr": float(econ[econ["fpr_target"] == 0.05]["rmb_gain_fisher_sample"].iloc[0]),
        "econ_extra_fisher_1fpr": float(econ[econ["fpr_target"] == 0.01]["extra_cases_fisher_sample"].iloc[0]),
        "econ_rmb_fisher_1fpr": float(econ[econ["fpr_target"] == 0.01]["rmb_gain_fisher_sample"].iloc[0]),
    }
    pd.Series(scalars).to_csv(RESULTS / "scalars.csv")
    print("\n=== scalars for manuscript ===")
    for k, v in scalars.items():
        print(f"  {k:20s} = {v}")

    print("\n[done] all outputs in qf/results/ and qf/figures/")


if __name__ == "__main__":
    main()
