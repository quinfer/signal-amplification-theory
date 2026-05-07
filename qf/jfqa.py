"""JFQA-targeted extensions to the empirical pipeline.

This module adds six analyses that the existing :mod:`empirical` pipeline
does not perform but that referees at the *Journal of Financial and
Quantitative Analysis* would expect to see:

1. **DeLong (1988) inference on paired ROC AUC differences.**  We implement
   the fast midrank algorithm of Sun and Xu (2014); see
   :func:`fast_delong` and :func:`delong_pairs`.

2. **Paired bootstrap on PR AUC differences.**  We resample test indices
   with replacement and recompute PR AUC for every rule on each resample,
   producing 95% percentile confidence intervals on every pairwise
   difference in PR AUC.

3. **Shrinkage Fisher direction.**  We re-estimate the Fisher direction
   using a Ledoit and Wolf (2004) shrinkage estimator for the benign
   covariance, providing a regularised linear benchmark that addresses
   the high-dimensional / noisy-covariance referee complaint.

4. **L1-penalised logistic regression with cross-validated penalty.**
   This is the natural regularised linear competitor to Fisher's rule
   under the common-covariance Gaussian assumption; if Fisher matches it,
   the linear-rule ceiling is robust to sparsity.

5. **Label-noise robustness.**  For each flipping fraction
   :math:`\eta\in\{0.02,0.05,0.10\}` we randomly recode that fraction of
   labelled-benign episodes as positives in *training* only, refit the
   horserace, and check that the Fisher-versus-singleton gap survives.

6. **Cluster bootstrap on the U-shape curvature.**  The 66 feature pairs
   share features and are therefore not independent.  We replace the
   pair-level bootstrap of :func:`empirical.ushape_test` with a feature
   bootstrap (resample 12 features with replacement, take all pairs of
   resampled features, refit the quadratic).  The CI widens honestly.

The module is designed to be *additive*: it does not modify the existing
pipeline.  Run ``python qf/jfqa.py`` after ``python qf/empirical.py`` to
produce the JFQA outputs in :file:`qf/results/jfqa_*.csv`.
"""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError, pinv, solve
from scipy import stats
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from qf.empirical import (
    DATA_ROOT,
    FEATURE_NAMES,
    RESULTS,
    X_PKL,
    Y_PKL,
    deflection,
    extract_features,
    fisher_weights,
    sigma0_cos2,
    summarise,
    tpr_at_fpr,
)


# ---------------------------------------------------------------------------
#  1. DeLong (1988) inference on paired ROC AUC differences
# ---------------------------------------------------------------------------

def _midrank(x: np.ndarray) -> np.ndarray:
    """Compute midranks of ``x`` in :math:`O(n\\log n)`.

    Ties are assigned the average of the ranks they would have been
    assigned in arbitrary order, the standard convention for DeLong's
    method.
    """
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    n = len(x)
    midranks_sorted = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        midranks_sorted[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    midranks = np.empty(n, dtype=np.float64)
    midranks[order] = midranks_sorted
    return midranks


def fast_delong(scores: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fast DeLong covariance for an array of paired AUCs.

    Parameters
    ----------
    scores : ndarray, shape (n_rules, n_obs)
        Each row holds the scores of one classifier on the *same* test
        set.
    y : ndarray, shape (n_obs,)
        Binary labels; positives have ``y == 1``.

    Returns
    -------
    aucs : ndarray, shape (n_rules,)
    cov : ndarray, shape (n_rules, n_rules)
        The DeLong covariance of the empirical AUCs.

    References
    ----------
    DeLong, DeLong, and Clarke-Pearson (1988); Sun and Xu (2014) for the
    midrank-based fast algorithm.
    """
    pos_mask = y == 1
    neg_mask = ~pos_mask
    m = int(pos_mask.sum())
    n = int(neg_mask.sum())
    if m == 0 or n == 0:
        raise ValueError("DeLong requires at least one positive and one negative")

    K = scores.shape[0]
    aucs = np.empty(K)
    V10 = np.empty((K, m))  # one row per rule, columns indexed by positives
    V01 = np.empty((K, n))  # one row per rule, columns indexed by negatives

    for r in range(K):
        s = scores[r]
        s_pos = s[pos_mask]
        s_neg = s[neg_mask]
        T_pos = _midrank(s_pos)
        T_neg = _midrank(s_neg)
        T_all = _midrank(s)
        T_all_pos = T_all[pos_mask]
        T_all_neg = T_all[neg_mask]
        # V10[i] = P(s_pos[i] > s_neg) for positive i
        V10[r] = (T_all_pos - T_pos) / n
        # V01[j] = P(s_pos > s_neg[j]) for negative j
        V01[r] = 1.0 - (T_all_neg - T_neg) / m
        aucs[r] = V10[r].mean()

    # Covariance of structural components, using sample covariance with
    # the (m-1) and (n-1) corrections in the conventional DeLong way.
    S10 = np.cov(V10) if K > 1 else np.array([[V10[0].var(ddof=1)]])
    S01 = np.cov(V01) if K > 1 else np.array([[V01[0].var(ddof=1)]])
    if S10.ndim == 0:
        S10 = S10.reshape(1, 1)
    if S01.ndim == 0:
        S01 = S01.reshape(1, 1)
    cov = S10 / m + S01 / n
    return aucs, cov


def delong_pairs(
    scores_dict: dict[str, np.ndarray],
    y: np.ndarray,
    reference: str,
) -> pd.DataFrame:
    """For every rule in ``scores_dict``, test the AUC difference against
    ``reference`` using DeLong.

    Returns a :class:`pandas.DataFrame` with one row per rule containing
    the AUC, its DeLong standard error, the AUC difference relative to
    the reference, the standard error of the difference, the two-sided
    Wald p-value, and a 95% Wald confidence interval on the difference.
    """
    rules = list(scores_dict.keys())
    if reference not in rules:
        raise ValueError(f"Reference rule {reference!r} not in scores_dict")
    score_matrix = np.vstack([scores_dict[r] for r in rules])
    aucs, cov = fast_delong(score_matrix, y)

    ref_idx = rules.index(reference)
    rows = []
    for i, name in enumerate(rules):
        diff = aucs[i] - aucs[ref_idx]
        var_diff = cov[i, i] + cov[ref_idx, ref_idx] - 2 * cov[i, ref_idx]
        se_diff = float(np.sqrt(max(var_diff, 0.0)))
        if se_diff > 0 and i != ref_idx:
            z = diff / se_diff
            p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
            lo = diff - 1.959964 * se_diff
            hi = diff + 1.959964 * se_diff
        else:
            z = np.nan
            p = np.nan
            lo = np.nan
            hi = np.nan
        rows.append({
            "rule": name,
            "auc": float(aucs[i]),
            "auc_se": float(np.sqrt(cov[i, i])),
            "auc_diff_vs_ref": float(diff),
            "diff_se": se_diff,
            "z": float(z) if np.isfinite(z) else np.nan,
            "p_value": float(p) if np.isfinite(p) else np.nan,
            "ci95_lo": float(lo) if np.isfinite(lo) else np.nan,
            "ci95_hi": float(hi) if np.isfinite(hi) else np.nan,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  2. Paired bootstrap on PR AUC differences
# ---------------------------------------------------------------------------

def pr_auc_paired_bootstrap(
    scores_dict: dict[str, np.ndarray],
    y: np.ndarray,
    reference: str,
    n_boot: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Paired bootstrap on PR AUC differences.

    For each of ``n_boot`` resamples we redraw test indices with
    replacement (preserving the pairing), recompute PR AUC for every
    rule, and form the difference relative to ``reference``.  We report
    the point estimate, the bootstrap standard error, and 95% percentile
    CIs on each difference.
    """
    rules = list(scores_dict.keys())
    if reference not in rules:
        raise ValueError(f"Reference rule {reference!r} not in scores_dict")
    n = len(y)
    rng = np.random.default_rng(seed)

    # Point estimates
    pr_full = {r: average_precision_score(y, scores_dict[r]) for r in rules}

    diffs = {r: np.empty(n_boot) for r in rules}
    skipped = 0
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == n:
            skipped += 1
            for r in rules:
                diffs[r][b] = np.nan
            continue
        ref_pr = average_precision_score(yb, scores_dict[reference][idx])
        for r in rules:
            pr_r = average_precision_score(yb, scores_dict[r][idx])
            diffs[r][b] = pr_r - ref_pr

    rows = []
    for r in rules:
        d = diffs[r][~np.isnan(diffs[r])]
        rows.append({
            "rule": r,
            "pr_auc": float(pr_full[r]),
            "pr_diff_vs_ref": float(pr_full[r] - pr_full[reference]),
            "boot_se": float(d.std(ddof=1)) if len(d) > 1 else np.nan,
            "ci95_lo": float(np.percentile(d, 2.5)) if len(d) > 0 else np.nan,
            "ci95_hi": float(np.percentile(d, 97.5)) if len(d) > 0 else np.nan,
            "n_boot_used": int(len(d)),
        })
    if skipped:
        print(f"[pr_bootstrap] skipped {skipped}/{n_boot} resamples with degenerate label distribution")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  3. Shrinkage Fisher direction (Ledoit-Wolf)
# ---------------------------------------------------------------------------

def shrinkage_fisher_weights(X_neg: np.ndarray, X_pos: np.ndarray):
    """Fisher direction with Ledoit-Wolf shrinkage of the benign covariance.

    Returns
    -------
    Sig0_lw : ndarray
        The shrunk benign covariance.
    mu : ndarray
        The mean shift.
    w : ndarray
        Fisher direction :math:`\\hat\\Sigma_0^{-1}\\hat\\mu` using the
        shrunk covariance.
    d_star2 : float
        Mahalanobis deflection in the shrunk metric.
    shrinkage : float
        The Ledoit-Wolf shrinkage intensity.
    """
    lw = LedoitWolf().fit(X_neg)
    Sig0_lw = lw.covariance_
    mu = X_pos.mean(axis=0) - X_neg.mean(axis=0)
    try:
        w = solve(Sig0_lw, mu)
    except LinAlgError:
        w = pinv(Sig0_lw) @ mu
    d_star2 = float(mu @ w)
    return Sig0_lw, mu, w, d_star2, float(lw.shrinkage_)


# ---------------------------------------------------------------------------
#  4. L1-penalised logistic regression
# ---------------------------------------------------------------------------

def l1_logistic_cv(Xtr: np.ndarray, ytr: np.ndarray, seed: int = 42) -> LogisticRegressionCV:
    """L1-penalised logistic regression with 5-fold cross-validated penalty.

    Class-balanced loss; ``saga`` solver to handle L1 with class weights.
    """
    return LogisticRegressionCV(
        Cs=10,
        cv=5,
        penalty="l1",
        solver="saga",
        scoring="roc_auc",
        max_iter=5000,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    ).fit(Xtr, ytr)


# ---------------------------------------------------------------------------
#  5. Label-noise robustness
# ---------------------------------------------------------------------------

def label_noise_horserace(
    Xtr: np.ndarray,
    Xte: np.ndarray,
    ytr: np.ndarray,
    yte: np.ndarray,
    flip_fractions: tuple[float, ...] = (0.02, 0.05, 0.10),
    n_seeds: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Re-run the linear part of the horserace under random benign-to-positive
    label flips on training data only.  Test labels are untouched.

    For each fraction :math:`\\eta` we sample :func:`n_seeds` distinct
    flip realisations and report the mean and standard deviation of:
    Fisher AUC, best-singleton AUC, the Fisher-minus-singleton gap, and
    the cosine alignment of the noisy Fisher direction with the
    clean-data Fisher direction.
    """
    Sig0_clean, mu_clean, w_clean, d_star2_clean = fisher_weights(
        Xtr[ytr == 0], Xtr[ytr == 1]
    )

    rows = []
    rng_seed_seq = np.random.SeedSequence(seed)
    seeds = rng_seed_seq.spawn(len(flip_fractions) * n_seeds)
    flat_idx = 0
    for eta in flip_fractions:
        for s in range(n_seeds):
            rng = np.random.default_rng(seeds[flat_idx])
            flat_idx += 1
            ytr_noisy = ytr.copy()
            neg_idx = np.where(ytr_noisy == 0)[0]
            n_flip = int(round(eta * len(neg_idx)))
            flip = rng.choice(neg_idx, size=n_flip, replace=False)
            ytr_noisy[flip] = 1

            X_neg = Xtr[ytr_noisy == 0]
            X_pos = Xtr[ytr_noisy == 1]
            try:
                Sig0_n, mu_n, w_n, d_star2_n = fisher_weights(X_neg, X_pos)
            except Exception as exc:  # pragma: no cover - degenerate case
                rows.append({
                    "eta": eta, "seed": s, "error": str(exc),
                    "auc_fisher": np.nan, "auc_best_singleton": np.nan,
                    "auc_gap": np.nan, "cos_alignment": np.nan,
                })
                continue
            scores_fisher = Xte @ w_n

            singleton_aucs = []
            for k in range(Xte.shape[1]):
                s_k = Xte[:, k] * np.sign(mu_n[k])
                singleton_aucs.append(roc_auc_score(yte, s_k))
            best_singleton = float(np.max(singleton_aucs))

            auc_fisher = roc_auc_score(yte, scores_fisher)
            cos_align = sigma0_cos2(w_n, w_clean, Sig0_clean) ** 0.5

            rows.append({
                "eta": eta,
                "seed": s,
                "auc_fisher": auc_fisher,
                "auc_best_singleton": best_singleton,
                "auc_gap": auc_fisher - best_singleton,
                "cos_alignment": float(cos_align),
                "d_star2": float(d_star2_n),
            })

    df = pd.DataFrame(rows)
    return df


def summarise_label_noise(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the per-seed label-noise results into mean/SD by eta."""
    g = df.groupby("eta")
    out = pd.DataFrame({
        "auc_fisher_mean": g["auc_fisher"].mean(),
        "auc_fisher_sd": g["auc_fisher"].std(),
        "auc_best_singleton_mean": g["auc_best_singleton"].mean(),
        "auc_gap_mean": g["auc_gap"].mean(),
        "auc_gap_sd": g["auc_gap"].std(),
        "cos_alignment_mean": g["cos_alignment"].mean(),
        "cos_alignment_min": g["cos_alignment"].min(),
        "n_seeds": g.size(),
    }).reset_index()
    return out


# ---------------------------------------------------------------------------
#  6. Cluster bootstrap for U-shape curvature
# ---------------------------------------------------------------------------

def ushape_cluster_bootstrap(
    X_neg_tr: np.ndarray,
    X_pos_tr: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict:
    """Cluster bootstrap on the U-shape curvature.

    The 66 feature pairs share features and are therefore not
    independent observations.  We instead resample features (12 of them)
    with replacement, take all unordered pairs of *distinct* indices in
    the resampled feature set, recompute pairwise rho_0 and the
    pairwise amplification ratio, and refit the quadratic.

    The resulting CI on the curvature is wider than the pair-level
    bootstrap in :func:`empirical.ushape_test` and is the honest
    inference under the natural clustering structure.
    """
    K = X_neg_tr.shape[1]
    mu = X_pos_tr.mean(axis=0) - X_neg_tr.mean(axis=0)

    # Original 66-pair fit (for comparison)
    rho_orig, amp_orig = _pairs_amplification(X_neg_tr, X_pos_tr, mu, np.arange(K))
    X_orig = np.column_stack([np.ones_like(rho_orig), rho_orig, rho_orig ** 2])
    beta_orig, *_ = np.linalg.lstsq(X_orig, amp_orig, rcond=None)
    c_orig = float(beta_orig[2])
    rho_star_orig = float(-beta_orig[1] / (2 * beta_orig[2])) if beta_orig[2] > 1e-8 else np.nan

    # Cluster (feature) bootstrap
    rng = np.random.default_rng(seed)
    cs = np.empty(n_boot)
    rho_stars = np.empty(n_boot)
    n_used = np.empty(n_boot, dtype=int)
    for b in range(n_boot):
        feat_idx = rng.integers(0, K, K)
        # Use the *unique* features so the LDA pair set is well-defined.
        # The clustering effect comes from the (random) restriction of
        # the pair set rather than from repeated rows.
        unique_feats = np.unique(feat_idx)
        if len(unique_feats) < 3:
            cs[b] = np.nan
            rho_stars[b] = np.nan
            n_used[b] = 0
            continue
        rho_b, amp_b = _pairs_amplification(X_neg_tr, X_pos_tr, mu, unique_feats)
        if len(rho_b) < 3:
            cs[b] = np.nan
            rho_stars[b] = np.nan
            n_used[b] = 0
            continue
        Xb = np.column_stack([np.ones_like(rho_b), rho_b, rho_b ** 2])
        bb, *_ = np.linalg.lstsq(Xb, amp_b, rcond=None)
        cs[b] = bb[2]
        rho_stars[b] = -bb[1] / (2 * bb[2]) if abs(bb[2]) > 1e-8 else np.nan
        n_used[b] = len(rho_b)

    cs = cs[~np.isnan(cs)]
    rho_stars_ok = rho_stars[~np.isnan(rho_stars)]
    return {
        "c_hat_orig": c_orig,
        "rho_star_orig": rho_star_orig,
        "c_ci_lo_cluster": float(np.percentile(cs, 2.5)),
        "c_ci_hi_cluster": float(np.percentile(cs, 97.5)),
        "p_c_positive_cluster": float((cs > 0).mean()),
        "rho_star_ci_lo_cluster": float(np.percentile(rho_stars_ok, 2.5))
            if len(rho_stars_ok) else np.nan,
        "rho_star_ci_hi_cluster": float(np.percentile(rho_stars_ok, 97.5))
            if len(rho_stars_ok) else np.nan,
        "n_pairs_orig": len(rho_orig),
        "median_pairs_per_boot": int(np.median(n_used[n_used > 0])),
    }


def _pairs_amplification(
    X_neg: np.ndarray, X_pos: np.ndarray, mu: np.ndarray, feat_idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Helper: compute rho_0 and amplification for all unordered pairs of
    features within ``feat_idx``.
    """
    rhos = []
    amps = []
    for a in range(len(feat_idx)):
        for b in range(a + 1, len(feat_idx)):
            i, j = int(feat_idx[a]), int(feat_idx[b])
            Sig_ij = np.cov(X_neg[:, [i, j]].T)
            mu_ij = mu[[i, j]]
            try:
                w = solve(Sig_ij, mu_ij)
            except LinAlgError:
                w = pinv(Sig_ij) @ mu_ij
            d_star2 = float(mu_ij @ w)
            d2_i = mu_ij[0] ** 2 / max(Sig_ij[0, 0], 1e-12)
            d2_j = mu_ij[1] ** 2 / max(Sig_ij[1, 1], 1e-12)
            d2_best = max(d2_i, d2_j)
            rho0 = Sig_ij[0, 1] / max(np.sqrt(Sig_ij[0, 0] * Sig_ij[1, 1]), 1e-12)
            rhos.append(rho0)
            amps.append(d_star2 / max(d2_best, 1e-12))
    return np.asarray(rhos), np.asarray(amps)


# ---------------------------------------------------------------------------
#  Orchestration
# ---------------------------------------------------------------------------

def run_jfqa_pipeline(
    Xtr: np.ndarray,
    Xte: np.ndarray,
    ytr: np.ndarray,
    yte: np.ndarray,
    seed: int = 42,
) -> dict:
    """Run the full JFQA extension on a single train/test split.

    Returns a dictionary of result DataFrames and writes each to
    :file:`qf/results/jfqa_*.csv`.
    """
    print("[jfqa] computing Fisher direction (sample covariance) ...")
    Sig0, mu, w_star, d_star2 = fisher_weights(Xtr[ytr == 0], Xtr[ytr == 1])

    print("[jfqa] computing Fisher direction (Ledoit-Wolf shrinkage) ...")
    Sig0_lw, _, w_lw, d_star2_lw, shrinkage = shrinkage_fisher_weights(
        Xtr[ytr == 0], Xtr[ytr == 1]
    )

    # Build score dictionary for inference: best singleton, equal-weight, Fisher,
    # shrunk Fisher, logistic, L1-logistic, RF, XGBoost.
    print("[jfqa] training linear and ML benchmarks ...")
    scores: dict[str, np.ndarray] = {}

    # Singletons (sign-aligned to mu); pick best on the *training* AUCs to avoid
    # peeking at test labels.
    train_singleton_aucs = []
    for k in range(Xtr.shape[1]):
        s_tr = Xtr[:, k] * np.sign(mu[k])
        train_singleton_aucs.append(roc_auc_score(ytr, s_tr))
    best_k = int(np.argmax(train_singleton_aucs))
    scores["best_singleton"] = Xte[:, best_k] * np.sign(mu[best_k])

    # Equal-weight composite (sign-aligned)
    scores["equal_weights"] = Xte @ np.sign(mu)

    # Fisher (sample) and Fisher (shrinkage)
    scores["fisher_sample"] = Xte @ w_star
    scores["fisher_shrunk"] = Xte @ w_lw

    # L2 logistic (class-balanced) and L1-CV logistic
    lr_l2 = LogisticRegression(
        max_iter=5000, class_weight="balanced", random_state=seed
    ).fit(Xtr, ytr)
    scores["logistic_l2"] = lr_l2.decision_function(Xte)

    lr_l1 = l1_logistic_cv(Xtr, ytr, seed=seed)
    scores["logistic_l1"] = lr_l1.decision_function(Xte)

    # Random forest and XGBoost (matching empirical.py settings)
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=20, n_jobs=-1,
        class_weight="balanced_subsample", random_state=seed,
    ).fit(Xtr, ytr)
    scores["random_forest"] = rf.predict_proba(Xte)[:, 1]

    scale = (ytr == 0).sum() / max(1, (ytr == 1).sum())
    xgb = XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale,
        eval_metric="aucpr", n_jobs=-1, random_state=seed, tree_method="hist",
    ).fit(Xtr, ytr)
    scores["xgboost"] = xgb.predict_proba(Xte)[:, 1]

    # ---------- (1) DeLong inference ----------
    print("[jfqa] running DeLong inference (reference: best_singleton) ...")
    df_delong_singleton = delong_pairs(scores, yte, reference="best_singleton")
    df_delong_singleton.to_csv(RESULTS / "jfqa_delong_vs_singleton.csv", index=False)

    print("[jfqa] running DeLong inference (reference: fisher_sample) ...")
    df_delong_fisher = delong_pairs(scores, yte, reference="fisher_sample")
    df_delong_fisher.to_csv(RESULTS / "jfqa_delong_vs_fisher.csv", index=False)

    # ---------- (2) PR AUC paired bootstrap ----------
    print("[jfqa] paired bootstrap on PR AUC differences (reference: best_singleton) ...")
    df_pr = pr_auc_paired_bootstrap(
        scores, yte, reference="best_singleton", n_boot=1000, seed=seed
    )
    df_pr.to_csv(RESULTS / "jfqa_pr_bootstrap_vs_singleton.csv", index=False)

    print("[jfqa] paired bootstrap on PR AUC differences (reference: fisher_sample) ...")
    df_pr_f = pr_auc_paired_bootstrap(
        scores, yte, reference="fisher_sample", n_boot=1000, seed=seed
    )
    df_pr_f.to_csv(RESULTS / "jfqa_pr_bootstrap_vs_fisher.csv", index=False)

    # ---------- (3+4) Shrinkage / regularised horserace summary ----------
    print("[jfqa] writing extended horserace ...")
    rows = []
    for name, s in scores.items():
        rows.append({
            "rule": name,
            "auc": float(roc_auc_score(yte, s)),
            "pr_auc": float(average_precision_score(yte, s)),
            "tpr@5fpr": float(tpr_at_fpr(yte, s, 0.05)),
        })
    df_extended = pd.DataFrame(rows)
    df_extended.to_csv(RESULTS / "jfqa_extended_horserace.csv", index=False)

    extra = pd.DataFrame([{
        "fisher_sample_d_star2_train": d_star2,
        "fisher_shrunk_d_star2_train": d_star2_lw,
        "ledoit_wolf_shrinkage_intensity": shrinkage,
        "l1_logistic_C": float(lr_l1.C_[0]),
        "l1_logistic_n_nonzero": int((lr_l1.coef_ != 0).sum()),
        "best_singleton_idx_train": best_k,
        "best_singleton_name": FEATURE_NAMES[best_k],
    }])
    extra.to_csv(RESULTS / "jfqa_diagnostics.csv", index=False)

    # ---------- (5) Label noise ----------
    print("[jfqa] label-noise robustness (eta in {0.02, 0.05, 0.10}) ...")
    df_ln = label_noise_horserace(
        Xtr, Xte, ytr, yte, flip_fractions=(0.02, 0.05, 0.10), n_seeds=5, seed=seed
    )
    df_ln.to_csv(RESULTS / "jfqa_label_noise_full.csv", index=False)
    df_ln_summary = summarise_label_noise(df_ln)
    df_ln_summary.to_csv(RESULTS / "jfqa_label_noise_summary.csv", index=False)

    # ---------- (6) Cluster bootstrap on U-shape ----------
    print("[jfqa] cluster bootstrap on U-shape curvature ...")
    ushape = ushape_cluster_bootstrap(
        Xtr[ytr == 0], Xtr[ytr == 1], n_boot=2000, seed=seed
    )
    pd.DataFrame([ushape]).to_csv(RESULTS / "jfqa_ushape_cluster.csv", index=False)

    return {
        "delong_vs_singleton": df_delong_singleton,
        "delong_vs_fisher": df_delong_fisher,
        "pr_bootstrap_vs_singleton": df_pr,
        "pr_bootstrap_vs_fisher": df_pr_f,
        "extended_horserace": df_extended,
        "diagnostics": extra,
        "label_noise_full": df_ln,
        "label_noise_summary": df_ln_summary,
        "ushape_cluster": ushape,
    }


def main():  # pragma: no cover - data-dependent
    print("[jfqa] loading data from", DATA_ROOT)
    x_list = joblib.load(X_PKL)
    y_all = joblib.load(Y_PKL)
    print(f"[jfqa] n_episodes = {len(x_list)}, pi1 = {y_all.mean():.4f}")

    print("[jfqa] extracting 12-dim feature vectors ...")
    X, y = extract_features(x_list, y_all)
    print(f"[jfqa] X shape = {X.shape}, n_pos = {int(y.sum())}")

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    print(f"[jfqa] train: {Xtr.shape[0]} (pos {int(ytr.sum())}); "
          f"test: {Xte.shape[0]} (pos {int(yte.sum())})")

    t0 = time.time()
    results = run_jfqa_pipeline(Xtr, Xte, ytr, yte, seed=42)
    print(f"[jfqa] done in {time.time() - t0:.1f}s")

    print("\n=== DeLong vs best_singleton ===")
    print(results["delong_vs_singleton"].to_string(
        index=False, float_format=lambda z: f"{z:.4f}"))

    print("\n=== DeLong vs fisher_sample ===")
    print(results["delong_vs_fisher"].to_string(
        index=False, float_format=lambda z: f"{z:.4f}"))

    print("\n=== PR AUC bootstrap vs best_singleton ===")
    print(results["pr_bootstrap_vs_singleton"].to_string(
        index=False, float_format=lambda z: f"{z:.4f}"))

    print("\n=== Label-noise summary ===")
    print(results["label_noise_summary"].to_string(
        index=False, float_format=lambda z: f"{z:.4f}"))

    print("\n=== U-shape cluster bootstrap ===")
    for k, v in results["ushape_cluster"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
