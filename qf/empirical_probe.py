"""Diagnostic probe on Chinese Level-2 market-manipulation data.

Goal: decide whether the Chinese Level-2 order-book data (Dai et al.)
supports the Fisher-Mahalanobis amplification theorem empirically. If
yes, a full empirical section is warranted; if no, we need to understand
why before expanding the paper.

Each of 43,050 windowed episodes has ~52 time-steps x 26 LOB features.
We reduce each window to a fixed-length feature vector, then compare
singleton features, equal-weight composite, and the Fisher-optimal
composite on held-out AUC/PR-AUC/TPR@5%-FPR.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError, pinv, solve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA_ROOT = Path(
    "/Users/quinference/Library/CloudStorage/Dropbox/Documents/Research/"
    "market_manipulation_anomaly_detection/market_manipulation_detection"
)
X_PKL = DATA_ROOT / "data" / "x_data.pkl"
Y_PKL = DATA_ROOT / "data" / "y_dex.pkl"

# Column order of the 26 LOB features inside each time-step slice. The
# `sample.csv` ships a 27-column header including a leading "Time_step"
# index; the pickled windows drop that index and retain 26 features.
COL_NAMES = [
    "Highpr", "Lowpr", "Begpr", "TVolume_accu1", "TSum_accu1", "CPrice",
    "Bidpr1", "Bidpr2", "Bidpr3", "Bidpr4", "Bidpr5",
    "Bidvol1", "Bidvol2", "Bidvol3", "Bidvol4", "Bidvol5",
    "Askpr1", "Askpr2", "Askpr3", "Askpr4", "Askpr5",
    "Askvol1", "Askvol2", "Askvol3", "Askvol4", "Askvol5",
]
assert len(COL_NAMES) == 26
# Convenience indices (all 0-based into the 26-column slice):
I_HIGH, I_LOW, I_BEG = 0, 1, 2
I_TVOL_CUM, I_TSUM_CUM, I_CPRICE = 3, 4, 5
I_BIDPR = slice(6, 11)
I_BIDVOL = slice(11, 16)
I_ASKPR = slice(16, 21)
I_ASKVOL = slice(21, 26)


def window_features(win: np.ndarray) -> np.ndarray:
    """Reduce a (T, 26) window to a fixed-length feature vector.

    Features chosen to capture microstructure dimensions most closely tied
    to manipulation (rush-order intensity, cancellation/order-book imbalance,
    spread pressure, volatility) without over-engineering.
    """
    # Valid rows: the raw data has zero-filled padding rows at the start; we
    # drop any timestep where the best bid and best ask are both zero.
    bid1 = win[:, 6]   # Bidpr1
    ask1 = win[:, 16]  # Askpr1
    valid = (bid1 > 0) & (ask1 > 0)
    if valid.sum() < 5:  # degenerate window
        return np.full(12, np.nan)

    w = win[valid]
    mid = 0.5 * (w[:, 6] + w[:, 16])
    spread = w[:, 16] - w[:, 6]
    rel_spread = spread / np.clip(mid, 1e-6, None) * 1e4  # bps
    bid_depth = w[:, 11:16].sum(axis=1)
    ask_depth = w[:, 21:26].sum(axis=1)
    total_depth = bid_depth + ask_depth
    imbalance = (bid_depth - ask_depth) / np.clip(total_depth, 1.0, None)
    ret = np.diff(np.log(np.clip(mid, 1e-6, None)))
    cum_vol = w[:, I_TVOL_CUM]
    vol_step = np.clip(np.diff(cum_vol), 0, None)
    hi = w[:, I_HIGH]
    lo = w[:, I_LOW]
    hi_nz = hi[hi > 0]
    lo_nz = lo[lo > 0]
    range_bps = (hi_nz.max() - lo_nz.min()) if (hi_nz.size and lo_nz.size) else 0.0

    feats = np.array([
        np.mean(rel_spread),                     # 0: mean spread bps
        np.std(rel_spread),                      # 1: spread volatility
        np.mean(imbalance),                      # 2: mean LOB imbalance
        np.std(imbalance),                       # 3: imbalance volatility
        np.mean(bid_depth),                      # 4: mean bid depth
        np.mean(ask_depth),                      # 5: mean ask depth
        np.log1p(vol_step.sum()),                # 6: log total traded volume
        np.log1p(vol_step.max()),                # 7: log max single-step volume
        np.std(ret) * 1e4 if ret.size else 0.0,  # 8: realised vol bps
        (ret > 0).mean() if ret.size else 0.0,   # 9: up-move frequency
        np.mean(np.abs(ret)) * 1e4 if ret.size else 0.0,  # 10: mean abs return
        range_bps / np.clip(mid.mean(), 1e-6, None) * 1e4,  # 11: range bps
    ])
    return feats


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
def tpr_at_fpr(y_true, scores, target_fpr=0.05):
    neg = scores[~y_true]
    thr = np.quantile(neg, 1 - target_fpr)
    tpr = (scores[y_true] >= thr).mean()
    return float(tpr)


def summarise(name, y, s):
    return {
        "rule": name,
        "auc": roc_auc_score(y, s),
        "pr_auc": average_precision_score(y, s),
        "tpr@5fpr": tpr_at_fpr(y, s, 0.05),
    }


# ---------- main probe ----------
def main():
    print("[load] reading pickles...")
    x_list = joblib.load(X_PKL)
    y = joblib.load(Y_PKL)
    print(f"[load] n_episodes = {len(x_list)}, pi1 = {y.mean():.4f}")

    print("[features] extracting 12-dim summary per window...")
    X, y = extract_features(x_list, y)
    print(f"[features] X shape {X.shape}, pi1 after mask = {y.mean():.4f}")

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xte_s = scaler.transform(Xte)

    # analytical Mahalanobis on training benign sample
    Sig0 = np.cov(Xtr_s[~ytr].T)
    mu = Xtr_s[ytr].mean(axis=0) - Xtr_s[~ytr].mean(axis=0)
    try:
        w_star = solve(Sig0, mu)
    except LinAlgError:
        w_star = pinv(Sig0) @ mu
    d_star2 = float(mu @ w_star)

    # fixed-weight (equal + sign of mu)
    w_fixed = np.sign(mu)
    # singleton scores on test
    results = []
    for k in range(Xte_s.shape[1]):
        s = Xte_s[:, k] * np.sign(mu[k])  # align sign
        results.append(summarise(f"singleton[{k}]", yte, s))
    results.append(summarise("fixed (sign(mu))", yte, Xte_s @ w_fixed))
    results.append(summarise("Fisher w*", yte, Xte_s @ w_star))

    # logistic regression as a general linear benchmark
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr_s, ytr)
    results.append(summarise("logistic", yte, lr.decision_function(Xte_s)))

    df = pd.DataFrame(results)
    print("\n=== out-of-sample performance (test n = {}, pos = {}) ===".format(
        len(yte), int(yte.sum())
    ))
    print(df.to_string(index=False, float_format=lambda z: f"{z:.4f}"))

    feat_names = [
        "rel_spread_mean", "rel_spread_std", "imbalance_mean", "imbalance_std",
        "bid_depth_mean", "ask_depth_mean", "log_total_volume", "log_max_step_volume",
        "realised_vol_bps", "up_move_freq", "abs_return_mean_bps", "range_bps",
    ]
    tr_neg = Xtr_s[~ytr]
    tr_pos = Xtr_s[ytr]
    print("\n=== per-feature analytical deflections (training) ===")
    d2_sing = np.zeros(Xtr_s.shape[1])
    for k in range(Xtr_s.shape[1]):
        num = (tr_pos[:, k].mean() - tr_neg[:, k].mean()) ** 2
        den = tr_neg[:, k].var() + 1e-12
        d2_sing[k] = num / den
        print(f"  d^2[{feat_names[k]:>22s}] = {d2_sing[k]:.4f}")
    print(f"\n  d^*2 (analytical Fisher, training)    = {d_star2:.4f}")
    print(f"  best singleton d^2 (training)         = {d2_sing.max():.4f} ({feat_names[d2_sing.argmax()]})")
    print(f"  amplification factor  d^*2 / d^2_best = {d_star2 / d2_sing.max():.3f}")
    print(f"  amplification (analytical, difference)= {d_star2 - d2_sing.max():.4f}")

    # dump results for reuse
    out = Path(os.path.dirname(__file__)) / "results" / "probe.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n[save] wrote {out}")


if __name__ == "__main__":
    main()
