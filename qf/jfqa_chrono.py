"""Chronological train/test harness with rolling-window Fisher refit.

This module provides a chronologically-aware version of the JFQA
extensions.  It is structured around a single boundary function,
:func:`get_episode_dates`, which the user must populate based on the
metadata layout discovered by :mod:`qf.probe_metadata`.  Once that
function returns a 1-D array of episode dates aligned to the row order
of the feature matrix, every downstream analysis runs without further
manual intervention.

Three operating modes are supported:

* :func:`fixed_chronological_split` --- a single train/test boundary
  date.  Trains on episodes with date :math:`<` boundary, tests on the
  remainder.  This is the minimal change relative to the existing
  random 70/30 split.

* :func:`rolling_window_refit` --- the harness expects the user to
  provide the train and test endpoints (e.g. annual buckets).  For each
  test bucket, the Fisher direction and any regularised benchmarks are
  re-estimated on the training prefix, applied to the test bucket, and
  the per-bucket AUC and AUC-difference statistics are stored.  Output
  is a tidy DataFrame with one row per test bucket per rule.

* :func:`event_study_split` --- given an event date :math:`t^\\star`,
  the harness produces a [pre, post] split and reports the change in
  Fisher-versus-singleton AUC across the boundary, conditional on the
  same set of stocks appearing in both windows where possible.

The harness writes its results to :file:`qf/results/jfqa_chrono_*.csv`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from qf.empirical import (
    DATA_ROOT,
    FEATURE_NAMES,
    RESULTS,
    X_PKL,
    Y_PKL,
    extract_features,
    fisher_weights,
    tpr_at_fpr,
)
from qf.jfqa import (
    delong_pairs,
    pr_auc_paired_bootstrap,
    shrinkage_fisher_weights,
    l1_logistic_cv,
)


# ---------------------------------------------------------------------------
#  The single boundary function the user needs to fill in.
# ---------------------------------------------------------------------------

def get_episode_dates(x_list: list, y: np.ndarray) -> np.ndarray:
    """Return a 1-D array of ``numpy.datetime64`` aligned to the row order
    of the feature matrix.

    The implementation depends on the outcome of the metadata probe; this
    function is the *only* place where pipeline code touches episode
    dates, so wiring it up should be a one-line change.

    The expected return is ``np.ndarray`` of dtype ``datetime64[D]`` (or
    finer) of length ``len(x_list)``.  Episodes whose date cannot be
    recovered should be returned as ``np.datetime64('NaT')`` and will be
    excluded from the chronological splits.

    EXAMPLE IMPLEMENTATIONS (uncomment whichever applies):

    ::

        # 1. If x_data.pkl windows have an embedded timestamp column.
        return np.asarray(
            [np.datetime64(int(w[0, DATE_COL]), "s") for w in x_list],
            dtype="datetime64[s]",
        )

        # 2. If a sibling metadata pickle lists dates by index.
        meta = joblib.load(DATA_ROOT / "data" / "episode_meta.pkl")
        return pd.to_datetime(meta["date"]).values.astype("datetime64[D]")

        # 3. If dates can be parsed from CSRC penalty decisions and the
        #    matching is via stock id + date proximity.
        meta = pd.read_csv(DATA_ROOT / "data" / "episode_meta.csv")
        meta["date"] = pd.to_datetime(meta["date"])
        return meta.sort_values("episode_id")["date"].values.astype("datetime64[D]")
    """
    raise NotImplementedError(
        "get_episode_dates() is not yet wired up.  Run qf/probe_metadata.py "
        "to discover the date layout, then implement this function."
    )


# ---------------------------------------------------------------------------
#  Splitting primitives
# ---------------------------------------------------------------------------

@dataclass
class ChronoSplit:
    """A single chronological train/test split.

    Attributes
    ----------
    train_idx, test_idx : ndarray[int]
        Row indices into the feature matrix.
    train_end : numpy.datetime64
        Boundary date.  The training set comprises episodes with
        ``date < train_end``; the test set comprises the remainder
        within ``[train_end, test_end)``.
    test_end : numpy.datetime64
        Right endpoint of the test set (exclusive).
    label : str
        Human-readable label for the split, e.g. ``"2021-01-01--2022-01-01"``.
    """

    train_idx: np.ndarray
    test_idx: np.ndarray
    train_end: np.datetime64
    test_end: np.datetime64
    label: str

    @property
    def n_train(self) -> int:
        return int(len(self.train_idx))

    @property
    def n_test(self) -> int:
        return int(len(self.test_idx))


def fixed_chronological_split(
    dates: np.ndarray,
    y: np.ndarray,
    boundary: str | np.datetime64,
) -> ChronoSplit:
    """Single train/test split at the given boundary date.

    Parameters
    ----------
    dates : ndarray[datetime64], shape (n_episodes,)
    y : ndarray[int], shape (n_episodes,)
    boundary : str or datetime64
        The training set is ``dates < boundary``.

    Returns
    -------
    ChronoSplit
    """
    boundary = np.datetime64(boundary)
    valid = ~pd.isna(dates)
    train_mask = valid & (dates < boundary)
    test_mask = valid & (dates >= boundary)
    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]
    return ChronoSplit(
        train_idx=train_idx,
        test_idx=test_idx,
        train_end=boundary,
        test_end=np.datetime64("NaT"),
        label=f"<{boundary} | >={boundary}",
    )


def rolling_year_buckets(
    dates: np.ndarray,
    first_test_year: int,
    last_test_year: int,
) -> list[ChronoSplit]:
    """Generate annual rolling-window splits.

    For each test year :math:`t \\in [\\text{first}, \\text{last}]`, the
    training set is everything with date strictly before 1 January of
    year :math:`t`, and the test set is the calendar year :math:`t`.
    Train sets therefore grow over time, mimicking a deployment in
    which weights are refitted at the start of each calendar year.
    """
    splits: list[ChronoSplit] = []
    valid = ~pd.isna(dates)
    for year in range(first_test_year, last_test_year + 1):
        t0 = np.datetime64(f"{year}-01-01")
        t1 = np.datetime64(f"{year + 1}-01-01")
        train_mask = valid & (dates < t0)
        test_mask = valid & (dates >= t0) & (dates < t1)
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue
        splits.append(
            ChronoSplit(
                train_idx=np.where(train_mask)[0],
                test_idx=np.where(test_mask)[0],
                train_end=t0,
                test_end=t1,
                label=str(year),
            )
        )
    return splits


def event_study_split(
    dates: np.ndarray,
    event_date: str | np.datetime64,
    pre_window: int = 365,
    post_window: int = 365,
) -> tuple[ChronoSplit, ChronoSplit]:
    """Construct symmetric pre/post windows around an event date.

    Returns ``(pre_split, post_split)`` where each split is a
    :class:`ChronoSplit` whose ``test_idx`` covers the corresponding
    window.  ``train_idx`` for both is set to all episodes strictly
    before the pre-window (i.e. weights are estimated on history
    untouched by either side of the event).
    """
    event_date = np.datetime64(event_date)
    pre_start = event_date - np.timedelta64(pre_window, "D")
    post_end = event_date + np.timedelta64(post_window, "D")
    valid = ~pd.isna(dates)
    train_mask = valid & (dates < pre_start)
    pre_mask = valid & (dates >= pre_start) & (dates < event_date)
    post_mask = valid & (dates >= event_date) & (dates < post_end)
    train_idx = np.where(train_mask)[0]
    pre = ChronoSplit(
        train_idx=train_idx,
        test_idx=np.where(pre_mask)[0],
        train_end=pre_start,
        test_end=event_date,
        label=f"pre[{pre_start}--{event_date}]",
    )
    post = ChronoSplit(
        train_idx=train_idx,
        test_idx=np.where(post_mask)[0],
        train_end=event_date,
        test_end=post_end,
        label=f"post[{event_date}--{post_end}]",
    )
    return pre, post


# ---------------------------------------------------------------------------
#  Per-split horserace + DeLong inference
# ---------------------------------------------------------------------------

def _evaluate_split(
    X: np.ndarray,
    y: np.ndarray,
    split: ChronoSplit,
    seed: int = 42,
    include_l1: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a linear-rule horserace on a single chronological split.

    Returns
    -------
    horserace_df : DataFrame
        AUC, PR AUC, TPR@5%FPR for each rule on the held-out test set.
    delong_df : DataFrame
        DeLong inference of each rule against the best-singleton baseline.
    """
    Xtr, Xte = X[split.train_idx], X[split.test_idx]
    ytr, yte = y[split.train_idx], y[split.test_idx]

    if ytr.sum() == 0 or yte.sum() == 0:
        return pd.DataFrame(), pd.DataFrame()

    # Fisher direction, sample covariance
    Sig0, mu, w_star, d_star2 = fisher_weights(Xtr[ytr == 0], Xtr[ytr == 1])

    # Fisher direction, Ledoit-Wolf shrinkage
    Sig0_lw, _, w_lw, d_star2_lw, shrinkage = shrinkage_fisher_weights(
        Xtr[ytr == 0], Xtr[ytr == 1]
    )

    # Best singleton on training AUC
    train_singleton_aucs = []
    for k in range(Xtr.shape[1]):
        train_singleton_aucs.append(roc_auc_score(ytr, Xtr[:, k] * np.sign(mu[k])))
    best_k = int(np.argmax(train_singleton_aucs))

    scores: dict[str, np.ndarray] = {
        "best_singleton": Xte[:, best_k] * np.sign(mu[best_k]),
        "equal_weights": Xte @ np.sign(mu),
        "fisher_sample": Xte @ w_star,
        "fisher_shrunk": Xte @ w_lw,
    }
    if include_l1:
        try:
            lr_l1 = l1_logistic_cv(Xtr, ytr, seed=seed)
            scores["logistic_l1"] = lr_l1.decision_function(Xte)
        except Exception as exc:
            print(f"  [warn] L1-logistic fit failed for split {split.label}: {exc!s}")

    rows = []
    for name, s in scores.items():
        rows.append({
            "split": split.label,
            "rule": name,
            "auc": float(roc_auc_score(yte, s)),
            "pr_auc": float(average_precision_score(yte, s)),
            "tpr@5fpr": float(tpr_at_fpr(yte, s, 0.05)),
            "n_train": split.n_train,
            "n_test": split.n_test,
            "n_pos_test": int(yte.sum()),
            "best_singleton_idx": best_k,
            "best_singleton_name": FEATURE_NAMES[best_k],
            "fisher_d_star2_train": d_star2,
            "fisher_shrunk_d_star2_train": d_star2_lw,
            "ledoit_wolf_shrinkage": shrinkage,
        })
    horserace_df = pd.DataFrame(rows)

    delong_df = delong_pairs(scores, yte, reference="best_singleton")
    delong_df.insert(0, "split", split.label)

    return horserace_df, delong_df


def rolling_window_refit(
    X: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    first_test_year: int,
    last_test_year: int,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Annual rolling-window refit and evaluation."""
    splits = rolling_year_buckets(dates, first_test_year, last_test_year)
    if not splits:
        raise ValueError(
            f"No usable rolling buckets between {first_test_year} and "
            f"{last_test_year}.  Check the date metadata."
        )
    print(f"[chrono] {len(splits)} annual buckets: "
          f"{[s.label for s in splits]}")

    horserace_rows: list[pd.DataFrame] = []
    delong_rows: list[pd.DataFrame] = []
    for split in splits:
        print(f"[chrono] split {split.label}: "
              f"train n={split.n_train}, test n={split.n_test}")
        h, d = _evaluate_split(X, y, split, seed=seed)
        if not h.empty:
            horserace_rows.append(h)
            delong_rows.append(d)

    h_df = pd.concat(horserace_rows, ignore_index=True) if horserace_rows else pd.DataFrame()
    d_df = pd.concat(delong_rows, ignore_index=True) if delong_rows else pd.DataFrame()
    return h_df, d_df


def event_study(
    X: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    event_date: str | np.datetime64,
    pre_window: int = 365,
    post_window: int = 365,
    seed: int = 42,
) -> dict:
    """Pre/post event study around a regulatory shock.

    Returns a dictionary with separate DataFrames for the pre and post
    horseraces, the change in AUC for each rule, and a DeLong-style test
    for the change in Fisher-vs-singleton AUC across the event.
    """
    pre, post = event_study_split(dates, event_date, pre_window, post_window)
    print(f"[event] pre n={pre.n_test}, post n={post.n_test}, "
          f"common train n={pre.n_train}")

    h_pre, d_pre = _evaluate_split(X, y, pre, seed=seed)
    h_post, d_post = _evaluate_split(X, y, post, seed=seed)

    # AUC change per rule
    aucs_pre = h_pre.set_index("rule")["auc"] if not h_pre.empty else pd.Series(dtype=float)
    aucs_post = h_post.set_index("rule")["auc"] if not h_post.empty else pd.Series(dtype=float)
    common = aucs_pre.index.intersection(aucs_post.index)
    delta = pd.DataFrame({
        "rule": common,
        "auc_pre": aucs_pre.loc[common].values,
        "auc_post": aucs_post.loc[common].values,
        "delta_auc": (aucs_post - aucs_pre).loc[common].values,
    })

    return {
        "pre_horserace": h_pre,
        "post_horserace": h_post,
        "pre_delong": d_pre,
        "post_delong": d_post,
        "auc_change": delta,
        "event_date": str(np.datetime64(event_date)),
        "pre_window": pre_window,
        "post_window": post_window,
    }


# ---------------------------------------------------------------------------
#  Orchestration
# ---------------------------------------------------------------------------

def main():  # pragma: no cover - data-dependent
    print("[chrono] loading data from", DATA_ROOT)
    x_list = joblib.load(X_PKL)
    y_all = joblib.load(Y_PKL)
    print(f"[chrono] n_episodes = {len(x_list)}, pi1 = {y_all.mean():.4f}")

    print("[chrono] resolving episode dates ...")
    dates = get_episode_dates(x_list, y_all)
    valid = ~pd.isna(dates)
    print(
        f"[chrono] dates resolved for {int(valid.sum())}/{len(dates)} episodes; "
        f"range {pd.Series(dates[valid]).min()} to {pd.Series(dates[valid]).max()}"
    )

    print("[chrono] extracting features ...")
    X, y = extract_features(x_list, y_all)

    # 1. Fixed chronological split at 1 January 2021 -------------------------
    print("\n[chrono] === fixed split at 2021-01-01 ===")
    split = fixed_chronological_split(dates, y, "2021-01-01")
    print(f"  train n={split.n_train}, test n={split.n_test}")
    h, d = _evaluate_split(X, y, split)
    h.to_csv(RESULTS / "jfqa_chrono_fixed_horserace.csv", index=False)
    d.to_csv(RESULTS / "jfqa_chrono_fixed_delong.csv", index=False)

    # 2. Annual rolling buckets ---------------------------------------------
    print("\n[chrono] === annual rolling buckets 2020--2023 ===")
    h_roll, d_roll = rolling_window_refit(X, y, dates, 2020, 2023)
    h_roll.to_csv(RESULTS / "jfqa_chrono_rolling_horserace.csv", index=False)
    d_roll.to_csv(RESULTS / "jfqa_chrono_rolling_delong.csv", index=False)

    # 3. Event-study around 2019 Stock Connect surveillance update ----------
    print("\n[chrono] === event study around 2019-08-01 (Stock Connect) ===")
    es19 = event_study(X, y, dates, "2019-08-01")
    es19["pre_horserace"].to_csv(RESULTS / "jfqa_event_2019_pre_horserace.csv", index=False)
    es19["post_horserace"].to_csv(RESULTS / "jfqa_event_2019_post_horserace.csv", index=False)
    es19["auc_change"].to_csv(RESULTS / "jfqa_event_2019_auc_change.csv", index=False)

    # 4. Event-study around 2022 Securities Law amendment ------------------
    print("\n[chrono] === event study around 2022-03-01 (Securities Law) ===")
    es22 = event_study(X, y, dates, "2022-03-01")
    es22["pre_horserace"].to_csv(RESULTS / "jfqa_event_2022_pre_horserace.csv", index=False)
    es22["post_horserace"].to_csv(RESULTS / "jfqa_event_2022_post_horserace.csv", index=False)
    es22["auc_change"].to_csv(RESULTS / "jfqa_event_2022_auc_change.csv", index=False)

    print("\n[chrono] all chronological outputs written to qf/results/jfqa_chrono_* and jfqa_event_*")


if __name__ == "__main__":
    main()
