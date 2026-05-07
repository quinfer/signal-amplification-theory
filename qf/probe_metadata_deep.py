"""Deeper metadata probe: numerical inspection of every LOB-window
column and the int64 Time columns in ``data_1115.pkl``.

The first probe (:mod:`qf.probe_metadata`) used coarse range checks to
look for YYYYMMDD or unix-epoch columns and found nothing.  This second
probe goes finer:

* For every one of the 26 columns of each LOB window, it reports per-
  column statistics (min, max, mean, monotonicity flag, distinct value
  count) across episodes and looks for plausible date or sequence
  encodings.
* For the ``Time_start`` and ``Time_end`` columns of ``data_1115.pkl``
  it reports the integer range and attempts to interpret the values as
  (a) seconds since midnight, (b) HHMMSS, (c) unix epoch in seconds,
  (d) unix epoch in milliseconds.
* It also dumps the first three rows of two LOB windows in full so
  that the column semantics can be eyeballed.

Output is appended to ``qf/results/metadata_probe.md``.
"""

from __future__ import annotations

import io
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from qf.probe_metadata import CANDIDATES  # noqa: E402

OUT = PROJECT_ROOT / "qf" / "results" / "metadata_probe.md"


# ---------------------------------------------------------------------------
#  LOB-window column profiling
# ---------------------------------------------------------------------------

def _column_summary(name: str, vals: np.ndarray) -> dict:
    """Per-column summary statistics across an episode-pooled sample."""
    out = {
        "column": name,
        "n": int(vals.size),
        "min": float(np.min(vals)) if vals.size else np.nan,
        "max": float(np.max(vals)) if vals.size else np.nan,
        "mean": float(np.mean(vals)) if vals.size else np.nan,
        "std": float(np.std(vals)) if vals.size else np.nan,
        "n_unique": int(np.unique(vals).size),
        "monotone_inside_episode": "?",
    }
    return out


def _interpret_column(stat: dict) -> list[str]:
    """Heuristic interpretation of a column's likely role."""
    notes = []
    lo, hi = stat["min"], stat["max"]
    if not np.isfinite(lo) or not np.isfinite(hi):
        return ["all-NaN column"]

    # Time-of-day in seconds since midnight
    if 30000 <= lo and hi <= 60000:
        notes.append("range fits seconds-since-midnight 08:20--16:40")
    # HHMMSS encoded (e.g. 93000 = 09:30:00)
    if 90000 <= lo and hi <= 160000:
        notes.append("range fits HHMMSS 09:00--16:00")
    # YYYYMMDD
    if 19000000 < lo and hi < 21000000:
        notes.append("range fits YYYYMMDD 1900--2100")
    # Unix epoch seconds
    if 1.0e9 < lo and hi < 2.0e9:
        notes.append("range fits unix epoch (seconds)")
    # Unix epoch milliseconds
    if 1.0e12 < lo and hi < 2.0e12:
        notes.append("range fits unix epoch (milliseconds)")
    # Microstructure prices are typically in range [0.1, 1000] for CN equity
    if 0.0 < lo < 5 and 5 <= hi < 1000:
        notes.append("range fits CNY mid-price")
    # Volumes
    if 0 <= lo and 100 < hi < 1e8:
        notes.append("range fits trade volume")
    # Sequence-id-like (small integer range)
    if 0 <= lo and hi < 100 and stat["n_unique"] < 50:
        notes.append("looks like a small categorical / sequence index")

    return notes if notes else ["no obvious interpretation"]


def probe_lob_windows(x_list: list, sample_size: int = 200) -> list[str]:
    lines: list[str] = []
    lines.append("### Deeper LOB-window column probe")
    lines.append("")
    lines.append(
        f"Pooled column statistics across the first {sample_size} episodes "
        f"of ``x_data.pkl`` (53 row-snapshots each, on average)."
    )
    lines.append("")

    sample = x_list[:sample_size]
    K = sample[0].shape[1]
    lines.append(f"- per-window shape: {sample[0].shape}; n_columns = {K}")
    lines.append("")

    rows = []
    for col in range(K):
        # Pool values across all rows of the sampled windows.
        vals = np.concatenate([w[:, col] for w in sample], axis=0)
        stat = _column_summary(f"col_{col}", vals)
        # Within-episode monotone test on a randomly chosen window.
        rng = np.random.default_rng(col)
        idx = rng.integers(0, len(sample))
        win = sample[idx][:, col]
        mono_up = bool(np.all(np.diff(win) >= 0))
        mono_dn = bool(np.all(np.diff(win) <= 0))
        if mono_up and not mono_dn:
            stat["monotone_inside_episode"] = "non-decreasing"
        elif mono_dn and not mono_up:
            stat["monotone_inside_episode"] = "non-increasing"
        elif mono_up and mono_dn:
            stat["monotone_inside_episode"] = "constant"
        else:
            stat["monotone_inside_episode"] = "no"
        rows.append(stat)

    df = pd.DataFrame(rows)
    buf = io.StringIO()
    df.to_string(buf, index=False, float_format=lambda z: f"{z:.4g}")
    lines.append("```")
    lines.append(buf.getvalue())
    lines.append("```")
    lines.append("")

    lines.append("#### Per-column heuristic interpretation")
    lines.append("")
    for stat in rows:
        notes = _interpret_column(stat)
        lines.append(
            f"- `col_{stat['column'].split('_')[1]}` "
            f"[{stat['min']:.3g}, {stat['max']:.3g}], "
            f"n_unique={stat['n_unique']}, "
            f"monotone={stat['monotone_inside_episode']}: "
            f"{'; '.join(notes)}"
        )
    lines.append("")

    # Eyeball the first three rows of two windows.
    lines.append("#### First three rows of two windows (for eyeballing)")
    lines.append("")
    for k in (0, 1):
        w = sample[k]
        lines.append(f"window {k}, shape {w.shape}:")
        lines.append("```")
        with np.printoptions(precision=2, suppress=True, linewidth=180):
            lines.append(str(w[:3]))
        lines.append("```")
        lines.append("")

    # Cross-episode column statistics: does column 0 (or any column) take a
    # *different* value in different episodes?  If so, that column might be
    # an episode-level identifier such as a date index.
    lines.append("#### Cross-episode constancy check")
    lines.append("")
    lines.append(
        "For each column we report the standard deviation of the per-episode "
        "mean.  A column whose per-episode mean varies a lot is more likely "
        "to be a feature; a column whose per-episode mean is concentrated on "
        "a small set of values may be a date/stock identifier."
    )
    lines.append("")
    rows_x = []
    for col in range(K):
        per_ep_means = np.array([w[:, col].mean() for w in sample])
        rows_x.append({
            "column": f"col_{col}",
            "between_episode_std": float(per_ep_means.std()),
            "between_episode_n_unique": int(np.unique(np.round(per_ep_means, 2)).size),
            "between_episode_min": float(per_ep_means.min()),
            "between_episode_max": float(per_ep_means.max()),
        })
    df_x = pd.DataFrame(rows_x)
    buf = io.StringIO()
    df_x.to_string(buf, index=False, float_format=lambda z: f"{z:.4g}")
    lines.append("```")
    lines.append(buf.getvalue())
    lines.append("```")

    return lines


# ---------------------------------------------------------------------------
#  Time-column inspection in data_1115.pkl
# ---------------------------------------------------------------------------

def probe_time_columns(df: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    lines.append("### Deeper inspection of `Time_start` / `Time_end` in `data_1115.pkl`")
    lines.append("")
    if "Time_start" not in df.columns or "Time_end" not in df.columns:
        lines.append("- Time_start / Time_end columns not present.")
        return lines

    for col in ("Time_start", "Time_end"):
        v = df[col].astype(np.int64).values
        lines.append(f"#### `{col}`")
        lines.append("")
        lines.append(f"- min = {int(v.min())}, max = {int(v.max())}")
        lines.append(f"- mean = {float(v.mean()):.0f}, n_unique = {int(np.unique(v).size)}")
        lines.append(f"- first 10 raw: {v[:10].tolist()}")
        lines.append(f"- last 10 raw : {v[-10:].tolist()}")

        # Try interpretations.
        interpretations = []
        # Seconds since midnight
        if v.min() >= 0 and v.max() < 90000:
            mn, mx = v.min(), v.max()
            interpretations.append(
                f"as seconds-since-midnight: {mn // 3600:02d}:"
                f"{(mn % 3600) // 60:02d}:{mn % 60:02d} -- "
                f"{mx // 3600:02d}:{(mx % 3600) // 60:02d}:{mx % 60:02d}"
            )
        # HHMMSS
        if 80000 <= v.min() and v.max() <= 160000:
            interpretations.append(
                f"as HHMMSS: {v.min() // 10000:02d}:"
                f"{(v.min() // 100) % 100:02d}:{v.min() % 100:02d} -- "
                f"{v.max() // 10000:02d}:{(v.max() // 100) % 100:02d}:{v.max() % 100:02d}"
            )
        # Unix seconds
        if 1.0e9 < v.min() and v.max() < 2.0e9:
            t0 = pd.to_datetime(int(v.min()), unit="s")
            t1 = pd.to_datetime(int(v.max()), unit="s")
            interpretations.append(f"as unix epoch (s): {t0} -- {t1}")
        # Unix milliseconds
        if 1.0e12 < v.min() and v.max() < 2.0e12:
            t0 = pd.to_datetime(int(v.min()), unit="ms")
            t1 = pd.to_datetime(int(v.max()), unit="ms")
            interpretations.append(f"as unix epoch (ms): {t0} -- {t1}")

        if interpretations:
            lines.append("- plausible interpretations:")
            for i in interpretations:
                lines.append(f"  - {i}")
        else:
            lines.append("- **no standard time encoding fits this range**")
        lines.append("")

    return lines


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    lines: list[str] = []
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("# Deeper metadata probe (appended)")
    lines.append("")
    lines.append(f"Generated at {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")

    x_path = CANDIDATES["x_data.pkl"]
    if x_path.exists():
        print(f"[deep] loading {x_path} ...")
        x_list = joblib.load(x_path)
        lines.extend(probe_lob_windows(x_list, sample_size=200))
    else:
        lines.append(f"x_data.pkl not found at {x_path}")
    lines.append("")

    d_path = CANDIDATES["data_1115.pkl"]
    if d_path.exists():
        print(f"[deep] loading {d_path} ...")
        df = joblib.load(d_path)
        lines.extend(probe_time_columns(df))
    else:
        lines.append(f"data_1115.pkl not found at {d_path}")
    lines.append("")

    # Also list anything in the SD-FMM project that might be a build script
    # or a metadata sidecar.
    detect_root = CANDIDATES["x_data.pkl"].parent.parent
    lines.append("### Sibling files in `market_manipulation_detection/`")
    lines.append("")
    lines.append("(searching for build scripts and metadata sidecars)")
    lines.append("")
    interesting = []
    for p in detect_root.rglob("*"):
        if p.is_file() and p.suffix in {".py", ".csv", ".pkl", ".json", ".parquet", ".txt", ".md"}:
            if "__pycache__" in str(p):
                continue
            try:
                rel = p.relative_to(detect_root)
            except ValueError:
                rel = p
            interesting.append((p.stat().st_size, str(rel)))
    interesting.sort(key=lambda t: t[1])
    lines.append("```")
    for sz, rel in interesting[:200]:
        lines.append(f"  {sz:>12d} bytes  {rel}")
    lines.append("```")

    with open(OUT, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[deep] appended results to {OUT}")


if __name__ == "__main__":
    main()
