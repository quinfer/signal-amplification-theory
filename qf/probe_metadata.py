"""Probe the source data files for episode-level date metadata.

This script is intended to be run on Barry's Mac (where the source data
lives outside the project tree).  It introspects each candidate file and
writes a structured summary to ``qf/results/metadata_probe.md`` so that
the chronological-split harness can be wired up cleanly.

The script is *read-only*: it never modifies any source file.  It loads
each pickle into memory, inspects its top-level structure, looks for
fields that resemble timestamps or stock identifiers, and reports what
it finds.

Run with::

    python qf/probe_metadata.py
"""

from __future__ import annotations

import io
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import joblib
except ImportError:  # pragma: no cover
    print("joblib is required; pip install joblib")
    sys.exit(1)


# Candidate paths on Barry's machine -----------------------------------------

DROPBOX_RESEARCH = Path.home() / (
    "Library/CloudStorage/Dropbox/Documents/Research/"
    "market_manipulation_anomaly_detection"
)
DETECT = DROPBOX_RESEARCH / "market_manipulation_detection"

CANDIDATES: dict[str, Path] = {
    "x_data.pkl": DETECT / "data" / "x_data.pkl",
    "y_dex.pkl": DETECT / "data" / "y_dex.pkl",
    "data_1115.pkl": DETECT / "Documents and Data" / "data_1115.pkl",
    "tick_sample_000001.csv": (
        DROPBOX_RESEARCH
        / "chinese market manipulation cases (sample)"
        / "000001.csv"
    ),
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "qf" / "results" / "metadata_probe.md"
OUT.parent.mkdir(parents=True, exist_ok=True)


# Helpers --------------------------------------------------------------------

def _looks_like_date(s: str) -> bool:
    """Heuristic: does a string parse as a date in any common format?"""
    if not isinstance(s, str) or len(s) < 6:
        return False
    candidates = [
        "%Y-%m-%d", "%Y/%m/%d", "%Y%m%d",
        "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f",
        "%Y%m%d %H:%M:%S",
    ]
    for fmt in candidates:
        try:
            datetime.strptime(s[:len(fmt) + 6], fmt)
            return True
        except ValueError:
            continue
    return False


def _summarise_array(name: str, arr: Any, max_show: int = 3) -> list[str]:
    lines: list[str] = []
    if isinstance(arr, np.ndarray):
        lines.append(f"- `{name}`: ndarray, shape={arr.shape}, dtype={arr.dtype}")
        flat = arr.ravel()
        if flat.size:
            try:
                head = [repr(flat[i]) for i in range(min(max_show, flat.size))]
                lines.append(f"  - first {len(head)}: {', '.join(head)}")
            except Exception:
                pass
            # If looks date-like
            try:
                sample = str(flat[0])
                if _looks_like_date(sample):
                    lines.append(f"  - **looks like a date field** (sample {sample!r})")
            except Exception:
                pass
            try:
                if np.issubdtype(arr.dtype, np.datetime64):
                    lines.append(
                        f"  - **datetime64**: min={arr.min()}, max={arr.max()}"
                    )
            except Exception:
                pass
    elif isinstance(arr, list):
        lines.append(f"- `{name}`: list, length={len(arr)}")
        if len(arr):
            x0 = arr[0]
            if isinstance(x0, np.ndarray):
                lines.append(
                    f"  - element[0] is ndarray, shape={x0.shape}, dtype={x0.dtype}"
                )
            else:
                lines.append(f"  - element[0]: type={type(x0).__name__}, value preview={str(x0)[:120]}")
                if isinstance(x0, str) and _looks_like_date(x0):
                    lines.append("  - **list of date-like strings**")
    elif isinstance(arr, dict):
        lines.append(f"- `{name}`: dict with {len(arr)} keys")
        for k in list(arr.keys())[:10]:
            v = arr[k]
            try:
                shape = getattr(v, "shape", None)
                dtype = getattr(v, "dtype", None)
                lines.append(f"  - key {k!r}: type={type(v).__name__}"
                             + (f", shape={shape}" if shape is not None else "")
                             + (f", dtype={dtype}" if dtype is not None else ""))
            except Exception:
                lines.append(f"  - key {k!r}: type={type(v).__name__}")
    elif isinstance(arr, pd.DataFrame):
        lines.append(f"- `{name}`: DataFrame, shape={arr.shape}")
        lines.append(f"  - columns: {list(arr.columns)}")
        lines.append(f"  - dtypes: {arr.dtypes.to_dict()}")
        for col in arr.columns:
            try:
                if pd.api.types.is_datetime64_any_dtype(arr[col]):
                    lines.append(
                        f"  - **datetime column** {col!r}: "
                        f"min={arr[col].min()}, max={arr[col].max()}"
                    )
                elif arr[col].dtype == "O":
                    sample = str(arr[col].dropna().iloc[0]) if len(arr[col].dropna()) else ""
                    if _looks_like_date(sample):
                        lines.append(f"  - **possibly date column** {col!r}: sample {sample!r}")
            except Exception:
                pass
    else:
        lines.append(f"- `{name}`: type={type(arr).__name__}, repr={repr(arr)[:200]}")
    return lines


def probe_pickle(path: Path) -> list[str]:
    lines: list[str] = []
    if not path.exists():
        return [f"### `{path.name}` --- **NOT FOUND** at `{path}`"]
    lines.append(f"### `{path.name}`")
    lines.append(f"- path: `{path}`")
    lines.append(f"- size: {path.stat().st_size / 1e6:.2f} MB")
    try:
        obj = joblib.load(path)
    except Exception as exc:
        lines.append(f"- **load failed**: {exc!s}")
        return lines
    lines.append(f"- top-level type: `{type(obj).__name__}`")
    lines.extend(_summarise_array(path.name, obj))
    # If it's a list of windows, probe element shapes for date/timestamp columns
    if isinstance(obj, list) and len(obj) and isinstance(obj[0], np.ndarray):
        a0 = obj[0]
        lines.append("")
        lines.append(
            f"#### Episode-window probe (n_episodes={len(obj)}, "
            f"shape[0]={a0.shape})"
        )
        # Look at the first three windows for any datetime-typed columns or
        # plausible date-encoding columns (large integers in the range of
        # YYYYMMDD or unix timestamps).
        for i in range(min(3, len(obj))):
            w = obj[i]
            lines.append(f"- window {i}: shape={w.shape}, dtype={w.dtype}")
            if w.dtype != object:
                # Check whether any column has values that look like dates
                for col in range(w.shape[1] if w.ndim > 1 else 0):
                    vmin = float(np.min(w[:, col])) if w.ndim > 1 else None
                    vmax = float(np.max(w[:, col])) if w.ndim > 1 else None
                    if vmin is None:
                        continue
                    if 19000000 < vmin < 21000000 and 19000000 < vmax < 21000000:
                        lines.append(
                            f"  - column {col}: range [{vmin}, {vmax}] "
                            "**looks like YYYYMMDD**"
                        )
                    elif 1e9 < vmin < 2e10 and 1e9 < vmax < 2e10:
                        lines.append(
                            f"  - column {col}: range [{vmin}, {vmax}] "
                            "**looks like unix epoch (seconds or ms)**"
                        )
            else:
                # Object-typed; show a sample of the first row
                lines.append(f"  - first row sample: {w[0, :min(8, w.shape[1])]!r}")
    return lines


def probe_csv(path: Path, n_rows: int = 5) -> list[str]:
    lines: list[str] = []
    if not path.exists():
        return [f"### `{path.name}` --- **NOT FOUND** at `{path}`"]
    lines.append(f"### `{path.name}`")
    lines.append(f"- path: `{path}`")
    lines.append(f"- size: {path.stat().st_size / 1e6:.2f} MB")
    try:
        df = pd.read_csv(path, nrows=n_rows)
    except Exception as exc:
        lines.append(f"- **load failed**: {exc!s}")
        return lines
    lines.append(f"- columns: {list(df.columns)}")
    lines.append(f"- dtypes: {df.dtypes.to_dict()}")
    lines.append(f"- first {n_rows} rows:")
    lines.append("")
    lines.append("```")
    buf = io.StringIO()
    df.to_string(buf, index=False)
    lines.append(buf.getvalue())
    lines.append("```")
    # Try parsing the Time column to see what date range is covered
    if "Time" in df.columns:
        try:
            full_df = pd.read_csv(path, usecols=["Time"], nrows=200000)
            t = pd.to_datetime(full_df["Time"], errors="coerce")
            lines.append(
                f"- Time range in first 200,000 rows: {t.min()} to {t.max()}"
            )
        except Exception as exc:
            lines.append(f"- Time parse attempt failed: {exc!s}")
    return lines


def main():
    lines: list[str] = []
    lines.append("# Source-data metadata probe")
    lines.append("")
    lines.append(f"Generated at {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append(
        "This file is produced by `qf/probe_metadata.py`.  It introspects the "
        "candidate source-data files for any timestamp or stock-identifier "
        "metadata that the chronological-split harness can use.  It does **not** "
        "modify the source data."
    )
    lines.append("")
    lines.append("## Candidate files")
    lines.append("")

    for label, path in CANDIDATES.items():
        if path.suffix == ".csv":
            lines.extend(probe_csv(path))
        else:
            lines.extend(probe_pickle(path))
        lines.append("")

    lines.append("## What to look for")
    lines.append("")
    lines.append(
        "The chronological-split harness needs a per-episode date that is "
        "comparable across episodes.  Likely sources, in order of preference:"
    )
    lines.append("")
    lines.append(
        "1. A datetime column inside each LOB-window ndarray inside `x_data.pkl`. "
        "If present, it allows the *exact* episode date (and time-of-day) to be "
        "recovered without reaching outside the existing pipeline."
    )
    lines.append(
        "2. A separate metadata structure inside `data_1115.pkl` aligned to the "
        "same episode index as `y_dex.pkl`.  If `data_1115.pkl` is a `dict` "
        "with keys like `episode_id`, `date`, or `stock_id`, this is the cleanest path."
    )
    lines.append(
        "3. The CSRC PDF folder.  Each penalty decision typically lists the "
        "manipulation date(s); these can be parsed into a per-case date table "
        "that is then matched to the labelled episodes via stock id and a date proximity rule."
    )
    lines.append("")
    lines.append(
        "Once the date source is identified, the next step is to fill in "
        "`get_episode_dates()` in `qf/jfqa_chrono.py` and run the chronological harness."
    )

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"[probe] wrote {OUT}")


if __name__ == "__main__":
    main()
