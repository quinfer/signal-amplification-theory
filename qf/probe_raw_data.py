"""Permissive listing of the raw-data directory.

The earlier probe only showed files with extensions in a small fixed
set, which could have hidden tick data stored as ``.h5``, ``.parquet``,
``.zip``, ``.feather``, ``.dat``, daily archive files, or files with
no extension at all.  This script lists *everything* under

    market_manipulation_detection/data/

and a few sibling folders, regardless of extension, plus the contents
of any archive files we encounter.  Output is appended to

    qf/results/metadata_probe.md
"""

from __future__ import annotations

import sys
import zipfile
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUT = PROJECT_ROOT / "qf" / "results" / "metadata_probe.md"

ROOTS = [
    Path.home()
    / "Library/CloudStorage/Dropbox/Documents/Research"
    / "market_manipulation_anomaly_detection"
    / "market_manipulation_detection"
    / "data",
    Path.home()
    / "Library/CloudStorage/Dropbox/Documents/Research"
    / "market_manipulation_anomaly_detection"
    / "market_manipulation_detection"
    / "Documents and Data",
    Path.home()
    / "Library/CloudStorage/Dropbox/Documents/Research"
    / "market_manipulation_anomaly_detection"
    / "market_manipulation_detection",
    Path.home()
    / "Library/CloudStorage/Dropbox/Documents/Research"
    / "market_manipulation_anomaly_detection",
]


def _format_size(n: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    s = float(n)
    u = 0
    while s > 1024 and u < len(units) - 1:
        s /= 1024
        u += 1
    return f"{s:7.2f} {units[u]}"


def list_everything(root: Path, max_depth: int = 2) -> list[str]:
    lines: list[str] = []
    if not root.exists():
        return [f"- `{root}` does not exist"]
    lines.append(f"### `{root}`")
    lines.append("")

    entries: list[tuple[Path, int, float, int]] = []  # path, size, mtime, depth
    base_depth = len(root.parts)
    for p in root.rglob("*"):
        try:
            depth = len(p.parts) - base_depth
            if depth > max_depth:
                continue
            if p.is_file():
                entries.append((p, p.stat().st_size, p.stat().st_mtime, depth))
            elif p.is_dir():
                entries.append((p, 0, p.stat().st_mtime, depth))
        except (PermissionError, OSError):
            continue

    if not entries:
        lines.append("- (empty or unreadable)")
        return lines

    # Sort by relative path
    entries.sort(key=lambda e: str(e[0]))
    lines.append("```")
    lines.append(f"{'kind':<5} {'size':>10}  {'mtime':<19}  path")
    for p, sz, mt, depth in entries[:300]:
        try:
            rel = p.relative_to(root)
        except ValueError:
            rel = p
        kind = "D" if p.is_dir() else "F"
        mtime = datetime.fromtimestamp(mt).strftime("%Y-%m-%d %H:%M:%S")
        lines.append(f"{kind:<5} {_format_size(sz):>10}  {mtime:<19}  {rel}")
    if len(entries) > 300:
        lines.append(f"... and {len(entries) - 300} more entries")
    lines.append("```")

    # Group counts by extension
    ext_counts: dict[str, list[int]] = {}
    for p, sz, mt, depth in entries:
        if p.is_file():
            ext = p.suffix.lower() or "(no-ext)"
            ext_counts.setdefault(ext, []).append(sz)

    if ext_counts:
        lines.append("")
        lines.append("Extension summary:")
        lines.append("")
        rows = sorted(ext_counts.items(), key=lambda kv: -sum(kv[1]))
        lines.append("```")
        for ext, sizes in rows:
            total = sum(sizes)
            lines.append(f"  {ext:<10}  n={len(sizes):>5}  total={_format_size(total)}")
        lines.append("```")

    return lines


def peek_inside_archive(zip_path: Path, max_entries: int = 30) -> list[str]:
    lines: list[str] = []
    if not zipfile.is_zipfile(zip_path):
        return []
    lines.append(f"### Contents of `{zip_path.name}`")
    lines.append("")
    try:
        with zipfile.ZipFile(zip_path) as z:
            names = z.namelist()
            lines.append(f"- archive contains {len(names)} entries")
            lines.append("```")
            for name in names[:max_entries]:
                info = z.getinfo(name)
                lines.append(
                    f"  {_format_size(info.file_size):>10}  "
                    f"{info.date_time[0]:04d}-"
                    f"{info.date_time[1]:02d}-"
                    f"{info.date_time[2]:02d}  {name}"
                )
            if len(names) > max_entries:
                lines.append(f"... and {len(names) - max_entries} more entries")
            lines.append("```")
            # Try peeking at the first entry if it looks like a CSV.
            for name in names[:5]:
                if name.lower().endswith(".csv"):
                    try:
                        with z.open(name) as fh:
                            head = fh.read(2048).decode("utf-8", errors="replace")
                        lines.append(f"\nFirst 2 KB of `{name}`:")
                        lines.append("```")
                        lines.append(head)
                        lines.append("```")
                        break
                    except Exception as exc:
                        lines.append(f"  could not peek inside: {exc!s}")
    except Exception as exc:
        lines.append(f"- could not open archive: {exc!s}")
    return lines


def main():
    out_lines: list[str] = []
    out_lines.append("")
    out_lines.append("---")
    out_lines.append("")
    out_lines.append("# Permissive raw-data listing (appended)")
    out_lines.append("")
    out_lines.append(f"Generated at {datetime.now().isoformat(timespec='seconds')}")
    out_lines.append("")

    for root in ROOTS:
        out_lines.extend(list_everything(root))
        out_lines.append("")

    # Find any zip / tar archives anywhere in the project tree and peek inside.
    out_lines.append("## Archive contents")
    out_lines.append("")
    seen: set[Path] = set()
    for root in ROOTS:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p in seen or not p.is_file():
                continue
            if p.suffix.lower() in (".zip",):
                seen.add(p)
                out_lines.extend(peek_inside_archive(p))
                out_lines.append("")

    with open(OUT, "a", encoding="utf-8") as fh:
        fh.write("\n".join(out_lines) + "\n")
    print(f"[probe] appended {len(out_lines)} lines to {OUT}")


if __name__ == "__main__":
    main()
