import argparse
import glob
import os
import csv
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


def load_rows(paths):
    rows = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                r = csv.DictReader(f)
                rows.extend(list(r))
        except FileNotFoundError:
            continue
    return rows


def to_float(x):
    try:
        return float(x)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description="Plot amplification vs rho with lines for sigma.")
    ap.add_argument("--glob", default="results/pub_grid_s*/grid_summary.csv",
                    help="Glob for per-seed CSV summaries")
    ap.add_argument("--size", type=int, default=200000,
                    help="Use rows with n_pos == n_neg == SIZE")
    ap.add_argument("--out", default="figures/amp_vs_rho.pdf",
                    help="Output PDF path")
    args = ap.parse_args()

    rows = load_rows(glob.glob(args.glob))
    if not rows:
        raise SystemExit("No grid summaries found. Run publication grid first.")

    # aggregate amplification by rho,sigma across seeds
    by_sigma = defaultdict(lambda: defaultdict(list))
    for row in rows:
        n_pos = int(float(row.get("n_pos", 0)))
        n_neg = int(float(row.get("n_neg", 0)))
        if n_pos != args.size or n_neg != args.size:
            continue
        rho = to_float(row.get("rho"))
        sigma = to_float(row.get("sigma_eps"))
        amp = to_float(row.get("amplification"))
        if rho is None or sigma is None or amp is None:
            continue
        by_sigma[sigma][rho].append(amp)

    sigmas = sorted(by_sigma.keys())
    colors = {0.3: "#1f77b4", 0.5: "#ff7f0e", 0.7: "#2ca02c"}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.figure(figsize=(6, 4))
    for s in sigmas:
        rho_vals = sorted(by_sigma[s].keys())
        amps = np.array([np.mean(by_sigma[s][r]) for r in rho_vals])
        sds = np.array([np.std(by_sigma[s][r], ddof=1) if len(by_sigma[s][r]) > 1 else 0.0 for r in rho_vals])
        c = colors.get(s, None)
        plt.plot(rho_vals, amps, label=f"sigma={s:.1f}", lw=2, color=c)
        plt.fill_between(rho_vals, amps - sds, amps + sds, color=c, alpha=0.2)

    plt.xlabel(r"$\rho$ (Covariance between $r$ and $c$)")
    plt.ylabel("Amplification: AUC(comp) − max AUC(single)")
    plt.title("Amplification vs rho (balanced; mean ± sd across seeds)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out)
    print(f"Saved figure to: {args.out}")


if __name__ == "__main__":
    main()

