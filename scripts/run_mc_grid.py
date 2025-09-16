import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import product
from typing import List, Tuple


@dataclass
class Job:
    rho: float
    sigma_eps: float
    n_pos: int
    n_neg: int
    bootstrap: int
    seed: int
    out: str
    plot: str | None


def build_job(cmd_py: str, rho: float, sigma_eps: float, n_pos: int, n_neg: int,
              bootstrap: int, seed: int, out_dir: str, do_plot: bool) -> Tuple[Job, List[str]]:
    os.makedirs(out_dir, exist_ok=True)
    base = f"rho{rho:+0.2f}_sigma{sigma_eps:0.2f}_npos{n_pos}_nneg{n_neg}_b{bootstrap}_s{seed}"
    out_json = os.path.join(out_dir, f"grid_{base}.json")
    plot_path = os.path.join(out_dir, f"roc_{base}.png") if do_plot else None
    job = Job(rho, sigma_eps, n_pos, n_neg, bootstrap, seed, out_json, plot_path)
    cmd = [
        cmd_py, "scripts/mc_fast_roc.py",
        "--rho", str(rho),
        "--sigma-eps", str(sigma_eps),
        "--n-pos", str(n_pos),
        "--n-neg", str(n_neg),
        "--bootstrap", str(bootstrap),
        "--seed", str(seed),
        "--out", out_json,
    ]
    if do_plot:
        cmd += ["--plot", plot_path]
    return job, cmd


def run_cmd(job: Job, cmd: List[str]) -> Tuple[Job, int, str]:
    try:
        res = subprocess.run(cmd, check=False, capture_output=True, text=True)
        code = res.returncode
        msg = res.stdout if code == 0 else (res.stdout + "\n" + res.stderr)
        return job, code, msg
    except Exception as e:
        return job, 1, f"Exception: {e}"


def write_summary(out_dir: str) -> str:
    import csv
    rows = []
    for fn in sorted(os.listdir(out_dir)):
        if not fn.endswith(".json"):
            continue
        path = os.path.join(out_dir, fn)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            cfg = data.get("config", {})
            tpr_map = data.get("tpr_at_fpr", {})
            fpr_key = next(iter(tpr_map.keys()), None)
            rows.append({
                "file": fn,
                "rho": cfg.get("rho"),
                "sigma_eps": cfg.get("sigma_eps"),
                "n_pos": cfg.get("n_pos"),
                "n_neg": cfg.get("n_neg"),
                "bootstrap": cfg.get("bootstrap"),
                "auc_r": data.get("auc_r"),
                "auc_c": data.get("auc_c"),
                "auc_comp": data.get("auc_comp"),
                "amplification": data.get("amplification"),
                "fpr": fpr_key,
                "tpr": tpr_map.get(fpr_key) if fpr_key is not None else None,
            })
        except Exception:
            continue
    summary_csv = os.path.join(out_dir, "grid_summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["file"]) 
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return summary_csv


def main():
    ap = argparse.ArgumentParser(description="Run a parameter grid for mc_fast_roc.py in parallel.")
    ap.add_argument("--python", default="python", help="Python executable to run mc_fast_roc.py")
    ap.add_argument("--out-dir", default="results/grid", help="Output directory for JSON/plots")
    ap.add_argument("--rhos", nargs="*", type=float, default=[-0.1, 0.0, 0.1, 0.2, 0.3, 0.4])
    ap.add_argument("--sigmas", nargs="*", type=float, default=[0.3, 0.5, 0.7])
    ap.add_argument("--sizes", nargs="*", type=int, default=[200_000, 500_000], help="Use as both n_pos and n_neg")
    ap.add_argument("--bootstrap-small", type=int, default=100, help="Bootstrap reps for the smaller size")
    ap.add_argument("--bootstrap-large", type=int, default=0, help="Bootstrap reps for the larger size")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("-j", "--jobs", type=int, default=min(8, os.cpu_count() or 4))
    ap.add_argument("--plots", action="store_true", help="Save ROC plots for each run")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    jobs_cmds: List[Tuple[Job, List[str]]] = []
    for rho, sigma, n in product(args.rhos, args.sigmas, args.sizes):
        b = args.bootstrap_small if n == min(args.sizes) else args.bootstrap_large
        job, cmd = build_job(args.python, rho, sigma, n, n, b, args.seed, args.out_dir, args.plots)
        jobs_cmds.append((job, cmd))

    print(f"Planned runs: {len(jobs_cmds)} | parallel jobs: {args.jobs}")
    successes = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(run_cmd, job, cmd) for job, cmd in jobs_cmds]
        for fut in as_completed(futs):
            job, code, msg = fut.result()
            tag = os.path.basename(job.out)
            if code == 0:
                successes += 1
                print(f"OK  | {tag}")
            else:
                print(f"ERR | {tag}\n{msg}")

    summary_csv = write_summary(args.out_dir)
    print(f"Summary written to: {summary_csv}")
    print(f"Completed OK: {successes}/{len(jobs_cmds)}")


if __name__ == "__main__":
    main()

