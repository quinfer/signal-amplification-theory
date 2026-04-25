"""Parameter robustness for the strategic equilibrium in Section 4.

For each parameter in the baseline calibration, perturb by {-50%, -25%,
+25%, +50%} and report (a) the sign of partial r*/partial alpha and
partial c*/partial beta (the deterrence signs), (b) the magnitudes of the
cross-effects, and (c) the sign of the private-social welfare wedge.

Outputs a LaTeX table at qf/results/robustness_tab.tex.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from equilibrium import (
    GameParams, best_response, comparative_statics,
    private_welfare, social_welfare, optimise_threshold,
)

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)

PARAMS_TO_PERTURB = [
    "k", "ell", "L", "alpha", "beta", "sigma", "H", "F", "xi", "zeta", "cD",
]
PERTURBATIONS = [-0.50, -0.25, +0.25, +0.50]


def _evaluate(p: GameParams) -> dict:
    r_star, c_star, _ = best_response(p)
    cs = comparative_statics(p)
    t_priv, w_priv = optimise_threshold(private_welfare, p)
    t_soc, w_soc = optimise_threshold(social_welfare, p)
    return {
        "r_star": r_star,
        "c_star": c_star,
        "dr_dalpha": cs["dr_dalpha"],
        "dc_dbeta": cs["dc_dbeta"],
        "dr_dbeta": cs["dr_dbeta"],
        "dc_dalpha": cs["dc_dalpha"],
        "tau_priv": t_priv,
        "tau_soc": t_soc,
        "wedge": t_priv - t_soc,
    }


def run():
    base = GameParams()
    rows = [{"param": "baseline", "mult": 1.0, "value": None, **_evaluate(base)}]
    baseline = rows[0]

    for name in PARAMS_TO_PERTURB:
        base_val = getattr(base, name)
        for mult in PERTURBATIONS:
            new_val = base_val * (1.0 + mult)
            d = asdict(base)
            d[name] = new_val
            p = GameParams(**d)
            rows.append({"param": name, "mult": mult, "value": new_val,
                         **_evaluate(p)})

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "robustness_full.csv", index=False)

    # Summary across all perturbations: count sign flips
    non_base = df[df["param"] != "baseline"]
    sign_summary = {
        "n_runs": int(len(non_base)),
        "deterrence_own_effect_sign_flips": int(
            ((non_base["dr_dalpha"] > 0) | (non_base["dc_dbeta"] > 0)).sum()
        ),
        "wedge_sign_flips": int((non_base["wedge"] * baseline["wedge"] < 0).sum()),
    }
    pd.Series(sign_summary).to_csv(RESULTS / "robustness_summary.csv")

    # Condensed LaTeX table: one row per parameter with min/max dr/dalpha
    grouped = []
    for name in PARAMS_TO_PERTURB:
        sub = df[df["param"] == name]
        grouped.append({
            "param": name,
            "dr_dalpha_min": sub["dr_dalpha"].min(),
            "dr_dalpha_max": sub["dr_dalpha"].max(),
            "dc_dbeta_min": sub["dc_dbeta"].min(),
            "dc_dbeta_max": sub["dc_dbeta"].max(),
            "wedge_min": sub["wedge"].min(),
            "wedge_max": sub["wedge"].max(),
        })
    gdf = pd.DataFrame(grouped)
    gdf.to_csv(RESULTS / "robustness_grouped.csv", index=False)

    # LaTeX table
    sym = {
        "k": r"$k$", "ell": r"$\ell$", "L": r"$L$",
        "alpha": r"$\alpha$", "beta": r"$\beta$", "sigma": r"$\sigma$",
        "H": r"$H$", "F": r"$F$", "xi": r"$\xi$", "zeta": r"$\zeta$", "cD": r"$c_D$",
    }
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Parameter & \multicolumn{2}{c}{$\partial r^\star/\partial\alpha$} & "
        r"\multicolumn{2}{c}{$\partial c^\star/\partial\beta$} & "
        r"\multicolumn{2}{c}{$\tau^{\rm priv}-\tau^{\rm soc}$} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r" & min & max & min & max & min & max \\",
        r"\midrule",
        f"baseline & \\multicolumn{{2}}{{c}}{{${baseline['dr_dalpha']:.3f}$}}"
        f" & \\multicolumn{{2}}{{c}}{{${baseline['dc_dbeta']:.3f}$}}"
        f" & \\multicolumn{{2}}{{c}}{{${baseline['wedge']:.3f}$}} \\\\",
        r"\midrule",
    ]
    for row in grouped:
        lines.append(
            f"{sym[row['param']]} & {row['dr_dalpha_min']:.3f} & {row['dr_dalpha_max']:.3f} "
            f"& {row['dc_dbeta_min']:.3f} & {row['dc_dbeta_max']:.3f} "
            f"& {row['wedge_min']:.3f} & {row['wedge_max']:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(RESULTS / "robustness_tab.tex", "w") as f:
        f.write("\n".join(lines))

    print("=== baseline ===")
    for k, v in baseline.items():
        print(f"  {k:12s} = {v}")
    print(f"\n=== sign summary across {sign_summary['n_runs']} perturbations ===")
    print(f"  deterrence own-effect sign flips (r wrt alpha OR c wrt beta): "
          f"{sign_summary['deterrence_own_effect_sign_flips']}")
    print(f"  private-social wedge sign flips: {sign_summary['wedge_sign_flips']}")

    return df, gdf, sign_summary


if __name__ == "__main__":
    run()
