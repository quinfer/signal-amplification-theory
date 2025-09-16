# Project Progress Snapshot

Date: 2025-09-16

This snapshot records the current state for the Economics Letters submission and the Quantitative Finance (QF) fallback, plus a quick resume checklist.

## Current State

- Letter manuscript (Overleaf main): `main.tex`
  - Clean theorem + short proof sketch, deterrence proposition.
  - Added “Code availability” note and a brief online appendix table summarizing imbalance results (10% and 2% positives).
  - Citations resolved via `\bibliography{references,refs}`.

- Full draft (for later expansion): `paper_full.tex`
  - Staging version of longer paper; appended a “Monte Carlo Sensitivity (Balanced)” section.
  - Includes hooks to input table `tables/qf_amplification_by_rho.tex` and figure `figures/amp_vs_rho.pdf`.

- MC/ROC/PR tooling:
  - `scripts/mc_fast_roc.py`: Fast vectorized ROC AUC + PR AUC + TPR@FPR; supports imbalance; optional bootstrap.
  - `scripts/run_mc_grid.py`: Parallel parameter grid runner; writes JSON per run + CSV summary.
  - `scripts/run_publication_grid.sh`: Publication-grade batch (seeds, bootstrap on smaller cells, imbalance stress tests, extra 1% FPR, mis-specified weights).
  - `scripts/make_qf_tables.py`: Aggregates per-seed summaries into LaTeX table of amplification by rho and sigma.
  - `scripts/plot_amp_vs_rho.py`: Builds amplification vs rho figure (mean ± sd across seeds).

- Generated artifacts:
  - Table: `tables/qf_amplification_by_rho.tex`
  - Figure: `figures/amp_vs_rho.pdf`
  - Results: `results/pub_grid_s*/grid_summary.csv` + per-run JSON/PNGs; imbalance summaries under `results/`.

## Resume Checklist

### Economics Letters (submit-ready)
- [ ] Quick proofread of `main.tex` (abstract tone; keep numeric claims illustrative).
- [ ] Optional: trim references to ~10–15 if requested by EL.
- [ ] Prepare cover letter (draft already outlined in conversation); add suggested referees if desired.
- [ ] Submit `main.tex` via Overleaf/Elsevier; include note on code availability.

### Quantitative Finance (fallback/parallel)
- [ ] Run or verify publication grid:
  ```bash
  bash scripts/run_publication_grid.sh
  ```
- [ ] Regenerate table and figure for sensitivity:
  ```bash
  python scripts/make_qf_tables.py --out tables/qf_amplification_by_rho.tex
  python scripts/plot_amp_vs_rho.py --out figures/amp_vs_rho.pdf
  ```
- [ ] Convert `paper_full.tex` to a clean class (elsarticle or article) and integrate:
  - K-feature generalization and full proofs
  - Equilibrium existence/uniqueness; private vs social thresholds
  - Dynamic/network extensions (appendix)
  - Insert `\input{tables/qf_amplification_by_rho.tex}` and figure reference

## Quick Commands

- Smoke test (balanced, fast):
  ```bash
  python scripts/mc_fast_roc.py --n-pos 100000 --n-neg 100000 --out results/mc_summary.json
  ```
- Imbalance checks (10% / 2% positives):
  ```bash
  python scripts/mc_fast_roc.py --n-pos 100000 --n-neg 900000 --rho 0.3 --sigma-eps 0.5 --out results/imbalance_10pct.json
  python scripts/mc_fast_roc.py --n-pos 20000  --n-neg 980000 --rho 0.3 --sigma-eps 0.5 --out results/imbalance_2pct.json
  ```

## Notes

- Overleaf main document is `main.tex` (EL letter). The longer draft is kept in `paper_full.tex` for QF expansion.
- All scripts are NumPy-based and Apple Silicon friendly; consider setting `OPENBLAS_NUM_THREADS=8` and `OMP_NUM_THREADS=8` for performance.

