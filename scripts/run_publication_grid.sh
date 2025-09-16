#!/usr/bin/env bash
set -euo pipefail

# Publication-grade ROC/AUC grid for QF prep
# - Sweeps rho, noise, and sample size across multiple seeds
# - Bootstrap CIs on smaller size only
# - Adds class-imbalance stress tests and mis-specified weights
# - Adds extra operating point at FPR=1% for highlight cells

# Tunables
JOBS=${JOBS:-8}
SEEDS=(42 777 1234 4242 98765)
RHOS=(-0.1 0.0 0.1 0.2 0.3 0.4)
SIGMAS=(0.3 0.5 0.7)
SIZES=(200000 500000) # used for both n_pos and n_neg
BOOTSTRAP_SMALL=200
BOOTSTRAP_LARGE=0
OUT_ROOT=${OUT_ROOT:-results/pub_grid}
PY=${PY:-python}

# Threading hints for BLAS/OpenMP
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

echo "Running publication-grade grid with ${#SEEDS[@]} seeds; parallel jobs: ${JOBS}"

# Core seeded grids
for s in "${SEEDS[@]}"; do
  OUT_DIR="${OUT_ROOT}_s${s}"
  ${PY} scripts/run_mc_grid.py \
    --python "${PY}" \
    --out-dir "${OUT_DIR}" \
    --rhos "${RHOS[@]}" \
    --sigmas "${SIGMAS[@]}" \
    --sizes "${SIZES[@]}" \
    --bootstrap-small ${BOOTSTRAP_SMALL} \
    --bootstrap-large ${BOOTSTRAP_LARGE} \
    --seed ${s} \
    --plots \
    -j ${JOBS}
done

echo "Core grids complete. Adding imbalance stress tests and extra operating points..."

# Imbalance stress tests (single seed is sufficient)
${PY} scripts/mc_fast_roc.py --n-pos 100000 --n-neg 900000 --rho 0.3 --sigma-eps 0.5 \
  --out ${OUT_ROOT}_imbalance_pi10pct.json
${PY} scripts/mc_fast_roc.py --n-pos 20000 --n-neg 980000 --rho 0.3 --sigma-eps 0.5 \
  --out ${OUT_ROOT}_imbalance_pi2pct.json

# Extra operating point: FPR=1% for highlight cells (balanced 500k)
for rho in 0.1 0.3 0.4; do
  for sigma in 0.3 0.5; do
    ${PY} scripts/mc_fast_roc.py --n-pos 500000 --n-neg 500000 --rho ${rho} --sigma-eps ${sigma} \
      --fpr 0.01 \
      --out ${OUT_ROOT}_op1pct_rho${rho}_sigma${sigma}.json
  done
done

# Mis-specified weights (robustness)
${PY} scripts/mc_fast_roc.py --n-pos 500000 --n-neg 500000 --rho 0.3 --sigma-eps 0.5 \
  --w1 0.8 --w2 0.6 \
  --out ${OUT_ROOT}_misW_rho03_sigma05.json

echo "All runs scheduled/completed. Check ${OUT_ROOT}_s*/grid_summary.csv and JSON files."

