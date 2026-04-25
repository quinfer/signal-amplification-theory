#!/usr/bin/env bash
# Build the QF submission PDF end-to-end.
# Usage:
#   bash build_qf.sh            # full pipeline (MC + empirical + figures + LaTeX)
#   bash build_qf.sh mc         # Monte Carlo grid only
#   bash build_qf.sh empirical  # Shenzhen empirical pipeline only
#   bash build_qf.sh figs       # theory/MC figures only
#   bash build_qf.sh latex      # LaTeX only
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

STAGE="${1:-all}"

if [[ "$STAGE" == "mc" || "$STAGE" == "all" ]]; then
  echo "[qf] Running Monte Carlo grid..."
  python qf/mc_amplification.py --mode grid \
    --n-pos 200000 --n-neg 200000 \
    --rho0-grid -0.6 -0.4 -0.2 0.0 0.2 0.4 0.6 0.8 \
    --rho1-grid 0.0 0.3 \
    --sigma-eps-grid 0.0 0.25 0.5 \
    --seeds 42 2024 7 \
    --out-csv qf/results/grid_primary.csv
fi

if [[ "$STAGE" == "empirical" || "$STAGE" == "all" ]]; then
  echo "[qf] Running Shenzhen empirical pipeline..."
  python qf/empirical.py
fi

if [[ "$STAGE" == "figs" || "$STAGE" == "all" ]]; then
  echo "[qf] Generating theory/MC figures..."
  ( cd qf && python make_figures.py )
fi

if [[ "$STAGE" == "latex" || "$STAGE" == "all" ]]; then
  echo "[qf] Compiling main_qf.tex..."
  pdflatex -interaction=nonstopmode main_qf.tex >/dev/null
  bibtex main_qf >/dev/null || true
  pdflatex -interaction=nonstopmode main_qf.tex >/dev/null
  pdflatex -interaction=nonstopmode main_qf.tex >/dev/null
  echo "[qf] PDF at $ROOT/main_qf.pdf"
fi

echo "[qf] Done."
