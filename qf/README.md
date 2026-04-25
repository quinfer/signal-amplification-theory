# Quantitative Finance submission — reproducibility package

This folder contains the replication code and outputs that accompany the
manuscript **"Signal Amplification and Strategic Deterrence in Market
Surveillance"** (`main_qf.tex` at the repository root).

## What's here

| Path                        | Contents                                                     |
|-----------------------------|--------------------------------------------------------------|
| `mc_amplification.py`       | Vectorised Monte Carlo for the Fisher-Mahalanobis theorem.   |
| `equilibrium.py`            | Best-response, IFT comparative statics, welfare thresholds.  |
| `empirical.py`              | Shenzhen Level-2 panel: horserace, misspecification bound, cross-sectional $\rho_0$. |
| `empirical_probe.py`        | Lightweight probe used during development.                   |
| `make_figures.py`           | Builds the theory and Monte Carlo figures.                   |
| `results/grid_primary.csv`  | Primary Monte Carlo grid ($\rho_0\times\rho_1\times\sigma_\varepsilon$, 3 seeds). |
| `results/horserace.csv`     | Out-of-sample AUC / PR AUC / TPR@5%FPR for 18 rules.         |
| `results/misspec.csv`       | 1,000 random-weight draws and their empirical deflection ratios. |
| `results/misspec_bins.csv`  | Conditional means of the deflection ratio within cos^2 quartiles. |
| `results/ushape_test.csv`   | Quadratic-fit coefficients and bootstrap CIs for the U-shape. |
| `results/xgb_permutation_importance.csv` | Per-feature AUC drop under permutation for XGBoost. |
| `results/fisher_quartile_strat.csv` | Within-Fisher-quartile AUC for Fisher vs. XGBoost. |
| `results/economic_magnitude.csv` | TPR at fixed FPR budgets and implied RMB magnitudes. |
| `results/robustness_tab.tex`  | LaTeX table: parameter-robustness summary for the equilibrium. |
| `results/robustness_full.csv` | Full 44-perturbation sweep of equilibrium and welfare quantities. |
| `results/rho0_cross_section.csv` | Two-feature amplifications across all 66 feature pairs. |
| `results/scalars.csv`       | Key scalars cited in the manuscript.                         |
| `figures/*.pdf`             | Publication-quality PDF figures.                             |

The top-level `build_qf.sh` wires these together and compiles the paper.

## Quick start

```bash
conda env create -f environment.yml      # first time only
conda activate market-manipulation
bash build_qf.sh                         # MC + figures + LaTeX
```

`build_qf.sh latex` rebuilds the PDF without re-running the Monte Carlo.

## What the paper establishes

1. **Mahalanobis amplification theorem** (Thm. 1). The Fisher-optimal linear
   surveillance rule $w^\star=\Sigma_0^{-1}\mu$ strictly dominates every
   singleton rule whenever $\mu$ has $\ge 2$ non-zero components.
2. **Closed-form 2D amplification** (Cor. 1). The deflection
   $d^{\star 2}(\rho_0) = (1-\rho_0^2)^{-1}[\mu_r^2/\sigma_r^2 + \mu_c^2/\sigma_c^2
   - 2\rho_0\mu_r\mu_c/(\sigma_r\sigma_c)]$ is U-shaped in $\rho_0$, with
   an interior minimiser $\rho_0^\star\in[0,1)$ — not monotone, contrary to
   prior informal claims.
3. **Misspecification bound** (Prop. 3). Suboptimal weights $\widehat w$
   retain a fraction $\cos^2\theta$ of optimal deflection, with $\theta$
   measured in the $\Sigma_0$-inner product.
4. **Equilibrium existence and uniqueness** (Prop. 4) under a strict-concavity
   assumption on $(k,\ell,L/\sigma^2)$.
5. **Strategic deterrence** (Prop. 5). $\partial r^\star/\partial\alpha<0$ and
   $\partial c^\star/\partial\beta<0$ via the implicit function theorem.
6. **Private–social wedge** (Prop. 6). The social planner monitors more
   intensely than the private detector whenever $H>L$ and externalities are
   positive; the wedge widens linearly in $H$.

## Key computational results

### Synthetic Monte Carlo
| Quantity                                   | Value (calibration) |
|--------------------------------------------|---------------------|
| Fisher deflection $d^{\star 2}$ at $\rho_0{=}0$ | 9.82          |
| Best-singleton deflection $d^2(e_k)$           | 6.25            |
| AUC amplification, $\rho_0{=}-0.6$, $\sigma_\varepsilon{=}0.25$ | 0.042 |
| Efficiency loss vs fixed weights $(1,0.8)$     | $\le$ 0.002     |
| Deterrence: $\partial r^\star/\partial\alpha$ at equilibrium | $-0.45$       |

### Shenzhen Level-2 empirical (42,072 episodes, $\pi_1 = 4.3\%$)
| Metric                          | Best singleton | Fisher $w^\star$ | Logistic | Random forest | XGBoost |
|---------------------------------|:--:|:--:|:--:|:--:|:--:|
| ROC AUC                         | 0.626 | 0.675 | 0.675 | 0.759 | 0.745 |
| PR AUC                          | 0.088 | 0.101 | 0.100 | 0.155 | 0.149 |
| TPR @ 5% FPR                    | 0.150 | 0.168 | 0.168 | 0.264 | 0.253 |
| $d^{\star 2}/d^2_{\text{best}}$ | —    | $2.18\times$ | — | — | — |
| Misspec. bound slope            | —    | 0.91 | — | — | — |
| Misspec. bound $R^2$            | —    | 0.79 | — | — | — |

## Monte Carlo details

* Primary grid: $\rho_0\in\{-0.6,\ldots,0.8\}$, $\rho_1\in\{0,0.3\}$,
  $\sigma_\varepsilon\in\{0,0.25,0.5\}$, three seeds per cell.
* Per-cell sample: $N_{\mathrm{pos}}=N_{\mathrm{neg}}=2\cdot 10^5$.
* Total synthetic observations: $1.44\cdot 10^8$.
* Full grid runs in ~25 s on Apple Silicon.

## Real-data details

* **Source**: Shenzhen Stock Exchange Level-2 order-book windows labelled
  against 165 CSRC administrative-penalty decisions (2016--2023).
  Proprietary data described in Dai, Quinn, and Kearney (2026, SD-FMM).
* **After cleaning**: 42,072 windowed episodes, 1,803 manipulative.
* **Split**: 70/30 stratified on the label, random-state 42.
* **Features**: 12-dim microstructure vector (spread, imbalance, depth,
  volume intensity, price dynamics).
* Full pipeline runs in ~7 s (feature extraction) + ~90 s (ML training).

## Citation

```bibtex
@article{dai2026signal,
  title  = {Signal Amplification and Strategic Deterrence in Market Surveillance},
  author = {Dai, Yongsheng and Quinn, Barry and Kearney, Fearghal},
  journal = {Quantitative Finance},
  year   = {2026},
  note   = {Submitted}
}
```
