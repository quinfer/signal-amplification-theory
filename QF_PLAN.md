# Fallback Plan: Quantitative Finance Expansion

Objective: Expand the letter into a full theory-forward article suitable for Quantitative Finance (primary) or Applied Mathematical Finance / Mathematics and Financial Economics (alternatives).

## Core Upgrades

- Formalize assumptions and objectives
  - Specify detection objective (Neyman–Pearson power at fixed size; or Bayes risk with Type I/II costs) and prove monotone mapping to the SNR criterion used.
  - State regularity: distributional assumptions on `(r,c)` under manipulation vs. normal regimes; independence/structure for noise `ε`.

- Complete proofs with lemmas
  - Amplification theorem with full detail: characterize the optimizer via generalized eigenvalue/Rayleigh quotient; provide strict-improvement condition and tight lower bound `ε(δ)`.
  - Existence and uniqueness of interior equilibrium under quadratic costs; concavity/supermodularity conditions; comparative statics `∂r*/∂w1<0`, `∂c*/∂w2<0`.
  - Private vs. social threshold result with explicit mapping between detection costs and welfare (and uniqueness conditions).

- Generalize beyond 2 features
  - Extend to `K`-dimensional feature vector: `S = w' x + ε`. Provide conditions on covariance structure `Σ_x` for strict dominance over best singleton and characterize `w*`.
  - Discuss robustness to misspecification (omitted features; correlated noise).

- Dynamic and network extensions (selectively formalized)
  - Dynamic game with adaptive manipulation and weight updates; Bellman formulation for detector; stability or comparative statics over time.
  - Multi-market spillovers: network matrix `Θ`; show when network complementarity strengthens amplification; implications for coordinated surveillance.

## Empirical/Computation Section

- Monte Carlo experiments
  - Synthetic order-flow generation with controllable covariance and noise; ROC/PR curves; power at fixed size; sensitivity to `Σ_x`, `σ^2`.
  - Robustness: starting values, solver choice, regularization/shrinkage on `w`.

- Numerical equilibrium analysis
  - Existence/uniqueness checks; comparative statics in `L, k, ℓ, σ, δ`.
  - Welfare analysis and private vs. social thresholds; quantify externality wedge.

- Replicability
  - Package scripts into a single `qf/` module with CLI; produce seeded runs and figures; archive parameters and outputs.

## Manuscript Structure (Target 25–30 pages)

1. Introduction and contribution (finance/microstructure positioning)
2. Model and detection objective (clear assumptions, mapping to econ costs)
3. Signal amplification theorem (K-dimensional version) + proofs
4. Game-theoretic equilibrium and deterrence + proofs
5. Private vs. social thresholds and welfare
6. Numerical illustrations and Monte Carlo
7. Extensions (dynamic; networks) — concise but formal
8. Conclusion and implications for surveillance design
Appendices: supplementary proofs; additional simulations

## Notation and Code Alignment

- Use `w` for detector weights throughout; reserve `α, β` for structural sensitivities only if needed and map them explicitly to `w`.
- Ensure `main` math and scripts share parameters; provide a single config for reproducibility.

## Target Outlets and Fit

- Primary: Quantitative Finance — theory + method contributions with finance relevance; comfortable with detection/estimation framing and microstructure applications.
- Alternatives: Applied Mathematical Finance; Mathematics and Financial Economics; Journal of Financial Markets (if empirics are expanded).

## Timeline (indicative)

- Weeks 1–2: Assumptions/objective formalization; K-feature theorem and proofs.
- Weeks 3–4: Equilibrium existence/uniqueness; welfare threshold analysis.
- Weeks 5–6: Monte Carlo suite, figures, and robustness; draft manuscript.
- Week 7: Polish, consistency, replication package; submission.

## Risks and Mitigations

- Novelty vs. standard linear detection: Emphasize strategic interaction and deterrence, network effects, and welfare mapping beyond pure SNR.
- Referee expectations on rigor: Provide complete proofs and clear assumptions; move heavy algebra to appendices.
- Scope creep: Keep dynamic/network sections focused with formal statements but short proofs in appendix.

