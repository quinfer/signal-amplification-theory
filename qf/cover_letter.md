# Cover letter — *Quantitative Finance*

Dear Editor,

We are pleased to submit our manuscript *"Signal Amplification and Strategic
Deterrence in Market Surveillance"* for consideration in *Quantitative
Finance*. The paper brings together two literatures that rarely meet: the
statistical theory of optimal composite detection and the game-theoretic
analysis of market manipulation under endogenous enforcement.

## Contribution

Our central result, the **Mahalanobis amplification theorem** (Theorem 1),
characterises when combining order-flow features strictly dominates
single-feature surveillance. The operative object is the covariance of
*benign* order flow, $\Sigma_0$, not the covariance of manipulation
strategies — a subtle point that has led some informal treatments to the
wrong comparative-static prediction. Our closed-form two-feature corollary
shows that the amplification is U-shaped in the benign-flow correlation
$\rho_0$ with an interior minimiser, and our misspecification bound
quantifies the exact efficiency loss from equal-weight or subject-matter-
expert weights rather than the Fisher-optimal rule.

Embedding this detection technology in a two-player quadratic-linear game, we
establish existence, uniqueness, and strict concavity of the manipulator's
interior best response; derive comparative-statics for strategic deterrence
via the implicit function theorem; and quantify the wedge between private
and social detection thresholds.

## Empirical validation

We validate the theory on **42,072 windowed Level-2 order-book episodes from
the Shenzhen Stock Exchange** labelled against **165 CSRC administrative-
penalty decisions** ($\pi_1 = 4.3\%$ conditional on the matching procedure).
On held-out data the Fisher rule beats the best single-feature rule by 4.9
AUC points (0.675 vs. 0.626) and matches a class-balanced logistic regression
to four decimal places, confirming that $w^\star = \Sigma_0^{-1}\mu$ is the
operative *linear* rule on real Chinese microstructure data. The
misspecification *identity* regresses on $\cos^2\theta$ with slope 0.91 and
$R^2 = 0.80$ across 1,000 weight draws, with violations above/below the
identity balanced at 48.5%/51.5% — the signature of symmetric finite-sample
noise around an exact equality. A formal bootstrap test on the 66 feature-
pair amplifications confirms the U-shape predicted by Corollary 1
(curvature 1.54, 95% CI [0.77, 2.49], interior minimiser $\hat\rho_0^\star =
0.38$ [0.28, 0.49]). Tree ensembles (random forest, XGBoost) add 8 AUC
points, which we decompose into a negligible monotone component and a large
interaction residual concentrated in the bottom Fisher-score quartiles —
pinpointing exactly where the linear theory leaves value on the table. At
realistic operating points the Fisher gain alone implies 20–43 additional
detected manipulation episodes per pass through the panel, an order-of-
magnitude RMB 60–130 million in CSRC penalties recovered over the
best-singleton benchmark. A full synthetic Monte Carlo on a 144-cell
parameter grid confirms the theory to within $10^{-3}$. The deterrence
sign in the strategic game is preserved across $\pm 50\%$ perturbations in
every structural parameter.

## Fit with the journal

The paper sits squarely within *Quantitative Finance*'s coverage of market
microstructure, high-frequency trading infrastructure, and quantitative
regulation. It complements recent empirical work on surveillance-based market
quality (Cumming-Johan-Li, Comerton-Forde-Putniņš) and machine-learning
manipulation detection (James-Leung-Prokhorov) by providing the Fisher-
geometric foundation that those classifiers implicitly use. The welfare
analysis speaks to the policy question of whether exchange-operated
surveillance internalises social externalities — a question sharpened by the
Xiong-Chen-Zhang (2024) evidence that information-infrastructure upgrades
reduce manipulation.

## Reviewer suggestions

With your permission we suggest the following referees, all of whom work at
the intersection of market microstructure and statistical detection theory:

- **Talis Putniņš** (UTS Business School), market-manipulation surveys and cryptocurrency manipulation.
- **Douglas Cumming** (Florida Atlantic University), exchange trading rules and market surveillance.
- **Yacine Aït-Sahalia** (Princeton), high-frequency market making and microstructure noise.
- **Robert James** (University of Sydney), machine-learning detection of illegal trading.
- **Mykola Khomyn** (University of Technology Sydney), algorithmic cancellation behaviour.

## Compliance

The manuscript is original, has not been published or submitted elsewhere,
and all three authors have approved the submission. The paper is
methodologically self-contained; all code and Monte Carlo outputs that support
the results are available in the accompanying `qf/` replication package and
can be rerun end-to-end in under one minute on a modern workstation.

We thank you in advance for considering this submission.

Sincerely,

Yongsheng Dai · Barry Quinn · Fearghal Kearney
