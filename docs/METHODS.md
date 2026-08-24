# Methods

This document is written to be lifted into a manuscript. Every claim it makes is
reproduced by `python scripts/train_models.py`, which regenerates the tables in
`outputs/tables/` and the provenance record in `models/model_card.json`.

---

## 1. Simulation campaign

A one-dimensional drift-diffusion model of a baseline p-i-n perovskite solar cell
(glass / ITO / hole-transport layer / mixed-cation perovskite absorber /
electron-transport layer / silver) was solved in COMSOL Multiphysics, with
depth-dependent Beer–Lambert photogeneration, mobile-ion transport, and coupled
degradation state variables.

The campaign is a **complete 6 × 6 full factorial** in two aging stressors:

| Factor | Levels |
|---|---|
| Aging illumination | 0.01, 0.2, 0.4, 0.6, 0.8, 1.0 suns |
| Aging temperature | 26.85, 46.85, 66.85, 86.85, 106.85, 126.85 °C (300–400 K) |

Each of the 36 runs was sampled at ten aging times — 0, 50, 100, 200, 300, 400,
500, 600, 800, 1000 h — giving **360 observations** and **360 J-V curves** on a
shared 72-point voltage grid spanning 0 to 1.5 V.

### 1.1 A protocol detail that governs every efficiency number

Cells are **aged** at the illumination and temperature above, but every
diagnostic J-V sweep is taken at **standard 1-sun illumination**
(`scan_light_suns = 1.0` for all 36 runs). Power conversion efficiency is
therefore referenced to a fixed 100 mW cm⁻² throughout, never to the aging
illumination.

This is not a cosmetic distinction. Dividing by the aging illumination would
inflate reported efficiency by up to a factor of 100 at the 0.01-sun corner. The
implementation asserts the fixed reference
(`psc_twin/surrogate/jv_pod.ONE_SUN_MW_CM2`) and a regression test pins it.

### 1.2 Response range

Retention at 1000 h spans 52.4% to 100%, and absolute efficiency spans 11.9% to
22.6%. The design therefore covers benign storage through to severe combined
light-and-heat stress, rather than probing a narrow band.

---

## 2. Scalar surrogate

### 2.1 Model

For each target *y* ∈ {PCE, V<sub>oc</sub>, J<sub>sc</sub>, FF, retention} an
independent Gaussian-process regressor is fitted over the three inputs
(illumination, temperature, aging time).

**Kernel.** `ConstantKernel × Matérn(ν = 5/2, ARD) + WhiteKernel`

- **Matérn ν = 5/2** rather than the squared-exponential limit. A degradation
  trajectory is smooth but not analytic; the RBF kernel assumes infinite
  differentiability and over-smooths the early-time knee.
- **Automatic relevance determination** — a separate length scale per input — so
  the fit reports which stressor the response is most sensitive to rather than
  assuming isotropy across variables with incomparable units.
- **WhiteKernel** absorbs solver jitter. The simulator is deterministic, so the
  fitted noise term collapses toward its lower bound, which is the expected and
  correct behaviour.

Inputs are standardised; outputs use `normalize_y`. Hyperparameters are found by
maximising the log marginal likelihood with 8 restarts from random initialisations
(`random_state = 0`), making the fit deterministic and reproducible.

### 2.2 Input warping

Degradation against time is strongly non-linear near *t* = 0. Both a raw aging-time
input and a `log1p` warp were evaluated under identical grouped cross-validation,
and the better-scoring option was adopted. The choice is recorded in
`GPConfig.time_warp` and the comparison is reproduced by
`scalar_gp.compare_time_warps()`.

### 2.3 Why a Gaussian process rather than a tree ensemble

Three reasons, all consequential rather than stylistic:

1. **Sample regime.** GP inference is O(n³); 360 observations is precisely where
   GPs are comfortable and where ensembles are data-starved.
2. **Calibrated uncertainty.** The GP returns a posterior, not a point estimate,
   and Section 4 reports the measured coverage of that posterior. The
   application's error bars are only defensible because this number exists.
3. **Graceful extrapolation.** Outside the design envelope a GP reverts toward its
   prior mean with a widening posterior. A random forest returns a flat,
   confident-looking value with no signal that it has left its training domain —
   the single most dangerous failure mode for an interactive tool.

The third point also supplies the active-learning acquisition function (Section 5)
for free.

---

## 3. Full-curve surrogate

### 3.1 Proper orthogonal decomposition

A J-V curve is a functional output. Regressing current independently at each of
the 72 voltage points discards that structure and admits non-physical
reconstructions — curves that wiggle, cross zero twice, or bend the wrong way
near the maximum power point.

Instead, the 360 × 72 curve matrix is centred on its mean curve and decomposed by
singular value decomposition. Every curve is then represented by a handful of
coefficients on an orthonormal empirical basis, and every reconstruction is a
linear combination of smooth modes — smooth and physically plausible by
construction.

### 3.2 Truncation criterion

Truncation uses two stopping rules and takes whichever is stricter: cumulative
energy ≥ 99.99%, **and** a cap of 0.25 mA cm⁻² on the worst pointwise residual
inside the operating region (V ≤ 1.30 V).

The second rule is necessary, and measurably so. Energy alone retains 3 modes,
which reproduces the bulk of every curve but leaves a **1.07 mA cm⁻² residual at
≈1.22 V** on the most degraded run — in the knee, which is exactly where V<sub>oc</sub>
and the fill factor are read. A fourth mode carries only 0.006% of the variance
yet corrects the knee:

| Modes | Cumulative energy | RMSE (mA cm⁻²) | Worst operating residual |
|---|---|---|---|
| 3 | 99.9936% | 0.0499 | 1.072 |
| **4 (adopted)** | **99.9998%** | **0.0087** | **0.097** |

Adopting the error-aware criterion improves RMSE 5.7× and the worst operating
residual 11×. Variance share is a poor proxy for accuracy where the physics is
extracted.

### 3.3 Coefficient regression and uncertainty propagation

One Gaussian process per retained mode coefficient, using the same kernel
construction as the scalar surrogate so the two models cannot drift apart.

A per-point band is propagated as σ²(V) = Σᵢ σᵢ² · φᵢ(V)², treating the mode
coefficients as independent. POD decorrelates coefficients over the *training*
set, not over the GP posterior, so this is an indicative band rather than a
rigorous joint credible region — stated wherever it is drawn.

### 3.4 Figure-of-merit extraction

J<sub>sc</sub> is interpolated at V = 0; V<sub>oc</sub> at the first downward zero
crossing; the maximum power point by direct search over V·J; FF = P<sub>max</sub>/(J<sub>sc</sub>V<sub>oc</sub>);
and efficiency against the fixed 1-sun reference of Section 1.1.

A curve that never crosses zero yields **NaN**, not the last voltage on the grid.
A failed extraction must stay visible.

**Verification.** Applied to the campaign's own curves, this extraction reproduces
the campaign's published metric table to within 1.1 × 10⁻¹⁴ (PCE), 2.2 × 10⁻¹⁶
(V<sub>oc</sub>), 0 (J<sub>sc</sub>) and 3.3 × 10⁻¹⁶ (FF) across all 360 curves —
floating-point agreement, confirming both the extraction and the efficiency
reference.

---

## 4. Validation

### 4.1 Grouped cross-validation

**Leave-one-run-out over the 36 design points, everywhere.**

Ten observations share each (illumination, temperature) design point. A random
train/test split would place rows from the same simulated condition on both
sides, letting the model interpolate between neighbours of the very point it is
being scored on. The resulting metric would be a fantasy.

Each of the 36 folds therefore trains on 35 runs (350 rows) and predicts the ten
rows of one entirely unseen stress condition. The same rule applies to the curve
surrogate, where **the POD basis is rebuilt inside every fold** — fitting the
basis on all runs and testing on a held-out one would leak that run's curve shape
into the basis, a subtler form of the same error.

### 4.2 Metrics

Per target: MAE, RMSE, R², and **coverage of the 95% predictive interval** — the
fraction of held-out observations falling inside the model's own error bars.
Coverage is reported exactly as measured. It is the only evidence that the bands
drawn throughout the application mean anything, and an uncertainty estimate
nobody has checked is decoration.

### 4.3 Learning curve

Models are retrained on 6, 12, 18, 24 and 30 randomly drawn *whole runs*, five
repeats each, and scored on the remainder. This answers the question a reviewer
will ask directly — *is 36 simulations enough?* — and both possible answers are
publishable. A flattened curve shows the design is saturated; a still-falling
curve shows the campaign is under-sampled and justifies the active-learning
workspace.

Sampling whole runs, never individual rows, preserves the grouping.

---

## 5. Active learning

A design point costs one COMSOL run, so candidates are (illumination,
temperature) pairs; aging time is sampled within a run at negligible extra cost.
Each candidate on a dense grid is scored by its mean posterior standard deviation
across the aging axis, normalised per target so that volts and mA cm⁻² contribute
comparably.

Selection is greedy maximum-variance with two exclusions: candidates within a
normalised radius of an existing design point, and, after each pick, the
neighbourhood of that pick — so a batch spreads across the space instead of
clustering in one hot spot.

---

## 6. Deliberate limitations

These are enforced in software, not merely noted. Requests that would cross them
raise rather than return numbers.

1. **A single architecture.** All 36 runs use the baseline p-i-n stack. n-i-p and
   tandem devices are unsupported.
2. **No humidity axis.** No dataset in this project varies relative humidity, so
   damp-heat and 85/85 protocols cannot be modelled.
3. **No electrical bias axis.** Aging was simulated at open circuit; MPP-tracked
   and reverse-bias aging are out of scope.
4. **Isothermal only.** Thermal cycling requires a time-varying temperature drive
   and cycle-counting damage accumulation.
5. **1000-hour horizon.** Longer predictions are extrapolation, flagged as such
   with widening intervals.
6. **Mechanism attribution is unvalidated.** The campaign records terminal J-V
   behaviour and carries no per-mechanism state variable, so mechanism weights are
   an interpretive overlay built from published stress-response behaviour.
7. **Climate profiles are archetypes.** The shipped climates are hand-built
   representative profiles, not measured typical-meteorological-year data, and
   chaining sequential states through a climate schedule is itself unvalidated.

## 7. Reproducibility

```bash
python scripts/train_models.py   # fit, validate, write the model card
python scripts/benchmark.py      # inference latency
python -m pytest tests -q        # invariants, including the no-leakage checks
```

All estimators carry fixed random states; two runs on the same data produce
identical artifacts and identical metrics. `models/model_card.json` records the
training data shape, design envelope, kernel specification, fitted
hyperparameters, every validation metric, library versions, and the limitations
above.

**On the speedup figure.** Surrogate inference latency is measured. The COMSOL
reference time is a declared constant in `scripts/benchmark.py`, not a timing on
the reporting machine, and every derived speedup is labelled as resting on an
estimated denominator. Replace it with a timed solve before publication.
