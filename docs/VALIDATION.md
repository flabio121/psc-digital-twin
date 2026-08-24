# Validation

Every number here was measured by `python scripts/train_models.py` and is
reproduced in `outputs/tables/`. Nothing is quoted from a training-set fit.

---

## Protocol

**Leave-one-run-out cross-validation over the 36 design points.**

Ten observations share each (illumination, temperature) design point. A random
train/test split would place rows from the same simulated condition on both sides
of the boundary, letting the model interpolate between immediate neighbours of the
very point it is being scored on. The resulting metric would be meaningless.

Each of the 36 folds trains on 35 runs (350 rows) and predicts the ten rows of one
entirely unseen stress condition. For the curve surrogate, the POD basis is
**rebuilt inside every fold** — fitting the basis on all runs and then testing on a
held-out one would leak that run's curve shape into the basis, a subtler form of
the same error.

The test `TestNoLeakage::test_grouped_split_keeps_runs_intact` asserts that no
run ever appears on both sides of a split.

---

## Scalar surrogate — held-out accuracy

360 held-out predictions per target (every observation, predicted while its own
design point was excluded from training).

| Target | MAE | RMSE | R² | 95% coverage | Mean predicted σ | Max abs. error |
|---|---|---|---|---|---|---|
| PCE (%) | 0.0262 | 0.0704 | 0.99889 | 0.911 | 0.0385 | 0.710 |
| V<sub>oc</sub> (V) | 0.00040 | 0.00077 | 0.99929 | 0.925 | 0.00056 | 0.00423 |
| J<sub>sc</sub> (mA cm⁻²) | 0.00518 | 0.01141 | 0.99934 | 0.958 | 0.01195 | 0.1019 |
| Fill factor | 0.00069 | 0.00177 | 0.99885 | 0.911 | 0.00110 | 0.0188 |
| Retention (%) | 0.1160 | 0.3111 | 0.99889 | 0.911 | 0.1703 | 3.138 |

**R² exceeds 0.9988 on every target.** In practical terms, efficiency is predicted
to within 0.026 percentage points on average for a cell whose efficiency ranges
from 11.9% to 22.6% — the surrogate reproduces a condition it has never seen to
roughly the third decimal place.

Wall-clock for the full 36-fold, 5-target cross-validation: **1329 s** (180 GP
fits, 8 optimiser restarts each).

### Calibration

This is the metric that licenses the app to draw error bars at all.

**Mean 95% interval coverage: 0.923** against a nominal 0.95, ranging from 0.911
(PCE, FF, retention) to 0.958 (J<sub>sc</sub>).

The intervals are therefore *slightly over-confident*: the truth falls outside the
95% band about 7.7% of the time rather than 5%. That is close enough to be useful
and is reported as measured rather than tuned. The app's Advanced → Model &
validation tab states this explicitly and, if coverage were to fall below 0.90 on
a retrain, switches to a warning banner telling the user to treat the bands as a
lower bound on uncertainty.

### On the optimiser warnings

Roughly 11% of L-BFGS restarts terminate with an `ABNORMAL` status (165 across
the 1440 restarts in a full cross-validation). This is expected here and does not
indicate a bad fit.

The simulator is deterministic, so the fitted `WhiteKernel` noise term is driven
toward its lower bound (1e-9), which flattens the likelihood surface along that
axis and makes a line search terminate without a clean gradient criterion.
scikit-learn retains the best result across the 8 restarts, and the held-out
accuracy and calibration above are the evidence that the retained fits are sound.

---

## Curve surrogate

### POD truncation

The basis truncates on operating-region error, not on variance share alone. The
comparison that motivates this:

| Modes | Cumulative energy | RMSE (mA cm⁻²) | Worst residual, V ≤ 1.30 V |
|---|---|---|---|
| 3 | 99.9936% | 0.0499 | 1.072 |
| **4 (adopted)** | **99.9998%** | **0.0087** | **0.097** |

Truncating at 99.99% of variance retains 3 modes and leaves a 1.07 mA cm⁻²
residual at ≈1.22 V on the most degraded run — in the knee, which is exactly where
V<sub>oc</sub> and the fill factor are read. The fourth mode carries only 0.006% of
the variance but improves the worst operating-region residual **11-fold** and the
RMSE **5.7-fold**.

Truncation error with the adopted basis: **0.0087 mA cm⁻², or 0.043% of
J<sub>sc</sub>**. That is the floor on achievable curve accuracy before any
regression error.

### Figure-of-merit extraction

Applied to the campaign's own curves, the extraction in
`jv_pod.curve_metrics()` reproduces the campaign's published metric table across
all 360 curves to:

| Quantity | Max abs. deviation |
|---|---|
| PCE | 1.1 × 10⁻¹⁴ |
| V<sub>oc</sub> | 2.2 × 10⁻¹⁶ |
| J<sub>sc</sub> | 0 (exact) |
| Fill factor | 3.3 × 10⁻¹⁶ |

Floating-point agreement. This validates the extraction code and confirms the
fixed 1-sun efficiency reference described in [METHODS.md](METHODS.md) §1.1.

### Held-out curve accuracy

Leave-one-run-out, POD basis rebuilt per fold (741 s across 12 cores):

| Quantity | Value |
|---|---|
| POD truncation RMSE (basis only, no GP) | 0.0087 mA cm⁻² |
| POD truncation NRMSE | 0.043% of J<sub>sc</sub> |
| **End-to-end curve RMSE (POD + GP)** | **0.398 mA cm⁻²** |
| **End-to-end curve NRMSE** | **1.95% of J<sub>sc</sub>** |
| Per-point 95% coverage | 0.969 |

An entire J-V curve for a never-seen stress condition is reproduced to within
about 2% of short-circuit current. The gap between the 0.043% truncation floor
and the 1.95% end-to-end figure is regression error, not basis error — the basis
is nowhere near the limiting factor, so adding more modes would not help. More
*simulations* would.

Per-point interval coverage of 0.969 is mildly **conservative** against the
nominal 0.95, in contrast to the slightly over-confident scalar bands. The curve
bands, if anything, over-state uncertainty.

### Predicting a number directly beats reading it off a predicted curve

A deliberate cross-check, because it is the obvious question and the answer is
not free: is it better to predict PCE directly, or to predict the whole curve and
extract PCE from it?

| Metric | MAE, direct scalar GP | MAE, extracted from predicted curve | Ratio |
|---|---|---|---|
| J<sub>sc</sub> (mA cm⁻²) | 0.00518 | 0.00821 | 1.58× worse |
| V<sub>oc</sub> (V) | 0.00040 | 0.00055 | 1.38× worse |
| Fill factor | 0.00069 | 0.00119 | 1.72× worse |
| PCE (%) | 0.02622 | 0.03773 | 1.44× worse |

**The direct route wins on every figure of merit**, by roughly 1.4–1.7×. This is
expected in hindsight — extraction compounds the curve's regression error through
a nonlinear operation (finding a zero crossing, maximising a product) — but it is
worth measuring rather than assuming.

The practical consequence, and how the app is built: **the scalar surrogate
supplies every number; the curve surrogate supplies curves.** Neither is
redundant, and the app never quietly substitutes one for the other. The curve
model's R² is still 0.998 on all four figures of merit, so it is not inaccurate —
merely second-best for a job the other model does directly.

---

## Learning curve

Models are retrained on 6, 12, 18, 24 and 30 randomly drawn **whole runs** and
scored on the remainder. Sampling whole runs, never individual rows, preserves the
grouping.

This answers the question a reviewer asks first — *is 36 simulations enough?*

Target: retention (%). Four repeats per size, held out on the remaining runs.

| Training runs | MAE | RMSE | 95% coverage |
|---|---|---|---|
| 6 | 2.596 | 5.042 | 0.730 |
| 12 | 0.505 | 0.942 | 0.909 |
| 18 | 0.381 | 0.868 | 0.897 |
| 24 | 0.155 | 0.313 | 0.950 |
| 30 | 0.106 | 0.240 | 0.958 |

**The measured verdict: the campaign is under-sampled.** Held-out RMSE was still
falling by 23.2% over the last step (24 → 30 runs). The curve has not flattened,
so additional simulations at the current spacing would still measurably improve
accuracy.

This is the less flattering of the two possible answers and it is reported as
measured. Two things follow, and both are useful:

1. **The model is nonetheless accurate enough to use.** At the full 36 runs,
   held-out R² exceeds 0.9988 on every target. "Not saturated" is not the same as
   "not good enough" — it means there is headroom, not that the current model is
   unreliable.
2. **It justifies the active-learning workspace concretely.** Since more runs pay,
   *which* runs to run next is a real question with real value, not a decorative
   feature. The maximum-variance recommendations in Advanced → Uncertainty & next
   runs are the answer.

Note also that **calibration improves with data**: coverage climbs from 0.730 at
six runs to 0.958 at thirty, tracking the nominal 0.95 from below. That is
textbook Gaussian-process behaviour and independent evidence that the posterior
is doing its job rather than being tuned.

The verdict is written into `models/model_card.json` under
`validation.learning_curve_verdict` and rendered in the app. Raw data in
`outputs/tables/learning_curve.csv`.

---

## What validation does *not* cover

Stated plainly, because a validation section that only lists successes is a sales
document.

1. **One architecture.** Every number above is conditional on the baseline p-i-n
   stack. Nothing here says anything about an n-i-p or tandem device.
2. **One simulator.** This validates the surrogate against COMSOL, not COMSOL
   against reality. The underlying physics model's own experimental calibration is
   a separate question and out of scope here.
3. **Interpolation, not extrapolation.** Held-out design points sit *inside* the
   convex hull of the training grid. The 0.923 coverage figure says nothing about
   accuracy beyond 1000 h or above 1 sun; those predictions are labelled
   `🟡 Preview` precisely because no held-out evidence supports them.
4. **No mechanism validation.** Mechanism attribution has no ground truth in this
   dataset and is not validated at all.
5. **Climate chaining is unvalidated.** Driving the surrogate through a climate
   schedule and accumulating state month over month has never been checked against
   a COMSOL run driven by the same series.

---

## Reproducing

```bash
python scripts/train_models.py     # writes every table above
python -m pytest tests -q          # 41 invariant tests, incl. no-leakage
```

The full validation is 180 scalar fits plus 144 curve-coefficient fits, each
with 8 optimiser restarts. Measured serial timings: scalar cross-validation
1329 s, J-V cross-validation roughly two CPU-hours, learning curve about a
minute.

`--n-jobs` parallelises both stages and is clearly worth it for the J-V stage
(741 s across 12 cores). It is *not* reliably worth it for the scalar stage: on
Windows, joblib's process spawning and array re-pickling can dominate, and a
measured attempt consumed six times the serial CPU without finishing sooner.
Measure on your own machine rather than assuming.

`--reuse-validation` regenerates `models/model_card.json` from tables already in
`outputs/tables/` without recomputing them, recording in the card that they were
reused.

All estimators use fixed random states. Two runs on the same data produce
identical artifacts and identical metrics.
