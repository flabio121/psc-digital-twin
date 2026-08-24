# PSC Digital Twin

**A climate-aware Gaussian-process surrogate for perovskite solar cell degradation.**

Perovskite solar cells are cheap and efficient, and they die young. Predicting how
fast one degrades normally means a COMSOL drift-diffusion solve that takes minutes
per condition — far too slow to sweep thousands of climates, designs, and
deployment scenarios.

This app replaces that solve with a surrogate trained on the simulations
themselves. It answers in **milliseconds**, reports **calibrated uncertainty**
with every number, and — the part that matters most — **refuses to answer**
outside the conditions it was actually trained on.

```bash
pip install -r requirements.txt
python scripts/train_models.py
streamlit run app.py
```

---

## What makes this different from a demo

Most surrogate demos will happily extrapolate into nonsense. This one is built
around a single rule:

> **A capability that is not validated never produces a number.**

Every feature in the interface resolves through one capability registry
([`psc_twin/capabilities.py`](psc_twin/capabilities.py)) that assigns it a tier:

| Tier | Meaning |
|---|---|
| ✅ **Validated** | Backed by the trained surrogate inside the tested design envelope, scored on held-out simulations. |
| 🟡 **Preview** | The model is extrapolating, or the quantity is a documented heuristic. Shown de-emphasised, with widened bands. |
| 🔒 **Planned** | No supporting data exists. Controls are disabled and the results area shows a roadmap card. **Never a number.** |

That registry drives the badges, the disabled controls, the roadmap page, and the
exported run bundle. There is no separate roadmap document to drift out of date:
if something is switched off in the UI, it is because of a row in that file.

A sidebar toggle, **Show roadmap features**, reveals planned capabilities greyed
in place. Off by default, so a first-time visitor sees a clean app; on, so a
reviewer can read where the work is going.

---

## The model

**Training data.** A complete 6×6 full-factorial COMSOL campaign on a baseline
p-i-n stack: six illumination levels (0.01–1.0 suns) × six temperatures
(27–127 °C), each aged through ten time points to 1000 h. 36 runs, 360
observations, 360 J-V curves on a shared 72-point voltage grid.

**Scalar metrics.** One Gaussian process per target (PCE, Voc, Jsc, FF,
retention). Kernel: `ConstantKernel × Matérn(ν=2.5, ARD) + WhiteKernel`,
standardised inputs, `normalize_y`. A GP rather than a tree ensemble for three
concrete reasons: 36 design points is the regime GPs are built for; it returns a
calibrated posterior instead of a point estimate; and that posterior is what
makes the active-learning workspace real rather than decorative.

**J-V curves.** Proper orthogonal decomposition (POD/PCA) reduces each 72-point
curve to a handful of mode coefficients, then one GP per coefficient. This
matters: regressing current at each voltage independently ignores that a curve is
a *functional* output and can produce wiggly, non-physical results. POD
reconstructions are smooth by construction.

Truncation stops on **operating-region error**, not variance share. Stopping at
99.99% of variance keeps 3 modes and leaves a 1.07 mA/cm² residual right in the
knee at ≈1.22 V — exactly where Voc and fill factor are read. A 4th mode carries
0.006% of the variance and fixes it, improving worst-case knee error 11× and RMSE
5.7×. Adopted: **4 modes, 99.9998% of curve energy, 0.043% truncation error**.

## Validation

Leave-one-run-out over the 36 design points, everywhere. Ten rows share each
design point, so any ungrouped split leaks the answer across the train/test
boundary and reports a fantasy. For the curve model the POD basis is rebuilt
inside every fold.

| Target | MAE | RMSE | R² | 95% coverage |
|---|---|---|---|---|
| PCE (%) | 0.026 | 0.070 | 0.9989 | 0.911 |
| Voc (V) | 0.0004 | 0.0008 | 0.9993 | 0.925 |
| Jsc (mA/cm²) | 0.005 | 0.011 | 0.9993 | 0.958 |
| Fill factor | 0.0007 | 0.0018 | 0.9989 | 0.911 |
| Retention (%) | 0.116 | 0.311 | 0.9989 | 0.911 |

**R² > 0.9988 on every target**, on conditions the model never saw. Mean coverage
of the 95% predictive interval is **0.923** against a nominal 0.95 — slightly
over-confident, reported as measured rather than tuned. That number is what
licenses the app to draw error bars at all.

Curve extraction reproduces the campaign's own metric table to floating-point
agreement (max deviation 1.1×10⁻¹⁴ on PCE) across all 360 curves.

Full detail in [docs/VALIDATION.md](docs/VALIDATION.md). Regenerate everything
with `python scripts/train_models.py`.

---

## What it will not do

These are switched off in the interface, not merely caveated in a footnote:

- **Other architectures.** Everything was trained on one baseline p-i-n stack.
  n-i-p and tandem are `🔒 Planned` and produce no numbers.
- **Humidity.** No dataset in this project has a humidity axis. Damp-heat and
  85/85 protocols are `🔒 Planned`.
- **Electrical bias / MPP tracking.** Aging was simulated at open circuit only.
- **Thermal cycling.** Only isothermal soaks were simulated.
- **Beyond 1000 h.** The model extrapolates and says so, with widening bands.
- **Mechanism attribution** is `🟡 Preview` — an interpretive overlay, since the
  campaign records terminal J-V behaviour and carries no per-mechanism state.

One protocol detail worth knowing: cells are **aged** at the illumination you
choose, but every diagnostic J-V sweep is taken at **standard 1 sun**. Efficiency
is therefore always referenced to 100 mW/cm², matching the campaign.

---

## Where this is heading

The long-term goal is a digital twin of entire solar installations under real
climates — not one cell in a lab oven, but a farm of modules living through a
measured weather year. The scale ladder:

| Rung | Status |
|---|---|
| Single-cell twin | ✅ Built and validated |
| Climate-driven deployment forecast | 🟡 Preview — archetypes drive the validated surrogate, but chaining states is itself unvalidated |
| Module and string scale | 🔒 v2 — needs interconnection and mismatch modelling |
| Whole solar farm under real weather | 🔒 v3 — needs module scale plus measured TMY data |
| Fleet forecasting and maintenance | 🔒 v3 — needs field telemetry assimilation |

The climate layer ships six representative archetypes (hot desert, humid
subtropical, temperate maritime, continental cold, tropical monsoon, high-altitude
arid). **These are hand-built illustrative profiles, not measured
typical-meteorological-year data**, and are labelled as such in the UI and in
every export.

---

## Layout

```
app.py                      Streamlit entry point
psc_twin/
  capabilities.py           the tier registry — the product's spine
  data.py                   dataset loading and the plain-language glossary
  surrogate/
    scalar_gp.py            per-metric Gaussian processes
    jv_pod.py               POD + GP full-curve surrogate
    predict.py              the single inference API the UI calls
    validation.py           grouped CV, learning curve, calibration
  lifetime.py               T80/T90/RUL, and refusing to invent them
  activelearn.py            max-variance "what to simulate next"
  climate.py                climate archetypes → stress schedules
  heuristic.py              mechanism attribution (clearly unvalidated)
  twin3d.py                 the Three.js digital twin
  plots.py                  every figure, publication-ready
  ui/                       theme, components, pages
scripts/
  train_models.py           fit, validate, write the model card
  benchmark.py              measure inference latency honestly
data/doe/                   the 6×6 campaign (CC BY 4.0)
```

## Reproducing

```bash
python scripts/train_models.py    # fit + full leave-one-run-out validation
python scripts/benchmark.py       # inference latency
python -m pytest tests -q         # test suite
```

`models/model_card.json` records the training data, design envelope, fitted
hyperparameters, every validation metric, library versions, and the stated
limitations.

> **On the speedup figure.** Inference latency is measured. The COMSOL reference
> time is a **user-supplied estimate** declared in `scripts/benchmark.py`, not a
> timing on this machine. Replace it with a timed solve before quoting any
> speedup in a publication; the script labels every derived number accordingly.

## Licence

Code MIT, data CC BY 4.0. See [LICENSE](LICENSE) and
[data/LICENSE-DATA](data/LICENSE-DATA). Citation metadata in
[CITATION.cff](CITATION.cff).
