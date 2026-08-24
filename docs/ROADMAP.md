# Roadmap

The authoritative roadmap is **[`psc_twin/capabilities.py`](../psc_twin/capabilities.py)**,
not this file. That registry drives the badges, the disabled controls, the
in-app roadmap page, and the exported capability table. If a feature is switched
off in the interface, it is because of a row there.

This document gives the narrative the registry cannot.

---

## The destination

A digital twin of entire solar installations under real Earth climates: not one
cell in a lab oven, but a farm of modules living through a measured weather year,
predicting output, degradation, and maintenance need decades ahead.

That is a long way from where this stands today, and the value of the tier system
is that it makes the distance legible instead of hiding it.

---

## The scale ladder

```
  ✅ Single cell        →  🟡 Climate deployment  →  🔒 Module & string
                                                      ↓
                                    🔒 Fleet forecasting  ←  🔒 Solar farm
```

### ✅ Single-cell twin — shipped

A validated Gaussian-process surrogate over illumination, temperature and aging
time for one p-i-n architecture, with calibrated uncertainty, full J-V curve
prediction, lifetime estimation, and a 3D twin whose appearance is driven by the
predicted state rather than by an animation timer.

### 🟡 Climate-driven deployment forecast — preview

Climate archetypes convert to schedules of illumination and cell temperature that
feed the validated surrogate directly. Two honest reasons this is not yet a
forecast:

- Chaining sequential states through a schedule — accumulating damage month over
  month — is itself unvalidated. No COMSOL run driven by a real weather series
  exists to check it against.
- A deployment year is ~8760 h against a 1000 h training horizon, so almost the
  whole schedule is extrapolation. `climate.envelope_report()` enumerates exactly
  which segments.

**What unlocks it:** a COMSOL run driven by a measured weather series, to validate
the chaining against ground truth.

### 🔒 Module and string scale — v2

Cells become modules through series/parallel interconnection, and modules fail in
ways cells do not: current mismatch between cells, hot spots, bypass-diode
activation, interconnect fatigue. None of that is in the physics today, so the
module scope renders as an explicit wireframe with no per-cell numbers.

**What unlocks it:** a network solver over cell models plus a mismatch-loss model.

### 🔒 Whole solar farm under real weather — v3

Module scale, replicated across an array, driven by measured
typical-meteorological-year data rather than archetypes. Adds row-to-row shading,
soiling, and albedo.

**What unlocks it:** module scale, plus ingesting real TMY files.

### 🔒 Fleet forecasting and maintenance — v3

Assimilating field telemetry from monitored plants so the twin tracks the real
asset rather than a nominal one, turning prediction into scheduling.

**What unlocks it:** farm scale, plus a data-assimilation layer.

---

## Physics gaps, in priority order

Each of these is a `🔒 Planned` capability today. The ordering reflects how much
each would widen the model's usefulness per simulation-hour spent.

| Gap | Version | Why it matters | What it needs |
|---|---|---|---|
| **Relative humidity** | v2 | Moisture ingress is among the dominant real-world perovskite failure pathways, and the app currently cannot represent it at all | Moisture transport in the COMSOL model, then an RH sweep |
| **Electrical bias / MPP** | v2 | Deployed cells operate at maximum power point, not open circuit, and ion redistribution is bias-dependent | Re-run the campaign under MPP and reverse-bias hold |
| **Other architectures** | v2–v3 | Every result is currently conditional on one stack | Repeat the 6 × 6 campaign per architecture |
| **Thermal cycling** | v3 | IEC 61215 qualification is cycling-based; only isothermal soaks exist | Time-varying temperature drive, cycle-counting damage |
| **Damp heat 85/85** | v3 | The standard accelerated protocol | Humidity axis first |
| **Inverse diagnosis** | v2 | Infer hidden material state from a measured J-V, closing the loop from experiment back to model | Port the multi-rate hysteretic J-V tensor workflow |

---

## Model work

- **Extend rather than densify.** The learning curve (Advanced → Model &
  validation) reports whether the current design is saturated. If it has
  flattened, further runs at the same spacing buy little and effort is better
  spent widening the envelope.
- **Validated mechanism attribution.** Today's weights are an interpretive
  overlay. Exporting per-mechanism state variables from COMSOL would make
  attribution a measurement.
- **Multi-output GP.** The five scalar targets are modelled independently; a
  coregionalised GP would share strength across them and give a joint posterior.
- **Measure the COMSOL baseline.** The speedup headline currently rests on a
  declared estimate. One timed solve replaces it with a measurement.

---

## Contributing a capability

1. Add a `Capability` row to `psc_twin/capabilities.py` at the honest tier.
2. If it is `PLANNED`, that is enough — the UI will render a roadmap card and
   refuse to produce numbers, with no further work.
3. To promote it, add the data and the validation first, then change the tier.
   Every tier claim should be traceable to a number in `outputs/tables/`.
