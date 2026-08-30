# Deploying

## Streamlit Community Cloud

1. Push this repository to GitHub (public).
2. At [share.streamlit.io](https://share.streamlit.io), create an app pointing at
   this repo with **main file** `app.py`.
3. Nothing else to configure. `requirements.txt` and `.python-version` pin the
   environment, and `.streamlit/config.toml` carries the theme.

### Why the trained models are committed

`models/scalar_gp.joblib` and `models/jv_pod.joblib` (~4.5 MB together) are
checked into the repository rather than built on deploy. Training takes several
minutes of CPU and Community Cloud will time out or thrash trying to do it during
app startup, so the app would simply never come up.

The consequence used to be that **exact pins mattered**. They now hurt more than
they help on Streamlit Community Cloud, which picks its own Python version and
ignores `.python-version`: pinning to 3.11-era wheels made `pip` fail to resolve
scikit-learn, and the app launched without it and crashed on import.

`requirements.txt` therefore declares **floors, not exact pins**. The joblib
artifacts were written by scikit-learn 1.5.2 and verified to load and predict
identically under scikit-learn 1.9 / NumPy 2.5 / pandas 3.0 / Python 3.12 --
same result to four decimal places, with only a benign version-mismatch warning
on unpickling. If you upgrade far enough that this stops holding, re-run
`python scripts/train_models.py` and commit the regenerated artifacts.

The app degrades gracefully if the artifacts are missing: every page shows a
"Models not built yet" notice with the command to run, rather than a traceback.

### Resource footprint

| | |
|---|---|
| Repository | ~6.5 MB |
| Data | ~510 KB |
| Model artifacts | ~4.5 MB |
| Dependencies | numpy, pandas, matplotlib, scikit-learn, streamlit |
| Memory at rest | comfortably inside the 1 GB Community Cloud limit |

Predictions are cached with `@st.cache_data` keyed on the input conditions, and
models load once per process behind an `lru_cache`, so repeated interaction does
not refit anything. A full trajectory prediction takes ~4 ms.

### The 3D twin and the CDN

`psc_twin/twin3d.py` loads Three.js from jsDelivr
(`three@0.160.0`) via an import map. That is the only external request the app
makes. If the CDN is unreachable — a locked-down network, an offline demo — the
canvas renders a styled explanatory panel instead of failing silently or showing
a blank iframe.

To run fully offline, vendor `three.module.js` and the three addons used
(`OrbitControls`, `EffectComposer`, `RenderPass`, `UnrealBloomPass`) into
`assets/` and point the import map at the local copies.

---

## Running locally

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
python scripts/train_models.py     # fit + full validation
streamlit run app.py
```

**On `--n-jobs`.** It helps the J-V stage substantially (741 s across 12 cores).
It does **not** reliably help the scalar stage: on Windows, joblib's process
spawning and array re-pickling can cost more than the fits themselves, and a
measured run burned 6x the total CPU of the serial path without finishing
sooner. Measure before relying on it. Serial timings: scalar CV 1329 s,
J-V CV ~2 CPU-hours, learning curve ~1 min.

`--quick` fits in ~2 min for a smoke test (explicitly not publication grade).
`--reuse-validation` rebuilds the model card from existing tables in
`outputs/tables/` without recomputing anything.

`python scripts/train_models.py --quick` fits in a couple of minutes with reduced
optimiser restarts. It is fine for checking that the pipeline runs, and the
console says plainly that the artifacts are **not** publication grade.

### Verifying an install

```bash
python -m pytest tests -q      # 41 invariant tests
python scripts/benchmark.py    # inference latency
```

The test suite covers the things that would make the app dishonest if broken:
that planned capabilities never emit numbers, that no cross-validation split
leaks across a design point, that J-V extraction reproduces the campaign's own
metrics, and that lifetimes are never invented when a threshold is not reached.

---

## Measuring the COMSOL baseline

The speedup headline needs a denominator, and it is the one number this project
cannot measure on an arbitrary machine. Run this once, from anywhere with a
COMSOL licence:

```bash
python scripts/measure_comsol_baseline.py --mph <model.mph> --list-studies
python scripts/measure_comsol_baseline.py --mph <model.mph> --study <tag>
```

`--list-studies` reads the study tags straight out of the `.mph` (which is a ZIP
holding an XML action log), so it needs no licence and takes no time. Pick the
study that corresponds to one aging run with its diagnostic J-V sweeps. If that
study contains a parametric sweep over several conditions, pass
`--parameter-points N` so the per-condition figure is right.

The result lands in `outputs/tables/comsol_baseline.json` and `benchmark.py`
uses it automatically, relabelling the speedup `MEASURED`.

**If it fails on a licence error**, that is a network problem rather than a model
problem — connect to your institution's VPN and retry. The script says so
explicitly and names the host it tried to reach.

---

## Notes for a reviewer or committee member

- **Advanced → Model & validation** has the held-out metrics, the parity plots,
  the calibration coverage, and the learning curve.
- **Advanced → Roadmap** is generated from the capability registry, so it cannot
  drift from what the interface actually does.
- **Show roadmap features** in the sidebar reveals every planned capability greyed
  in place. It is off by default.
- `models/model_card.json` is the full provenance record: data shape, design
  envelope, fitted hyperparameters, every validation metric, library versions,
  and the stated limitations.
