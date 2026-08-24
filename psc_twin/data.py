"""Data access for the COMSOL campaign that backs the surrogate.

Two datasets ship with this app:

``data/doe/``   the 6 x 6 full-factorial stress campaign on the baseline p-i-n
                stack. 36 runs, 10 aging times each, 72 voltage points per
                J-V curve. This is what the surrogate is trained on.
``data/seed/``  a separate 12-scenario mechanism-isolation study, browsable in
                the app as COMSOL ground truth but not used for training.

Every loader is pure and cached by the caller; nothing here touches Streamlit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DOE_DIR = DATA_DIR / "doe"
SEED_DIR = DATA_DIR / "seed"

DOE_METRICS = DOE_DIR / "doe_pv_metrics.csv"
DOE_CURVES = DOE_DIR / "doe_jv_curves.csv.gz"
DOE_RUNS = DOE_DIR / "doe_run_index.csv"
SEED_TIMESERIES = SEED_DIR / "comsol_scenario_timeseries.csv"
SEED_LIFETIME = SEED_DIR / "comsol_lifetime_summary.csv"

# The three inputs the surrogate is conditioned on. Order matters: every
# feature matrix in this package is built in exactly this order.
FEATURES = ("aging_light_suns", "aging_temperature_C", "aging_h")

FEATURE_LABELS = {
    "aging_light_suns": "Illumination (suns)",
    "aging_temperature_C": "Temperature (C)",
    "aging_h": "Aging time (h)",
}

# Scalar targets the surrogate predicts, with display metadata.
TARGETS = ("PCE_pct", "Voc_V", "Jsc_mAcm2", "FF", "PCE_retention_pct")

TARGET_LABELS = {
    "PCE_pct": "Power conversion efficiency (%)",
    "Voc_V": "Open-circuit voltage (V)",
    "Jsc_mAcm2": "Short-circuit current density (mA/cm2)",
    "FF": "Fill factor",
    "PCE_retention_pct": "Efficiency retained (%)",
}

TARGET_SHORT = {
    "PCE_pct": "PCE",
    "Voc_V": "Voc",
    "Jsc_mAcm2": "Jsc",
    "FF": "FF",
    "PCE_retention_pct": "Retention",
}

TARGET_UNITS = {
    "PCE_pct": "%",
    "Voc_V": "V",
    "Jsc_mAcm2": "mA/cm2",
    "FF": "",
    "PCE_retention_pct": "%",
}

# Plain-language glossary surfaced as tooltips throughout the UI.
GLOSSARY = {
    "PCE": (
        "Power conversion efficiency: the share of sunlight energy the cell "
        "turns into electricity. A brand-new lab perovskite cell sits around "
        "20-25%."
    ),
    "Voc": (
        "Open-circuit voltage: the voltage the cell produces when no current "
        "is drawn. It falls as defects accumulate."
    ),
    "Jsc": (
        "Short-circuit current density: the current per square centimetre when "
        "the terminals are shorted. It tracks how much light is collected."
    ),
    "FF": (
        "Fill factor: how square the current-voltage curve is, between 0 and 1. "
        "Resistive and recombination losses push it down."
    ),
    "J-V curve": (
        "The current-voltage curve. Sweeping the voltage and recording current "
        "is the standard fingerprint of a solar cell; every performance number "
        "on this page is extracted from it."
    ),
    "T80": (
        "The time until only 80% of the original efficiency remains. The most "
        "widely quoted stability lifetime for perovskite cells."
    ),
    "T90": "The time until 90% of the original efficiency remains.",
    "RUL": (
        "Remaining useful life: how much longer until the cell reaches its "
        "end-of-life threshold, given how far it has already degraded."
    ),
    "Surrogate": (
        "A fast statistical stand-in for a slow physics simulation. It learns "
        "the input-output behaviour of the simulator, then answers in "
        "milliseconds instead of minutes."
    ),
    "Gaussian process": (
        "A model that predicts a value together with an honest error bar. It "
        "reports large uncertainty where it has seen little data, which is why "
        "it suits a small, expensive simulation campaign."
    ),
    "POD": (
        "Proper orthogonal decomposition: finds the handful of characteristic "
        "shapes that combine to reproduce every curve in a family, so a whole "
        "J-V curve can be described by a few numbers."
    ),
    "Design envelope": (
        "The box of conditions the simulations actually covered. Predictions "
        "inside it are validated; outside it the model is guessing, and says so."
    ),
    "COMSOL": (
        "The finite-element physics package used to run the underlying "
        "drift-diffusion device simulations."
    ),
    "Digital twin": (
        "A live virtual copy of a physical object, kept in step with real "
        "conditions so you can ask what happens next."
    ),
}


@dataclass(frozen=True)
class DoeBundle:
    """The training campaign in the shapes the model code needs."""

    metrics: pd.DataFrame          # 360 rows, one per (run, aging time)
    curves: pd.DataFrame           # 25 920 rows, long format
    runs: pd.DataFrame             # 36 rows, one per design point
    voltage: np.ndarray            # shared 72-point voltage grid
    curve_matrix: np.ndarray       # (360, 72) current density
    curve_keys: pd.DataFrame       # (360, 4) run_id + features, aligned to rows

    @property
    def n_runs(self) -> int:
        return int(self.runs["run_id"].nunique())

    @property
    def n_observations(self) -> int:
        return len(self.metrics)

    def feature_matrix(self) -> np.ndarray:
        return self.metrics.loc[:, list(FEATURES)].to_numpy(dtype=float)

    def target_vector(self, target: str) -> np.ndarray:
        return self.metrics[target].to_numpy(dtype=float)

    def groups(self) -> np.ndarray:
        """Run ids, for grouped cross-validation.

        Splitting by run rather than by row is essential: rows from the same
        run share a design point, so a random split would leak.
        """
        return self.metrics["run_id"].to_numpy()


def load_doe(doe_dir: Path | None = None) -> DoeBundle:
    """Load the training campaign and assemble the aligned curve matrix."""
    base = Path(doe_dir) if doe_dir is not None else DOE_DIR
    metrics = pd.read_csv(base / "doe_pv_metrics.csv")
    curves = pd.read_csv(base / "doe_jv_curves.csv.gz")
    runs = pd.read_csv(base / "doe_run_index.csv")

    metrics = metrics.sort_values(["run_id", "aging_h"]).reset_index(drop=True)

    voltage = np.sort(curves["voltage_V"].unique())
    wide = (
        curves.pivot_table(
            index=["run_id", "aging_h"],
            columns="voltage_V",
            values="current_density_mAcm2",
            aggfunc="first",
        )
        .sort_index(axis=1)
        .reset_index()
    )
    wide = wide.sort_values(["run_id", "aging_h"]).reset_index(drop=True)

    keys = metrics.loc[:, ["run_id", *FEATURES]].copy()
    merged = keys.merge(wide, on=["run_id", "aging_h"], how="left", validate="one_to_one")
    curve_matrix = merged.loc[:, list(voltage)].to_numpy(dtype=float)

    if np.isnan(curve_matrix).any():
        missing = int(np.isnan(curve_matrix).any(axis=1).sum())
        raise ValueError(
            f"{missing} of {len(curve_matrix)} J-V curves could not be aligned to "
            "the metrics table; the DOE export is inconsistent."
        )

    return DoeBundle(
        metrics=metrics,
        curves=curves,
        runs=runs,
        voltage=voltage,
        curve_matrix=curve_matrix,
        curve_keys=keys,
        )


def load_seed(seed_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the 12-scenario mechanism-isolation study shown as ground truth."""
    base = Path(seed_dir) if seed_dir is not None else SEED_DIR
    timeseries = pd.read_csv(base / "comsol_scenario_timeseries.csv")
    lifetime = pd.read_csv(base / "comsol_lifetime_summary.csv")
    return timeseries, lifetime


def doe_summary(bundle: DoeBundle) -> dict[str, object]:
    """Headline numbers about the training campaign, for the UI and docs."""
    metrics = bundle.metrics
    return {
        "n_runs": bundle.n_runs,
        "n_observations": bundle.n_observations,
        "n_curves": int(bundle.curve_matrix.shape[0]),
        "n_voltage_points": int(bundle.curve_matrix.shape[1]),
        "illumination_levels": sorted(metrics["aging_light_suns"].unique().tolist()),
        "temperature_levels": sorted(round(v, 2) for v in metrics["aging_temperature_C"].unique()),
        "aging_times_h": sorted(metrics["aging_h"].unique().tolist()),
        "voltage_min_v": float(bundle.voltage.min()),
        "voltage_max_v": float(bundle.voltage.max()),
        "pce_min_pct": float(metrics["PCE_pct"].min()),
        "pce_max_pct": float(metrics["PCE_pct"].max()),
        "retention_min_pct": float(metrics["PCE_retention_pct"].min()),
        "retention_max_pct": float(metrics["PCE_retention_pct"].max()),
    }


def glossary_lookup(term: str) -> str:
    """Tooltip text for a glossary term, or an empty string if unknown."""
    return GLOSSARY.get(term, "")
