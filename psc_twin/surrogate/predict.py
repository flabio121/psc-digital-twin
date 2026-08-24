"""The single inference entry point for the whole application.

Every page calls :func:`predict` and nothing else. All the routing logic --
which capability was requested, whether the conditions fall inside the design
envelope, which tier the answer carries, whether an answer may be given at all
-- lives here so no UI page ever has to decide it. A rule enforced in one place
is a rule; a rule re-implemented on five pages is a bug waiting to happen.

The hard product rule is implemented by :class:`PlannedCapabilityError`: a
Tier.PLANNED capability raises rather than returning numbers. Callers are
expected to catch it and render a roadmap card.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

from psc_twin import heuristic, lifetime as lifetime_mod
from psc_twin.capabilities import (
    ARCHITECTURE_VALIDATED,
    ENVELOPE,
    Capability,
    Tier,
    envelope_excursions,
    resolve,
)
from psc_twin.surrogate import jv_pod, scalar_gp


class PlannedCapabilityError(RuntimeError):
    """Raised when a caller asks for something the app must not answer.

    Carries the capability so the UI can render the roadmap card without
    re-resolving anything.
    """

    def __init__(self, capability: Capability):
        self.capability = capability
        super().__init__(
            f"{capability.label} is planned for {capability.version or 'a future release'} "
            f"and produces no numbers. {capability.backing}"
        )


class ModelsMissingError(RuntimeError):
    """Raised when the trained artifacts are not on disk."""


# --------------------------------------------------------------------------
# model loading, once per process
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _load_models() -> tuple[scalar_gp.ScalarSurrogateSet, jv_pod.JvSurrogate]:
    try:
        return scalar_gp.load(), jv_pod.load()
    except FileNotFoundError as exc:
        raise ModelsMissingError(
            "Trained surrogate artifacts were not found in models/. "
            "Run:  python scripts/train_models.py"
        ) from exc


def models_available() -> bool:
    """Cheap check so a page can show a helpful message instead of a traceback."""
    return scalar_gp.DEFAULT_ARTIFACT.exists() and jv_pod.DEFAULT_ARTIFACT.exists()


def clear_cache() -> None:
    """Drop the cached models, so a retrain is picked up without a restart."""
    _load_models.cache_clear()


def format_lifetime(estimate: lifetime_mod.LifetimeEstimate) -> str:
    """Render a lifetime for display without ever inventing a number.

    A threshold the trajectory never reaches has no numeric answer, and this
    says so in words rather than substituting the horizon or a projection the
    caller did not ask for.
    """
    if estimate.value_h is None or not np.isfinite(estimate.value_h):
        return "beyond the simulated window"
    suffix = " (projected)" if estimate.is_projection else ""
    return f"{estimate.value_h:,.0f} h{suffix}"


# --------------------------------------------------------------------------
# result container
# --------------------------------------------------------------------------
@dataclass
class TwinPrediction:
    """Everything a results page needs, resolved and tiered."""

    architecture: str
    illumination_suns: float
    temperature_c: float
    horizon_h: float

    trajectories: pd.DataFrame
    curve: jv_pod.PredictedCurve | None
    curve_reference: jv_pod.PredictedCurve | None
    mechanisms: pd.DataFrame

    t80: lifetime_mod.LifetimeEstimate
    t90: lifetime_mod.LifetimeEstimate

    tier: Tier
    excursions: list[str] = field(default_factory=list)
    engine: str = "surrogate"
    latency_ms: float = 0.0

    @property
    def in_envelope(self) -> bool:
        return not self.excursions

    @property
    def final(self) -> pd.Series:
        return self.trajectories.iloc[-1]

    def at_time(self, aging_h: float) -> pd.Series:
        """Nearest predicted row to a requested aging time."""
        idx = int(np.argmin(np.abs(self.trajectories["aging_h"].to_numpy() - float(aging_h))))
        return self.trajectories.iloc[idx]

    def summary_frame(self) -> pd.DataFrame:
        """Headline numbers, formatted for display and export."""
        final = self.final
        rows = [
            {"quantity": "Starting efficiency", "value": f"{self.trajectories['PCE_pct'].iloc[0]:.2f} %"},
            {"quantity": f"Efficiency at {self.horizon_h:,.0f} h", "value": f"{final['PCE_pct']:.2f} %"},
            {"quantity": "Efficiency retained", "value": f"{final['PCE_retention_pct']:.1f} %"},
            {"quantity": "T90", "value": format_lifetime(self.t90)},
            {"quantity": "T80", "value": format_lifetime(self.t80)},
            {"quantity": "Confidence tier", "value": self.tier.value},
        ]
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# the entry point
# --------------------------------------------------------------------------
def predict(
    illumination_suns: float,
    temperature_c: float,
    horizon_h: float = 1000.0,
    architecture: str = ARCHITECTURE_VALIDATED,
    n_points: int = 61,
    curve_at_h: float | None = None,
) -> TwinPrediction:
    """Run the surrogate and package a complete, tiered answer.

    Raises :class:`PlannedCapabilityError` when ``architecture`` names a
    capability the app has no data for, and :class:`ModelsMissingError` when the
    trained artifacts are absent.
    """
    cap = resolve(architecture)
    if cap is None:
        raise ValueError(f"Unknown architecture {architecture!r}")
    if cap.blocks_numbers:
        raise PlannedCapabilityError(cap)

    scalars, curves = _load_models()

    start = time.perf_counter()

    aging = np.linspace(0.0, float(horizon_h), int(n_points))
    traj = scalar_gp.trajectory_grid(scalars, illumination_suns, temperature_c, aging)

    target_h = float(horizon_h if curve_at_h is None else curve_at_h)
    curve = curves.predict_curve(illumination_suns, temperature_c, target_h)
    reference = curves.predict_curve(illumination_suns, temperature_c, 0.0)

    latency_ms = (time.perf_counter() - start) * 1000.0

    retention = traj["PCE_retention_pct"].to_numpy(dtype=float)
    retention_std = (
        traj["PCE_retention_pct_std"].to_numpy(dtype=float)
        if "PCE_retention_pct_std" in traj.columns
        else None
    )
    t80 = lifetime_mod.lifetime_with_uncertainty(aging, retention, retention_std, threshold_pct=80.0) \
        if retention_std is not None else lifetime_mod.t80(aging, retention)
    t90 = lifetime_mod.lifetime_with_uncertainty(aging, retention, retention_std, threshold_pct=90.0) \
        if retention_std is not None else lifetime_mod.t90(aging, retention)

    excursions = envelope_excursions(illumination_suns, temperature_c, horizon_h)
    tier = Tier.VALIDATED if not excursions else Tier.PREVIEW
    engine = "surrogate" if not excursions else "surrogate (extrapolating)"

    mechanisms = heuristic.mechanism_attribution(
        illumination_suns, temperature_c, min(float(horizon_h), ENVELOPE["aging_h"][1])
    )

    return TwinPrediction(
        architecture=cap.label,
        illumination_suns=float(illumination_suns),
        temperature_c=float(temperature_c),
        horizon_h=float(horizon_h),
        trajectories=traj,
        curve=curve,
        curve_reference=reference,
        mechanisms=mechanisms,
        t80=t80,
        t90=t90,
        tier=tier,
        excursions=excursions,
        engine=engine,
        latency_ms=latency_ms,
    )


def predict_curve_family(
    illumination_suns: float,
    temperature_c: float,
    aging_times: list[float],
) -> list[jv_pod.PredictedCurve]:
    """A fan of J-V curves across aging times, for the degradation view."""
    _scalars, curves = _load_models()
    return curves.predict_family(illumination_suns, temperature_c, aging_times)


# --------------------------------------------------------------------------
# timing, for the speedup headline
# --------------------------------------------------------------------------
def measure_latency(n_repeats: int = 50, n_points: int = 61) -> dict[str, float]:
    """Median and spread of single-prediction latency, after a warm-up.

    Reported as a distribution rather than a single best case, because a demo
    that quotes its fastest run is quoting noise.
    """
    scalars, curves = _load_models()

    # Warm up: first call pays import and BLAS setup costs.
    scalar_gp.trajectory_grid(scalars, 0.5, 70.0, np.linspace(0, 1000, n_points))
    curves.predict_curve(0.5, 70.0, 500.0)

    aging = np.linspace(0.0, 1000.0, n_points)
    rng = np.random.default_rng(0)
    samples_traj, samples_curve, samples_point = [], [], []

    for _ in range(int(n_repeats)):
        suns = float(rng.uniform(*ENVELOPE["illumination_suns"]))
        temp = float(rng.uniform(*ENVELOPE["temperature_c"]))

        t0 = time.perf_counter()
        scalar_gp.trajectory_grid(scalars, suns, temp, aging)
        samples_traj.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        curves.predict_curve(suns, temp, 500.0)
        samples_curve.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        scalar_gp.trajectory_grid(scalars, suns, temp, np.array([500.0]))
        samples_point.append((time.perf_counter() - t0) * 1000.0)

    def stats(values: list[float], name: str) -> dict[str, float]:
        arr = np.asarray(values)
        return {
            f"{name}_median_ms": float(np.median(arr)),
            f"{name}_p25_ms": float(np.percentile(arr, 25)),
            f"{name}_p75_ms": float(np.percentile(arr, 75)),
        }

    out: dict[str, float] = {"n_repeats": float(n_repeats), "n_points": float(n_points)}
    out.update(stats(samples_point, "single_point"))
    out.update(stats(samples_traj, "trajectory"))
    out.update(stats(samples_curve, "jv_curve"))
    return out
