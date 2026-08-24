"""Which simulation to run next.

A Gaussian process reports how unsure it is at every point in the design space,
not just what it predicts. That posterior variance is a map of the model's own
ignorance, and the cheapest way to improve the surrogate is to spend the next
expensive COMSOL run where the map is darkest. This is classical
design-of-computer-experiments active learning, and it is the reason the GP was
worth choosing over a tree ensemble: a random forest has no comparable notion of
where it does not know.

A design point here is a ``(illumination, temperature)`` pair, because that is
what one COMSOL run costs. Aging time is sampled *within* a run at essentially
no extra cost, so a candidate is scored by its mean posterior uncertainty
across the whole aging axis rather than at a single time.

Capability key: ``out_active_learning`` (Tier.VALIDATED) -- the recommendations
are a direct readout of the fitted model, not a heuristic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from psc_twin.capabilities import ENVELOPE, ENVELOPE_LEVELS
from psc_twin.data import TARGETS, DoeBundle, load_doe
from psc_twin.surrogate.scalar_gp import ScalarSurrogateSet

#: Candidate grid resolution per stress axis.
GRID_N = 41

#: Minimum separation from an existing design point, in normalised units where
#: each axis spans 0-1 across the design envelope. Roughly a third of the 6x6
#: grid spacing (0.2), so a recommendation must sit meaningfully away from a
#: simulation that already exists rather than nudging one by a pixel.
EXCLUSION_RADIUS = 0.07

#: Aging times at which uncertainty is averaged for scoring.
SCORE_TIMES_H = (0.0, 100.0, 300.0, 600.0, 1000.0)


@dataclass(frozen=True)
class Recommendation:
    """One proposed COMSOL run."""

    illumination_suns: float
    temperature_c: float
    score: float
    nearest_existing_distance: float
    reason: str


def _normalise(suns: np.ndarray, temp: np.ndarray) -> np.ndarray:
    """Map stress coordinates onto the unit square spanning the envelope."""
    lo_s, hi_s = ENVELOPE["illumination_suns"]
    lo_t, hi_t = ENVELOPE["temperature_c"]
    return np.column_stack(
        [
            (np.asarray(suns, dtype=float) - lo_s) / max(hi_s - lo_s, 1e-9),
            (np.asarray(temp, dtype=float) - lo_t) / max(hi_t - lo_t, 1e-9),
        ]
    )


def uncertainty_surface(
    surrogates: ScalarSurrogateSet,
    grid_n: int = GRID_N,
    targets: tuple[str, ...] = TARGETS,
    score_times_h: tuple[float, ...] = SCORE_TIMES_H,
) -> pd.DataFrame:
    """Mean normalised posterior uncertainty over the stress plane.

    Each target's standard deviation is divided by that target's own spread
    before averaging, so a volt and a milliamp per square centimetre contribute
    comparably instead of whichever happens to have larger units dominating.
    """
    lo_s, hi_s = ENVELOPE["illumination_suns"]
    lo_t, hi_t = ENVELOPE["temperature_c"]
    suns_axis = np.linspace(lo_s, hi_s, grid_n)
    temp_axis = np.linspace(lo_t, hi_t, grid_n)
    grid_s, grid_t = np.meshgrid(suns_axis, temp_axis)
    flat_s, flat_t = grid_s.ravel(), grid_t.ravel()

    accum = np.zeros(flat_s.size, dtype=float)
    used = 0
    for target in targets:
        model = surrogates.models.get(target) if hasattr(surrogates, "models") else None
        if model is None:
            continue
        per_time = []
        for t in score_times_h:
            X = np.column_stack([flat_s, flat_t, np.full(flat_s.size, float(t))])
            _mean, std = model.predict(X, return_std=True)
            per_time.append(std)
        std_mean = np.mean(np.vstack(per_time), axis=0)
        scale = float(np.mean(std_mean))
        accum += std_mean / scale if scale > 0 else std_mean
        used += 1

    score = accum / max(used, 1)
    return pd.DataFrame(
        {
            "aging_light_suns": flat_s,
            "aging_temperature_C": flat_t,
            "uncertainty_score": score,
        }
    )


def existing_design_points(bundle: DoeBundle | None = None) -> np.ndarray:
    """The 36 (illumination, temperature) pairs already simulated."""
    bundle = bundle if bundle is not None else load_doe()
    pairs = (
        bundle.metrics.loc[:, ["aging_light_suns", "aging_temperature_C"]]
        .drop_duplicates()
        .to_numpy(dtype=float)
    )
    return pairs


def recommend_runs(
    surrogates: ScalarSurrogateSet,
    n: int = 5,
    bundle: DoeBundle | None = None,
    grid_n: int = GRID_N,
    exclusion_radius: float = EXCLUSION_RADIUS,
) -> pd.DataFrame:
    """The ``n`` most informative simulations to run next.

    Greedy max-variance selection: take the highest-uncertainty candidate, then
    exclude its neighbourhood before taking the next, so the batch spreads out
    instead of clustering in one hot spot. Points too close to an existing
    simulation are excluded from the start.
    """
    bundle = bundle if bundle is not None else load_doe()
    surface = uncertainty_surface(surrogates, grid_n=grid_n)

    cand = _normalise(surface["aging_light_suns"], surface["aging_temperature_C"])
    existing = _normalise(existing_design_points(bundle)[:, 0], existing_design_points(bundle)[:, 1])

    dist_to_existing = np.min(
        np.linalg.norm(cand[:, None, :] - existing[None, :, :], axis=2), axis=1
    )
    surface = surface.assign(nearest_existing_distance=dist_to_existing)

    available = surface[surface["nearest_existing_distance"] >= exclusion_radius].copy()
    if available.empty:
        available = surface.copy()

    picks: list[Recommendation] = []
    chosen_norm: list[np.ndarray] = []

    for _ in range(int(n)):
        if available.empty:
            break
        if chosen_norm:
            pts = _normalise(available["aging_light_suns"], available["aging_temperature_C"])
            spread = np.min(
                np.linalg.norm(pts[:, None, :] - np.array(chosen_norm)[None, :, :], axis=2),
                axis=1,
            )
            available = available[spread >= exclusion_radius * 2.0]
            if available.empty:
                break

        row = available.loc[available["uncertainty_score"].idxmax()]
        suns = float(row["aging_light_suns"])
        temp = float(row["aging_temperature_C"])
        picks.append(
            Recommendation(
                illumination_suns=suns,
                temperature_c=temp,
                score=float(row["uncertainty_score"]),
                nearest_existing_distance=float(row["nearest_existing_distance"]),
                reason=_reason(suns, temp, float(row["nearest_existing_distance"])),
            )
        )
        chosen_norm.append(_normalise([suns], [temp])[0])
        available = available.drop(index=row.name, errors="ignore")

    return pd.DataFrame(
        [
            {
                "rank": i + 1,
                "aging_light_suns": round(p.illumination_suns, 3),
                "aging_temperature_C": round(p.temperature_c, 1),
                "uncertainty_score": round(p.score, 4),
                "distance_to_nearest_run": round(p.nearest_existing_distance, 3),
                "reason": p.reason,
            }
            for i, p in enumerate(picks)
        ]
    )


def _reason(suns: float, temp: float, distance: float) -> str:
    """Plain-language justification a non-specialist can act on."""
    s_levels = np.array(ENVELOPE_LEVELS["illumination_suns"], dtype=float)
    t_levels = np.array(ENVELOPE_LEVELS["temperature_c"], dtype=float)
    near_s = float(s_levels[np.argmin(np.abs(s_levels - suns))])
    near_t = float(t_levels[np.argmin(np.abs(t_levels - temp))])

    where = []
    if abs(suns - near_s) > 0.05:
        where.append(f"illumination sits between the tested {near_s:g} sun levels")
    if abs(temp - near_t) > 5.0:
        where.append(f"temperature falls between the tested {near_t:.0f} C levels")

    gap = "a sparse gap in the grid" if not where else " and ".join(where)
    return (
        f"The model is least certain here: {gap}, and the nearest existing run is "
        f"{distance:.2f} grid units away. Simulating this point would tighten the "
        "predictions across the surrounding region."
    )


def coverage_summary(bundle: DoeBundle | None = None) -> dict[str, float | int]:
    """How much of the stress plane the existing campaign actually pins down."""
    bundle = bundle if bundle is not None else load_doe()
    pairs = existing_design_points(bundle)
    return {
        "n_design_points": int(len(pairs)),
        "illumination_levels": int(len(np.unique(pairs[:, 0]))),
        "temperature_levels": int(len(np.unique(pairs[:, 1]))),
        "is_full_factorial": bool(
            len(pairs) == len(np.unique(pairs[:, 0])) * len(np.unique(pairs[:, 1]))
        ),
    }
