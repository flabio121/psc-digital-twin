"""Interpretive overlay: which degradation mechanism plausibly dominates.

This module is deliberately small, and deliberately *not* a second predictive
engine.

An earlier design had a separate Arrhenius fallback take over outside the
design envelope. That was dropped: a Gaussian process already extrapolates
honestly, reverting toward its prior mean with a widening posterior, and a
second engine could only ever disagree with the first at the boundary. One
model with truthful error bars beats two models with a seam in the middle.
So extrapolation is handled by the surrogate itself and merely *labelled*
Tier.PREVIEW.

What remains here is mechanism attribution, which the surrogate genuinely
cannot supply. The COMSOL campaign records terminal behaviour -- J-V curves and
the figures of merit read off them -- and carries no per-mechanism state
variable. Nothing in the data says "this loss was ion migration". The weights
below are therefore an *interpretation* built from published stress-response
behaviour of perovskite devices, not a fit to anything in this repository.

Capability key: ``out_mechanism`` (Tier.PREVIEW). Never present these numbers
as measured, and never let them drive a prediction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from psc_twin.capabilities import ENVELOPE

#: Boltzmann constant in eV/K.
K_B_EV = 8.617333262e-5

#: Reference temperature for the Arrhenius term, in kelvin (~26.85 C, the
#: coolest corner of the design).
T_REF_K = 300.0

#: Activation energy used for the thermal term, in eV. Reported activation
#: energies for thermally driven degradation of mixed-cation perovskite
#: absorbers cluster in the 0.6-1.0 eV range; 0.8 eV sits mid-range. This is a
#: literature-typical value used to *order* mechanisms, not a fitted parameter.
E_ACTIVATION_EV = 0.8


@dataclass(frozen=True)
class Mechanism:
    """One candidate degradation pathway and how it responds to stress."""

    key: str
    label: str
    #: Weight given to the Arrhenius (thermal) driver, 0-1.
    thermal: float
    #: Weight given to the photo-dose (illumination) driver, 0-1.
    photo: float
    #: Weight given to accumulated time, 0-1.
    temporal: float
    evidence: str


MECHANISMS: tuple[Mechanism, ...] = (
    Mechanism(
        key="ion_migration",
        label="Mobile-ion redistribution",
        thermal=0.75,
        photo=0.55,
        temporal=0.35,
        evidence=(
            "Halide vacancies become mobile well below 100 C and redistribute "
            "under illumination, screening the built-in field. Shows up first "
            "as fill-factor loss at moderate stress."
        ),
    ),
    Mechanism(
        key="trap_recombination",
        label="Trap-assisted recombination",
        thermal=0.60,
        photo=0.70,
        temporal=0.65,
        evidence=(
            "Photo-generated defect states accumulate roughly with light dose "
            "and depress open-circuit voltage without changing collected current "
            "very much."
        ),
    ),
    Mechanism(
        key="interface_degradation",
        label="Interface and contact degradation",
        thermal=0.90,
        photo=0.30,
        temporal=0.55,
        evidence=(
            "Transport-layer and electrode interfaces are the most strongly "
            "thermally activated pathway, raising series resistance and "
            "flattening the curve near the maximum power point."
        ),
    ),
    Mechanism(
        key="shunt_formation",
        label="Shunt path formation",
        thermal=0.45,
        photo=0.40,
        temporal=0.85,
        evidence=(
            "Pinholes and filamentary shorts develop slowly and largely "
            "independently of stress level, so their share grows with elapsed "
            "time rather than with intensity."
        ),
    ),
)


def _arrhenius(temperature_c: float) -> float:
    """Relative thermal acceleration versus the coolest tested corner."""
    t_k = float(temperature_c) + 273.15
    t_k = max(t_k, 1.0)
    return float(np.exp(-E_ACTIVATION_EV / K_B_EV * (1.0 / t_k - 1.0 / T_REF_K)))


def _normalised_drivers(
    illumination_suns: float,
    temperature_c: float,
    aging_h: float,
) -> tuple[float, float, float]:
    """Map raw conditions onto three 0-1 stress drivers.

    Each driver is scaled against the design envelope so the attribution is
    stable and interpretable, rather than being dominated by raw magnitudes.
    """
    lo_i, hi_i = ENVELOPE["illumination_suns"]
    lo_a, hi_a = ENVELOPE["aging_h"]

    thermal_raw = _arrhenius(temperature_c)
    thermal_max = _arrhenius(ENVELOPE["temperature_c"][1])
    thermal = float(np.clip(np.log1p(thermal_raw) / max(np.log1p(thermal_max), 1e-9), 0.0, 1.5))

    photo = float(np.clip((float(illumination_suns) - lo_i) / max(hi_i - lo_i, 1e-9), 0.0, 1.5))
    temporal = float(np.clip(float(aging_h) / max(hi_a, 1e-9), 0.0, 1.5))
    return thermal, photo, temporal


def mechanism_attribution(
    illumination_suns: float,
    temperature_c: float,
    aging_h: float,
) -> pd.DataFrame:
    """Relative plausibility of each degradation pathway at these conditions.

    Returns weights that sum to 1. They describe which mechanism is *most
    consistent* with the stress applied; they are not a decomposition of the
    predicted efficiency loss, and no row of the training data supports them.
    """
    thermal, photo, temporal = _normalised_drivers(illumination_suns, temperature_c, aging_h)

    raw = np.array(
        [
            m.thermal * thermal + m.photo * photo + m.temporal * temporal
            for m in MECHANISMS
        ],
        dtype=float,
    )
    raw = np.maximum(raw, 1e-9)
    weights = raw / raw.sum()

    frame = pd.DataFrame(
        {
            "mechanism": [m.key for m in MECHANISMS],
            "label": [m.label for m in MECHANISMS],
            "weight": weights,
            "evidence": [m.evidence for m in MECHANISMS],
        }
    ).sort_values("weight", ascending=False, ignore_index=True)

    # Carried on every row so the flag survives a CSV export or a copy-paste
    # into a slide, where the surrounding caveat would be lost.
    frame["is_heuristic"] = True
    frame["validated"] = False
    return frame


def dominant_mechanism(
    illumination_suns: float,
    temperature_c: float,
    aging_h: float,
) -> str:
    """Label of the highest-weighted pathway. Presentation only."""
    return str(mechanism_attribution(illumination_suns, temperature_c, aging_h)["label"].iloc[0])


def driver_table(
    illumination_suns: float,
    temperature_c: float,
    aging_h: float,
) -> pd.DataFrame:
    """The three normalised stress drivers, for showing the reasoning."""
    thermal, photo, temporal = _normalised_drivers(illumination_suns, temperature_c, aging_h)
    return pd.DataFrame(
        [
            {
                "driver": "Thermal (Arrhenius)",
                "value": thermal,
                "basis": f"Ea = {E_ACTIVATION_EV} eV, relative to {T_REF_K - 273.15:.0f} C",
            },
            {
                "driver": "Photo dose",
                "value": photo,
                "basis": "Illumination scaled across the tested 0.01-1.0 sun range",
            },
            {
                "driver": "Elapsed time",
                "value": temporal,
                "basis": "Aging time scaled against the 1000 h tested horizon",
            },
        ]
    )


DISCLAIMER = (
    "Mechanism weights are an interpretive overlay. The simulation campaign "
    "records terminal J-V behaviour only and contains no per-mechanism state "
    "variable, so nothing here has been validated against data. Use them to "
    "frame a hypothesis, never as a measurement."
)
