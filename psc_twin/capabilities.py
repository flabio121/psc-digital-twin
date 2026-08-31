"""Single source of truth for what this app can and cannot do.

Every claim the UI makes about model support resolves through this registry.
The rule the whole product depends on:

    A capability that is not VALIDATED never produces a number.

Tier.VALIDATED  backed by the trained surrogate, inside the COMSOL design
                envelope. Full results, full colour.
Tier.PREVIEW    the surrogate is extrapolating outside its design envelope, or
                the quantity is an interpretation rather than a measurement.
                Results shown, visibly de-emphasised, with a banner.
Tier.PLANNED    no supporting data exists. Controls are disabled and the
                results area is replaced by a roadmap card. Never a number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd


class Tier(str, Enum):
    VALIDATED = "validated"
    PREVIEW = "preview"
    PLANNED = "planned"


TIER_LABEL = {
    Tier.VALIDATED: "Validated",
    Tier.PREVIEW: "Preview",
    Tier.PLANNED: "Planned",
}

TIER_ICON = {
    Tier.VALIDATED: "✅",
    Tier.PREVIEW: "\U0001f7e1",
    Tier.PLANNED: "\U0001f512",
}

TIER_BLURB = {
    Tier.VALIDATED: (
        "Backed by a Gaussian-process surrogate trained on COMSOL "
        "drift-diffusion runs and scored on held-out runs."
    ),
    Tier.PREVIEW: (
        "Shown, but not backed by held-out validation. Either the surrogate is "
        "predicting outside the conditions it was trained on -- in which case its "
        "error bars widen accordingly -- or the quantity is an interpretation "
        "rather than a measurement. Treat the shape as indicative."
    ),
    Tier.PLANNED: (
        "Not yet supported. The simulation campaign that would ground this "
        "has not been run, so the app deliberately shows no numbers."
    ),
}


@dataclass(frozen=True)
class Capability:
    """One thing the app either can or cannot do, and the evidence for it."""

    key: str
    group: str
    label: str
    tier: Tier
    backing: str
    unlocks: str = ""
    version: str = ""
    aliases: tuple[str, ...] = field(default_factory=tuple)

    @property
    def icon(self) -> str:
        return TIER_ICON[self.tier]

    @property
    def is_validated(self) -> bool:
        return self.tier is Tier.VALIDATED

    @property
    def blocks_numbers(self) -> bool:
        """True when this capability must never render a predicted value."""
        return self.tier is Tier.PLANNED

    def badge(self) -> str:
        if self.tier is Tier.VALIDATED:
            return self.label
        suffix = self.version or TIER_LABEL[self.tier]
        return f"{self.label}  {self.icon} {suffix}"


# --------------------------------------------------------------------------
# The design envelope of the COMSOL campaign that backs the trained model.
# A 6 x 6 full factorial in (illumination, temperature), 10 aging times each.
# Everything in the registry below is justified against these bounds.
# --------------------------------------------------------------------------
ENVELOPE = {
    "illumination_suns": (0.01, 1.0),
    "temperature_c": (26.85, 126.85),
    "aging_h": (0.0, 1000.0),
}

ENVELOPE_LEVELS = {
    "illumination_suns": (0.01, 0.2, 0.4, 0.6, 0.8, 1.0),
    "temperature_c": (26.85, 46.85, 66.85, 86.85, 106.85, 126.85),
    "aging_h": (0, 50, 100, 200, 300, 400, 500, 600, 800, 1000),
}

ARCHITECTURE_VALIDATED = "p-i-n (baseline)"


CAPABILITIES: tuple[Capability, ...] = (
    # ---- device architecture -------------------------------------------
    Capability(
        key="arch_pin",
        group="Architecture",
        label=ARCHITECTURE_VALIDATED,
        tier=Tier.VALIDATED,
        backing="36-run COMSOL campaign on the baseline p-i-n stack.",
        aliases=("p-i-n", "pin"),
    ),
    Capability(
        key="arch_nip",
        group="Architecture",
        label="n-i-p",
        tier=Tier.PLANNED,
        version="v2",
        backing="No COMSOL runs exist for an inverted stack.",
        unlocks="Repeat the 6x6 stress campaign with the n-i-p geometry.",
        aliases=("n-i-p", "nip"),
    ),
    Capability(
        key="arch_tandem",
        group="Architecture",
        label="Perovskite/Si tandem",
        tier=Tier.PLANNED,
        version="v3",
        backing="Needs a two-junction model with a recombination junction.",
        unlocks="Build the tandem COMSOL model, then a matched stress campaign.",
        aliases=("tandem",),
    ),
    Capability(
        key="materials_custom",
        group="Device materials",
        label="Custom layer materials",
        tier=Tier.PLANNED,
        version="v2",
        backing=(
            "The Arizona State University thesis COMSOL campaign contains only the "
            "baseline glass / ITO / "
            "NiOx / MeO-2PACz SAM / Cs0.2FA0.8PbI3 / C60 / BCP / Ag physical stack. "
            "The COMSOL geometry groups each transport/interface pair into one domain."
        ),
        unlocks=(
            "Run and validate a material-resolved COMSOL design campaign, then train "
            "the surrogate with material identity and properties as inputs."
        ),
    ),
    Capability(
        key="geometry_thickness",
        group="Device geometry",
        label="Custom layer thickness",
        tier=Tier.PLANNED,
        version="v2",
        backing=(
            "The ASU thesis COMSOL campaign uses one fixed thickness for each "
            "layer; thickness was not a surrogate input."
        ),
        unlocks=(
            "Run a thickness-resolved COMSOL design campaign and retrain the "
            "surrogate with layer thicknesses as inputs."
        ),
    ),
    Capability(
        key="barrier_layers",
        group="Device geometry",
        label="Optional barrier and encapsulation layers",
        tier=Tier.PLANNED,
        version="v2",
        backing=(
            "The ASU thesis COMSOL stack contains no explicit moisture-barrier, "
            "encapsulant, or edge-seal domain."
        ),
        unlocks=(
            "Add barrier transport and interface properties to COMSOL, then validate "
            "material and thickness sweeps against encapsulated devices."
        ),
    ),
    Capability(
        key="internal_interlayers",
        group="Device geometry",
        label="Internal SAM, passivation, and bilayer interfaces",
        tier=Tier.PLANNED,
        version="v2",
        backing=(
            "The ASU thesis baseline includes MeO-2PACz and BCP interface layers, "
            "although the COMSOL geometry groups them with NiOx and C60. The surrogate "
            "does not independently vary their identity, thickness, defect density, "
            "or ion-blocking properties."
        ),
        unlocks=(
            "Add explicit interface domains or boundary conditions to COMSOL, sweep "
            "SAM and bilayer properties, and validate against matched devices."
        ),
    ),
    # ---- stress dimensions ---------------------------------------------
    Capability(
        key="stress_illumination",
        group="Stress",
        label="Illumination 0.01-1.0 suns",
        tier=Tier.VALIDATED,
        backing="Swept axis of the factorial design (6 levels).",
    ),
    Capability(
        key="stress_temperature",
        group="Stress",
        label="Temperature 27-127 C",
        tier=Tier.VALIDATED,
        backing="Swept axis of the factorial design (6 levels).",
    ),
    Capability(
        key="stress_duration",
        group="Stress",
        label="Aging 0-1000 h",
        tier=Tier.VALIDATED,
        backing="10 sampled aging times per run.",
    ),
    Capability(
        key="stress_extrapolation",
        group="Stress",
        label="Conditions outside the tested envelope",
        tier=Tier.PREVIEW,
        backing="The surrogate reverts toward its prior with widening error bars.",
        unlocks="Extend the campaign to the requested corner of stress space.",
    ),
    Capability(
        key="stress_humidity",
        group="Stress",
        label="Relative humidity",
        tier=Tier.PLANNED,
        version="v2",
        backing="No dataset in this project carries a humidity axis.",
        unlocks="Add moisture ingress to the COMSOL model, then sweep RH.",
    ),
    Capability(
        key="stress_bias",
        group="Stress",
        label="Electrical bias / MPP tracking",
        tier=Tier.PLANNED,
        version="v2",
        backing="Aging was simulated at open circuit only.",
        unlocks="Re-run the campaign with MPP and reverse-bias hold protocols.",
    ),
    Capability(
        key="stress_cycling",
        group="Stress",
        label="Thermal cycling (IEC 61215)",
        tier=Tier.PLANNED,
        version="v3",
        backing="Only isothermal soaks were simulated.",
        unlocks="Add time-varying temperature drive and cycle-counting damage.",
    ),
    Capability(
        key="stress_damp_heat",
        group="Stress",
        label="Damp heat 85 C / 85% RH",
        tier=Tier.PLANNED,
        version="v3",
        backing="Requires the humidity axis first.",
        unlocks="Combine the moisture model with the 85/85 protocol.",
    ),
    Capability(
        key="stress_live_weather",
        group="Stress",
        label="Live rooftop weather",
        tier=Tier.PLANNED,
        version="v2",
        backing=(
            "No live weather provider, rooftop thermal model, or measured "
            "site-validation series is connected to this alpha."
        ),
        unlocks=(
            "Connect a weather source, translate ambient conditions into cell "
            "temperature and irradiance, then validate against rooftop telemetry."
        ),
    ),
    # ---- predicted outputs ---------------------------------------------
    Capability(
        key="out_scalars",
        group="Predictions",
        label="PCE, Voc, Jsc, FF trajectories",
        tier=Tier.VALIDATED,
        backing="Independent Gaussian-process surrogate per metric.",
    ),
    Capability(
        key="out_jv",
        group="Predictions",
        label="Full J-V curves",
        tier=Tier.VALIDATED,
        backing="Proper-orthogonal-decomposition modes with GP mode coefficients.",
    ),
    Capability(
        key="out_lifetime",
        group="Predictions",
        label="T80 / T90 / remaining useful life",
        tier=Tier.VALIDATED,
        backing="Derived from the predicted retention trajectory.",
    ),
    Capability(
        key="out_uncertainty",
        group="Predictions",
        label="Calibrated uncertainty bands",
        tier=Tier.VALIDATED,
        backing="Gaussian-process posterior standard deviation.",
    ),
    Capability(
        key="out_active_learning",
        group="Predictions",
        label="Next-simulation recommendations",
        tier=Tier.VALIDATED,
        backing="Maximum-posterior-variance acquisition over the stress space.",
    ),
    Capability(
        key="out_mechanism",
        group="Predictions",
        label="Degradation mechanism attribution",
        tier=Tier.PREVIEW,
        backing="Heuristic weighting; the campaign does not label mechanisms.",
        unlocks="Export per-mechanism state variables from COMSOL.",
    ),
    Capability(
        key="out_inverse",
        group="Predictions",
        label="Inverse diagnosis from measured J-V",
        tier=Tier.PLANNED,
        version="v2",
        backing="The inverse training pipeline is not part of this release.",
        unlocks="Port the multi-rate hysteretic J-V tensor workflow.",
    ),
    # ---- the digital-twin vision ---------------------------------------
    Capability(
        key="twin_cell",
        group="Digital twin",
        label="Single-cell twin",
        tier=Tier.VALIDATED,
        backing="Rendered directly from the surrogate state vector.",
    ),
    Capability(
        key="twin_climate",
        group="Digital twin",
        label="Climate-driven deployment forecast",
        tier=Tier.PREVIEW,
        backing=(
            "Climate archetypes drive the validated surrogate hour by hour, but "
            "the chaining of hourly states is itself unvalidated."
        ),
        unlocks="Validate against a COMSOL run driven by a real weather series.",
    ),
    Capability(
        key="twin_module",
        group="Digital twin",
        label="Module and string scale",
        tier=Tier.PLANNED,
        version="v2",
        backing="No cell-to-module interconnection or mismatch model yet.",
        unlocks="Add series/parallel network solving and mismatch losses.",
    ),
    Capability(
        key="twin_farm",
        group="Digital twin",
        label="Whole solar farm under real weather",
        tier=Tier.PLANNED,
        version="v3",
        backing="The long-horizon vision. Needs module scale plus real TMY data.",
        unlocks="Couple the module model to measured typical-meteorological-year files.",
    ),
    Capability(
        key="twin_fleet",
        group="Digital twin",
        label="Fleet forecasting and maintenance scheduling",
        tier=Tier.PLANNED,
        version="v3",
        backing="Requires farm scale plus field telemetry assimilation.",
        unlocks="Add data assimilation from monitored plant performance.",
    ),
)


BY_KEY = {cap.key: cap for cap in CAPABILITIES}
GROUPS = tuple(dict.fromkeys(cap.group for cap in CAPABILITIES))


def get(key: str) -> Capability:
    return BY_KEY[key]


def resolve(name: str) -> Capability | None:
    """Look a capability up by key, label, or a declared alias."""
    needle = str(name).strip().lower()
    for cap in CAPABILITIES:
        if needle == cap.key.lower() or needle == cap.label.lower():
            return cap
        if any(needle == alias.lower() for alias in cap.aliases):
            return cap
    return None


def in_group(group: str) -> tuple[Capability, ...]:
    return tuple(cap for cap in CAPABILITIES if cap.group == group)


def by_tier(tier: Tier) -> tuple[Capability, ...]:
    return tuple(cap for cap in CAPABILITIES if cap.tier is tier)


def within_envelope(illumination_suns: float, temperature_c: float, aging_h: float) -> bool:
    lo_i, hi_i = ENVELOPE["illumination_suns"]
    lo_t, hi_t = ENVELOPE["temperature_c"]
    lo_a, hi_a = ENVELOPE["aging_h"]
    return (
        lo_i <= float(illumination_suns) <= hi_i
        and lo_t <= float(temperature_c) <= hi_t
        and lo_a <= float(aging_h) <= hi_a
    )


def envelope_excursions(illumination_suns: float, temperature_c: float, aging_h: float) -> list[str]:
    """Plain-language description of how far outside the tested box we are."""
    out: list[str] = []
    checks = (
        ("Illumination", float(illumination_suns), *ENVELOPE["illumination_suns"], "suns"),
        ("Temperature", float(temperature_c), *ENVELOPE["temperature_c"], "C"),
        ("Aging time", float(aging_h), *ENVELOPE["aging_h"], "h"),
    )
    for name, value, low, high, unit in checks:
        if value < low:
            out.append(f"{name} {value:g} {unit} is below the tested {low:g} {unit}.")
        elif value > high:
            out.append(f"{name} {value:g} {unit} is above the tested {high:g} {unit}.")
    return out


def registry_table() -> pd.DataFrame:
    """The whole registry, for the roadmap page and the exported run bundle."""
    return pd.DataFrame(
        [
            {
                "group": cap.group,
                "capability": cap.label,
                "tier": TIER_LABEL[cap.tier],
                "target_version": cap.version or "shipped",
                "backed_by": cap.backing,
                "unlocked_by": cap.unlocks,
            }
            for cap in CAPABILITIES
        ]
    )


def tier_counts() -> dict[str, int]:
    return {
        TIER_LABEL[tier]: len(by_tier(tier))
        for tier in (Tier.VALIDATED, Tier.PREVIEW, Tier.PLANNED)
    }
