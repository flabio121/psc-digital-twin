"""The 3D digital twin: a perovskite cell you can look at while it degrades.

This module turns a *predicted device state* into a self-contained WebGL scene.
It is deliberately split in two so the interesting half stays testable:

``build_twin_config``  pure function, plain numbers in, plain dict out. No
                       Streamlit, no model imports, no I/O. Every visual
                       quantity the renderer will use is decided here, in
                       Python, where it can be unit-tested.
``build_twin_html``    string templating. Takes that dict and produces one
                       HTML document suitable for ``st.components.v1.html``.

--------------------------------------------------------------------------
VISUAL HONESTY -- the rule this file exists to enforce
--------------------------------------------------------------------------
Every appearance of damage is a **pure function of the predicted state that
was passed in**. There is no decay timer anywhere in the renderer. Concretely:

* absorber bleaching, defect speckle density, contact/interface darkening and
  edge-ingress width are all derived in :func:`_appearance` from the predicted
  retention (and fill factor) at the requested aging time, and are then frozen
  into the config as constants;
* the JavaScript reads those constants once and applies them. It never
  advances them;
* the only time-varying quantities in the browser are camera azimuth (orbit
  auto-rotate) and a small bounded shimmer on the defect cloud and light
  beams, neither of which changes any degradation quantity;
* when a *timeline* is supplied (see :func:`build_twin_timeline`) the renderer
  may animate, but only by interpolating between states it was handed. It
  cannot extrapolate past the last frame and it invents nothing in between
  beyond a linear blend of two given states.

If the surrogate says the cell is fine, the picture shows a cell that is fine.
That is the entire difference between a visualisation and a cartoon.

--------------------------------------------------------------------------
Scale scopes
--------------------------------------------------------------------------
``SCOPE_CELL``    capability ``twin_cell``   -> VALIDATED. Full render, numbers.
``SCOPE_MODULE``  capability ``twin_module`` -> PLANNED v2. Wireframe, no numbers.
``SCOPE_FARM``    capability ``twin_farm``   -> PLANNED v3. Wireframe, no numbers.

A PLANNED scope can never produce a number, and the renderer enforces that by
never being given any: ``config["metrics"]`` is ``None`` for those scopes.
The same treatment kicks in at cell scope if the resolved tier is PLANNED
(for example an n-i-p architecture), because the rule is about the tier, not
about the geometry.

--------------------------------------------------------------------------
Robustness
--------------------------------------------------------------------------
* Missing, ``None``, non-numeric, NaN and infinite inputs are replaced by the
  documented defaults in :data:`DEFAULTS` and then clamped to the ranges in
  :data:`CLAMPS`. Nothing in this module raises for bad input.
* If the Three.js CDN is unreachable, or WebGL is unavailable, or the module
  script throws, the page shows a styled explanatory panel *with a static SVG
  cross-section of the same stack*, rendered from the same config. It never
  shows a blank iframe or a bare stack trace.
* The layout is fluid from 700 px to 1200 px. No pointer lock, no audio, no
  network calls other than the two Three.js module files.
"""

from __future__ import annotations

import html
import json
import math
import re
from typing import Any, Iterable, Mapping, Sequence

from psc_twin.capabilities import (
    ARCHITECTURE_VALIDATED,
    TIER_ICON,
    TIER_LABEL,
    Capability,
    Tier,
)
from psc_twin.capabilities import get as get_capability

__all__ = [
    "SCOPES",
    "SCOPE_CELL",
    "SCOPE_MODULE",
    "SCOPE_FARM",
    "SCOPE_LABEL",
    "SCOPE_CAPABILITY",
    "DEFAULTS",
    "CLAMPS",
    "PIN_STACK",
    "THREE_MODULE_URL",
    "THREE_ADDONS_URL",
    "build_twin_config",
    "build_twin_timeline",
    "build_twin_html",
]


# --------------------------------------------------------------------------
# palette
#
# These are the literals from psc_twin/ui/theme.py. They are duplicated rather
# than imported because theme.py imports Streamlit and matplotlib, and this
# module must stay importable (and unit-testable) without either. The __main__
# block below asserts the two stay in sync whenever theme.py can be imported.
# --------------------------------------------------------------------------
PALETTE: dict[str, str] = {
    "bg": "#F7F9FC",
    "surface": "#FFFFFF",
    "surfaceSunk": "#F1F5F9",
    "border": "#E3E8EF",
    "borderStrong": "#CBD5E1",
    "text": "#0F172A",
    "textMuted": "#5B6B82",
    "textFaint": "#94A3B8",
    "primary": "#2563EB",
    "primarySoft": "#EFF6FF",
    "validated": "#059669",
    "preview": "#D97706",
    "planned": "#64748B",
    "validatedBg": "#ECFDF5",
    "previewBg": "#FFFBEB",
    "plannedBg": "#F1F5F9",
    "validatedBorder": "#A7F3D0",
    "previewBorder": "#FDE68A",
    "plannedBorder": "#CBD5E1",
}

_TIER_KEY = {Tier.VALIDATED: "validated", Tier.PREVIEW: "preview", Tier.PLANNED: "planned"}


# --------------------------------------------------------------------------
# scopes
# --------------------------------------------------------------------------
SCOPE_CELL = "cell"
SCOPE_MODULE = "module"
SCOPE_FARM = "farm"
SCOPES: tuple[str, ...] = (SCOPE_CELL, SCOPE_MODULE, SCOPE_FARM)

SCOPE_LABEL = {
    SCOPE_CELL: "Cell",
    SCOPE_MODULE: "Module",
    SCOPE_FARM: "Farm",
}

SCOPE_TITLE = {
    SCOPE_CELL: "Single-cell twin",
    SCOPE_MODULE: "Module and string scale",
    SCOPE_FARM: "Whole solar farm",
}

SCOPE_CAPABILITY = {
    SCOPE_CELL: "twin_cell",
    SCOPE_MODULE: "twin_module",
    SCOPE_FARM: "twin_farm",
}

SCOPE_BLURB = {
    SCOPE_CELL: (
        "The p-i-n stack rendered from the surrogate state vector. Bleaching, "
        "defect speckle, interface darkening and edge ingress are all functions "
        "of the predicted retention shown below."
    ),
    SCOPE_MODULE: (
        "What a module of interconnected cells would look like. Drawn as a "
        "wireframe on purpose: no cell-to-module interconnection or mismatch "
        "model exists yet, so there is nothing to colour in and no number to show."
    ),
    SCOPE_FARM: (
        "The long-horizon ambition: an array of modules under real weather. "
        "Drawn as a wireframe on purpose. It needs module scale first, then "
        "measured meteorological data."
    ),
}


# --------------------------------------------------------------------------
# input handling
#
# DEFAULTS is the documented substitute for a missing / None / NaN / infinite
# input. CLAMPS is the range each value is then squeezed into. Together they
# guarantee build_twin_config never raises and never emits a non-finite number
# into the JSON payload.
# --------------------------------------------------------------------------
DEFAULTS: dict[str, float | str] = {
    "architecture": ARCHITECTURE_VALIDATED,
    "illumination_suns": 1.0,      # 1 sun, the standard test condition
    "temperature_c": 25.0,         # room temperature
    "aging_h": 0.0,                # fresh device
    "horizon_h": 1000.0,           # the campaign's longest soak
    "pce_pct": 20.0,               # a plausible fresh perovskite cell
    "retention_pct": 100.0,        # assume undegraded rather than degraded
    "voc_v": 1.10,
    "jsc_macm2": 23.0,
    "ff": 0.78,
    "pce_sd": 0.0,                 # unknown uncertainty is shown as none,
    "retention_sd": 0.0,           # never as a fabricated band
    "voc_sd": 0.0,
    "jsc_sd": 0.0,
    "ff_sd": 0.0,
}

CLAMPS: dict[str, tuple[float, float]] = {
    "illumination_suns": (0.0, 10.0),
    "temperature_c": (-50.0, 300.0),
    "aging_h": (0.0, 1.0e6),
    "horizon_h": (1.0, 1.0e6),
    "pce_pct": (0.0, 35.0),
    "retention_pct": (0.0, 110.0),   # >100 allows for light-soak burn-in gain
    "voc_v": (0.0, 2.5),
    "jsc_macm2": (0.0, 50.0),
    "ff": (0.0, 1.0),
    "pce_sd": (0.0, 20.0),
    "retention_sd": (0.0, 60.0),
    "voc_sd": (0.0, 1.0),
    "jsc_sd": (0.0, 25.0),
    "ff_sd": (0.0, 0.5),
}

#: Fill factor of a healthy baseline cell. Used only as the reference point for
#: the "how square is the J-V curve still" part of the contact-darkening map.
FF_REFERENCE = 0.80

#: Retention loss (as a fraction) at which each visual channel saturates.
#: These are display constants, not physics. They are exposed so a reader can
#: check exactly how a predicted number became a pixel.
APPEARANCE_SATURATION: dict[str, float] = {
    "bleach": 0.55,     # absorber fully PbI2-yellow at 55% efficiency lost
    "defect": 0.45,     # defect speckle at maximum density at 45% lost
    "edge": 0.50,       # edge ingress at maximum width at 50% lost
    "contact": 0.40,    # contact/interface darkening saturates at 40% lost
}


# --------------------------------------------------------------------------
# the p-i-n stack
#
# Order is the physical one, illuminated side first:
#   glass -> ITO -> HTL -> perovskite -> ETL -> metal
# which is what "p-i-n" (a.k.a. inverted) means: holes are extracted at the
# front. Thicknesses are the nominal values of the baseline device.
#
# Drawn thickness is compressed as d ** THICKNESS_EXPONENT and normalised, so
# the perovskite still reads as by far the thickest layer while a 20 nm HTL
# stays visible. True nanometres are carried alongside and shown in the legend.
# The glass superstrate is 1.1 mm -- four orders of magnitude thicker than
# everything else -- so it is drawn at a fixed height and flagged not to scale.
# --------------------------------------------------------------------------
THICKNESS_EXPONENT = 0.5
STACK_WORLD_HEIGHT = 1.95
COVER_WORLD_HEIGHT = 0.34
MIN_LAYER_WORLD = 0.075

PIN_STACK: tuple[dict[str, Any], ...] = (
    {
        "name": "Glass",
        "sub": "superstrate",
        "role": "cover",
        "thickness_nm": 1_100_000.0,
        "to_scale": False,
        "color": "#DCE9F7",
        "opacity": 0.20,
        "metalness": 0.0,
        "roughness": 0.03,
        "transmission": 0.92,
    },
    {
        "name": "ITO",
        "sub": "transparent contact",
        "role": "front_contact",
        "thickness_nm": 150.0,
        "to_scale": True,
        "color": "#0891B2",
        "opacity": 0.55,
        "metalness": 0.25,
        "roughness": 0.16,
        "transmission": 0.45,
    },
    {
        "name": "HTL",
        "sub": "NiOx / PTAA",
        "role": "htl",
        "thickness_nm": 20.0,
        "to_scale": True,
        "color": "#7C3AED",
        "opacity": 0.94,
        "metalness": 0.0,
        "roughness": 0.44,
        "transmission": 0.0,
    },
    {
        "name": "Perovskite",
        "sub": "absorber",
        "role": "absorber",
        "thickness_nm": 500.0,
        "to_scale": True,
        "color": "#2E1A47",       # fresh: near-black aubergine
        "degraded_color": "#C99A2E",  # spent: PbI2 yellow
        "opacity": 1.0,
        "metalness": 0.0,
        "roughness": 0.34,
        "transmission": 0.0,
    },
    {
        "name": "ETL",
        "sub": "C60 / BCP",
        "role": "etl",
        "thickness_nm": 38.0,
        "to_scale": True,
        "color": "#2563EB",
        "opacity": 0.92,
        "metalness": 0.0,
        "roughness": 0.42,
        "transmission": 0.0,
    },
    {
        "name": "Ag",
        "sub": "back contact",
        "role": "back_contact",
        "thickness_nm": 100.0,
        "to_scale": True,
        "color": "#CBD5E1",
        "degraded_color": "#4B5563",  # tarnished / reacted contact
        "opacity": 1.0,
        "metalness": 1.0,
        "roughness": 0.18,
        "transmission": 0.0,
    },
)


# --------------------------------------------------------------------------
# CDN pin -- the same one the thesis renderer used.
# Exposed as arguments to build_twin_html so the failure path can be exercised
# in a test without editing this file.
# --------------------------------------------------------------------------
_CDN = "https://cdn.jsdelivr.net/npm/three@0.160.0"
THREE_MODULE_URL = f"{_CDN}/build/three.module.js"
THREE_ADDONS_URL = f"{_CDN}/examples/jsm/"


# --------------------------------------------------------------------------
# small pure helpers
# --------------------------------------------------------------------------
def _num(value: Any, key: str) -> float:
    """Coerce ``value`` to a finite float, falling back to ``DEFAULTS[key]``."""
    fallback = float(DEFAULTS.get(key, 0.0))  # type: ignore[arg-type]
    try:
        if value is None or isinstance(value, bool) or value == "":
            return fallback
        out = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(out):
        return fallback
    low, high = CLAMPS.get(key, (-math.inf, math.inf))
    return float(min(max(out, low), high))


def _clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return float(min(max(value, 0.0), 1.0))


def _text(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    out = str(value).strip()
    return out or fallback


def _round(value: float, places: int = 4) -> float:
    out = round(float(value), places)
    return 0.0 if out == 0 else out


def _resolve_tier(tier: Any) -> Tier:
    """Accept a Tier, its value, or its label. Anything unknown -> PREVIEW.

    PREVIEW rather than VALIDATED, because an unrecognised tier is exactly the
    situation in which the app must not claim more than it can back.
    """
    if isinstance(tier, Tier):
        return tier
    needle = _text(tier).lower()
    for candidate in (Tier.VALIDATED, Tier.PREVIEW, Tier.PLANNED):
        if needle in {candidate.value, TIER_LABEL[candidate].lower()}:
            return candidate
    return Tier.PREVIEW


def _resolve_scope(scope: Any) -> str:
    needle = _text(scope, SCOPE_CELL).lower()
    if needle in SCOPES:
        return needle
    for key, label in SCOPE_LABEL.items():
        if needle == label.lower():
            return key
    return SCOPE_CELL


def _capability(scope: str) -> Capability | None:
    try:
        return get_capability(SCOPE_CAPABILITY[scope])
    except KeyError:
        return None


# --------------------------------------------------------------------------
# geometry of the stack
# --------------------------------------------------------------------------
def _stack_geometry(layers: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Attach a drawn world height to every layer, top of stack first.

    ``d ** THICKNESS_EXPONENT`` normalised to :data:`STACK_WORLD_HEIGHT`, with a
    floor of :data:`MIN_LAYER_WORLD` so the thinnest layer is still clickable
    and legible. Layers flagged ``to_scale: False`` (the glass) are given a
    fixed height instead and excluded from the normalisation.
    """
    scaled = [dict(layer) for layer in layers]
    weights = [
        max(1e-9, float(layer.get("thickness_nm", 1.0))) ** THICKNESS_EXPONENT
        if layer.get("to_scale", True)
        else 0.0
        for layer in scaled
    ]
    total = sum(weights)
    for layer, weight in zip(scaled, weights):
        if not layer.get("to_scale", True):
            layer["draw_h"] = COVER_WORLD_HEIGHT
        elif total <= 0:
            layer["draw_h"] = MIN_LAYER_WORLD
        else:
            layer["draw_h"] = max(MIN_LAYER_WORLD, STACK_WORLD_HEIGHT * weight / total)

    # bottom-up cumulative centres: the metal sits on y = 0, glass on top.
    y = 0.0
    for layer in reversed(scaled):
        layer["draw_y"] = _round(y + layer["draw_h"] / 2.0, 5)
        y += layer["draw_h"]
    for layer in scaled:
        layer["draw_h"] = _round(layer["draw_h"], 5)
        layer["thickness_label"] = _thickness_label(float(layer.get("thickness_nm", 0.0)))
    return scaled


def _thickness_label(nm: float) -> str:
    if nm >= 1_000_000:
        return f"{nm / 1_000_000:g} mm"
    if nm >= 1000:
        return f"{nm / 1000:g} um"
    return f"{nm:g} nm"


# --------------------------------------------------------------------------
# the mapping from predicted state to appearance
# --------------------------------------------------------------------------
def _appearance(
    retention_pct: float,
    retention_sd: float,
    ff: float,
    illumination_suns: float,
    temperature_c: float,
) -> dict[str, float]:
    """Turn the predicted state into the scalars the renderer consumes.

    Only ``retention`` and ``ff`` drive *damage*. Illumination and temperature
    drive *environment* cues only -- how bright the light shafts are and how
    much heat shimmer sits above the stack -- so a hot, bright, undegraded cell
    still renders as an undegraded cell.
    """
    loss = _clamp01(1.0 - retention_pct / 100.0)
    loss_high = _clamp01(1.0 - (retention_pct - retention_sd) / 100.0)

    sat = APPEARANCE_SATURATION
    ff_shortfall = _clamp01((FF_REFERENCE - ff) / 0.25)

    return {
        # damage channels -- functions of the predicted state, nothing else
        "loss": _round(loss),
        "lossHigh": _round(loss_high),
        "bleach": _round(_clamp01(loss / sat["bleach"])),
        "defectDensity": _round(_clamp01(loss / sat["defect"])),
        "defectDensityHigh": _round(_clamp01(loss_high / sat["defect"])),
        "edgeIngress": _round(_clamp01(loss / sat["edge"]) ** 0.75),
        "contactDarkening": _round(
            _clamp01(0.6 * _clamp01(loss / sat["contact"]) + 0.4 * ff_shortfall)
        ),
        # environment channels -- never touch the damage look
        "photonFlux": _round(_clamp01(illumination_suns / 1.0)),
        "thermalLoad": _round(_clamp01((temperature_c - 25.0) / 100.0)),
    }


# --------------------------------------------------------------------------
# 1. the pure config builder
# --------------------------------------------------------------------------
def build_twin_config(
    architecture: Any = ARCHITECTURE_VALIDATED,
    illumination_suns: Any = None,
    temperature_c: Any = None,
    aging_h: Any = None,
    pce_pct: Any = None,
    retention_pct: Any = None,
    voc_v: Any = None,
    jsc_macm2: Any = None,
    ff: Any = None,
    *,
    pce_sd: Any = None,
    retention_sd: Any = None,
    voc_sd: Any = None,
    jsc_sd: Any = None,
    ff_sd: Any = None,
    tier: Any = Tier.VALIDATED,
    scope: Any = SCOPE_CELL,
    horizon_h: Any = None,
    title: Any = None,
) -> dict[str, Any]:
    """Turn one predicted device state into a render config.

    Pure: plain numbers in, a JSON-serialisable dict out. No Streamlit, no
    model, no file access, so the page layer stays a two-line call and the
    interesting logic can be unit-tested.

    Any input that is missing, ``None``, non-numeric, NaN or infinite is
    replaced by :data:`DEFAULTS` and clamped to :data:`CLAMPS`. This function
    does not raise.

    Scope and tier interact by one rule: a PLANNED capability never carries a
    number. ``scope="module"`` and ``scope="farm"`` therefore force
    ``Tier.PLANNED`` (their capabilities are planned in the registry) and the
    returned ``metrics`` is ``None``. A PLANNED tier at cell scope -- an n-i-p
    device, say -- gets the same treatment.
    """
    scope_key = _resolve_scope(scope)
    cap = _capability(scope_key)

    resolved_tier = _resolve_tier(tier)
    if cap is not None and cap.tier is Tier.PLANNED:
        # the registry, not the caller, decides that module/farm are unbuilt
        resolved_tier = Tier.PLANNED
    planned = resolved_tier is Tier.PLANNED
    show_numbers = not planned

    arch = _text(architecture, ARCHITECTURE_VALIDATED)

    illum = _num(illumination_suns, "illumination_suns")
    temp = _num(temperature_c, "temperature_c")
    age = _num(aging_h, "aging_h")
    horizon = max(_num(horizon_h, "horizon_h"), age, 1.0)

    pce = _num(pce_pct, "pce_pct")
    retention = _num(retention_pct, "retention_pct")
    voc = _num(voc_v, "voc_v")
    jsc = _num(jsc_macm2, "jsc_macm2")
    fill = _num(ff, "ff")

    sd = {
        "pce": _num(pce_sd, "pce_sd"),
        "retention": _num(retention_sd, "retention_sd"),
        "voc": _num(voc_sd, "voc_sd"),
        "jsc": _num(jsc_sd, "jsc_sd"),
        "ff": _num(ff_sd, "ff_sd"),
    }

    appearance = _appearance(retention, sd["retention"], fill, illum, temp)

    metrics: dict[str, float] | None = None
    if show_numbers:
        metrics = {
            "pce_pct": _round(pce, 3),
            "pce_sd": _round(sd["pce"], 3),
            "retention_pct": _round(retention, 3),
            "retention_sd": _round(sd["retention"], 3),
            "voc_v": _round(voc, 4),
            "voc_sd": _round(sd["voc"], 4),
            "jsc_macm2": _round(jsc, 3),
            "jsc_sd": _round(sd["jsc"], 3),
            "ff": _round(fill, 4),
            "ff_sd": _round(sd["ff"], 4),
        }

    if retention >= 90:
        health = "Above T90"
    elif retention >= 80:
        health = "Between T90 and T80"
    else:
        health = "Below T80"

    tier_key = _TIER_KEY[resolved_tier]
    config: dict[str, Any] = {
        "version": 1,
        "scope": scope_key,
        "scopeLabel": SCOPE_LABEL[scope_key],
        "scopeBlurb": SCOPE_BLURB[scope_key],
        "title": _text(title, SCOPE_TITLE[scope_key]),
        "capabilityKey": SCOPE_CAPABILITY[scope_key],
        "capabilityLabel": cap.label if cap else SCOPE_LABEL[scope_key],
        "tier": resolved_tier.value,
        "tierKey": tier_key,
        "tierLabel": TIER_LABEL[resolved_tier],
        "tierIcon": TIER_ICON[resolved_tier],
        "tierColor": PALETTE[tier_key],
        "tierBg": PALETTE[f"{tier_key}Bg"],
        "tierBorder": PALETTE[f"{tier_key}Border"],
        "planned": planned,
        "preview": resolved_tier is Tier.PREVIEW,
        "showNumbers": show_numbers,
        "plannedVersion": (cap.version if cap else "") if planned else "",
        "architecture": arch,
        "conditions": {
            "illumination_suns": _round(illum, 4),
            "temperature_c": _round(temp, 3),
            "aging_h": _round(age, 3),
            "horizon_h": _round(horizon, 3),
            "progress": _round(_clamp01(age / horizon)),
        },
        "metrics": metrics,
        "health": health if show_numbers else "",
        "appearance": appearance,
        "layers": _stack_geometry(PIN_STACK),
        "thicknessNote": (
            f"Drawn thickness is compressed as d^{THICKNESS_EXPONENT:g} so a 20 nm "
            "layer stays visible; true values are listed. Glass is not to scale."
        ),
        "palette": dict(PALETTE),
        "roadmap": {
            "version": cap.version if cap else "",
            "backing": cap.backing if cap else "",
            "unlocks": cap.unlocks if cap else "",
        },
        "motion": {
            "autoRotate": True,
            "shimmer": True,
        },
        "frames": [],
        "playback": False,
    }
    return config


def build_twin_timeline(states: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Merge configs from :func:`build_twin_config` into one animatable config.

    The renderer will interpolate between consecutive frames and nothing else:
    it cannot run past the last frame, and every in-between value is a plain
    linear blend of two states the surrogate actually produced. Feed it the
    aging times you predicted, not a start and an end.

    The first state supplies scope, tier, architecture and layers. States are
    sorted by aging time. An empty iterable yields a default fresh-cell config.
    """
    frames: list[dict[str, Any]] = []
    base: dict[str, Any] | None = None
    for state in states:
        if not isinstance(state, Mapping):
            continue
        if base is None:
            base = dict(state)
        conditions = state.get("conditions") or {}
        frames.append(
            {
                "aging_h": float(conditions.get("aging_h", 0.0) or 0.0),
                "appearance": dict(state.get("appearance") or {}),
                "metrics": dict(state["metrics"]) if state.get("metrics") else None,
                "health": _text(state.get("health")),
                "conditions": dict(conditions),
            }
        )

    if base is None:
        return build_twin_config()

    frames.sort(key=lambda frame: frame["aging_h"])
    if len(frames) > 1:
        base["frames"] = frames
        base["playback"] = True
        # start the scene on the last predicted state so a static screenshot
        # of the page is the end-of-life picture, not a misleadingly fresh one
        base["appearance"] = dict(frames[-1]["appearance"])
        base["metrics"] = dict(frames[-1]["metrics"]) if frames[-1]["metrics"] else None
        base["health"] = frames[-1]["health"]
        base["conditions"] = dict(frames[-1]["conditions"])
    return base


# --------------------------------------------------------------------------
# 2. HTML rendering
# --------------------------------------------------------------------------
def _e(value: Any) -> str:
    return html.escape(_text(value), quote=True)


def _fmt(value: float, places: int = 2) -> str:
    if not math.isfinite(value):
        return "-"
    return f"{value:,.{places}f}"


def _pm(value: float, sd: float, places: int = 2, unit: str = "") -> str:
    body = _fmt(value, places)
    if sd and sd > 0:
        body += f" &plusmn; {_fmt(sd, places)}"
    if unit:
        body += f" {unit}"
    return body


def _hud_html(config: Mapping[str, Any]) -> str:
    cond = config.get("conditions") or {}
    metrics = config.get("metrics")
    rows: list[str] = []

    def row(label: str, value: str, ident: str = "") -> str:
        attr = f' id="{ident}"' if ident else ""
        return (
            '<div class="hud-cell">'
            f'<div class="hud-label">{label}</div>'
            f'<div class="hud-value"{attr}>{value}</div>'
            "</div>"
        )

    rows.append(row("Architecture", _e(config.get("architecture"))))
    rows.append(row("Illumination", f"{_fmt(float(cond.get('illumination_suns', 0)), 2)} suns"))
    rows.append(row("Temperature", f"{_fmt(float(cond.get('temperature_c', 0)), 1)} &deg;C"))
    rows.append(row("Aging time", f"{_fmt(float(cond.get('aging_h', 0)), 0)} h", "hud-age"))

    if metrics:
        rows.append(
            row("PCE", _pm(float(metrics["pce_pct"]), float(metrics["pce_sd"]), 2, "%"), "hud-pce")
        )
        rows.append(
            row(
                "Retention",
                _pm(float(metrics["retention_pct"]), float(metrics["retention_sd"]), 1, "%"),
                "hud-ret",
            )
        )
        rows.append(row("Voc", _pm(float(metrics["voc_v"]), float(metrics["voc_sd"]), 3, "V"), "hud-voc"))
        rows.append(
            row("Jsc", _pm(float(metrics["jsc_macm2"]), float(metrics["jsc_sd"]), 2, "mA/cm2"), "hud-jsc")
        )
        rows.append(row("FF", _pm(float(metrics["ff"]), float(metrics["ff_sd"]), 3), "hud-ff"))
        rows.append(row("Lifetime", _e(config.get("health")), "hud-health"))
    else:
        rows.append(row("PCE", "not modelled"))
        rows.append(row("Retention", "not modelled"))
        rows.append(row("Per-cell output", "not modelled"))
        rows.append(row("Target release", _e(config.get("plannedVersion") or "-")))

    chip = (
        f'<span class="chip chip-{_e(config.get("tierKey"))}">'
        f'{_e(config.get("tierIcon"))} {_e(config.get("tierLabel"))}</span>'
    )

    if config.get("planned"):
        note = (
            "<strong>This scale is not built yet.</strong> "
            f"{_e((config.get('roadmap') or {}).get('backing'))} "
            "No predicted values are shown here, at any zoom level."
        )
    elif config.get("preview"):
        note = (
            "<strong>Preview.</strong> The shape is indicative and the numbers are "
            "soft; the underlying chaining of states is not validated."
        )
    else:
        note = (
            "Damage in this view is a function of the predicted retention above. "
            "No decay timer runs in this canvas."
        )

    return (
        '<section id="hud" aria-label="Digital twin state">'
        '<header class="hud-head">'
        '<div class="hud-eyebrow">Digital twin</div>'
        f'<h1 class="hud-title">{_e(config.get("title"))}</h1>'
        f'<div class="hud-chips">{chip}'
        f'<span class="chip chip-scope">{_e(config.get("scopeLabel"))} scale</span></div>'
        "</header>"
        f'<div class="hud-grid">{"".join(rows)}</div>'
        f'<p class="hud-note">{note}</p>'
        "</section>"
    )


def _legend_html(config: Mapping[str, Any]) -> str:
    if config.get("planned"):
        road = config.get("roadmap") or {}
        unlocks = _e(road.get("unlocks")) or "Run the simulation campaign that grounds it."
        return (
            '<aside id="legend" aria-label="Roadmap">'
            f'<div class="legend-head">Planned{" " + _e(road.get("version")) if road.get("version") else ""}</div>'
            f'<p class="legend-note">{_e(config.get("scopeBlurb"))}</p>'
            f'<div class="legend-head">What unlocks it</div>'
            f'<p class="legend-note">{unlocks}</p>'
            "</aside>"
        )

    layers = config.get("layers") or []
    swatches = "".join(
        '<div class="swatch-row">'
        f'<span class="swatch" style="background:{_e(layer.get("color"))}"></span>'
        f'<span class="swatch-name">{_e(layer.get("name"))}</span>'
        f'<span class="swatch-sub">{_e(layer.get("sub"))}</span>'
        f'<span class="swatch-nm">{_e(layer.get("thickness_label"))}</span>'
        "</div>"
        for layer in layers
    )

    app = config.get("appearance") or {}
    bars = []
    for key, label in (
        ("defectDensity", "Defect density"),
        ("contactDarkening", "Contact darkening"),
        ("edgeIngress", "Edge ingress"),
        ("bleach", "Absorber bleaching"),
    ):
        value = _clamp01(float(app.get(key, 0.0)))
        bars.append(
            '<div class="bar-row">'
            f'<span class="bar-label">{label}</span>'
            '<span class="bar-shell">'
            f'<span class="bar-fill" id="bar-{key}" style="width:{value * 100:.1f}%"></span>'
            "</span>"
            f'<span class="bar-pct" id="pct-{key}">{value * 100:.0f}%</span>'
            "</div>"
        )

    return (
        '<aside id="legend" aria-label="Layer stack and degradation channels">'
        '<div class="legend-head">Stack, illuminated side first</div>'
        f'<div class="swatches">{swatches}</div>'
        f'<p class="legend-note">{_e(config.get("thicknessNote"))}</p>'
        '<div class="legend-head">Appearance driven by predicted retention</div>'
        f'<div class="bars">{"".join(bars)}</div>'
        '<p class="legend-note">Faint outer speckle marks the extra defects implied '
        "by the +1&sigma; retention uncertainty.</p>"
        "</aside>"
    )


def _timeline_html(config: Mapping[str, Any]) -> str:
    frames = config.get("frames") or []
    if len(frames) < 2:
        return ""
    last = len(frames) - 1
    return (
        '<div id="timeline" aria-label="Predicted aging timeline">'
        '<button id="tl-play" type="button" aria-label="Play the predicted aging sequence">Play</button>'
        f'<input id="tl-range" type="range" min="0" max="{last}" step="0.001" value="{last}" '
        'aria-label="Aging time, interpolated between predicted states" />'
        f'<span id="tl-readout">{_fmt(float(frames[-1]["aging_h"]), 0)} h</span>'
        "</div>"
        '<p class="tl-note">Interpolates between '
        f"{len(frames)} predicted states. Nothing between them is extrapolated.</p>"
    )


def _fallback_svg(config: Mapping[str, Any]) -> str:
    """A static cross-section of the same stack, for when WebGL never arrives.

    Built from the identical config, so the fallback tells the same story: the
    same layer order, the same relative thickness, the same bleaching and the
    same edge ingress -- just flat and without a camera.
    """
    try:
        layers = list(config.get("layers") or [])
        if not layers:
            return ""
        app = config.get("appearance") or {}
        planned = bool(config.get("planned"))
        bleach = 0.0 if planned else _clamp01(float(app.get("bleach", 0.0)))
        edge = 0.0 if planned else _clamp01(float(app.get("edgeIngress", 0.0)))
        defect = 0.0 if planned else _clamp01(float(app.get("defectDensity", 0.0)))
        dark = 0.0 if planned else _clamp01(float(app.get("contactDarkening", 0.0)))

        width, pad_x, pad_y = 420.0, 14.0, 12.0
        plate_w = 236.0
        total = sum(float(layer.get("draw_h", 0.1)) for layer in layers) or 1.0
        height = 232.0
        usable = height - 2 * pad_y

        def lerp_hex(a: str, b: str, t: float) -> str:
            try:
                ar, ag, ab = (int(a[i : i + 2], 16) for i in (1, 3, 5))
                br, bg, bb = (int(b[i : i + 2], 16) for i in (1, 3, 5))
            except (ValueError, IndexError):
                return a
            mix = lambda x, y: int(round(x + (y - x) * t))  # noqa: E731
            return f"#{mix(ar, br):02X}{mix(ag, bg):02X}{mix(ab, bb):02X}"

        parts: list[str] = []
        y = pad_y
        absorber_box = None
        for layer in layers:
            h = usable * float(layer.get("draw_h", 0.1)) / total
            color = _text(layer.get("color"), "#94A3B8")
            if planned:
                color = PALETTE["planned"]
            elif layer.get("role") == "absorber" and layer.get("degraded_color"):
                color = lerp_hex(color, str(layer["degraded_color"]), bleach)
            elif layer.get("role") == "back_contact" and layer.get("degraded_color"):
                color = lerp_hex(color, str(layer["degraded_color"]), dark)
            opacity = 0.25 if planned else float(layer.get("opacity", 1.0))
            parts.append(
                f'<rect x="{pad_x:.1f}" y="{y:.1f}" width="{plate_w:.1f}" height="{max(h, 1.5):.1f}" '
                f'fill="{color}" fill-opacity="{opacity:.2f}" stroke="{PALETTE["border"]}" '
                'stroke-width="0.8" />'
            )
            label_y = y + max(h, 1.5) / 2 + 3.6
            parts.append(
                f'<text x="{pad_x + plate_w + 12:.1f}" y="{label_y:.1f}" '
                f'font-size="10.5" fill="{PALETTE["text"]}" font-weight="600">'
                f"{_e(layer.get('name'))}</text>"
            )
            parts.append(
                f'<text x="{pad_x + plate_w + 12:.1f}" y="{label_y + 11:.1f}" '
                f'font-size="9" fill="{PALETTE["textFaint"]}">'
                f"{_e(layer.get('thickness_label'))}</text>"
            )
            if layer.get("role") == "absorber":
                absorber_box = (pad_x, y, plate_w, max(h, 1.5))
            y += h

        if absorber_box and not planned:
            ax, ay, aw, ah = absorber_box
            if edge > 0.01:
                inset = max(2.0, aw * 0.16 * edge)
                for rx in (ax, ax + aw - inset):
                    parts.append(
                        f'<rect x="{rx:.1f}" y="{ay:.1f}" width="{inset:.1f}" height="{ah:.1f}" '
                        f'fill="#C99A2E" fill-opacity="{0.45 + 0.35 * edge:.2f}" />'
                    )
            count = int(round(6 + 90 * defect))
            for i in range(count):
                # deterministic scatter -- same seed idea as the WebGL cloud
                fx = (math.sin(i * 12.9898) * 43758.5453) % 1.0
                fy = (math.sin(i * 78.233 + 3.1) * 43758.5453) % 1.0
                parts.append(
                    f'<circle cx="{ax + 3 + fx * (aw - 6):.1f}" cy="{ay + 2 + fy * max(ah - 4, 1):.1f}" '
                    f'r="{1.0 + 0.9 * defect:.2f}" fill="#DC2626" fill-opacity="0.72" />'
                )

        parts.append(
            f'<text x="{pad_x:.1f}" y="{height - 1:.1f}" font-size="9" '
            f'fill="{PALETTE["textFaint"]}">Static cross-section, same predicted state</text>'
        )
        return (
            f'<svg viewBox="0 0 {width:.0f} {height:.0f}" width="100%" '
            f'style="max-width:{width:.0f}px" role="img" '
            'aria-label="Cross-section of the layer stack" '
            'xmlns="http://www.w3.org/2000/svg">' + "".join(parts) + "</svg>"
        )
    except Exception:  # pragma: no cover - the fallback must never be the bug
        return ""


def _fallback_html(config: Mapping[str, Any]) -> str:
    metrics = config.get("metrics")
    cond = config.get("conditions") or {}
    if metrics:
        summary = (
            f'<li><span>PCE</span><strong>{_pm(float(metrics["pce_pct"]), float(metrics["pce_sd"]), 2, "%")}</strong></li>'
            f'<li><span>Retention</span><strong>{_pm(float(metrics["retention_pct"]), float(metrics["retention_sd"]), 1, "%")}</strong></li>'
            f'<li><span>Aging</span><strong>{_fmt(float(cond.get("aging_h", 0)), 0)} h</strong></li>'
        )
    else:
        summary = (
            f'<li><span>Scale</span><strong>{_e(config.get("scopeLabel"))}</strong></li>'
            '<li><span>Status</span><strong>Planned, no values</strong></li>'
        )

    return (
        '<div id="fallback" role="status">'
        '<div class="fb-card">'
        '<div class="fb-body">'
        '<div class="fb-eyebrow">3D view unavailable</div>'
        '<h2 class="fb-title">The interactive twin could not start</h2>'
        '<p class="fb-text" id="fb-reason">The Three.js library could not be loaded.</p>'
        '<p class="fb-text fb-quiet">The 3D view is presentation only. Every number on '
        "this page comes from the surrogate and is unaffected. The cross-section on the "
        "right is drawn from exactly the same predicted state.</p>"
        f'<ul class="fb-list">{summary}</ul>'
        "</div>"
        f'<div class="fb-figure">{_fallback_svg(config)}</div>'
        "</div>"
        "</div>"
    )


def _sr_summary(config: Mapping[str, Any]) -> str:
    cond = config.get("conditions") or {}
    metrics = config.get("metrics")
    bits = [
        f'{_e(config.get("scopeLabel"))}-scale digital twin of a {_e(config.get("architecture"))} device.',
        f'Tier: {_e(config.get("tierLabel"))}.',
        (
            f'Aged {_fmt(float(cond.get("aging_h", 0)), 0)} hours at '
            f'{_fmt(float(cond.get("temperature_c", 0)), 0)} degrees Celsius and '
            f'{_fmt(float(cond.get("illumination_suns", 0)), 2)} suns.'
        ),
    ]
    if metrics:
        app = config.get("appearance") or {}
        bits.append(
            f'Predicted efficiency {_fmt(float(metrics["pce_pct"]), 2)} percent, '
            f'retaining {_fmt(float(metrics["retention_pct"]), 1)} percent of its initial value.'
        )
        bits.append(
            "The absorber is drawn "
            f'{_fmt(_clamp01(float(app.get("bleach", 0.0))) * 100, 0)} percent bleached toward '
            "lead-iodide yellow, with defect speckle and edge ingress scaled to the same loss."
        )
    else:
        bits.append(
            "This scale is not modelled yet, so it is drawn as a grey wireframe and "
            "no predicted values are shown."
        )
    return " ".join(bits)


def _json_payload(config: Mapping[str, Any]) -> str:
    """JSON that is safe to drop inside a <script> element."""
    try:
        raw = json.dumps(config, ensure_ascii=True, allow_nan=False, default=str)
    except (TypeError, ValueError):
        raw = json.dumps(build_twin_config(), ensure_ascii=True, allow_nan=False)
    raw = raw.replace("</", "<\\/").replace("<!--", "<\\!--")
    return raw


def build_twin_html(
    config: Mapping[str, Any],
    *,
    three_module_url: str = THREE_MODULE_URL,
    three_addons_url: str = THREE_ADDONS_URL,
) -> str:
    """Render ``config`` into a self-contained HTML document.

    Drop the result straight into ``st.components.v1.html(html, height=560)``.
    The only network requests it makes are the two pinned Three.js module files
    (three@0.160.0 via jsdelivr). If either is unreachable, or WebGL is missing,
    or anything in the scene code throws, the page swaps in an explanatory panel
    with a static SVG cross-section instead of failing blank.

    ``three_module_url`` / ``three_addons_url`` exist so the failure path can be
    exercised by pointing them somewhere that does not resolve.
    """
    if not isinstance(config, Mapping):
        config = build_twin_config()
    cfg = dict(config)
    cfg.setdefault("palette", dict(PALETTE))

    try:
        body = (
            _TEMPLATE.replace("__TWIN_TITLE__", _e(cfg.get("title") or "PSC digital twin"))
            .replace("__TWIN_THREE_URL__", _e(three_module_url))
            .replace("__TWIN_ADDONS_URL__", _e(three_addons_url))
            .replace("__TWIN_HUD__", _hud_html(cfg))
            .replace("__TWIN_LEGEND__", _legend_html(cfg))
            .replace("__TWIN_TIMELINE__", _timeline_html(cfg))
            .replace("__TWIN_FALLBACK__", _fallback_html(cfg))
            .replace("__TWIN_SR__", _e(_sr_summary(cfg)))
            .replace("__TWIN_WATERMARK__", "" if not cfg.get("planned") else "visible")
            .replace(
                "__TWIN_WATERMARK_TEXT__",
                _e(f"Planned {cfg.get('plannedVersion', '')}".strip()),
            )
            .replace("__TWIN_CONFIG__", _json_payload(cfg))
        )
    except Exception as exc:  # pragma: no cover - a render must never 500 a page
        return _minimal_error_page(str(exc))
    return body


def _minimal_error_page(reason: str) -> str:
    return (
        "<!doctype html><html><head><meta charset='utf-8'></head>"
        f"<body style=\"margin:0;background:{PALETTE['bg']};color:{PALETTE['text']};"
        'font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif">'
        f"<div style=\"margin:24px;padding:18px 20px;background:{PALETTE['surface']};"
        f"border:1px solid {PALETTE['border']};border-left:4px solid {PALETTE['preview']};"
        'border-radius:12px">'
        "<strong>The 3D twin could not be assembled.</strong><br>"
        f"<span style=\"color:{PALETTE['textMuted']}\">{html.escape(reason)}</span><br>"
        f"<span style=\"color:{PALETTE['textMuted']}\">Every number on this page comes "
        "from the surrogate and is unaffected.</span></div></body></html>"
    )


# --------------------------------------------------------------------------
# the document template
#
# Token substitution rather than str.format, because the CSS and JS below are
# full of braces. Tokens: __TWIN_TITLE__, __TWIN_THREE_URL__,
# __TWIN_ADDONS_URL__, __TWIN_HUD__, __TWIN_LEGEND__, __TWIN_TIMELINE__,
# __TWIN_FALLBACK__, __TWIN_SR__, __TWIN_WATERMARK__, __TWIN_WATERMARK_TEXT__,
# __TWIN_CONFIG__.
# --------------------------------------------------------------------------
_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>__TWIN_TITLE__</title>
<style>
  :root {
    color-scheme: light;
    --bg: #F7F9FC;
    --surface: #FFFFFF;
    --surface-sunk: #F1F5F9;
    --border: #E3E8EF;
    --border-strong: #CBD5E1;
    --text: #0F172A;
    --text-muted: #5B6B82;
    --text-faint: #94A3B8;
    --primary: #2563EB;
    --primary-soft: #EFF6FF;
    --validated: #059669;
    --preview: #D97706;
    --planned: #64748B;
    --validated-bg: #ECFDF5;
    --preview-bg: #FFFBEB;
    --planned-bg: #F1F5F9;
    --validated-border: #A7F3D0;
    --preview-border: #FDE68A;
    --planned-border: #CBD5E1;
    --font: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font);
    font-size: 13px;
    overflow: hidden;
  }
  #stage {
    position: absolute;
    inset: 0;
    overflow: hidden;
    border: 1px solid var(--border);
    border-radius: 14px;
    background:
      radial-gradient(120% 90% at 18% 6%, #FFFFFF 0%, rgba(255,255,255,0) 58%),
      radial-gradient(90% 70% at 88% 96%, #EEF3FA 0%, rgba(238,243,250,0) 60%),
      var(--bg);
  }
  #viewport { position: absolute; inset: 0; }
  #viewport canvas { display: block; width: 100%; height: 100%; outline: none; }
  #viewport:focus-visible { outline: 2px solid var(--primary); outline-offset: -3px; }

  .sr-only {
    position: absolute; width: 1px; height: 1px; margin: -1px;
    padding: 0; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; border: 0;
  }

  /* ---------------- HUD ---------------- */
  #hud {
    position: absolute; left: 14px; top: 14px;
    width: clamp(250px, 34%, 348px);
    padding: 13px 15px 12px;
    background: rgba(255,255,255,0.93);
    border: 1px solid var(--border);
    border-radius: 12px;
    box-shadow: 0 8px 26px rgba(15,23,42,0.07);
    backdrop-filter: blur(9px);
    pointer-events: none;
  }
  .hud-eyebrow {
    font-size: 9.5px; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--text-faint);
  }
  .hud-title { margin: 2px 0 7px; font-size: 15px; font-weight: 680; letter-spacing: -0.01em; }
  .hud-chips { display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 10px; }
  .chip {
    display: inline-flex; align-items: center; gap: 4px;
    border-radius: 999px; padding: 2px 9px;
    font-size: 10.5px; font-weight: 650; letter-spacing: 0.01em;
    border: 1px solid transparent; white-space: nowrap;
  }
  .chip-validated { background: var(--validated-bg); color: var(--validated); border-color: var(--validated-border); }
  .chip-preview   { background: var(--preview-bg);   color: var(--preview);   border-color: var(--preview-border); }
  .chip-planned   { background: var(--planned-bg);   color: var(--planned);   border-color: var(--planned-border); }
  .chip-scope     { background: var(--primary-soft); color: var(--primary);   border-color: #BFDBFE; }
  .hud-grid { display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 7px 12px; }
  .hud-cell { min-width: 0; }
  .hud-label { font-size: 9.5px; color: var(--text-faint); letter-spacing: 0.02em; }
  .hud-value {
    font-size: 12.5px; font-weight: 660; color: var(--text);
    font-variant-numeric: tabular-nums;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }
  .hud-note {
    margin: 10px 0 0; padding-top: 9px; border-top: 1px solid var(--border);
    font-size: 10.5px; line-height: 1.45; color: var(--text-muted);
  }

  /* ---------------- legend ---------------- */
  #legend {
    position: absolute; right: 14px; top: 14px;
    width: clamp(200px, 26%, 268px);
    max-height: calc(100% - 96px);
    overflow: auto;
    padding: 12px 14px;
    background: rgba(255,255,255,0.93);
    border: 1px solid var(--border);
    border-radius: 12px;
    box-shadow: 0 8px 26px rgba(15,23,42,0.07);
    backdrop-filter: blur(9px);
    pointer-events: none;
  }
  .legend-head {
    font-size: 9.5px; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--text-faint); margin-bottom: 7px;
  }
  .legend-head + .legend-head, .bars + .legend-head, .legend-note + .legend-head { margin-top: 12px; }
  .swatch-row {
    display: grid; grid-template-columns: 12px 1fr auto; gap: 6px;
    align-items: baseline; margin-bottom: 5px;
  }
  .swatch {
    width: 10px; height: 10px; border-radius: 3px;
    border: 1px solid rgba(15,23,42,0.16); display: inline-block; transform: translateY(1px);
  }
  .swatch-name { font-size: 11px; font-weight: 640; }
  .swatch-sub { display: none; }
  .swatch-nm { font-size: 10px; color: var(--text-faint); font-variant-numeric: tabular-nums; }
  .legend-note { margin: 8px 0 0; font-size: 9.8px; line-height: 1.45; color: var(--text-muted); }
  .bar-row { display: grid; grid-template-columns: 1fr 62px 30px; gap: 6px; align-items: center; margin-bottom: 5px; }
  .bar-label { font-size: 10.5px; color: var(--text-muted); }
  .bar-shell { height: 6px; border-radius: 999px; background: var(--surface-sunk); border: 1px solid var(--border); overflow: hidden; }
  .bar-fill { display: block; height: 100%; background: linear-gradient(90deg, var(--primary), #7C3AED); border-radius: inherit; }
  .bar-pct { font-size: 10px; color: var(--text-faint); text-align: right; font-variant-numeric: tabular-nums; }

  /* ---------------- planned watermark ---------------- */
  #watermark { position: absolute; inset: 0; display: none; place-items: center; pointer-events: none; }
  #watermark[data-on="visible"] {
    display: grid;
    background: repeating-linear-gradient(135deg,
      rgba(100,116,139,0.055) 0 13px, rgba(100,116,139,0) 13px 26px);
  }
  #watermark span {
    transform: rotate(-16deg);
    font-size: clamp(26px, 6.4vw, 58px); font-weight: 800; letter-spacing: 0.14em;
    text-transform: uppercase; color: rgba(100,116,139,0.22);
    border: 3px dashed rgba(100,116,139,0.26);
    border-radius: 16px; padding: 0.18em 0.42em;
  }

  /* ---------------- timeline ---------------- */
  #timeline {
    position: absolute; left: 14px; right: 14px; bottom: 34px;
    display: flex; align-items: center; gap: 10px;
    padding: 8px 12px;
    background: rgba(255,255,255,0.94);
    border: 1px solid var(--border); border-radius: 999px;
    box-shadow: 0 8px 26px rgba(15,23,42,0.07);
  }
  #tl-play {
    flex: none; border: 1px solid var(--primary); background: var(--primary); color: #fff;
    font: 600 11px/1 var(--font); padding: 6px 13px; border-radius: 999px; cursor: pointer;
  }
  #tl-play:hover { filter: brightness(1.06); }
  #tl-range { flex: 1 1 auto; accent-color: var(--primary); min-width: 60px; }
  #tl-readout { flex: none; font-size: 11px; font-weight: 650; color: var(--text-muted); font-variant-numeric: tabular-nums; min-width: 54px; text-align: right; }
  .tl-note { position: absolute; left: 22px; right: 22px; bottom: 14px; margin: 0; font-size: 9.5px; color: var(--text-faint); text-align: center; }

  /* ---------------- loading + fallback ---------------- */
  #loading {
    position: absolute; inset: 0; display: grid; place-items: center;
    color: var(--text-muted); font-size: 12px; letter-spacing: 0.02em;
  }
  #loading[hidden] { display: none; }
  #loading .dot {
    width: 7px; height: 7px; border-radius: 999px; background: var(--primary);
    display: inline-block; margin-right: 7px; animation: pulse 1.1s ease-in-out infinite;
  }
  @keyframes pulse { 0%,100% { opacity: 0.25; } 50% { opacity: 1; } }

  #fallback {
    position: absolute; inset: 0; display: grid; place-items: center;
    padding: 18px; background: var(--bg); overflow: auto;
  }
  #fallback[hidden] { display: none; }
  .fb-card {
    display: grid; grid-template-columns: minmax(0, 1.15fr) minmax(0, 0.85fr);
    gap: 18px; align-items: center;
    width: min(880px, 100%);
    padding: 20px 22px;
    background: var(--surface); border: 1px solid var(--border);
    border-left: 4px solid var(--primary);
    border-radius: 14px; box-shadow: 0 8px 26px rgba(15,23,42,0.06);
  }
  .fb-eyebrow {
    font-size: 9.5px; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--text-faint);
  }
  .fb-title { margin: 4px 0 8px; font-size: 16px; font-weight: 680; }
  .fb-text { margin: 0 0 8px; font-size: 12px; line-height: 1.55; color: var(--text-muted); }
  .fb-quiet { color: var(--text-faint); }
  .fb-list { list-style: none; margin: 12px 0 0; padding: 0; display: grid; gap: 5px; }
  .fb-list li {
    display: flex; justify-content: space-between; gap: 12px;
    padding: 6px 10px; background: var(--surface-sunk);
    border: 1px solid var(--border); border-radius: 8px; font-size: 11.5px;
  }
  .fb-list span { color: var(--text-muted); }
  .fb-list strong { font-variant-numeric: tabular-nums; }
  .fb-figure { min-width: 0; }

  /* ---------------- responsive: 700 -> 1200 px ---------------- */
  @media (max-width: 1000px) {
    #legend { width: clamp(180px, 30%, 232px); padding: 10px 12px; }
    #hud { width: clamp(228px, 40%, 300px); }
  }
  @media (max-width: 840px) {
    #legend { display: none; }
    #hud { width: calc(100% - 28px); }
    .hud-grid { grid-template-columns: repeat(3, minmax(0,1fr)); }
    .fb-card { grid-template-columns: minmax(0,1fr); }
  }
  @media (max-height: 420px) {
    .hud-note { display: none; }
    .tl-note { display: none; }
  }
  @media (prefers-reduced-motion: reduce) {
    #loading .dot { animation: none; opacity: 0.7; }
  }
</style>
<script type="importmap">
{
  "imports": {
    "three": "__TWIN_THREE_URL__",
    "three/addons/": "__TWIN_ADDONS_URL__"
  }
}
</script>
</head>
<body>
<main id="stage">
  <div id="viewport" tabindex="0" role="img" aria-label="__TWIN_SR__"></div>
  <div id="watermark" data-on="__TWIN_WATERMARK__" aria-hidden="true"><span>__TWIN_WATERMARK_TEXT__</span></div>
  __TWIN_HUD__
  __TWIN_LEGEND__
  __TWIN_TIMELINE__
  <div id="loading" hidden><span><span class="dot"></span>Loading the 3D twin...</span></div>
  __TWIN_FALLBACK__
  <p class="sr-only" id="sr-summary">__TWIN_SR__</p>
</main>

<script>
  window.TWIN_CONFIG = __TWIN_CONFIG__;
</script>

<script>
/* Boot guard. Runs as a classic script so it executes even when the module
   graph never resolves. The fallback panel ships VISIBLE in the markup, so a
   browser with JavaScript switched off still sees the explanation and the
   static cross-section; this script hides it and shows a loading state, and
   puts it back if the scene does not come up. */
(function () {
  var fallback = document.getElementById('fallback');
  var loading = document.getElementById('loading');
  window.TWIN_BOOTED = false;
  window.TWIN_FAILED = false;

  if (fallback) { fallback.setAttribute('hidden', ''); }
  if (loading) { loading.removeAttribute('hidden'); }

  window.twinFail = function (reason) {
    if (window.TWIN_BOOTED || window.TWIN_FAILED) { return; }
    window.TWIN_FAILED = true;
    if (loading) { loading.setAttribute('hidden', ''); }
    var slot = document.getElementById('fb-reason');
    if (slot && reason) { slot.textContent = reason; }
    ['hud', 'legend', 'timeline', 'watermark'].forEach(function (id) {
      var node = document.getElementById(id);
      if (node) { node.style.display = 'none'; }
    });
    var notes = document.querySelector('.tl-note');
    if (notes) { notes.style.display = 'none'; }
    if (fallback) { fallback.removeAttribute('hidden'); }
  };

  window.twinBooted = function () {
    window.TWIN_BOOTED = true;
    if (loading) { loading.setAttribute('hidden', ''); }
    if (fallback) { fallback.setAttribute('hidden', ''); }
  };

  /* capture phase catches <script> / network resource failures, which is how a
     blocked or unreachable CDN shows up */
  window.addEventListener('error', function (event) {
    var target = event && event.target;
    if (target && target !== window && (target.tagName === 'SCRIPT' || target.tagName === 'LINK')) {
      window.twinFail('The Three.js library could not be fetched from the jsdelivr CDN. '
        + 'The network may be offline, or the CDN blocked by a proxy or content policy.');
    }
  }, true);

  window.addEventListener('unhandledrejection', function () {
    window.twinFail('The 3D scene failed while loading its Three.js modules.');
  });

  /* belt and braces: if nothing has booted after 8 s, assume it never will */
  setTimeout(function () {
    window.twinFail('The Three.js library did not load within 8 seconds. '
      + 'The network may be offline, or the CDN blocked by a proxy or content policy.');
  }, 8000);
})();
</script>

<script nomodule>
  window.twinFail('This browser does not support JavaScript modules, which the 3D view requires.');
</script>

<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

/* ==========================================================================
   Renderer.

   Reads window.TWIN_CONFIG once. Every degradation quantity below comes from
   cfg.appearance, which Python computed from the predicted state. No decay is
   generated here. When cfg.frames is populated the timeline lerps between the
   supplied frames, and only between them.
   ========================================================================== */

const cfg = window.TWIN_CONFIG || {};
const P = cfg.palette || {};
const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

const FRESH_PVK = '#2E1A47';
const SPENT_PVK = '#C99A2E';
const DEFECT_COLOR = '#DC2626';
const PLANNED_COLOR = P.planned || '#64748B';

function clamp01(v) { return Math.max(0, Math.min(1, Number(v) || 0)); }
function lerp(a, b, t) { return a + (b - a) * t; }

/* deterministic pseudo-random: the same config always renders the same scene,
   which matters because these get screenshotted into a thesis */
function seeded(i, salt) {
  const x = Math.sin(i * 127.1 + (salt || 0) * 311.7 + 17.3) * 43758.5453;
  return x - Math.floor(x);
}

function main() {
  const stage = document.getElementById('viewport');
  const width = () => Math.max(1, stage.clientWidth || 900);
  const height = () => Math.max(1, stage.clientHeight || 520);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  /* updateStyle=false: the stylesheet already pins the canvas to 100%/100% of
     the stage. Letting three.js also write inline pixel sizes lets the drawing
     buffer and the CSS box disagree whenever the container is laid out after
     this line runs, which reads on screen as an off-centre, stretched scene. */
  renderer.setSize(width(), height(), false);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.02;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.setClearAlpha(0);
  renderer.domElement.setAttribute('aria-hidden', 'true');
  stage.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(34, width() / height(), 0.05, 400);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.075;
  controls.enablePan = false;              /* you cannot lose the device */
  controls.minPolarAngle = 0.16;
  controls.maxPolarAngle = Math.PI * 0.49; /* never go under the ground plane */
  controls.autoRotate = !reduced && !!(cfg.motion && cfg.motion.autoRotate);
  controls.autoRotateSpeed = cfg.scope === 'cell' ? 0.55 : 0.34;

  /* --- lighting: a light studio, to match a light page ------------------ */
  scene.add(new THREE.HemisphereLight(0xffffff, 0xd7e2f0, 1.5));
  const key = new THREE.DirectionalLight(0xffffff, 2.35);
  key.position.set(-6, 10, 6.5);
  key.castShadow = true;
  key.shadow.mapSize.set(1024, 1024);
  key.shadow.camera.near = 1;
  key.shadow.camera.far = 42;
  key.shadow.camera.left = -9; key.shadow.camera.right = 9;
  key.shadow.camera.top = 9; key.shadow.camera.bottom = -9;
  key.shadow.bias = -0.0012;
  scene.add(key);
  const fill = new THREE.DirectionalLight(0xdbeafe, 0.85);
  fill.position.set(7, 4, -6);
  scene.add(fill);
  const rim = new THREE.DirectionalLight(0xffffff, 0.5);
  rim.position.set(0, -5, -4);
  scene.add(rim);

  const root = new THREE.Group();
  scene.add(root);

  /* --- ground: shadow catcher + faint grid ------------------------------ */
  const groundY = -0.55;
  const shadowPlane = new THREE.Mesh(
    new THREE.PlaneGeometry(60, 60),
    new THREE.ShadowMaterial({ opacity: 0.17 })
  );
  shadowPlane.rotation.x = -Math.PI / 2;
  shadowPlane.position.y = groundY;
  shadowPlane.receiveShadow = true;
  scene.add(shadowPlane);

  const grid = new THREE.GridHelper(48, 48, new THREE.Color(P.borderStrong || '#CBD5E1'), new THREE.Color(P.border || '#E3E8EF'));
  grid.material.transparent = true;
  grid.material.opacity = 0.55;
  grid.position.y = groundY + 0.001;
  scene.add(grid);

  /* --- shared state the timeline mutates ------------------------------- */
  const dyn = {
    absorberMat: null,
    metalMat: null,
    interfaceMats: [],
    edgeBars: [],
    edgeMat: null,
    defects: null,
    defectsHigh: null,
    defectMax: 0,
    beams: [],
    plateW: 3.5,
    plateD: 2.4
  };

  /* ====================================================================== */
  /* CELL SCOPE                                                             */
  /* ====================================================================== */
  function makeLabel(name, sub) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2) * 2;
    const w = 320, h = 84;
    const canvas = document.createElement('canvas');
    canvas.width = w * dpr; canvas.height = h * dpr;
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);
    ctx.font = '700 27px Inter, system-ui, -apple-system, "Segoe UI", sans-serif';
    ctx.fillStyle = P.text || '#0F172A';
    ctx.textBaseline = 'top';
    ctx.fillText(name, 0, 4);
    ctx.font = '500 21px Inter, system-ui, -apple-system, "Segoe UI", sans-serif';
    ctx.fillStyle = P.textFaint || '#94A3B8';
    ctx.fillText(sub, 0, 40);
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.anisotropy = 4;
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
      map: texture, transparent: true, depthTest: false, depthWrite: false
    }));
    sprite.scale.set(1.28, 0.336, 1);
    sprite.renderOrder = 20;
    return sprite;
  }

  function defectTexture() {
    const s = 64;
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = s;
    const ctx = canvas.getContext('2d');
    const g = ctx.createRadialGradient(s / 2, s / 2, 0, s / 2, s / 2, s / 2);
    g.addColorStop(0, 'rgba(255,255,255,1)');
    g.addColorStop(0.34, 'rgba(255,255,255,0.92)');
    g.addColorStop(1, 'rgba(255,255,255,0)');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, s, s);
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    return texture;
  }

  function buildCell() {
    const layers = cfg.layers || [];
    const W = 3.5, D = 2.4;
    dyn.plateW = W; dyn.plateD = D;
    const planned = !!cfg.planned;
    let absorber = null;
    const labelBits = [];

    layers.forEach((layer, index) => {
      const h = Math.max(0.02, Number(layer.draw_h) || 0.08);
      const y = Number(layer.draw_y) || 0;
      const isCover = layer.role === 'cover';

      const material = planned
        ? new THREE.MeshStandardMaterial({
            color: new THREE.Color(PLANNED_COLOR),
            transparent: true, opacity: 0.16, roughness: 0.8, metalness: 0.0
          })
        : new THREE.MeshPhysicalMaterial({
            color: new THREE.Color(layer.color || '#94A3B8'),
            transparent: (layer.opacity ?? 1) < 1,
            opacity: layer.opacity ?? 1,
            metalness: layer.metalness ?? 0,
            roughness: layer.roughness ?? 0.35,
            transmission: layer.transmission ?? 0,
            thickness: isCover ? 0.6 : 0.1,
            ior: isCover ? 1.5 : 1.9,
            clearcoat: isCover || layer.role === 'front_contact' ? 0.9 : 0.15,
            clearcoatRoughness: 0.06,
            envMapIntensity: 0.7
          });

      const mesh = new THREE.Mesh(new THREE.BoxGeometry(W, h, D), material);
      mesh.position.y = y;
      mesh.name = layer.name || ('layer' + index);
      mesh.castShadow = !isCover && !planned;
      mesh.receiveShadow = !planned;
      root.add(mesh);

      const edges = new THREE.LineSegments(
        new THREE.EdgesGeometry(mesh.geometry),
        new THREE.LineBasicMaterial({
          color: new THREE.Color(planned ? PLANNED_COLOR : (P.text || '#0F172A')),
          transparent: true,
          opacity: planned ? 0.75 : 0.16
        })
      );
      edges.position.copy(mesh.position);
      root.add(edges);

      if (layer.role === 'absorber') {
        dyn.absorberMat = material;
        absorber = { y: y, h: h };
      }
      if (layer.role === 'back_contact') { dyn.metalMat = material; }
      labelBits.push({ layer: layer, y: y, h: h });
    });

    /* leader lines + floating labels, cell scope only */
    if (!planned) {
      labelBits.forEach((bit) => {
        const x0 = W / 2;
        const x1 = W / 2 + 0.62;
        const line = new THREE.Line(
          new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(x0, bit.y, D / 2 * 0.42),
            new THREE.Vector3(x1, bit.y, D / 2 * 0.42)
          ]),
          new THREE.LineBasicMaterial({ color: new THREE.Color(P.borderStrong || '#CBD5E1'), transparent: true, opacity: 0.9 })
        );
        line.renderOrder = 19;
        root.add(line);
        const label = makeLabel(bit.layer.name || '', bit.layer.thickness_label || '');
        label.position.set(x1 + 0.66, bit.y, D / 2 * 0.42);
        root.add(label);
      });
    }

    if (!absorber || planned) { return; }

    /* ---- edge ingress: a frame creeping in from the perimeter ----------
       Unit boxes so the timeline can rescale the width without rebuilding. */
    dyn.edgeMat = new THREE.MeshPhysicalMaterial({
      color: new THREE.Color(SPENT_PVK),
      roughness: 0.86, metalness: 0.0, transparent: true, opacity: 0.85
    });
    const bars = [
      { axis: 'z', sign: 1 }, { axis: 'z', sign: -1 },
      { axis: 'x', sign: 1 }, { axis: 'x', sign: -1 }
    ];
    bars.forEach((spec) => {
      const bar = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), dyn.edgeMat);
      bar.castShadow = true;
      bar.userData = spec;
      bar.position.y = absorber.y;
      root.add(bar);
      dyn.edgeBars.push(bar);
    });
    dyn.absorberBox = absorber;

    /* ---- interface darkening at the two transport contacts ------------- */
    [absorber.y + absorber.h / 2, absorber.y - absorber.h / 2].forEach((y) => {
      const mat = new THREE.MeshBasicMaterial({
        color: new THREE.Color(P.text || '#0F172A'), transparent: true, opacity: 0.0, depthWrite: false
      });
      const skin = new THREE.Mesh(new THREE.BoxGeometry(W * 1.002, 0.018, D * 1.002), mat);
      skin.position.y = y;
      root.add(skin);
      dyn.interfaceMats.push(mat);
    });

    /* ---- defect cloud inside the absorber ------------------------------
       Built once at the worst density this config will ever show, then
       revealed with setDrawRange. Points beyond the mean count represent the
       extra defects implied by +1 sigma of retention uncertainty. */
    const frames = cfg.frames || [];
    let peak = clamp01((cfg.appearance || {}).defectDensityHigh || 0);
    frames.forEach((f) => { peak = Math.max(peak, clamp01((f.appearance || {}).defectDensityHigh || 0)); });
    const maxCount = Math.round(60 + 1500 * peak);
    dyn.defectMax = maxCount;

    const positions = new Float32Array(maxCount * 3);
    const edgeBias = clamp01((cfg.appearance || {}).edgeIngress || 0);
    for (let i = 0; i < maxCount; i += 1) {
      const onEdge = seeded(i, 1) < 0.18 + 0.5 * edgeBias;
      const u = seeded(i, 2), v = seeded(i, 3), w = seeded(i, 4);
      let x, z;
      if (onEdge) {
        const side = seeded(i, 5) < 0.5;
        const t = 0.5 - 0.5 * Math.pow(seeded(i, 6), 2.1);
        x = side ? (u < 0.5 ? -1 : 1) * W * t : (u - 0.5) * W * 0.95;
        z = side ? (v - 0.5) * D * 0.95 : (v < 0.5 ? -1 : 1) * D * t;
      } else {
        x = (u - 0.5) * W * 0.9;
        z = (v - 0.5) * D * 0.9;
      }
      positions[i * 3] = x;
      positions[i * 3 + 1] = absorber.y + (w - 0.5) * absorber.h * 0.86;
      positions[i * 3 + 2] = z;
    }

    const sprite = defectTexture();
    function cloud(color, opacity, size) {
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      const points = new THREE.Points(geometry, new THREE.PointsMaterial({
        color: new THREE.Color(color), size: size, sizeAttenuation: true,
        map: sprite, transparent: true, opacity: opacity, depthWrite: false
      }));
      points.renderOrder = 12;
      root.add(points);
      return points;
    }
    dyn.defectsHigh = cloud(DEFECT_COLOR, 0.30, 0.052);
    dyn.defects = cloud(DEFECT_COLOR, 0.92, 0.05);

    /* ---- incoming light shafts (environment cue only) ------------------ */
    const flux = clamp01((cfg.appearance || {}).photonFlux || 0);
    const beamCount = Math.round(4 + 8 * flux);
    const beamMat = new THREE.MeshBasicMaterial({
      color: new THREE.Color('#F6C86B'), transparent: true,
      opacity: 0.05 + 0.12 * flux, depthWrite: false, side: THREE.DoubleSide
    });
    for (let i = 0; i < beamCount; i += 1) {
      const beam = new THREE.Mesh(new THREE.PlaneGeometry(0.055 + 0.05 * flux, 3.4), beamMat.clone());
      beam.position.set((seeded(i, 7) - 0.5) * W * 0.95, layersTop(layers) + 1.55, (seeded(i, 8) - 0.5) * D * 0.9);
      beam.rotation.set(0, seeded(i, 9) * Math.PI, 0.20);
      beam.userData.phase = seeded(i, 10) * Math.PI * 2;
      root.add(beam);
      dyn.beams.push(beam);
    }
  }

  function layersTop(layers) {
    let top = 0;
    (layers || []).forEach((l) => { top = Math.max(top, (Number(l.draw_y) || 0) + (Number(l.draw_h) || 0) / 2); });
    return top;
  }

  /* ====================================================================== */
  /* PLANNED SCOPES -- module and farm                                      */
  /* Wireframe, grey, no per-cell anything. Deliberately unfinished-looking. */
  /* ====================================================================== */
  function plannedMaterials() {
    return {
      face: new THREE.MeshStandardMaterial({
        color: new THREE.Color(PLANNED_COLOR), transparent: true, opacity: 0.13,
        roughness: 0.9, metalness: 0.0, side: THREE.DoubleSide
      }),
      line: new THREE.LineBasicMaterial({
        color: new THREE.Color(PLANNED_COLOR), transparent: true, opacity: 0.7
      }),
      lineFaint: new THREE.LineBasicMaterial({
        color: new THREE.Color(PLANNED_COLOR), transparent: true, opacity: 0.34
      })
    };
  }

  function wirePlate(parent, w, h, d, mats, faint) {
    const geometry = new THREE.BoxGeometry(w, h, d);
    const mesh = new THREE.Mesh(geometry, mats.face);
    parent.add(mesh);
    const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geometry), faint ? mats.lineFaint : mats.line);
    parent.add(edges);
    return mesh;
  }

  function buildModule() {
    const mats = plannedMaterials();
    const panel = new THREE.Group();
    root.add(panel);
    const cols = 6, rows = 10, cw = 0.62, ch = 0.42, gap = 0.045;
    for (let r = 0; r < rows; r += 1) {
      for (let c = 0; c < cols; c += 1) {
        const cell = new THREE.Group();
        cell.position.set(
          (c - (cols - 1) / 2) * (cw + gap),
          0.02,
          (r - (rows - 1) / 2) * (ch + gap)
        );
        panel.add(cell);
        wirePlate(cell, cw, 0.035, ch, mats, false);
      }
    }
    /* interconnect ribbons -- drawn, not solved */
    for (let c = 0; c < cols - 1; c += 1) {
      const x = ((c - (cols - 1) / 2) + 0.5) * (cw + gap);
      const line = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(x, 0.06, -rows * (ch + gap) / 2),
          new THREE.Vector3(x, 0.06, rows * (ch + gap) / 2)
        ]), mats.lineFaint);
      panel.add(line);
    }
    wirePlate(panel, cols * (cw + gap) + 0.22, 0.05, rows * (ch + gap) + 0.22, mats, true);
    panel.rotation.x = -0.30;
    panel.position.y = 0.5;
  }

  function buildFarm() {
    const mats = plannedMaterials();
    const rows = 4, perRow = 5;
    for (let r = 0; r < rows; r += 1) {
      for (let c = 0; c < perRow; c += 1) {
        const table = new THREE.Group();
        table.position.set((c - (perRow - 1) / 2) * 2.35, 0.42, (r - (rows - 1) / 2) * 2.6);
        table.rotation.x = -0.55;
        root.add(table);
        wirePlate(table, 2.05, 0.05, 1.15, mats, false);
        /* legs */
        [-0.85, 0.85].forEach((dx) => {
          const leg = new THREE.Line(
            new THREE.BufferGeometry().setFromPoints([
              new THREE.Vector3(dx, 0, 0), new THREE.Vector3(dx, -0.55, -0.35)
            ]), mats.lineFaint);
          table.add(leg);
        });
      }
    }
    /* an inverter, also unbuilt */
    const inverter = new THREE.Group();
    inverter.position.set(6.6, 0.35, 0);
    root.add(inverter);
    wirePlate(inverter, 0.7, 0.9, 0.45, mats, false);
    const ground = new THREE.Mesh(
      new THREE.PlaneGeometry(26, 18),
      new THREE.MeshStandardMaterial({ color: new THREE.Color(P.surfaceSunk || '#F1F5F9'), roughness: 1.0 })
    );
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = groundY + 0.002;
    ground.receiveShadow = true;
    scene.add(ground);
  }

  /* ====================================================================== */
  /* appearance application -- the ONLY place damage is expressed           */
  /* ====================================================================== */
  function applyAppearance(a) {
    const bleach = clamp01(a.bleach);
    const contact = clamp01(a.contactDarkening);
    const edge = clamp01(a.edgeIngress);
    const density = clamp01(a.defectDensity);
    const densityHigh = Math.max(density, clamp01(a.defectDensityHigh));

    if (dyn.absorberMat) {
      dyn.absorberMat.color.copy(new THREE.Color(FRESH_PVK).lerp(new THREE.Color(SPENT_PVK), bleach));
      dyn.absorberMat.roughness = lerp(0.34, 0.86, bleach);
    }
    if (dyn.metalMat) {
      dyn.metalMat.color.copy(new THREE.Color('#CBD5E1').lerp(new THREE.Color('#4B5563'), contact));
      dyn.metalMat.roughness = lerp(0.18, 0.62, contact);
      dyn.metalMat.metalness = lerp(1.0, 0.55, contact);
    }
    dyn.interfaceMats.forEach((mat) => { mat.opacity = 0.04 + 0.5 * contact; });

    if (dyn.edgeBars.length && dyn.absorberBox) {
      const W = dyn.plateW, D = dyn.plateD;
      const inset = Math.max(0.001, (0.02 + 0.16 * edge) * Math.min(W, D));
      const h = dyn.absorberBox.h * 1.03;
      dyn.edgeBars.forEach((bar) => {
        const spec = bar.userData;
        if (spec.axis === 'z') {
          bar.scale.set(W * 1.004, h, inset);
          bar.position.set(0, dyn.absorberBox.y, spec.sign * (D / 2 - inset / 2));
        } else {
          bar.scale.set(inset, h, D * 1.004 - 2 * inset);
          bar.position.set(spec.sign * (W / 2 - inset / 2), dyn.absorberBox.y, 0);
        }
        bar.visible = edge > 0.005;
      });
      if (dyn.edgeMat) { dyn.edgeMat.opacity = 0.45 + 0.45 * edge; }
    }

    if (dyn.defects && dyn.defectMax > 0) {
      const n = Math.round(dyn.defectMax * (density / Math.max(densityHigh, 1e-6)) * (densityHigh > 0 ? 1 : 0));
      const meanCount = Math.min(dyn.defectMax, Math.max(0, Math.round(60 * density + 1500 * density * density / Math.max(density, 1e-6) * 0)) || n);
      const shown = Math.min(dyn.defectMax, Math.round(60 * (density > 0 ? 1 : 0) + 1500 * density));
      const shownHigh = Math.min(dyn.defectMax, Math.round(60 * (densityHigh > 0 ? 1 : 0) + 1500 * densityHigh));
      dyn.defects.geometry.setDrawRange(0, shown);
      dyn.defectsHigh.geometry.setDrawRange(0, shownHigh);
      dyn.defects.visible = shown > 0;
      dyn.defectsHigh.visible = shownHigh > shown;
      void meanCount;
    }
  }

  /* ====================================================================== */
  /* build                                                                  */
  /* ====================================================================== */
  if (cfg.scope === 'module') { buildModule(); }
  else if (cfg.scope === 'farm') { buildFarm(); }
  else { buildCell(); }

  applyAppearance(cfg.appearance || {});

  /* --- framing --------------------------------------------------------- */
  const box = new THREE.Box3().setFromObject(root);
  const sphere = box.getBoundingSphere(new THREE.Sphere());
  function frameScene() {
    const aspect = width() / height();
    camera.aspect = aspect;
    const fov = camera.fov * Math.PI / 180;
    let dist = (sphere.radius / 0.78) / Math.tan(fov / 2);
    if (aspect < 1.55) { dist *= 1.55 / Math.max(aspect, 0.55); }
    const dir = new THREE.Vector3(0.82, 0.46, 1).normalize();
    camera.position.copy(sphere.center).addScaledVector(dir, dist);
    controls.target.copy(sphere.center);
    controls.minDistance = dist * 0.42;
    controls.maxDistance = dist * 2.4;
    camera.updateProjectionMatrix();
    controls.update();
  }
  frameScene();

  /* Re-frame until the user takes control. The container is frequently still
     being laid out when main() runs -- inside a Streamlit iframe it always is --
     so the first frameScene() sees the wrong aspect. Once somebody orbits, stop
     stealing their camera and only keep the projection honest. */
  let userTookControl = false;
  controls.addEventListener('start', () => { userTookControl = true; });

  function resize() {
    renderer.setSize(width(), height(), false);
    camera.aspect = width() / height();
    camera.updateProjectionMatrix();
    if (!userTookControl) { frameScene(); }
  }
  window.addEventListener('resize', resize);
  if (window.ResizeObserver) { new ResizeObserver(resize).observe(stage); }
  /* Two frames is enough for the iframe to settle at its final size. */
  requestAnimationFrame(() => requestAnimationFrame(resize));

  /* --- keyboard orbit, so the canvas is not mouse-only ------------------ */
  stage.addEventListener('keydown', (event) => {
    const step = 0.16;
    if (event.key === 'ArrowLeft') { controls.autoRotate = false; rotateBy(-step); }
    else if (event.key === 'ArrowRight') { controls.autoRotate = false; rotateBy(step); }
    else if (event.key === 'ArrowUp') { controls.autoRotate = false; pitchBy(-step * 0.5); }
    else if (event.key === 'ArrowDown') { controls.autoRotate = false; pitchBy(step * 0.5); }
    else { return; }
    event.preventDefault();
  });
  function rotateBy(delta) {
    const offset = camera.position.clone().sub(controls.target);
    const spherical = new THREE.Spherical().setFromVector3(offset);
    spherical.theta += delta;
    camera.position.copy(controls.target).add(new THREE.Vector3().setFromSpherical(spherical));
    controls.update();
  }
  function pitchBy(delta) {
    const offset = camera.position.clone().sub(controls.target);
    const spherical = new THREE.Spherical().setFromVector3(offset);
    spherical.phi = Math.max(0.18, Math.min(Math.PI * 0.48, spherical.phi + delta));
    camera.position.copy(controls.target).add(new THREE.Vector3().setFromSpherical(spherical));
    controls.update();
  }

  /* ====================================================================== */
  /* timeline -- interpolation between GIVEN states, never beyond them      */
  /* ====================================================================== */
  const frames = Array.isArray(cfg.frames) ? cfg.frames : [];
  const tlRange = document.getElementById('tl-range');
  const tlPlay = document.getElementById('tl-play');
  const tlReadout = document.getElementById('tl-readout');
  let playing = false;
  let position = frames.length ? frames.length - 1 : 0;

  function blend(a, b, t) {
    const out = {};
    const keys = new Set([...Object.keys(a || {}), ...Object.keys(b || {})]);
    keys.forEach((k) => { out[k] = lerp(Number(a[k]) || 0, Number(b[k]) || 0, t); });
    return out;
  }
  function setText(id, value) {
    const node = document.getElementById(id);
    if (node) { node.textContent = value; }
  }
  function fixed(v, n) { return (Number(v) || 0).toFixed(n); }

  function seek(pos) {
    if (frames.length < 2) { return; }
    position = Math.max(0, Math.min(frames.length - 1, pos));
    const i = Math.min(frames.length - 2, Math.floor(position));
    const t = position - i;
    const A = frames[i], B = frames[i + 1];
    const appearance = blend(A.appearance, B.appearance, t);
    applyAppearance(appearance);

    const ageH = lerp(A.aging_h, B.aging_h, t);
    if (tlReadout) { tlReadout.textContent = Math.round(ageH) + ' h'; }
    if (tlRange && Math.abs(Number(tlRange.value) - position) > 1e-6) { tlRange.value = String(position); }
    setText('hud-age', Math.round(ageH).toLocaleString() + ' h');

    if (A.metrics && B.metrics) {
      const m = blend(A.metrics, B.metrics, t);
      const pm = (v, sd, n, unit) => fixed(v, n) + (sd > 0 ? ' ± ' + fixed(sd, n) : '') + (unit ? ' ' + unit : '');
      setText('hud-pce', pm(m.pce_pct, m.pce_sd, 2, '%'));
      setText('hud-ret', pm(m.retention_pct, m.retention_sd, 1, '%'));
      setText('hud-voc', pm(m.voc_v, m.voc_sd, 3, 'V'));
      setText('hud-jsc', pm(m.jsc_macm2, m.jsc_sd, 2, 'mA/cm2'));
      setText('hud-ff', pm(m.ff, m.ff_sd, 3, ''));
      setText('hud-health', m.retention_pct >= 90 ? 'Above T90' : (m.retention_pct >= 80 ? 'Between T90 and T80' : 'Below T80'));
    }
    ['defectDensity', 'contactDarkening', 'edgeIngress', 'bleach'].forEach((k) => {
      const bar = document.getElementById('bar-' + k);
      const pct = document.getElementById('pct-' + k);
      const v = clamp01(appearance[k]);
      if (bar) { bar.style.width = (v * 100).toFixed(1) + '%'; }
      if (pct) { pct.textContent = Math.round(v * 100) + '%'; }
    });
  }

  if (frames.length > 1) {
    if (tlRange) {
      tlRange.addEventListener('input', () => { playing = false; if (tlPlay) { tlPlay.textContent = 'Play'; } seek(Number(tlRange.value)); });
    }
    if (tlPlay) {
      tlPlay.addEventListener('click', () => {
        if (!playing && position >= frames.length - 1) { position = 0; }
        playing = !playing;
        tlPlay.textContent = playing ? 'Pause' : 'Play';
      });
    }
    seek(position);
  }

  /* ====================================================================== */
  /* loop                                                                   */
  /* ====================================================================== */
  const clock = new THREE.Clock();
  let last = 0;
  function animate() {
    requestAnimationFrame(animate);
    const t = clock.getElapsedTime();
    const dt = Math.min(0.05, t - last);
    last = t;

    if (playing && frames.length > 1) {
      /* 6 seconds to walk the whole predicted sequence, then stop dead at
         the last predicted state -- it does not loop into invented futures */
      position += dt * (frames.length - 1) / 6.0;
      if (position >= frames.length - 1) { position = frames.length - 1; playing = false; if (tlPlay) { tlPlay.textContent = 'Play'; } }
      seek(position);
    }

    if (!reduced && cfg.motion && cfg.motion.shimmer) {
      /* bounded cosmetic shimmer. Amplitudes are constants; none of these
         touch a degradation quantity. */
      if (dyn.defects) { dyn.defects.material.opacity = 0.86 + 0.08 * Math.sin(t * 1.6); }
      dyn.beams.forEach((beam, i) => {
        beam.material.opacity = beam.material.opacity;
        beam.position.y += Math.sin(t * 0.9 + (beam.userData.phase || 0)) * 0.0006;
        void i;
      });
    }

    controls.update();
    renderer.render(scene, camera);
  }
  animate();

  window.TWIN_DIAG = {
    scope: cfg.scope,
    tier: cfg.tier,
    triangles: renderer.info.render.triangles,
    canvasWidth: renderer.domElement.width,
    canvasHeight: renderer.domElement.height
  };
  window.twinBooted();
}

try {
  main();
} catch (err) {
  const message = (err && err.message) ? err.message : String(err);
  window.twinFail('The 3D scene could not start in this browser: ' + message
    + ' WebGL may be unavailable or disabled.');
}
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------
# 3. sample renders, so the scene can be eyeballed without Streamlit
# --------------------------------------------------------------------------
def _sample_states() -> list[tuple[str, dict[str, Any]]]:
    """Three points on one plausible 1-sun / 65 C aging trajectory."""
    return [
        (
            "twin3d_cell_fresh.html",
            build_twin_config(
                architecture=ARCHITECTURE_VALIDATED,
                illumination_suns=1.0,
                temperature_c=26.85,
                aging_h=0.0,
                pce_pct=20.84,
                retention_pct=100.0,
                voc_v=1.132,
                jsc_macm2=23.41,
                ff=0.786,
                pce_sd=0.21,
                retention_sd=0.0,
                voc_sd=0.006,
                jsc_sd=0.18,
                ff_sd=0.004,
                tier=Tier.VALIDATED,
                scope=SCOPE_CELL,
                title="Fresh cell",
            ),
        ),
        (
            "twin3d_cell_midlife.html",
            build_twin_config(
                architecture=ARCHITECTURE_VALIDATED,
                illumination_suns=1.0,
                temperature_c=66.85,
                aging_h=400.0,
                pce_pct=17.02,
                retention_pct=81.7,
                voc_v=1.081,
                jsc_macm2=22.63,
                ff=0.696,
                pce_sd=0.44,
                retention_sd=2.1,
                voc_sd=0.011,
                jsc_sd=0.31,
                ff_sd=0.012,
                tier=Tier.VALIDATED,
                scope=SCOPE_CELL,
                title="Mid-life cell",
            ),
        ),
        (
            "twin3d_cell_degraded.html",
            build_twin_config(
                architecture=ARCHITECTURE_VALIDATED,
                illumination_suns=1.0,
                temperature_c=106.85,
                aging_h=1000.0,
                pce_pct=9.63,
                retention_pct=46.2,
                voc_v=0.968,
                jsc_macm2=20.11,
                ff=0.495,
                pce_sd=0.91,
                retention_sd=4.4,
                voc_sd=0.022,
                jsc_sd=0.55,
                ff_sd=0.024,
                tier=Tier.VALIDATED,
                scope=SCOPE_CELL,
                title="Heavily degraded cell",
            ),
        ),
    ]


def _main() -> int:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    out_dir = root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # keep the duplicated palette honest whenever theme.py is importable
    try:
        from psc_twin.ui import theme  # noqa: PLC0415

        assert PALETTE["bg"] == theme.BG
        assert PALETTE["surface"] == theme.SURFACE
        assert PALETTE["border"] == theme.BORDER
        assert PALETTE["text"] == theme.TEXT
        assert PALETTE["primary"] == theme.PRIMARY
        assert PALETTE["validated"] == theme.TIER_COLOR[Tier.VALIDATED]
        assert PALETTE["preview"] == theme.TIER_COLOR[Tier.PREVIEW]
        assert PALETTE["planned"] == theme.TIER_COLOR[Tier.PLANNED]
        print("palette check: twin3d.PALETTE matches psc_twin.ui.theme")
    except Exception as exc:  # streamlit not installed, or drift
        print(f"palette check skipped/failed: {exc}")

    written: list[Path] = []
    states = _sample_states()
    for filename, config in states:
        path = out_dir / filename
        path.write_text(build_twin_html(config), encoding="utf-8")
        written.append(path)

    # the two planned scopes, so the unbuilt treatment can be eyeballed too
    for filename, scope in (
        ("twin3d_module_planned.html", SCOPE_MODULE),
        ("twin3d_farm_planned.html", SCOPE_FARM),
    ):
        config = build_twin_config(
            architecture=ARCHITECTURE_VALIDATED,
            illumination_suns=1.0,
            temperature_c=66.85,
            aging_h=400.0,
            pce_pct=17.02,
            retention_pct=81.7,
            voc_v=1.081,
            jsc_macm2=22.63,
            ff=0.696,
            tier=Tier.VALIDATED,   # deliberately optimistic; the registry overrules it
            scope=scope,
        )
        assert config["metrics"] is None, "a planned scope must never carry numbers"
        path = out_dir / filename
        path.write_text(build_twin_html(config), encoding="utf-8")
        written.append(path)

    # the timeline, interpolating only between the three predicted states
    timeline = build_twin_timeline([config for _, config in states])
    timeline["title"] = "Predicted aging sequence"
    path = out_dir / "twin3d_cell_timeline.html"
    path.write_text(build_twin_html(timeline), encoding="utf-8")
    written.append(path)

    # the CDN-failure path, rendered against a host that cannot resolve
    path = out_dir / "twin3d_cdn_failure.html"
    path.write_text(
        build_twin_html(
            states[2][1],
            three_module_url="https://cdn.invalid.example/three.module.js",
            three_addons_url="https://cdn.invalid.example/addons/",
        ),
        encoding="utf-8",
    )
    written.append(path)

    for item in written:
        print(f"wrote {item}  ({item.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
