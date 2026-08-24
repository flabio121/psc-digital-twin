"""The climate layer: turning a place into a stress the surrogate understands.

The surrogate in this app was trained on a lab-stress campaign -- 36 COMSOL
runs on a 6 x 6 grid of (illumination, temperature), held isothermal and
iso-illumination for up to 1000 h. A real deployment is nothing like that. It
is a repeating daily cycle of light and dark, riding on a seasonal cycle of
ambient temperature, running for decades. This module is the bridge, and it is
deliberately the most heavily caveated file in the package.

WHAT THIS MODULE IS
    A set of CLIMATE ARCHETYPES: hand-specified, internally consistent monthly
    profiles that stand in for genuinely different degradation regimes (hot and
    dry, hot and wet, cold and dim, high-altitude and bright). They exist so a
    reader can ask "does the physics behave differently in Phoenix and in
    Anchorage?" and get a defensible qualitative answer.

WHAT THIS MODULE IS NOT
    It is NOT weather data. Not a typical meteorological year, not a satellite
    reanalysis, not a station record. No value in this file was measured, and
    none is traceable to a named site. Every archetype carries a ``source``
    field that says so in its first sentence, ``archetype_table()`` repeats the
    disclaimer on every exported row, and ``PROVENANCE_NOTICE`` is the single
    string the UI must render next to any climate result. If you ever need
    real provenance, that is capability ``twin_farm`` (Tier.PLANNED), which is
    explicitly gated on "measured typical-meteorological-year files".

THE CHAIN OF ASSUMPTIONS, IN ORDER
    1. Monthly mean ambient temperature + a monthly mean diurnal range give a
       daytime mean and a night mean ambient (sinusoidal day, see
       ``_SINUSOID_HALF_MEAN``).
    2. Astronomical daylight hours come from latitude and Cooper's declination
       formula -- no cloud, no horizon, no terrain.
    3. Monthly mean daily irradiation (kWh/m2/day, plane-of-array) is spread
       over those daylight hours to give an effective irradiance, hence an
       effective suns level. The spread is dose-preserving: the total
       equivalent full-sun hours in the schedule always equals the archetype's
       stated annual insolation, whatever ``diurnal_bins`` is set to.
    4. Irradiance plus ambient gives a cell temperature through the standard
       NOCT model with NOCT = 45 C ASSUMED (``NOCT_C``).
    5. The result is a table of (suns, cell temperature, hours) segments --
       exactly the coordinates the surrogate was trained in.

WHAT IS DELIBERATELY NOT HERE
    Damage accumulation. This module never chains one segment into the next,
    never carries a degradation state forward, and never predicts a
    performance number. Capability ``twin_climate`` is Tier.PREVIEW precisely
    because sequential state accumulation across a schedule is itself
    unvalidated: the COMSOL campaign only ever held a cell at ONE condition,
    so nothing in the training data says how to compose two conditions. Doing
    the composition here would smuggle an unvalidated model into a module that
    otherwise only does arithmetic on stated assumptions. Callers that want a
    trajectory must own that step and must label it PREVIEW.

WHAT THE ENVELOPE REPORT WILL TELL YOU (and it is not flattering)
    ``envelope_report()`` runs every schedule segment through
    ``capabilities.envelope_excursions()``. Measured on the archetypes shipped
    here (see ``__main__``), the honest picture is:

    * Cell temperature NEVER reaches the 126.85 C training ceiling. With
      NOCT = 45 C the hot-desert archetype peaks near 55 C on a daylight-mean
      basis and near 65 C in the brightest diurnal bin. Claims that a climate
      twin is "extrapolating to high temperature" would be wrong here.
    * Cell temperature is instead far BELOW the 26.85 C training floor for
      most of the year in the cold archetypes, and for the night segments of
      nearly all of them. This is the temperature-side extrapolation, and it
      is the opposite of the one people expect.
    * Illumination in the dark segments is 0.0 suns, below the 0.01 suns
      floor. The surrogate has never seen true darkness.
    * The horizon is the dominant excursion. The training envelope stops at
      1000 h. One deployment year is 8760 h, so after roughly six weeks EVERY
      subsequent segment is a time extrapolation, and a 25-year forecast is
      219 000 h -- 219x beyond anything simulated.

    None of that is hidden or softened. The UI renders this report.

Nothing here imports Streamlit, every function is pure, and the module has no
randomness to seed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import capabilities
from .capabilities import ENVELOPE, TIER_LABEL, Tier, envelope_excursions

__all__ = [
    "PROVENANCE_NOTICE",
    "SIMPLIFICATIONS",
    "NOCT_C",
    "ClimateArchetype",
    "EnvelopeReport",
    "ARCHETYPES",
    "ARCHETYPE_KEYS",
    "get_archetype",
    "solar_declination_deg",
    "daylight_hours",
    "cell_temperature_noct",
    "to_stress_schedule",
    "envelope_report",
    "archetype_table",
    "monthly_table",
    "consistency_report",
    "climate_capability",
]


# ---------------------------------------------------------------------------
# Provenance. This string is the contract with the reader; the UI must show it
# anywhere a climate number appears.
# ---------------------------------------------------------------------------
PROVENANCE_NOTICE = (
    "REPRESENTATIVE CLIMATE ARCHETYPES -- FOR ILLUSTRATION ONLY. These monthly "
    "profiles were constructed by hand to span distinct degradation regimes. "
    "They are NOT measured weather data, NOT a typical meteorological year, and "
    "NOT traceable to any weather station. Do not report them as observations."
)

_SOURCE_PREFIX = (
    "Representative archetype for illustration only; constructed by hand, not "
    "measured, not a typical meteorological year, not tied to any station. "
    "Reasoning for the chosen values: "
)


def _source(reasoning: str) -> str:
    """Build a ``source`` string that always leads with the disclaimer."""
    return _SOURCE_PREFIX + reasoning


# ---------------------------------------------------------------------------
# The NOCT thermal model. Stated as an assumption because it is one.
# ---------------------------------------------------------------------------
NOCT_C = 45.0
"""Nominal operating cell temperature, ASSUMED. 45 C is a mid-range value for a
glass/backsheet module on an open rack. It is not measured for this device
stack, and a roof-parallel or building-integrated mounting would be hotter
(NOCT 48-55 C), a bifacial tracker cooler."""

NOCT_REFERENCE_AMBIENT_C = 20.0
NOCT_REFERENCE_IRRADIANCE_W_M2 = 800.0

MONTH_NAMES = (
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
)

# Fixed 365-day non-leap calendar. Leap years would move the annual dose by
# 0.3%, far below the uncertainty in the archetype values themselves.
DAYS_IN_MONTH = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
HOURS_PER_YEAR = 24.0 * sum(DAYS_IN_MONTH)  # 8760.0

# Klein's recommended mean days of each month: the day whose extraterrestrial
# irradiation is closest to the monthly mean. Using the 15th of every month
# instead would bias the solstice months.
KLEIN_MEAN_DAY_OF_YEAR = (17, 47, 75, 105, 135, 162, 198, 228, 258, 288, 318, 344)

# Mean of a sinusoid over the half cycle centred on its peak, as a fraction of
# the half amplitude: (1/(pi/2)) * integral of sin = 2/pi. Used to split a
# 24-hour mean ambient temperature into a daytime mean and a night mean given
# the diurnal range.
_SINUSOID_HALF_MEAN = 2.0 / math.pi

CLIMATE_CAPABILITY_KEY = "twin_climate"

SIMPLIFICATIONS: tuple[str, ...] = (
    "Climate values are hand-built archetypes, not measurements (see PROVENANCE_NOTICE).",
    "Irradiation is treated as already plane-of-array; no tilt/azimuth transposition "
    "model is applied and no albedo or bifacial gain is modelled.",
    "Daylight hours are astronomical (Cooper declination, flat horizon). Cloud, "
    "terrain shading and row-to-row shading are absorbed into the stated irradiation.",
    "The diurnal cycle is collapsed onto a half-sine irradiance profile discretised "
    "into `diurnal_bins` segments; with the default of 1 bin the whole daylight "
    "period becomes a SINGLE mean condition. Degradation is generally nonlinear in "
    "irradiance and temperature, so a mean condition is not equivalent to the cycle "
    "it replaces -- it is dose-preserving, not damage-preserving.",
    "Ambient temperature is split into a daytime and a night mean using a sinusoidal "
    "day of the stated diurnal range; within daylight, ambient is held flat.",
    "Cell temperature uses the NOCT model with NOCT = 45 C assumed, which ignores "
    "wind speed, mounting style and the power actually extracted from the cell.",
    "Every simulated year repeats the same climatological year: no interannual "
    "variability, no climate trend, no soiling, no snow cover, no maintenance.",
    "Relative humidity is carried through the schedule for context only. The "
    "surrogate has no humidity axis at all (capability 'stress_humidity').",
    "No damage accumulation, no state carried between segments. That is capability "
    "'twin_climate', Tier.PREVIEW, and it is not implemented in this module.",
)


# ---------------------------------------------------------------------------
# The archetype
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ClimateArchetype:
    """One representative operating climate. NOT a weather record.

    All twelve-element tuples are indexed January (0) to December (11) in
    calendar order, including for southern-hemisphere archetypes -- so a
    southern archetype has its warm, bright months at the ends of the tuple.

    Attributes
    ----------
    key
        Stable machine identifier, safe for URLs and export filenames.
    name
        Human label for the UI.
    koppen_class
        Koppen-Geiger class this archetype is built to resemble (BWh, Cfa, ...).
        The archetype approximates the class; it does not define it.
    description
        What degradation regime this archetype is here to represent, and where
        the numbers are rough. Approximations are called out in the text.
    latitude_band
        Human description of the latitude range this profile is drawn from.
    representative_latitude_deg
        The single latitude used to compute daylight hours, positive north.
        A band cannot be integrated, so one representative value is required.
    monthly_ambient_temperature_c
        Monthly mean of the full 24-hour ambient air temperature, degrees C.
    monthly_diurnal_range_c
        Monthly mean day-night ambient swing, degrees C. This is what separates
        a desert from a maritime climate at the same mean temperature, and it
        drives the day/night ambient split.
    monthly_irradiation_kwh_m2_day
        Monthly mean daily irradiation in the module plane, kWh/m2/day.
    monthly_relative_humidity_pct
        Monthly mean relative humidity, %. Carried for context; the surrogate
        cannot consume it.
    source
        Provenance. Always begins with the "for illustration only" disclaimer,
        then the reasoning behind the numbers.
    """

    key: str
    name: str
    koppen_class: str
    description: str
    latitude_band: str
    representative_latitude_deg: float
    monthly_ambient_temperature_c: tuple[float, ...]
    monthly_diurnal_range_c: tuple[float, ...]
    monthly_irradiation_kwh_m2_day: tuple[float, ...]
    monthly_relative_humidity_pct: tuple[float, ...]
    source: str

    def __post_init__(self) -> None:
        # Structural validation only. The physical plausibility check (can the
        # sun actually deliver this much energy in this many daylight hours?)
        # lives in consistency_report(), because it needs the solar geometry.
        for field_name in (
            "monthly_ambient_temperature_c",
            "monthly_diurnal_range_c",
            "monthly_irradiation_kwh_m2_day",
            "monthly_relative_humidity_pct",
        ):
            values = getattr(self, field_name)
            if len(values) != 12:
                raise ValueError(
                    f"{self.key}.{field_name} has {len(values)} values; 12 are required."
                )
        if not -90.0 <= self.representative_latitude_deg <= 90.0:
            raise ValueError(f"{self.key}: latitude out of range.")
        if any(v < 0.0 for v in self.monthly_diurnal_range_c):
            raise ValueError(f"{self.key}: negative diurnal range.")
        if any(v < 0.0 for v in self.monthly_irradiation_kwh_m2_day):
            raise ValueError(f"{self.key}: negative irradiation.")
        if any(not 0.0 <= v <= 100.0 for v in self.monthly_relative_humidity_pct):
            raise ValueError(f"{self.key}: relative humidity outside 0-100%.")
        if not self.source.startswith(_SOURCE_PREFIX):
            raise ValueError(
                f"{self.key}: source field must lead with the illustration-only "
                "disclaimer. Build it with _source()."
            )

    @property
    def annual_insolation_kwh_m2(self) -> float:
        """Plane-of-array insolation over one climatological year."""
        return float(
            sum(
                h * d
                for h, d in zip(self.monthly_irradiation_kwh_m2_day, DAYS_IN_MONTH)
            )
        )

    @property
    def annual_mean_temperature_c(self) -> float:
        """Day-weighted mean of the monthly 24-hour mean temperatures."""
        total_days = sum(DAYS_IN_MONTH)
        return float(
            sum(t * d for t, d in zip(self.monthly_ambient_temperature_c, DAYS_IN_MONTH))
            / total_days
        )

    @property
    def annual_mean_relative_humidity_pct(self) -> float:
        total_days = sum(DAYS_IN_MONTH)
        return float(
            sum(r * d for r, d in zip(self.monthly_relative_humidity_pct, DAYS_IN_MONTH))
            / total_days
        )

    def daylight_hours_by_month(self) -> tuple[float, ...]:
        """Astronomical daylight hours on each month's Klein mean day."""
        return tuple(
            daylight_hours(self.representative_latitude_deg, day)
            for day in KLEIN_MEAN_DAY_OF_YEAR
        )


# ---------------------------------------------------------------------------
# Solar geometry. Deterministic closed forms; no scipy, no tables.
# ---------------------------------------------------------------------------
def solar_declination_deg(day_of_year: int) -> float:
    """Solar declination via Cooper's approximation.

    Accurate to about 0.5 degrees, which moves daylight length by minutes --
    negligible against the uncertainty in a hand-built irradiation profile.
    """
    return 23.45 * math.sin(math.radians(360.0 * (284 + int(day_of_year)) / 365.0))


def daylight_hours(latitude_deg: float, day_of_year: int) -> float:
    """Hours between sunrise and sunset for a flat horizon.

    From the sunset hour angle, ``cos(ws) = -tan(lat) tan(decl)``. The cosine
    argument is clamped so polar night returns 0 h and polar day returns 24 h
    instead of raising; the archetypes shipped here stay below 62 degrees, but
    the function must not explode if a caller supplies a polar latitude.
    """
    phi = math.radians(float(latitude_deg))
    delta = math.radians(solar_declination_deg(day_of_year))
    cos_hour_angle = -math.tan(phi) * math.tan(delta)
    if cos_hour_angle >= 1.0:
        return 0.0
    if cos_hour_angle <= -1.0:
        return 24.0
    return (2.0 / 15.0) * math.degrees(math.acos(cos_hour_angle))


def cell_temperature_noct(
    ambient_temperature_c: float | np.ndarray,
    irradiance_w_m2: float | np.ndarray,
    noct_c: float = NOCT_C,
) -> float | np.ndarray:
    """Standard NOCT cell-temperature model.

        T_cell = T_ambient + (NOCT - 20) / 800 * G

    This is the step that converts a climate into a stress the surrogate
    understands: everything upstream is meteorology, everything downstream is
    a point in the COMSOL design space.

    ASSUMPTIONS, all of them consequential:

    * ``NOCT = 45 C`` (``NOCT_C``) is assumed, not measured for this stack.
      The 20 C ambient and 800 W/m2 irradiance in the formula are the fixed
      NOCT test conditions, not free parameters.
    * The model is linear in irradiance and carries no wind term. Real modules
      run several degrees cooler in wind and hotter when mounted close to a
      roof; that spread is not represented.
    * NOCT is defined at open circuit, so this returns an open-circuit cell
      temperature. A module operating at maximum power exports a few percent
      of the absorbed energy as electricity and sits roughly 1-3 C cooler.
      Here the open-circuit reading is the CONSISTENT choice rather than a
      conservative one: capability ``stress_bias`` records that the COMSOL
      aging campaign was itself simulated at open circuit only.
    * It is a steady-state model. Thermal mass, and therefore morning lag and
      evening carry-over, are ignored.
    """
    gain = (float(noct_c) - NOCT_REFERENCE_AMBIENT_C) / NOCT_REFERENCE_IRRADIANCE_W_M2
    return ambient_temperature_c + gain * irradiance_w_m2


def _half_sine_bin_irradiance(
    daily_irradiation_kwh_m2: float,
    daylight_h: float,
    bins: int,
) -> list[float]:
    """Mean irradiance (W/m2) in each of ``bins`` equal-length daylight slices.

    The daily irradiance profile is modelled as a half sine that rises at
    sunrise and sets at sunset, scaled so its integral reproduces the stated
    daily irradiation exactly. Each bin mean is the analytic integral of that
    half sine over the bin, so the discretisation is dose-preserving for any
    number of bins.

    With ``bins == 1`` this reduces exactly to ``H * 1000 / N``, the flat
    daylight-mean irradiance -- which is the reviewer-visible simplification
    the module docstring warns about. Raising ``bins`` recovers the shape of
    the day (and a hotter midday cell temperature) without changing the total
    energy dose.
    """
    if daylight_h <= 0.0 or bins < 1:
        return []
    peak = math.pi * 1000.0 * daily_irradiation_kwh_m2 / (2.0 * daylight_h)
    means: list[float] = []
    for i in range(bins):
        t_start = daylight_h * i / bins
        t_end = daylight_h * (i + 1) / bins
        integral = (daylight_h / math.pi) * (
            math.cos(math.pi * t_start / daylight_h)
            - math.cos(math.pi * t_end / daylight_h)
        )
        means.append(peak * integral / (t_end - t_start))
    return means


# ---------------------------------------------------------------------------
# The archetypes.
#
# Every number below was chosen by hand for internal consistency, not copied
# from a record. The pattern used throughout: pick a mean temperature season,
# pick a diurnal range that matches the humidity regime (dry air swings, moist
# air does not), then pick a plane-of-array irradiation whose annual total
# lands in the range normally associated with that climate class. The physical
# check that the sun can actually deliver the stated energy in the available
# daylight is in consistency_report().
# ---------------------------------------------------------------------------
HOT_DESERT = ClimateArchetype(
    key="hot_desert",
    name="Hot desert",
    koppen_class="BWh",
    description=(
        "The classic high-stress PV climate: extreme summer heat, very high "
        "insolation, dry air and a large day-night swing. This is the archetype "
        "that pushes cell temperature hardest. Approximation: the late-summer "
        "humidity bump that a monsoon-influenced desert shows is included in a "
        "smoothed form, and dust soiling -- a first-order effect in real "
        "deserts -- is not modelled at all."
    ),
    latitude_band="Subtropical, roughly 30-35 degrees N",
    representative_latitude_deg=33.5,
    monthly_ambient_temperature_c=(13, 15, 18, 23, 28, 33, 36, 35, 31, 24, 17, 12),
    monthly_diurnal_range_c=(14, 15, 16, 17, 17, 17, 14, 13, 15, 16, 15, 14),
    monthly_irradiation_kwh_m2_day=(
        5.0, 5.6, 6.4, 7.1, 7.3, 7.4, 6.8, 6.7, 6.6, 6.1, 5.2, 4.7,
    ),
    monthly_relative_humidity_pct=(40, 35, 30, 22, 18, 15, 28, 32, 30, 30, 36, 42),
    source=_source(
        "annual plane-of-array insolation set near 2270 kWh/m2 to represent a very "
        "clear subtropical desert with a latitude-tilted array; summer mean above "
        "35 C with a 13-17 C diurnal range because dry air radiates strongly at "
        "night; humidity held below 45% year round with a mid-summer rise to "
        "represent monsoon incursion."
    ),
)

HUMID_SUBTROPICAL = ClimateArchetype(
    key="humid_subtropical",
    name="Humid subtropical coastal",
    koppen_class="Cfa",
    description=(
        "Warm and persistently wet. Temperatures never reach desert extremes, "
        "but the cell is warm at night as well as by day and the air is near "
        "saturation year round -- the regime where moisture ingress, not heat, "
        "is expected to dominate. Approximation: convective cloud suppresses "
        "summer insolation below the spring peak, which is represented as a "
        "smooth dip rather than the day-to-day variability that actually causes it."
    ),
    latitude_band="Subtropical, roughly 25-30 degrees N",
    representative_latitude_deg=27.0,
    monthly_ambient_temperature_c=(20, 21, 23, 25, 27, 29, 30, 30, 29, 27, 24, 21),
    monthly_diurnal_range_c=(8, 8, 8, 8, 8, 7, 7, 7, 7, 7, 8, 8),
    monthly_irradiation_kwh_m2_day=(
        4.3, 4.9, 5.6, 6.2, 6.0, 5.6, 5.7, 5.6, 5.2, 5.0, 4.4, 4.0,
    ),
    monthly_relative_humidity_pct=(74, 72, 72, 71, 74, 79, 78, 79, 80, 78, 76, 75),
    source=_source(
        "annual plane-of-array insolation near 1900 kWh/m2, clearly below the desert "
        "case because summer convection clouds the brightest months; diurnal range "
        "held at 7-8 C because humid maritime air resists night-time radiative "
        "cooling; relative humidity kept in the 70-80% band all year, which is the "
        "defining feature of this archetype."
    ),
)

TEMPERATE_MARITIME = ClimateArchetype(
    key="temperate_maritime",
    name="Temperate maritime",
    koppen_class="Cfb",
    description=(
        "Northern-European conditions: mild, damp, cloudy, with a severe winter "
        "light deficit. The interesting regime for a degradation study because "
        "the cell spends most of the year well below any temperature the "
        "COMSOL campaign simulated. Approximation: persistent overcast is "
        "folded into the low stated irradiation, so the schedule shows a dim "
        "but continuous day rather than the alternation of bright and dark days "
        "that produces it."
    ),
    latitude_band="Mid-latitude, roughly 50-55 degrees N",
    representative_latitude_deg=52.0,
    monthly_ambient_temperature_c=(4, 4, 6, 9, 13, 16, 18, 18, 15, 11, 7, 5),
    monthly_diurnal_range_c=(6, 7, 8, 9, 10, 10, 10, 10, 9, 8, 6, 5),
    monthly_irradiation_kwh_m2_day=(
        1.1, 1.9, 3.0, 4.3, 5.0, 5.2, 5.0, 4.4, 3.4, 2.2, 1.2, 0.9,
    ),
    monthly_relative_humidity_pct=(87, 85, 80, 75, 74, 75, 76, 78, 82, 85, 88, 89),
    source=_source(
        "annual plane-of-array insolation near 1140 kWh/m2, the value typically "
        "associated with cloudy north-west European sites; a 6:1 summer-to-winter "
        "irradiation ratio driven jointly by short winter days at 52 degrees N and "
        "heavy winter cloud; small diurnal range and 74-89% humidity throughout "
        "because the air mass is maritime."
    ),
)

CONTINENTAL_SUBARCTIC = ClimateArchetype(
    key="continental_subarctic",
    name="Continental subarctic",
    koppen_class="Dfc",
    description=(
        "Long, dark, deeply sub-zero winters and short, mild, bright summers. "
        "This archetype exists to make the cold end of the extrapolation "
        "problem impossible to ignore: the cell is below the surrogate's "
        "26.85 C training floor for essentially the entire year. Approximation: "
        "snow cover, which both reflects extra light onto a tilted array and "
        "can bury it, is not modelled in either direction."
    ),
    latitude_band="High latitude, roughly 58-63 degrees N",
    representative_latitude_deg=61.0,
    monthly_ambient_temperature_c=(-8, -6, -2, 4, 10, 14, 16, 15, 10, 2, -5, -8),
    monthly_diurnal_range_c=(7, 8, 9, 10, 10, 10, 10, 9, 8, 7, 6, 6),
    monthly_irradiation_kwh_m2_day=(
        0.4, 1.3, 2.9, 4.3, 4.9, 5.2, 4.6, 3.6, 2.4, 1.2, 0.5, 0.2,
    ),
    monthly_relative_humidity_pct=(74, 73, 71, 67, 66, 68, 72, 76, 80, 78, 76, 75),
    source=_source(
        "annual plane-of-array insolation near 960 kWh/m2 concentrated almost "
        "entirely in April-August, because at 61 degrees N the December day is "
        "roughly five hours long and low-sun; winter means near -8 C with a "
        "modest diurnal range because a snow-covered surface under cloud damps "
        "the daily cycle."
    ),
)

TROPICAL_MONSOON = ClimateArchetype(
    key="tropical_monsoon",
    name="Tropical monsoon",
    koppen_class="Am",
    description=(
        "Hot all year with almost no seasonal temperature signal, but a violent "
        "seasonal swing in cloud and humidity. The bright, dry, high-swing "
        "winter and the dim, saturated, low-swing wet season are two different "
        "degradation regimes inside one archetype. Approximation: the wet "
        "season is represented as uniformly dim rather than as alternating "
        "downpours and bright intervals."
    ),
    latitude_band="Tropical, roughly 10-18 degrees N",
    representative_latitude_deg=13.0,
    monthly_ambient_temperature_c=(25, 27, 29, 30, 30, 28, 27, 27, 27, 27, 26, 25),
    monthly_diurnal_range_c=(11, 11, 10, 9, 8, 6, 5, 5, 6, 7, 9, 10),
    monthly_irradiation_kwh_m2_day=(
        5.6, 6.2, 6.5, 6.4, 5.6, 4.6, 4.4, 4.4, 4.6, 4.9, 5.2, 5.4,
    ),
    monthly_relative_humidity_pct=(62, 63, 68, 74, 80, 86, 88, 88, 87, 83, 73, 65),
    source=_source(
        "annual plane-of-array insolation near 1940 kWh/m2 with the peak in the "
        "pre-monsoon dry months rather than at the solstice, which is the "
        "signature of a monsoon climate; diurnal range collapses from 11 C in the "
        "dry season to 5 C under monsoon cloud and near-saturated air; temperature "
        "varies by only 5 C across the year because the latitude is low."
    ),
)

HIGH_ALTITUDE_ARID = ClimateArchetype(
    key="high_altitude_arid",
    name="High-altitude cold desert",
    koppen_class="BWk",
    description=(
        "A thin, dry, extremely clear atmosphere over a cold plateau: the "
        "highest irradiation of any archetype here combined with cold ambient "
        "air and the largest diurnal swing. It decouples light stress from heat "
        "stress, which is exactly the pairing the 6 x 6 factorial was built to "
        "separate. Southern hemisphere, so the calendar is inverted relative to "
        "the other archetypes. Approximation: the altitude effect is folded "
        "entirely into the irradiation and temperature values -- there is no "
        "explicit pressure, air-mass or spectral treatment, and the enhanced "
        "UV fraction of a high-altitude spectrum is not represented at all."
    ),
    latitude_band="Tropical southern hemisphere, roughly 15-22 degrees S, high plateau",
    representative_latitude_deg=-18.0,
    monthly_ambient_temperature_c=(10, 10, 10, 8, 5, 3, 3, 5, 7, 9, 10, 10),
    monthly_diurnal_range_c=(13, 13, 14, 16, 18, 19, 19, 19, 18, 17, 15, 14),
    monthly_irradiation_kwh_m2_day=(
        6.8, 6.7, 6.8, 6.9, 6.6, 6.4, 6.7, 7.1, 7.4, 7.6, 7.6, 7.0,
    ),
    monthly_relative_humidity_pct=(55, 57, 54, 45, 35, 30, 28, 28, 33, 35, 40, 48),
    source=_source(
        "annual plane-of-array insolation near 2540 kWh/m2, above the hot-desert "
        "case because a high plateau has less atmosphere and almost no aerosol; "
        "mean temperatures held near 3-10 C despite tropical latitude because of "
        "altitude; diurnal range pushed to 13-19 C, the largest in this set, "
        "because thin dry air cannot retain daytime heat; the seasonal minimum in "
        "irradiation falls in the southern winter around June."
    ),
)

MEDITERRANEAN = ClimateArchetype(
    key="mediterranean",
    name="Mediterranean summer-dry",
    koppen_class="Csa",
    description=(
        "Hot dry summers and mild wet winters. It sits between the desert and "
        "the maritime cases and is the closest archetype to the conditions of "
        "most installed European utility-scale PV. Approximation: coastal and "
        "inland variants of this class differ by several degrees in diurnal "
        "range; a moderate inland value is used."
    ),
    latitude_band="Mid-latitude, roughly 35-42 degrees N",
    representative_latitude_deg=38.0,
    monthly_ambient_temperature_c=(9, 10, 12, 15, 19, 24, 27, 27, 23, 18, 13, 10),
    monthly_diurnal_range_c=(9, 10, 11, 11, 12, 13, 14, 14, 13, 11, 9, 8),
    monthly_irradiation_kwh_m2_day=(
        2.9, 3.7, 4.8, 5.6, 6.3, 6.9, 7.1, 6.7, 5.6, 4.3, 3.1, 2.6,
    ),
    monthly_relative_humidity_pct=(78, 75, 70, 66, 62, 55, 50, 52, 60, 70, 77, 80),
    source=_source(
        "annual plane-of-array insolation near 1810 kWh/m2 with a pronounced "
        "summer maximum, because the summer is not merely sunnier but genuinely "
        "cloud-free; humidity anti-correlated with irradiation, falling to 50% in "
        "July and rising to 80% in December, which is the defining Csa pattern."
    ),
)

HUMID_CONTINENTAL = ClimateArchetype(
    key="humid_continental",
    name="Humid continental",
    koppen_class="Dfa",
    description=(
        "Large seasonal amplitude with hot humid summers and hard-freezing "
        "winters, so a single site spans most of the temperature axis over a "
        "year. Approximation: the many freeze-thaw crossings this climate "
        "produces are a real and well-documented module-failure driver, but "
        "thermal cycling is capability 'stress_cycling' (Tier.PLANNED) and this "
        "schedule cannot represent it -- monthly means hide every crossing."
    ),
    latitude_band="Mid-latitude, roughly 40-45 degrees N",
    representative_latitude_deg=42.0,
    monthly_ambient_temperature_c=(-5, -3, 3, 10, 16, 21, 24, 23, 19, 12, 5, -2),
    monthly_diurnal_range_c=(9, 10, 11, 12, 12, 12, 12, 11, 11, 11, 9, 8),
    monthly_irradiation_kwh_m2_day=(
        2.4, 3.2, 4.2, 4.9, 5.4, 5.7, 5.8, 5.4, 4.6, 3.5, 2.3, 2.0,
    ),
    monthly_relative_humidity_pct=(74, 72, 68, 64, 66, 68, 69, 72, 74, 72, 75, 77),
    source=_source(
        "annual plane-of-array insolation near 1500 kWh/m2, between the maritime "
        "and Mediterranean cases; a 29 C swing between January and July means, "
        "which is the continentality that names this class; humidity held in a "
        "narrow 64-77% band because both summer humidity and winter cloud are high."
    ),
)

ARCHETYPES: tuple[ClimateArchetype, ...] = (
    HOT_DESERT,
    HUMID_SUBTROPICAL,
    TROPICAL_MONSOON,
    MEDITERRANEAN,
    TEMPERATE_MARITIME,
    HUMID_CONTINENTAL,
    CONTINENTAL_SUBARCTIC,
    HIGH_ALTITUDE_ARID,
)

_BY_KEY = {a.key: a for a in ARCHETYPES}
_BY_NAME = {a.name.lower(): a for a in ARCHETYPES}
ARCHETYPE_KEYS: tuple[str, ...] = tuple(a.key for a in ARCHETYPES)


def get_archetype(name: str) -> ClimateArchetype:
    """Look an archetype up by key or display name (case-insensitive)."""
    needle = str(name).strip().lower()
    if needle in _BY_KEY:
        return _BY_KEY[needle]
    if needle in _BY_NAME:
        return _BY_NAME[needle]
    raise KeyError(
        f"Unknown climate archetype {name!r}. Available: {', '.join(ARCHETYPE_KEYS)}"
    )


def climate_capability() -> capabilities.Capability:
    """The registry entry that governs anything built on these schedules."""
    return capabilities.get(CLIMATE_CAPABILITY_KEY)


# ---------------------------------------------------------------------------
# Climate -> stress schedule
# ---------------------------------------------------------------------------
SCHEDULE_COLUMNS = (
    "archetype_key",
    "archetype",
    "year_index",
    "month",
    "month_name",
    "segment",
    "illumination_suns",
    "irradiance_W_m2",
    "ambient_temperature_C",
    "cell_temperature_C",
    "relative_humidity_pct",
    "hours_at_condition",
    "equivalent_full_sun_hours",
    "elapsed_hours_start",
    "elapsed_hours_end",
    "daylight_h_per_day",
    "irradiation_kWh_m2_day",
    "days_in_month",
)


def to_stress_schedule(
    archetype: ClimateArchetype | str,
    years: int = 1,
    *,
    diurnal_bins: int = 1,
    include_dark_hours: bool = True,
    noct_c: float = NOCT_C,
) -> pd.DataFrame:
    """Convert a climate archetype into a sequence of surrogate stress points.

    Returns one row per (year, month, segment). A segment is a block of hours
    spent at one nominal (illumination, cell temperature) condition. Every
    column is documented in ``SCHEDULE_COLUMNS`` order:

    ``illumination_suns``        effective irradiance / 1000 W/m2
    ``cell_temperature_C``       NOCT model output for that segment
    ``hours_at_condition``       hours the schedule assigns to that condition
    ``elapsed_hours_end``        cumulative deployment hours, used as the
                                 aging-time coordinate by ``envelope_report``

    THE CONVERSION, AND WHAT IT ASSUMES
    -----------------------------------
    For each month, ``N`` astronomical daylight hours are computed from the
    archetype's representative latitude on Klein's mean day for that month.
    The stated daily irradiation ``H`` (kWh/m2/day, plane-of-array) is spread
    over those ``N`` hours as a half sine normalised to preserve ``H`` exactly,
    then discretised into ``diurnal_bins`` equal-duration slices. Effective
    suns for a slice is its mean irradiance divided by 1000 W/m2.

    THE SIMPLIFICATION A REVIEWER WILL PROBE, STATED PLAINLY: with the default
    ``diurnal_bins=1`` this collapses the entire diurnal cycle into ONE mean
    condition. A day that actually runs 0 -> 900 -> 0 W/m2 is replaced by a
    flat 520 W/m2 for the same number of hours. The two carry identical energy
    but they are not identical stresses, because neither the surrogate's
    response nor the underlying degradation chemistry is linear in irradiance
    or temperature -- and the flat version never reaches the peak cell
    temperature the real day does. The collapse is dose-preserving, not
    damage-preserving. Raising ``diurnal_bins`` (4-8 is a reasonable probe)
    recovers the shape and the midday temperature peak while preserving the
    same total dose, and is the intended sensitivity check. It does not make
    the result validated: nothing about composing conditions is validated.

    Ambient air temperature is split into a daytime mean and a night mean
    around the monthly 24-hour mean using half the stated diurnal range scaled
    by ``2/pi`` (the mean of a sinusoid over the half cycle containing its
    peak). Ambient is then held flat across all daylight bins, so within
    daylight only irradiance -- not air temperature -- shapes the cell
    temperature profile.

    Parameters
    ----------
    archetype
        A ``ClimateArchetype`` or a key/name accepted by ``get_archetype``.
    years
        Whole deployment years. The same climatological year repeats; there is
        no interannual variability and no degradation trend applied to the
        climate.
    diurnal_bins
        Daylight slices per month. 1 (default) collapses the day to its mean.
    include_dark_hours
        When True (default) each month also gets a night segment at exactly
        0.0 suns and the night mean ambient, so the schedule accounts for all
        8760 h of the year. Those segments are BELOW the surrogate's 0.01 suns
        floor and ``envelope_report`` flags every one of them. Setting this
        False produces a daylight-only schedule that does not span the year and
        silently discards the thermal-only stress of the night; that is a
        worse assumption, not a better one, and it is offered only so a caller
        can isolate the illuminated segments.
    noct_c
        NOCT assumption passed to ``cell_temperature_noct``.

    Notes
    -----
    Pure and deterministic: no sampling, no state, nothing to seed. This
    function does NOT accumulate damage and does NOT call the surrogate --
    see the module docstring and capability ``twin_climate`` (Tier.PREVIEW).
    """
    arch = archetype if isinstance(archetype, ClimateArchetype) else get_archetype(archetype)
    years = int(years)
    diurnal_bins = int(diurnal_bins)
    if years < 1:
        raise ValueError("years must be a positive whole number of years.")
    if diurnal_bins < 1:
        raise ValueError("diurnal_bins must be at least 1.")

    rows: list[dict[str, object]] = []
    elapsed = 0.0

    for year_index in range(1, years + 1):
        for month_index in range(12):
            days = DAYS_IN_MONTH[month_index]
            daylight = daylight_hours(
                arch.representative_latitude_deg, KLEIN_MEAN_DAY_OF_YEAR[month_index]
            )
            irradiation = arch.monthly_irradiation_kwh_m2_day[month_index]
            mean_ambient = arch.monthly_ambient_temperature_c[month_index]
            half_swing = 0.5 * arch.monthly_diurnal_range_c[month_index] * _SINUSOID_HALF_MEAN
            day_ambient = mean_ambient + half_swing
            night_ambient = mean_ambient - half_swing
            humidity = arch.monthly_relative_humidity_pct[month_index]

            bin_irradiance = _half_sine_bin_irradiance(irradiation, daylight, diurnal_bins)
            for bin_index, irradiance in enumerate(bin_irradiance):
                hours = days * daylight / diurnal_bins
                if hours <= 0.0:
                    continue
                suns = irradiance / 1000.0
                cell_t = float(cell_temperature_noct(day_ambient, irradiance, noct_c))
                label = (
                    "daylight"
                    if diurnal_bins == 1
                    else f"daylight {bin_index + 1}/{diurnal_bins}"
                )
                rows.append(
                    {
                        "archetype_key": arch.key,
                        "archetype": arch.name,
                        "year_index": year_index,
                        "month": month_index + 1,
                        "month_name": MONTH_NAMES[month_index],
                        "segment": label,
                        "illumination_suns": suns,
                        "irradiance_W_m2": irradiance,
                        "ambient_temperature_C": day_ambient,
                        "cell_temperature_C": cell_t,
                        "relative_humidity_pct": float(humidity),
                        "hours_at_condition": hours,
                        "equivalent_full_sun_hours": suns * hours,
                        "elapsed_hours_start": elapsed,
                        "elapsed_hours_end": elapsed + hours,
                        "daylight_h_per_day": daylight,
                        "irradiation_kWh_m2_day": float(irradiation),
                        "days_in_month": days,
                    }
                )
                elapsed += hours

            if include_dark_hours:
                dark_hours = days * (24.0 - daylight)
                if dark_hours > 0.0:
                    # Exactly zero suns, not the 0.01 floor. Clipping to the
                    # envelope here would hide the excursion the report exists
                    # to surface.
                    cell_t = float(cell_temperature_noct(night_ambient, 0.0, noct_c))
                    rows.append(
                        {
                            "archetype_key": arch.key,
                            "archetype": arch.name,
                            "year_index": year_index,
                            "month": month_index + 1,
                            "month_name": MONTH_NAMES[month_index],
                            "segment": "dark",
                            "illumination_suns": 0.0,
                            "irradiance_W_m2": 0.0,
                            "ambient_temperature_C": night_ambient,
                            "cell_temperature_C": cell_t,
                            "relative_humidity_pct": float(humidity),
                            "hours_at_condition": dark_hours,
                            "equivalent_full_sun_hours": 0.0,
                            "elapsed_hours_start": elapsed,
                            "elapsed_hours_end": elapsed + dark_hours,
                            "daylight_h_per_day": daylight,
                            "irradiation_kWh_m2_day": float(irradiation),
                            "days_in_month": days,
                        }
                    )
                    elapsed += dark_hours

    schedule = pd.DataFrame(rows, columns=list(SCHEDULE_COLUMNS))
    # attrs survive most pandas operations but not all; the archetype columns
    # above are the load-bearing copy. These are convenience metadata only.
    schedule.attrs["archetype_key"] = arch.key
    schedule.attrs["diurnal_bins"] = diurnal_bins
    schedule.attrs["include_dark_hours"] = include_dark_hours
    schedule.attrs["noct_c"] = float(noct_c)
    schedule.attrs["provenance"] = PROVENANCE_NOTICE
    return schedule


# ---------------------------------------------------------------------------
# Envelope reporting
# ---------------------------------------------------------------------------
_ENVELOPE_DIMENSIONS = (
    # (report label, schedule column, ENVELOPE key)
    ("Illumination", "illumination_suns", "illumination_suns"),
    ("Cell temperature", "cell_temperature_C", "temperature_c"),
    ("Elapsed time", "elapsed_hours_end", "aging_h"),
)


@dataclass(frozen=True)
class EnvelopeReport:
    """Where a climate schedule leaves the surrogate's tested design box.

    Produced by ``envelope_report``. Every hour count is real: they are summed
    from the schedule, not estimated.
    """

    archetype: str
    archetype_key: str
    years: float
    n_segments: int
    total_hours: float
    hours_inside: float
    segments_inside: int
    first_excursion_elapsed_h: float | None
    illumination_range_suns: tuple[float, float]
    cell_temperature_range_c: tuple[float, float]
    elapsed_hours_range: tuple[float, float]
    excursions: pd.DataFrame
    detail: pd.DataFrame
    notes: tuple[str, ...]

    @property
    def hours_outside(self) -> float:
        return self.total_hours - self.hours_inside

    @property
    def fraction_hours_inside(self) -> float:
        return self.hours_inside / self.total_hours if self.total_hours else 0.0

    @property
    def is_fully_validated(self) -> bool:
        """True only if every single segment sits inside the tested box."""
        return self.segments_inside == self.n_segments

    def headline(self) -> str:
        pct = 100.0 * self.fraction_hours_inside
        return (
            f"{self.archetype}: {self.segments_inside}/{self.n_segments} schedule "
            f"segments inside the tested envelope, covering {self.hours_inside:.0f} of "
            f"{self.total_hours:.0f} h ({pct:.1f}% of deployment time)."
        )

    def to_frame(self) -> pd.DataFrame:
        """The excursion summary, for direct display or export."""
        return self.excursions.copy()

    def lines(self) -> tuple[str, ...]:
        """Everything the UI needs to render, as plain text."""
        out = [PROVENANCE_NOTICE, self.headline()]
        if self.excursions.empty:
            out.append("No envelope excursions: the whole schedule is inside the tested box.")
        else:
            for row in self.excursions.itertuples(index=False):
                out.append(
                    f"[{row.dimension} {row.direction} envelope] {row.n_segments} segments, "
                    f"{row.hours:.0f} h ({row.pct_of_hours:.1f}% of time). "
                    f"Worst value {row.worst_value:g} vs limit {row.limit:g}. "
                    f"Example: {row.example_message}"
                )
        out.extend(self.notes)
        return tuple(out)


def envelope_report(schedule: pd.DataFrame) -> EnvelopeReport:
    """State exactly which parts of a climate schedule are extrapolation.

    Each schedule segment is passed to ``capabilities.envelope_excursions()``
    with

        illumination = ``illumination_suns``
        temperature  = ``cell_temperature_C``   (NOT ambient -- the surrogate's
                        temperature axis is the temperature of the device)
        aging time   = ``elapsed_hours_end``    (cumulative deployment hours,
                        the only sensible analogue of the campaign's aging
                        coordinate)

    The plain-language message text comes from the capability registry so the
    UI and this report can never disagree about wording. Aggregation into
    per-dimension counts re-reads ``capabilities.ENVELOPE`` directly, so the
    bounds are also single-sourced.

    A note on the aging coordinate, because it is the weakest link. The
    campaign's ``aging_h`` is time held at ONE fixed condition. Deployment
    elapsed time is time spent across MANY conditions. Equating them is
    already an approximation, and it is the approximation that makes
    ``twin_climate`` Tier.PREVIEW rather than Tier.VALIDATED. Using cumulative
    equivalent full-sun hours instead would flatter the result -- roughly a
    fifth of the number -- and would still be beyond 1000 h within the first
    year for every archetype here, so nothing is gained by the substitution
    and the more conservative reading is used.

    Returns an ``EnvelopeReport``. Pure; does not mutate ``schedule``.
    """
    if schedule.empty:
        raise ValueError("Cannot build an envelope report from an empty schedule.")

    total_hours = float(schedule["hours_at_condition"].sum())
    detail_rows: list[dict[str, object]] = []
    inside_hours = 0.0
    inside_segments = 0
    first_excursion: float | None = None

    for row in schedule.itertuples(index=False):
        messages = envelope_excursions(
            row.illumination_suns, row.cell_temperature_C, row.elapsed_hours_end
        )
        if not messages:
            inside_hours += float(row.hours_at_condition)
            inside_segments += 1
            continue
        if first_excursion is None:
            first_excursion = float(row.elapsed_hours_start)
        for label, column, envelope_key in _ENVELOPE_DIMENSIONS:
            value = float(getattr(row, column))
            low, high = ENVELOPE[envelope_key]
            if value < low:
                direction, limit = "below", low
            elif value > high:
                direction, limit = "above", high
            else:
                continue
            # Pair our classification with the registry's own wording for the
            # same dimension, so the exported text is the registry's, not ours.
            example = next(
                (m for m in messages if m.lower().startswith(label.split()[0].lower())),
                messages[0],
            )
            detail_rows.append(
                {
                    "year_index": row.year_index,
                    "month_name": row.month_name,
                    "segment": row.segment,
                    "dimension": label,
                    "direction": direction,
                    "value": value,
                    "limit": limit,
                    "hours": float(row.hours_at_condition),
                    "elapsed_hours_end": float(row.elapsed_hours_end),
                    "message": example,
                }
            )

    detail = pd.DataFrame(
        detail_rows,
        columns=[
            "year_index", "month_name", "segment", "dimension", "direction",
            "value", "limit", "hours", "elapsed_hours_end", "message",
        ],
    )

    if detail.empty:
        excursions = pd.DataFrame(
            columns=[
                "dimension", "direction", "n_segments", "hours", "pct_of_hours",
                "worst_value", "limit", "example_message",
            ]
        )
    else:
        grouped = []
        for (dimension, direction), block in detail.groupby(
            ["dimension", "direction"], sort=False
        ):
            worst = block["value"].min() if direction == "below" else block["value"].max()
            hours = float(block["hours"].sum())
            grouped.append(
                {
                    "dimension": dimension,
                    "direction": direction,
                    "n_segments": int(len(block)),
                    "hours": hours,
                    "pct_of_hours": 100.0 * hours / total_hours if total_hours else 0.0,
                    "worst_value": float(worst),
                    "limit": float(block["limit"].iloc[0]),
                    "example_message": str(block["message"].iloc[0]),
                }
            )
        excursions = (
            pd.DataFrame(grouped)
            .sort_values("hours", ascending=False)
            .reset_index(drop=True)
        )

    cap = climate_capability()
    humidity_cap = capabilities.get("stress_humidity")
    extrapolation_cap = capabilities.get("stress_extrapolation")
    diurnal_bins = int(schedule.attrs.get("diurnal_bins", 1))

    notes = [
        f"Damage accumulation across this schedule is NOT performed here. Capability "
        f"'{cap.key}' ({cap.label}) is {TIER_LABEL[cap.tier]}: {cap.backing}",
        f"Any segment outside the box falls under '{extrapolation_cap.key}' "
        f"({TIER_LABEL[extrapolation_cap.tier]}): {extrapolation_cap.backing}",
        f"Relative humidity is carried in this schedule but the surrogate cannot use "
        f"it. Capability '{humidity_cap.key}' is {TIER_LABEL[humidity_cap.tier]}: "
        f"{humidity_cap.backing}",
    ]
    if diurnal_bins == 1:
        notes.append(
            "The diurnal cycle is collapsed to one mean condition per month "
            "(diurnal_bins=1). Dose is preserved exactly; peak cell temperature is "
            "not. Re-run with diurnal_bins=6 to see how much that matters."
        )
    else:
        notes.append(
            f"The daylight period is resolved into {diurnal_bins} half-sine bins per "
            "month. Total dose is identical to the collapsed case by construction."
        )
    notes.append(
        "Climate inputs are hand-built archetypes, so these excursion figures "
        "describe the archetype, not any real site."
    )

    years = total_hours / HOURS_PER_YEAR

    return EnvelopeReport(
        archetype=str(schedule["archetype"].iloc[0]),
        archetype_key=str(schedule["archetype_key"].iloc[0]),
        years=years,
        n_segments=int(len(schedule)),
        total_hours=total_hours,
        hours_inside=inside_hours,
        segments_inside=inside_segments,
        first_excursion_elapsed_h=first_excursion,
        illumination_range_suns=(
            float(schedule["illumination_suns"].min()),
            float(schedule["illumination_suns"].max()),
        ),
        cell_temperature_range_c=(
            float(schedule["cell_temperature_C"].min()),
            float(schedule["cell_temperature_C"].max()),
        ),
        elapsed_hours_range=(
            float(schedule["elapsed_hours_start"].min()),
            float(schedule["elapsed_hours_end"].max()),
        ),
        excursions=excursions,
        detail=detail,
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# Tables for the UI and the exported run bundle
# ---------------------------------------------------------------------------
def archetype_table() -> pd.DataFrame:
    """All archetypes, one row each, with the disclaimer on every row.

    The ``provenance`` column is repeated deliberately. A table gets copied
    into a slide or a supplement without its caption; the row must carry its
    own warning.
    """
    return pd.DataFrame(
        [
            {
                "key": a.key,
                "archetype": a.name,
                "koppen_class": a.koppen_class,
                "latitude_band": a.latitude_band,
                "representative_latitude_deg": a.representative_latitude_deg,
                "annual_insolation_kWh_m2": round(a.annual_insolation_kwh_m2, 1),
                "annual_mean_temperature_C": round(a.annual_mean_temperature_c, 1),
                "annual_mean_RH_pct": round(a.annual_mean_relative_humidity_pct, 1),
                "min_month_temperature_C": min(a.monthly_ambient_temperature_c),
                "max_month_temperature_C": max(a.monthly_ambient_temperature_c),
                "max_diurnal_range_C": max(a.monthly_diurnal_range_c),
                "provenance": "Representative archetype - illustrative, not measured data",
                "description": a.description,
                "source": a.source,
            }
            for a in ARCHETYPES
        ]
    )


def monthly_table(archetype: ClimateArchetype | str) -> pd.DataFrame:
    """The twelve monthly rows of one archetype, with derived daylight hours."""
    arch = archetype if isinstance(archetype, ClimateArchetype) else get_archetype(archetype)
    daylight = arch.daylight_hours_by_month()
    return pd.DataFrame(
        {
            "month": range(1, 13),
            "month_name": list(MONTH_NAMES),
            "ambient_mean_C": list(arch.monthly_ambient_temperature_c),
            "diurnal_range_C": list(arch.monthly_diurnal_range_c),
            "irradiation_kWh_m2_day": list(arch.monthly_irradiation_kwh_m2_day),
            "relative_humidity_pct": list(arch.monthly_relative_humidity_pct),
            "daylight_h": [round(h, 2) for h in daylight],
            "daylight_mean_suns": [
                round(h_irr / h_day, 4) if h_day > 0 else 0.0
                for h_irr, h_day in zip(arch.monthly_irradiation_kwh_m2_day, daylight)
            ],
            "provenance": "Representative archetype - illustrative, not measured data",
        }
    )


def consistency_report() -> pd.DataFrame:
    """Physical plausibility check on every archetype.

    The binding constraint: the daylight-mean effective irradiance can never
    exceed one sun, because a stated daily irradiation ``H`` spread over ``N``
    daylight hours implies a mean of ``H/N`` suns and no plane on Earth
    averages above 1000 W/m2 across the whole day. If ``max_daylight_mean_suns``
    ever exceeds 1.0, the archetype's irradiation and latitude are mutually
    impossible and the numbers must be fixed, not explained.

    Also reports the peak single-bin irradiance under a 12-bin half-sine
    discretisation, whose ceiling is the roughly 1000-1100 W/m2 a clear sky
    delivers at normal incidence (a little more at high altitude).
    """
    rows = []
    for arch in ARCHETYPES:
        daylight = arch.daylight_hours_by_month()
        means = [
            (h / d) if d > 0 else 0.0
            for h, d in zip(arch.monthly_irradiation_kwh_m2_day, daylight)
        ]
        peaks = [
            max(_half_sine_bin_irradiance(h, d, 12), default=0.0)
            for h, d in zip(arch.monthly_irradiation_kwh_m2_day, daylight)
        ]
        rows.append(
            {
                "key": arch.key,
                "koppen_class": arch.koppen_class,
                "annual_insolation_kWh_m2": round(arch.annual_insolation_kwh_m2, 1),
                "min_daylight_h": round(min(daylight), 2),
                "max_daylight_h": round(max(daylight), 2),
                "max_daylight_mean_suns": round(max(means), 3),
                "peak_bin_irradiance_W_m2": round(max(peaks), 1),
                "physically_possible": bool(max(means) <= 1.0),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Demonstration. Run with:
#   .venv/Scripts/python.exe -m psc_twin.climate
# ---------------------------------------------------------------------------
def _demo() -> None:  # pragma: no cover - developer-facing output
    pd.set_option("display.width", 150)
    pd.set_option("display.max_columns", 30)

    print("=" * 100)
    print(PROVENANCE_NOTICE)
    print("=" * 100)

    cap = climate_capability()
    print(
        f"\nGoverning capability: {cap.key} / {cap.label} -> {TIER_LABEL[cap.tier]}"
        f"\n  backing: {cap.backing}"
        f"\n  unlocked by: {cap.unlocks}"
    )
    print(f"\nNOCT assumption: {NOCT_C} C   (T_cell = T_amb + ({NOCT_C}-20)/800 * G)")

    print("\n\n--- ARCHETYPE REGISTRY ---")
    table = archetype_table()
    print(
        table[
            [
                "key", "koppen_class", "representative_latitude_deg",
                "annual_insolation_kWh_m2", "annual_mean_temperature_C",
                "annual_mean_RH_pct", "max_diurnal_range_C",
            ]
        ].to_string(index=False)
    )

    print("\n--- PHYSICAL CONSISTENCY CHECK ---")
    print(consistency_report().to_string(index=False))

    for key in ("hot_desert", "continental_subarctic"):
        arch = get_archetype(key)
        print("\n\n" + "=" * 100)
        print(f"ARCHETYPE: {arch.name}  ({arch.koppen_class})   {arch.latitude_band}")
        print("=" * 100)
        print(f"description: {arch.description}")
        print(f"source     : {arch.source}")

        print("\n-- monthly climate profile --")
        print(monthly_table(arch).drop(columns=["provenance"]).to_string(index=False))

        schedule = to_stress_schedule(arch, years=1)
        print("\n-- stress schedule, 1 year, diurnal_bins=1 (all 24 rows) --")
        print(
            schedule[
                [
                    "month_name", "segment", "illumination_suns",
                    "ambient_temperature_C", "cell_temperature_C",
                    "hours_at_condition", "elapsed_hours_end",
                ]
            ]
            .round(3)
            .to_string(index=False)
        )
        print(
            f"   total hours = {schedule['hours_at_condition'].sum():.1f} "
            f"(one year = {HOURS_PER_YEAR:.0f} h)"
            f"   equivalent full-sun hours = "
            f"{schedule['equivalent_full_sun_hours'].sum():.1f} "
            f"(archetype annual insolation = {arch.annual_insolation_kwh_m2:.1f} kWh/m2)"
        )

        report = envelope_report(schedule)
        print("\n-- ENVELOPE REPORT (1 year) --")
        for line in report.lines():
            print(f"   {line}")
        print(
            f"   ranges seen: illumination {report.illumination_range_suns[0]:.3f} to "
            f"{report.illumination_range_suns[1]:.3f} suns | cell temperature "
            f"{report.cell_temperature_range_c[0]:.1f} to "
            f"{report.cell_temperature_range_c[1]:.1f} C | elapsed "
            f"{report.elapsed_hours_range[0]:.0f} to {report.elapsed_hours_range[1]:.0f} h"
        )
        print(
            f"   first excursion at elapsed hour: {report.first_excursion_elapsed_h}"
        )

        fine = to_stress_schedule(arch, years=1, diurnal_bins=6)
        print(
            f"\n-- diurnal_bins=6 sensitivity: peak cell temperature rises from "
            f"{schedule['cell_temperature_C'].max():.1f} C to "
            f"{fine['cell_temperature_C'].max():.1f} C; equivalent full-sun hours "
            f"{fine['equivalent_full_sun_hours'].sum():.1f} (dose unchanged)"
        )

    print("\n\n" + "=" * 100)
    print("25-YEAR HORIZON, ALL ARCHETYPES (diurnal_bins=1)")
    print("=" * 100)
    summary = []
    for arch in ARCHETYPES:
        sched = to_stress_schedule(arch, years=25)
        rep = envelope_report(sched)
        summary.append(
            {
                "archetype": arch.key,
                "segments": rep.n_segments,
                "total_h": round(rep.total_hours),
                "segments_inside": rep.segments_inside,
                "pct_hours_inside": round(100.0 * rep.fraction_hours_inside, 2),
                "min_cell_C": round(rep.cell_temperature_range_c[0], 1),
                "max_cell_C": round(rep.cell_temperature_range_c[1], 1),
                "max_suns": round(rep.illumination_range_suns[1], 3),
                "first_excursion_h": rep.first_excursion_elapsed_h,
            }
        )
    print(pd.DataFrame(summary).to_string(index=False))

    print("\n--- SIMPLIFICATIONS DECLARED BY THIS MODULE ---")
    for i, item in enumerate(SIMPLIFICATIONS, 1):
        print(f"  {i:2d}. {item}")


if __name__ == "__main__":  # pragma: no cover
    _demo()
