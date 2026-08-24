"""Turn a retention trajectory into a lifetime number, and say how it was obtained.

A retention trajectory is easy to plot and hard to summarise honestly. The field
quotes a single number -- T80, the time until only 80% of the original efficiency
remains -- but that number can be arrived at in three very different ways, and a
reader who cannot tell them apart is being misled:

    interpolated   The trajectory actually crosses the threshold inside the
                   simulated window. The number is read off the curve. Trustworthy.
    extrapolated   The window ends while the cell is still above threshold, so an
                   exponential decay is fitted to the tail and solved forward.
                   A projection, not a measurement.
    not reached    The cell never approaches the threshold, the tail carries no
                   usable decay trend, or the projection would land so far beyond
                   the data that it is a guess. No number -- ``value_h`` is None.

Every function here returns the method alongside the value, and the two are never
blended. ``LifetimeEstimate.method`` is the field a plot caption or a table must
read before it prints a lifetime.

WHY EXTRAPOLATION IS THE INTERESTING CASE
-----------------------------------------
In the 36-run COMSOL campaign only 8 of 36 runs reach 80% retention inside the
1000 h simulated (14 of 36 reach 90%). For the other 28 runs any T80 at all is a
projection. Refusing to extrapolate would leave most of the design space blank;
extrapolating without a warning label would be dishonest. Hence the third field.

HOW GOOD IS THE EXTRAPOLATION? (measured, not assumed)
------------------------------------------------------
Fitting ``retention = A * exp(-k t)`` to the last 5 points of each of the 36 runs
gives a mean R^2 of 0.998 (min 0.991) in retention space. That looks excellent and
it is misleading. The R^2 describes how well the model fits the tail it was shown,
not how well it predicts forward.

The honest test is a truncation study, which this module's authors ran on the 8
runs whose true T80 is known by interpolation: truncate the trajectory to a short
horizon, extrapolate, and compare against the known answer (n = 21 truncations):

    reach = estimate / horizon      median error      worst error
    up to 1.25x                        -7.0%            -10.7%
    1.25x to 1.75x                    -15.6%            -22.1%
    1.75x to 2.5x                     -22.7%            -27.4%
    beyond 2.5x                       -27.9%            -27.9%

Every single one of the 21 errors is negative. The bias is systematic, not noise:
the true decay decelerates (it is sub-exponential), so an exponential fitted to an
early tail keeps decaying too fast and reports a lifetime that is too short. R^2
above 0.98 accompanies errors of -28%, which is precisely why ``fit_quality`` is
documented as a fit diagnostic and never as an accuracy guarantee.

The practical consequence, stated plainly because it ships with a paper:
**extrapolated lifetimes from this module are conservative (too short), by roughly
5-10% at 1.25x reach and 20-30% at 2.5x reach.** ``LifetimeEstimate.reach_factor``
exposes the reach so a caller can apply this table, and ``note()`` says so in
words. No correction factor is applied -- correcting a bias measured on 8 runs
would be over-fitting a small sample, and a silently corrected number is worse
than a labelled biased one.

THERE IS A HARD LIMIT ON REACH
------------------------------
The truncation study above covers reach factors up to about 2.9x. Beyond that
there is no evidence at all, and the arithmetic will happily keep going: the
gentlest run in the campaign (0.01 suns, 26.85 C) still holds 99.97% retention at
1000 h, and solving its fitted tail for 80% returns 676,186 h -- 77 years, 676x
beyond anything simulated, with R^2 = 1.0000. That number is not a projection, it
is a fabrication with a decimal point.

So ``lifetime_estimate`` refuses it. Any solution beyond ``max_reach_factor``
(default 3.0, just past the validated range) is discarded and the estimate comes
back as ``not reached`` with the rejected reach recorded in
``declined_reach_factor``. On the 36-run campaign this is the difference between
36 confident-looking T80 numbers and the honest split of 8 interpolated, 7
extrapolated, 21 declined. Raise the cap deliberately if a downstream analysis
genuinely wants unvalidated projections; do not raise it to fill a table.

MODELLING CHOICE: THE AMPLITUDE IS FITTED, NOT PINNED TO 100
------------------------------------------------------------
The nominal model is ``retention = 100 * exp(-k t)``. Pinning the amplitude to
exactly 100 forces the fitted curve through the undegraded starting point, which
the tail has long left behind; on these runs that costs a great deal of fit
quality (mean R^2 0.897, min 0.238 on a 5-point tail, versus 0.998 / 0.991 with a
free amplitude). So ``fit_exponential_tail`` fits both amplitude and rate by
default and reports the amplitude it used. ``pin_amplitude=True`` recovers the
literal ``100 * exp(-k t)`` form for anyone who wants it.

The fit is a least-squares line through log-retention via ``numpy.polyfit`` --
that is a deliberately simple, dependency-free choice (there is no scipy in this
project). It weights the log residuals, so early tail points with high retention
carry slightly less influence than a retention-space fit would give them.

This module is pure numpy/pandas. It imports no model, no sklearn, no Streamlit,
and holds no state, so it can be applied equally to a COMSOL ground-truth
trajectory and to a surrogate-predicted one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

# The three ways a lifetime number can come to exist. Kept as a Literal rather
# than an Enum so the value survives a round trip through CSV/JSON unchanged.
Method = Literal["interpolated", "extrapolated", "not reached"]

INTERPOLATED: Method = "interpolated"
EXTRAPOLATED: Method = "extrapolated"
NOT_REACHED: Method = "not reached"

# 95% of a normal lies within this many standard deviations. Used only to turn a
# predicted mean/std retention band into a pair of bounding trajectories.
Z_95 = 1.96

# Default number of trailing points used for the tail fit. Five is the largest
# tail that stays inside the decelerating late-time regime for these runs; using
# all 10 points drags the early, faster-decaying part into the fit and lowers
# retention-space R^2 from 0.998 to 0.988.
DEFAULT_TAIL_POINTS = 5

# Largest allowed ratio of extrapolated lifetime to simulated horizon. The
# truncation study behind this module validated reach factors up to ~2.9x; 3.0 is
# the edge of the evidence, not a round number chosen for looks. Past it the
# exponential fit still returns a value and that value is not worth printing.
MAX_REACH_FACTOR = 3.0


@dataclass(frozen=True)
class ExponentialTailFit:
    """A least-squares exponential fitted to the tail of a retention trajectory."""

    k_per_h: float          # decay rate, 1/h. Positive means retention is falling.
    amplitude_pct: float    # fitted retention at t = 0, in percent
    r2: float               # coefficient of determination in RETENTION space
    r2_log: float           # ... and in log space, which is what polyfit minimised
    n_points: int           # tail points used
    t_start_h: float        # first time in the fitted tail
    pinned_amplitude: bool  # True if amplitude was forced to 100

    def retention_at(self, time_h: float | np.ndarray) -> np.ndarray:
        """Evaluate the fitted decay at one or more times."""
        return self.amplitude_pct * np.exp(-self.k_per_h * np.asarray(time_h, dtype=float))

    def solve_time(self, threshold_pct: float) -> float | None:
        """Time at which the fitted decay reaches ``threshold_pct``.

        Returns None when the fit cannot deliver the threshold at all: a
        non-decaying tail (k <= 0), or an amplitude already at or below the
        threshold, which would put the crossing at or before t = 0.
        """
        if not np.isfinite(self.k_per_h) or self.k_per_h <= 0.0:
            return None
        if not np.isfinite(self.amplitude_pct) or self.amplitude_pct <= threshold_pct:
            return None
        return float(np.log(self.amplitude_pct / threshold_pct) / self.k_per_h)


@dataclass(frozen=True)
class LifetimeEstimate:
    """One lifetime number and the full provenance a reader needs to judge it."""

    value_h: float | None
    method: Method
    threshold_pct: float
    confidence_low_h: float | None = None
    confidence_high_h: float | None = None
    fit_quality: float | None = None          # tail-fit R^2; None when interpolated
    horizon_h: float | None = None            # last simulated time
    final_retention_pct: float | None = None  # retention at the horizon
    decay_k_per_h: float | None = None        # fitted rate; None when interpolated
    fit_amplitude_pct: float | None = None
    # Bound provenance is tracked separately: a band's lower edge can cross
    # inside the window while its upper edge has to be extrapolated, and
    # collapsing that difference would hide exactly what the reader must see.
    confidence_low_method: Method | None = None
    confidence_high_method: Method | None = None
    # Set when a projection was computed and then thrown away for reaching too far
    # past the data. Keeping the number visible makes the refusal auditable, but it
    # lives in its own field so nothing can mistake it for a lifetime.
    declined_reach_factor: float | None = None

    @property
    def label(self) -> str:
        """Conventional name of this metric, e.g. 'T80'."""
        return f"T{self.threshold_pct:g}"

    @property
    def is_projection(self) -> bool:
        return self.method == EXTRAPOLATED

    @property
    def reach_factor(self) -> float | None:
        """How far past the simulated horizon the estimate sits.

        1.0 means the estimate lands exactly at the end of the data; 2.5 means it
        is two and a half times further out than anything that was simulated.
        Read this against the bias table in the module docstring.
        """
        if self.value_h is None or not self.horizon_h:
            return None
        return float(self.value_h / self.horizon_h)

    def note(self) -> str:
        """A plain-language sentence a non-specialist can read without context."""
        pct = f"{self.threshold_pct:g}%"
        horizon = f"{self.horizon_h:g} h" if self.horizon_h is not None else "the simulated window"

        if self.method == NOT_REACHED:
            left = (
                f" It still holds {self.final_retention_pct:.1f}% at the end."
                if self.final_retention_pct is not None
                else ""
            )
            return (
                f"Efficiency stays above {pct} for the whole {horizon} simulated, so "
                f"{self.label} is beyond the tested window.{left} No lifetime number "
                "is reported, because any value would be invented rather than measured."
            )

        if self.method == INTERPOLATED:
            base = (
                f"Efficiency drops to {pct} of its starting value after about "
                f"{self.value_h:,.0f} h. That happens inside the {horizon} simulated, "
                f"so {self.label} is read straight off the curve rather than projected."
            )
            return base + self._band_clause()

        # Extrapolated: state the projection, the evidence, and the known bias.
        left = (
            f", still at {self.final_retention_pct:.1f}%"
            if self.final_retention_pct is not None
            else ""
        )
        reach = self.reach_factor
        strength = (
            "a modest step past the data"
            if reach is not None and reach <= 1.25
            else "well past the data"
            if reach is not None and reach <= 2.5
            else "far past the data"
        )
        base = (
            f"The simulation ends at {horizon}{left}, before efficiency reaches {pct}. "
            f"Continuing the fitted decay trend puts {self.label} at roughly "
            f"{self.value_h:,.0f} h -- {strength}"
        )
        if reach is not None:
            base += f" ({reach:.1f}x the simulated window)"
        base += (
            ". This is a projection, not a simulated result. On runs where the true "
            "answer is known, this method came out too short by about 5-10% at this "
            "kind of short reach and by 20-30% when projecting far out, so treat it "
            "as a conservative floor."
        )
        return base + self._band_clause()

    def _band_clause(self) -> str:
        lo, hi = self.confidence_low_h, self.confidence_high_h
        if lo is None and hi is None:
            return ""
        if hi is None:
            return (
                f" Allowing for the model's uncertainty, it could be as early as "
                f"{lo:,.0f} h; the optimistic edge of the band never reaches {self.threshold_pct:g}%, "
                "so there is no upper bound."
            )
        if lo is None:
            return f" Allowing for the model's uncertainty, it could be as late as {hi:,.0f} h."
        return (
            f" Allowing for the model's uncertainty, the plausible range is roughly "
            f"{lo:,.0f} to {hi:,.0f} h."
        )


@dataclass(frozen=True)
class RemainingUsefulLife:
    """How much life is left at a given age, and the end-of-life it was measured against."""

    rul_h: float | None
    eol_h: float | None          # the end-of-life time the RUL was taken from
    current_age_h: float
    threshold_pct: float
    method: Method               # provenance of eol_h, carried through unchanged
    already_past_eol: bool = False
    estimate: LifetimeEstimate | None = None

    def note(self) -> str:
        pct = f"{self.threshold_pct:g}%"
        if self.rul_h is None:
            return (
                f"At {self.current_age_h:,.0f} h the cell has not reached {pct} and the "
                "simulated window is too short to say when it will, so remaining useful "
                "life cannot be stated."
            )
        if self.already_past_eol:
            return (
                f"At {self.current_age_h:,.0f} h the cell is already past its {pct} "
                f"end-of-life point ({self.eol_h:,.0f} h), so no useful life remains by "
                "this criterion."
            )
        projected = " (based on a projected end-of-life, not a simulated one)" if self.method == EXTRAPOLATED else ""
        return (
            f"After {self.current_age_h:,.0f} h of ageing, about {self.rul_h:,.0f} h remain "
            f"before efficiency falls to {pct} of its original value, which happens at "
            f"{self.eol_h:,.0f} h{projected}."
        )


# ---------------------------------------------------------------------------
# Trajectory hygiene
# ---------------------------------------------------------------------------

def _as_trajectory(
    time_h: np.ndarray | list[float] | pd.Series,
    retention_pct: np.ndarray | list[float] | pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate, clean and time-sort a trajectory.

    Non-finite samples are dropped rather than tolerated: a NaN in the middle of a
    trajectory would silently move a crossing, and a fabricated crossing is worse
    than a missing one. Sorting is defensive -- callers assembling a prediction
    from a dict of times cannot be assumed to have kept them ordered.
    """
    t = np.asarray(time_h, dtype=float).ravel()
    r = np.asarray(retention_pct, dtype=float).ravel()
    if t.shape != r.shape:
        raise ValueError(f"time_h and retention_pct must be the same length, got {t.shape} and {r.shape}")
    if t.size < 2:
        raise ValueError("a trajectory needs at least 2 points")

    keep = np.isfinite(t) & np.isfinite(r)
    if not keep.any():
        raise ValueError("trajectory contains no finite samples")
    t, r = t[keep], r[keep]

    order = np.argsort(t, kind="stable")
    return t[order], r[order]


# ---------------------------------------------------------------------------
# 1-2. Threshold crossing by interpolation
# ---------------------------------------------------------------------------

def threshold_time(
    time_h: np.ndarray | list[float] | pd.Series,
    retention_pct: np.ndarray | list[float] | pd.Series,
    threshold_pct: float,
) -> float | None:
    """Time of the FIRST downward crossing of ``threshold_pct``, by linear interpolation.

    Returns None -- never a fabricated number -- when the trajectory does not cross
    the threshold anywhere inside the supplied horizon.

    Only downward crossings count: a segment qualifies when it starts strictly
    above the threshold and ends at or below it. Trajectories from a noisy
    surrogate can wander back up and re-cross, so scanning stops at the first
    qualifying segment. That is the conservative reading -- a cell that has once
    fallen to 80% has reached T80, whatever it does afterwards.

    A trajectory that is already at or below the threshold at its first sample
    crossed before the record begins; the first sample time is returned, since
    that is the earliest time supported by the data.
    """
    t, r = _as_trajectory(time_h, retention_pct)

    if r[0] <= threshold_pct:
        return float(t[0])

    below = r <= threshold_pct
    if not below.any():
        return None

    # First index at or below threshold; by the guard above it cannot be 0, so
    # the preceding sample exists and is strictly above the threshold.
    i = int(np.argmax(below))
    r_hi, r_lo = r[i - 1], r[i]
    span = r_hi - r_lo  # strictly positive: r_hi > threshold >= r_lo
    frac = (r_hi - threshold_pct) / span
    return float(t[i - 1] + frac * (t[i] - t[i - 1]))


def t80_time(time_h, retention_pct) -> float | None:
    """Interpolated time to 80% retention, or None if never reached in-window."""
    return threshold_time(time_h, retention_pct, 80.0)


def t90_time(time_h, retention_pct) -> float | None:
    """Interpolated time to 90% retention, or None if never reached in-window."""
    return threshold_time(time_h, retention_pct, 90.0)


# ---------------------------------------------------------------------------
# 3. Exponential tail fit for the too-short-horizon case
# ---------------------------------------------------------------------------

def fit_exponential_tail(
    time_h: np.ndarray | list[float] | pd.Series,
    retention_pct: np.ndarray | list[float] | pd.Series,
    n_tail: int = DEFAULT_TAIL_POINTS,
    pin_amplitude: bool = False,
) -> ExponentialTailFit | None:
    """Fit ``retention = A * exp(-k t)`` to the last ``n_tail`` points.

    Taking logs turns this into a straight line, ``log(retention) = log(A) - k t``,
    which ``numpy.polyfit`` solves directly -- no scipy, no iterative optimiser, no
    starting guess to get wrong. The cost is that the fit minimises log residuals,
    not retention residuals; both R^2 values are reported so the difference is
    visible rather than assumed away.

    With ``pin_amplitude=True`` the amplitude is forced to 100, giving the literal
    ``100 * exp(-k t)`` form, and only the rate is fitted. See the module docstring
    for why that is not the default.

    Returns None when the tail cannot support a fit: fewer than two usable points,
    no spread in time, or non-positive retention values (log undefined).
    """
    t, r = _as_trajectory(time_h, retention_pct)

    n = max(2, int(n_tail))
    t_tail, r_tail = t[-n:], r[-n:]

    # Retention must be strictly positive to take a log. A fully dead cell at 0%
    # is physically meaningful but unusable here, so it is dropped rather than
    # nudged to an epsilon that would invent a decay rate.
    usable = r_tail > 0.0
    t_tail, r_tail = t_tail[usable], r_tail[usable]
    if t_tail.size < 2 or np.ptp(t_tail) <= 0.0:
        return None

    y = np.log(r_tail)
    if pin_amplitude:
        # Least squares for y = log(100) - k t with no free intercept.
        offset = y - np.log(100.0)
        denom = float(np.dot(t_tail, t_tail))
        if denom <= 0.0:
            return None
        k = -float(np.dot(t_tail, offset) / denom)
        amplitude = 100.0
    else:
        slope, intercept = np.polyfit(t_tail, y, 1)
        k = -float(slope)
        amplitude = float(np.exp(intercept))

    if not (np.isfinite(k) and np.isfinite(amplitude)):
        return None

    pred = amplitude * np.exp(-k * t_tail)
    r2 = _r_squared(r_tail, pred)
    r2_log = _r_squared(y, np.log(amplitude) - k * t_tail)

    return ExponentialTailFit(
        k_per_h=k,
        amplitude_pct=amplitude,
        r2=r2,
        r2_log=r2_log,
        n_points=int(t_tail.size),
        t_start_h=float(t_tail[0]),
        pinned_amplitude=bool(pin_amplitude),
    )


def _r_squared(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Coefficient of determination, with the degenerate flat-target case pinned to 0.

    A perfectly flat target has zero variance, so R^2 is undefined. Reporting 0
    there is the honest reading: the fit explains no variation because there is
    none to explain. Reporting 1.0 would flatter a meaningless fit.
    """
    ss_res = float(np.sum((observed - predicted) ** 2))
    ss_tot = float(np.sum((observed - np.mean(observed)) ** 2))
    if ss_tot <= 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# The main entry point: interpolate if you can, extrapolate if you must, and say which
# ---------------------------------------------------------------------------

def lifetime_estimate(
    time_h: np.ndarray | list[float] | pd.Series,
    retention_pct: np.ndarray | list[float] | pd.Series,
    threshold_pct: float = 80.0,
    allow_extrapolation: bool = True,
    n_tail: int = DEFAULT_TAIL_POINTS,
    pin_amplitude: bool = False,
    max_reach_factor: float = MAX_REACH_FACTOR,
) -> LifetimeEstimate:
    """Time to ``threshold_pct`` retention, labelled with how it was obtained.

    The order is strict and never blended:

    1. If the trajectory crosses the threshold in-window, interpolate and stop.
       ``fit_quality`` stays None because no fit was involved.
    2. Otherwise, if ``allow_extrapolation``, fit the tail and solve forward. The
       result is flagged ``extrapolated`` and carries the fitted k and R^2 --
       but only if it lands within ``max_reach_factor`` times the horizon.
    3. Otherwise -- tail unusable, or projection too far past the data -- return
       ``not reached`` with ``value_h = None`` and, where a projection was
       computed and rejected, its reach in ``declined_reach_factor``.

    Set ``allow_extrapolation=False`` wherever the product must show only what was
    actually simulated.
    """
    t, r = _as_trajectory(time_h, retention_pct)
    horizon = float(t[-1])
    final_retention = float(r[-1])

    crossed = threshold_time(t, r, threshold_pct)
    if crossed is not None:
        return LifetimeEstimate(
            value_h=crossed,
            method=INTERPOLATED,
            threshold_pct=float(threshold_pct),
            horizon_h=horizon,
            final_retention_pct=final_retention,
        )

    declined: float | None = None
    if allow_extrapolation:
        fit = fit_exponential_tail(t, r, n_tail=n_tail, pin_amplitude=pin_amplitude)
        if fit is not None:
            solved = fit.solve_time(threshold_pct)
            # A solved time inside the horizon would contradict step 1 (the data
            # did not cross), so it is the fit disagreeing with the data, not a
            # lifetime. Reject it rather than report a crossing that never happened.
            if solved is not None and solved > horizon:
                reach = solved / horizon if horizon > 0 else np.inf
                if reach <= max_reach_factor:
                    return LifetimeEstimate(
                        value_h=solved,
                        method=EXTRAPOLATED,
                        threshold_pct=float(threshold_pct),
                        fit_quality=fit.r2,
                        horizon_h=horizon,
                        final_retention_pct=final_retention,
                        decay_k_per_h=fit.k_per_h,
                        fit_amplitude_pct=fit.amplitude_pct,
                    )
                # Too far past the data to be evidence. Remember how far, then decline.
                declined = float(reach)

    return LifetimeEstimate(
        value_h=None,
        method=NOT_REACHED,
        threshold_pct=float(threshold_pct),
        horizon_h=horizon,
        final_retention_pct=final_retention,
        declined_reach_factor=declined,
    )


def t80(time_h, retention_pct, **kwargs) -> LifetimeEstimate:
    """T80: time until 80% of the original efficiency remains."""
    return lifetime_estimate(time_h, retention_pct, threshold_pct=80.0, **kwargs)


def t90(time_h, retention_pct, **kwargs) -> LifetimeEstimate:
    """T90: time until 90% of the original efficiency remains."""
    return lifetime_estimate(time_h, retention_pct, threshold_pct=90.0, **kwargs)


# ---------------------------------------------------------------------------
# 4. Remaining useful life
# ---------------------------------------------------------------------------

def remaining_useful_life(
    current_age_h: float,
    time_h: np.ndarray | list[float] | pd.Series,
    retention_pct: np.ndarray | list[float] | pd.Series,
    threshold_pct: float = 80.0,
    allow_extrapolation: bool = True,
    n_tail: int = DEFAULT_TAIL_POINTS,
) -> RemainingUsefulLife:
    """Hours left before end-of-life, together with the end-of-life time used.

    RUL is a subtraction, so it is only ever as sound as the end-of-life it is
    subtracted from. The end-of-life time and its provenance are both returned, and
    ``method`` propagates unchanged from the underlying estimate: an RUL derived
    from an extrapolated T80 is itself a projection and says so.

    A cell already past end-of-life gets ``rul_h = 0.0`` with ``already_past_eol``
    set, rather than a negative number that a table might render as a lifetime.
    """
    age = float(current_age_h)
    estimate = lifetime_estimate(
        time_h,
        retention_pct,
        threshold_pct=threshold_pct,
        allow_extrapolation=allow_extrapolation,
        n_tail=n_tail,
    )

    if estimate.value_h is None:
        return RemainingUsefulLife(
            rul_h=None,
            eol_h=None,
            current_age_h=age,
            threshold_pct=float(threshold_pct),
            method=estimate.method,
            estimate=estimate,
        )

    remaining = estimate.value_h - age
    past = remaining <= 0.0
    return RemainingUsefulLife(
        rul_h=0.0 if past else float(remaining),
        eol_h=float(estimate.value_h),
        current_age_h=age,
        threshold_pct=float(threshold_pct),
        method=estimate.method,
        already_past_eol=bool(past),
        estimate=estimate,
    )


# ---------------------------------------------------------------------------
# 5. Uncertainty-aware variant
# ---------------------------------------------------------------------------

def lifetime_with_uncertainty(
    time_h: np.ndarray | list[float] | pd.Series,
    mean_retention_pct: np.ndarray | list[float] | pd.Series,
    std_retention_pct: np.ndarray | list[float] | pd.Series,
    threshold_pct: float = 80.0,
    z: float = Z_95,
    allow_extrapolation: bool = True,
    n_tail: int = DEFAULT_TAIL_POINTS,
) -> LifetimeEstimate:
    """Lifetime from a mean trajectory, bounded by its +/- z-sigma trajectories.

    WHAT THIS IS NOT: a posterior over T80. It is a **trajectory-band mapping**. The
    two bounds are the lifetimes of two specific deterministic curves -- the
    pointwise mean minus z sigma and plus z sigma -- pushed through exactly the same
    interpolate-then-extrapolate logic as the central estimate.

    Those curves are not samples from the model. A pointwise band assumes the
    retention error is perfectly correlated across time, whereas a real posterior
    would draw whole trajectories and take the distribution of their crossing
    times. Because the crossing time is a nonlinear functional of the trajectory,
    the resulting interval is not a 95% credible interval for T80 and its coverage
    has not been measured. It is a legible sensitivity range: "if retention really
    sat at the pessimistic edge of the band the whole way, life would end here."

    The sign flip matters and is easy to get backwards: LOWER retention degrades to
    the threshold SOONER. So the mean-minus-z-sigma curve produces
    ``confidence_low_h`` (the early bound) and mean-plus-z-sigma produces
    ``confidence_high_h`` (the late bound). Either bound may be None when that edge
    of the band never reaches the threshold -- an open-ended interval, reported as
    such rather than closed with the horizon.
    """
    t, mean = _as_trajectory(time_h, mean_retention_pct)
    _, sigma = _as_trajectory(time_h, std_retention_pct)
    if sigma.shape != mean.shape:
        raise ValueError("mean and std trajectories must have the same usable length")
    # A negative standard deviation is a caller bug; clip rather than propagate a
    # band that crosses itself.
    sigma = np.clip(sigma, 0.0, None)

    central = lifetime_estimate(
        t, mean, threshold_pct=threshold_pct,
        allow_extrapolation=allow_extrapolation, n_tail=n_tail,
    )

    pessimistic = lifetime_estimate(
        t, mean - z * sigma, threshold_pct=threshold_pct,
        allow_extrapolation=allow_extrapolation, n_tail=n_tail,
    )
    optimistic = lifetime_estimate(
        t, mean + z * sigma, threshold_pct=threshold_pct,
        allow_extrapolation=allow_extrapolation, n_tail=n_tail,
    )

    return LifetimeEstimate(
        value_h=central.value_h,
        method=central.method,
        threshold_pct=central.threshold_pct,
        confidence_low_h=pessimistic.value_h,
        confidence_high_h=optimistic.value_h,
        fit_quality=central.fit_quality,
        horizon_h=central.horizon_h,
        final_retention_pct=central.final_retention_pct,
        decay_k_per_h=central.decay_k_per_h,
        fit_amplitude_pct=central.fit_amplitude_pct,
        confidence_low_method=pessimistic.method,
        confidence_high_method=optimistic.method,
    )


# ---------------------------------------------------------------------------
# 7. Export
# ---------------------------------------------------------------------------

def estimate_to_frame(estimate: LifetimeEstimate, label: str | None = None) -> pd.DataFrame:
    """One tidy row for display and export.

    ``value_h`` stays genuinely missing (pd.NA) when nothing was measured, so a
    downstream mean() or CSV round trip cannot turn "not reached" into a number.
    """
    return pd.DataFrame(
        [
            {
                "metric": label or estimate.label,
                "threshold_pct": estimate.threshold_pct,
                "value_h": estimate.value_h if estimate.value_h is not None else pd.NA,
                "method": estimate.method,
                "confidence_low_h": estimate.confidence_low_h if estimate.confidence_low_h is not None else pd.NA,
                "confidence_high_h": estimate.confidence_high_h if estimate.confidence_high_h is not None else pd.NA,
                "confidence_low_method": estimate.confidence_low_method or pd.NA,
                "confidence_high_method": estimate.confidence_high_method or pd.NA,
                "fit_quality_r2": estimate.fit_quality if estimate.fit_quality is not None else pd.NA,
                "decay_k_per_h": estimate.decay_k_per_h if estimate.decay_k_per_h is not None else pd.NA,
                "fit_amplitude_pct": estimate.fit_amplitude_pct if estimate.fit_amplitude_pct is not None else pd.NA,
                "horizon_h": estimate.horizon_h if estimate.horizon_h is not None else pd.NA,
                "final_retention_pct": estimate.final_retention_pct if estimate.final_retention_pct is not None else pd.NA,
                "reach_factor": estimate.reach_factor if estimate.reach_factor is not None else pd.NA,
                "note": estimate.note(),
            }
        ]
    )


def rul_to_frame(rul: RemainingUsefulLife, label: str | None = None) -> pd.DataFrame:
    """One tidy row for a remaining-useful-life result."""
    return pd.DataFrame(
        [
            {
                "metric": label or f"RUL@{rul.current_age_h:g}h",
                "threshold_pct": rul.threshold_pct,
                "current_age_h": rul.current_age_h,
                "rul_h": rul.rul_h if rul.rul_h is not None else pd.NA,
                "eol_h": rul.eol_h if rul.eol_h is not None else pd.NA,
                "method": rul.method,
                "already_past_eol": rul.already_past_eol,
                "note": rul.note(),
            }
        ]
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)

    RNG = np.random.default_rng(0)  # deterministic: fixed seed, no exceptions
    TIMES = np.array([0, 50, 100, 200, 300, 400, 500, 600, 800, 1000], dtype=float)

    def banner(title: str) -> None:
        print("\n" + "=" * 78)
        print(title)
        print("=" * 78)

    def report(name: str, est: LifetimeEstimate) -> None:
        val = "None" if est.value_h is None else f"{est.value_h:.1f} h"
        print(f"\n[{name}]  {est.label} = {val}   method = {est.method!r}")
        if est.fit_quality is not None:
            print(f"    fitted k = {est.decay_k_per_h:.6g} /h   amplitude = {est.fit_amplitude_pct:.2f}%   R2 = {est.fit_quality:.5f}")
        if est.reach_factor is not None and est.is_projection:
            print(f"    reach    = {est.reach_factor:.2f}x the simulated window")
        print(f"    note: {est.note()}")

    # ---- Case 1: fast degrader that genuinely crosses both thresholds -------
    banner("CASE 1  fast-degrading trajectory (crosses 90% and 80% in-window)")
    fast = 100.0 * np.exp(-TIMES / 900.0) - 0.004 * TIMES  # decays through 80% well before 1000 h
    print("retention:", np.round(fast, 2))
    report("fast", t90(TIMES, fast))
    report("fast", t80(TIMES, fast))

    rul = remaining_useful_life(300.0, TIMES, fast, threshold_pct=80.0)
    print(f"\n[fast]  RUL at 300 h = {rul.rul_h:.1f} h   (EOL {rul.eol_h:.1f} h, method {rul.method!r})")
    print(f"    note: {rul.note()}")
    rul_late = remaining_useful_life(900.0, TIMES, fast, threshold_pct=80.0)
    print(f"\n[fast]  RUL at 900 h = {rul_late.rul_h:.1f} h   already_past_eol = {rul_late.already_past_eol}")
    print(f"    note: {rul_late.note()}")

    # ---- Case 2: flat trajectory that never gets near the threshold ---------
    banner("CASE 2  flat trajectory (never reaches 80%)")
    flat = 100.0 - 0.0008 * TIMES  # loses less than 1% over the whole window
    print("retention:", np.round(flat, 3))
    est_flat = t80(TIMES, flat)
    report("flat", est_flat)
    print(f"    raw threshold_time() returns: {threshold_time(TIMES, flat, 80.0)!r}   <- None, not a fabricated number")
    fit_flat = fit_exponential_tail(TIMES, flat)
    print(f"    tail fit: k = {fit_flat.k_per_h:.6g} /h, R2 = {fit_flat.r2:.5f}, "
          f"solve(80%) = {fit_flat.solve_time(80.0):.0f} h  (rejected: {est_flat.method!r})")

    est_flat_noextrap = t80(TIMES, flat, allow_extrapolation=False)
    print(f"    with allow_extrapolation=False: value = {est_flat_noextrap.value_h!r}, method = {est_flat_noextrap.method!r}")

    rul_flat = remaining_useful_life(300.0, TIMES, flat, threshold_pct=80.0, allow_extrapolation=False)
    print(f"\n[flat]  RUL at 300 h = {rul_flat.rul_h!r}   note: {rul_flat.note()}")

    # ---- Case 3: noisy, non-monotone trajectory ----------------------------
    banner("CASE 3  noisy non-monotone trajectory (first crossing wins)")
    smooth = 100.0 * np.exp(-TIMES / 1500.0)
    noise = RNG.normal(0.0, 2.5, size=TIMES.size)
    noisy = smooth + noise
    noisy[5] = 79.0   # a dip below 80% at 400 h ...
    noisy[6] = 86.0   # ... that recovers afterwards
    print("retention:", np.round(noisy, 2))
    print("monotone decreasing?", bool(np.all(np.diff(noisy) <= 0)))
    report("noisy", t80(TIMES, noisy))
    print(f"    all in-window samples <= 80%: indices {np.flatnonzero(noisy <= 80.0).tolist()} "
          f"-> first crossing is taken, later ones ignored")

    # ---- Uncertainty-aware variant -----------------------------------------
    banner("CASE 4  uncertainty-aware T80 (trajectory-band mapping, not a posterior)")
    mean_traj = 100.0 * np.exp(-TIMES / 2600.0)
    std_traj = 0.4 + 0.006 * TIMES  # uncertainty grows with extrapolated age
    print("mean:", np.round(mean_traj, 2))
    print("std :", np.round(std_traj, 2))
    band = lifetime_with_uncertainty(TIMES, mean_traj, std_traj, threshold_pct=80.0)
    report("band", band)
    print(f"    low  bound {band.confidence_low_h:.1f} h  (method {band.confidence_low_method!r})")
    print(f"    high bound {band.confidence_high_h:.1f} h  (method {band.confidence_high_method!r})")
    print(f"    ordering low <= central <= high: {band.confidence_low_h <= band.value_h <= band.confidence_high_h}")

    banner("CASE 5  uncertainty band with an open upper end")
    mean2 = 100.0 - 0.011 * TIMES
    std2 = 0.5 + 0.010 * TIMES
    band2 = lifetime_with_uncertainty(TIMES, mean2, std2, threshold_pct=90.0)
    report("open-band", band2)
    print(f"    low bound {band2.confidence_low_h!r} ({band2.confidence_low_method!r}), "
          f"high bound {band2.confidence_high_h!r} ({band2.confidence_high_method!r})")

    # ---- Pinned vs free amplitude ------------------------------------------
    banner("CASE 6  pinned amplitude (literal 100*exp(-kt)) vs fitted amplitude")
    decayed = 100.0 * np.exp(-TIMES / 900.0) - 0.0035 * TIMES
    free = fit_exponential_tail(TIMES, decayed, n_tail=5, pin_amplitude=False)
    pinned = fit_exponential_tail(TIMES, decayed, n_tail=5, pin_amplitude=True)
    print(f"    free   : k = {free.k_per_h:.6g} /h, A = {free.amplitude_pct:.2f}%, R2 = {free.r2:.5f}, T80 = {free.solve_time(80.0):.1f} h")
    print(f"    pinned : k = {pinned.k_per_h:.6g} /h, A = {pinned.amplitude_pct:.2f}%, R2 = {pinned.r2:.5f}, T80 = {pinned.solve_time(80.0):.1f} h")

    # ---- Export frames ------------------------------------------------------
    banner("CASE 7  tidy export frames")
    frame = pd.concat(
        [
            estimate_to_frame(t80(TIMES, fast), label="T80 fast"),
            estimate_to_frame(t80(TIMES, flat), label="T80 flat"),
            estimate_to_frame(t80(TIMES, noisy), label="T80 noisy"),
            estimate_to_frame(band, label="T80 band"),
        ],
        ignore_index=True,
    )
    cols = ["metric", "threshold_pct", "value_h", "method", "confidence_low_h",
            "confidence_high_h", "fit_quality_r2", "reach_factor", "horizon_h", "final_retention_pct"]
    print(frame[cols].to_string(index=False))
    print("\nvalue_h dtype:", frame["value_h"].dtype, " -- 'not reached' stays <NA>, never 0")
    print("\nRUL frame:")
    print(rul_to_frame(rul).drop(columns=["note"]).to_string(index=False))

    # ---- Edge cases ---------------------------------------------------------
    banner("CASE 8  edge cases")
    print("already below threshold at t=0:",
          threshold_time([0.0, 100.0], [75.0, 70.0], 80.0), "h")
    print("exact hit on a sample point   :",
          threshold_time([0.0, 100.0, 200.0], [100.0, 80.0, 60.0], 80.0), "h")
    print("rising trajectory, no crossing:",
          threshold_time([0.0, 100.0], [90.0, 95.0], 80.0))
    nan_traj = np.array([100.0, np.nan, 90.0, 85.0, 78.0])
    print("NaN dropped, crossing still found:",
          round(threshold_time([0.0, 50.0, 100.0, 200.0, 300.0], nan_traj, 80.0), 2), "h")
    flat_line = fit_exponential_tail(TIMES, np.full_like(TIMES, 95.0))
    print(f"perfectly flat tail: k = {flat_line.k_per_h:.3g}, R2 = {flat_line.r2:.3g} (0.0 by convention), "
          f"solve(80%) = {flat_line.solve_time(80.0)!r}")
    try:
        threshold_time([0.0], [100.0], 80.0)
    except ValueError as exc:
        print("too-short trajectory raises:", exc)

    # ---- Real COMSOL trajectories, if the dataset is importable -------------
    try:
        from psc_twin.data import load_doe
    except Exception as exc:  # pragma: no cover
        print(f"\n(skipping real-data check: {exc})")
    else:
        banner("CASE 9  real COMSOL runs from the 36-run campaign")
        metrics = load_doe().metrics
        rows = []
        for run_id, block in metrics.groupby("run_id"):
            t = block["aging_h"].to_numpy(float)
            r = block["PCE_retention_pct"].to_numpy(float)
            est = t80(t, r)
            rows.append(
                {
                    "suns": block["aging_light_suns"].iloc[0],
                    "temp_C": block["aging_temperature_C"].iloc[0],
                    "final_ret_pct": round(float(r[-1]), 2),
                    "T80_h": None if est.value_h is None else round(est.value_h, 1),
                    "method": est.method,
                    "R2": None if est.fit_quality is None else round(est.fit_quality, 4),
                }
            )
        real = pd.DataFrame(rows).sort_values(["temp_C", "suns"]).reset_index(drop=True)
        print(real.to_string(index=False))
        print("\nmethod counts over all 36 runs:")
        print(real["method"].value_counts().to_string())
