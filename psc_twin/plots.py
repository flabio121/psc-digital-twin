"""Figures for the app and the paper.

Every figure is built by a pure function returning a Matplotlib ``Figure``, so
the same code makes the on-screen chart and the publication asset. Nothing here
imports Streamlit.

Two conventions run through the module:

* An uncertainty band is always the 95% predictive interval, drawn as a shaded
  region behind the mean line. If a caller has no ``_std`` column the band is
  simply omitted rather than faked.
* The tested design envelope is drawn wherever it is meaningful, because the
  single most useful thing a reader can know about any prediction here is
  whether it sits inside the box the simulations actually covered.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from psc_twin.capabilities import ENVELOPE, ENVELOPE_LEVELS
from psc_twin.data import TARGET_LABELS, TARGET_SHORT
from psc_twin.materials import LAYERS
from psc_twin.ui.theme import (
    BAND_ALPHA,
    BORDER_STRONG,
    PRIMARY,
    SERIES,
    TEXT_FAINT,
    TEXT_MUTED,
    TIER_COLOR,
    apply_matplotlib_style,
)
from psc_twin.capabilities import Tier

apply_matplotlib_style()

_Z95 = 1.959963985


def _finish(fig: Figure) -> Figure:
    fig.tight_layout()
    return fig


def _band(ax, x, mean, std, color: str, label: str | None = None) -> None:
    if std is None:
        return
    std = np.asarray(std, dtype=float)
    if not np.isfinite(std).any() or np.allclose(std, 0.0):
        return
    lo = np.asarray(mean) - _Z95 * std
    hi = np.asarray(mean) + _Z95 * std
    ax.fill_between(x, lo, hi, color=color, alpha=BAND_ALPHA, linewidth=0, label=label)


def cell_stack(materials: dict[str, str]) -> Figure:
    """Draw the currently selected p-i-n stack, illuminated side first."""
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    heights = {
        "front_barrier": 0.28,
        "substrate": 0.62,
        "front_contact": 0.48,
        "htl": 0.42,
        "absorber": 1.55,
        "etl": 0.46,
        "rear_contact": 0.52,
        "rear_barrier": 0.28,
    }
    y = 0.0

    active_layers = [layer for layer in LAYERS if materials[layer.key] != "None"]
    for layer in reversed(active_layers):
        height = heights[layer.key]
        selected = materials[layer.key]
        changed = selected != layer.baseline
        face = "#E2E8F0" if changed else layer.color
        text_color = TEXT_MUTED if changed else ("white" if layer.key in {"htl", "absorber", "etl"} else "#0F172A")
        rect = Rectangle(
            (0.1, y),
            0.72,
            height,
            facecolor=face,
            edgecolor="#94A3B8" if changed else "white",
            linewidth=1.5,
            hatch="///" if changed else None,
        )
        ax.add_patch(rect)
        lock = "  LOCKED" if changed else ""
        ax.text(
            0.46,
            y + height / 2,
            f"{layer.label}\n{selected}{lock}",
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="600",
            color=text_color,
        )
        ax.text(
            0.86,
            y + height / 2,
            layer.thickness if layer.baseline != "None" else "planned",
            ha="left",
            va="center",
            fontsize=8,
            color=TEXT_FAINT,
        )
        y += height

    ax.annotate(
        "LIGHT",
        xy=(0.46, y + 0.02),
        xytext=(0.46, y + 0.52),
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="700",
        color="#D97706",
        arrowprops={"arrowstyle": "-|>", "color": "#D97706", "lw": 2.0},
    )
    ax.text(0.46, -0.25, "rear electrode", ha="center", va="top", fontsize=8, color=TEXT_FAINT)
    ax.set_xlim(0.0, 1.15)
    ax.set_ylim(-0.42, y + 0.72)
    ax.axis("off")
    ax.set_title("Your layer stack", pad=8)
    return _finish(fig)


# --------------------------------------------------------------------------
# trajectories
# --------------------------------------------------------------------------
def trajectory(
    df: pd.DataFrame,
    target: str,
    *,
    time_col: str = "aging_h",
    color: str = PRIMARY,
    show_band: bool = True,
    title: str | None = None,
) -> Figure:
    """One predicted metric against aging time, with its 95% interval."""
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    x = df[time_col].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=float)

    std_col = f"{target}_std"
    if show_band and std_col in df.columns:
        _band(ax, x, y, df[std_col].to_numpy(dtype=float), color, "95% predictive interval")

    ax.plot(x, y, color=color, label=TARGET_SHORT.get(target, target), zorder=3)
    ax.set_xlabel("Aging time (h)")
    ax.set_ylabel(TARGET_LABELS.get(target, target))
    if title:
        ax.set_title(title)

    hi = ENVELOPE["aging_h"][1]
    if x.max() > hi:
        ax.axvspan(hi, x.max(), color=TIER_COLOR[Tier.PREVIEW], alpha=0.07, zorder=0)
        ax.axvline(hi, color=TIER_COLOR[Tier.PREVIEW], linestyle="--", linewidth=1.2, zorder=2)
        ax.text(
            hi,
            ax.get_ylim()[1],
            "  beyond simulated window",
            color=TIER_COLOR[Tier.PREVIEW],
            fontsize=8,
            va="top",
        )
    if any(h for h in ax.get_legend_handles_labels()[1]):
        ax.legend(loc="best")
    return _finish(fig)


def retention_with_lifetime(
    df: pd.DataFrame,
    *,
    time_col: str = "aging_h",
    retention_col: str = "PCE_retention_pct",
    t80_h: float | None = None,
    t90_h: float | None = None,
) -> Figure:
    """Retention curve annotated with the lifetime thresholds a reader cares about."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = df[time_col].to_numpy(dtype=float)
    y = df[retention_col].to_numpy(dtype=float)

    std_col = f"{retention_col}_std"
    if std_col in df.columns:
        _band(ax, x, y, df[std_col].to_numpy(dtype=float), PRIMARY, "95% predictive interval")

    ax.plot(x, y, color=PRIMARY, zorder=3, label="Predicted retention")

    for threshold, tval, name, colour in (
        (90.0, t90_h, "T90", SERIES[4]),
        (80.0, t80_h, "T80", SERIES[1]),
    ):
        ax.axhline(threshold, color=colour, linewidth=1.0, linestyle=":", zorder=1)
        ax.text(x.min(), threshold + 0.4, f"{threshold:g}%", color=colour, fontsize=8)
        if tval is not None and np.isfinite(tval) and x.min() <= tval <= x.max():
            ax.axvline(tval, color=colour, linewidth=1.0, linestyle="--", zorder=1)
            ax.plot([tval], [threshold], "o", color=colour, markersize=6, zorder=4)
            ax.annotate(
                f"{name} = {tval:,.0f} h",
                xy=(tval, threshold),
                xytext=(6, 10),
                textcoords="offset points",
                fontsize=8.5,
                color=colour,
                fontweight="600",
            )

    ax.set_xlabel("Aging time (h)")
    ax.set_ylabel("Efficiency retained (%)")
    ax.set_ylim(min(70.0, float(np.nanmin(y)) - 3.0), 102.0)
    ax.legend(loc="lower left")
    return _finish(fig)


def multi_metric(df: pd.DataFrame, targets: Sequence[str], *, time_col: str = "aging_h") -> Figure:
    """Small-multiples of several predicted metrics sharing a time axis."""
    n = len(targets)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.9 * ncols, 2.9 * nrows), squeeze=False)
    x = df[time_col].to_numpy(dtype=float)

    for ax, target, colour in zip(axes.ravel(), targets, SERIES):
        y = df[target].to_numpy(dtype=float)
        std_col = f"{target}_std"
        if std_col in df.columns:
            _band(ax, x, y, df[std_col].to_numpy(dtype=float), colour)
        ax.plot(x, y, color=colour, zorder=3)
        ax.set_title(TARGET_LABELS.get(target, target), fontsize=9.5)
        ax.set_xlabel("Aging time (h)", fontsize=8.5)

    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    return _finish(fig)


# --------------------------------------------------------------------------
# J-V curves
# --------------------------------------------------------------------------
def jv_curve(
    voltage: np.ndarray,
    current: np.ndarray,
    *,
    std: np.ndarray | None = None,
    reference: tuple[np.ndarray, np.ndarray] | None = None,
    mpp: tuple[float, float] | None = None,
    title: str | None = None,
) -> Figure:
    """A predicted J-V curve, optionally against a fresh-cell reference."""
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    v = np.asarray(voltage, dtype=float)
    j = np.asarray(current, dtype=float)

    if reference is not None:
        ax.plot(
            np.asarray(reference[0], dtype=float),
            np.asarray(reference[1], dtype=float),
            color=TEXT_FAINT,
            linestyle="--",
            linewidth=1.5,
            label="Fresh cell (0 h)",
            zorder=2,
        )

    _band(ax, v, j, std, PRIMARY, "95% predictive interval")
    ax.plot(v, j, color=PRIMARY, label="Predicted", zorder=3)

    if mpp is not None and all(np.isfinite(mpp)):
        vm, jm = mpp
        ax.plot([vm], [jm], "o", color=SERIES[2], markersize=7, zorder=5)
        ax.annotate(
            "maximum power point",
            xy=(vm, jm),
            xytext=(8, -14),
            textcoords="offset points",
            fontsize=8.5,
            color=SERIES[2],
            fontweight="600",
        )
        ax.add_patch(
            Rectangle(
                (0, 0), vm, jm, facecolor=SERIES[2], alpha=0.08, edgecolor="none", zorder=1
            )
        )

    ax.axhline(0, color=BORDER_STRONG, linewidth=1.0, zorder=1)
    ax.axvline(0, color=BORDER_STRONG, linewidth=1.0, zorder=1)

    # Past the open-circuit voltage the current dives steeply negative. That is
    # real physics but it is not the part anyone reads, and letting it set the
    # axis limits squashes the whole power-producing quadrant into a sliver.
    # Frame on the operating region instead.
    jsc = float(np.interp(0.0, v, j))
    if np.isfinite(jsc) and jsc > 0:
        ax.set_ylim(-0.18 * jsc, 1.25 * jsc)
        crossing = np.where((j[:-1] > 0) & (j[1:] <= 0))[0]
        if crossing.size:
            voc = float(v[crossing[0] + 1])
            ax.set_xlim(float(v.min()), min(float(v.max()), voc * 1.15))

    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Current density (mA/cm$^2$)")
    if title:
        ax.set_title(title)
    ax.legend(loc="lower left")
    return _finish(fig)


def jv_family(
    voltage: np.ndarray,
    curves: Sequence[tuple[float, np.ndarray]],
    *,
    title: str | None = None,
) -> Figure:
    """A fan of J-V curves across aging times: degradation at a glance."""
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    v = np.asarray(voltage, dtype=float)
    cmap = plt.get_cmap("viridis")
    times = [t for t, _ in curves]
    lo, hi = (min(times), max(times)) if times else (0.0, 1.0)
    span = max(hi - lo, 1e-9)

    jsc_ref = 0.0
    for t, j in curves:
        arr = np.asarray(j, dtype=float)
        ax.plot(v, arr, color=cmap(0.12 + 0.78 * (t - lo) / span), linewidth=1.7)
        jsc_ref = max(jsc_ref, float(np.interp(0.0, v, arr)))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=lo, vmax=hi))
    fig.colorbar(sm, ax=ax, label="Aging time (h)", pad=0.02)

    # Same framing rule as `jv_curve`: keep the operating quadrant readable.
    if jsc_ref > 0:
        ax.set_ylim(-0.18 * jsc_ref, 1.25 * jsc_ref)

    ax.axhline(0, color=BORDER_STRONG, linewidth=1.0)
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Current density (mA/cm$^2$)")
    if title:
        ax.set_title(title)
    return _finish(fig)


# --------------------------------------------------------------------------
# the design envelope and where a query sits in it
# --------------------------------------------------------------------------
def envelope_map(
    query: tuple[float, float] | None = None,
    *,
    recommendations: pd.DataFrame | None = None,
    title: str | None = None,
) -> Figure:
    """The 6x6 design grid, the query point, and any suggested next runs.

    This is the clearest single explanation of what the model does and does not
    know: the dots are simulations that were actually run.
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    suns = np.array(ENVELOPE_LEVELS["illumination_suns"], dtype=float)
    temps = np.array(ENVELOPE_LEVELS["temperature_c"], dtype=float)
    grid_s, grid_t = np.meshgrid(suns, temps)

    lo_s, hi_s = ENVELOPE["illumination_suns"]
    lo_t, hi_t = ENVELOPE["temperature_c"]
    ax.add_patch(
        Rectangle(
            (lo_s, lo_t),
            hi_s - lo_s,
            hi_t - lo_t,
            facecolor=TIER_COLOR[Tier.VALIDATED],
            alpha=0.06,
            edgecolor=TIER_COLOR[Tier.VALIDATED],
            linestyle="--",
            linewidth=1.2,
            zorder=0,
        )
    )

    ax.scatter(
        grid_s.ravel(),
        grid_t.ravel(),
        s=42,
        facecolor="white",
        edgecolor=TIER_COLOR[Tier.VALIDATED],
        linewidth=1.5,
        zorder=3,
        label="Simulated (36 runs)",
    )

    if recommendations is not None and len(recommendations):
        cols = set(recommendations.columns)
        scol = "aging_light_suns" if "aging_light_suns" in cols else "illumination_suns"
        tcol = "aging_temperature_C" if "aging_temperature_C" in cols else "temperature_c"
        if scol in cols and tcol in cols:
            ax.scatter(
                recommendations[scol],
                recommendations[tcol],
                s=120,
                marker="*",
                color=SERIES[3],
                edgecolor="white",
                linewidth=0.8,
                zorder=4,
                label="Suggested next runs",
            )

    if query is not None:
        qs, qt = query
        inside = lo_s <= qs <= hi_s and lo_t <= qt <= hi_t
        colour = PRIMARY if inside else TIER_COLOR[Tier.PREVIEW]
        ax.scatter(
            [qs], [qt], s=190, marker="X", color=colour, edgecolor="white",
            linewidth=1.4, zorder=5,
            label="Your conditions" if inside else "Your conditions (outside tested range)",
        )

    ax.set_xlabel("Illumination (suns)")
    ax.set_ylabel("Temperature (C)")
    ax.set_title(title or "Where your conditions sit in the simulated design space")
    ax.legend(loc="upper left", fontsize=8.5)
    return _finish(fig)


# --------------------------------------------------------------------------
# validation figures
# --------------------------------------------------------------------------
def parity(
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    std: np.ndarray | None = None,
    target: str = "",
    metrics: dict[str, float] | None = None,
) -> Figure:
    """Held-out actual vs predicted. The first plot any reviewer looks for."""
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)

    lo = float(min(a.min(), p.min()))
    hi = float(max(a.max(), p.max()))
    pad = 0.05 * max(hi - lo, 1e-9)
    lims = (lo - pad, hi + pad)

    ax.plot(lims, lims, color=TEXT_FAINT, linestyle="--", linewidth=1.2, zorder=1)
    if std is not None:
        ax.errorbar(
            a, p, yerr=_Z95 * np.asarray(std, dtype=float), fmt="none",
            ecolor=PRIMARY, alpha=0.28, linewidth=0.9, zorder=2,
        )
    ax.scatter(a, p, s=26, color=PRIMARY, alpha=0.75, edgecolor="white", linewidth=0.5, zorder=3)

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"COMSOL {TARGET_SHORT.get(target, target)}")
    ax.set_ylabel(f"Surrogate {TARGET_SHORT.get(target, target)}")
    ax.set_title(TARGET_LABELS.get(target, target), fontsize=10)

    if metrics:
        text = "\n".join(
            f"{k} = {v:.4g}" for k, v in metrics.items() if isinstance(v, (int, float))
        )
        ax.text(
            0.04, 0.96, text, transform=ax.transAxes, va="top", ha="left",
            fontsize=8.5, color=TEXT_MUTED,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=BORDER_STRONG, linewidth=0.8),
        )
    return _finish(fig)


def learning_curve(
    df: pd.DataFrame,
    *,
    x_col: str = "n_training_runs",
    y_col: str = "rmse",
    target: str = "",
) -> Figure:
    """Error against training-set size: the evidence for "is 36 runs enough?"."""
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    grouped = df.groupby(x_col)[y_col]
    x = np.array(sorted(grouped.groups))
    mean = grouped.mean().reindex(x).to_numpy()

    if grouped.count().max() > 1:
        sd = grouped.std().reindex(x).to_numpy()
        ax.fill_between(x, mean - sd, mean + sd, color=PRIMARY, alpha=BAND_ALPHA, linewidth=0)

    ax.plot(x, mean, color=PRIMARY, marker="o", markersize=5, zorder=3)
    ax.set_xlabel("Number of COMSOL runs used for training")
    ax.set_ylabel(f"Held-out {y_col.upper()}")
    ax.set_title(f"Learning curve{' - ' + TARGET_SHORT.get(target, target) if target else ''}", fontsize=10)
    return _finish(fig)


def residual_map(df: pd.DataFrame, *, value_col: str = "residual") -> Figure:
    """Residuals across the stress plane, to expose systematic bias."""
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    scol = "aging_light_suns" if "aging_light_suns" in df.columns else "illumination_suns"
    tcol = "aging_temperature_C" if "aging_temperature_C" in df.columns else "temperature_c"

    v = df[value_col].to_numpy(dtype=float)
    lim = float(np.nanmax(np.abs(v))) if np.isfinite(v).any() else 1.0
    sc = ax.scatter(
        df[scol], df[tcol], c=v, s=90, cmap="RdBu_r", vmin=-lim, vmax=lim,
        edgecolor="white", linewidth=0.7, zorder=3,
    )
    fig.colorbar(sc, ax=ax, label="Residual (predicted - actual)", pad=0.02)
    ax.set_xlabel("Illumination (suns)")
    ax.set_ylabel("Temperature (C)")
    ax.set_title("Where the surrogate is biased", fontsize=10)
    return _finish(fig)


def pod_modes(voltage: np.ndarray, mean_curve: np.ndarray, modes: np.ndarray, explained: Sequence[float]) -> Figure:
    """The characteristic shapes the J-V surrogate builds every curve from."""
    k = int(modes.shape[0])
    fig, axes = plt.subplots(1, k + 1, figsize=(2.7 * (k + 1), 3.1), squeeze=False)
    v = np.asarray(voltage, dtype=float)

    axes[0][0].plot(v, np.asarray(mean_curve, dtype=float), color=TEXT_MUTED)
    axes[0][0].set_title("Mean curve", fontsize=9.5)
    axes[0][0].set_xlabel("V (V)")
    axes[0][0].set_ylabel("J (mA/cm$^2$)")

    for i in range(k):
        ax = axes[0][i + 1]
        ax.plot(v, modes[i], color=SERIES[i % len(SERIES)])
        share = explained[i] if i < len(explained) else float("nan")
        ax.set_title(f"Mode {i + 1} ({share * 100:.2f}%)", fontsize=9.5)
        ax.set_xlabel("V (V)")
        ax.axhline(0, color=BORDER_STRONG, linewidth=0.9)
    return _finish(fig)


# --------------------------------------------------------------------------
# climate
# --------------------------------------------------------------------------
def climate_schedule(df: pd.DataFrame, *, name: str = "") -> Figure:
    """Monthly cell temperature and effective illumination for a climate."""
    fig, ax1 = plt.subplots(figsize=(7.2, 4.0))
    tcol = next((c for c in ("cell_temperature_C", "temperature_c", "T_cell_C") if c in df.columns), None)
    scol = next((c for c in ("illumination_suns", "suns", "effective_suns") if c in df.columns), None)
    xcol = next((c for c in ("month", "month_index", "step") if c in df.columns), None)
    x = df[xcol].to_numpy() if xcol else np.arange(len(df))

    if tcol:
        ax1.plot(x, df[tcol], color=SERIES[1], marker="o", markersize=4, label="Cell temperature")
        ax1.set_ylabel("Cell temperature (C)", color=SERIES[1])
        ax1.tick_params(axis="y", labelcolor=SERIES[1])
        hi = ENVELOPE["temperature_c"][1]
        ax1.axhline(hi, color=TIER_COLOR[Tier.PREVIEW], linestyle="--", linewidth=1.1)
        ax1.text(x[0], hi + 1.5, "tested ceiling", fontsize=8, color=TIER_COLOR[Tier.PREVIEW])

    if scol:
        ax2 = ax1.twinx()
        ax2.plot(x, df[scol], color=SERIES[0], marker="s", markersize=4, label="Illumination")
        ax2.set_ylabel("Mean daylight illumination (suns)", color=SERIES[0])
        ax2.tick_params(axis="y", labelcolor=SERIES[0])
        ax2.grid(False)
        # Anchor at zero. Mean daylight intensity barely moves through the year
        # (a desert sits near 0.5 suns in every month), and autoscaling a few
        # percent of variation draws it as a dramatic seasonal swing it is not.
        top = float(np.nanmax(df[scol].to_numpy(dtype=float)))
        ax2.set_ylim(0.0, max(top * 1.25, 0.1))

    ax1.set_xlabel("Month")
    ax1.set_title(f"Climate stress schedule{' - ' + name if name else ''}", fontsize=10)
    return _finish(fig)


def mechanism_bars(df: pd.DataFrame, *, value_col: str = "weight", label_col: str = "mechanism") -> Figure:
    """Relative mechanism weighting. Always rendered as heuristic, never as fact."""
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    order = df.sort_values(value_col, ascending=True)
    labels = [str(v).replace("_", " ") for v in order[label_col]]
    ax.barh(labels, order[value_col].to_numpy(dtype=float),
            color=TIER_COLOR[Tier.PREVIEW], alpha=0.85, edgecolor="white")
    ax.set_xlabel("Relative weight (heuristic, not validated)")
    ax.grid(axis="y", visible=False)
    return _finish(fig)
