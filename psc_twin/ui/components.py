"""Reusable UI pieces that enforce the product's honesty rules.

These are not decoration. The tier system only works if every page renders it
identically, so pages are expected to reach for these helpers rather than
hand-rolling their own badges and warnings.

The one invariant worth restating: ``planned_card`` exists so that a page can
say "here is what this will do" in the exact place a number would otherwise
go. A Tier.PLANNED capability must route to it and stop.
"""

from __future__ import annotations

import html
from collections.abc import Iterable, Sequence

import pandas as pd
import streamlit as st

from psc_twin.capabilities import (
    TIER_BLURB,
    TIER_ICON,
    TIER_LABEL,
    Capability,
    Tier,
)
from psc_twin.data import GLOSSARY

_TIER_CLASS = {
    Tier.VALIDATED: "tier-validated",
    Tier.PREVIEW: "tier-preview",
    Tier.PLANNED: "tier-planned",
}


# --------------------------------------------------------------------------
# tier vocabulary
# --------------------------------------------------------------------------
def tier_pill_html(tier: Tier, text: str | None = None) -> str:
    label = text or TIER_LABEL[tier]
    return (
        f'<span class="tier-pill {_TIER_CLASS[tier]}">'
        f"{TIER_ICON[tier]} {html.escape(label)}</span>"
    )


def tier_pill(tier: Tier, text: str | None = None) -> None:
    st.markdown(tier_pill_html(tier, text), unsafe_allow_html=True)


def tier_row(tiers: Iterable[tuple[Tier, str]]) -> None:
    """A run of pills on one line, e.g. the legend under the page title."""
    st.markdown(
        " ".join(tier_pill_html(tier, text) for tier, text in tiers),
        unsafe_allow_html=True,
    )


def banner(body: str, kind: str = "info", title: str | None = None) -> None:
    """A left-accented notice. ``kind`` is validated/preview/planned/info."""
    head = f"<strong>{html.escape(title)}</strong><br>" if title else ""
    st.markdown(
        f'<div class="tw-banner tw-banner-{kind}">{head}{body}</div>',
        unsafe_allow_html=True,
    )


def tier_banner(cap: Capability) -> None:
    """Explain, in the page itself, why a capability is at the tier it is."""
    kind = {
        Tier.VALIDATED: "validated",
        Tier.PREVIEW: "preview",
        Tier.PLANNED: "planned",
    }[cap.tier]
    body = (
        f"{html.escape(TIER_BLURB[cap.tier])}<br>"
        f'<span class="tw-caption">Backing: {html.escape(cap.backing)}</span>'
    )
    if cap.unlocks:
        body += (
            f'<br><span class="tw-caption">Unlocked by: '
            f"{html.escape(cap.unlocks)}</span>"
        )
    banner(body, kind=kind, title=f"{TIER_ICON[cap.tier]} {cap.label} - {TIER_LABEL[cap.tier]}")


def planned_card(cap: Capability) -> None:
    """Stand in for results that must never be shown.

    This is the enforcement point for the product's central rule. When a page
    resolves a Tier.PLANNED capability it calls this and returns; no numbers,
    no chart, no placeholder trajectory that a screenshot could misrepresent.
    """
    version = cap.version or "a future release"
    st.markdown(
        f"""
        <div class="tw-card tw-card-planned">
          <h4>{TIER_ICON[Tier.PLANNED]} {html.escape(cap.label)} &mdash; planned for {html.escape(version)}</h4>
          <p><strong>Why there are no numbers here.</strong> {html.escape(cap.backing)}</p>
          <p style="margin-top:0.55rem;"><strong>What would unlock it.</strong> {html.escape(cap.unlocks or 'Further simulation work.')}</p>
          <p style="margin-top:0.55rem;">This app deliberately shows nothing rather than a
          guess, because a plausible-looking number here would be indistinguishable from a
          validated one.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def preview_zone_open() -> None:
    """Begin a visually de-emphasised region for Tier.PREVIEW results."""
    st.markdown('<div class="tw-preview-zone">', unsafe_allow_html=True)


def preview_zone_close() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# orientation
# --------------------------------------------------------------------------
def step_rail(steps: Sequence[str], active: int, done: Iterable[int] = ()) -> None:
    """A breadcrumb of the three-step flow, so nobody wonders where they are."""
    done_set = set(done)
    chunks = []
    for i, name in enumerate(steps):
        cls = "tw-step"
        if i == active:
            cls += " tw-step-active"
        elif i in done_set:
            cls += " tw-step-done"
        chunks.append(
            f'<span class="{cls}"><span class="tw-step-num">{i + 1}</span>'
            f"{html.escape(name)}</span>"
        )
    st.markdown(f'<div class="tw-steps">{"".join(chunks)}</div>', unsafe_allow_html=True)


def eyebrow(text: str) -> None:
    st.markdown(f'<div class="tw-eyebrow">{html.escape(text)}</div>', unsafe_allow_html=True)


def caption(text: str) -> None:
    st.markdown(f'<div class="tw-caption">{text}</div>', unsafe_allow_html=True)


def card(title: str, body: str, planned: bool = False) -> None:
    cls = "tw-card tw-card-planned" if planned else "tw-card"
    st.markdown(
        f'<div class="{cls}"><h4>{html.escape(title)}</h4><p>{body}</p></div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# glossary
# --------------------------------------------------------------------------
def term(name: str, display: str | None = None) -> str:
    """Inline glossary term with a native browser tooltip."""
    definition = GLOSSARY.get(name, "")
    shown = html.escape(display or name)
    if not definition:
        return shown
    return f'<span class="tw-term" title="{html.escape(definition)}">{shown}</span>'


def glossary_expander(keys: Sequence[str] | None = None) -> None:
    """A plain-language dictionary. Beginners open it; experts ignore it."""
    items = keys or list(GLOSSARY)
    with st.expander("Plain-language glossary", expanded=False):
        for key in items:
            if key in GLOSSARY:
                st.markdown(f"**{key}** &mdash; {GLOSSARY[key]}")


def help_for(*keys: str) -> str:
    """Assemble tooltip text for a Streamlit widget's ``help=`` argument."""
    return "\n\n".join(f"{k}: {GLOSSARY[k]}" for k in keys if k in GLOSSARY)


# --------------------------------------------------------------------------
# the design envelope, made visible
# --------------------------------------------------------------------------
def envelope_meter(
    label: str,
    value: float,
    tested_low: float,
    tested_high: float,
    axis_low: float,
    axis_high: float,
    unit: str = "",
) -> None:
    """Show where a chosen condition sits relative to what was simulated.

    The green band is the tested range; the marker is the user's choice and
    turns amber when it leaves the band. This single control does more to
    convey the model's limits than any amount of prose.
    """
    span = max(axis_high - axis_low, 1e-9)

    def pct(x: float) -> float:
        return max(0.0, min(100.0, (float(x) - axis_low) / span * 100.0))

    left, right = pct(tested_low), pct(tested_high)
    marker = pct(value)
    outside = not (tested_low <= float(value) <= tested_high)
    marker_cls = "tw-meter-marker tw-meter-marker-out" if outside else "tw-meter-marker"

    st.markdown(
        f"""
        <div class="tw-meter">
          <div class="tw-caption" style="margin-bottom:0.3rem;">
            {html.escape(label)}: <strong>{value:g}{html.escape(unit)}</strong>
            {'<span style="color:#D97706;"> &mdash; outside the tested range</span>' if outside else ''}
          </div>
          <div class="tw-meter-track">
            <div class="tw-meter-band" style="left:{left:.2f}%; width:{max(right - left, 0.5):.2f}%;"></div>
            <div class="{marker_cls}" style="left:{marker:.2f}%;"></div>
          </div>
          <div class="tw-meter-labels">
            <span>{axis_low:g}{html.escape(unit)}</span>
            <span>tested {tested_low:g}&ndash;{tested_high:g}{html.escape(unit)}</span>
            <span>{axis_high:g}{html.escape(unit)}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def excursion_notice(excursions: Sequence[str]) -> None:
    """Render the plain-language list of envelope violations, if any."""
    if not excursions:
        return
    items = "".join(f"<li>{html.escape(item)}</li>" for item in excursions)
    banner(
        "The request sits outside the simulated design envelope, so these results come "
        "from the fallback engine and carry deliberately widened error bars."
        f"<ul style='margin:0.45rem 0 0 1.1rem; padding:0;'>{items}</ul>",
        kind="preview",
        title="Extrapolating beyond the tested conditions",
    )


# --------------------------------------------------------------------------
# misc
# --------------------------------------------------------------------------
def metric_grid(entries: Sequence[tuple[str, str, str]]) -> None:
    """A row of metrics as (label, value, help) triples."""
    cols = st.columns(len(entries))
    for col, (label, value, helptext) in zip(cols, entries):
        col.metric(label, value, help=helptext or None)


def dataframe(df: pd.DataFrame, **kwargs) -> None:
    st.dataframe(df, width="stretch", hide_index=True, **kwargs)


def roadmap_toggle() -> bool:
    """The switch that reconciles a clean default with an ambitious vision.

    Off, the app shows only what it can actually do. On, planned capabilities
    appear greyed in place so a visitor can read the trajectory of the work.
    """
    return st.sidebar.toggle(
        "Show roadmap features",
        value=False,
        help=(
            "Reveal capabilities that are planned but not yet supported. They "
            "appear greyed out and never produce numbers."
        ),
    )
