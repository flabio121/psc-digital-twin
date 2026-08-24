"""Landing page: what this is, in the first fifteen seconds.

Written for someone who has never heard of a perovskite. The jargon is
introduced once, in plain words, with the technical term in brackets after the
explanation rather than before it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import streamlit as st

from psc_twin.capabilities import (
    TIER_BLURB,
    Tier,
    by_tier,
    tier_counts,
)
from psc_twin.ui import components as ui

CARD_PATH = Path(__file__).resolve().parents[3] / "models" / "model_card.json"


def _headline_speedup() -> tuple[str, str] | None:
    """Pull the measured latency out of the benchmark table, if it exists."""
    bench = Path(__file__).resolve().parents[3] / "outputs" / "tables" / "benchmark.csv"
    if not bench.exists():
        return None
    try:
        import pandas as pd

        df = pd.read_csv(bench)
        row = df[df["quantity"].str.contains("trajectory", case=False, na=False)]
        if row.empty:
            return None
        ms = float(row["median_ms"].iloc[0])
        return f"{ms:.1f} ms", "per 61-point aging trajectory"
    except Exception:
        return None


def render(goto: Callable[[str], None]) -> None:
    st.title("See a solar cell age, in milliseconds")

    st.markdown(
        """
        <div class="tw-banner tw-banner-info">
        Perovskite solar cells are cheap, efficient, and frustratingly short-lived.
        Predicting how fast one degrades normally means a physics simulation that
        takes minutes per condition &mdash; far too slow to explore thousands of
        climates and designs. This app replaces that simulation with a
        <strong>surrogate</strong>: a statistical model trained on those simulations
        that answers in milliseconds, and reports how confident it is.
        </div>
        """,
        unsafe_allow_html=True,
    )

    counts = tier_counts()
    speed = _headline_speedup()

    cols = st.columns(4)
    cols[0].metric(
        "Prediction time",
        speed[0] if speed else "milliseconds",
        help="Measured median latency for a full aging trajectory. Run scripts/benchmark.py to refresh.",
    )
    cols[1].metric(
        "Simulations trained on",
        "36 runs",
        help="A complete 6x6 grid of illumination and temperature, 10 aging times each.",
    )
    cols[2].metric(
        "Validated capabilities",
        counts["Validated"],
        help="Features backed by the trained model and scored on held-out simulations.",
    )
    cols[3].metric(
        "On the roadmap",
        counts["Planned"],
        help="Features the app deliberately refuses to guess at. Turn on 'Show roadmap features' to see them.",
    )

    st.markdown("---")

    left, right = st.columns([1.35, 1])

    with left:
        st.markdown("## Try it")
        st.markdown(
            "Pick a starting point and the app will jump straight to the results. "
            "You can change everything afterwards."
        )

        a, b, c = st.columns(3)
        if a.button("Desert rooftop", width="stretch", help="Bright and hot: 1 sun, 85 C"):
            st.session_state.update(illumination_suns=1.0, temperature_c=85.0, horizon_h=1000.0)
            goto("Results")
        if b.button("Temperate rooftop", width="stretch", help="Milder: 0.6 sun, 47 C"):
            st.session_state.update(illumination_suns=0.6, temperature_c=46.85, horizon_h=1000.0)
            goto("Results")
        if c.button("Dark shelf storage", width="stretch", help="Almost no light, room temperature"):
            st.session_state.update(illumination_suns=0.01, temperature_c=26.85, horizon_h=1000.0)
            goto("Results")

        st.markdown("")
        if st.button("Build my own cell instead", type="primary", width="stretch"):
            goto("Build a cell")

    with right:
        st.markdown("## How to read this app")
        st.markdown(
            "Every number carries a label saying how much to trust it. "
            "You will see these three throughout:"
        )
        for tier in (Tier.VALIDATED, Tier.PREVIEW, Tier.PLANNED):
            st.markdown(
                f"{ui.tier_pill_html(tier)} &nbsp; "
                f'<span class="tw-caption">{TIER_BLURB[tier]}</span>',
                unsafe_allow_html=True,
            )
            st.markdown("")

    st.markdown("---")
    st.markdown("## What is actually happening under the hood")

    c1, c2, c3 = st.columns(3)
    with c1:
        ui.card(
            "1. Physics, run 36 times",
            "A finite-element drift-diffusion model simulated a perovskite cell aging "
            "under every combination of six light levels and six temperatures, "
            "recording its full current-voltage behaviour at ten points in time.",
        )
    with c2:
        ui.card(
            "2. A model that knows its limits",
            "A Gaussian process learns that input-output behaviour. Unlike most machine "
            "learning, it returns an honest error bar with every prediction, and that "
            "error bar grows where the simulations were sparse.",
        )
    with c3:
        ui.card(
            "3. Answers, and a shopping list",
            "Predictions arrive instantly. Because the model tracks its own ignorance, "
            "it can also point at the single most useful simulation to run next.",
        )

    st.markdown("")
    st.markdown(
        f"""
        <div class="tw-banner tw-banner-preview">
        <strong>What this app will not do.</strong>
        It was trained on one device design (a baseline p-i-n stack) aged under
        light and heat only, for up to 1000 hours. It has never seen humidity,
        electrical bias, thermal cycling, or any other cell architecture. Ask it
        about those and it will tell you it does not know rather than guess &mdash;
        {tier_counts()['Planned']} capabilities are deliberately switched off for
        exactly that reason.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Where this is heading", expanded=False):
        st.markdown(
            "The long-term goal is a digital twin of entire solar installations under "
            "real climates: not one cell in a lab oven, but a farm of modules living "
            "through a measured weather year, predicting output and maintenance needs "
            "decades ahead. The scale ladder looks like this:"
        )
        for cap in by_tier(Tier.PLANNED):
            if cap.group == "Digital twin":
                st.markdown(f"- **{cap.label}** ({cap.version}) &mdash; {cap.unlocks}")
        st.markdown(
            "The single-cell rung of that ladder is built and validated today. "
            "The Advanced workspace has the full roadmap."
        )

    ui.glossary_expander(["PCE", "J-V curve", "T80", "Surrogate", "Gaussian process", "Design envelope"])
