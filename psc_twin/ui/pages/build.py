"""Step 1: describe the cell and the conditions it has to survive.

The design principle here is that the tested envelope should be visible while
you choose, not announced afterwards. Every slider sits above a meter showing
the range the simulations actually covered, and the marker turns amber the
moment you leave it. Nobody should be able to wander outside the validated
region without noticing.
"""

from __future__ import annotations

from typing import Callable

import streamlit as st

from psc_twin import plots
from psc_twin.capabilities import (
    ARCHITECTURE_VALIDATED,
    ENVELOPE,
    Tier,
    envelope_excursions,
    in_group,
    resolve,
)
from psc_twin.ui import components as ui

_ARCH_ORDER = ("arch_pin", "arch_nip", "arch_tandem")


def render(goto: Callable[[str], None]) -> None:
    st.title("Build a cell")
    st.markdown(
        '<div class="tw-caption">Choose a device and the conditions it has to live '
        "through. The green band on each control is the range the underlying "
        "simulations actually covered.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    show_roadmap = st.session_state.get("show_roadmap", False)

    left, right = st.columns([1, 1.1], gap="large")

    # ---- device -----------------------------------------------------------
    with left:
        st.markdown("## Device")

        caps = [resolve(k) or in_group("Architecture")[0] for k in _ARCH_ORDER]
        available = [c for c in caps if c.tier is Tier.VALIDATED or show_roadmap]
        labels = [c.badge() for c in available]
        current = st.session_state.get("architecture", ARCHITECTURE_VALIDATED)
        try:
            index = [c.label for c in available].index(current)
        except ValueError:
            index = 0

        chosen_label = st.selectbox(
            "Cell architecture",
            labels,
            index=index,
            help=(
                "The layer stack of the solar cell. Only the baseline p-i-n design "
                "has simulation data behind it; the others are shown for context "
                "and produce no numbers."
            ),
        )
        chosen = available[labels.index(chosen_label)]
        st.session_state["architecture"] = chosen.label

        if chosen.tier is Tier.PLANNED:
            ui.planned_card(chosen)
            st.markdown("")
            if st.button("Go back to the validated design", type="primary"):
                st.session_state["architecture"] = ARCHITECTURE_VALIDATED
                st.rerun()
            return

        st.markdown(
            f'<div class="tw-caption">Layer stack: glass / ITO / hole-transport layer / '
            f"perovskite absorber / electron-transport layer / silver. "
            f"{ui.tier_pill_html(Tier.VALIDATED)}</div>",
            unsafe_allow_html=True,
        )

        if not show_roadmap:
            st.caption(
                "Other architectures exist on the roadmap. Turn on "
                "**Show roadmap features** in the sidebar to see them."
            )

        st.markdown("---")
        st.markdown("## Stress conditions")

        lo_i, hi_i = ENVELOPE["illumination_suns"]
        illumination = st.slider(
            "Illumination while aging (suns)",
            min_value=0.0,
            max_value=1.5,
            value=float(st.session_state["illumination_suns"]),
            step=0.01,
            help=(
                "How much light the cell sits under while it degrades. "
                "1 sun is full midday sunlight; 0.01 is effectively a dark shelf."
            ),
        )
        ui.envelope_meter("Illumination", illumination, lo_i, hi_i, 0.0, 1.5, " suns")

        lo_t, hi_t = ENVELOPE["temperature_c"]
        temperature = st.slider(
            "Temperature while aging (C)",
            min_value=0.0,
            max_value=160.0,
            value=float(st.session_state["temperature_c"]),
            step=1.0,
            help=(
                "The cell's own temperature during aging. A rooftop module in "
                "summer commonly reaches 60-80 C."
            ),
        )
        ui.envelope_meter("Temperature", temperature, lo_t, hi_t, 0.0, 160.0, " C")

        lo_a, hi_a = ENVELOPE["aging_h"]
        horizon = st.slider(
            "How long to age it (hours)",
            min_value=100.0,
            max_value=3000.0,
            value=float(st.session_state["horizon_h"]),
            step=50.0,
            help=(
                "The simulations ran to 1000 h. Beyond that the model is "
                "extrapolating and says so."
            ),
        )
        ui.envelope_meter("Aging time", horizon, lo_a, hi_a, 0.0, 3000.0, " h")

        st.session_state.update(
            illumination_suns=illumination,
            temperature_c=temperature,
            horizon_h=horizon,
            curve_time_h=min(float(st.session_state.get("curve_time_h", horizon)), horizon),
            twin_time_h=min(float(st.session_state.get("twin_time_h", horizon * 0.6)), horizon),
        )

    # ---- envelope readout -------------------------------------------------
    with right:
        st.markdown("## Where that sits in the tested design")
        st.pyplot(
            plots.envelope_map(query=(illumination, temperature)),
            clear_figure=True,
        )
        st.markdown(
            '<div class="tw-caption">Each open circle is a COMSOL simulation that was '
            "actually run. The model interpolates confidently between them and grows "
            "less certain the further your cross sits from the cluster.</div>",
            unsafe_allow_html=True,
        )

        excursions = envelope_excursions(illumination, temperature, horizon)
        st.markdown("")
        if excursions:
            ui.excursion_notice(excursions)
        else:
            ui.banner(
                "These conditions sit inside the simulated design envelope, so the "
                "prediction is backed by held-out validation and the error bars are "
                "the model's calibrated posterior.",
                kind="validated",
                title="✅ Inside the validated region",
            )

        st.markdown("")
        if st.button("See how it ages →", type="primary", width="stretch"):
            goto("Results")

    ui.glossary_expander(["Design envelope", "PCE", "T80"])
