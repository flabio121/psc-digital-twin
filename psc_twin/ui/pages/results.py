"""Step 2: the payoff. What happens to this cell, and how sure are we.

Three tabs, ordered by how much the reader already knows: the headline, then
the individual electrical parameters, then the raw current-voltage behaviour
everything else is derived from.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import streamlit as st

from psc_twin import plots
from psc_twin.capabilities import get
from psc_twin.data import TARGET_LABELS
from psc_twin.materials import is_baseline_design, selected_materials
from psc_twin.surrogate import predict as predict_mod
from psc_twin.ui import components as ui


@st.cache_data(show_spinner=False, ttl=3600)
def _predict(illumination: float, temperature: float, horizon: float, architecture: str, curve_h: float):
    """Cached so that moving an unrelated widget does not refit anything."""
    return predict_mod.predict(
        illumination_suns=illumination,
        temperature_c=temperature,
        horizon_h=horizon,
        architecture=architecture,
        curve_at_h=curve_h,
    )


def render(goto: Callable[[str], None]) -> None:
    st.title("How this cell ages")

    if not is_baseline_design(selected_materials(st.session_state)):
        ui.planned_card(get("materials_custom"))
        if st.button("Return to the cell builder", type="primary"):
            goto("Build a cell")
        return

    if not predict_mod.models_available():
        ui.banner(
            "The trained model files are not present. From the project root run "
            "<code>python scripts/train_models.py</code>, then reload.",
            kind="planned",
            title="Models not built yet",
        )
        return

    illumination = float(st.session_state["illumination_suns"])
    temperature = float(st.session_state["temperature_c"])
    horizon = float(st.session_state["horizon_h"])
    architecture = st.session_state["architecture"]
    curve_h = float(st.session_state.get("curve_time_h", horizon))

    try:
        pred = _predict(illumination, temperature, horizon, architecture, curve_h)
    except predict_mod.PlannedCapabilityError as exc:
        ui.planned_card(exc.capability)
        if st.button("Choose a supported design"):
            goto("Build a cell")
        return
    except predict_mod.ModelsMissingError as exc:
        ui.banner(str(exc), kind="planned", title="Models not built yet")
        return

    st.markdown(
        f'<div class="tw-caption">{architecture} &middot; {illumination:g} suns &middot; '
        f"{temperature:.0f} C &middot; aged to {horizon:,.0f} h &nbsp; "
        f"{ui.tier_pill_html(pred.tier)}</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    if pred.excursions:
        ui.excursion_notice(pred.excursions)

    tab_overview, tab_params, tab_jv = st.tabs(
        ["Overview", "Electrical parameters", "Current-voltage curves"]
    )

    # ---- overview ---------------------------------------------------------
    with tab_overview:
        first = pred.trajectories.iloc[0]
        final = pred.final

        cols = st.columns(4)
        cols[0].metric(
            "Starting efficiency",
            f"{first['PCE_pct']:.2f} %",
            help=ui.help_for("PCE"),
        )
        cols[1].metric(
            f"After {horizon:,.0f} h",
            f"{final['PCE_pct']:.2f} %",
            delta=f"{final['PCE_pct'] - first['PCE_pct']:+.2f} pts",
            delta_color="inverse",
        )
        cols[2].metric(
            "Efficiency retained",
            f"{final['PCE_retention_pct']:.1f} %",
            help="How much of its original efficiency the cell still has.",
        )
        cols[3].metric(
            "T80",
            predict_mod.format_lifetime(pred.t80),
            help=ui.help_for("T80"),
        )

        st.markdown("")
        left, right = st.columns([1.6, 1], gap="large")

        with left:
            st.pyplot(
                plots.retention_with_lifetime(
                    pred.trajectories,
                    t80_h=pred.t80.value_h,
                    t90_h=pred.t90.value_h,
                ),
                clear_figure=True,
            )
            st.markdown(
                '<div class="tw-caption">The shaded band is the 95% predictive interval '
                "from the Gaussian process. Where it is narrow, several nearby "
                "simulations pin the answer down; where it widens, the model is "
                "telling you it is working from less evidence.</div>",
                unsafe_allow_html=True,
            )

        with right:
            st.markdown("### In plain words")
            st.markdown(pred.t80.note())
            st.markdown("")
            st.markdown(pred.t90.note())

            st.markdown("---")
            st.markdown("### Summary")
            ui.dataframe(pred.summary_frame())

            st.caption(f"Predicted in {pred.latency_ms:.1f} ms using the {pred.engine}.")

        st.markdown("---")
        cta_l, cta_r = st.columns(2)
        if cta_l.button("Watch it in 3D →", width="stretch"):
            goto("Digital twin")
        if cta_r.button("Change the conditions", width="stretch"):
            goto("Build a cell")

    # ---- parameters -------------------------------------------------------
    with tab_params:
        st.markdown(
            "Efficiency is a product of three measurable quantities. Watching which "
            "one falls tells you *how* the cell is failing, not just that it is."
        )
        st.markdown("")

        targets = ["PCE_pct", "Voc_V", "Jsc_mAcm2", "FF"]
        st.pyplot(plots.multi_metric(pred.trajectories, targets), clear_figure=True)

        st.markdown("")
        c1, c2, c3 = st.columns(3)
        with c1:
            ui.card(
                "Voltage (Voc)",
                "Falls when defects give charges somewhere to recombine before they "
                "escape. A steep voltage drop points at trap formation.",
            )
        with c2:
            ui.card(
                "Current (Jsc)",
                "Falls when less light is absorbed or fewer charges are collected. "
                "Usually the most stable of the three.",
            )
        with c3:
            ui.card(
                "Fill factor (FF)",
                "Falls when resistance rises or mobile ions screen the internal "
                "field. Often the first thing to move.",
            )

        st.markdown("---")
        chosen = st.selectbox(
            "Look at one parameter closely",
            targets,
            format_func=lambda t: TARGET_LABELS.get(t, t),
        )
        st.pyplot(plots.trajectory(pred.trajectories, chosen), clear_figure=True)

        with st.expander("Show the numbers"):
            ui.dataframe(pred.trajectories.round(4))

    # ---- J-V --------------------------------------------------------------
    with tab_jv:
        cap = get("out_jv")
        st.markdown(
            "Every performance number on this page is read off a current-voltage "
            f"curve. {ui.term('J-V curve', 'These curves')} are predicted directly by the "
            "model, not reconstructed from the numbers above.",
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="tw-caption">{ui.tier_pill_html(cap.tier)} &nbsp; {cap.backing}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("")

        curve_time = st.slider(
            "Aging time to show (hours)",
            min_value=0.0,
            max_value=float(horizon),
            value=float(min(curve_h, horizon)),
            step=max(horizon / 100.0, 1.0),
        )
        st.session_state["curve_time_h"] = curve_time

        pred_at = _predict(illumination, temperature, horizon, architecture, curve_time)
        curve = pred_at.curve
        ref = pred_at.curve_reference

        left, right = st.columns([1.6, 1], gap="large")
        with left:
            st.pyplot(
                plots.jv_curve(
                    curve.voltage,
                    curve.current_density,
                    std=curve.current_std,
                    reference=(ref.voltage, ref.current_density) if ref else None,
                    mpp=(curve.metrics.vmp_v, curve.metrics.jmp_macm2),
                    title=f"Predicted J-V at {curve_time:,.0f} h",
                ),
                clear_figure=True,
            )
        with right:
            m = curve.metrics
            st.markdown("### Read off this curve")
            st.metric("Short-circuit current (Jsc)", f"{m.jsc_macm2:.2f} mA/cm²", help=ui.help_for("Jsc"))
            st.metric("Open-circuit voltage (Voc)", f"{m.voc_v:.3f} V", help=ui.help_for("Voc"))
            st.metric("Fill factor (FF)", f"{m.ff:.3f}", help=ui.help_for("FF"))
            st.metric("Efficiency (PCE)", f"{m.pce_pct:.2f} %", help=ui.help_for("PCE"))
            if not m.valid:
                ui.banner(
                    f"This predicted curve failed a physical sanity check: {m.note}. "
                    "The numbers above are not trustworthy.",
                    kind="preview",
                    title="Curve extraction warning",
                )

        st.markdown("---")
        st.markdown("### The whole aging sequence")
        times = list(np.linspace(0.0, horizon, 8))
        family = predict_mod.predict_curve_family(illumination, temperature, times)
        st.pyplot(
            plots.jv_family(
                family[0].voltage,
                [(t, c.current_density) for t, c in zip(times, family)],
                title="Predicted J-V curves across the aging window",
            ),
            clear_figure=True,
        )
        st.markdown(
            '<div class="tw-caption">Each line is one predicted curve. The curve '
            "collapsing toward the origin is the cell losing power; which axis it "
            "collapses along tells you the mechanism.</div>",
            unsafe_allow_html=True,
        )

    ui.glossary_expander(["PCE", "Voc", "Jsc", "FF", "J-V curve", "T80", "RUL"])
