"""Step 3: the digital twin, and the ladder of scales above it.

The cell scope is real: the geometry darkens, speckles and bleaches according to
the retention the surrogate predicted for the time on the slider. Nothing in the
canvas invents its own decay.

The module and farm scopes are the vision, rendered honestly as unbuilt. They
carry a watermark, no numbers, and an explanation of what would unlock them.
"""

from __future__ import annotations

from typing import Callable

import streamlit as st
import streamlit.components.v1 as components

from psc_twin import twin3d
from psc_twin.capabilities import Tier, get
from psc_twin.surrogate import predict as predict_mod
from psc_twin.ui import components as ui

SCOPES = {
    "cell": ("Single cell", "twin_cell"),
    "module": ("Module and string", "twin_module"),
    "farm": ("Solar farm", "twin_farm"),
}


def render(goto: Callable[[str], None]) -> None:
    st.title("Digital twin")

    if not predict_mod.models_available():
        ui.banner(
            "The trained model files are not present. Run "
            "<code>python scripts/train_models.py</code> first.",
            kind="planned",
            title="Models not built yet",
        )
        return

    show_roadmap = st.session_state.get("show_roadmap", False)

    illumination = float(st.session_state["illumination_suns"])
    temperature = float(st.session_state["temperature_c"])
    horizon = float(st.session_state["horizon_h"])
    architecture = st.session_state["architecture"]

    keys = [k for k in SCOPES if k == "cell" or show_roadmap]
    labels = []
    for k in keys:
        name, cap_key = SCOPES[k]
        cap = get(cap_key)
        labels.append(name if cap.tier is Tier.VALIDATED else f"{name}  {cap.icon} {cap.version}")

    top = st.columns([1, 1, 2])
    chosen_label = top[0].selectbox("Scale", labels, index=0)
    scope = keys[labels.index(chosen_label)]
    st.session_state["twin_scope"] = scope

    cap = get(SCOPES[scope][1])

    time_h = top[1].slider(
        "Time (hours)",
        min_value=0.0,
        max_value=float(horizon),
        value=float(min(st.session_state.get("twin_time_h", horizon * 0.6), horizon)),
        step=max(horizon / 100.0, 1.0),
        disabled=cap.tier is Tier.PLANNED,
        help="Drag to watch the same cell at different points in its life.",
    )
    st.session_state["twin_time_h"] = time_h

    with top[2]:
        st.markdown("")
        st.markdown(
            f'<div class="tw-caption">{ui.tier_pill_html(cap.tier)} &nbsp; {cap.backing}</div>',
            unsafe_allow_html=True,
        )

    if cap.tier is Tier.PLANNED:
        st.markdown("")
        config = twin3d.build_twin_config(
            architecture=architecture,
            illumination_suns=illumination,
            temperature_c=temperature,
            aging_h=time_h,
            scope=scope,
            horizon_h=horizon,
        )
        components.html(twin3d.build_twin_html(config), height=560, scrolling=False)
        st.markdown("")
        ui.planned_card(cap)
        if not show_roadmap:
            st.caption("Turn off **Show roadmap features** in the sidebar to hide unbuilt scales.")
        return

    try:
        pred = predict_mod.predict(
            illumination_suns=illumination,
            temperature_c=temperature,
            horizon_h=horizon,
            architecture=architecture,
            curve_at_h=time_h,
        )
    except predict_mod.PlannedCapabilityError as exc:
        ui.planned_card(exc.capability)
        return

    row = pred.at_time(time_h)

    if pred.excursions:
        ui.excursion_notice(pred.excursions)

    config = twin3d.build_twin_config(
        architecture=architecture,
        illumination_suns=illumination,
        temperature_c=temperature,
        aging_h=time_h,
        pce_pct=float(row["PCE_pct"]),
        retention_pct=float(row["PCE_retention_pct"]),
        voc_v=float(row.get("Voc_V", float("nan"))),
        jsc_macm2=float(row.get("Jsc_mAcm2", float("nan"))),
        ff=float(row.get("FF", float("nan"))),
        pce_sd=float(row.get("PCE_pct_std", 0.0) or 0.0),
        retention_sd=float(row.get("PCE_retention_pct_std", 0.0) or 0.0),
        tier=pred.tier,
        scope=scope,
        horizon_h=horizon,
    )
    components.html(twin3d.build_twin_html(config), height=620, scrolling=False)

    st.markdown(
        '<div class="tw-caption">Drag to orbit, scroll to zoom, arrow keys also work. '
        "Everything you see &mdash; the defect speckle, the darkened contacts, the "
        "bleaching of the absorber &mdash; is driven by the predicted retention at the "
        "time on the slider. There is no decay animation running independently of the "
        "model.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    cols = st.columns(4)
    cols[0].metric("Time", f"{time_h:,.0f} h")
    cols[1].metric("Efficiency", f"{row['PCE_pct']:.2f} %")
    cols[2].metric("Retained", f"{row['PCE_retention_pct']:.1f} %")
    cols[3].metric("Confidence", cap.tier.value if pred.tier is Tier.VALIDATED else pred.tier.value)

    if not show_roadmap:
        st.markdown("")
        st.caption(
            "This is the single-cell rung of a longer ladder. Turn on "
            "**Show roadmap features** in the sidebar to see the module and farm scales."
        )

    ui.glossary_expander(["Digital twin", "PCE", "Design envelope"])
