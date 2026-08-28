"""Step 1: assemble a cell, then choose the stress it must survive."""

from __future__ import annotations

from typing import Callable

import streamlit as st

from psc_twin import plots
from psc_twin.capabilities import ARCHITECTURE_VALIDATED, ENVELOPE, envelope_excursions
from psc_twin.materials import LAYERS, changed_layers, is_baseline_design, selected_materials
from psc_twin.ui import components as ui


def _material_label(layer, material: str) -> str:
    if material == layer.baseline:
        return f"{material}  -  simulated baseline"
    return f"{material}  -  locked until COMSOL data exists"


def render(goto: Callable[[str], None]) -> None:
    if st.session_state.pop("_reset_materials", False):
        for layer in LAYERS:
            st.session_state[f"material_{layer.key}"] = layer.baseline

    st.title("Build your perovskite solar cell")
    st.markdown(
        '<div class="tw-caption">Start with the physical device. Choose one material '
        "for each layer and watch the stack update. The current surrogate can only "
        "predict the validated baseline; other combinations remain visible as the "
        "next model-building targets.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    ui.banner(
        "The baseline p-i-n stack is backed by a 36-run COMSOL degradation campaign "
        "generated for Favian Tippin's Arizona State University thesis, "
        "<em>Physics-Based Drift-Diffusion Modeling and Machine-Learning Surrogates "
        "for Perovskite Solar Cell Degradation</em>. Alternative architectures and "
        "materials can be explored here, but their aging predictions stay locked "
        "until matching simulations exist.",
        kind="info",
        title="ASU thesis data: build first, simulate second",
    )

    st.markdown("## 1. Choose the cell architecture")
    arch_cols = st.columns(3)
    with arch_cols[0]:
        ui.card(
            "p-i-n baseline  ·  Validated",
            "Light enters through ITO and the hole-transport layer. This is the "
            "only architecture represented in the ASU thesis COMSOL campaign.",
        )
    with arch_cols[1]:
        ui.card(
            "Locked  ·  n-i-p",
            "Reverses the transport-layer order. It needs a new COMSOL geometry "
            "and matched degradation data before predictions can be enabled.",
            planned=True,
        )
    with arch_cols[2]:
        ui.card(
            "Locked  ·  Perovskite/Si tandem",
            "Adds a silicon bottom junction and recombination junction. This is a "
            "future multi-junction model, not an option the current surrogate can guess.",
            planned=True,
        )
    st.session_state["architecture"] = ARCHITECTURE_VALIDATED

    st.markdown("---")
    st.markdown("## 2. Choose a material for every layer")

    controls, visual = st.columns([1.05, 1.15], gap="large")
    with controls:
        for number, layer in enumerate(LAYERS, start=1):
            key = f"material_{layer.key}"
            current = st.session_state.get(key, layer.baseline)
            if current not in layer.options:
                current = layer.baseline
            st.selectbox(
                f"{number}. {layer.label}",
                layer.options,
                index=layer.options.index(current),
                key=key,
                format_func=lambda value, spec=layer: _material_label(spec, value),
                help=f"{layer.role}. Baseline thickness: {layer.thickness}.",
            )

        if st.button("Reset all layers to the simulated baseline", width="stretch"):
            st.session_state["_reset_materials"] = True
            st.rerun()

    materials = selected_materials(st.session_state)
    changed = changed_layers(materials)
    design_is_baseline = is_baseline_design(materials)

    with visual:
        st.pyplot(plots.cell_stack(materials), clear_figure=True)
        if design_is_baseline:
            ui.banner(
                "Every selected layer matches the ASU thesis COMSOL training stack. "
                "Stress controls and aging predictions are available.",
                kind="validated",
                title="Validated material stack",
            )
        else:
            names = ", ".join(layer.label for layer in changed)
            ui.banner(
                f"Changed layers: <strong>{names}</strong>. The stack drawing updates, "
                "but results remain locked because these material identities were not "
                "inputs to the ASU thesis COMSOL campaign.",
                kind="planned",
                title="Custom material design - exploration only",
            )
            st.caption(
                "Locked layers use grey hatching in the diagram. Return every layer "
                "to its baseline material to unlock the validated surrogate."
            )

    st.markdown("---")
    st.markdown("## 3. Choose the aging conditions")
    if not design_is_baseline:
        st.caption("These controls are locked because the selected material stack has no trained model yet.")

    stress, envelope = st.columns([1, 1.1], gap="large")
    with stress:
        lo_i, hi_i = ENVELOPE["illumination_suns"]
        illumination = st.slider(
            "Illumination while aging (suns)",
            min_value=0.0,
            max_value=1.5,
            value=float(st.session_state["illumination_suns"]),
            step=0.01,
            disabled=not design_is_baseline,
            help="1 sun is full midday sunlight; 0.01 is effectively dark storage.",
        )
        ui.envelope_meter("Illumination", illumination, lo_i, hi_i, 0.0, 1.5, " suns")

        lo_t, hi_t = ENVELOPE["temperature_c"]
        temperature = st.slider(
            "Temperature while aging (C)",
            min_value=0.0,
            max_value=160.0,
            value=float(st.session_state["temperature_c"]),
            step=1.0,
            disabled=not design_is_baseline,
            help="The cell temperature during aging, not the surrounding air temperature.",
        )
        ui.envelope_meter("Temperature", temperature, lo_t, hi_t, 0.0, 160.0, " C")

        lo_a, hi_a = ENVELOPE["aging_h"]
        horizon = st.slider(
            "How long to age it (hours)",
            min_value=100.0,
            max_value=3000.0,
            value=float(st.session_state["horizon_h"]),
            step=50.0,
            disabled=not design_is_baseline,
            help="The ASU thesis simulations run to 1000 hours.",
        )
        ui.envelope_meter("Aging time", horizon, lo_a, hi_a, 0.0, 3000.0, " h")

        st.session_state.update(
            illumination_suns=illumination,
            temperature_c=temperature,
            horizon_h=horizon,
            curve_time_h=min(float(st.session_state.get("curve_time_h", horizon)), horizon),
            twin_time_h=min(float(st.session_state.get("twin_time_h", horizon * 0.6)), horizon),
        )

    with envelope:
        st.pyplot(plots.envelope_map(query=(illumination, temperature)), clear_figure=True)
        excursions = envelope_excursions(illumination, temperature, horizon)
        if design_is_baseline and excursions:
            ui.excursion_notice(excursions)
        elif design_is_baseline:
            ui.banner(
                "These conditions sit inside the ASU thesis simulated design envelope.",
                kind="validated",
                title="Inside the validated stress range",
            )
        else:
            ui.banner(
                "Stress prediction will unlock only after the material stack returns "
                "to the validated baseline or new COMSOL training data is added.",
                kind="planned",
                title="Prediction locked",
            )

        if st.button(
            "See how this cell ages",
            type="primary",
            width="stretch",
            disabled=not design_is_baseline,
        ):
            goto("Results")

    ui.glossary_expander(["Design envelope", "PCE", "T80"])
