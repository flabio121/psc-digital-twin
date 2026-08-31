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
        return f"{material}  -  validated physical baseline"
    return f"{material}  -  locked until COMSOL data exists"


def _thickness_parts(thickness: str) -> tuple[float | None, str]:
    parts = thickness.split(maxsplit=1)
    if len(parts) != 2:
        return None, thickness
    try:
        return float(parts[0]), parts[1]
    except ValueError:
        return None, thickness


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
    st.caption(
        "The physical baseline separates NiOx from its MeO-2PACz SAM and C60 from "
        "its BCP buffer layer. The ASU thesis COMSOL export grouped each pair into "
        "one computational domain; that bookkeeping does not make them one material. "
        "Additional barriers, passivation layers, and bilayers remain optional and locked."
    )

    controls, visual = st.columns([1.05, 1.15], gap="large")
    with controls:
        for number, layer in enumerate(LAYERS, start=1):
            key = f"material_{layer.key}"
            current = st.session_state.get(key, layer.baseline)
            if current not in layer.options:
                current = layer.baseline
                st.session_state[key] = current
            st.selectbox(
                f"{number}. {layer.label}",
                layer.options,
                index=layer.options.index(current),
                key=key,
                format_func=lambda value, spec=layer: _material_label(spec, value),
                help=(
                    f"{layer.role}. Baseline thickness: {layer.thickness}. "
                    f"{layer.thickness_note}"
                ).strip(),
            )

        if st.button("Reset all layers to the validated physical baseline", width="stretch"):
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
    st.markdown("## 3. Set each layer thickness")
    ui.banner(
        "Thickness values are shown at the ASU thesis COMSOL baseline. The controls "
        "are visible but locked because the current simulation campaign did not "
        "sweep geometry, so changing them cannot produce a defensible prediction. "
        "MeO-2PACz and BCP are shown separately as physical layers, but their "
        "individual thicknesses were not resolved from the combined COMSOL domains.",
        kind="planned",
        title="Planned · thickness-aware model",
    )
    thickness_cols = st.columns(3)
    for index, layer in enumerate(LAYERS):
        baseline_value, unit = _thickness_parts(layer.thickness)
        with thickness_cols[index % 3]:
            help_text = (
                f"ASU thesis baseline: {layer.thickness}. {layer.thickness_note} "
                "Unlocks after a thickness-resolved COMSOL campaign is validated."
            ).strip()
            if baseline_value is None:
                st.text_input(
                    layer.label,
                    value=layer.thickness,
                    disabled=True,
                    key=f"thickness_{layer.key}",
                    help=help_text,
                )
            else:
                st.number_input(
                    f"{layer.label} ({unit})",
                    value=baseline_value,
                    disabled=True,
                    key=f"thickness_{layer.key}",
                    help=help_text,
                )

    st.markdown("---")
    st.markdown("## 4. Choose the aging conditions")
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

        st.markdown("### Live rooftop conditions")
        ui.banner(
            "A future connector will use a rooftop location and current weather to "
            "estimate plane-of-array irradiance, cell temperature, humidity, and "
            "wind cooling. It is disabled until the weather-to-cell model is "
            "validated against field telemetry.",
            kind="planned",
            title="Planned · real-time weather",
        )
        st.text_input(
            "Rooftop location",
            value="Connect a weather source to choose a site",
            disabled=True,
            key="live_weather_location",
        )
        weather_cols = st.columns(2)
        with weather_cols[0]:
            st.text_input("Solar irradiance", value="Not connected", disabled=True)
            st.text_input("Cell temperature", value="Not connected", disabled=True)
        with weather_cols[1]:
            st.text_input("Relative humidity", value="Not connected", disabled=True)
            st.text_input("Wind cooling", value="Not connected", disabled=True)
        st.toggle(
            "Use live rooftop conditions",
            value=False,
            disabled=True,
            help="Planned capability; no live data is requested or used in this alpha.",
        )

        if st.button(
            "See how this cell ages",
            type="primary",
            width="stretch",
            disabled=not design_is_baseline,
        ):
            goto("Results")

    ui.glossary_expander(["Design envelope", "PCE", "T80"])
