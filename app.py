"""PSC Digital Twin - application entry point.

    streamlit run app.py

The navigation is deliberately short. Four steps carry the whole story a first
-time visitor needs -- what this is, describe a cell, see it age, watch it --
and everything a researcher wants sits behind a single Advanced entry. The
sidebar toggle reveals planned capabilities greyed in place, so the default view
stays uncluttered without hiding where the work is going.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from psc_twin import __version__
from psc_twin.capabilities import ARCHITECTURE_VALIDATED, ENVELOPE
from psc_twin.surrogate import predict as predict_mod
from psc_twin.ui import components as ui
from psc_twin.ui import theme
from psc_twin.ui.pages import advanced, build, results, start, twin

st.set_page_config(
    page_title="PSC Digital Twin",
    page_icon="\U0001f31e",
    layout="wide",
    initial_sidebar_state="expanded",
)

theme.inject()


# --------------------------------------------------------------------------
# shared state
# --------------------------------------------------------------------------
DEFAULTS = {
    "illumination_suns": 1.0,
    "temperature_c": 85.0,
    "horizon_h": 1000.0,
    "architecture": ARCHITECTURE_VALIDATED,
    "nav": "Build a cell",
    "curve_time_h": 1000.0,
    "twin_time_h": 600.0,
    "twin_scope": "cell",
}

for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


PAGES = {
    "Build a cell": build.render,
    "Results": results.render,
    "Digital twin": twin.render,
    "Advanced": advanced.render,
    "About the model": start.render,
}

STEPS = ("Build a cell", "Results", "Digital twin")


def goto(page: str) -> None:
    """Navigate programmatically, e.g. from a call-to-action button.

    Streamlit forbids writing to a key that a widget already owns during the
    same run, so this parks the destination and reruns. ``main`` applies it
    before the radio is instantiated, which is the only moment the assignment
    is legal.
    """
    st.session_state["_pending_nav"] = page
    st.rerun()


def main() -> None:
    # Apply a programmatic navigation request from the previous run. This has to
    # happen before st.radio(key="nav") exists.
    if "_pending_nav" in st.session_state:
        st.session_state["nav"] = st.session_state.pop("_pending_nav")
    if st.session_state.get("nav") not in PAGES:
        st.session_state["nav"] = "Build a cell"

    with st.sidebar:
        st.markdown("### \U0001f31e PSC Digital Twin")
        st.caption(
            "A fast stand-in for COMSOL drift-diffusion simulations of "
            "perovskite solar cell degradation."
        )
        st.caption("Data source: Favian Tippin's Arizona State University thesis COMSOL campaign.")
        st.markdown("---")

        nav = st.radio(
            "Go to",
            list(PAGES),
            index=list(PAGES).index(st.session_state["nav"]),
            key="nav",
            label_visibility="collapsed",
        )

        st.link_button(
            "Share alpha feedback",
            "https://github.com/flabio121/psc-digital-twin/issues/new?"
            "title=Alpha%20feedback&labels=feedback",
            width="stretch",
            help="Report something confusing, broken, or worth adding in the public GitHub repository.",
        )

        st.markdown("---")
        show_roadmap = ui.roadmap_toggle()
        st.session_state["show_roadmap"] = show_roadmap

        if not predict_mod.models_available():
            st.error(
                "Trained models are missing.\n\n"
                "Run `python scripts/train_models.py` from the project root, "
                "then reload this page."
            )

        st.markdown("---")
        st.caption(f"Version {__version__}")
        st.caption(
            f"Validated envelope: {ENVELOPE['illumination_suns'][0]}-"
            f"{ENVELOPE['illumination_suns'][1]} suns, "
            f"{ENVELOPE['temperature_c'][0]:.0f}-{ENVELOPE['temperature_c'][1]:.0f} C, "
            f"0-{ENVELOPE['aging_h'][1]:.0f} h."
        )

    if nav in STEPS:
        idx = STEPS.index(nav)
        ui.step_rail(STEPS, active=idx, done=range(idx))

    PAGES[nav](goto=goto)


if __name__ == "__main__":
    main()
