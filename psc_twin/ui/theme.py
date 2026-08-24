"""Visual system for the app.

One deliberate light theme rather than a light/dark pair. This is a scientific
instrument that people screenshot into slides and papers, so a single stable
appearance is worth more than theme flexibility.

The palette encodes the product's central idea. Three tier colours -- green,
amber, slate -- carry the meaning of "validated", "preview" and "planned"
everywhere they appear, so a reader learns the vocabulary once and then reads
every page fluently.
"""

from __future__ import annotations

import matplotlib as mpl
import streamlit as st

from psc_twin.capabilities import Tier

# --- palette --------------------------------------------------------------
BG = "#F7F9FC"
SURFACE = "#FFFFFF"
SURFACE_SUNK = "#F1F5F9"
BORDER = "#E3E8EF"
BORDER_STRONG = "#CBD5E1"
TEXT = "#0F172A"
TEXT_MUTED = "#5B6B82"
TEXT_FAINT = "#94A3B8"
PRIMARY = "#2563EB"
PRIMARY_SOFT = "#EFF6FF"

TIER_COLOR = {
    Tier.VALIDATED: "#059669",
    Tier.PREVIEW: "#D97706",
    Tier.PLANNED: "#64748B",
}

TIER_BG = {
    Tier.VALIDATED: "#ECFDF5",
    Tier.PREVIEW: "#FFFBEB",
    Tier.PLANNED: "#F1F5F9",
}

TIER_BORDER = {
    Tier.VALIDATED: "#A7F3D0",
    Tier.PREVIEW: "#FDE68A",
    Tier.PLANNED: "#CBD5E1",
}

# Series colours for plots. Chosen to stay distinguishable in greyscale print,
# which matters because these figures are meant to survive a journal PDF.
SERIES = ("#2563EB", "#DC2626", "#059669", "#7C3AED", "#D97706", "#0891B2")

BAND_ALPHA = 0.18


def apply_matplotlib_style() -> None:
    """Match figures to the page so exported plots look native in the paper."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": BORDER_STRONG,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "axes.titlesize": 11,
            "axes.titleweight": "600",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": BORDER,
            "grid.linewidth": 0.8,
            "grid.alpha": 0.9,
            "xtick.color": TEXT_MUTED,
            "ytick.color": TEXT_MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "figure.dpi": 120,
            "font.size": 10,
        }
    )


_CSS = f"""
<style>
:root {{ color-scheme: light; }}

.stApp, [data-testid="stAppViewContainer"] {{
    background: {BG};
    color: {TEXT};
}}
[data-testid="stHeader"] {{
    background: rgba(247, 249, 252, 0.94);
    border-bottom: 1px solid {BORDER};
}}
[data-testid="stSidebar"] {{
    background: {SURFACE};
    border-right: 1px solid {BORDER};
}}
[data-testid="stSidebar"] * {{ color: {TEXT}; }}
.block-container {{ padding-top: 2.1rem; padding-bottom: 3rem; max-width: 1400px; }}

h1 {{ font-size: 1.9rem; font-weight: 700; letter-spacing: -0.02em; color: {TEXT}; }}
h2 {{ font-size: 1.28rem; font-weight: 650; letter-spacing: -0.01em; color: {TEXT}; margin-top: 0.4rem; }}
h3 {{ font-size: 1.04rem; font-weight: 650; color: {TEXT}; }}

/* --- tier pills --------------------------------------------------------- */
.tier-pill {{
    display: inline-flex; align-items: center; gap: 0.32rem;
    border-radius: 999px; padding: 0.16rem 0.62rem;
    font-size: 0.74rem; font-weight: 650; letter-spacing: 0.01em;
    border: 1px solid transparent; white-space: nowrap;
}}
.tier-validated {{ background: {TIER_BG[Tier.VALIDATED]}; color: {TIER_COLOR[Tier.VALIDATED]}; border-color: {TIER_BORDER[Tier.VALIDATED]}; }}
.tier-preview   {{ background: {TIER_BG[Tier.PREVIEW]};   color: {TIER_COLOR[Tier.PREVIEW]};   border-color: {TIER_BORDER[Tier.PREVIEW]}; }}
.tier-planned   {{ background: {TIER_BG[Tier.PLANNED]};   color: {TIER_COLOR[Tier.PLANNED]};   border-color: {TIER_BORDER[Tier.PLANNED]}; }}

/* --- banners ------------------------------------------------------------ */
.tw-banner {{
    border: 1px solid {BORDER}; border-left-width: 4px;
    background: {SURFACE}; border-radius: 10px;
    padding: 0.85rem 1.05rem; margin: 0.55rem 0 1.05rem 0;
    line-height: 1.5; font-size: 0.9rem; color: {TEXT};
}}
.tw-banner strong {{ font-weight: 680; }}
.tw-banner-validated {{ border-left-color: {TIER_COLOR[Tier.VALIDATED]}; }}
.tw-banner-preview   {{ border-left-color: {TIER_COLOR[Tier.PREVIEW]}; background: {TIER_BG[Tier.PREVIEW]}; }}
.tw-banner-planned   {{ border-left-color: {TIER_COLOR[Tier.PLANNED]}; background: {TIER_BG[Tier.PLANNED]}; }}
.tw-banner-info      {{ border-left-color: {PRIMARY}; background: {PRIMARY_SOFT}; }}

/* --- cards -------------------------------------------------------------- */
.tw-card {{
    border: 1px solid {BORDER}; background: {SURFACE};
    border-radius: 12px; padding: 1.1rem 1.25rem; height: 100%;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}}
.tw-card h4 {{ margin: 0 0 0.4rem 0; font-size: 0.98rem; font-weight: 660; color: {TEXT}; }}
.tw-card p  {{ margin: 0; font-size: 0.87rem; line-height: 1.55; color: {TEXT_MUTED}; }}
.tw-card-planned {{ background: {SURFACE_SUNK}; border-style: dashed; }}
.tw-card-planned h4 {{ color: {TEXT_MUTED}; }}
.tw-card-planned p  {{ color: {TEXT_FAINT}; }}

/* --- the de-emphasis wrapper for preview-tier results ------------------- */
.tw-preview-zone {{
    border: 1px dashed {TIER_BORDER[Tier.PREVIEW]};
    background: linear-gradient(0deg, rgba(255,251,235,0.55), rgba(255,251,235,0.55));
    border-radius: 12px; padding: 0.35rem 0.75rem 0.1rem 0.75rem; margin-bottom: 0.9rem;
}}

/* --- step rail ---------------------------------------------------------- */
.tw-steps {{ display: flex; gap: 0.45rem; flex-wrap: wrap; margin: 0.2rem 0 1.15rem 0; }}
.tw-step {{
    display: inline-flex; align-items: center; gap: 0.4rem;
    border: 1px solid {BORDER}; background: {SURFACE};
    border-radius: 999px; padding: 0.28rem 0.75rem;
    font-size: 0.8rem; color: {TEXT_MUTED};
}}
.tw-step-num {{
    display: inline-flex; align-items: center; justify-content: center;
    width: 1.15rem; height: 1.15rem; border-radius: 999px;
    background: {SURFACE_SUNK}; color: {TEXT_MUTED};
    font-size: 0.68rem; font-weight: 700;
}}
.tw-step-active {{ border-color: {PRIMARY}; background: {PRIMARY_SOFT}; color: {PRIMARY}; font-weight: 600; }}
.tw-step-active .tw-step-num {{ background: {PRIMARY}; color: #FFFFFF; }}
.tw-step-done {{ color: {TIER_COLOR[Tier.VALIDATED]}; border-color: {TIER_BORDER[Tier.VALIDATED]}; }}
.tw-step-done .tw-step-num {{ background: {TIER_COLOR[Tier.VALIDATED]}; color: #FFFFFF; }}

/* --- key/value readouts ------------------------------------------------- */
div[data-testid="stMetric"] {{
    background: {SURFACE}; border: 1px solid {BORDER};
    border-radius: 10px; padding: 0.8rem 0.95rem;
}}
div[data-testid="stMetricLabel"] p {{ color: {TEXT_MUTED}; font-size: 0.8rem; }}
div[data-testid="stMetricValue"] {{ color: {TEXT}; font-size: 1.5rem; font-weight: 660; }}

[data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 10px; }}

/* --- envelope meter ----------------------------------------------------- */
.tw-meter {{ margin: 0.35rem 0 0.9rem 0; }}
.tw-meter-track {{
    position: relative; height: 0.5rem; border-radius: 999px;
    background: {SURFACE_SUNK}; border: 1px solid {BORDER};
}}
.tw-meter-band {{
    position: absolute; top: 0; bottom: 0;
    background: {TIER_BG[Tier.VALIDATED]}; border-left: 2px solid {TIER_COLOR[Tier.VALIDATED]};
    border-right: 2px solid {TIER_COLOR[Tier.VALIDATED]};
}}
.tw-meter-marker {{
    position: absolute; top: -0.28rem; width: 0.62rem; height: 1.06rem;
    border-radius: 3px; background: {PRIMARY}; border: 2px solid {SURFACE};
    box-shadow: 0 0 0 1px {PRIMARY}; transform: translateX(-50%);
}}
.tw-meter-marker-out {{ background: {TIER_COLOR[Tier.PREVIEW]}; box-shadow: 0 0 0 1px {TIER_COLOR[Tier.PREVIEW]}; }}
.tw-meter-labels {{
    display: flex; justify-content: space-between;
    font-size: 0.72rem; color: {TEXT_FAINT}; margin-top: 0.28rem;
}}

/* --- glossary term ------------------------------------------------------ */
.tw-term {{
    border-bottom: 1px dotted {TEXT_FAINT}; cursor: help; font-weight: 600;
}}

.tw-caption {{ font-size: 0.82rem; color: {TEXT_MUTED}; line-height: 1.5; }}
.tw-eyebrow {{
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; color: {TEXT_FAINT}; margin-bottom: 0.15rem;
}}
hr {{ border: none; border-top: 1px solid {BORDER}; margin: 1.4rem 0; }}
</style>
"""


def inject() -> None:
    """Install the stylesheet and the matplotlib style. Call once per run."""
    st.markdown(_CSS, unsafe_allow_html=True)
    apply_matplotlib_style()
