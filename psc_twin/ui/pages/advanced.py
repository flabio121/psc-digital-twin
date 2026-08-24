"""Everything research-facing, behind one menu entry.

Seven workspaces that would each have been a sidebar item in a naive design sit
here as tabs. The main navigation stays four items long and a first-time visitor
never has to decide whether "Data contract" is something they need.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import streamlit as st

from psc_twin import activelearn, climate, heuristic, plots
from psc_twin.capabilities import (
    GROUPS,
    TIER_ICON,
    Tier,
    get,
    in_group,
    registry_table,
    tier_counts,
)
from psc_twin.data import TARGETS, TARGET_LABELS, doe_summary, load_doe, load_seed
from psc_twin.surrogate import predict as predict_mod
from psc_twin.ui import components as ui

ROOT = Path(__file__).resolve().parents[3]
TABLES = ROOT / "outputs" / "tables"
CARD = ROOT / "models" / "model_card.json"


@st.cache_data(show_spinner=False)
def _table(name: str) -> pd.DataFrame | None:
    path = TABLES / name
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def _card() -> dict | None:
    return json.loads(CARD.read_text(encoding="utf-8")) if CARD.exists() else None


@st.cache_data(show_spinner=False)
def _doe():
    bundle = load_doe()
    return bundle.metrics, doe_summary(bundle)


def render(goto: Callable[[str], None]) -> None:
    st.title("Advanced")
    st.markdown(
        '<div class="tw-caption">The model itself, the data behind it, its limits, '
        "and where the project is going.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    tabs = st.tabs(
        [
            "Model & validation",
            "Simulation data",
            "Uncertainty & next runs",
            "Climate deployment",
            "Mechanisms",
            "Roadmap",
            "Export",
        ]
    )

    with tabs[0]:
        _model_validation()
    with tabs[1]:
        _simulation_data()
    with tabs[2]:
        _uncertainty()
    with tabs[3]:
        _climate()
    with tabs[4]:
        _mechanisms()
    with tabs[5]:
        _roadmap()
    with tabs[6]:
        _export()


# --------------------------------------------------------------------------
def _model_validation() -> None:
    st.markdown("## How well does it actually work?")
    card = _card()
    cv = _table("cv_scalar_metrics.csv")

    if cv is None:
        ui.banner(
            "No validation tables yet. Run <code>python scripts/train_models.py</code> "
            "to fit the models and generate them.",
            kind="planned",
            title="Not trained yet",
        )
        return

    ui.banner(
        "Every number below comes from <strong>leave-one-run-out</strong> "
        "cross-validation: the model is retrained 36 times, each time with one "
        "entire simulation condition held out, and scored only on that unseen "
        "condition. Because ten rows share each design point, an ordinary random "
        "split would leak the answer and report a far better number than the model "
        "deserves.",
        kind="info",
        title="Validation protocol",
    )

    st.markdown("### Held-out accuracy")
    ui.dataframe(cv.round(5))

    cov = float(cv["coverage_95"].mean()) if "coverage_95" in cv.columns else float("nan")
    cols = st.columns(3)
    cols[0].metric(
        "Mean R² (held out)",
        f"{cv['R2'].mean():.4f}" if "R2" in cv.columns else "n/a",
        help="1.0 is perfect. Computed on simulation conditions the model never saw.",
    )
    cols[1].metric(
        "Mean 95% coverage",
        f"{cov:.3f}" if np.isfinite(cov) else "n/a",
        help=(
            "Share of held-out points falling inside the model's own 95% error bar. "
            "Should be near 0.95. Below that, the error bars are too confident."
        ),
    )
    # n_held_out is per target, so the sum counts predictions, not simulations.
    # Every one of the 360 observations is predicted once per target.
    cols[2].metric(
        "Held-out predictions",
        f"{int(cv['n_held_out'].sum()):,}" if "n_held_out" in cv.columns else "n/a",
        help=(
            "36 design points x 10 aging times = 360 observations, each predicted "
            "once per target while its own design point was excluded from training."
        ),
    )

    if np.isfinite(cov):
        if cov >= 0.90:
            ui.banner(
                f"Coverage of {cov:.3f} against a nominal 0.95 means the shaded bands "
                "shown throughout the app are broadly trustworthy.",
                kind="validated",
                title="Uncertainty is calibrated",
            )
        else:
            ui.banner(
                f"Coverage of {cov:.3f} is below the nominal 0.95, so the error bars "
                "are over-confident: the true value falls outside them more often "
                "than advertised. Treat the bands as a lower bound on uncertainty.",
                kind="preview",
                title="Uncertainty is over-confident",
            )

    preds = _table("cv_scalar_predictions.csv")
    if preds is not None:
        st.markdown("---")
        st.markdown("### Predicted against simulated")
        target = st.selectbox(
            "Target", list(TARGETS), format_func=lambda t: TARGET_LABELS.get(t, t)
        )
        if f"{target}_true" in preds.columns:
            row = cv[cv["target"] == target]
            metrics = {}
            if not row.empty:
                metrics = {
                    "R2": float(row["R2"].iloc[0]),
                    "MAE": float(row["MAE"].iloc[0]),
                    "coverage": float(row["coverage_95"].iloc[0]),
                }
            c1, c2 = st.columns(2)
            with c1:
                st.pyplot(
                    plots.parity(
                        preds[f"{target}_true"],
                        preds[f"{target}_pred"],
                        std=preds.get(f"{target}_std"),
                        target=target,
                        metrics=metrics,
                    ),
                    clear_figure=True,
                )
            with c2:
                resid = pd.DataFrame(
                    {
                        "aging_light_suns": preds["aging_light_suns"],
                        "aging_temperature_C": preds["aging_temperature_C"],
                        "residual": preds[f"{target}_pred"] - preds[f"{target}_true"],
                    }
                ).groupby(["aging_light_suns", "aging_temperature_C"], as_index=False)["residual"].mean()
                st.pyplot(plots.residual_map(resid), clear_figure=True)
                st.markdown(
                    '<div class="tw-caption">Structure here &mdash; a whole corner biased '
                    "one way &mdash; would mean the kernel is mis-specified rather than "
                    "merely uncertain.</div>",
                    unsafe_allow_html=True,
                )

    lc = _table("learning_curve.csv")
    if lc is not None:
        st.markdown("---")
        st.markdown("### Is 36 simulations enough?")
        c1, c2 = st.columns([1.4, 1])
        with c1:
            st.pyplot(plots.learning_curve(lc), clear_figure=True)
        with c2:
            verdict = (card or {}).get("validation", {}).get("learning_curve_verdict")
            if verdict:
                st.markdown(verdict)
            st.markdown(
                "The curve is built by retraining on 6, 12, 18, 24 and 30 randomly "
                "chosen *whole runs* and scoring on the rest. If it has flattened, "
                "more simulations at the same spacing would add little; if it is "
                "still falling, the campaign is under-sampled."
            )

    jv_sum = _table("cv_jv_summary.csv")
    if jv_sum is not None:
        st.markdown("---")
        st.markdown("### The J-V curve model")
        c1, c2 = st.columns(2)
        with c1:
            ui.dataframe(jv_sum.round(6))
        with c2:
            scores = _table("cv_jv_metric_scores.csv")
            if scores is not None:
                st.markdown("Metrics recovered from the **predicted curve**:")
                ui.dataframe(scores.round(5))
                st.markdown(
                    '<div class="tw-caption">Comparing these against the direct scalar '
                    "model above answers a fair question: is it better to predict a "
                    "number directly, or to predict the whole curve and read the number "
                    "off it?</div>",
                    unsafe_allow_html=True,
                )
        modes = _table("pod_modes.csv")
        if modes is not None:
            st.markdown("**Retained curve shapes**")
            ui.dataframe(modes.round(6))

    if card:
        st.markdown("---")
        with st.expander("Model card (full provenance)"):
            st.json(card)


# --------------------------------------------------------------------------
def _simulation_data() -> None:
    st.markdown("## The simulations behind the model")
    metrics, summary = _doe()

    cols = st.columns(4)
    cols[0].metric("COMSOL runs", summary["n_runs"])
    cols[1].metric("Observations", summary["n_observations"])
    cols[2].metric("J-V curves", summary["n_curves"])
    cols[3].metric("Points per curve", summary["n_voltage_points"])

    ui.banner(
        "A complete 6&times;6 factorial: every combination of six illumination levels "
        "and six temperatures, each aged through ten time points. Balanced designs "
        "like this are what make a surrogate defensible on a small budget &mdash; there "
        "are no confounded corners.<br>"
        "<span class='tw-caption'>Important protocol detail: cells are <em>aged</em> at "
        "the illumination shown, but every diagnostic J-V sweep is taken at standard "
        "1 sun. Efficiency is therefore always referenced to 100 mW/cm².</span>",
        kind="info",
        title="Design of the campaign",
    )

    st.pyplot(plots.envelope_map(title="The 36 simulated conditions"), clear_figure=True)

    st.markdown("---")
    st.markdown("### Browse a single run")
    c1, c2 = st.columns(2)
    suns = c1.selectbox("Illumination (suns)", sorted(metrics["aging_light_suns"].unique()))
    temp = c2.selectbox(
        "Temperature (C)",
        sorted(metrics["aging_temperature_C"].unique()),
        format_func=lambda v: f"{v:.1f}",
    )
    run = metrics[
        (metrics["aging_light_suns"] == suns) & (metrics["aging_temperature_C"] == temp)
    ].sort_values("aging_h")

    if run.empty:
        st.info("No simulation at that combination.")
    else:
        left, right = st.columns([1.5, 1])
        with left:
            st.pyplot(
                plots.trajectory(
                    run.rename(columns={"aging_h": "aging_h"}),
                    "PCE_retention_pct",
                    show_band=False,
                    title=f"Simulated retention at {suns:g} suns, {temp:.0f} C",
                ),
                clear_figure=True,
            )
        with right:
            ui.dataframe(
                run[["aging_h", "PCE_pct", "Voc_V", "Jsc_mAcm2", "FF", "PCE_retention_pct"]].round(4)
            )

    st.markdown("---")
    with st.expander("A second, separate dataset: mechanism-isolation study"):
        st.markdown(
            "The repository also ships a 12-scenario COMSOL study in which individual "
            "degradation mechanisms were switched on in isolation. It is **not** used "
            "to train the surrogate and is included as reference context only."
        )
        try:
            seed_ts, seed_life = load_seed()
            ui.dataframe(seed_life.round(4).head(12))
            st.caption(f"{len(seed_ts)} rows across {seed_ts['case_id'].nunique()} cases.")
        except Exception as exc:  # pragma: no cover
            st.info(f"Reference dataset unavailable: {exc}")


# --------------------------------------------------------------------------
def _uncertainty() -> None:
    st.markdown("## Where the model is unsure, and what to simulate next")
    cap = get("out_active_learning")
    st.markdown(
        f'<div class="tw-caption">{ui.tier_pill_html(cap.tier)} &nbsp; {cap.backing}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    ui.banner(
        "This is the practical payoff of choosing a Gaussian process. The model "
        "reports how uncertain it is everywhere, not just what it predicts, so the "
        "next expensive simulation can be aimed at whichever corner of the design "
        "space is currently darkest. A tree ensemble has no comparable notion of "
        "where it does not know.",
        kind="info",
        title="Active learning, not guesswork",
    )

    recs = _table("active_learning_recommendations.csv")
    if recs is None:
        if not predict_mod.models_available():
            ui.banner(
                "Train the models first: <code>python scripts/train_models.py</code>.",
                kind="planned",
                title="Not trained yet",
            )
            return
        with st.spinner("Scoring the design space ..."):
            scalars, _ = predict_mod._load_models()
            recs = activelearn.recommend_runs(scalars, n=5)

    left, right = st.columns([1.2, 1])
    with left:
        st.pyplot(plots.envelope_map(recommendations=recs, title="Suggested next simulations"), clear_figure=True)
    with right:
        st.markdown("### The next five runs")
        ui.dataframe(
            recs[["rank", "aging_light_suns", "aging_temperature_C", "uncertainty_score"]].round(3)
        )

    st.markdown("### Why these")
    for row in recs.itertuples():
        st.markdown(
            f"**{row.rank}. {row.aging_light_suns:g} suns, {row.aging_temperature_C:.0f} C** "
            f"&mdash; {row.reason}"
        )

    st.markdown("---")
    cov = activelearn.coverage_summary()
    c = st.columns(4)
    c[0].metric("Design points", cov["n_design_points"])
    c[1].metric("Illumination levels", cov["illumination_levels"])
    c[2].metric("Temperature levels", cov["temperature_levels"])
    c[3].metric("Full factorial", "yes" if cov["is_full_factorial"] else "no")


# --------------------------------------------------------------------------
def _climate() -> None:
    st.markdown("## From a lab oven to a real climate")
    cap = get("twin_climate")
    st.markdown(
        f'<div class="tw-caption">{ui.tier_pill_html(cap.tier)} &nbsp; {cap.backing}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    ui.banner(
        "The surrogate understands illumination and temperature. A climate is just a "
        "schedule of those two quantities through the year, so a climate can be fed "
        "straight into it. That is the bridge from this app to the long-term goal of "
        "twinning real installations.<br><br>"
        "<strong>Read these as archetypes, not measurements.</strong> The monthly "
        "profiles below are hand-built representative values for each climate class, "
        "not typical-meteorological-year data from any weather station. They are here "
        "to demonstrate the coupling, and are labelled as such in every export.",
        kind="preview",
        title="Climate archetypes, honestly labelled",
    )

    try:
        table = climate.archetype_table()
    except Exception as exc:  # pragma: no cover
        st.error(f"Climate module unavailable: {exc}")
        return

    # 'key' is the lookup handle, 'archetype' the human-readable name.
    keys = list(table["key"])
    labels = list(table["archetype"])

    picked = st.selectbox("Climate archetype", labels)
    key = keys[labels.index(picked)]
    arche = climate.get_archetype(key)
    years = st.slider("Deployment length (years)", 1, 5, 1)

    schedule = climate.to_stress_schedule(arche, years=years)
    report = climate.envelope_report(schedule)

    # The schedule splits each month into daylight and dark segments. Only the
    # daylight rows carry meaningful illumination, so plot those.
    daylight = (
        schedule[schedule["segment"] == "daylight"]
        if "segment" in schedule.columns
        else schedule
    )

    left, right = st.columns([1.5, 1])
    with left:
        if daylight.empty:
            ui.dataframe(schedule.head(24).round(3))
        else:
            st.pyplot(plots.climate_schedule(daylight, name=picked), clear_figure=True)
    with right:
        st.markdown(f"### {picked}")
        st.markdown(f"*{getattr(arche, 'description', '')}*")
        st.markdown("")
        row = table[table["key"] == key].iloc[0]
        st.metric("Annual insolation", f"{row['annual_insolation_kWh_m2']:,.0f} kWh/m²")
        st.metric("Mean ambient temperature", f"{row['annual_mean_temperature_C']:.1f} °C")
        st.metric("Mean relative humidity", f"{row['annual_mean_RH_pct']:.0f} %")

    st.markdown("")
    ui.banner(
        "<ul style='margin:0.2rem 0 0 1.1rem; padding:0;'>"
        + "".join(f"<li>{note}</li>" for note in (getattr(report, "notes", None) or []))
        + "</ul>",
        kind="preview",
        title="Why this is not yet a forecast",
    )

    st.markdown("---")
    st.markdown("### The stress schedule")
    ui.dataframe(
        daylight.loc[
            :,
            [
                c
                for c in (
                    "year_index", "month_name", "illumination_suns",
                    "ambient_temperature_C", "cell_temperature_C",
                    "relative_humidity_pct", "hours_at_condition", "elapsed_hours_end",
                )
                if c in daylight.columns
            ],
        ].head(24).round(2)
    )
    st.caption(
        f"Daylight segments only, first 24 of {len(daylight)}. "
        f"Relative humidity is carried through but the surrogate cannot use it."
    )


# --------------------------------------------------------------------------
def _mechanisms() -> None:
    st.markdown("## Which mechanism is doing the damage?")
    cap = get("out_mechanism")
    st.markdown(
        f'<div class="tw-caption">{ui.tier_pill_html(cap.tier)} &nbsp; {cap.backing}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    ui.banner(heuristic.DISCLAIMER, kind="preview", title="Interpretation, not measurement")

    illumination = float(st.session_state["illumination_suns"])
    temperature = float(st.session_state["temperature_c"])
    horizon = float(st.session_state["horizon_h"])

    attribution = heuristic.mechanism_attribution(illumination, temperature, horizon)

    ui.preview_zone_open()
    left, right = st.columns([1.3, 1])
    with left:
        st.pyplot(plots.mechanism_bars(attribution, label_col="label"), clear_figure=True)
    with right:
        st.markdown("**Stress drivers at these conditions**")
        ui.dataframe(heuristic.driver_table(illumination, temperature, horizon).round(3))
    ui.preview_zone_close()

    for row in attribution.itertuples():
        st.markdown(f"**{row.label}** ({row.weight:.0%}) &mdash; {row.evidence}")


# --------------------------------------------------------------------------
def _roadmap() -> None:
    st.markdown("## What is built, what is coming")
    counts = tier_counts()

    cols = st.columns(3)
    cols[0].metric(f"{TIER_ICON[Tier.VALIDATED]} Validated", counts["Validated"])
    cols[1].metric(f"{TIER_ICON[Tier.PREVIEW]} Preview", counts["Preview"])
    cols[2].metric(f"{TIER_ICON[Tier.PLANNED]} Planned", counts["Planned"])

    st.markdown("")
    ui.banner(
        "This table is generated from the same registry that drives every badge and "
        "every disabled control in the app. There is no separate roadmap document to "
        "drift out of date: if a capability is switched off in the interface, it is "
        "because of a row in here.",
        kind="info",
        title="One source of truth",
    )

    for group in GROUPS:
        st.markdown(f"### {group}")
        for cap in in_group(group):
            icon = TIER_ICON[cap.tier]
            version = f" &middot; {cap.version}" if cap.version else ""
            body = f"{cap.backing}"
            if cap.unlocks:
                body += f"<br><em>Unlocked by: {cap.unlocks}</em>"
            st.markdown(
                f'<div class="tw-card{" tw-card-planned" if cap.tier is Tier.PLANNED else ""}" '
                f'style="margin-bottom:0.55rem;">'
                f"<h4>{icon} {cap.label}{version}</h4><p>{body}</p></div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")
    with st.expander("Full registry as a table"):
        ui.dataframe(registry_table())


# --------------------------------------------------------------------------
def _export() -> None:
    st.markdown("## Take the results with you")
    st.markdown(
        "Everything the app computed for the current conditions, in open formats."
    )

    illumination = float(st.session_state["illumination_suns"])
    temperature = float(st.session_state["temperature_c"])
    horizon = float(st.session_state["horizon_h"])
    architecture = st.session_state["architecture"]

    if not predict_mod.models_available():
        ui.banner(
            "Train the models first: <code>python scripts/train_models.py</code>.",
            kind="planned",
            title="Not trained yet",
        )
        return

    try:
        pred = predict_mod.predict(
            illumination_suns=illumination,
            temperature_c=temperature,
            horizon_h=horizon,
            architecture=architecture,
        )
    except predict_mod.PlannedCapabilityError as exc:
        ui.planned_card(exc.capability)
        return

    c1, c2, c3 = st.columns(3)
    c1.download_button(
        "Predicted trajectory (CSV)",
        data=pred.trajectories.to_csv(index=False).encode("utf-8"),
        file_name=f"psc_trajectory_{illumination:g}sun_{temperature:.0f}C.csv",
        mime="text/csv",
        width="stretch",
    )
    if pred.curve is not None:
        c2.download_button(
            "Predicted J-V curve (CSV)",
            data=pred.curve.to_frame().to_csv(index=False).encode("utf-8"),
            file_name=f"psc_jv_{illumination:g}sun_{temperature:.0f}C_{horizon:.0f}h.csv",
            mime="text/csv",
            width="stretch",
        )
    c3.download_button(
        "Capability registry (CSV)",
        data=registry_table().to_csv(index=False).encode("utf-8"),
        file_name="psc_capabilities.csv",
        mime="text/csv",
        width="stretch",
    )

    card = _card()
    if card:
        st.download_button(
            "Model card (JSON)",
            data=json.dumps(card, indent=2).encode("utf-8"),
            file_name="psc_model_card.json",
            mime="application/json",
        )

    st.markdown("---")
    st.markdown("### Cite this")
    st.code(
        "Tippin, F. PSC Digital Twin: a climate-aware Gaussian-process surrogate for\n"
        "perovskite solar cell degradation. Software, 2026.\n"
        "See CITATION.cff in the repository for the machine-readable record.",
        language="text",
    )
