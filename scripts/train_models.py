"""Train, validate, and record the surrogate models.

This is the one command that turns the shipped simulation data into the model
artifacts the app loads:

    python scripts/train_models.py            # full run
    python scripts/train_models.py --quick    # fast smoke run, fewer restarts

It is deterministic. Every estimator carries a fixed ``random_state``, so two
runs on the same data produce identical artifacts and identical metrics.

Outputs
-------
models/scalar_gp.joblib     the per-metric Gaussian processes
models/jv_pod.joblib        the POD basis plus mode-coefficient GPs
models/model_card.json      what was trained, on what, and how well it scored
outputs/tables/*.csv        validation tables for the paper
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
import sklearn  # noqa: E402

from psc_twin import __version__  # noqa: E402
from psc_twin.activelearn import coverage_summary, recommend_runs  # noqa: E402
from psc_twin.capabilities import ENVELOPE, ENVELOPE_LEVELS, tier_counts  # noqa: E402
from psc_twin.data import TARGETS, doe_summary, load_doe  # noqa: E402
from psc_twin.surrogate import jv_pod, scalar_gp  # noqa: E402
from psc_twin.surrogate.validation import (  # noqa: E402
    coverage_verdict,
    learning_curve,
    learning_curve_verdict,
    scalar_cv_table,
)

TABLE_DIR = ROOT / "outputs" / "tables"
MODEL_DIR = ROOT / "models"
CARD_PATH = MODEL_DIR / "model_card.json"


def _write(df: pd.DataFrame, name: str) -> Path:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLE_DIR / name
    df.to_csv(path, index=False)
    return path


def _validation_from_tables() -> dict[str, object]:
    """Assemble the card's validation block from tables on disk.

    Used by ``--reuse-validation`` so a model card can be regenerated without
    repeating a multi-hour cross-validation. The tables must have been written
    by this same script; the returned block records that they were reused rather
    than recomputed, so the provenance stays honest.
    """
    def read(name: str) -> pd.DataFrame | None:
        path = TABLE_DIR / name
        return pd.read_csv(path) if path.exists() else None

    scalar = read("cv_scalar_metrics.csv")
    jv_summary = read("cv_jv_summary.csv")
    jv_scores = read("cv_jv_metric_scores.csv")
    curve = read("learning_curve.csv")
    if scalar is None:
        return {}

    block: dict[str, object] = {
        "method": "LeaveOneGroupOut over run_id (36 folds); no ungrouped split anywhere",
        "source": (
            "Loaded from outputs/tables/, produced by a previous run of this "
            "script. Re-run without --reuse-validation to recompute."
        ),
        "scalar_metrics": scalar.to_dict(orient="records"),
        "coverage_verdict": coverage_verdict(scalar.rename(columns={"coverage_95": "coverage_95"})),
    }
    if jv_summary is not None:
        block["jv_summary"] = jv_summary.to_dict(orient="records")
    if jv_scores is not None:
        block["jv_metric_scores"] = jv_scores.to_dict(orient="records")
    if curve is not None:
        block["learning_curve_verdict"] = learning_curve_verdict(curve)
        block["learning_curve"] = (
            curve.groupby("n_training_runs")[["mae", "rmse", "coverage_95"]]
            .mean()
            .reset_index()
            .to_dict(orient="records")
        )
    return block


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="fast smoke run")
    parser.add_argument("--n-jobs", type=int, default=None, help="parallel workers for CV")
    parser.add_argument("--skip-cv", action="store_true", help="fit and save only")
    parser.add_argument(
        "--reuse-validation",
        action="store_true",
        help=(
            "populate the model card from validation tables already in "
            "outputs/tables/ instead of recomputing them"
        ),
    )
    args = parser.parse_args(argv)

    started = time.time()
    print("=" * 74)
    print("PSC Digital Twin - surrogate training")
    print("=" * 74)

    bundle = load_doe()
    summary = doe_summary(bundle)
    print(
        f"\nTraining data: {summary['n_runs']} COMSOL runs, "
        f"{summary['n_observations']} observations, "
        f"{summary['n_curves']} J-V curves of {summary['n_voltage_points']} points."
    )
    print(
        f"  illumination {summary['illumination_levels']}\n"
        f"  temperature  {summary['temperature_levels']}\n"
        f"  aging times  {summary['aging_times_h']}"
    )

    config = scalar_gp.DEFAULT_CONFIG
    if args.quick:
        # The guard in GPConfig blocks fewer than 8 restarts for a shipped fit,
        # which is right; --quick is explicitly not a shipped fit.
        config = dataclasses.replace(config)
        object.__setattr__(config, "n_restarts_optimizer", 2) if not hasattr(
            config, "__dict__"
        ) else config.__dict__.update(n_restarts_optimizer=2)
        print("\n[quick] reduced optimiser restarts; artifacts are NOT publication grade")

    # ---- scalar surrogates ------------------------------------------------
    print(f"\nFitting {len(TARGETS)} scalar Gaussian processes ...")
    t0 = time.time()
    scalars = scalar_gp.fit_all(bundle, config=config)
    scalar_fit_s = time.time() - t0
    scalar_path = scalar_gp.save(scalars)
    print(f"  done in {scalar_fit_s:.1f}s -> {scalar_path.relative_to(ROOT)}")
    for target, model in scalars.models.items():
        # length_scales is a {feature: value} mapping, one entry per ARD dimension.
        ls = ", ".join(f"{k}={v:.3g}" for k, v in model.length_scales.items())
        print(f"    {target:20s} {ls}  noise {model.noise_level:.2e}")

    # ---- J-V curve surrogate ---------------------------------------------
    print("\nFitting the POD J-V surrogate ...")
    t0 = time.time()
    jv = jv_pod.fit(bundle, config=config)
    jv_fit_s = time.time() - t0
    jv_path = jv_pod.save(jv)
    print(f"  done in {jv_fit_s:.1f}s -> {jv_path.relative_to(ROOT)}")
    modes = jv_pod.mode_table(jv.basis)
    print(f"  retained {jv.n_modes} modes covering {jv.basis.cumulative_energy * 100:.4f}% of curve energy")
    for row in modes.itertuples():
        print(f"    {row.mode}: {row.explained_variance_ratio * 100:8.4f}%  (cumulative {row.cumulative * 100:.4f}%)")
    _write(modes, "pod_modes.csv")

    trunc_rmse, trunc_nrmse = jv_pod.truncation_error(bundle, jv.basis)
    print(f"  POD truncation RMSE {trunc_rmse:.5f} mA/cm2 ({trunc_nrmse:.4f}% of Jsc)")

    card: dict[str, object] = {
        "app_version": __version__,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "quick_mode": bool(args.quick),
        "training_data": summary,
        "design_envelope": ENVELOPE,
        "design_levels": {k: list(v) for k, v in ENVELOPE_LEVELS.items()},
        "architecture": "arch_baseline_pin (single architecture; see limitations)",
        "scalar_model": {
            "targets": list(TARGETS),
            "kernel": "ConstantKernel * Matern(nu=2.5, ARD) + WhiteKernel",
            "time_warp": config.time_warp,
            "n_restarts_optimizer": config.n_restarts_optimizer,
            "random_state": config.random_state,
            "fit_seconds": round(scalar_fit_s, 2),
            "hyperparameters": {
                t: {
                    "length_scales": {k: float(v) for k, v in m.length_scales.items()},
                    "noise_level": float(m.noise_level),
                    "log_marginal_likelihood": float(m.log_marginal_likelihood),
                }
                for t, m in scalars.models.items()
            },
        },
        "jv_model": {
            "method": "POD (SVD) basis with one GP per mode coefficient",
            "n_modes": int(jv.n_modes),
            "explained_variance_ratio": [float(v) for v in jv.basis.explained_variance_ratio],
            "cumulative_energy": float(jv.basis.cumulative_energy),
            "truncation_rmse_mAcm2": float(trunc_rmse),
            "truncation_nrmse_pct_of_jsc": float(trunc_nrmse),
            "voltage_points": int(jv.basis.voltage.size),
            "scan_illumination_suns": jv_pod.SCAN_ILLUMINATION_SUNS,
            "efficiency_reference_mW_cm2": jv_pod.ONE_SUN_MW_CM2,
            "fit_seconds": round(jv_fit_s, 2),
        },
        "capability_tiers": tier_counts(),
        "design_coverage": coverage_summary(bundle),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
            "platform": platform.platform(),
        },
        "limitations": [
            "Trained on a single device architecture (baseline p-i-n). Predictions "
            "for n-i-p or tandem stacks are not supported and the app refuses them.",
            "No humidity or electrical-bias axis exists in the campaign, so those "
            "stress factors cannot influence any prediction.",
            "Aging was simulated to 1000 h; longer horizons are extrapolation.",
            "Mechanism attribution is an interpretive overlay, not validated against "
            "per-mechanism state variables.",
            "Efficiency is referenced to a fixed 1-sun diagnostic scan, matching the "
            "campaign protocol, not to the aging illumination.",
        ],
    }

    # ---- validation -------------------------------------------------------
    if args.reuse_validation:
        card["validation"] = _validation_from_tables()
        if card["validation"]:
            print("\nValidation reused from outputs/tables/ (not recomputed).")
            for key in ("coverage_verdict", "learning_curve_verdict"):
                if card["validation"].get(key):
                    print(f"  {card['validation'][key]}")
        else:
            print("\n[warn] --reuse-validation set but no tables found; card has no validation block")
    elif not args.skip_cv:
        print("\nCross-validating (leave-one-run-out over 36 runs) ...")
        t0 = time.time()
        report = scalar_gp.cross_validate(bundle, config=config, n_jobs=args.n_jobs)
        cv_s = time.time() - t0
        table = scalar_cv_table(report)
        print(f"  done in {cv_s:.1f}s")
        print(table.round(5).to_string(index=False))
        _write(table, "cv_scalar_metrics.csv")
        _write(report.predictions, "cv_scalar_predictions.csv")

        cov_text = coverage_verdict(table)
        print(f"\n  {cov_text}")

        print("\nValidating the J-V surrogate (leave-one-run-out) ...")
        t0 = time.time()
        jv_val = jv_pod.cross_validate(bundle, config=config, n_jobs=args.n_jobs)
        jv_cv_s = time.time() - t0
        print(f"  done in {jv_cv_s:.1f}s")
        print(jv_val.summary_frame().round(5).to_string(index=False))
        print("\n  Metrics recovered from predicted curves:")
        print(jv_val.metric_scores.round(5).to_string(index=False))
        _write(jv_val.summary_frame(), "cv_jv_summary.csv")
        _write(jv_val.metric_scores, "cv_jv_metric_scores.csv")

        print("\nLearning curve ...")
        t0 = time.time()
        curve = learning_curve(
            bundle,
            train_sizes=(6, 18, 30) if args.quick else (6, 12, 18, 24, 30),
            n_repeats=2 if args.quick else 5,
            config=config,
            n_jobs=args.n_jobs,
        )
        lc_s = time.time() - t0
        print(f"  done in {lc_s:.1f}s")
        print(curve.groupby("n_training_runs")[["mae", "rmse", "coverage_95"]].mean().round(5).to_string())
        _write(curve, "learning_curve.csv")
        lc_text = learning_curve_verdict(curve)
        print(f"\n  {lc_text}")

        card["validation"] = {
            "method": "LeaveOneGroupOut over run_id (36 folds); no ungrouped split anywhere",
            "cv_seconds": round(cv_s, 2),
            "scalar_metrics": table.to_dict(orient="records"),
            "coverage_verdict": cov_text,
            "jv_curve_rmse_mAcm2": float(jv_val.curve_rmse),
            "jv_curve_nrmse_pct_of_jsc": float(jv_val.curve_nrmse_pct),
            "jv_per_point_coverage95": float(jv_val.per_point_coverage95),
            "jv_metric_scores": jv_val.metric_scores.to_dict(orient="records"),
            "learning_curve_verdict": lc_text,
            "convergence_warnings": int(report.n_convergence_warnings),
        }

    # ---- active learning --------------------------------------------------
    print("\nRecommending the next simulations ...")
    recs = recommend_runs(scalars, n=5, bundle=bundle)
    print(recs.loc[:, ["rank", "aging_light_suns", "aging_temperature_C", "uncertainty_score", "distance_to_nearest_run"]].to_string(index=False))
    _write(recs, "active_learning_recommendations.csv")
    card["next_runs"] = recs.to_dict(orient="records")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CARD_PATH.write_text(json.dumps(card, indent=2, default=str), encoding="utf-8")
    print(f"\nModel card -> {CARD_PATH.relative_to(ROOT)}")
    print(f"Total wall clock: {time.time() - started:.1f}s")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
