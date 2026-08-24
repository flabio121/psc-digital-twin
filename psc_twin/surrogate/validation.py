"""Validation evidence for the surrogate, in the form a reviewer will ask for.

Three questions decide whether this model is publishable, and there is one
function here for each:

1. *Does it generalise to conditions it has not seen?*
   :func:`scalar_cv_table` -- leave-one-run-out over the 36 design points.
2. *Do the error bars mean anything?*
   Coverage of the 95% predictive interval, reported in the same table. An
   uncertainty band nobody has checked is decoration.
3. *Is 36 runs enough, or is the campaign under-sampled?*
   :func:`learning_curve` -- accuracy as a function of how many runs the model
   was allowed to train on. A curve that has flattened says the design is
   saturated; one still falling says more simulations would pay, which is a
   perfectly publishable answer and is exactly what the active-learning
   workspace acts on.

Every split in this module is grouped by ``run_id``. Ten rows share each design
point, so an ungrouped split would leak the answer into the training set and
report a fantasy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from psc_twin.data import DoeBundle, load_doe
from psc_twin.surrogate.scalar_gp import (
    DEFAULT_CONFIG,
    CvReport,
    GPConfig,
    cross_validate,
    fit_target,
)

_Z95 = 1.959963985


def scalar_cv_table(report: CvReport) -> pd.DataFrame:
    """Per-target held-out metrics as a presentable table."""
    rows = []
    for score in report.scores:
        rows.append(
            {
                "target": score.target,
                "n_held_out": score.n,
                "MAE": score.mae,
                "RMSE": score.rmse,
                "R2": score.r2,
                "coverage_95": score.coverage95,
                "mean_predicted_sd": score.mean_pred_std,
                "max_abs_error": score.max_abs_error,
            }
        )
    return pd.DataFrame(rows)


def parity_frame(report: CvReport, target: str) -> pd.DataFrame:
    """Held-out actual/predicted pairs for one target, ready to plot."""
    preds = report.predictions
    return pd.DataFrame(
        {
            "actual": preds[f"{target}_true"],
            "predicted": preds[f"{target}_pred"],
            "sd": preds[f"{target}_std"],
            "aging_light_suns": preds["aging_light_suns"],
            "aging_temperature_C": preds["aging_temperature_C"],
            "aging_h": preds["aging_h"],
        }
    )


def residual_frame(report: CvReport, target: str) -> pd.DataFrame:
    """Residuals across the stress plane, averaged per design point.

    Systematic structure here -- a whole corner biased one way -- would mean the
    kernel is mis-specified rather than merely uncertain.
    """
    preds = report.predictions
    out = pd.DataFrame(
        {
            "aging_light_suns": preds["aging_light_suns"],
            "aging_temperature_C": preds["aging_temperature_C"],
            "residual": preds[f"{target}_pred"] - preds[f"{target}_true"],
        }
    )
    return (
        out.groupby(["aging_light_suns", "aging_temperature_C"], as_index=False)["residual"]
        .mean()
    )


def learning_curve(
    bundle: DoeBundle | None = None,
    target: str = "PCE_retention_pct",
    train_sizes: tuple[int, ...] = (6, 12, 18, 24, 30),
    n_repeats: int = 5,
    config: GPConfig = DEFAULT_CONFIG,
    random_state: int = 0,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    """Held-out error against the number of training runs.

    For each size, sample that many whole runs at random for training and hold
    the rest out, repeating to average over which runs were drawn. Sampling
    whole runs -- never individual rows -- keeps the grouping intact.
    """
    bundle = bundle if bundle is not None else load_doe()
    X = bundle.feature_matrix()
    y = bundle.target_vector(target)
    groups = bundle.groups()
    unique_runs = np.array(pd.unique(groups))
    n_runs = len(unique_runs)

    jobs = []
    for size in train_sizes:
        size = int(min(size, n_runs - 1))
        splitter = GroupShuffleSplit(
            n_splits=n_repeats, train_size=size / n_runs, random_state=random_state
        )
        for repeat, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups)):
            jobs.append((size, repeat, train_idx, test_idx))

    def run(size: int, repeat: int, train_idx: np.ndarray, test_idx: np.ndarray) -> dict:
        model = fit_target(X[train_idx], y[train_idx], target, config)
        mean, std = model.predict(X[test_idx], return_std=True)
        err = mean - y[test_idx]
        return {
            "n_training_runs": int(len(np.unique(groups[train_idx]))),
            "repeat": repeat,
            "target": target,
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err**2))),
            "coverage_95": float(np.mean(np.abs(err) <= _Z95 * std)),
        }

    try:
        import joblib

        if n_jobs is not None:
            rows = joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(run)(s, r, tr, te) for s, r, tr, te in jobs
            )
        else:
            rows = [run(s, r, tr, te) for s, r, tr, te in jobs]
    except ImportError:  # pragma: no cover
        rows = [run(s, r, tr, te) for s, r, tr, te in jobs]

    return pd.DataFrame(rows)


def learning_curve_verdict(curve: pd.DataFrame, tol: float = 0.10) -> str:
    """Read the learning curve honestly and say what it implies.

    ``tol`` is the relative RMSE improvement over the last step below which the
    curve counts as flattened.
    """
    grouped = curve.groupby("n_training_runs")["rmse"].mean().sort_index()
    if len(grouped) < 2:
        return "Not enough points on the learning curve to judge saturation."

    last, prev = float(grouped.iloc[-1]), float(grouped.iloc[-2])
    improvement = (prev - last) / prev if prev > 0 else 0.0

    if improvement < tol:
        return (
            f"Held-out RMSE improved only {improvement * 100:.1f}% between "
            f"{grouped.index[-2]} and {grouped.index[-1]} training runs, so the "
            "design is close to saturated: more runs at the same spacing would "
            "buy little. Extra effort is better spent widening the envelope than "
            "filling it in."
        )
    return (
        f"Held-out RMSE was still falling ({improvement * 100:.1f}% over the last "
        f"step, {grouped.index[-2]} to {grouped.index[-1]} runs), so the campaign "
        "is under-sampled and additional simulations should still pay. The "
        "active-learning workspace ranks where to spend them."
    )


def coverage_verdict(table: pd.DataFrame, target_coverage: float = 0.95) -> str:
    """State plainly whether the uncertainty bands are trustworthy."""
    cov = table["coverage_95"].to_numpy(dtype=float)
    mean_cov = float(np.mean(cov))
    worst = table.loc[table["coverage_95"].idxmin()]

    if mean_cov >= target_coverage - 0.05:
        quality = "well calibrated"
    elif mean_cov >= target_coverage - 0.15:
        quality = "somewhat over-confident"
    else:
        quality = "clearly over-confident"

    return (
        f"Mean 95% interval coverage across targets is {mean_cov:.3f}, which is "
        f"{quality} against the nominal 0.95. Worst target is "
        f"{worst['target']} at {worst['coverage_95']:.3f}."
    )


def full_report(
    bundle: DoeBundle | None = None,
    config: GPConfig = DEFAULT_CONFIG,
    n_jobs: int | None = None,
    quick: bool = False,
) -> dict[str, object]:
    """Everything needed for the model card and the paper's validation section."""
    bundle = bundle if bundle is not None else load_doe()
    report = cross_validate(bundle, config=config, n_jobs=n_jobs)
    table = scalar_cv_table(report)

    curve = learning_curve(
        bundle,
        train_sizes=(6, 18, 30) if quick else (6, 12, 18, 24, 30),
        n_repeats=2 if quick else 5,
        config=config,
        n_jobs=n_jobs,
    )

    return {
        "cv_report": report,
        "cv_table": table,
        "learning_curve": curve,
        "learning_curve_verdict": learning_curve_verdict(curve),
        "coverage_verdict": coverage_verdict(table),
    }
