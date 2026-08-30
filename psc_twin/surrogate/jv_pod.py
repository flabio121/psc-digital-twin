"""Full J-V curve surrogate: proper orthogonal decomposition + GP coefficients.

Why not regress the current at each voltage independently? A J-V curve is a
*functional* output, not 72 unrelated numbers. Per-point regression is free to
produce a curve that wiggles, double-crosses zero, or bends the wrong way near
the maximum power point, because nothing in that formulation knows the points
belong to one physical object.

Proper orthogonal decomposition (POD, the same computation as PCA) instead finds
the handful of characteristic curve *shapes* whose weighted sum reproduces every
curve in the family. Three consequences matter here:

* Any reconstruction is a linear combination of smooth empirical modes, so it is
  smooth and physically plausible by construction.
* The learning problem shrinks from 72 outputs to a few mode coefficients, which
  is what makes 36 runs a workable training set at all.
* The modes are interpretable -- the leading one is almost always "overall
  current scale", the next "voltage-axis softening", and so on.

Each retained coefficient gets its own Gaussian process over
(illumination, temperature, aging time), reusing the kernel and input warp
defined in :mod:`psc_twin.surrogate.scalar_gp` so the two surrogates cannot
drift apart.

Validation is grouped by ``run_id`` throughout. Ten curves share each design
point, so an ungrouped split would leak.
"""

from __future__ import annotations

import pickle
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from psc_twin.data import DoeBundle, load_doe
from psc_twin.surrogate.scalar_gp import (
    DEFAULT_CONFIG,
    GPConfig,
    ScalarSurrogate,
    fit_target,
)

try:  # joblib ships with scikit-learn; fall back to pickle if that changes.
    import joblib as _joblib
except ImportError:  # pragma: no cover
    _joblib = None

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
#: Canonical artifact location. Named to match scalar_gp.DEFAULT_ARTIFACT so the
#: two surrogate modules present the same interface to predict.py.
DEFAULT_ARTIFACT = MODELS_DIR / "jv_pod.joblib"
MODEL_PATH = DEFAULT_ARTIFACT  # backwards-compatible alias

#: Energy the retained modes must capture before truncation stops.
ENERGY_TARGET = 0.9999

#: Hard ceiling on retained modes. With 36 design points, more coefficient GPs
#: than this would be fitting numerical noise in the SVD tail.
MAX_MODES = 8

#: Largest pointwise reconstruction residual tolerated in the operating region,
#: in mA/cm^2.
#:
#: Energy alone is a poor stopping rule here, and measurably so. Truncating at
#: 99.99% of variance keeps 3 modes, which reproduces the bulk of every curve
#: but leaves a 1.07 mA/cm^2 residual at V ~ 1.22 V on the most degraded run --
#: right in the knee, which is precisely where Voc and the fill factor are read.
#: A fourth mode carries little variance yet fixes the knee, cutting the worst
#: residual roughly nine-fold to 0.12 mA/cm^2 and the RMSE from 0.050 to 0.009.
#: So truncation is driven by the error where the physics is extracted, not by
#: variance share.
MAX_OPERATING_ERROR = 0.25

#: Upper edge of the operating region, in volts. Every figure of merit comes
#: from between short circuit and just past open circuit; past that the current
#: dives steeply and nobody reads it.
OPERATING_VOLTAGE_MAX = 1.30

#: Reference irradiance for 1 sun, in mW/cm^2.
#:
#: Efficiency here is *always* referenced to this value, never to the aging
#: illumination. The campaign follows the standard stability protocol: a cell is
#: aged at some illumination between 0.01 and 1 sun, but every diagnostic J-V
#: sweep is taken at standard 1-sun AM1.5G. ``scan_light_suns`` is 1.0 for all
#: 36 runs while ``aging_light_suns`` sweeps, and the campaign's own PCE column
#: matches a fixed 100 mW/cm^2 denominator exactly.
#:
#: Dividing by the aging illumination instead would inflate efficiency by up to
#: 100x at the dim corner of the design.
ONE_SUN_MW_CM2 = 100.0

#: Illumination at which every diagnostic J-V sweep is performed.
SCAN_ILLUMINATION_SUNS = 1.0


# --------------------------------------------------------------------------
# PV metric extraction
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CurveMetrics:
    """Photovoltaic figures of merit read off a single J-V curve."""

    jsc_macm2: float
    voc_v: float
    ff: float
    pce_pct: float
    vmp_v: float
    jmp_macm2: float
    pmax_mwcm2: float
    valid: bool
    note: str = ""

    def as_dict(self) -> dict[str, float | bool | str]:
        return {
            "Jsc_mAcm2": self.jsc_macm2,
            "Voc_V": self.voc_v,
            "FF": self.ff,
            "PCE_pct": self.pce_pct,
            "Vmp_V": self.vmp_v,
            "Jmp_mAcm2": self.jmp_macm2,
            "Pmax_mWcm2": self.pmax_mwcm2,
            "valid": self.valid,
            "note": self.note,
        }


def _first_zero_crossing(voltage: np.ndarray, current: np.ndarray) -> float:
    """Interpolated voltage where the current first falls through zero.

    Returns NaN when the curve never crosses, which happens for a badly
    extrapolated prediction. Returning NaN rather than the last voltage keeps a
    failed extraction visible instead of quietly reporting a plausible Voc.
    """
    sign = np.sign(current)
    for i in range(len(voltage) - 1):
        if sign[i] > 0 >= sign[i + 1]:
            j0, j1 = current[i], current[i + 1]
            if j1 == j0:
                return float(voltage[i])
            frac = j0 / (j0 - j1)
            return float(voltage[i] + frac * (voltage[i + 1] - voltage[i]))
    return float("nan")


def curve_metrics(voltage: np.ndarray, current: np.ndarray) -> CurveMetrics:
    """Extract Jsc, Voc, FF and efficiency from one J-V curve.

    Efficiency is referenced to a fixed 1-sun input (:data:`ONE_SUN_MW_CM2`),
    not to the aging illumination, because every diagnostic sweep in the
    campaign is taken at standard 1-sun conditions regardless of how the cell
    was aged. See the note on :data:`ONE_SUN_MW_CM2`.

    Verified against the campaign's own metric table: Jsc, Voc, FF and PCE
    reproduce to within floating-point noise across all 360 curves.
    """
    v = np.asarray(voltage, dtype=float)
    j = np.asarray(current, dtype=float)
    order = np.argsort(v)
    v, j = v[order], j[order]

    if not np.isfinite(j).all():
        return CurveMetrics(*(float("nan"),) * 7, valid=False, note="non-finite current")

    jsc = float(np.interp(0.0, v, j))
    voc = _first_zero_crossing(v, j)

    power = v * j  # mW/cm^2 when J is mA/cm^2 and V is volts
    idx = int(np.argmax(power))
    pmax = float(power[idx])
    vmp, jmp = float(v[idx]), float(j[idx])

    pce = 100.0 * pmax / ONE_SUN_MW_CM2

    note = ""
    valid = True
    if not np.isfinite(voc) or voc <= 0:
        valid, note = False, "no zero crossing; Voc undefined"
        ff = float("nan")
    elif jsc <= 0:
        valid, note = False, "non-positive Jsc"
        ff = float("nan")
    else:
        denom = jsc * voc
        ff = float(pmax / denom) if denom > 0 else float("nan")
        if not (0.0 < ff < 1.0):
            valid, note = False, f"fill factor out of range ({ff:.3f})"

    return CurveMetrics(jsc, voc, ff, pce, vmp, jmp, pmax, valid, note)


# --------------------------------------------------------------------------
# the POD basis
# --------------------------------------------------------------------------
@dataclass
class PodBasis:
    """Mean curve plus orthonormal modes truncated at ``n_modes``."""

    voltage: np.ndarray
    mean_curve: np.ndarray
    modes: np.ndarray               # (k, 72), rows orthonormal
    explained_variance_ratio: np.ndarray
    singular_values: np.ndarray

    @property
    def n_modes(self) -> int:
        return int(self.modes.shape[0])

    @property
    def cumulative_energy(self) -> float:
        return float(np.sum(self.explained_variance_ratio))

    def project(self, curves: np.ndarray) -> np.ndarray:
        """Curves -> mode coefficients. Shape (n, k)."""
        centred = np.atleast_2d(np.asarray(curves, dtype=float)) - self.mean_curve
        return centred @ self.modes.T

    def reconstruct(self, coefficients: np.ndarray) -> np.ndarray:
        """Mode coefficients -> curves. Inverse of :meth:`project`."""
        coefficients = np.atleast_2d(np.asarray(coefficients, dtype=float))
        return self.mean_curve + coefficients @ self.modes


def build_basis(
    curve_matrix: np.ndarray,
    voltage: np.ndarray,
    energy_target: float = ENERGY_TARGET,
    max_modes: int = MAX_MODES,
    max_operating_error: float = MAX_OPERATING_ERROR,
    operating_voltage_max: float = OPERATING_VOLTAGE_MAX,
) -> PodBasis:
    """Truncated POD of the curve family, by singular value decomposition.

    Truncation uses two stopping rules and takes whichever demands more modes:
    a cumulative-energy threshold, and a cap on the worst pointwise residual
    inside the operating region. See :data:`MAX_OPERATING_ERROR` for why energy
    alone is not sufficient on this data.
    """
    curves = np.asarray(curve_matrix, dtype=float)
    voltage = np.asarray(voltage, dtype=float)
    mean_curve = curves.mean(axis=0)
    centred = curves - mean_curve

    # full_matrices=False gives the thin SVD; rows of Vt are the modes.
    _u, s, vt = np.linalg.svd(centred, full_matrices=False)
    energy = s**2
    ratio = energy / energy.sum() if energy.sum() > 0 else np.zeros_like(energy)

    cumulative = np.cumsum(ratio)
    k = int(np.searchsorted(cumulative, energy_target) + 1)
    k = max(1, min(k, max_modes, len(s)))

    # Extend until the knee is reproduced well enough to read Voc and FF off it.
    operating = voltage <= operating_voltage_max
    if operating.any() and max_operating_error > 0:
        while k < min(max_modes, len(s)):
            modes = vt[:k]
            recon = mean_curve + (centred @ modes.T) @ modes
            worst = float(np.abs(recon - curves)[:, operating].max())
            if worst <= max_operating_error:
                break
            k += 1

    return PodBasis(
        voltage=np.asarray(voltage, dtype=float),
        mean_curve=mean_curve,
        modes=vt[:k],
        explained_variance_ratio=ratio[:k],
        singular_values=s[:k],
    )


# --------------------------------------------------------------------------
# the fitted surrogate
# --------------------------------------------------------------------------
@dataclass
class PredictedCurve:
    """One predicted J-V curve with a per-point uncertainty band."""

    voltage: np.ndarray
    current_density: np.ndarray
    current_std: np.ndarray
    metrics: CurveMetrics
    illumination_suns: float
    temperature_c: float
    aging_h: float

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "voltage_V": self.voltage,
                "current_density_mAcm2": self.current_density,
                "current_density_std": self.current_std,
            }
        )


@dataclass
class JvSurrogate:
    """POD basis plus one Gaussian process per mode coefficient."""

    basis: PodBasis
    coefficient_models: list[ScalarSurrogate]
    config: GPConfig
    n_train: int
    n_runs: int
    coefficient_scale: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def n_modes(self) -> int:
        return self.basis.n_modes

    def predict_coefficients(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        means, stds = [], []
        for model in self.coefficient_models:
            mean, std = model.predict(X, return_std=True)
            means.append(mean)
            stds.append(std)
        return np.column_stack(means), np.column_stack(stds)

    def predict_curves(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Curves and their per-point standard deviation for a batch of inputs.

        The band is propagated as ``var_j = sum_i var_i * mode_i(V)^2``. That
        treats the mode coefficients as independent, which they are only
        approximately: POD decorrelates the coefficients over the *training*
        set, not over the GP posterior. The band is therefore a good indicator
        and not a rigorous joint credible region, and is documented as such
        wherever it is drawn.
        """
        coef_mean, coef_std = self.predict_coefficients(X)
        curves = self.basis.reconstruct(coef_mean)
        var = (coef_std**2) @ (self.basis.modes**2)
        return curves, np.sqrt(np.maximum(var, 0.0))

    def predict_curve(
        self,
        illumination_suns: float,
        temperature_c: float,
        aging_h: float,
    ) -> PredictedCurve:
        X = np.array([[illumination_suns, temperature_c, aging_h]], dtype=float)
        curves, stds = self.predict_curves(X)
        current = curves[0]
        return PredictedCurve(
            voltage=self.basis.voltage.copy(),
            current_density=current,
            current_std=stds[0],
            metrics=curve_metrics(self.basis.voltage, current),
            illumination_suns=float(illumination_suns),
            temperature_c=float(temperature_c),
            aging_h=float(aging_h),
        )

    def predict_family(
        self,
        illumination_suns: float,
        temperature_c: float,
        aging_times: Sequence[float],
    ) -> list[PredictedCurve]:
        return [
            self.predict_curve(illumination_suns, temperature_c, float(t))
            for t in aging_times
        ]


def fit(
    bundle: DoeBundle | None = None,
    config: GPConfig = DEFAULT_CONFIG,
    energy_target: float = ENERGY_TARGET,
    max_modes: int = MAX_MODES,
) -> JvSurrogate:
    """Build the POD basis and fit one GP per retained coefficient."""
    bundle = bundle if bundle is not None else load_doe()
    basis = build_basis(bundle.curve_matrix, bundle.voltage, energy_target, max_modes)
    coefficients = basis.project(bundle.curve_matrix)
    X = bundle.feature_matrix()

    models = [
        fit_target(X, coefficients[:, i], f"pod_mode_{i + 1}", config)
        for i in range(basis.n_modes)
    ]

    return JvSurrogate(
        basis=basis,
        coefficient_models=models,
        config=config,
        n_train=int(X.shape[0]),
        n_runs=int(pd.unique(bundle.groups()).size),
        coefficient_scale=coefficients.std(axis=0),
    )


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------
@dataclass
class JvValidation:
    """Everything needed to defend the curve surrogate to a reviewer."""

    n_folds: int
    n_modes: int
    explained_variance_ratio: np.ndarray
    truncation_rmse: float
    truncation_nrmse_pct: float
    curve_rmse: float
    curve_nrmse_pct: float
    per_point_coverage95: float
    metric_scores: pd.DataFrame
    predictions: pd.DataFrame

    def summary_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "quantity": "POD truncation RMSE (mA/cm2)",
                    "value": self.truncation_rmse,
                    "note": f"{self.n_modes} modes, no GP involved",
                },
                {
                    "quantity": "POD truncation NRMSE (% of Jsc)",
                    "value": self.truncation_nrmse_pct,
                    "note": "floor on achievable curve accuracy",
                },
                {
                    "quantity": "End-to-end curve RMSE (mA/cm2)",
                    "value": self.curve_rmse,
                    "note": "leave-one-run-out, POD + GP",
                },
                {
                    "quantity": "End-to-end curve NRMSE (% of Jsc)",
                    "value": self.curve_nrmse_pct,
                    "note": "leave-one-run-out, POD + GP",
                },
                {
                    "quantity": "Per-point 95% coverage",
                    "value": self.per_point_coverage95,
                    "note": "share of held-out points inside the band",
                },
            ]
        )


def truncation_error(bundle: DoeBundle, basis: PodBasis) -> tuple[float, float]:
    """How much the basis alone loses, before any regression error."""
    recon = basis.reconstruct(basis.project(bundle.curve_matrix))
    resid = recon - bundle.curve_matrix
    rmse = float(np.sqrt(np.mean(resid**2)))
    jsc_scale = float(np.mean(np.abs(bundle.curve_matrix[:, 0])))
    return rmse, 100.0 * rmse / jsc_scale if jsc_scale > 0 else float("nan")


def cross_validate(
    bundle: DoeBundle | None = None,
    config: GPConfig = DEFAULT_CONFIG,
    energy_target: float = ENERGY_TARGET,
    max_modes: int = MAX_MODES,
    n_jobs: int | None = None,
    verbose: bool = False,
) -> JvValidation:
    """Leave-one-run-out validation of the whole curve pipeline.

    The POD basis is rebuilt inside every fold. Fitting the basis on all runs
    and then testing on a held-out one would leak that run's curve shape into
    the basis -- a subtle form of the same mistake as an ungrouped split.
    """
    # Imported here rather than at module scope: this is the only function that
    # needs it, and keeping it out of the import chain means the inference path
    # (predict -> jv_pod) does not depend on sklearn.model_selection at all.
    from sklearn.model_selection import LeaveOneGroupOut

    bundle = bundle if bundle is not None else load_doe()
    X = bundle.feature_matrix()
    curves = bundle.curve_matrix
    groups = bundle.groups()
    voltage = bundle.voltage
    illum = bundle.metrics["aging_light_suns"].to_numpy(dtype=float)

    splitter = LeaveOneGroupOut()
    folds = list(splitter.split(X, groups=groups))

    def run_fold(train_idx: np.ndarray, test_idx: np.ndarray):
        basis = build_basis(curves[train_idx], voltage, energy_target, max_modes)
        coefs = basis.project(curves[train_idx])
        models = [
            fit_target(X[train_idx], coefs[:, i], f"pod_mode_{i + 1}", config)
            for i in range(basis.n_modes)
        ]
        means, stds = [], []
        for model in models:
            m, s = model.predict(X[test_idx], return_std=True)
            means.append(m)
            stds.append(s)
        coef_mean = np.column_stack(means)
        coef_std = np.column_stack(stds)
        pred = basis.reconstruct(coef_mean)
        var = (coef_std**2) @ (basis.modes**2)
        return pred, np.sqrt(np.maximum(var, 0.0)), basis.n_modes

    if n_jobs is not None and _joblib is not None:
        results = _joblib.Parallel(n_jobs=n_jobs, verbose=5 if verbose else 0)(
            _joblib.delayed(run_fold)(tr, te) for tr, te in folds
        )
    else:
        results = [run_fold(tr, te) for tr, te in folds]

    pred_all = np.full_like(curves, np.nan)
    std_all = np.full_like(curves, np.nan)
    mode_counts = []
    for (_tr, te), (pred, std, k) in zip(folds, results):
        pred_all[te] = pred
        std_all[te] = std
        mode_counts.append(k)

    resid = pred_all - curves
    curve_rmse = float(np.sqrt(np.nanmean(resid**2)))
    jsc_scale = float(np.mean(np.abs(curves[:, 0])))
    curve_nrmse = 100.0 * curve_rmse / jsc_scale if jsc_scale > 0 else float("nan")
    coverage = float(np.nanmean(np.abs(resid) <= 1.959963985 * std_all))

    # Metrics recovered from the predicted curve, compared with the truth.
    rows = []
    for i in range(curves.shape[0]):
        true_m = curve_metrics(voltage, curves[i])
        pred_m = curve_metrics(voltage, pred_all[i])
        rows.append(
            {
                "run_id": bundle.metrics["run_id"].iloc[i],
                "aging_h": bundle.metrics["aging_h"].iloc[i],
                "aging_light_suns": illum[i],
                "aging_temperature_C": bundle.metrics["aging_temperature_C"].iloc[i],
                **{f"{k}_true": v for k, v in true_m.as_dict().items() if isinstance(v, float)},
                **{f"{k}_pred": v for k, v in pred_m.as_dict().items() if isinstance(v, float)},
                "pred_valid": pred_m.valid,
            }
        )
    predictions = pd.DataFrame(rows)

    scores = []
    for metric in ("Jsc_mAcm2", "Voc_V", "FF", "PCE_pct"):
        t = predictions[f"{metric}_true"].to_numpy(dtype=float)
        p = predictions[f"{metric}_pred"].to_numpy(dtype=float)
        ok = np.isfinite(t) & np.isfinite(p)
        err = p[ok] - t[ok]
        ss_res = float(np.sum(err**2))
        ss_tot = float(np.sum((t[ok] - t[ok].mean()) ** 2))
        scores.append(
            {
                "metric": metric,
                "n": int(ok.sum()),
                "mae": float(np.mean(np.abs(err))),
                "rmse": float(np.sqrt(np.mean(err**2))),
                "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
                "source": "extracted from predicted curve",
            }
        )

    trunc_rmse, trunc_nrmse = truncation_error(bundle, build_basis(curves, voltage, energy_target, max_modes))

    return JvValidation(
        n_folds=len(folds),
        n_modes=int(np.median(mode_counts)),
        explained_variance_ratio=build_basis(curves, voltage, energy_target, max_modes).explained_variance_ratio,
        truncation_rmse=trunc_rmse,
        truncation_nrmse_pct=trunc_nrmse,
        curve_rmse=curve_rmse,
        curve_nrmse_pct=curve_nrmse,
        per_point_coverage95=coverage,
        metric_scores=pd.DataFrame(scores),
        predictions=predictions,
    )


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------
def save(surrogate: JvSurrogate, path: Path | str = DEFAULT_ARTIFACT) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if _joblib is not None:
        _joblib.dump(surrogate, path, compress=3)
    else:  # pragma: no cover
        with path.open("wb") as fh:
            pickle.dump(surrogate, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load(path: Path | str = DEFAULT_ARTIFACT) -> JvSurrogate:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No J-V surrogate at {path}. Run scripts/train_models.py first."
        )
    if _joblib is not None:
        return _joblib.load(path)
    with path.open("rb") as fh:  # pragma: no cover
        return pickle.load(fh)


def mode_table(basis: PodBasis) -> pd.DataFrame:
    """Human-readable description of the retained modes."""
    return pd.DataFrame(
        {
            "mode": [f"Mode {i + 1}" for i in range(basis.n_modes)],
            "singular_value": basis.singular_values,
            "explained_variance_ratio": basis.explained_variance_ratio,
            "cumulative": np.cumsum(basis.explained_variance_ratio),
        }
    )
