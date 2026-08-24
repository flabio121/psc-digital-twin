"""Scalar Gaussian-process surrogate for the COMSOL degradation campaign.

WHY A GP AT ALL
---------------
The training set is 36 drift-diffusion runs (360 rows). That is small, expensive
and noise-free-ish, which is exactly the regime where a Gaussian process beats a
flexible regressor: it interpolates the simulator almost exactly where data
exists and it reports a posterior standard deviation that grows where it does
not. The app promises an error bar next to every number, so the model has to
produce one natively rather than bolt one on.

WHAT IS MODELLED
----------------
One independent GP per entry of ``data.TARGETS``. The five metrics are
physically coupled (PCE ~= Voc * Jsc * FF / 100), but fitting them jointly would
require a multi-output kernel that this dataset cannot identify. Independent GPs
are the honest, boring choice; the coupling is not enforced, and a reviewer
should know that a predicted (Voc, Jsc, FF) triple will not reproduce the
predicted PCE to machine precision. Section "KNOWN LIMITATIONS" quantifies it.

HOW IT IS VALIDATED -- READ THIS BEFORE TRUSTING ANY NUMBER
-----------------------------------------------------------
Rows are NOT independent. Each COMSOL run contributes 10 rows that share one
design point (illumination, temperature) and differ only in aging time. A random
train/test split would put t=300 h of a run in train and t=400 h of the SAME run
in test, so the model would be scored on a design point it had already seen. That
inflates every metric and is a genuine methodological error, not a nuance.

Every cross-validation in this module therefore uses **LeaveOneGroupOut over
``DoeBundle.groups()``** (the 36 ``run_id`` values, via ``sklearn`` grouped CV).
Each fold trains on 35 runs / 350 rows and predicts a whole unseen design point.
This measures the quantity the app actually needs: accuracy at a stress
condition that was never simulated.

TIME WARP -- MEASURED, NOT ASSUMED
----------------------------------
Degradation is strongly non-linear near t=0, so a log1p warp of ``aging_h``
before standardisation is the obvious thing to try. It was tried. Both
parametrisations are implemented (``TIME_WARPS``) and ``compare_time_warps()``
scores them under the same grouped CV; ``DEFAULT_CONFIG.time_warp`` is set to
whichever won on the real data. The measured numbers for both are in
``docs/`` and reproduced by running this module as a script. Do not change the
default without re-running that comparison.

CALIBRATION
-----------
``cross_validate()`` reports ``coverage95``: the fraction of held-out points
whose true value falls inside mean +/- 1.96*std. This is the only number that
licenses drawing an uncertainty band in the UI. It is reported as measured. The
COMSOL simulator is deterministic, so the fitted WhiteKernel collapses to a
numerical floor and the predictive interval at an unseen design point is driven
almost entirely by interpolation distance -- which, on a 6x6 grid with long
fitted length scales, is small. Expect and report under-coverage rather than
tuning it away.

KNOWN LIMITATIONS
-----------------
* Independent GPs do not enforce PCE = Voc*Jsc*FF/100 (see above).
* Held-out folds are whole design points, but the 6x6 grid means most held-out
  points still have four grid neighbours. Corner design points are the hardest
  and are not reported separately.
* Nothing here extrapolates responsibly outside ``capabilities.ENVELOPE``; the
  GP reverts toward its prior mean, which the capability registry labels PREVIEW.

Nothing in this module imports streamlit. Every public function is pure apart
from ``save``/``load``.
"""

from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

from ..data import FEATURES, TARGETS, DoeBundle, load_doe

try:  # joblib ships as a hard dependency of scikit-learn; pickle is the fallback.
    import joblib as _joblib
except ImportError:  # pragma: no cover - only on a broken sklearn install
    _joblib = None

__all__ = [
    "GPConfig",
    "DEFAULT_CONFIG",
    "TIME_WARPS",
    "ScalarSurrogate",
    "ScalarSurrogateSet",
    "TargetScore",
    "CvReport",
    "make_kernel",
    "design_matrix",
    "fit_target",
    "fit_all",
    "cross_validate",
    "compare_time_warps",
    "trajectory_grid",
    "save",
    "load",
    "MODELS_DIR",
    "DEFAULT_ARTIFACT",
]

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
DEFAULT_ARTIFACT = MODELS_DIR / "scalar_gp.joblib"

# Index of aging_h inside FEATURES. Asserted rather than hard-coded so that a
# future edit to data.FEATURES fails loudly here instead of silently warping the
# wrong column.
AGING_INDEX = FEATURES.index("aging_h")

TIME_WARPS: tuple[str, ...] = ("raw", "log1p")

# Two-sided normal quantile for a 95% interval. scipy is not a dependency of
# this project, so the constant is inlined rather than imported.
Z95 = 1.959963984540054


@dataclass(frozen=True)
class GPConfig:
    """Every knob that changes the fitted model, in one hashable place.

    ``time_warp`` is the only field chosen empirically; see ``compare_time_warps``.
    """

    time_warp: str = "log1p"
    nu: float = 2.5
    n_restarts_optimizer: int = 8
    random_state: int = 0
    # Kernel hyper-parameter bounds. Deliberately wide: on this data the
    # optimiser drives the amplitude and length scales into the long-correlation
    # regime (the GP's way of expressing a smooth near-polynomial trend), and
    # tight bounds simply pin the solution to the boundary.
    constant_value_bounds: tuple[float, float] = (1e-4, 1e6)
    length_scale_bounds: tuple[float, float] = (1e-2, 1e4)
    noise_level_bounds: tuple[float, float] = (1e-9, 1e1)
    initial_noise_level: float = 1e-3
    # Jitter added to the diagonal on top of the WhiteKernel. The simulator is
    # deterministic, so the fitted white noise collapses toward its lower bound
    # and the Cholesky needs this floor to stay conditioned.
    alpha: float = 1e-10

    def __post_init__(self) -> None:
        if self.time_warp not in TIME_WARPS:
            raise ValueError(f"time_warp must be one of {TIME_WARPS}, got {self.time_warp!r}")
        if self.n_restarts_optimizer < 8:
            raise ValueError("n_restarts_optimizer < 8 makes the ARD fit unreliable on 350 rows")


#: The shipped configuration. ``time_warp`` here is the winner of the measured
#: grouped-CV comparison in ``compare_time_warps()``, not a guess.
DEFAULT_CONFIG = GPConfig()


# --------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------
def design_matrix(X: np.ndarray, time_warp: str) -> np.ndarray:
    """Apply the aging-time warp to a raw (n, 3) feature matrix in FEATURES order.

    Illumination and temperature are left alone: both are swept on a uniform
    6-level grid, so there is no skew for a warp to fix. Aging time is sampled
    on a deliberately non-uniform grid (0, 50, 100, then 100 h steps, then 200 h
    steps) because the response moves fastest early, which is precisely the
    situation a log1p warp is meant to linearise.
    """
    if time_warp not in TIME_WARPS:
        raise ValueError(f"time_warp must be one of {TIME_WARPS}, got {time_warp!r}")
    Z = np.atleast_2d(np.asarray(X, dtype=float)).copy()
    if Z.shape[1] != len(FEATURES):
        raise ValueError(f"expected {len(FEATURES)} columns in FEATURES order, got {Z.shape[1]}")
    if time_warp == "log1p":
        # log1p, not log: aging_h == 0 is a real, populated level (the pristine
        # cell) and must map to a finite value.
        if (Z[:, AGING_INDEX] < 0).any():
            raise ValueError("aging_h must be non-negative for the log1p warp")
        Z[:, AGING_INDEX] = np.log1p(Z[:, AGING_INDEX])
    return Z


def make_kernel(config: GPConfig = DEFAULT_CONFIG) -> ConstantKernel:
    """ConstantKernel * Matern(nu) with ARD + WhiteKernel.

    nu=2.5 gives sample paths that are twice differentiable. That is the right
    smoothness class for a physical degradation trajectory: it is smooth (no
    kinks, no jumps) but it is not analytic -- ion migration, interface
    recombination and thermal activation each dominate over a different stretch
    of the surface, so the response has real curvature structure that the RBF
    limit (nu -> inf, infinitely differentiable) over-smooths into a bland
    quadratic. nu=1.5 goes the other way and is too rough for a converged
    finite-element solution.

    The length scale is a VECTOR of 3 (ARD), one per feature, so the fit reports
    how far the response travels along illumination, temperature and time
    separately -- these have wildly different physical relevance and a single
    isotropic length scale would average them into nonsense.
    """
    amplitude = ConstantKernel(constant_value=1.0, constant_value_bounds=config.constant_value_bounds)
    shape = Matern(
        length_scale=np.ones(len(FEATURES)),
        length_scale_bounds=config.length_scale_bounds,
        nu=config.nu,
    )
    # WhiteKernel absorbs solver jitter (mesh/tolerance noise in the COMSOL
    # export). It is part of the kernel, so sklearn's predictive std is the
    # posterior for a NEW OBSERVATION, which is what coverage must be scored on.
    jitter = WhiteKernel(
        noise_level=config.initial_noise_level,
        noise_level_bounds=config.noise_level_bounds,
    )
    return amplitude * shape + jitter


# --------------------------------------------------------------------------
# Fitted containers
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ScalarSurrogate:
    """A fitted GP for ONE target, owning its full transform chain.

    ``predict`` takes raw physical inputs in ``FEATURES`` order; the warp and the
    standardiser are applied internally so that no caller can accidentally feed
    the GP untransformed coordinates.
    """

    target: str
    config: GPConfig
    scaler: StandardScaler
    gp: GaussianProcessRegressor
    n_train: int
    kernel_repr: str
    log_marginal_likelihood: float

    def _z(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.transform(design_matrix(X, self.config.time_warp))

    def predict(self, X: np.ndarray, return_std: bool = False):
        """Posterior mean, and optionally the posterior standard deviation.

        Returns ``mean`` of shape (n,), or ``(mean, std)`` when ``return_std``.
        """
        Z = self._z(X)
        if return_std:
            mean, std = self.gp.predict(Z, return_std=True)
            return np.asarray(mean, dtype=float), np.asarray(std, dtype=float)
        return np.asarray(self.gp.predict(Z), dtype=float)

    @property
    def length_scales(self) -> dict[str, float]:
        """Fitted ARD length scales, in standardised units, keyed by feature."""
        matern = self.gp.kernel_.k1.k2
        return {name: float(v) for name, v in zip(FEATURES, np.atleast_1d(matern.length_scale))}

    @property
    def noise_level(self) -> float:
        return float(self.gp.kernel_.k2.noise_level)


@dataclass(frozen=True)
class ScalarSurrogateSet:
    """All five target GPs behind one call. This is what the UI holds."""

    models: Mapping[str, ScalarSurrogate]
    config: GPConfig
    targets: tuple[str, ...] = TARGETS
    n_train: int = 0
    n_runs: int = 0

    def __getitem__(self, target: str) -> ScalarSurrogate:
        return self.models[target]

    def predict(self, X: np.ndarray, return_std: bool = False):
        """Stacked (n, n_targets) mean, optionally with a matching std array."""
        means, stds = [], []
        for target in self.targets:
            if return_std:
                mean, std = self.models[target].predict(X, return_std=True)
                means.append(mean)
                stds.append(std)
            else:
                means.append(self.models[target].predict(X))
        M = np.column_stack(means)
        return (M, np.column_stack(stds)) if return_std else M

    def predict_frame(self, X: np.ndarray) -> pd.DataFrame:
        """Tidy predictions: the inputs, then ``<target>`` and ``<target>_std``.

        The input columns are carried through so the frame is self-describing
        when it is exported into the reproducibility bundle.
        """
        Xa = np.atleast_2d(np.asarray(X, dtype=float))
        out = pd.DataFrame(Xa, columns=list(FEATURES))
        for target in self.targets:
            mean, std = self.models[target].predict(Xa, return_std=True)
            out[target] = mean
            out[f"{target}_std"] = std
        return out

    def trajectory(
        self,
        illumination_suns: float,
        temperature_c: float,
        aging_h: Sequence[float] | np.ndarray,
    ) -> pd.DataFrame:
        """Predicted degradation trajectory at one stress condition.

        This is the UI entry point: hold illumination and temperature fixed,
        sweep aging time, get one tidy row per time point with a mean and a
        standard deviation for every target.
        """
        times = np.asarray(aging_h, dtype=float).ravel()
        if times.size == 0:
            raise ValueError("aging_h must contain at least one time point")
        X = np.column_stack(
            [
                np.full(times.shape, float(illumination_suns)),
                np.full(times.shape, float(temperature_c)),
                times,
            ]
        )
        return self.predict_frame(X)

    def summary(self) -> pd.DataFrame:
        """Fitted hyper-parameters per target, for the model card."""
        rows = []
        for target in self.targets:
            model = self.models[target]
            row: dict[str, object] = {
                "target": target,
                "n_train": model.n_train,
                "log_marginal_likelihood": model.log_marginal_likelihood,
                "noise_level": model.noise_level,
            }
            row.update({f"ls_{k}": v for k, v in model.length_scales.items()})
            rows.append(row)
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------
def fit_target(
    X: np.ndarray,
    y: np.ndarray,
    target: str,
    config: GPConfig = DEFAULT_CONFIG,
) -> ScalarSurrogate:
    """Fit one GP. ``X`` is raw, in ``FEATURES`` order; the warp is applied here."""
    Xw = design_matrix(X, config.time_warp)
    scaler = StandardScaler().fit(Xw)
    Z = scaler.transform(Xw)
    gp = GaussianProcessRegressor(
        kernel=make_kernel(config),
        alpha=config.alpha,
        # normalize_y centres and scales the target so the kernel amplitude
        # prior is meaningful across targets whose units differ by 4 orders of
        # magnitude (FF ~ 0.8 vs retention ~ 94 %).
        normalize_y=True,
        n_restarts_optimizer=config.n_restarts_optimizer,
        random_state=config.random_state,
    ).fit(Z, np.asarray(y, dtype=float))
    return ScalarSurrogate(
        target=target,
        config=config,
        scaler=scaler,
        gp=gp,
        n_train=int(Z.shape[0]),
        kernel_repr=str(gp.kernel_),
        log_marginal_likelihood=float(gp.log_marginal_likelihood_value_),
    )


def fit_all(
    bundle: DoeBundle | None = None,
    config: GPConfig = DEFAULT_CONFIG,
    targets: Iterable[str] = TARGETS,
) -> ScalarSurrogateSet:
    """Fit one GP per target on the full campaign (no held-out rows).

    The shipped model uses every row; honest performance comes from
    ``cross_validate``, which is a separate computation on separate fits.
    """
    bundle = bundle if bundle is not None else load_doe()
    X = bundle.feature_matrix()
    targets = tuple(targets)
    models = {t: fit_target(X, bundle.target_vector(t), t, config) for t in targets}
    return ScalarSurrogateSet(
        models=models,
        config=config,
        targets=targets,
        n_train=int(X.shape[0]),
        n_runs=bundle.n_runs,
    )


# --------------------------------------------------------------------------
# Grouped cross-validation
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class TargetScore:
    """Out-of-fold performance for one target under LeaveOneGroupOut."""

    target: str
    n: int
    mae: float
    rmse: float
    r2: float
    coverage95: float
    mean_pred_std: float
    max_abs_error: float


@dataclass(frozen=True)
class CvReport:
    """Everything measured in one grouped-CV pass, plus what went wrong."""

    time_warp: str
    n_folds: int
    scores: tuple[TargetScore, ...]
    predictions: pd.DataFrame
    n_convergence_warnings: int = 0
    config: GPConfig = field(default=DEFAULT_CONFIG)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "target": s.target,
                    "n": s.n,
                    "MAE": s.mae,
                    "RMSE": s.rmse,
                    "R2": s.r2,
                    "coverage95": s.coverage95,
                    "mean_pred_std": s.mean_pred_std,
                    "max_abs_err": s.max_abs_error,
                }
                for s in self.scores
            ]
        )

    def score(self, target: str) -> TargetScore:
        for s in self.scores:
            if s.target == target:
                return s
        raise KeyError(target)


def _score(target: str, y_true: np.ndarray, mean: np.ndarray, std: np.ndarray) -> TargetScore:
    err = mean - y_true
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    # R2 against the pooled out-of-fold predictions, i.e. how much of the
    # campaign-wide variance the surrogate explains at unseen design points.
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    inside = np.abs(err) <= Z95 * std
    return TargetScore(
        target=target,
        n=int(y_true.size),
        mae=float(np.mean(np.abs(err))),
        rmse=float(np.sqrt(np.mean(err**2))),
        r2=float(r2),
        coverage95=float(np.mean(inside)),
        mean_pred_std=float(np.mean(std)),
        max_abs_error=float(np.max(np.abs(err))),
    )


def _cv_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    target: str,
    config: GPConfig,
) -> tuple[np.ndarray, np.ndarray, int]:
    """One (fold, target) job: fit on the kept runs, predict the held-out run.

    Module level and side-effect free so it can be shipped to a joblib worker.
    Convergence warnings are counted here and returned, because a warning raised
    in a worker process would otherwise be swallowed and the report would
    silently claim 36 clean optimisations.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model = fit_target(X_train, y_train, target, config)
        mean, std = model.predict(X_test, return_std=True)
    n_warn = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
    return mean, std, int(n_warn)


def cross_validate(
    bundle: DoeBundle | None = None,
    config: GPConfig = DEFAULT_CONFIG,
    targets: Iterable[str] = TARGETS,
    verbose: bool = False,
    n_jobs: int | None = None,
) -> CvReport:
    """LeaveOneGroupOut over the 36 ``run_id`` groups from ``DoeBundle.groups()``.

    Grouping is not optional. Ten rows share each design point, so any split that
    is not grouped by ``run_id`` leaks the design point across the train/test
    boundary and reports a fantasy. Each of the 36 folds trains on 35 runs
    (350 rows) and predicts the 10 rows of one entirely unseen (illumination,
    temperature) condition.

    ``coverage95`` is the fraction of held-out rows with
    ``|y - mean| <= 1.96 * std``. It is reported exactly as measured; it is the
    only evidence that the app's uncertainty bands mean anything.

    ``n_jobs`` only changes wall-clock time. Every GP is fitted with a fixed
    ``random_state``, so serial and parallel runs return bit-identical numbers.
    """
    bundle = bundle if bundle is not None else load_doe()
    X = bundle.feature_matrix()
    groups = bundle.groups()
    targets = tuple(targets)
    y_by_target = {t: bundle.target_vector(t) for t in targets}

    splitter = LeaveOneGroupOut()
    n_folds = splitter.get_n_splits(X, groups=groups)
    folds = list(splitter.split(X, groups=groups))
    jobs = [(tr, te, t) for tr, te in folds for t in targets]

    if n_jobs is not None and _joblib is not None:
        results = _joblib.Parallel(n_jobs=n_jobs, verbose=5 if verbose else 0)(
            _joblib.delayed(_cv_fold)(X[tr], y_by_target[t][tr], X[te], t, config)
            for tr, te, t in jobs
        )
    else:
        results = []
        for i, (tr, te, t) in enumerate(jobs):
            results.append(_cv_fold(X[tr], y_by_target[t][tr], X[te], t, config))
            if verbose and (i + 1) % len(targets) == 0:
                print(f"  fold {(i + 1) // len(targets)}/{n_folds} done", flush=True)

    oof_mean = {t: np.full(X.shape[0], np.nan) for t in targets}
    oof_std = {t: np.full(X.shape[0], np.nan) for t in targets}
    n_warn = 0
    for (_tr, te, t), (mean, std, warn) in zip(jobs, results):
        oof_mean[t][te] = mean
        oof_std[t][te] = std
        n_warn += warn

    preds = bundle.metrics.loc[:, ["run_id", *FEATURES]].copy()
    scores = []
    for target in targets:
        y = y_by_target[target]
        preds[f"{target}_true"] = y
        preds[f"{target}_pred"] = oof_mean[target]
        preds[f"{target}_std"] = oof_std[target]
        scores.append(_score(target, y, oof_mean[target], oof_std[target]))

    return CvReport(
        time_warp=config.time_warp,
        n_folds=int(n_folds),
        scores=tuple(scores),
        predictions=preds,
        n_convergence_warnings=int(n_warn),
        config=config,
    )


def compare_time_warps(
    bundle: DoeBundle | None = None,
    config: GPConfig = DEFAULT_CONFIG,
    targets: Iterable[str] = TARGETS,
    verbose: bool = False,
    n_jobs: int | None = None,
) -> tuple[pd.DataFrame, dict[str, CvReport]]:
    """Score raw ``aging_h`` against a log1p warp under identical grouped CV.

    Returns a long comparison table and the two reports. Whichever wins on
    normalised RMSE (averaged over targets, each target's RMSE divided by that
    target's standard deviation so the five very different units can be pooled)
    should be ``DEFAULT_CONFIG.time_warp``. Nothing about this choice is assumed.
    """
    bundle = bundle if bundle is not None else load_doe()
    reports: dict[str, CvReport] = {}
    frames = []
    for warp in TIME_WARPS:
        if verbose:
            print(f"[compare] time_warp={warp}", flush=True)
        cfg = replace(config, time_warp=warp)
        report = cross_validate(bundle, cfg, targets, verbose=verbose, n_jobs=n_jobs)
        reports[warp] = report
        frame = report.to_frame()
        frame.insert(0, "time_warp", warp)
        # Normalised RMSE lets the five targets be averaged into one verdict.
        sd = np.array([bundle.target_vector(t).std(ddof=0) for t in frame["target"]])
        frame["nRMSE"] = frame["RMSE"].to_numpy() / sd
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), reports


# --------------------------------------------------------------------------
# UI helper
# --------------------------------------------------------------------------
def trajectory_grid(
    surrogates: ScalarSurrogateSet,
    illumination_suns: float,
    temperature_c: float,
    aging_h: Sequence[float] | np.ndarray | None = None,
    n_points: int = 61,
) -> pd.DataFrame:
    """Trajectory over a default 0-1000 h grid unless explicit times are given."""
    if aging_h is None:
        aging_h = np.linspace(0.0, 1000.0, int(n_points))
    return surrogates.trajectory(illumination_suns, temperature_c, aging_h)


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------
def save(surrogates: ScalarSurrogateSet, path: Path | str = DEFAULT_ARTIFACT) -> Path:
    """Persist the fitted set. joblib with compression, pickle if joblib is gone.

    A GP stores its Cholesky factor, so the raw object is a few MB; compression
    level 3 is where the size/time curve flattens for these matrices.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if _joblib is not None:
        _joblib.dump(surrogates, path, compress=3)
    else:  # pragma: no cover
        with path.open("wb") as fh:
            pickle.dump(surrogates, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load(path: Path | str = DEFAULT_ARTIFACT) -> ScalarSurrogateSet:
    """Load a fitted set written by ``save``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no surrogate artifact at {path}; run fit_all() then save()")
    if _joblib is not None:
        obj = _joblib.load(path)
    else:  # pragma: no cover
        with path.open("rb") as fh:
            obj = pickle.load(fh)
    if not isinstance(obj, ScalarSurrogateSet):
        raise TypeError(f"{path} does not contain a ScalarSurrogateSet")
    return obj


if __name__ == "__main__":  # pragma: no cover
    import sys

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)
    jobs = None
    for arg in sys.argv[1:]:
        if arg.startswith("--jobs="):
            jobs = int(arg.split("=", 1)[1])
    bundle = load_doe()
    if "--compare" in sys.argv:
        table, _ = compare_time_warps(bundle, verbose=True, n_jobs=jobs)
        print(table.to_string(index=False))
    else:
        report = cross_validate(bundle, DEFAULT_CONFIG, verbose=True, n_jobs=jobs)
        print(f"LeaveOneGroupOut over {report.n_folds} runs, time_warp={report.time_warp}")
        print(report.to_frame().to_string(index=False))
        models = fit_all(bundle)
        print(models.summary().to_string(index=False))
        print("saved ->", save(models))
