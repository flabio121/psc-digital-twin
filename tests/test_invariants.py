"""The invariants this product would be broken without.

These are not coverage-chasing unit tests. Each one guards a specific claim the
app makes to its users, and each would fail loudly if a refactor quietly
weakened that claim.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from psc_twin import activelearn, capabilities, climate, heuristic, lifetime
from psc_twin.capabilities import Tier
from psc_twin.data import FEATURES, TARGETS, load_doe
from psc_twin.materials import BASELINE_MATERIALS, LAYERS, is_baseline_design
from psc_twin.surrogate import jv_pod

MODELS_READY = (
    (ROOT / "models" / "scalar_gp.joblib").exists()
    and (ROOT / "models" / "jv_pod.joblib").exists()
)
needs_models = pytest.mark.skipif(not MODELS_READY, reason="run scripts/train_models.py first")


@pytest.fixture(scope="module")
def bundle():
    return load_doe()


# --------------------------------------------------------------------------
# The central product rule
# --------------------------------------------------------------------------
class TestPlannedNeverAnswers:
    """A Tier.PLANNED capability must never yield a number, by any route."""

    @needs_models
    @pytest.mark.parametrize("architecture", ["n-i-p", "tandem", "arch_nip", "arch_tandem"])
    def test_planned_architecture_raises(self, architecture):
        from psc_twin.surrogate import predict

        with pytest.raises(predict.PlannedCapabilityError):
            predict.predict(1.0, 85.0, 1000.0, architecture=architecture)

    def test_planned_capabilities_declare_blocking(self):
        for cap in capabilities.by_tier(Tier.PLANNED):
            assert cap.blocks_numbers, f"{cap.key} is planned but does not block numbers"

    def test_validated_capabilities_do_not_block(self):
        for cap in capabilities.by_tier(Tier.VALIDATED):
            assert not cap.blocks_numbers

    def test_every_planned_capability_explains_itself(self):
        """A roadmap card with no explanation is worse than no card."""
        for cap in capabilities.by_tier(Tier.PLANNED):
            assert cap.backing.strip(), f"{cap.key} has no backing text"
            assert cap.unlocks.strip(), f"{cap.key} does not say what would unlock it"

    def test_only_the_thesis_material_stack_is_prediction_ready(self):
        assert is_baseline_design(BASELINE_MATERIALS)
        for layer in LAYERS:
            custom = dict(BASELINE_MATERIALS)
            custom[layer.key] = layer.alternatives[0]
            assert not is_baseline_design(custom), f"{layer.label} alternative did not lock results"

    def test_optional_barriers_default_off_and_lock_when_enabled(self):
        assert BASELINE_MATERIALS["front_barrier"] == "None"
        assert BASELINE_MATERIALS["rear_barrier"] == "None"
        custom = dict(BASELINE_MATERIALS)
        custom["front_barrier"] = "Al2O3"
        assert not is_baseline_design(custom)
        assert capabilities.get("barrier_layers").tier is Tier.PLANNED

    def test_internal_interlayers_default_off_and_lock_when_enabled(self):
        assert BASELINE_MATERIALS["htl_absorber_barrier"] == "None"
        assert BASELINE_MATERIALS["absorber_etl_barrier"] == "None"
        custom = dict(BASELINE_MATERIALS)
        custom["htl_absorber_barrier"] = "SAM + Al2O3 bilayer"
        assert not is_baseline_design(custom)
        assert capabilities.get("internal_interlayers").tier is Tier.PLANNED


# --------------------------------------------------------------------------
# Validation methodology
# --------------------------------------------------------------------------
class TestNoLeakage:
    """Rows sharing a design point must never straddle a train/test split."""

    def test_ten_rows_per_design_point(self, bundle):
        counts = bundle.metrics.groupby("run_id").size()
        assert set(counts.unique()) == {10}, "expected 10 aging times per run"

    def test_grouped_split_keeps_runs_intact(self, bundle):
        from sklearn.model_selection import LeaveOneGroupOut

        X = bundle.feature_matrix()
        groups = bundle.groups()
        for train_idx, test_idx in LeaveOneGroupOut().split(X, groups=groups):
            train_runs = set(groups[train_idx])
            test_runs = set(groups[test_idx])
            assert not (train_runs & test_runs), "a run appeared on both sides"
            assert len(test_runs) == 1

    def test_design_is_full_factorial(self, bundle):
        summary = activelearn.coverage_summary(bundle)
        assert summary["n_design_points"] == 36
        assert summary["is_full_factorial"]


# --------------------------------------------------------------------------
# Physics extraction
# --------------------------------------------------------------------------
class TestCurveMetrics:
    """Our J-V extraction must reproduce the campaign's own numbers."""

    def test_matches_campaign_metrics(self, bundle):
        metrics = bundle.metrics
        worst = {"PCE_pct": 0.0, "Voc_V": 0.0, "Jsc_mAcm2": 0.0, "FF": 0.0}
        for i in range(len(metrics)):
            m = jv_pod.curve_metrics(bundle.voltage, bundle.curve_matrix[i])
            worst["PCE_pct"] = max(worst["PCE_pct"], abs(m.pce_pct - metrics["PCE_pct"].iloc[i]))
            worst["Voc_V"] = max(worst["Voc_V"], abs(m.voc_v - metrics["Voc_V"].iloc[i]))
            worst["Jsc_mAcm2"] = max(worst["Jsc_mAcm2"], abs(m.jsc_macm2 - metrics["Jsc_mAcm2"].iloc[i]))
            worst["FF"] = max(worst["FF"], abs(m.ff - metrics["FF"].iloc[i]))
        for key, value in worst.items():
            assert value < 1e-9, f"{key} drifted from the campaign by {value:g}"

    def test_efficiency_uses_fixed_one_sun_reference(self, bundle):
        """Aging illumination must not enter the efficiency denominator.

        Every diagnostic sweep is taken at 1 sun regardless of aging condition.
        Scaling by aging illumination inflates PCE up to 100x at the dim corner,
        which is exactly the bug this test exists to prevent recurring.
        """
        dim = bundle.metrics["aging_light_suns"] == 0.01
        idx = int(np.flatnonzero(dim.to_numpy())[0])
        m = jv_pod.curve_metrics(bundle.voltage, bundle.curve_matrix[idx])
        assert m.pce_pct == pytest.approx(bundle.metrics["PCE_pct"].iloc[idx], abs=1e-9)
        assert m.pce_pct < 30.0, "efficiency above 30% means the reference is wrong"

    def test_no_zero_crossing_is_flagged_not_guessed(self):
        v = np.linspace(0.0, 1.5, 72)
        j = np.full_like(v, 5.0)  # never crosses zero
        m = jv_pod.curve_metrics(v, j)
        assert not m.valid
        assert np.isnan(m.voc_v), "a missing Voc must be NaN, not a plausible number"


class TestPodBasis:
    def test_reconstruction_is_accurate(self, bundle):
        basis = jv_pod.build_basis(bundle.curve_matrix, bundle.voltage)
        rmse, nrmse = jv_pod.truncation_error(bundle, basis)
        assert nrmse < 0.1, f"POD truncation loses {nrmse:.3f}% of Jsc"

    def test_modes_are_orthonormal(self, bundle):
        basis = jv_pod.build_basis(bundle.curve_matrix, bundle.voltage)
        gram = basis.modes @ basis.modes.T
        np.testing.assert_allclose(gram, np.eye(basis.n_modes), atol=1e-10)

    def test_project_reconstruct_roundtrip(self, bundle):
        """Reconstruction must be tight where the physics is actually read.

        The worst residual sits at roughly 1.22 V -- in the knee near open
        circuit on the most degraded run, which is exactly where Voc and the
        fill factor are extracted. That is why ``build_basis`` truncates on
        operating-region error rather than on variance share alone, and why this
        test asserts the same contract.

        Errors are normalised by Jsc, the conventional scale for a J-V residual.
        Normalising pointwise by the local current would be meaningless: the
        curve passes through zero at Voc, so any absolute error there looks
        infinite in relative terms.
        """
        basis = jv_pod.build_basis(bundle.curve_matrix, bundle.voltage)
        recon = basis.reconstruct(basis.project(bundle.curve_matrix))
        assert recon.shape == bundle.curve_matrix.shape

        residual = np.abs(recon - bundle.curve_matrix)

        # The criterion build_basis() actually truncates on.
        operating = bundle.voltage <= jv_pod.OPERATING_VOLTAGE_MAX
        assert residual[:, operating].max() <= jv_pod.MAX_OPERATING_ERROR

        # Whole sweep, as a fraction of each curve's own short-circuit current.
        jsc = np.abs(bundle.curve_matrix[:, 0])[:, None]
        assert (residual / np.maximum(jsc, 1e-9)).max() < 0.01


# --------------------------------------------------------------------------
# Refusing to invent lifetimes
# --------------------------------------------------------------------------
class TestLifetime:
    def test_crossing_is_interpolated(self):
        t = np.linspace(0, 2000, 80)
        e = lifetime.t80(t, 100 * np.exp(-t / 1200))
        assert e.value_h is not None
        assert e.method == "interpolated"
        assert 200 < e.value_h < 400

    def test_never_reached_returns_none(self):
        """The single most important honesty behaviour in this module."""
        t = np.linspace(0, 1000, 50)
        e = lifetime.t80(t, 100 - 0.5 * t / 1000, allow_extrapolation=False)
        assert e.value_h is None
        assert "beyond" in e.note().lower() or "stays above" in e.note().lower()

    def test_note_is_plain_language(self):
        t = np.linspace(0, 2000, 80)
        note = lifetime.t80(t, 100 * np.exp(-t / 1200)).note()
        assert len(note) > 40
        assert "%" in note or "efficiency" in note.lower()

    def test_projection_is_labelled(self):
        t = np.linspace(0, 500, 40)
        e = lifetime.t80(t, 100 * np.exp(-t / 3000), allow_extrapolation=True)
        if e.value_h is not None:
            assert e.is_projection, "an out-of-horizon estimate must declare itself"


# --------------------------------------------------------------------------
# The design envelope
# --------------------------------------------------------------------------
class TestEnvelope:
    def test_inside_is_clean(self):
        assert capabilities.within_envelope(0.5, 70.0, 500.0)
        assert capabilities.envelope_excursions(0.5, 70.0, 500.0) == []

    @pytest.mark.parametrize(
        "args,expected",
        [((1.4, 70.0, 500.0), 1), ((1.4, 150.0, 500.0), 2), ((1.4, 150.0, 5000.0), 3)],
    )
    def test_excursions_are_counted(self, args, expected):
        assert len(capabilities.envelope_excursions(*args)) == expected

    def test_levels_lie_inside_declared_bounds(self):
        for axis, levels in capabilities.ENVELOPE_LEVELS.items():
            low, high = capabilities.ENVELOPE[axis]
            assert min(levels) >= low - 1e-9
            assert max(levels) <= high + 1e-9

    def test_envelope_matches_the_actual_data(self, bundle):
        """The declared envelope must not overstate what was simulated."""
        m = bundle.metrics
        low, high = capabilities.ENVELOPE["illumination_suns"]
        assert m["aging_light_suns"].min() >= low - 1e-9
        assert m["aging_light_suns"].max() <= high + 1e-9
        low, high = capabilities.ENVELOPE["aging_h"]
        assert m["aging_h"].max() <= high + 1e-9


# --------------------------------------------------------------------------
# Honesty of the unvalidated parts
# --------------------------------------------------------------------------
class TestHeuristicIsLabelled:
    def test_attribution_carries_its_own_warning(self):
        frame = heuristic.mechanism_attribution(1.0, 85.0, 500.0)
        assert frame["is_heuristic"].all()
        assert not frame["validated"].any()

    def test_weights_sum_to_one(self):
        frame = heuristic.mechanism_attribution(0.5, 60.0, 300.0)
        assert frame["weight"].sum() == pytest.approx(1.0)

    def test_mechanism_capability_is_preview(self):
        assert capabilities.get("out_mechanism").tier is Tier.PREVIEW


class TestClimateIsLabelled:
    def test_archetypes_declare_they_are_not_measurements(self):
        table = climate.archetype_table()
        blob = " ".join(str(v) for v in table.to_numpy().ravel()).lower()
        assert any(
            word in blob for word in ("archetype", "representative", "illustrat", "not measured")
        ), "climate archetypes must not read as measured station data"

    def test_climate_capability_is_preview(self):
        assert capabilities.get("twin_climate").tier is Tier.PREVIEW


# --------------------------------------------------------------------------
# Inference contract
# --------------------------------------------------------------------------
@needs_models
class TestPrediction:
    def test_in_envelope_is_validated(self):
        from psc_twin.surrogate import predict

        p = predict.predict(1.0, 85.0, 1000.0)
        assert p.tier is Tier.VALIDATED
        assert p.excursions == []
        assert len(p.trajectories) > 10

    def test_outside_envelope_downgrades_to_preview(self):
        from psc_twin.surrogate import predict

        p = predict.predict(1.4, 150.0, 2500.0)
        assert p.tier is Tier.PREVIEW
        assert len(p.excursions) == 3

    def test_trajectory_carries_uncertainty(self):
        from psc_twin.surrogate import predict

        p = predict.predict(0.6, 60.0, 800.0)
        for target in TARGETS:
            assert target in p.trajectories.columns
            assert f"{target}_std" in p.trajectories.columns
            assert (p.trajectories[f"{target}_std"] >= 0).all()

    def test_predictions_are_deterministic(self):
        from psc_twin.surrogate import predict

        a = predict.predict(0.8, 75.0, 900.0)
        b = predict.predict(0.8, 75.0, 900.0)
        pd.testing.assert_frame_equal(a.trajectories, b.trajectories)

    def test_predicted_curve_is_physical(self):
        from psc_twin.surrogate import predict

        p = predict.predict(1.0, 85.0, 500.0)
        assert p.curve is not None
        assert p.curve.metrics.valid, p.curve.metrics.note
        assert 0.0 < p.curve.metrics.ff < 1.0
        assert p.curve.metrics.jsc_macm2 > 0

    def test_recommendations_avoid_existing_design_points(self):
        from psc_twin.surrogate import predict

        scalars, _ = predict._load_models()
        recs = activelearn.recommend_runs(scalars, n=5)
        assert len(recs) > 0
        assert (recs["distance_to_nearest_run"] >= activelearn.EXCLUSION_RADIUS).all()


# --------------------------------------------------------------------------
# Data contract
# --------------------------------------------------------------------------
class TestDataContract:
    def test_feature_order_is_stable(self):
        assert FEATURES == ("aging_light_suns", "aging_temperature_C", "aging_h")

    def test_curves_align_with_metrics(self, bundle):
        assert bundle.curve_matrix.shape[0] == len(bundle.metrics)
        assert not np.isnan(bundle.curve_matrix).any()

    def test_shared_voltage_grid(self, bundle):
        assert bundle.voltage.size == 72
        assert bundle.voltage[0] == pytest.approx(0.0)

    def test_no_missing_targets(self, bundle):
        for target in TARGETS:
            assert bundle.metrics[target].notna().all()
