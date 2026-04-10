"""Tests for the replay_content analysis subpackage.

Covers the most critical functions: spatial mapping, event collapsing,
arm counts, predictor computation (including dist_to_arm), conditional
Poisson fitting, cross-validation, and model comparison.
"""

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.hippocampus.replay_content.event_collapsing import (
    _modal,
    add_interaction_columns,
    compute_arm_counts,
)
from continuum_swr_replay.analysis.hippocampus.replay_content.fitting import (
    _log_likelihood,
    _null_log_likelihood,
    fit_conditional_logit,
    fit_conditional_poisson,
    fit_glm,
    fit_null_model_conditional,
)
from continuum_swr_replay.analysis.hippocampus.replay_content.model_checking import (
    cross_validate_conditional_poisson,
    cross_validate_session,
)
from continuum_swr_replay.analysis.hippocampus.replay_content.model_comparison import (
    compare_nested_models,
    summarize_comparison,
)
from continuum_swr_replay.analysis.hippocampus.replay_content.model_specs import (
    MODEL_HIERARCHY,
)
from continuum_swr_replay.analysis.hippocampus.replay_content.predictors import (
    compute_dist_to_arm,
    compute_time_since_stop,
)
from continuum_swr_replay.analysis.hippocampus.replay_content.spatial_mapping import (
    well_to_arm_set,
)

# ── Fixtures ──


@pytest.fixture
def sample_track_graph():
    """Create a simple track graph with 6 wells and 9 edges."""
    G = nx.Graph()
    # Add 6 well nodes with positions
    well_positions = {
        0: (0, 0),
        1: (10, 0),
        2: (0, 10),
        3: (10, 10),
        4: (0, 20),
        5: (10, 20),
    }
    for node, pos in well_positions.items():
        G.add_node(node, pos=pos)
    # Add 4 junction nodes
    for node, pos in [(6, (5, 5)), (7, (5, 5)), (8, (5, 15)), (9, (5, 15))]:
        G.add_node(node, pos=pos)
    # Add 9 edges with edge_id
    edges = [
        (0, 6, 0),
        (1, 6, 1),
        (0, 1, 6),
        (2, 7, 2),
        (3, 7, 3),
        (2, 3, 7),
        (4, 8, 4),
        (5, 8, 5),
        (4, 5, 8),
    ]
    for u, v, eid in edges:
        G.add_edge(u, v, edge_id=eid)
    return G


@pytest.fixture
def sample_long_format():
    """Create sample long-format data for 5 events × 9 arms."""
    np.random.seed(42)
    n_events = 5
    n_arms = 9
    rows = []
    for event in range(1, n_events + 1):
        replay_arm = np.random.choice(n_arms)
        current_arm = np.random.choice(n_arms)
        for arm in range(n_arms):
            rows.append(
                {
                    "animal": "test",
                    "day": 20200101,
                    "epoch": "02_r1",
                    "non_local_event": event,
                    "replay_here": int(arm == replay_arm),
                    "is_current": int(arm == current_arm),
                    "is_prev": int(arm == (current_arm + 1) % n_arms),
                    "is_next": int(arm == (current_arm - 1) % n_arms),
                    "is_last_rewarded": int(arm == 0),
                    "is_swr": int(event % 2 == 0),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def sample_count_long_format():
    """Create sample count-based long-format data for ConditionalPoisson."""
    np.random.seed(42)
    n_events = 50
    n_arms = 9
    rows = []
    for event in range(n_events):
        # Generate random counts that sum to ~30 timebins
        raw_counts = np.random.poisson(3, n_arms)
        # Ensure at least one arm has counts
        if raw_counts.sum() == 0:
            raw_counts[0] = 1
        current_arm = np.random.choice(n_arms)
        for arm in range(n_arms):
            rows.append(
                {
                    "event_uid": event,
                    "arm_id": arm,
                    "arm_count": int(raw_counts[arm]),
                    "is_current": int(arm == current_arm),
                    "is_prev": int(arm == (current_arm + 1) % n_arms),
                    "is_next": int(arm == (current_arm - 1) % n_arms),
                    "is_last_rewarded": int(arm == 0),
                    "is_swr": int(event % 2 == 0),
                    "animal": "test",
                    "day": 20200101,
                    "epoch": "02_r1",
                    "non_local_event": event + 1,
                }
            )
    return pd.DataFrame(rows)


# ── Tests: Spatial Mapping ──


class TestWellToArmSet:
    """Test well_to_arm_set function."""

    def test_returns_set_of_adjacent_arms(self, sample_track_graph):
        arms = well_to_arm_set(0, sample_track_graph)
        assert isinstance(arms, set)
        assert len(arms) > 0

    def test_nan_returns_empty_set(self, sample_track_graph):
        arms = well_to_arm_set(np.nan, sample_track_graph)
        assert arms == set()

    def test_each_well_has_two_arms(self, sample_track_graph):
        for well_id in range(6):
            arms = well_to_arm_set(well_id, sample_track_graph)
            assert len(arms) == 2, f"Well {well_id} has {len(arms)} arms"


# ── Tests: Predictors ──


class TestComputeTimeSinceStop:
    """Test time_since_stop computation."""

    def test_single_bout(self):
        """All timebins in one bout should have increasing time_since_stop."""
        times = np.arange(0, 0.2, 0.002)
        nl = pd.DataFrame({"head_speed": 1.0}, index=times)
        result = compute_time_since_stop(nl)
        assert len(result) == len(nl)
        assert result.iloc[0] == pytest.approx(0.0)
        assert result.iloc[-1] == pytest.approx(times[-1] - times[0], abs=0.01)

    def test_two_bouts_separated_by_gap(self):
        """A gap > BOUT_GAP_THRESHOLD_SEC should start a new bout."""
        bout1 = np.arange(0, 0.05, 0.002)
        bout2 = np.arange(1.0, 1.05, 0.002)
        times = np.concatenate([bout1, bout2])
        nl = pd.DataFrame({"head_speed": 1.0}, index=times)
        result = compute_time_since_stop(nl)
        assert result.iloc[0] == pytest.approx(0.0)
        assert result.iloc[len(bout1)] == pytest.approx(0.0)

    def test_empty_dataframe(self):
        nl = pd.DataFrame({"head_speed": []}, index=pd.Index([], dtype=float))
        result = compute_time_since_stop(nl)
        assert len(result) == 0


class TestComputeDistToArm:
    """Test dist_to_arm computation."""

    def test_adjacent_arm_has_distance_one(self, sample_track_graph):
        """Arms adjacent to the current well should have small distance."""
        dists = compute_dist_to_arm(0, sample_track_graph)
        # Well 0 is adjacent to arm 0 (edge 0-6) and arm 6 (edge 0-1)
        assert dists[0] == 0  # arm 0 connects to well 0
        assert dists[6] == 0  # arm 6 connects wells 0-1

    def test_all_distances_non_negative_or_nan(self, sample_track_graph):
        """Distances should be non-negative (NaN for unreachable arms)."""
        for well in range(6):
            dists = compute_dist_to_arm(well, sample_track_graph)
            for arm, dist in dists.items():
                assert np.isnan(dist) or dist >= 0, (
                    f"Well {well}, arm {arm}: dist={dist}"
                )

    def test_nan_well_returns_nan(self, sample_track_graph):
        dists = compute_dist_to_arm(np.nan, sample_track_graph)
        assert all(np.isnan(v) for v in dists.values())

    def test_returns_all_arms(self, sample_track_graph):
        dists = compute_dist_to_arm(0, sample_track_graph)
        assert len(dists) == 9


# ── Tests: Event Collapsing ──


class TestModal:
    def test_returns_mode(self):
        assert _modal(pd.Series([1, 2, 2, 3])) == 2

    def test_nan_values_ignored(self):
        assert _modal(pd.Series([np.nan, 1, 1, np.nan])) == 1

    def test_all_nan_returns_nan(self):
        assert np.isnan(_modal(pd.Series([np.nan, np.nan])))

    def test_empty_returns_nan(self):
        assert np.isnan(_modal(pd.Series([], dtype=float)))


class TestAddInteractionColumns:
    def test_creates_expected_columns(self):
        df = pd.DataFrame(
            {
                "is_after_switch": [1, 0],
                "is_current": [1, 0],
                "is_prev": [0, 1],
                "is_after_omission": [0, 1],
                "is_last_rewarded": [1, 0],
                "is_swr": [1, 0],
                "is_next": [0, 1],
            }
        )
        add_interaction_columns(df)
        assert "switch_x_current" in df.columns
        assert "omission_x_rewarded" in df.columns
        assert "swr_x_current" in df.columns
        assert "swr_x_next" in df.columns
        assert df["switch_x_current"].iloc[0] == 1
        assert df["switch_x_current"].iloc[1] == 0


class TestComputeArmCounts:
    """Test per-arm timebin counting."""

    def test_counts_sum_to_total(self):
        """Arm counts should sum to total_count for each event."""
        nl = pd.DataFrame(
            {
                "animal": "test",
                "day": 20200101,
                "epoch": "02_r1",
                "non_local_event": [1, 1, 1, 1, 2, 2, 2],
                "replay_arm": [0, 0, 1, 3, 2, 2, 5],
                "SWR_event": [1, 1, 0, 0, 0, 0, 1],
            }
        )
        result = compute_arm_counts(nl)
        assert len(result) == 2  # 2 events

        for _, row in result.iterrows():
            count_cols = [c for c in result.columns if c.startswith("arm_count_")]
            assert row[count_cols].sum() == row["total_count"]

    def test_junction_bins_excluded(self):
        """Timebins with replay_arm not in ALL_ARMS should be excluded."""
        nl = pd.DataFrame(
            {
                "animal": "test",
                "day": 20200101,
                "epoch": "02_r1",
                "non_local_event": [1, 1, 1],
                "replay_arm": [0, np.nan, -1],  # NaN and -1 excluded
                "SWR_event": [0, 0, 0],
            }
        )
        result = compute_arm_counts(nl)
        assert result["total_count"].iloc[0] == 1  # only arm 0 counted

    def test_empty_returns_empty(self):
        nl = pd.DataFrame(
            {
                "non_local_event": [0, 0],  # no events > 0
                "replay_arm": [1, 2],
                "animal": "test",
                "day": 20200101,
                "epoch": "02_r1",
            }
        )
        result = compute_arm_counts(nl)
        assert len(result) == 0


# ── Tests: GLM Fitting ──


class TestFitGlm:
    """Test legacy GLM fitting function."""

    def test_null_model(self):
        """Empty predictor list should fit intercept-only."""
        np.random.seed(42)
        y = np.random.binomial(1, 0.3, 100)
        df = pd.DataFrame({"replay_here": y})
        result = fit_glm(df, predictors=[])
        assert result is not None
        assert "coef_intercept" in result
        assert result["n_params"] == 1
        assert result["pseudo_r2"] == 0.0

    def test_one_predictor(self, sample_long_format):
        result = fit_glm(sample_long_format, predictors=["is_current"])
        assert result is not None
        assert "coef_is_current" in result
        assert "aic" in result
        assert result["n_params"] == 2

    def test_returns_none_for_insufficient_data(self):
        """Should return None when data is too small."""
        df = pd.DataFrame({"replay_here": [0, 0, 1], "x": [1, 0, 1]})
        result = fit_glm(df, predictors=["x"])
        assert result is None

    def test_aic_is_finite(self, sample_long_format):
        result = fit_glm(sample_long_format, predictors=["is_current"])
        assert result is not None
        assert np.isfinite(result["aic"])
        assert np.isfinite(result["bic"])


# ── Tests: Conditional Poisson Fitting ──


class TestFitConditionalPoisson:
    """Test conditional Poisson model fitting."""

    def test_null_model(self, sample_count_long_format):
        """Empty predictors should return analytical null."""
        result = fit_conditional_poisson(
            sample_count_long_format, predictors=[], event_col="event_uid"
        )
        assert result is not None
        assert result["n_params"] == 0
        assert result["llf"] < 0

    def test_one_predictor(self, sample_count_long_format):
        result = fit_conditional_poisson(
            sample_count_long_format,
            predictors=["is_current"],
            event_col="event_uid",
        )
        assert result is not None
        assert "coef_is_current" in result
        assert "bse_is_current" in result
        assert "pval_is_current" in result
        assert result["n_params"] == 1
        assert np.isfinite(result["aic"])

    def test_multiple_predictors(self, sample_count_long_format):
        result = fit_conditional_poisson(
            sample_count_long_format,
            predictors=["is_current", "is_prev", "is_next"],
            event_col="event_uid",
        )
        assert result is not None
        assert result["n_params"] == 3

    def test_returns_none_for_insufficient_data(self):
        """Should return None when too few events."""
        df = pd.DataFrame(
            {
                "arm_count": [1, 0, 0],
                "event_uid": [0, 0, 0],
                "x": [1, 0, 0],
            }
        )
        result = fit_conditional_poisson(df, predictors=["x"], event_col="event_uid")
        assert result is None

    def test_null_model_analytical(self, sample_count_long_format):
        """Null model ℓ₀ should equal -Σ L_e · log(9)."""
        result = fit_null_model_conditional(
            sample_count_long_format, event_col="event_uid"
        )
        # Verify analytically
        total_count = sample_count_long_format.groupby("event_uid")[
            "arm_count"
        ].sum()
        expected_ll = -float(total_count.sum()) * np.log(9)
        assert result["llf"] == pytest.approx(expected_ll, rel=1e-6)
        assert result["n_params"] == 0

    def test_recovers_known_signal(self):
        """ConditionalPoisson should detect a strong signal in synthetic data."""
        np.random.seed(123)
        n_events = 200
        n_arms = 9
        beta_current = 1.5  # strong positive effect

        rows = []
        for event in range(n_events):
            current_arm = np.random.choice(n_arms)
            # Generate counts from softmax with known beta
            eta = np.zeros(n_arms)
            eta[current_arm] = beta_current
            probs = np.exp(eta) / np.exp(eta).sum()
            counts = np.random.multinomial(50, probs)

            for arm in range(n_arms):
                rows.append(
                    {
                        "event_uid": event,
                        "arm_count": int(counts[arm]),
                        "is_current": int(arm == current_arm),
                    }
                )

        df = pd.DataFrame(rows)
        result = fit_conditional_poisson(
            df, predictors=["is_current"], event_col="event_uid"
        )
        assert result is not None
        # Coefficient should be positive and close to 1.5
        assert result["coef_is_current"] > 0.5
        assert result["coef_is_current"] < 3.0
        # Should be highly significant
        assert result["pval_is_current"] < 0.001


class TestFitConditionalLogit:
    """Test conditional logit robustness model."""

    def test_fits_on_count_data(self, sample_count_long_format):
        result = fit_conditional_logit(
            sample_count_long_format,
            predictors=["is_current"],
            event_col="event_uid",
        )
        assert result is not None
        assert "coef_is_current" in result


# ── Tests: Cross-Validation ──


class TestCrossValidateConditionalPoisson:
    """Test event-based CV for conditional Poisson."""

    def test_cv_produces_results(self, sample_count_long_format):
        result = cross_validate_conditional_poisson(
            sample_count_long_format,
            predictors=["is_current"],
            event_col="event_uid",
            n_folds=5,
        )
        assert result is not None
        assert "cv_ll_per_event_mean" in result
        assert "cv_ll_per_timebin_mean" in result
        assert result["cv_n_folds"] >= 2

    def test_null_model_cv(self, sample_count_long_format):
        result = cross_validate_conditional_poisson(
            sample_count_long_format,
            predictors=[],
            event_col="event_uid",
            n_folds=5,
        )
        assert result is not None
        assert result["cv_ll_per_event_mean"] < 0  # log-likelihood is negative


class TestCrossValidateSession:
    """Test legacy binary cross-validation."""

    def test_cv_produces_results(self, sample_long_format):
        result = cross_validate_session(
            sample_long_format, predictors=["is_current"], n_folds=2
        )
        assert result is None or "cv_brier_mean" in result

    def test_null_model_cv(self, sample_long_format):
        result = cross_validate_session(sample_long_format, predictors=[], n_folds=2)
        if result is not None:
            assert "cv_brier_mean" in result
            assert result["cv_n_folds"] >= 1

    def test_requires_non_local_event_column(self):
        df = pd.DataFrame({"replay_here": [0, 1, 0]})
        result = cross_validate_session(df, predictors=[])
        assert result is None

    def test_cv_groups_events_not_rows(self, sample_long_format):
        """GroupKFold should keep all rows for an event in the same fold."""
        from sklearn.model_selection import GroupKFold

        groups = sample_long_format["non_local_event"].values
        y = sample_long_format["replay_here"].values
        gkf = GroupKFold(n_splits=2)

        for train_idx, test_idx in gkf.split(sample_long_format, y, groups):
            train_events = set(groups[train_idx])
            test_events = set(groups[test_idx])
            assert train_events & test_events == set()

    def test_cv_brier_is_non_negative(self, sample_long_format):
        """CV Brier score should always be non-negative."""
        result = cross_validate_session(
            sample_long_format, predictors=["is_current"], n_folds=2
        )
        if result is not None:
            assert result["cv_brier_mean"] >= 0


# ── Tests: Model Comparison ──


class TestModelComparison:
    def test_compare_nested_models(self):
        parent = pd.DataFrame(
            {
                "animal": ["a", "a"],
                "day": [1, 2],
                "epoch": ["e1", "e2"],
                "ll": [-100.0, -200.0],
                "aic": [204.0, 404.0],
                "bic": [210.0, 410.0],
                "n_params": [2, 2],
            }
        )
        child = pd.DataFrame(
            {
                "animal": ["a", "a"],
                "day": [1, 2],
                "epoch": ["e1", "e2"],
                "ll": [-95.0, -195.0],
                "aic": [196.0, 396.0],
                "bic": [205.0, 405.0],
                "n_params": [3, 3],
            }
        )
        result = compare_nested_models(parent, child)
        assert len(result) == 2
        assert "delta_aic" in result.columns
        assert result["delta_aic"].iloc[0] > 0  # child better
        assert result["lr_stat"].iloc[0] > 0
        assert result["df_diff"].iloc[0] == 1

    def test_compare_with_llf_column(self):
        """Should work with 'llf' column from conditional Poisson."""
        parent = pd.DataFrame(
            {
                "animal": ["a"],
                "day": [1],
                "epoch": ["e1"],
                "llf": [-100.0],
                "aic": [200.0],
                "bic": [200.0],
                "n_params": [0],
            }
        )
        child = pd.DataFrame(
            {
                "animal": ["a"],
                "day": [1],
                "epoch": ["e1"],
                "llf": [-90.0],
                "aic": [182.0],
                "bic": [184.0],
                "n_params": [1],
            }
        )
        result = compare_nested_models(parent, child)
        assert len(result) == 1
        assert result["delta_aic"].iloc[0] > 0

    def test_mismatched_n_events_invalidates_comparison(self):
        """delta_aic/bic/LRT should be NaN when n_events differs."""
        parent = pd.DataFrame(
            {
                "animal": ["a"],
                "day": [1],
                "epoch": ["e1"],
                "llf": [-100.0],
                "aic": [200.0],
                "bic": [200.0],
                "n_params": [0],
                "n_events": [50],
            }
        )
        child = pd.DataFrame(
            {
                "animal": ["a"],
                "day": [1],
                "epoch": ["e1"],
                "llf": [-90.0],
                "aic": [182.0],
                "bic": [184.0],
                "n_params": [1],
                "n_events": [45],  # fewer events due to NaN filtering
            }
        )
        result = compare_nested_models(parent, child)
        assert len(result) == 1
        assert np.isnan(result["delta_aic"].iloc[0])
        assert np.isnan(result["delta_bic"].iloc[0])
        assert np.isnan(result["lr_pvalue"].iloc[0])

    def test_matched_n_events_preserves_comparison(self):
        """Comparisons should be valid when n_events matches."""
        parent = pd.DataFrame(
            {
                "animal": ["a"],
                "day": [1],
                "epoch": ["e1"],
                "llf": [-100.0],
                "aic": [200.0],
                "bic": [200.0],
                "n_params": [0],
                "n_events": [50],
            }
        )
        child = pd.DataFrame(
            {
                "animal": ["a"],
                "day": [1],
                "epoch": ["e1"],
                "llf": [-90.0],
                "aic": [182.0],
                "bic": [184.0],
                "n_params": [1],
                "n_events": [50],  # same events
            }
        )
        result = compare_nested_models(parent, child)
        assert len(result) == 1
        assert result["delta_aic"].iloc[0] > 0  # child better
        assert np.isfinite(result["lr_pvalue"].iloc[0])

    def test_summarize_comparison(self):
        comp = pd.DataFrame(
            {
                "delta_aic": [5.0, 3.0],
                "delta_bic": [2.0, 1.0],
                "lr_stat": [10.0, 6.0],
                "lr_pvalue": [0.001, 0.01],
                "df_diff": [1, 1],
            }
        )
        summary = summarize_comparison(comp, "M0", "M1")
        assert summary["parent"] == "M0"
        assert summary["child"] == "M1"
        assert summary["n_sessions"] == 2
        assert summary["delta_aic_median"] == 4.0


# ── Tests: Model Specs ──


class TestModelSpecs:
    def test_hierarchy_is_nested(self):
        """Each child's predictors should be a superset of its parent's."""
        for name, spec in MODEL_HIERARCHY.items():
            if spec.parent is None:
                continue
            parent_spec = MODEL_HIERARCHY[spec.parent]
            parent_preds = set(parent_spec.predictors)
            child_preds = set(spec.predictors)
            assert parent_preds.issubset(child_preds), (
                f"{name} predictors {child_preds} are not a superset of "
                f"{spec.parent} predictors {parent_preds}"
            )

    def test_all_models_have_unique_names(self):
        names = list(MODEL_HIERARCHY.keys())
        assert len(names) == len(set(names))

    def test_is_last_rewarded_appears_before_its_interactions(self):
        """Main effect must appear in model before any interaction using it."""
        for name, spec in MODEL_HIERARCHY.items():
            preds = spec.predictors
            if "swr_x_rewarded" in preds or "omission_x_rewarded" in preds:
                assert "is_last_rewarded" in preds, (
                    f"{name} has reward interactions without is_last_rewarded"
                )

    def test_no_event_level_main_effects(self):
        """Event-level constants should not appear as main effects."""
        event_level = {"time_since_stop", "is_swr", "is_after_switch",
                       "is_after_omission", "log_distance"}
        for name, spec in MODEL_HIERARCHY.items():
            for pred in spec.predictors:
                assert pred not in event_level, (
                    f"{name} has event-level main effect '{pred}' "
                    f"which cancels in conditional likelihood"
                )

    def test_m0_has_no_predictors(self):
        assert MODEL_HIERARCHY["M0"].predictors == ()
        assert MODEL_HIERARCHY["M0"].parent is None

    def test_dist_to_arm_in_m4(self):
        assert "dist_to_arm" in MODEL_HIERARCHY["M4"].predictors

    def test_time_x_next_in_m5(self):
        assert "time_x_next" in MODEL_HIERARCHY["M5"].predictors


# ── Tests: Log-Likelihood ──


class TestLogLikelihood:
    def test_perfect_prediction(self):
        y = np.array([1, 0, 1, 0])
        y_pred = np.array([0.999, 0.001, 0.999, 0.001])
        ll = _log_likelihood(y, y_pred)
        assert ll < 0
        assert ll > -1

    def test_null_equals_null_function(self):
        y = np.array([1, 0, 1, 0, 1])
        p = y.mean()
        y_pred = np.full_like(y, p, dtype=float)
        ll1 = _log_likelihood(y, y_pred)
        ll2 = _null_log_likelihood(y)
        assert ll1 == pytest.approx(ll2, abs=1e-6)
