"""Scientific-contract tests for the recommended marked-model family."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.pfc.glm import event_basis_functions
from continuum_swr_replay.analysis.pfc.heldout import assign_blocked_folds
from continuum_swr_replay.analysis.pfc.marked_glm import (
    EVENT_DURATION_MARK,
    MARKED_EVENT_CLASSES,
    build_marked_model_ladder,
    fit_marked_kernel_ladder,
)
from continuum_swr_replay.analysis.pfc.marked_model_family import (
    CONTENT_UNCERTAINTY,
    CONTENT_VALUE,
    CURRENT_ARM_FRACTION,
    CURRENT_PATCH_FRACTION,
    CURRENT_X_SWITCH,
    DECODE_ENTROPY,
    LEAF_CONTENT_FRACTION,
    LOG_SPATIAL_EXTENT,
    MARK_DESCRIPTIONS,
    PREVIOUS_ARM_FRACTION,
    RECOMMENDED_MODEL_SPECS,
    SWITCH_CONTEXT,
    MarkedFamilySpec,
    attach_recommended_event_marks,
    bounded_design_batch_size,
    build_marked_family_ladder,
    build_recommended_event_mark_table,
    fit_marked_family_ladder,
)

SAMPLING_FREQUENCY = 500.0


def _tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    arm_rows: list[dict[str, object]] = []
    weights_by_event = {
        "e1": np.array([0.10, 0.20, 0.10, 0.10, 0.05, 0.05, 0.10, 0.15, 0.15]),
        "e2": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.4, 0.3]),
    }
    for event_index, (event_id, weights) in enumerate(weights_by_event.items()):
        for arm, weight in enumerate(weights):
            has_context = event_id == "e1"
            arm_rows.append(
                {
                    "event_id": event_id,
                    "candidate_arm": arm,
                    "arm_fraction": weight,
                    "event_start_s": (1.0, 2.0)[event_index],
                    "event_end_s": (1.04, 2.08)[event_index],
                    "event_duration_s": (0.042, 0.082)[event_index],
                    "is_swr": bool(event_index),
                    "is_current_arm": (arm == 1) if has_context else np.nan,
                    "is_prev_arm": arm == 2,
                    "is_same_patch_as_current": (
                        arm in {0, 1, 6} if has_context else np.nan
                    ),
                    # Large structural trunk values must never enter the
                    # leaf-normalized weighted summaries.
                    "reward_belief_variance_candidate_arm_preoutcome": (
                        0.01 * (arm + 1) if arm < 6 else 99.0
                    ),
                    "reward_belief_mean_candidate_arm_preoutcome": (
                        0.1 * (arm + 1) if arm < 6 else 99.0
                    ),
                    "reward_fraction_candidate_arm_last10_visits": (
                        0.05 * arm if arm < 6 else 99.0
                    ),
                    "visit_recency_trials_event_leaf_centered": (
                        float(arm - 2) if arm < 6 else 99.0
                    ),
                    "is_candidate_arm_never_visited": (
                        float(arm == 5) if arm < 6 else 0.0
                    ),
                }
            )
    event_rows = pd.DataFrame(
        {
            "event_id": ["e1", "e2"],
            "event_start_s": [1.0, 2.0],
            "event_end_s": [1.04, 2.08],
            "event_duration_s": [0.042, 0.082],
            "is_swr": [False, True],
            "time_since_stop_start_s": [0.5, 1.5],
            "did_last_completed_trip_switch_patch": [1.0, 0.0],
            "is_rewarded_outcome": [0.0, 1.0],
            "spatial_extent_cm": [20.0, 80.0],
            "n_reversals": [2, 8],
            "pause_fraction": [0.1, 0.3],
            "max_ahead_cm": [10.0, 20.0],
            "max_behind_cm": [40.0, 100.0],
            "ripple_power_z": [-0.5, 2.0],
            "mua_rate_z": [0.2, 1.4],
            "theta_z": [0.1, -0.2],
            "posterior_entropy_mean": [0.4, 0.8],
        }
    )
    return pd.DataFrame(arm_rows), event_rows


def test_recommended_marks_exclude_structural_trunk_zeros_and_preserve_leaf_mass():
    arm, event = _tables()

    marks = build_recommended_event_mark_table(arm, event).set_index("event_id")

    weights = np.array([0.10, 0.20, 0.10, 0.10, 0.05, 0.05])
    expected_uncertainty = np.sum(weights * (0.01 * np.arange(1, 7))) / weights.sum()
    expected_value = np.sum(weights * (0.1 * np.arange(1, 7))) / weights.sum()
    assert marks.loc["e1", LEAF_CONTENT_FRACTION] == pytest.approx(0.60)
    assert marks.loc["e1", CONTENT_UNCERTAINTY] == pytest.approx(expected_uncertainty)
    assert marks.loc["e1", CONTENT_VALUE] == pytest.approx(expected_value)
    assert marks.loc["e1", CURRENT_ARM_FRACTION] == pytest.approx(0.20)
    assert marks.loc["e1", PREVIOUS_ARM_FRACTION] == pytest.approx(0.10)
    assert marks.loc["e1", CURRENT_PATCH_FRACTION] == pytest.approx(0.40)
    assert marks.loc["e1", CURRENT_X_SWITCH] == pytest.approx(0.20)
    assert marks.loc["e1", LOG_SPATIAL_EXTENT] == pytest.approx(np.log1p(20.0))

    # An all-trunk event has no leaf uncertainty/value estimand. It is not
    # silently assigned the structural zero carried by trunk-arm table rows.
    assert marks.loc["e2", LEAF_CONTENT_FRACTION] == 0.0
    assert np.isnan(marks.loc["e2", CONTENT_UNCERTAINTY])
    assert np.isnan(marks.loc["e2", CONTENT_VALUE])
    assert np.isnan(marks.loc["e2", CURRENT_ARM_FRACTION])


def test_recommended_marks_refuse_partial_leaf_covariates():
    arm, event = _tables()
    arm.loc[
        (arm.event_id == "e1") & (arm.candidate_arm == 3),
        "reward_belief_variance_candidate_arm_preoutcome",
    ] = np.nan

    with pytest.raises(ValueError, match="partially available leaf values"):
        build_recommended_event_mark_table(arm, event)


def test_mark_attachment_preserves_class_order_and_validates_timing():
    arm, event = _tables()
    marks = build_recommended_event_mark_table(arm, event)
    classes = {
        "non_local_no_SWR": pd.DataFrame(
            {"event_id": ["e1"], "start_time": [1.0], "end_time": [1.04]}
        ),
        "non_local_SWR": pd.DataFrame(
            {"event_id": ["e2"], "start_time": [2.0], "end_time": [2.08]}
        ),
    }

    attached = attach_recommended_event_marks(classes, marks)

    assert attached["non_local_no_SWR"].event_id.tolist() == ["e1"]
    assert attached["non_local_no_SWR"][SWITCH_CONTEXT].tolist() == [1.0]
    wrong = {name: frame.copy() for name, frame in classes.items()}
    wrong["non_local_SWR"].loc[0, "start_time"] += 0.01
    with pytest.raises(ValueError, match="timing disagrees"):
        attach_recommended_event_marks(wrong, marks)


def _family_ladder():
    fit_mask = pd.Series(
        np.r_[
            np.ones(100, dtype=bool), np.zeros(20, dtype=bool), np.ones(100, dtype=bool)
        ],
        index=np.arange(220) / SAMPLING_FREQUENCY,
    )
    time = fit_mask.index.to_numpy()
    lags = np.arange(-4, 5)
    basis, _, _ = event_basis_functions(lags, SAMPLING_FREQUENCY)
    events = {}
    for class_index, event_class in enumerate(MARKED_EVENT_CLASSES):
        bins = np.array([20, 40, 60, 140, 160, 180]) + class_index
        events[event_class] = pd.DataFrame(
            {
                "event_id": class_index * 10 + np.arange(len(bins)),
                "start_time": time[bins],
                EVENT_DURATION_MARK: np.linspace(-3.0, -1.0, len(bins)),
                CURRENT_ARM_FRACTION: np.linspace(0.1, 0.9, len(bins)),
                PREVIOUS_ARM_FRACTION: np.linspace(0.8, 0.2, len(bins)) ** 2,
            }
        )
    spec = MarkedFamilySpec(
        "test_family",
        (CURRENT_ARM_FRACTION, PREVIOUS_ARM_FRACTION),
        (EVENT_DURATION_MARK,),
        "test two target blocks",
    )
    return build_marked_family_ladder(
        events,
        fit_mask,
        basis,
        lags,
        spec,
        min_marked_events=2,
    )


def test_family_ladder_drops_exactly_one_target_block_per_comparison():
    ladder = _family_ladder()
    n_basis = ladder.spline_basis.shape[1]
    expected_full_width = 1 + 2 * (1 + 1 + 2) * n_basis

    assert ladder.full.matrix.shape[1] == expected_full_width
    assert tuple(ladder.reduced) == tuple(
        (event_class, mark)
        for event_class in MARKED_EVENT_CLASSES
        for mark in (CURRENT_ARM_FRACTION, PREVIOUS_ARM_FRACTION)
    )
    for (event_class, mark), reduced in ladder.reduced.items():
        span = ladder.full.block_slices[f"{event_class}:{mark}"]
        expected = np.delete(
            ladder.full.matrix, np.arange(span.start, span.stop), axis=1
        )
        np.testing.assert_array_equal(reduced.matrix, expected)
        assert reduced.matrix.shape[1] == expected_full_width - n_basis


def test_single_target_family_is_numerically_identical_to_single_mark_estimator():
    source = _family_ladder()
    spec = MarkedFamilySpec(
        "single_target_equivalence",
        (CURRENT_ARM_FRACTION,),
        (EVENT_DURATION_MARK,),
        "equivalence reference",
    )
    family = build_marked_family_ladder(
        source.events,
        source.fit_mask,
        source.spline_basis,
        source.lags,
        spec,
        min_marked_events=2,
    )
    single = build_marked_model_ladder(
        source.events,
        source.fit_mask,
        source.spline_basis,
        source.lags,
        scientific_mark=CURRENT_ARM_FRACTION,
        nuisance_marks=(EVENT_DURATION_MARK,),
        min_marked_events=2,
    )
    np.testing.assert_array_equal(family.full.matrix, single.full.matrix)
    for event_class in MARKED_EVENT_CLASSES:
        np.testing.assert_array_equal(
            family.reduced[(event_class, CURRENT_ARM_FRACTION)].matrix,
            single.reduced[event_class].matrix,
        )

    rng = np.random.default_rng(82)
    counts = rng.poisson(0.15, size=(family.full.matrix.shape[0], 2)).astype(float)
    folds = assign_blocked_folds(
        family.fit_mask,
        n_folds=2,
        seed=9,
        support_radius_bins=int(np.abs(family.lags).max()),
    )
    family_result = fit_marked_family_ladder(
        family, counts, folds, penalty_sigma=3.0, design_batch_size=2
    )
    single_result = fit_marked_kernel_ladder(
        single,
        counts,
        folds,
        penalty_sigma=3.0,
        n_shuffles=0,
        design_batch_size=2,
    )

    np.testing.assert_allclose(
        family_result.statistics["family_full_heldout_deviance"],
        single_result.statistics["marked_full_heldout_deviance"],
        atol=1e-8,
        rtol=1e-10,
        equal_nan=True,
    )
    for event_class in MARKED_EVENT_CLASSES:
        prefix = f"{event_class}:{CURRENT_ARM_FRACTION}"
        np.testing.assert_allclose(
            family_result.statistics[f"{prefix}: heldout_delta_deviance"],
            single_result.statistics[f"{event_class}: marked_heldout_delta_dev"],
            atol=1e-8,
            rtol=1e-10,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            family_result.typical_curves[event_class],
            single_result.typical_curves[event_class],
            atol=1e-8,
            rtol=1e-10,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            family_result.mark_curves[event_class][CURRENT_ARM_FRACTION],
            single_result.mark_curves[event_class],
            atol=1e-8,
            rtol=1e-10,
            equal_nan=True,
        )


def test_recommended_specs_share_compiled_widths_within_each_tier():
    core_counts = {
        len(spec.all_marks)
        for spec in RECOMMENDED_MODEL_SPECS.values()
        if spec.tier == "core"
    }
    interaction_counts = {
        len(spec.all_marks)
        for spec in RECOMMENDED_MODEL_SPECS.values()
        if spec.tier == "interaction"
    }

    assert core_counts == {6, 9}
    assert (
        sum(
            len(spec.all_marks) == 6
            for spec in RECOMMENDED_MODEL_SPECS.values()
            if spec.tier == "core"
        )
        == 6
    )
    assert (
        sum(
            len(spec.all_marks) == 9
            for spec in RECOMMENDED_MODEL_SPECS.values()
            if spec.tier == "core"
        )
        == 3
    )
    assert interaction_counts == {9}
    assert set(MARK_DESCRIPTIONS) >= {
        mark for spec in RECOMMENDED_MODEL_SPECS.values() for mark in spec.all_marks
    }


def test_design_batch_bound_never_exceeds_profiled_feature_workload():
    assert bounded_design_batch_size(4, 67) == 4
    assert bounded_design_batch_size(4, 144) == 2
    assert bounded_design_batch_size(10, 50) == 5
    with pytest.raises(ValueError, match="must be positive"):
        bounded_design_batch_size(0, 50)


def test_interaction_specs_include_both_constituent_main_effects():
    exploration = RECOMMENDED_MODEL_SPECS["exploration_content_interaction"]

    assert CURRENT_ARM_FRACTION in exploration.nuisance_marks
    assert CONTENT_UNCERTAINTY in exploration.nuisance_marks
    assert SWITCH_CONTEXT in exploration.nuisance_marks
    assert DECODE_ENTROPY in exploration.nuisance_marks
    assert exploration.required_varying_nuisance_marks == (SWITCH_CONTEXT,)
