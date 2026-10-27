"""Scientific-contract tests for the recommended marked-model family."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.pfc.design import event_basis_functions
from continuum_swr_replay.analysis.pfc.heldout import assign_blocked_folds
from continuum_swr_replay.analysis.pfc.marked_glm import (
    EVENT_DURATION_MARK,
    MARKED_EVENT_CLASSES,
    build_marked_model_ladder,
    fit_marked_kernel_ladder,
)
from continuum_swr_replay.analysis.pfc.marked_model_family import (
    CONTENT_NOVELTY,
    CONTENT_RECENCY,
    CONTENT_RECENT_REWARD,
    CONTENT_UNCERTAINTY,
    CONTENT_VALUE,
    CURRENT_ARM_FRACTION,
    CURRENT_PATCH_FRACTION,
    CURRENT_X_REWARD,
    CURRENT_X_RIPPLE,
    CURRENT_X_SWITCH,
    DECODE_ENTROPY,
    LEAF_CONTENT_FRACTION,
    LOG_REVERSALS,
    LOG_SPATIAL_EXTENT,
    LOG_STOP_TIME,
    MARK_DESCRIPTIONS,
    NEXT_TRIP_SWITCHES_PATCH,
    PREVIOUS_ARM_FRACTION,
    RECOMMENDED_MODEL_SPECS,
    RETROSPECTIVE_BIAS,
    REWARDED_CONTEXT,
    RIPPLE_POWER,
    SWITCH_CONTEXT,
    TRIPS_TO_PATCH_SWITCH,
    UNCERTAINTY_X_REWARD,
    UNCERTAINTY_X_RIPPLE,
    UNCERTAINTY_X_SWITCH,
    MarkedFamilySpec,
    ModelIneligibleError,
    attach_recommended_event_marks,
    bounded_design_batch_size,
    build_marked_family_ladder,
    build_recommended_event_mark_table,
    fit_marked_family_ladder,
    stop_patch_switch_marks,
)

SAMPLING_FREQUENCY = 500.0


def _tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    arm_rows: list[dict[str, object]] = []
    weights_by_event = {
        "e1": np.array([0.10, 0.20, 0.10, 0.10, 0.05, 0.05, 0.10, 0.15, 0.15]),
        "e2": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.4, 0.3]),
        # All represented leaf content sits on the never-visited arm 5.
        "e3": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.2, 0.2, 0.2]),
    }
    for event_index, (event_id, weights) in enumerate(weights_by_event.items()):
        for arm, weight in enumerate(weights):
            has_context = event_id == "e1"
            arm_rows.append(
                {
                    "event_id": event_id,
                    "candidate_arm": arm,
                    "arm_fraction": weight,
                    "event_start_s": (1.0, 2.0, 3.0)[event_index],
                    "event_end_s": (1.04, 2.08, 3.05)[event_index],
                    "event_duration_s": (0.042, 0.082, 0.052)[event_index],
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
                    # Arm 5 was never visited, so its recency entry is the
                    # right-censoring fill, not a measured recency.
                    "visit_recency_trials_event_leaf_centered": (
                        150.0 if arm == 5 else (float(arm - 2) if arm < 6 else 99.0)
                    ),
                    "is_candidate_arm_never_visited": (
                        float(arm == 5) if arm < 6 else 0.0
                    ),
                }
            )
    event_rows = pd.DataFrame(
        {
            "event_id": ["e1", "e2", "e3"],
            "event_start_s": [1.0, 2.0, 3.0],
            "event_end_s": [1.04, 2.08, 3.05],
            "event_duration_s": [0.042, 0.082, 0.052],
            "is_swr": [False, True, True],
            "time_since_stop_start_s": [0.5, 1.5, 2.5],
            "did_last_completed_trip_switch_patch": [1.0, 0.0, 1.0],
            "is_rewarded_outcome": [0.0, 1.0, 0.0],
            "spatial_extent_cm": [20.0, 80.0, 50.0],
            "n_reversals": [2, 8, 4],
            "pause_fraction": [0.1, 0.3, 0.2],
            "max_ahead_cm": [10.0, 20.0, 15.0],
            "max_behind_cm": [40.0, 100.0, 60.0],
            "ripple_power_z": [-0.5, 2.0, 1.0],
            "mua_rate_z": [0.2, 1.4, 0.6],
            "theta_z": [0.1, -0.2, 0.3],
            "posterior_entropy_mean": [0.4, 0.8, 0.5],
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


def test_content_recency_excludes_never_visited_right_censoring_fill():
    arm, event = _tables()

    marks = build_recommended_event_mark_table(arm, event).set_index("event_id")

    # Arms 0-4 were visited (centered recency -2..+2); arm 5 carries the
    # right-censoring fill 150.0 and must not enter either sum.
    visited_weights = np.array([0.10, 0.20, 0.10, 0.10, 0.05])
    visited_recency = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    expected_recency = np.sum(visited_weights * visited_recency) / visited_weights.sum()
    assert expected_recency == pytest.approx(-0.2 / 0.55)
    assert marks.loc["e1", CONTENT_RECENCY] == pytest.approx(expected_recency)
    # The censored fill would have dominated a mean taken over all leaf arms.
    all_leaf_mean = (-0.2 + 0.05 * 150.0) / 0.60
    assert marks.loc["e1", CONTENT_RECENCY] != pytest.approx(all_leaf_mean)

    # The never-visited arm's represented mass is still carried by the novelty
    # mark, and recent reward still averages over every finite leaf row.
    assert marks.loc["e1", CONTENT_NOVELTY] == pytest.approx(0.05 / 0.60)
    expected_recent_reward = (
        np.sum(np.array([0.10, 0.20, 0.10, 0.10, 0.05, 0.05]) * (0.05 * np.arange(6)))
        / 0.60
    )
    assert expected_recent_reward == pytest.approx(0.0575 / 0.60)
    assert marks.loc["e1", CONTENT_RECENT_REWARD] == pytest.approx(
        expected_recent_reward
    )

    # An event whose represented leaf content is entirely never-visited has no
    # recency estimand at all, while novelty and value remain defined.
    assert marks.loc["e3", LEAF_CONTENT_FRACTION] == pytest.approx(0.40)
    assert np.isnan(marks.loc["e3", CONTENT_RECENCY])
    assert marks.loc["e3", CONTENT_NOVELTY] == pytest.approx(1.0)
    assert marks.loc["e3", CONTENT_VALUE] == pytest.approx(0.6)


def test_derived_marks_use_canonical_duration_signed_bias_and_interactions():
    arm, event = _tables()

    marks = build_recommended_event_mark_table(arm, event).set_index("event_id")

    # The canonical duration is inclusive of one sampled decode bin; taking
    # end - start would shorten every event, most on the shortest ones.
    assert marks.loc["e1", EVENT_DURATION_MARK] == pytest.approx(np.log(0.042))
    assert marks.loc["e1", EVENT_DURATION_MARK] != pytest.approx(np.log(1.04 - 1.0))
    assert marks.loc["e2", EVENT_DURATION_MARK] == pytest.approx(np.log(0.082))

    # Retrospective bias keeps its sign: positive means the trajectory reached
    # further behind the animal than ahead of it.
    assert marks.loc["e1", RETROSPECTIVE_BIAS] == pytest.approx(40.0 - 10.0)
    assert marks.loc["e2", RETROSPECTIVE_BIAS] == pytest.approx(100.0 - 20.0)

    assert marks.loc["e1", LOG_STOP_TIME] == pytest.approx(np.log1p(0.5))
    assert marks.loc["e2", LOG_REVERSALS] == pytest.approx(np.log1p(8.0))
    assert marks.loc["e2", LOG_SPATIAL_EXTENT] == pytest.approx(np.log1p(80.0))
    assert marks.loc["e1", REWARDED_CONTEXT] == 0.0
    assert marks.loc["e2", REWARDED_CONTEXT] == 1.0
    assert marks.loc["e1", DECODE_ENTROPY] == pytest.approx(0.4)

    # The all-trunk event has a defined previous-arm fraction of zero (the role
    # is known on every arm row and carries no weight) but no current-patch
    # estimand at all (the role itself is undefined for that event).
    assert marks.loc["e2", PREVIOUS_ARM_FRACTION] == 0.0
    assert np.isnan(marks.loc["e2", CURRENT_PATCH_FRACTION])
    assert np.isnan(marks.loc["e2", CONTENT_NOVELTY])

    # Every interaction is the plain product of its two constituents, and is
    # undefined wherever either constituent is.
    uncertainty = marks.loc["e1", CONTENT_UNCERTAINTY]
    assert marks.loc["e1", CURRENT_X_SWITCH] == pytest.approx(0.20 * 1.0)
    assert marks.loc["e1", UNCERTAINTY_X_SWITCH] == pytest.approx(uncertainty * 1.0)
    assert marks.loc["e1", CURRENT_X_REWARD] == pytest.approx(0.0)
    assert marks.loc["e1", UNCERTAINTY_X_REWARD] == pytest.approx(0.0)
    assert marks.loc["e1", CURRENT_X_RIPPLE] == pytest.approx(0.20 * -0.5)
    assert marks.loc["e1", UNCERTAINTY_X_RIPPLE] == pytest.approx(uncertainty * -0.5)
    assert marks.loc["e2", RIPPLE_POWER] == pytest.approx(2.0)
    for interaction in (
        CURRENT_X_SWITCH,
        UNCERTAINTY_X_SWITCH,
        CURRENT_X_REWARD,
        UNCERTAINTY_X_REWARD,
        CURRENT_X_RIPPLE,
        UNCERTAINTY_X_RIPPLE,
    ):
        assert np.isnan(marks.loc["e2", interaction])


def test_recency_availability_invariant_is_checked_before_exclusion():
    arm, event = _tables()
    arm.loc[
        (arm.event_id == "e1") & (arm.candidate_arm == 5),
        "visit_recency_trials_event_leaf_centered",
    ] = np.nan

    with pytest.raises(ValueError, match="partially available leaf values"):
        build_recommended_event_mark_table(arm, event)


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


def test_mark_attachment_preserves_out_of_order_rows_within_each_class():
    mark_table = pd.DataFrame(
        {
            "event_id": [10, 11, 12, 13],
            "event_start_s": [1.0, 2.0, 3.0, 4.0],
            "event_end_s": [1.04, 2.04, 3.04, 4.04],
            "is_swr": [False, True, False, True],
            CURRENT_ARM_FRACTION: [0.1, 0.2, 0.3, 0.4],
            SWITCH_CONTEXT: [0.0, 1.0, 1.0, 0.0],
        }
    )
    classes = {
        "non_local_no_SWR": pd.DataFrame(
            {"event_id": [12, 10], "start_time": [3.0, 1.0], "end_time": [3.04, 1.04]}
        ),
        "non_local_SWR": pd.DataFrame(
            {"event_id": [13, 11], "start_time": [4.0, 2.0], "end_time": [4.04, 2.04]}
        ),
    }

    attached = attach_recommended_event_marks(classes, mark_table)

    assert attached["non_local_no_SWR"].event_id.tolist() == [12, 10]
    assert attached["non_local_SWR"].event_id.tolist() == [13, 11]
    assert attached["non_local_no_SWR"][CURRENT_ARM_FRACTION].tolist() == [0.3, 0.1]
    assert attached["non_local_SWR"][CURRENT_ARM_FRACTION].tolist() == [0.4, 0.2]
    assert attached["non_local_no_SWR"][SWITCH_CONTEXT].tolist() == [1.0, 0.0]
    assert attached["non_local_SWR"][SWITCH_CONTEXT].tolist() == [0.0, 1.0]


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
    # The single-mark estimator has one prior for every block, so the family
    # is asked to share it here; the shipped family default is tighter.
    family_result = fit_marked_family_ladder(
        family,
        counts,
        folds,
        penalty_sigma=3.0,
        mark_penalty_sigma=None,
        design_batch_size=2,
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
            family_result.statistics[f"{prefix}: reduced_heldout_deviance"],
            single_result.statistics[f"{event_class}: marked_reduced_heldout_deviance"],
            atol=1e-8,
            rtol=1e-10,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            family_result.statistics[f"{prefix}: heldout_delta_deviance"],
            single_result.statistics[f"{event_class}: marked_heldout_delta_dev"],
            atol=1e-8,
            rtol=1e-10,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            np.stack(
                family_result.statistics[
                    f"{prefix}: heldout_delta_deviance_folds"
                ].to_numpy()
            ),
            np.stack(
                single_result.statistics[
                    f"{event_class}: marked_heldout_delta_dev_folds"
                ].to_numpy()
            ),
            atol=1e-8,
            rtol=1e-10,
            equal_nan=True,
        )
        # The family writes the shared full-model convergence once, unprefixed;
        # the single-mark estimator repeats it under every class prefix.
        np.testing.assert_array_equal(
            family_result.statistics["full_all_folds_converged"],
            single_result.statistics[f"{event_class}: marked_full_all_folds_converged"],
        )
        np.testing.assert_array_equal(
            family_result.statistics[f"{prefix}: reduced_all_folds_converged"],
            single_result.statistics[
                f"{event_class}: marked_reduced_all_folds_converged"
            ],
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
    # This pins the shipped model inventory: the counts below are the number of
    # six-mark and nine-mark core models and the single interaction width.
    # Adding, removing or re-marking a model must edit this test deliberately,
    # because a new mark count adds a compiled design width to every session.
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


def test_design_batch_bound_floors_at_one_design_for_the_shipped_widths():
    # The shipped family compiles two widths, 155 and 221 columns; neither
    # fits twice in the profiled budget, so both fall to one design per batch
    # rather than to a zero-sized batch.
    assert bounded_design_batch_size(4, 221) == 1
    assert bounded_design_batch_size(4, 210) == 1
    assert bounded_design_batch_size(4, 155) == 1
    assert bounded_design_batch_size(4, 144) == 2


def test_interaction_specs_include_both_constituent_main_effects():
    exploration = RECOMMENDED_MODEL_SPECS["exploration_content_interaction"]

    assert CURRENT_ARM_FRACTION in exploration.nuisance_marks
    assert CONTENT_UNCERTAINTY in exploration.nuisance_marks
    assert SWITCH_CONTEXT in exploration.nuisance_marks
    assert DECODE_ENTROPY in exploration.nuisance_marks
    assert exploration.required_varying_nuisance_marks == (SWITCH_CONTEXT,)


def _eligibility_inputs(
    n_events: int = 12,
) -> tuple[pd.Series, np.ndarray, np.ndarray, dict[str, pd.DataFrame]]:
    """Two classes of well-separated events carrying every mark a spec needs."""
    fit_mask = pd.Series(
        np.ones(600, dtype=bool), index=np.arange(600) / SAMPLING_FREQUENCY
    )
    time = fit_mask.index.to_numpy()
    lags = np.arange(-4, 5)
    basis, _, _ = event_basis_functions(lags, SAMPLING_FREQUENCY)
    rng = np.random.default_rng(404)
    events = {}
    for class_index, event_class in enumerate(MARKED_EVENT_CLASSES):
        bins = 20 + 40 * np.arange(n_events) + class_index
        current = rng.uniform(0.1, 0.9, n_events)
        switch = np.resize([0.0, 1.0], n_events)
        events[event_class] = pd.DataFrame(
            {
                "event_id": 100 * class_index + np.arange(n_events),
                "start_time": time[bins],
                EVENT_DURATION_MARK: rng.normal(size=n_events),
                CURRENT_ARM_FRACTION: current,
                SWITCH_CONTEXT: switch,
                CURRENT_X_SWITCH: current * switch,
            }
        )
    return fit_mask, basis, lags, events


def test_a_constant_target_mark_makes_the_session_ineligible_for_the_model():
    fit_mask, basis, lags, events = _eligibility_inputs()
    events["non_local_no_SWR"][CURRENT_ARM_FRACTION] = 0.5
    spec = MarkedFamilySpec(
        name="constant_target",
        scientific_marks=(CURRENT_ARM_FRACTION,),
        nuisance_marks=(EVENT_DURATION_MARK,),
        question="a target with no variation cannot be estimated",
    )

    with pytest.raises(
        ModelIneligibleError,
        match=(
            r"model 'constant_target', event class 'non_local_no_SWR': event mark "
            r"'current_arm_fraction' has no within-class variation"
        ),
    ):
        build_marked_family_ladder(events, fit_mask, basis, lags, spec)


def test_a_constant_adjustment_mark_is_carried_as_an_all_zero_block():
    fit_mask, basis, lags, events = _eligibility_inputs()
    for event_class in MARKED_EVENT_CLASSES:
        events[event_class][SWITCH_CONTEXT] = 1.0
    spec = MarkedFamilySpec(
        name="constant_adjustment",
        scientific_marks=(CURRENT_ARM_FRACTION,),
        nuisance_marks=(EVENT_DURATION_MARK, SWITCH_CONTEXT),
        question="a constant adjustment must not refuse the session",
    )

    ladder = build_marked_family_ladder(events, fit_mask, basis, lags, spec)

    for event_class in MARKED_EVENT_CLASSES:
        assert ladder.scales[event_class][SWITCH_CONTEXT].std == 0.0
        block = f"{event_class}:{SWITCH_CONTEXT}"
        np.testing.assert_array_equal(
            ladder.full.matrix[:, ladder.full.block_slices[block]], 0.0
        )
        for reduced in ladder.reduced.values():
            assert block in reduced.block_slices
            np.testing.assert_array_equal(
                reduced.matrix[:, reduced.block_slices[block]], 0.0
            )


def test_an_interaction_refuses_a_session_whose_modifier_never_varies():
    fit_mask, basis, lags, events = _eligibility_inputs()
    for event_class in MARKED_EVENT_CLASSES:
        events[event_class][SWITCH_CONTEXT] = 1.0
        events[event_class][CURRENT_X_SWITCH] = events[event_class][
            CURRENT_ARM_FRACTION
        ]
    spec = MarkedFamilySpec(
        name="interaction_needs_a_varying_modifier",
        scientific_marks=(CURRENT_X_SWITCH,),
        nuisance_marks=(EVENT_DURATION_MARK, CURRENT_ARM_FRACTION, SWITCH_CONTEXT),
        question="an interaction is uninterpretable without a varying modifier",
        tier="interaction",
        required_varying_nuisance_marks=(SWITCH_CONTEXT,),
    )

    with pytest.raises(
        ModelIneligibleError,
        match=(
            r"model 'interaction_needs_a_varying_modifier', event class "
            r"'non_local_no_SWR': event mark 'switch_patch_context' has no "
            r"within-class variation"
        ),
    ):
        build_marked_family_ladder(events, fit_mask, basis, lags, spec)


def test_the_marked_event_floor_is_ten_finite_events_per_class():
    spec = MarkedFamilySpec(
        name="event_floor",
        scientific_marks=(CURRENT_ARM_FRACTION,),
        nuisance_marks=(EVENT_DURATION_MARK,),
        question="a slope through nine events is not reported",
    )
    fit_mask, basis, lags, events = _eligibility_inputs(n_events=10)
    nine_finite = {name: frame.copy() for name, frame in events.items()}
    nine_finite["non_local_no_SWR"].loc[0, CURRENT_ARM_FRACTION] = np.nan

    with pytest.raises(
        ModelIneligibleError,
        match=r"'current_arm_fraction' has only 9 finite events; at least 10",
    ):
        build_marked_family_ladder(nine_finite, fit_mask, basis, lags, spec)

    ladder = build_marked_family_ladder(events, fit_mask, basis, lags, spec)

    assert ladder.scales["non_local_no_SWR"][CURRENT_ARM_FRACTION].n_events == 10


@pytest.mark.slow
def test_planted_content_kernel_is_recovered_only_on_its_own_target_and_class():
    rng = np.random.default_rng(20260918)
    runs = [np.r_[np.ones(1200, dtype=bool), np.zeros(120, dtype=bool)]] * 10
    values = np.concatenate(runs)
    fit_mask = pd.Series(values, index=np.arange(len(values)) / SAMPLING_FREQUENCY)
    time = fit_mask.index.to_numpy()
    fit_rows = fit_mask.to_numpy()
    lags = np.arange(-30, 31)
    basis, lag_time, _ = event_basis_functions(lags, SAMPLING_FREQUENCY)

    usable_bins = np.flatnonzero(fit_rows)
    onset_bins = usable_bins[200:-200:180]
    classes = np.resize(np.array(MARKED_EVENT_CLASSES), len(onset_bins))
    current = rng.uniform(0.0, 1.0, len(onset_bins))
    previous = rng.uniform(0.0, 1.0, len(onset_bins))
    durations = rng.uniform(0.02, 0.2, len(onset_bins))
    events = {}
    for event_class in MARKED_EVENT_CLASSES:
        keep = classes == event_class
        events[event_class] = pd.DataFrame(
            {
                "event_id": np.flatnonzero(keep),
                "start_time": time[onset_bins[keep]],
                CURRENT_ARM_FRACTION: current[keep],
                PREVIOUS_ARM_FRACTION: previous[keep],
                EVENT_DURATION_MARK: np.log(durations[keep]),
            }
        )

    # Ground truth, built without any of the module's design machinery: a
    # constant baseline plus, at every driven-class onset, that event's own
    # standardized current-arm mark times a fixed peri-event curve. Nothing is
    # planted on the second target mark or on the other event class.
    driven_class = "non_local_no_SWR"
    gaussian = np.exp(-0.5 * ((lag_time - 0.025) / 0.025) ** 2)
    planted_curve = basis @ np.linalg.lstsq(basis, gaussian, rcond=None)[0]
    planted_curve *= 0.9 / planted_curve.max()
    peak_lag_index = int(np.argmax(planted_curve))

    driven_marks = current[classes == driven_class]
    standardized = (driven_marks - driven_marks.mean()) / driven_marks.std(ddof=0)
    log_rate = np.full(len(time), np.log(25.0 / SAMPLING_FREQUENCY))
    for onset_bin, mark_z in zip(
        onset_bins[classes == driven_class], standardized, strict=True
    ):
        positions = np.clip(onset_bin + lags, 0, len(time) - 1)
        np.add.at(log_rate, positions, mark_z * planted_curve)

    driven = rng.poisson(np.exp(log_rate))[fit_rows]
    flat = rng.poisson(25.0 / SAMPLING_FREQUENCY, size=len(driven))
    counts = np.column_stack([driven, flat]).astype(float)
    folds = assign_blocked_folds(
        fit_mask, n_folds=5, seed=12, support_radius_bins=int(np.abs(lags).max())
    )

    spec = MarkedFamilySpec(
        name="two_target_recovery",
        scientific_marks=(CURRENT_ARM_FRACTION, PREVIOUS_ARM_FRACTION),
        nuisance_marks=(EVENT_DURATION_MARK,),
        question="is a planted kernel confined to its own target and class?",
    )
    ladder = build_marked_family_ladder(events, fit_mask, basis, lags, spec)
    result = fit_marked_family_ladder(
        ladder, counts, folds, penalty_sigma=3.0, design_batch_size=2
    )

    def delta(event_class: str, mark: str) -> np.ndarray:
        return result.statistics[
            f"{event_class}:{mark}: heldout_delta_deviance"
        ].to_numpy()

    planted = delta(driven_class, CURRENT_ARM_FRACTION)
    recovered = result.mark_curves[driven_class][CURRENT_ARM_FRACTION]
    unplanted = result.mark_curves[driven_class][PREVIOUS_ARM_FRACTION]

    assert planted[0] > 0
    assert recovered[peak_lag_index, 0] == pytest.approx(0.9, abs=0.35)
    # Negative controls. The second target of the same class was never planted,
    # neither target of the other class was, and the unmodulated unit buys
    # essentially nothing from the planted comparison. The flat unit's other
    # comparisons are pure noise and are deliberately not pinned.
    # Under the shipped mark prior (0.5) an unplanted block costs little out of
    # fold either way, so its delta sits near zero on both sides; the bound is
    # a tenth of the planted gain (measured: planted +30, unplanted +1.8 and
    # the two SWR-class comparisons within a few units of zero).
    noise_bound = max(2.0, 0.1 * planted[0])
    assert delta(driven_class, PREVIOUS_ARM_FRACTION)[0] < noise_bound
    assert delta("non_local_SWR", CURRENT_ARM_FRACTION)[0] < noise_bound
    assert delta("non_local_SWR", PREVIOUS_ARM_FRACTION)[0] < noise_bound
    assert planted[1] < noise_bound
    assert np.nanmax(np.abs(unplanted[:, 0])) < 0.6


def test_family_ridge_penalty_gives_marked_blocks_their_own_prior():
    from continuum_swr_replay.analysis.pfc.batched_glm import make_ridge_penalty
    from continuum_swr_replay.analysis.pfc.marked_model_family import (
        family_ridge_penalty,
    )

    fit_mask, basis, lags, events = _eligibility_inputs()
    spec = MarkedFamilySpec(
        name="penalty_probe",
        scientific_marks=(CURRENT_ARM_FRACTION,),
        nuisance_marks=(EVENT_DURATION_MARK,),
        question="which columns carry which prior",
    )
    design = build_marked_family_ladder(events, fit_mask, basis, lags, spec).full
    shared = family_ridge_penalty(design, 3.0)
    np.testing.assert_array_equal(
        shared, make_ridge_penalty(design.matrix.shape[1], 3.0)
    )
    split = family_ridge_penalty(design, 3.0, 1.0)
    assert split[0] == 0.0
    for block in design.blocks:
        expected = 1.0 if block.mark_column is not None else 1.0 / 9.0
        np.testing.assert_allclose(split[design.block_slices[block.name]], expected)
    with pytest.raises(ValueError, match="mark_penalty_sigma"):
        family_ridge_penalty(design, 3.0, 0.0)


def _stop_sequence(arms) -> pd.DataFrame:
    """One stop per completed trial; trial ``i`` ends at ``arms[i]``."""
    arms = np.asarray(arms, dtype=float)
    return pd.DataFrame(
        {
            "stop_idx": np.arange(len(arms)) + 100,
            "completed_trial_id": pd.array(np.arange(len(arms)), dtype="Int64"),
            "completed_trial_arm": arms,
            "next_arm": np.r_[arms[1:], np.nan],
        }
    )


def test_trips_to_patch_switch_counts_forward():
    # wells 0, 1, 0, 2, 3, 4 sit in patches 1, 1, 1, 2, 2, 3
    marks = stop_patch_switch_marks(_stop_sequence([0, 1, 0, 2, 3, 4]))

    np.testing.assert_array_equal(
        marks[TRIPS_TO_PATCH_SWITCH].to_numpy(), [2, 1, 0, 1, 0, np.nan]
    )
    np.testing.assert_array_equal(
        marks[NEXT_TRIP_SWITCHES_PATCH].to_numpy(), [0, 0, 1, 0, 1, np.nan]
    )
    assert marks.index.tolist() == list(range(100, 106))

    # Six trips stay in patch 1 before the switch: capped at three.
    capped = stop_patch_switch_marks(_stop_sequence([0, 1, 0, 1, 0, 1, 0, 2, 4]))
    np.testing.assert_array_equal(
        capped[TRIPS_TO_PATCH_SWITCH].to_numpy(), [3, 3, 3, 3, 2, 1, 0, 0, np.nan]
    )


def test_trips_to_patch_switch_reads_every_stop_of_a_trial_and_checks_them():
    stops = _stop_sequence([0, 1, 2])
    # A second stop after trial 0 (e.g. off the well) shares its next trip.
    extra = stops.iloc[[0]].assign(stop_idx=999)
    # A stop before the first completed trial has no known origin.
    first = pd.DataFrame(
        {
            "stop_idx": [99],
            "completed_trial_id": pd.array([pd.NA], dtype="Int64"),
            "completed_trial_arm": [np.nan],
            "next_arm": [0.0],
        }
    )
    marks = stop_patch_switch_marks(pd.concat([first, stops, extra]))

    assert (
        marks.loc[999, TRIPS_TO_PATCH_SWITCH] == marks.loc[100, TRIPS_TO_PATCH_SWITCH]
    )
    assert marks.loc[100, TRIPS_TO_PATCH_SWITCH] == 1.0
    assert np.isnan(marks.loc[99, TRIPS_TO_PATCH_SWITCH])

    inconsistent = stops.copy()
    inconsistent.loc[0, "next_arm"] = 4.0  # disagrees with trial 1's arm, 1
    with pytest.raises(ValueError, match="disagree on the arm of trial 1"):
        stop_patch_switch_marks(inconsistent)


def test_stop_marks_join_each_event_through_its_stop():
    arm, event = _tables()
    event["stop_idx"] = [100, 102, 102]
    stop_marks = stop_patch_switch_marks(_stop_sequence([0, 1, 0, 2, 3, 4]))

    marks = build_recommended_event_mark_table(arm, event, stop_marks).set_index(
        "event_id"
    )

    assert marks[TRIPS_TO_PATCH_SWITCH].tolist() == [2.0, 0.0, 0.0]
    assert marks[NEXT_TRIP_SWITCHES_PATCH].tolist() == [0.0, 1.0, 1.0]
    assert marks["stop_idx"].tolist() == [100, 102, 102]
    event.loc[0, "stop_idx"] = 7
    with pytest.raises(ValueError, match="stop_idx the stop marks lack"):
        build_recommended_event_mark_table(arm, event, stop_marks)
    # Without stop marks the forward marks are simply absent.
    assert TRIPS_TO_PATCH_SWITCH not in build_recommended_event_mark_table(arm, event)


def test_followup_specs_use_known_marks():
    from continuum_swr_replay.analysis.pfc.marked_glm import (
        EXTENDED_MARKED_EVENT_CLASSES,
    )
    from continuum_swr_replay.analysis.pfc.marked_model_family import (
        FOLLOWUP_MODEL_SPECS,
        RECOMMENDED_MODEL_SPECS,
    )

    # Marked blocks per class (adjustments plus targets); this pins the
    # compiled design widths the follow-up family adds to a session.
    widths = {name: len(spec.all_marks) for name, spec in FOLLOWUP_MODEL_SPECS.items()}
    assert widths == {
        "replay_content": 6,
        "event_strength": 4,
        "stop_context": 5,
        "ripple_power": 2,
        "ripple_power_pooled": 2,
    }
    for spec in FOLLOWUP_MODEL_SPECS.values():
        assert set(spec.all_marks) <= set(MARK_DESCRIPTIONS)
    for name in ("replay_content", "event_strength", "stop_context"):
        assert FOLLOWUP_MODEL_SPECS[name].event_classes == MARKED_EVENT_CLASSES
    for name in ("ripple_power", "ripple_power_pooled"):
        assert FOLLOWUP_MODEL_SPECS[name].event_classes == EXTENDED_MARKED_EVENT_CLASSES
    assert FOLLOWUP_MODEL_SPECS["ripple_power_pooled"].pooled
    assert FOLLOWUP_MODEL_SPECS["stop_context"].baseline_context_marks == (
        TRIPS_TO_PATCH_SWITCH,
    )
    # The v1 inventory is untouched.
    assert len(RECOMMENDED_MODEL_SPECS) == 12
    # A canonical-set session fits the ripple-power models on the replay
    # classes only.
    restricted = FOLLOWUP_MODEL_SPECS["ripple_power"].for_classes(MARKED_EVENT_CLASSES)
    assert restricted.event_classes == MARKED_EVENT_CLASSES


def test_marginal_specs_cover_every_followup_target_once():
    from continuum_swr_replay.analysis.pfc.marked_model_family import (
        FOLLOWUP_MODEL_SPECS,
        MARGINAL_MODEL_NAMES,
        MARGINAL_MODEL_SPECS,
        MUA_RATE,
    )

    targets = {
        m for spec in FOLLOWUP_MODEL_SPECS.values() for m in spec.scientific_marks
    }
    assert set(MARGINAL_MODEL_NAMES) == {
        f"marginal_{mark}" for mark in (*targets, NEXT_TRIP_SWITCHES_PATCH)
    }
    for name, spec in MARGINAL_MODEL_SPECS.items():
        assert spec.tier == "marginal"
        assert len(spec.scientific_marks) == 1
        assert name == f"marginal_{spec.scientific_marks[0]}"
        assert spec.event_classes == MARKED_EVENT_CLASSES
    duration = MARGINAL_MODEL_SPECS[f"marginal_{EVENT_DURATION_MARK}"]
    assert duration.nuisance_marks == (DECODE_ENTROPY,)
    for content in (CURRENT_ARM_FRACTION, CONTENT_RECENT_REWARD, CONTENT_UNCERTAINTY):
        assert MARGINAL_MODEL_SPECS[f"marginal_{content}"].nuisance_marks == (
            EVENT_DURATION_MARK,
            LEAF_CONTENT_FRACTION,
            DECODE_ENTROPY,
        )
    for other in (RIPPLE_POWER, MUA_RATE, REWARDED_CONTEXT, TRIPS_TO_PATCH_SWITCH):
        assert MARGINAL_MODEL_SPECS[f"marginal_{other}"].nuisance_marks == (
            EVENT_DURATION_MARK,
            DECODE_ENTROPY,
        )
    # Stop-level context targets other than the reward outcome (already a
    # baseline column) are also baseline context columns.
    for mark in (TRIPS_TO_PATCH_SWITCH, NEXT_TRIP_SWITCHES_PATCH):
        assert MARGINAL_MODEL_SPECS[f"marginal_{mark}"].baseline_context_marks == (
            mark,
        )
    assert (
        MARGINAL_MODEL_SPECS[f"marginal_{REWARDED_CONTEXT}"].baseline_context_marks
        == ()
    )


def test_swrs_without_replay_carry_only_duration_and_ripple_power():
    from continuum_swr_replay.analysis.pfc.marked_glm import (
        EXTENDED_MARKED_EVENT_CLASSES,
    )

    with pytest.raises(ValueError, match="carry only"):
        MarkedFamilySpec(
            "content_on_swrs",
            (CURRENT_ARM_FRACTION,),
            (EVENT_DURATION_MARK,),
            "an SWR without replay has no content",
            event_classes=EXTENDED_MARKED_EVENT_CLASSES,
        )


def _three_class_inputs(n_events: int = 12):
    """Both replay classes and SWRs without replay, carrying ripple power."""
    from continuum_swr_replay.analysis.pfc.marked_glm import (
        EXTENDED_MARKED_EVENT_CLASSES,
    )

    fit_mask, basis, lags, _ = _eligibility_inputs(n_events)
    time = fit_mask.index.to_numpy()
    rng = np.random.default_rng(31)
    events = {
        event_class: pd.DataFrame(
            {
                "start_time": time[20 + 40 * np.arange(n_events) + 3 * index],
                EVENT_DURATION_MARK: rng.normal(-3.0, 0.5, n_events),
                RIPPLE_POWER: rng.normal(index, 1.0 + index, n_events),
            }
        )
        for index, event_class in enumerate(EXTENDED_MARKED_EVENT_CLASSES)
    }
    return fit_mask, basis, lags, events


def test_pooled_block_equals_the_per_class_blocks_shared_slope():
    from continuum_swr_replay.analysis.pfc.marked_glm import (
        EXTENDED_MARKED_EVENT_CLASSES,
    )
    from continuum_swr_replay.analysis.pfc.marked_model_family import (
        FOLLOWUP_MODEL_SPECS,
    )

    fit_mask, basis, lags, events = _three_class_inputs()
    per_class = build_marked_family_ladder(
        events, fit_mask, basis, lags, FOLLOWUP_MODEL_SPECS["ripple_power"]
    )
    pooled = build_marked_family_ladder(
        events, fit_mask, basis, lags, FOLLOWUP_MODEL_SPECS["ripple_power_pooled"]
    )

    assert per_class.target_classes == EXTENDED_MARKED_EVENT_CLASSES
    assert pooled.target_classes == ("pooled",)
    assert tuple(pooled.reduced) == (("pooled", RIPPLE_POWER),)
    n_basis = basis.shape[1]
    assert per_class.full.matrix.shape[1] - pooled.full.matrix.shape[1] == 2 * n_basis
    # Constraining the three per-class slopes to one curve sums their columns.
    shared = sum(
        per_class.full.matrix[:, per_class.full.block_slices[f"{c}:{RIPPLE_POWER}"]]
        for c in EXTENDED_MARKED_EVENT_CLASSES
    )
    np.testing.assert_allclose(
        pooled.full.matrix[:, pooled.full.block_slices[f"pooled:{RIPPLE_POWER}"]],
        shared,
        atol=1e-12,
    )
    # Every class keeps its own base kernel and duration block.
    for event_class in EXTENDED_MARKED_EVENT_CLASSES:
        for block in (f"{event_class}:event", f"{event_class}:{EVENT_DURATION_MARK}"):
            np.testing.assert_array_equal(
                pooled.full.matrix[:, pooled.full.block_slices[block]],
                per_class.full.matrix[:, per_class.full.block_slices[block]],
            )
    # Each class's mark is standardized within that class before pooling.
    for event_class in EXTENDED_MARKED_EVENT_CLASSES:
        z = pooled.events["pooled"].loc[
            pooled.events["pooled"].event_class == event_class, f"{RIPPLE_POWER}_z"
        ]
        assert z.mean() == pytest.approx(0.0, abs=1e-12)
        assert z.std(ddof=0) == pytest.approx(1.0)


def test_ripple_power_fit_reads_one_curve_per_class_or_one_pooled_curve():
    from continuum_swr_replay.analysis.pfc.marked_glm import (
        EXTENDED_MARKED_EVENT_CLASSES,
    )
    from continuum_swr_replay.analysis.pfc.marked_model_family import (
        FOLLOWUP_MODEL_SPECS,
    )

    fit_mask, basis, lags, events = _three_class_inputs()
    counts = np.random.default_rng(3).poisson(0.2, size=(int(fit_mask.sum()), 2))
    folds = np.repeat([0, 1], len(counts) // 2)
    for name, targets in (
        ("ripple_power", EXTENDED_MARKED_EVENT_CLASSES),
        ("ripple_power_pooled", ("pooled",)),
    ):
        ladder = build_marked_family_ladder(
            events, fit_mask, basis, lags, FOLLOWUP_MODEL_SPECS[name]
        )
        result = fit_marked_family_ladder(ladder, counts, folds, penalty_sigma=3.0)
        assert tuple(result.mark_curves) == targets
        assert tuple(result.typical_curves) == EXTENDED_MARKED_EVENT_CLASSES
        for target in targets:
            assert (
                f"{target}:{RIPPLE_POWER}: heldout_delta_deviance" in result.statistics
            )
