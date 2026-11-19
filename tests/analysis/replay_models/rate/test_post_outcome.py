"""The post-outcome rules of the per-bin rate models: the covariate mask and the bin gate."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay_models.model_specs import (
    MODEL_UNCERTAINTY_RATE,
    RATE_BIN_S,
)
from continuum_swr_replay.analysis.replay_models.rate.per_bin_rate_model import (
    _design_columns,
)
from continuum_swr_replay.analysis.replay_models.rate.post_outcome import (
    gate_bins_after_outcome,
    mask_post_outcome_covariates,
)

MU_UNCERTAINTY = MODEL_UNCERTAINTY_RATE["MU_uncertainty"]


@pytest.fixture
def stop_frame() -> pd.DataFrame:
    """Three stops: one starts before its outcome is known, two do not."""
    return pd.DataFrame(
        {
            "stop_id": [1, 2, 3],
            "stop_start": [10.0, 20.0, 30.0],
            # stop 2 starts 1 s before the outcome is known; stop 3 starts
            # exactly at it (not before -> keeps its post-outcome values).
            "outcome_known_time": [9.5, 21.0, 30.0],
            "occupancy_state": ["at_well", "at_well", "in_transit"],
            "rpe_postoutcome": [0.4, -0.6, 0.2],
            "reward_belief_mean_update_l1_all_ports_postoutcome": [0.1, 0.2, 0.3],
            "relative_value_switch_minus_stay_preoutcome": [1.0, 2.0, 3.0],
        }
    )


def test_post_outcome_gate_nans_only_pre_outcome_stops(stop_frame):
    gated, counts = mask_post_outcome_covariates(stop_frame)

    assert gated["rpe_postoutcome"].tolist()[0] == 0.4
    assert np.isnan(gated["rpe_postoutcome"].iloc[1])
    assert gated["rpe_postoutcome"].iloc[2] == 0.2
    update = gated["reward_belief_mean_update_l1_all_ports_postoutcome"]
    assert update.iloc[0] == 0.1
    assert np.isnan(update.iloc[1])
    assert update.iloc[2] == 0.3
    # Pre-outcome covariates are untouched on the same rows.
    assert gated["relative_value_switch_minus_stay_preoutcome"].tolist() == [
        1.0,
        2.0,
        3.0,
    ]
    assert counts["at_well"] == 1
    assert counts.get("in_transit", 0) == 0


def test_post_outcome_gate_does_not_mutate_the_input(stop_frame):
    mask_post_outcome_covariates(stop_frame)
    assert stop_frame["rpe_postoutcome"].tolist() == [0.4, -0.6, 0.2]


def test_post_outcome_gate_requires_the_outcome_clock(stop_frame):
    with pytest.raises(ValueError, match="outcome_known_time"):
        mask_post_outcome_covariates(stop_frame.drop(columns="outcome_known_time"))


def _with_mu_fit_columns(data: dict) -> pd.DataFrame:
    """Complete a tiny physical-gate fixture with the exact MU design schema."""
    design = pd.DataFrame(data)
    # ``tc`` is the within-stop bin center, not a filler: the gate audit
    # reports the retained time-course support, so a constant here would
    # make that report vacuous.
    if "tc" not in design and "binj" in design:
        design["tc"] = (design["binj"].to_numpy(dtype=float) + 0.5) * RATE_BIN_S
    for column in _design_columns(MU_UNCERTAINTY):
        if column not in design:
            design[column] = 0.0
    if "exposure" not in design:
        design["exposure"] = RATE_BIN_S
    return design


def test_post_outcome_bin_gate_uses_physical_left_edge_and_keeps_equality():
    design = _with_mu_fit_columns(
        {
            "stop_id": [1, 1, 1, 2, 2, 2],
            "stop_start": [10.0] * 6,
            "binj": [0, 1, 2, 0, 1, 2],
            "outcome_known_time": [10.0 + RATE_BIN_S] * 6,
            "trial_reward": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        }
    )

    gated, audit = gate_bins_after_outcome(design, MU_UNCERTAINTY)

    assert gated["binj"].tolist() == [1, 2, 1, 2]
    assert gated["response_time"].tolist() == pytest.approx(
        [
            10.0 + RATE_BIN_S,
            10.0 + 2 * RATE_BIN_S,
            10.0 + RATE_BIN_S,
            10.0 + 2 * RATE_BIN_S,
        ]
    )
    assert audit == {
        "n_bins_before_post_outcome_gate": 6,
        "n_bins_after_post_outcome_gate": 4,
        "n_rewarded_bins_before_post_outcome_gate": 3,
        "n_rewarded_bins_after_post_outcome_gate": 2,
        "n_omission_bins_before_post_outcome_gate": 3,
        "n_omission_bins_after_post_outcome_gate": 2,
        "n_bins_before_complete_case": 4,
        "n_rewarded_bins_before_complete_case": 2,
        "n_omission_bins_before_complete_case": 2,
        "n_bins_after_complete_case": 4,
        "n_rewarded_bins_after_complete_case": 2,
        "n_omission_bins_after_complete_case": 2,
        "n_bins_excluded_missing_timing": 0,
        "n_bins_excluded_pre_outcome": 2,
        "n_bins_excluded_incomplete_fit": 0,
        "n_bins_excluded_nonpositive_exposure": 0,
        "outcome_balance_valid": True,
        "outcome_balance_status": "both_outcomes_retained",
        "n_bins_after_outcome_balance_guard": 4,
        # Symmetric here by construction: both classes share a boundary.
        "min_retained_tc_rewarded": pytest.approx(1.5 * RATE_BIN_S),
        "median_retained_tc_rewarded": pytest.approx(2.0 * RATE_BIN_S),
        "min_retained_tc_omission": pytest.approx(1.5 * RATE_BIN_S),
        "median_retained_tc_omission": pytest.approx(2.0 * RATE_BIN_S),
        "min_retained_tc_omission_minus_rewarded": pytest.approx(0.0),
    }


def test_gate_audit_exposes_asymmetric_time_course_truncation():
    """An omission's later boundary shortens its retained time course.

    The spline's boundary knots are fixed, so a class whose early bins are
    all gated away contributes nothing to the low-``tc`` region and the
    basis stops being orthogonal to outcome. Counts alone cannot show this;
    the retained support can.
    """
    design = _with_mu_fit_columns(
        {
            "stop_id": [1, 1, 1, 1, 2, 2, 2, 2],
            "stop_start": [10.0] * 8,
            "binj": [0, 1, 2, 3, 0, 1, 2, 3],
            # Rewarded stop 1 clears at once; omission stop 2 only at bin 3.
            "outcome_known_time": [10.0] * 4 + [10.0 + 3 * RATE_BIN_S] * 4,
            "trial_reward": [1.0] * 4 + [0.0] * 4,
        }
    )

    _, audit = gate_bins_after_outcome(design, MU_UNCERTAINTY)

    assert audit["outcome_balance_valid"]
    assert audit["min_retained_tc_rewarded"] == pytest.approx(0.5 * RATE_BIN_S)
    assert audit["min_retained_tc_omission"] == pytest.approx(3.5 * RATE_BIN_S)
    assert audit["min_retained_tc_omission_minus_rewarded"] == pytest.approx(
        3.0 * RATE_BIN_S
    )


def test_post_outcome_bin_gate_fails_loud_without_physical_timing():
    design = pd.DataFrame({"stop_id": [1], "binj": [0]})
    with pytest.raises(ValueError, match="outcome_known_time"):
        gate_bins_after_outcome(design, MU_UNCERTAINTY)


def test_post_outcome_bin_gate_rejects_inconsistent_stop_boundary():
    design = pd.DataFrame(
        {
            "stop_id": [1, 1],
            "stop_start": [10.0, 10.0],
            "binj": [0, 1],
            "outcome_known_time": [10.0, 10.1],
            "trial_reward": [1.0, 1.0],
        }
    )
    with pytest.raises(ValueError, match="constant within stop_id"):
        gate_bins_after_outcome(design, MU_UNCERTAINTY)


def test_post_outcome_bin_gate_suppresses_one_outcome_survivor_set():
    design = _with_mu_fit_columns(
        {
            "stop_id": [1, 1, 2, 2],
            "stop_start": [10.0] * 4,
            "binj": [0, 1, 0, 1],
            "outcome_known_time": [10.0, 10.0, 11.0, 11.0],
            "trial_reward": [1.0, 1.0, 0.0, 0.0],
        }
    )

    gated, audit = gate_bins_after_outcome(design, MU_UNCERTAINTY)

    assert gated.empty
    assert audit["n_rewarded_bins_after_post_outcome_gate"] == 2
    assert audit["n_omission_bins_after_post_outcome_gate"] == 0
    assert audit["outcome_balance_valid"] is False
    assert audit["n_bins_after_outcome_balance_guard"] == 0


def test_post_outcome_bin_gate_judges_completeness_on_the_specs_columns():
    """A NaN in a column the rung fits removes the bin; one in another does not."""
    design = _with_mu_fit_columns(
        {
            "stop_id": [1, 1, 2, 2],
            "stop_start": [10.0] * 4,
            "binj": [0, 1, 0, 1],
            "outcome_known_time": [10.0] * 4,
            "trial_reward": [1.0, 1.0, 0.0, 0.0],
            "not_a_design_column": [np.nan, 0.0, 0.0, 0.0],
        }
    )
    fitted = _design_columns(MU_UNCERTAINTY)[-1]
    design.loc[3, fitted] = np.nan

    gated, audit = gate_bins_after_outcome(design, MU_UNCERTAINTY)

    assert gated["binj"].tolist() == [0, 1, 0]
    assert audit["n_bins_excluded_incomplete_fit"] == 1
