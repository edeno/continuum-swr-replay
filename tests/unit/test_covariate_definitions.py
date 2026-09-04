"""Contracts for the machine-readable scientific covariate catalog."""

from __future__ import annotations

import pytest

from continuum_swr_replay.analysis.replay.covariate_timing import (
    FUTURE_REFERENCING,
    KNOWN_ATOMS,
    POST_OUTCOME,
)
from continuum_swr_replay.analysis.replay.covariates import (
    COVARIATE_DEFINITIONS,
    POST_CHOICE_COVARIATES,
    Availability,
    QueryClock,
    definition_for,
    definitions_frame,
)


def test_every_model_facing_timing_atom_has_a_scientific_definition() -> None:
    assert set(KNOWN_ATOMS) <= set(COVARIATE_DEFINITIONS)


@pytest.mark.parametrize("name", sorted(KNOWN_ATOMS))
def test_definition_availability_matches_timing_registry(name: str) -> None:
    expected = (
        Availability.POST_CHOICE
        if name in POST_CHOICE_COVARIATES
        else Availability.POST_OUTCOME
        if name in POST_OUTCOME
        else Availability.FUTURE_REFERENCING
        if name in FUTURE_REFERENCING
        else Availability.PRE_OUTCOME
    )
    assert definition_for(name).availability is expected


def test_timing_class_membership_is_pinned() -> None:
    """Golden pin on the leak-class membership, independent of ``definitions.py``.

    ``test_definition_availability_matches_timing_registry`` is circular (both
    sides derive from the same ``.availability`` field), so it cannot catch a
    covariate being *mis-marked*. This snapshot fails loudly if any covariate's
    Availability flips into or out of the enforcement-relevant classes, since a
    silent PRE<->POST<->FUTURE change alters model-fitting eligibility gating.
    Update the literals below only alongside a deliberate scientific decision.
    """
    assert POST_OUTCOME == frozenset(
        {
            "completed_trial_reward",
            "is_omitted_outcome",
            "is_rewarded_outcome",
            "reward_belief_mean_chosen_arm_preoutcome",
            "reward_belief_mean_update_l1_all_ports_postoutcome",
            "reward_belief_mean_update_l1_alternative_patches_postoutcome",
            "rpe_postoutcome",
            "task_reward_probability_chosen_arm",
        }
    )
    assert FUTURE_REFERENCING == frozenset(
        {
            "chosen_arm_fraction",
            "contains_path_to_next",
            "ends_at_next",
            "is_next_arm",
            "is_periswitch_post",
            "is_stay",
            "next_arm",
            "periswitch_index_trials",
            "wsls_deviation",
        }
    )


def test_definitions_are_complete_human_readable_records() -> None:
    frame = definitions_frame()
    assert frame["name"].is_unique
    required_text = (
        "name",
        "description",
        "grain",
        "units",
        "support",
        "missingness",
        "producer",
    )
    for column in required_text:
        assert (
            frame[column]
            .map(lambda value: isinstance(value, str) and bool(value.strip()))
            .all()
        )


def test_representative_clocks_encode_the_scientific_estimands() -> None:
    assert definition_for("visit_recency_trials").query_clock is QueryClock.EVENT_START
    assert (
        definition_for("known_reward_outcome_lag_1").query_clock
        is QueryClock.STOP_START
    )
    assert definition_for("leaky_q_value_diff").query_clock is QueryClock.GOAL_ARRIVAL
    assert (
        definition_for("rpe_postoutcome").query_clock is QueryClock.OUTCOME_AVAILABILITY
    )
    assert definition_for("next_arm").query_clock is QueryClock.FUTURE_CHOICE


def test_unknown_covariate_definition_fails_loudly() -> None:
    with pytest.raises(KeyError, match="no scientific covariate definition"):
        definition_for("unreviewed_future_predictor")
