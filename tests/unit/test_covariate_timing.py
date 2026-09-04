"""Covariate-timing registry: leak detection with the right remedy per leak type."""

from __future__ import annotations

import pytest

from continuum_swr_replay.analysis.replay import model_specs as ms
from continuum_swr_replay.analysis.replay.covariate_timing import (
    KNOWN_ATOMS,
    assert_predictor_timing,
    timing_of,
)
from continuum_swr_replay.analysis.replay.per_bin_rate_model import _design_columns
from continuum_swr_replay.analysis.replay.uncertainty_content import (
    HORSERACE_PREDICTORS,
)


def test_rejects_derived_post_outcome():
    """A derived interaction whose source is post-outcome must be caught (the name does
    not contain the source scalar, so a string heuristic would miss it)."""
    with pytest.raises(ValueError, match="post-outcome"):
        assert_predictor_timing(
            ["reward_belief_mean_update_l1_all_ports_postoutcome_z_x_is_current_arm"]
        )
    # ...but only when the source really is post-outcome; relval is safe.
    assert_predictor_timing(
        ["relative_value_switch_minus_stay_preoutcome_z_x_is_current_arm"]
    )


def test_rejects_derived_future_referencing():
    for p in (
        "periswitch_index_trials_z_x_is_current_arm",
        "is_periswitch_post_x_is_current_arm",
    ):
        with pytest.raises(ValueError, match="future-referencing"):
            assert_predictor_timing([p])


def test_rate_swr_interaction_caught():
    """The patsy expansion of the S7/S9 is_swr*(...+rpe/update) rungs must resolve the
    post-outcome operand, not hide behind the is_swr gating factor."""
    assert timing_of("is_swr:rpe_postoutcome") == ["rpe_postoutcome"]
    with pytest.raises(ValueError, match="post-outcome"):
        assert_predictor_timing(["is_swr:rpe_postoutcome"])
    assert timing_of("is_swr:periswitch_index_trials") == [
        "periswitch_index_trials"
    ]
    with pytest.raises(ValueError, match="future-referencing"):
        assert_predictor_timing(["is_swr:periswitch_index_trials"])


def test_post_outcome_fixable_by_retrospective_but_future_is_not():
    # POST_OUTCOME clears with retrospective gating...
    assert_predictor_timing(["rpe_postoutcome"], retrospective=True)  # no raise
    assert_predictor_timing(
        ["reward_belief_mean_update_l1_all_ports_postoutcome_z_x_is_current_arm"],
        retrospective=True,
    )
    # ...FUTURE_REFERENCING does NOT — a temporal gate can't fix it.
    with pytest.raises(ValueError, match="future-referencing"):
        assert_predictor_timing(["periswitch_index_trials"], retrospective=True)
    with pytest.raises(ValueError, match="future-referencing"):
        assert_predictor_timing(
            ["periswitch_index_trials_z_x_is_current_arm"], retrospective=True
        )
    # Only descriptive clears future-referencing.
    assert_predictor_timing(["periswitch_index_trials"], descriptive=True)  # no raise


@pytest.mark.parametrize(
    "predictor",
    [
        "rpe_postoutcome",
        "reward_belief_mean_update_l1_alternative_patches_postoutcome",
    ],
)
def test_uncertainty_rate_post_outcome_names_require_retrospective_gate(predictor):
    with pytest.raises(ValueError, match="post-outcome"):
        assert_predictor_timing([predictor])
    assert_predictor_timing([predictor], retrospective=True)


def test_is_after_switch_is_safe():
    """Last-completed-trip switch status is past-only and never rejected."""
    assert_predictor_timing(["did_last_completed_trip_switch_patch"])  # no raise
    assert timing_of("did_last_completed_trip_switch_patch") == ["did_last_completed_trip_switch_patch"]


@pytest.mark.parametrize("predictor", ["is_rewarded_outcome", "is_omitted_outcome"])
def test_completion_indexed_outcome_flags_still_require_a_gate(predictor):
    """Their finite values are meaningful only after outcome availability.

    ``add_trial_context`` now leaves these fields missing between arrival and
    reward onset/poke-out. They remain POST_OUTCOME because whenever they are
    finite they encode the current realized outcome, so callers must preserve
    the physical gate rather than treating them as ordinary predictors.
    """
    with pytest.raises(ValueError, match="post-outcome"):
        assert_predictor_timing([predictor])
    # The gated path is how a fit legitimately uses them.
    assert_predictor_timing([predictor], retrospective=True)


def test_unknown_timing_atom_fails_closed():
    with pytest.raises(ValueError, match="unclassified timing atom"):
        assert_predictor_timing(["plausible_but_unregistered_covariate"])


@pytest.mark.parametrize("predictor", ["is_stay", "wsls_deviation"])
def test_eventual_choice_covariates_are_future_referencing(predictor):
    with pytest.raises(ValueError, match="future-referencing"):
        assert_predictor_timing([predictor], retrospective=True)
    assert_predictor_timing([predictor], descriptive=True)


def test_descriptive_whitelists_both_leak_types():
    assert_predictor_timing(
        [
            "rpe_postoutcome",
            "periswitch_index_trials",
            "reward_belief_mean_update_l1_all_ports_postoutcome_z_x_is_current_arm",
        ],
        descriptive=True,
    )  # no raise


def _all_ladder_predictors():
    preds: set[str] = set()
    for lad in (
        ms.SWR_CONTRAST_CONTENT_ARM_LADDER,
        ms.SWR_CONTRAST_CONTENT_PATCH_LADDER,
    ):
        for _name, (pr, _scope) in lad.items():
            preds.update(pr)
    for lad in (ms.SWR_CONTRAST_RATE_G_SPECS, ms.SWR_CONTRAST_RATE_S_SPECS):
        for _name, (_formula, need) in lad.items():
            preds.update(need)
    for _name, spec in ms.MODEL_PER_STOP_RATE.items():
        preds.update(spec.predictors)
    for family in (
        ms.MODEL_HIERARCHY,
        ms.MODEL_HIERARCHY_SWR,
        ms.MODEL_HIERARCHY_MUA,
        # Previously unenumerated: the patch, switch and choice families are
        # shipped and fitted, so an unclassified atom in any of them must fail
        # here rather than at whichever runner happens to load it.
        ms.MODEL_PATCH,
        ms.MODEL_PATCH_SWR,
        ms.MODEL_PATCH_MUA,
        ms.MODEL_PATCH_SWITCH,
        ms.MODEL_CHOICE,
    ):
        for _name, spec in family.items():
            preds.update(spec.predictors)
    for _name, spec in ms.MODEL_PER_BIN_RATE.items():
        preds.update(c for c in _design_columns(spec) if c != "count")
    # The within-arm horse-race is not a ModelSpec ladder, so enumerating the
    # spec families misses it -- and its coefficients are cited. Include it, or a
    # leaky predictor added there stays invisible to every completeness check.
    preds.update(HORSERACE_PREDICTORS)
    return preds


def test_every_ladder_predictor_resolvable():
    """Every predictor in every shipped ladder resolves via timing_of into KNOWN_ATOMS —
    a new interaction with no source map (or a stray covariate) fails here rather than
    passing silently as its own name."""
    unresolved = {}
    for p in _all_ladder_predictors():
        bad = [a for a in timing_of(p) if a not in KNOWN_ATOMS]
        if bad:
            unresolved[p] = bad
    assert not unresolved, f"predictors with atoms outside KNOWN_ATOMS: {unresolved}"
