"""Novel-arm controls accompany every recency term in regression inputs."""

from continuum_swr_replay.analysis.behavior import model_specs as specs
from continuum_swr_replay.analysis.behavior.covariate_timing import timing_of


def _has_recency(predictors) -> bool:
    """Both the linear and log-scale recency forms carry the same hazard."""
    return any(
        p in predictors
        for p in (
            "visit_recency_trials_event_leaf_centered",
            *specs.RECENCY_SPLINE_COLUMNS,
        )
    )


def test_arm_hierarchy_adds_novelty_main_effect_with_recency():
    for name in ("M4", "M5", "M6", "Mv", "Mu", "Mvu"):
        assert any(
            c in specs.MODEL_HIERARCHY[name].predictors
            for c in specs.RECENCY_SPLINE_COLUMNS
        )
        assert (
            "is_candidate_arm_never_visited" in specs.MODEL_HIERARCHY[name].predictors
        )
    assert any(
        f"swr_x_{c}" in specs.MODEL_HIERARCHY["M6"].predictors
        for c in specs.RECENCY_SPLINE_COLUMNS
    )
    # The sparse interaction remains materialized for sensitivity analyses but
    # is not in the primary ladder: it makes five otherwise estimable shipped
    # sessions rank-deficient and would silently change the M5->M6 session set.
    assert "swr_x_arm_never_visited" not in specs.MODEL_HIERARCHY["M6"].predictors


def test_choice_and_contrast_bases_add_novelty_alongside_recency():
    for model in specs.MODEL_CHOICE.values():
        if "visit_recency_trials" in model.predictors:
            assert "is_candidate_arm_never_visited" in model.predictors
    assert "is_candidate_arm_never_visited" in specs.SWR_CONTRAST_CONTENT_BASEC
    assert "is_patch_never_visited" in specs.SWR_CONTRAST_CONTENT_BASEC_PATCH


def test_primary_choice_and_patch_models_use_rate_and_learned_value():
    choice = set(specs.MODEL_CHOICE["C_behavior"].predictors)
    assert "reward_belief_mean_candidate_arm_preoutcome" in choice
    assert "reward_fraction_candidate_arm_last10_visits" in choice
    assert "reward_count_candidate_arm_last10_known_trials" not in choice
    # Nominal contingency is retained in the table as a distinct sensitivity
    # variable, not co-fit with learned Q in the manuscript-aligned baseline.
    assert "task_reward_probability_candidate_arm" not in choice

    for name in ("P2", "P3", "Pv"):
        predictors = set(specs.MODEL_PATCH[name].predictors)
        assert "reward_fraction_candidate_arm_last10_visits_patch" in predictors
        assert "reward_count_candidate_arm_last10_known_trials_patch" not in predictors


def test_novelty_timing_is_past_only():
    assert timing_of("is_candidate_arm_never_visited") == [
        "is_candidate_arm_never_visited"
    ]
    assert timing_of("is_patch_never_visited") == ["is_patch_never_visited"]
    assert timing_of("swr_x_is_candidate_arm_never_visited") == [
        "is_candidate_arm_never_visited"
    ]


def test_every_recency_rung_carries_the_novelty_companion():
    """Either recency form must never appear without ``arm_never_visited``.

    Centering excludes censored arms from the *mean*, but their own values are
    still the right-censoring rule (``n_completed + 1`` minus the visited
    mean): 4.78% of leaf rows, mean +9.66, max +163.5 against ~0 for visited
    arms. ``arm_never_visited`` is what absorbs that state, so a rung carrying
    the recency term alone would estimate roughly 5% of its slope off the
    censoring rule. The pairing currently holds everywhere by convention --
    this makes it a contract.
    """
    from continuum_swr_replay.analysis.behavior import model_specs as ms

    unpaired = []
    families = {
        "MODEL_HIERARCHY": ms.MODEL_HIERARCHY,
        "MODEL_HIERARCHY_SWR": ms.MODEL_HIERARCHY_SWR,
        "MODEL_HIERARCHY_MUA": ms.MODEL_HIERARCHY_MUA,
        "MODEL_PATCH": ms.MODEL_PATCH,
        "MODEL_CHOICE": ms.MODEL_CHOICE,
    }
    for family, ladder in families.items():
        for name, spec in ladder.items():
            predictors = set(spec.predictors)
            if _has_recency(predictors) and (
                "is_candidate_arm_never_visited" not in predictors
            ):
                unpaired.append(f"{family}:{name}")
    for name, (predictors, _scope) in ms.SWR_CONTRAST_CONTENT_ARM_LADDER.items():
        if _has_recency(predictors) and (
            "is_candidate_arm_never_visited" not in predictors
        ):
            unpaired.append(f"SWR_CONTRAST_ARM:{name}")

    assert not unpaired, (
        f"rungs carry a recency term without arm_never_visited: {unpaired}"
    )
