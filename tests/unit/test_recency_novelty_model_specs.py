"""Novel-arm controls accompany every recency term in regression inputs."""

from continuum_swr_replay.analysis.replay import model_specs as specs
from continuum_swr_replay.analysis.replay.covariate_timing import timing_of


def test_arm_hierarchy_adds_novelty_main_effect_with_recency():
    for name in ("M4", "M5", "M6", "M7", "Mv", "Mu", "Mvu"):
        assert "visit_recency_demeaned" in specs.MODEL_HIERARCHY[name].predictors
        assert "arm_never_visited" in specs.MODEL_HIERARCHY[name].predictors
    assert "swr_x_visit_recency_demeaned" in specs.MODEL_HIERARCHY["M7"].predictors
    # The sparse interaction remains materialized for sensitivity analyses but
    # is not in the primary ladder: it makes five otherwise estimable shipped
    # sessions rank-deficient and would silently change the M6->M7 session set.
    assert "swr_x_arm_never_visited" not in specs.MODEL_HIERARCHY["M7"].predictors


def test_choice_and_contrast_bases_add_novelty_alongside_recency():
    for model in specs.MODEL_CHOICE.values():
        if "visit_recency" in model.predictors:
            assert "arm_never_visited" in model.predictors
    assert "arm_never_visited" in specs.SWR_CONTRAST_CONTENT_BASEC
    assert "arm_never_visited_patch" in specs.SWR_CONTRAST_CONTENT_BASEC_PATCH


def test_novelty_timing_is_past_only():
    assert timing_of("arm_never_visited") == ["arm_never_visited"]
    assert timing_of("arm_never_visited_patch") == ["arm_never_visited_patch"]
    assert timing_of("swr_x_arm_never_visited") == ["arm_never_visited"]
