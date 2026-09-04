"""The immediate-switch flag is retired from general-purpose fitted models.

``is_after_switch`` is often constant within a session's complete-case cohort,
which makes the design rank-deficient and rejects the whole session. The
naming spec ("Immediate switch is not a nuisance adjustment") therefore removes
it from R1, from the per-bin stop covariates, and retires the M8
per-bin switch main-effect rung -- while preserving the *dedicated* post-switch
patch analysis, which uses the flag to define a cohort, not as a coefficient.
"""

from __future__ import annotations

from continuum_swr_replay.analysis.replay import covariate_registry as reg
from continuum_swr_replay.analysis.replay import model_specs as ms
from continuum_swr_replay.analysis.replay import per_bin_rate_model as pbrm


def test_per_stop_rate_ladders_drop_the_switch_flag() -> None:
    for name in ("R0", "R1"):
        assert "did_last_completed_trip_switch_patch" not in ms.MODEL_PER_STOP_RATE[name].predictors


def test_stop_covariates_drop_the_switch_flag() -> None:
    assert "did_last_completed_trip_switch_patch" not in ms.STOP_COVARIATES


def test_m8_switch_rung_is_gone() -> None:
    assert "M8_full_switch" not in ms.MODEL_PER_BIN_RATE


def test_no_per_bin_model_emits_the_switch_flag() -> None:
    """Neither the patsy formula nor the design-column builder may add it."""
    for name, spec in ms.MODEL_PER_BIN_RATE.items():
        assert "did_last_completed_trip_switch_patch" not in spec.formula(), name
        assert "did_last_completed_trip_switch_patch" not in pbrm._design_columns(spec), name
        assert spec.switch is False, name


def test_constructing_a_switch_spec_fails_loud() -> None:
    """The retired switch flag cannot be silently re-enabled."""
    import pytest

    with pytest.raises(ValueError, match="retired"):
        ms.PerBinRateModelSpec(
            "bad", True, 0, False, 0, True, False, "switch on", "M0_const"
        )


def test_dedicated_post_switch_patch_model_is_preserved() -> None:
    """The permitted dedicated analysis (cohort defined by the switch flag,
    not a nuisance coefficient) must survive."""
    assert ms.MODEL_PATCH_SWITCH, "the post-switch patch model was removed"


def test_registry_and_model_specs_agree_switch_is_not_a_predictor() -> None:
    """The naming registry marks is_after_switch as provenance, not a
    predictor; model_specs must not carry it in any fitted ladder. This pins
    the two files against drift -- if a future edit re-adds the flag to a
    ladder while leaving the registry note stale, one of these fails."""
    entry = next(
        e
        for e in reg.REGISTRY
        if e.canonical == "did_last_completed_trip_switch_patch"
    )
    assert entry.role == "provenance", "registry still calls the flag a predictor"
    fitted = (
        *ms.MODEL_PER_STOP_RATE.values(),
        *ms.MODEL_HIERARCHY.values(),
    )
    for spec in fitted:
        assert "did_last_completed_trip_switch_patch" not in getattr(spec, "predictors", ()), spec.name
    for spec in ms.MODEL_PER_BIN_RATE.values():
        assert "did_last_completed_trip_switch_patch" not in spec.formula(), spec.name


def test_ladder_parent_chain_survives_m8_removal() -> None:
    """M9 parents off M7, so removing M8 leaves the ladder connected."""
    assert ms.MODEL_PER_BIN_RATE["M9_rew_x_time"].parent == "M7_time_hist_rew"
    parents = {s.parent for s in ms.MODEL_PER_BIN_RATE.values()} - {None}
    assert "M8_full_switch" not in parents
