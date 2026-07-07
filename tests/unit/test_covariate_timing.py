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


def test_rejects_derived_post_outcome():
    """A derived interaction whose source is post-outcome must be caught (the name does
    not contain the source scalar, so a string heuristic would miss it)."""
    with pytest.raises(ValueError, match="post-outcome"):
        assert_predictor_timing(["update_x_current"])  # source hmm_value_update
    # ...but only when the source really is post-outcome; relval is safe.
    assert_predictor_timing(["relval_x_current"])  # hmm_relative_value -> no raise


def test_rejects_derived_future_referencing():
    for p in ("tss_x_current", "postsw_x_current"):
        with pytest.raises(ValueError, match="future-referencing"):
            assert_predictor_timing([p])


def test_rate_swr_interaction_caught():
    """The patsy expansion of the S7/S9 is_swr*(...+rpe/update) rungs must resolve the
    post-outcome operand, not hide behind the is_swr gating factor."""
    assert timing_of("is_swr:hmm_rpe") == ["hmm_rpe"]
    with pytest.raises(ValueError, match="post-outcome"):
        assert_predictor_timing(["is_swr:hmm_rpe"])
    assert timing_of("is_swr:trials_since_switch") == ["trials_since_switch"]
    with pytest.raises(ValueError, match="future-referencing"):
        assert_predictor_timing(["is_swr:trials_since_switch"])


def test_post_outcome_fixable_by_retrospective_but_future_is_not():
    # POST_OUTCOME clears with retrospective gating...
    assert_predictor_timing(["hmm_rpe"], retrospective=True)  # no raise
    assert_predictor_timing(["update_x_current"], retrospective=True)  # no raise
    # ...FUTURE_REFERENCING does NOT — a temporal gate can't fix it.
    with pytest.raises(ValueError, match="future-referencing"):
        assert_predictor_timing(["trials_since_switch"], retrospective=True)
    with pytest.raises(ValueError, match="future-referencing"):
        assert_predictor_timing(["tss_x_current"], retrospective=True)
    # Only descriptive clears future-referencing.
    assert_predictor_timing(["trials_since_switch"], descriptive=True)  # no raise


def test_is_after_switch_is_safe():
    """Destination is chosen at departure -> pre-outcome-safe, never rejected."""
    assert_predictor_timing(["is_after_switch"])  # no raise
    assert timing_of("is_after_switch") == ["is_after_switch"]


def test_descriptive_whitelists_both_leak_types():
    assert_predictor_timing(
        ["hmm_rpe", "trials_since_switch", "update_x_current"], descriptive=True
    )  # no raise


def _all_ladder_predictors():
    preds: set[str] = set()
    for lad in (ms.SWR_CONTRAST_CONTENT_ARM_LADDER, ms.SWR_CONTRAST_CONTENT_PATCH_LADDER):
        for _name, (pr, _scope) in lad.items():
            preds.update(pr)
    for lad in (ms.SWR_CONTRAST_RATE_G_SPECS, ms.SWR_CONTRAST_RATE_S_SPECS):
        for _name, (_formula, need) in lad.items():
            preds.update(need)
    for _name, spec in ms.MODEL_PER_STOP_RATE.items():
        preds.update(spec.predictors)
    for _name, spec in ms.MODEL_HIERARCHY.items():
        preds.update(spec.predictors)
    for _name, spec in ms.MODEL_PER_BIN_RATE.items():
        preds.update(c for c in _design_columns(spec) if c != "count")
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
