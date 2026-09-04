"""Tests for analysis/replay/fitting.py.

Pins the behavior of the zero-total-count event filter: both
``fit_conditional_poisson`` and ``fit_conditional_logit`` must drop
events where every arm has zero count, because the conditional
likelihood (and the logit's dominant-arm logic) is degenerate there.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay import fitting as fitting_mod
from continuum_swr_replay.analysis.replay.fitting import (
    fit_conditional_logit,
    fit_conditional_poisson,
)


def _make_long_df(event_arm_counts, predictor_values=None):
    """Build a long-format frame: one row per (event_id, arm).

    Parameters
    ----------
    event_arm_counts : list[list[int]]
        Outer list is events, inner is per-arm counts (fixed 3 arms for
        compactness). A row of all zeros is the zero-count event that
        both fitters must filter out.
    predictor_values : list[list[float]] | None
        Per-(event, arm) values for the single ``is_prev`` predictor.
        Defaults to 1 on arm 0, 0 elsewhere.
    """
    rows = []
    for ev, counts in enumerate(event_arm_counts):
        for arm, count in enumerate(counts):
            row = {
                "event_id": ev + 1,
                "candidate_arm": arm,
                "arm_count_hard": count,
                "is_prev_arm": 1.0 if arm == 0 else 0.0,
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    if predictor_values is not None:
        df["is_prev_arm"] = np.asarray(predictor_values).ravel()
    return df


def test_fit_conditional_logit_filters_zero_count_events():
    """fit_conditional_poisson drops events with total count == 0 at
    fitting.py:83-84. fit_conditional_logit must do the same, or
    zero-count events get an arbitrary 'dominant' arm via idxmax and
    enter the fit as spurious observations."""
    long_df = _make_long_df(
        event_arm_counts=[
            [0, 0, 0],  # event 1: zero total — must be filtered out
            [3, 1, 0],  # event 2
            [0, 2, 1],  # event 3
            [2, 0, 1],  # event 4
            [1, 1, 2],  # event 5
            [4, 1, 0],  # event 6
        ]
    )
    result = fit_conditional_logit(long_df, predictors=["is_prev_arm"])
    assert result is not None, "logit fit returned None on valid input"
    assert result["n_events"] == 5, (
        f"zero-count event not filtered: n_events={result['n_events']} "
        "(expected 5; event 1 had all-zero arm_count_hard)"
    )


def test_fit_conditional_poisson_filters_zero_count_events():
    """Same invariant for the Poisson fitter (sanity pin on existing
    behavior — the logit fix brings it in line with this function)."""
    long_df = _make_long_df(
        event_arm_counts=[
            [0, 0, 0],
            [3, 1, 0],
            [0, 2, 1],
            [2, 0, 1],
            [1, 1, 2],
            [4, 1, 0],
        ]
    )
    result = fit_conditional_poisson(long_df, predictors=["is_prev_arm"])
    assert result is not None
    assert result["n_events"] == 5


def test_null_model_filters_zero_count_events_like_non_null_model():
    """M0 must use the same non-degenerate event set as M1+.

    Otherwise M0/M1 comparisons see mismatched ``n_events`` and
    correctly invalidate every metric, even though the caller supplied
    the same input table to both fits.
    """
    long_df = _make_long_df(
        event_arm_counts=[
            [0, 0, 0],
            [3, 1, 0],
            [0, 2, 1],
            [2, 0, 1],
            [1, 1, 2],
            [4, 1, 0],
        ]
    )
    null = fit_conditional_poisson(long_df, predictors=[], n_categories=3)
    child = fit_conditional_poisson(long_df, predictors=["is_prev_arm"], n_categories=3)
    assert null["n_events"] == child["n_events"] == 5
    assert null["n_rows"] == 15


def test_fit_conditional_logit_ties_break_by_lowest_arm_id(monkeypatch):
    """Dominant-arm robustness fit should not depend on row order."""
    long_df = _make_long_df(
        event_arm_counts=[
            [2, 2, 0],
            [3, 1, 0],
            [0, 2, 1],
            [2, 0, 1],
            [1, 1, 2],
            [4, 1, 0],
        ],
        predictor_values=[
            [0.0, 1.0, 0.0],  # event 1 tie: arm 0 should be dominant
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ],
    )
    # Put tied arm 1 before tied arm 0; first-row idxmax would choose arm 1.
    event1 = long_df[long_df["event_id"] == 1].iloc[[1, 0, 2]]
    rest = long_df[long_df["event_id"] != 1]
    long_df = pd.concat([event1, rest], ignore_index=True)

    captured: dict[str, np.ndarray] = {}

    class FakeConditionalLogit:
        def __init__(self, y, X, groups):
            captured["y"] = np.asarray(y)
            captured["groups"] = np.asarray(groups)

        def fit(self, disp=0, **kwargs):
            # **kwargs so the stub tracks the real fit() signature. The fitters
            # pass maxiter (statsmodels' default of 100 leaves the larger rungs
            # short of the optimum); a stub that accepts only what it was written
            # against fails on the call rather than on the behaviour under test.
            captured["fit_kwargs"] = kwargs

            class Result:
                params = np.array([0.0])
                bse = np.array([1.0])
                pvalues = np.array([1.0])
                llf = -1.0

            return Result()

    monkeypatch.setattr(fitting_mod, "ConditionalLogit", FakeConditionalLogit)

    result = fit_conditional_logit(long_df, predictors=["is_prev_arm"])
    assert result is not None
    event1_y = captured["y"][captured["groups"] == 1]
    # Event 1 rows are ordered arm 1, arm 0, arm 2. Lowest-arm tie-break
    # should therefore mark the second row, not the first.
    np.testing.assert_array_equal(event1_y, np.array([0, 1, 0]))


def test_rank_deficient_design_is_rejected():
    """A perfectly collinear (duplicated) predictor must be rejected, not
    fit with finite-but-meaningless coefficients. Enough events so the
    n_events >= 2*n_predictors gate passes and the rank check is what
    triggers the None."""
    # 6 events, 3 arms; arm 0 carries the count so the dominant-arm logit
    # is well defined; all events non-zero count.
    long_df = _make_long_df([[3, 1, 1]] * 6)
    long_df["is_prev_dup"] = long_df["is_prev_arm"]  # exact duplicate -> rank 1 of 2

    preds = ["is_prev_arm", "is_prev_dup"]
    assert fit_conditional_poisson(long_df, predictors=preds) is None
    assert fit_conditional_logit(long_df, predictors=preds) is None

    # Sanity: the single non-duplicated predictor is full-rank and fits.
    assert fit_conditional_poisson(long_df, predictors=["is_prev_arm"]) is not None


def test_within_group_constant_predictor_is_rejected():
    """A predictor that varies across events but is CONSTANT within every
    event carries no within-group information for a conditional model;
    statsmodels would fit it with an exploding SE. The within-group rank
    check must reject it (a raw matrix_rank(X) would not, since the column
    is globally full-rank)."""
    long_df = _make_long_df([[3, 1, 1]] * 6)
    # "ev_level" is constant within each event (all 3 arms share the event
    # id) but varies across events -> full global rank, zero within-group.
    long_df["ev_level"] = long_df["event_id"].astype(float)

    assert fit_conditional_poisson(long_df, predictors=["ev_level"]) is None
    assert fit_conditional_logit(long_df, predictors=["ev_level"]) is None


# ── Collinearity diagnostics ──


def test_condition_number_is_scale_invariant_and_on_the_design_not_the_gram():
    """The reported number must be comparable with the conventional bands.

    Two variants are wrong here and both were live at some point: ``cond(X)``
    unscaled reports the columns' arbitrary relative units, and
    ``cond(X.T @ X)`` squares whatever that is. Only the column-scaled design
    condition number is comparable to the 10/30 bands -- and to the same
    column emitted by ``within_arm_horserace``.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3))

    plain = fitting_mod._compute_conditional_diagnostics(X, ["a", "b", "c"])
    # Rescaling one column by 1000 changes its units, not its collinearity.
    rescaled = X.copy()
    rescaled[:, 1] *= 1000.0
    scaled = fitting_mod._compute_conditional_diagnostics(rescaled, ["a", "b", "c"])

    assert plain["cond_number"] == pytest.approx(scaled["cond_number"], rel=1e-6)
    # Near-orthogonal columns sit far below the warning band; the Gram-matrix
    # variant would report roughly this squared.
    assert plain["cond_number"] < 5.0


def test_collinear_design_warns_without_refusing_to_fit(caplog):
    """A structurally collinear rung is reported, not withheld."""
    rng = np.random.default_rng(1)
    base = rng.normal(size=(500, 1))
    # Second column is the first plus a whisker of noise: near-dependent.
    X = np.column_stack([base, base + rng.normal(scale=1e-3, size=(500, 1))])
    diagnostics = fitting_mod._compute_conditional_diagnostics(X, ["recency", "novelty"])

    assert diagnostics["cond_number"] > fitting_mod.COND_NUMBER_WARN
    with caplog.at_level(logging.WARNING, logger=fitting_mod.__name__):
        fitting_mod._warn_on_collinear_design(diagnostics, ["recency", "novelty"])
    assert any("Collinear design" in r.message for r in caplog.records)
    assert any("condition number" in r.message for r in caplog.records)


def test_well_conditioned_design_is_silent(caplog):
    rng = np.random.default_rng(2)
    X = rng.normal(size=(400, 3))
    diagnostics = fitting_mod._compute_conditional_diagnostics(X, ["a", "b", "c"])
    with caplog.at_level(logging.WARNING, logger=fitting_mod.__name__):
        fitting_mod._warn_on_collinear_design(diagnostics, ["a", "b", "c"])
    assert not [r for r in caplog.records if "Collinear design" in r.message]


def test_zero_variance_column_does_not_divide_by_zero():
    """An arm indicator absent from a session is structurally zero."""
    rng = np.random.default_rng(3)
    X = np.column_stack([rng.normal(size=200), np.zeros(200)])
    diagnostics = fitting_mod._compute_conditional_diagnostics(X, ["a", "absent_arm"])
    assert np.isfinite(diagnostics["cond_number"]) or np.isinf(
        diagnostics["cond_number"]
    )
