"""Tests for analysis/replay/fitting.py.

Pins the behavior of the zero-total-count event filter: both
``fit_conditional_poisson`` and ``fit_conditional_logit`` must drop
events where every arm has zero count, because the conditional
likelihood (and the logit's dominant-arm logic) is degenerate there.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

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
                "arm": arm,
                "arm_count_hard": count,
                "is_prev": 1.0 if arm == 0 else 0.0,
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    if predictor_values is not None:
        df["is_prev"] = np.asarray(predictor_values).ravel()
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
    result = fit_conditional_logit(long_df, predictors=["is_prev"])
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
    result = fit_conditional_poisson(long_df, predictors=["is_prev"])
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
    child = fit_conditional_poisson(long_df, predictors=["is_prev"], n_categories=3)
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

        def fit(self, disp=0):
            class Result:
                params = np.array([0.0])
                bse = np.array([1.0])
                pvalues = np.array([1.0])
                llf = -1.0

            return Result()

    monkeypatch.setattr(fitting_mod, "ConditionalLogit", FakeConditionalLogit)

    result = fit_conditional_logit(long_df, predictors=["is_prev"])
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
    long_df["is_prev_dup"] = long_df["is_prev"]  # exact duplicate -> rank 1 of 2

    preds = ["is_prev", "is_prev_dup"]
    assert fit_conditional_poisson(long_df, predictors=preds) is None
    assert fit_conditional_logit(long_df, predictors=preds) is None

    # Sanity: the single non-duplicated predictor is full-rank and fits.
    assert fit_conditional_poisson(long_df, predictors=["is_prev"]) is not None
