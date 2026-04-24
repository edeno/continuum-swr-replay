"""Tests for analysis/replay/fitting.py.

Pins the behavior of the zero-total-count event filter: both
``fit_conditional_poisson`` and ``fit_conditional_logit`` must drop
events where every arm has zero count, because the conditional
likelihood (and the logit's dominant-arm logic) is degenerate there.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

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
    n_arms = 3
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
            [0, 0, 0],   # event 1: zero total — must be filtered out
            [3, 1, 0],   # event 2
            [0, 2, 1],   # event 3
            [2, 0, 1],   # event 4
            [1, 1, 2],   # event 5
            [4, 1, 0],   # event 6
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
