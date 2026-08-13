"""Regression tests for event-gallery interval selection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.figures.generate_event_gallery import (
    _active_trip_at_time,
    _completed_trials_at,
    _intervals_overlap_any,
    _regression_footer,
)


def test_gallery_overlap_ignores_point_intervals_on_either_side():
    """Gallery categories use the same positive-duration overlap convention."""
    starts = np.array([0.0, 1.0, 4.0])
    ends = np.array([2.0, 1.0, 6.0])
    other_starts = np.array([1.0, 5.0])
    other_ends = np.array([1.0, 7.0])

    observed = _intervals_overlap_any(starts, ends, other_starts, other_ends)

    assert observed.tolist() == [False, False, True]


def test_gallery_separates_active_trip_from_completed_regression_history():
    """An underway trip is visible but its future outcome is not in history."""
    trials = pd.DataFrame(
        {
            "start_time": [0.0, 4.0],
            "goal_arrival_time": [3.0, 7.0],
            "end_time": [3.5, 7.5],
            "from_well": [0, 1],
            "to_well": [1, 2],
            "is_reward": [1, 0],
        }
    )

    active = _active_trip_at_time(trials, 5.0)
    completed = _completed_trials_at(trials, 5.0)

    assert int(active["to_well"]) == 2
    assert completed["to_well"].tolist() == [1]
    assert completed["_completion_time"].tolist() == [3.0]


def test_gallery_completion_is_inclusive_at_goal_arrival():
    trials = pd.DataFrame(
        {
            "start_time": [0.0],
            "goal_arrival_time": [3.0],
            "end_time": [3.5],
            "to_well": [1],
            "is_reward": [1],
        }
    )

    completed = _completed_trials_at(trials, 3.0)

    assert completed["to_well"].tolist() == [1]


def test_gallery_footer_formats_unknown_boolean_context_as_na():
    row = pd.Series(
        {
            "category": "non_swr",
            "event_id": 1,
            "duration_ms": 30.0,
            "peak_nonlocal_prob": 0.9,
        }
    )
    event = pd.Series(
        {
            "current_arm": pd.NA,
            "prev_arm": pd.NA,
            "next_arm": pd.NA,
            "last_rewarded_arm": pd.NA,
            "current_patch": pd.NA,
            "prev_patch": pd.NA,
            "next_patch": pd.NA,
            "is_mua": pd.NA,
            "is_after_switch": np.nan,
            "is_after_reward": pd.NA,
            "is_after_omission": np.nan,
            "time_since_reward": np.nan,
            "reward_outcome": np.nan,
        }
    )

    footer = _regression_footer(row, event)

    assert "is_mua=n/a" in footer
    assert "after_switch=n/a" in footer
    assert "after_reward=n/a" in footer
    assert "after_omission=n/a" in footer
