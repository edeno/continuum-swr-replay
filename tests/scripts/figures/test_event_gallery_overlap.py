"""Regression tests for the event gallery's trial lookups and footer."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.figures.generate_event_gallery import (
    _active_trip_at_time,
    _completed_trials_at,
    _outcome_known_trials_at,
    _regression_footer,
)


def test_gallery_separates_active_trip_from_completed_regression_history():
    """An underway trip is visible but its future outcome is not in history."""
    trials = pd.DataFrame(
        {
            "movement_start_time": [0.0, 4.0],
            "destination_first_poke_in_time": [3.0, 7.0],
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
            "movement_start_time": [0.0],
            "destination_first_poke_in_time": [3.0],
            "end_time": [3.5],
            "to_well": [1],
            "is_reward": [1],
        }
    )

    completed = _completed_trials_at(trials, 3.0)

    assert completed["to_well"].tolist() == [1]


def test_gallery_distinguishes_arrived_destination_from_pending_outcome():
    trials = pd.DataFrame(
        {
            "movement_start_time": [0.0],
            "destination_first_poke_in_time": [3.0],
            # 0.2 s pump latency, inside the observed distribution and under
            # OMISSION_KNOWN_LATENCY_SEC (see the guard in trial_outcome_times).
            "reward_on_time": [3.2],
            "outcome_known_time": [3.2],
            "end_time": [3.8],
            "to_well": [1],
            "is_reward": [1],
        }
    )

    # Arrived at 3.0, pump at 3.2: between the two the destination is known but
    # the outcome is not; at pump onset equality is eligible.
    assert _completed_trials_at(trials, 3.1)["to_well"].tolist() == [1]
    assert _outcome_known_trials_at(trials, 3.1).empty
    assert _outcome_known_trials_at(trials, 3.2)["to_well"].tolist() == [1]


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
            "occupancy_well_arm": pd.NA,
            "prev_arm": pd.NA,
            "next_arm": pd.NA,
            "last_rewarded_arm": pd.NA,
            "current_patch": pd.NA,
            "prev_patch": pd.NA,
            "next_patch": pd.NA,
            "is_mua": pd.NA,
            "did_last_completed_trip_switch_patch": np.nan,
            "is_rewarded_outcome": pd.NA,
            "time_since_last_reward_s": np.nan,
        }
    )

    footer = _regression_footer(row, event)

    assert "is_mua=n/a" in footer
    assert "after_switch=n/a" in footer
    assert "rewarded_outcome=n/a" in footer
