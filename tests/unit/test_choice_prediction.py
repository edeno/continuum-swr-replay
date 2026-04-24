"""Tests for analysis/replay/choice_prediction.py.

Pins the current_arm NaN-exclusion behavior: a stop whose
``current_arm`` is NaN must not contribute a 9-arm choice set (the
``arm != current_arm`` filter silently keeps all 9 when current_arm is
NaN, because ``arm != NaN`` evaluates True in IEEE 754).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.replay.choice_prediction import (
    build_choice_dataset,
)


def _make_minimal_tables(stops):
    """Construct minimal event_table / event_arm_table / stop_table for
    build_choice_dataset.

    ``stops`` is a list of dicts with at least ``stop_id``,
    ``current_arm``, ``next_arm``. Other stop_table columns are filled
    with safe defaults. One event per stop; every arm gets a
    uniform replay fraction.
    """
    stop_table = pd.DataFrame(
        [
            {
                "animal": "test",
                "day": 1,
                "epoch": "02_r1",
                "has_replay": True,
                "n_events": 1,
                "reward_outcome": 0,
                "is_after_reward": False,
                "is_after_omission": False,
                **s,
            }
            for s in stops
        ]
    )
    # One event per stop.
    event_table = pd.DataFrame(
        {
            "event_id": range(1, len(stops) + 1),
            "stop_id": [s["stop_id"] for s in stops],
        }
    )
    # Nine arms per event with uniform replay fraction.
    arms = list(range(9))
    event_arm_rows = []
    for s in stops:
        for arm in arms:
            event_arm_rows.append(
                {
                    "event_id": s["stop_id"],
                    "stop_id": s["stop_id"],
                    "arm": arm,
                    "arm_fraction": 1.0 / 9,
                    "visit_recency": 0.0,
                    "cumulative_reward": 0.0,
                    "recent_reward_rate": 0.0,
                    "dist_to_arm": 1.0,
                    "is_same_patch": 0,
                }
            )
    event_arm_table = pd.DataFrame(event_arm_rows)
    return event_table, event_arm_table, stop_table


def test_build_choice_dataset_drops_stops_with_nan_current_arm():
    """arm != NaN is True for every arm in IEEE 754, so the
    ``choice_df[choice_df['arm'] != choice_df['current_arm']]`` filter
    silently keeps all 9 arms when current_arm is NaN. build_choice_dataset
    must filter such stops out at the stop level — they have no defined
    current arm to exclude from the 9-arm choice set, and including
    them with 9 arms instead of 8 inflates the baseline probability for
    those observations."""
    event_table, event_arm_table, stop_table = _make_minimal_tables(
        [
            {"stop_id": 1, "current_arm": 0, "next_arm": 3},
            {"stop_id": 2, "current_arm": np.nan, "next_arm": 2},
            {"stop_id": 3, "current_arm": 5, "next_arm": 2},
        ]
    )
    ds = build_choice_dataset(event_table, event_arm_table, stop_table)

    # Every surviving stop must have exactly 8 arms (9 minus the
    # excluded current_arm).
    sizes = ds.groupby("stop_id").size()
    assert (sizes == 8).all(), (
        f"stop sizes {sizes.to_dict()} — NaN-current_arm stop kept 9 arms"
    )
    # Stop 2 (NaN current_arm) must be absent entirely.
    assert 2 not in sizes.index, (
        "stop with NaN current_arm should have been filtered"
    )
    # The two valid stops must still be present.
    assert set(sizes.index) == {1, 3}
