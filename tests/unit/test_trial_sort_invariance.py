"""Regression test: trial-history helpers must be sort-invariant.

The helpers use ``np.searchsorted`` on raw ``trials`` columns, which
silently assumes chronological order. Spyglass-loaded or manually
passed frames are not guaranteed to be sorted; the user reproduced
different ``current_well`` / ``visit_recency`` / ``recent_reward_rate``
values from the same trials sorted vs unsorted.

The fix sorts defensively at the head of each helper. This test
shuffles the trial order and asserts the helper outputs match the
sorted-trial baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.replay._helpers import (
    add_trial_context,
    cumulative_reward_all_events,
    recent_reward_rate_all_events,
    visit_recency_all_events,
)


def _toy_trials(n: int = 6) -> pd.DataFrame:
    """6 chronologically-ordered trials with reward and patch-change patterns."""
    return pd.DataFrame(
        {
            "start_time": np.arange(n, dtype=float),
            "end_time": np.arange(n, dtype=float) + 0.5,
            "from_well": [0, 1, 2, 3, 4, 5][:n],
            "to_well": [1, 2, 3, 4, 5, 0][:n],
            "is_reward": [True, False, True, False, True, False][:n],
            "is_patch_change": [False, True, False, True, False, True][:n],
        }
    )


def _well_to_arm() -> dict[int, int]:
    # 6 wells, leaf-arm map 0→0, 1→1, …, 5→5 (matches degree-1 wells).
    return {w: w for w in range(6)}


class TestTrialSortInvariance:
    def test_add_trial_context_sort_invariant(self):
        sorted_trials = _toy_trials()
        shuffled = sorted_trials.iloc[[3, 0, 5, 1, 4, 2]].reset_index(drop=True)

        dot_times = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
        dot_sorted = pd.DataFrame(index=dot_times)
        dot_shuffled = pd.DataFrame(index=dot_times)

        out_sorted = add_trial_context(dot_sorted, sorted_trials)
        out_shuffled = add_trial_context(dot_shuffled, shuffled)

        for col in (
            "prev_well", "current_well", "next_well", "last_rewarded_well",
            "is_after_switch", "is_after_reward",
        ):
            np.testing.assert_array_equal(
                out_sorted[col].to_numpy(), out_shuffled[col].to_numpy(),
                err_msg=f"{col} differs after shuffle",
            )

    def test_visit_recency_sort_invariant(self):
        sorted_trials = _toy_trials()
        shuffled = sorted_trials.iloc[[3, 0, 5, 1, 4, 2]].reset_index(drop=True)
        event_times = np.array([0.5, 2.5, 4.5])

        out_sorted = visit_recency_all_events(
            event_times, sorted_trials, _well_to_arm()
        )
        out_shuffled = visit_recency_all_events(
            event_times, shuffled, _well_to_arm()
        )
        np.testing.assert_array_equal(out_sorted, out_shuffled)

    def test_cumulative_reward_sort_invariant(self):
        sorted_trials = _toy_trials()
        shuffled = sorted_trials.iloc[[3, 0, 5, 1, 4, 2]].reset_index(drop=True)
        event_times = np.array([0.5, 2.5, 4.5])

        out_sorted = cumulative_reward_all_events(
            event_times, sorted_trials, _well_to_arm()
        )
        out_shuffled = cumulative_reward_all_events(
            event_times, shuffled, _well_to_arm()
        )
        np.testing.assert_array_equal(out_sorted, out_shuffled)

    def test_recent_reward_rate_sort_invariant(self):
        sorted_trials = _toy_trials()
        shuffled = sorted_trials.iloc[[3, 0, 5, 1, 4, 2]].reset_index(drop=True)
        event_times = np.array([0.5, 2.5, 4.5])

        out_sorted = recent_reward_rate_all_events(
            event_times, sorted_trials, _well_to_arm()
        )
        out_shuffled = recent_reward_rate_all_events(
            event_times, shuffled, _well_to_arm()
        )
        # Element-wise equality, treating NaN==NaN.
        np.testing.assert_array_equal(
            np.where(np.isnan(out_sorted), -1, out_sorted),
            np.where(np.isnan(out_shuffled), -1, out_shuffled),
        )
