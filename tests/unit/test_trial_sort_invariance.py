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


class TestNoPastTrialsContract:
    """All three trial-history helpers must agree that an event with
    no completed trial behind it has UNKNOWN history (NaN), not a
    fabricated zero. Pre-fix ``cumulative_reward_all_events`` returned
    zeros for the no-past-trials case while its sibling helpers
    correctly returned NaN, letting M3 (+ ``cumulative_reward``) treat
    before-trial events as valid neutral observations.
    """

    def test_event_before_first_trial_yields_nan_cumulative_reward(self):
        # User repro: event at 0.5s before the first trial ending
        # at 1.5s. Pre-fix returned all-zero cumulative_reward.
        event_times = np.array([0.5])
        trials = pd.DataFrame(
            [{"end_time": 1.5, "to_well": 0, "is_reward": True}]
        )
        out = cumulative_reward_all_events(
            event_times, trials, _well_to_arm()
        )
        assert np.all(np.isnan(out)), (
            f"cumulative_reward returned {out} for an event with no "
            "past trials — should be all-NaN to match the contract "
            "enforced by visit_recency_all_events / "
            "recent_reward_rate_all_events."
        )

    def test_three_helpers_agree_on_no_past_trials_case(self):
        event_times = np.array([0.5])
        trials = pd.DataFrame(
            [{"end_time": 1.5, "to_well": 0, "is_reward": True}]
        )
        w2a = _well_to_arm()
        vr = visit_recency_all_events(event_times, trials, w2a)
        cr = cumulative_reward_all_events(event_times, trials, w2a)
        rrr = recent_reward_rate_all_events(event_times, trials, w2a)
        assert np.all(np.isnan(vr))
        assert np.all(np.isnan(cr))
        assert np.all(np.isnan(rrr))

    def test_post_trial_events_still_get_real_reward_counts(self):
        # Sanity: an event AFTER the first trial must still see the
        # real reward count (the no-past-trials NaN is gated on
        # ``cutoffs == 0``, not applied universally).
        event_times = np.array([0.5, 2.0])  # before, then after
        trials = pd.DataFrame(
            [{"end_time": 1.5, "to_well": 0, "is_reward": True}]
        )
        out = cumulative_reward_all_events(
            event_times, trials, _well_to_arm()
        )
        assert np.all(np.isnan(out[0]))
        # Event 1 is after a single rewarded trial to well 0 (arm 0).
        assert out[1, 0] == 1.0
        assert np.all(out[1, 1:] == 0.0)

    def test_zero_trials_returns_all_nan(self):
        # ``n_trials_total == 0`` early-return path: pre-fix returned
        # ``np.zeros``, post-fix returns NaN to match the sibling
        # helpers' contract for the same case.
        event_times = np.array([1.0, 2.0, 3.0])
        empty_trials = pd.DataFrame(
            {"end_time": [], "to_well": [], "is_reward": []}
        )
        out = cumulative_reward_all_events(
            event_times, empty_trials, _well_to_arm()
        )
        assert np.all(np.isnan(out))
