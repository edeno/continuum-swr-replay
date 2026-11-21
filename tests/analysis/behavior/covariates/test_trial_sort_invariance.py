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
import pytest

from continuum_swr_replay.analysis.behavior.covariates import (
    add_trial_context as _add_trial_context,
)
from continuum_swr_replay.analysis.behavior.covariates import (
    lagged_trial_rewards as _lagged_trial_rewards,
)
from continuum_swr_replay.analysis.behavior.covariates import (
    recent_reward_rate_all_events as _recent_reward_rate_all_events,
)
from continuum_swr_replay.analysis.behavior.covariates import (
    recent_trial_reward_rate as _recent_trial_reward_rate,
)
from continuum_swr_replay.analysis.behavior.covariates import (
    rolling_reward_count_all_events as _rolling_reward_count_all_events,
)
from continuum_swr_replay.analysis.behavior.covariates import (
    visit_recency_all_events as _visit_recency_all_events,
)
from continuum_swr_replay.analysis.behavior.covariates.clocks import (
    trial_arrival_times,
)
from tests.support._port_experience_fixtures import (
    port_experiences_from_trials,
    with_outcome_known_time,
)


def _experiences(trials: pd.DataFrame) -> pd.DataFrame:
    return port_experiences_from_trials(trials, prepend_initial=False)


def add_trial_context(dot: pd.DataFrame, trials: pd.DataFrame) -> pd.DataFrame:
    trials = with_outcome_known_time(trials)
    return _add_trial_context(
        dot, trials, port_experiences_from_trials(trials, prepend_initial=True)
    )


def visit_recency_all_events(times, trials, well_to_arm):
    return _visit_recency_all_events(times, _experiences(trials), well_to_arm)


def rolling_reward_count_all_events(times, trials, well_to_arm):
    return _rolling_reward_count_all_events(times, _experiences(trials), well_to_arm)


def recent_reward_rate_all_events(times, trials, well_to_arm, k=10):
    return _recent_reward_rate_all_events(times, _experiences(trials), well_to_arm, k)


def recent_trial_reward_rate(times, trials, k=10):
    return _recent_trial_reward_rate(times, _experiences(trials), k)


def lagged_trial_rewards(times, trials, k=10):
    return _lagged_trial_rewards(times, _experiences(trials), k)


def _toy_trials(n: int = 6) -> pd.DataFrame:
    """6 chronologically-ordered trials with reward and patch-change patterns."""
    return pd.DataFrame(
        {
            "movement_start_time": np.arange(n, dtype=float),
            "destination_first_poke_in_time": np.arange(n, dtype=float) + 0.5,
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
            "prev_well",
            "last_visited_well",
            "next_well",
            "last_rewarded_well",
            "did_last_completed_trip_switch_patch",
        ):
            np.testing.assert_array_equal(
                out_sorted[col].to_numpy(),
                out_shuffled[col].to_numpy(),
                err_msg=f"{col} differs after shuffle",
            )

    def test_visit_recency_sort_invariant(self):
        sorted_trials = _toy_trials()
        shuffled = sorted_trials.iloc[[3, 0, 5, 1, 4, 2]].reset_index(drop=True)
        event_times = np.array([0.5, 2.5, 4.5])

        out_sorted = visit_recency_all_events(
            event_times, sorted_trials, _well_to_arm()
        )
        out_shuffled = visit_recency_all_events(event_times, shuffled, _well_to_arm())
        np.testing.assert_array_equal(out_sorted, out_shuffled)

    def test_cumulative_reward_sort_invariant(self):
        sorted_trials = _toy_trials()
        shuffled = sorted_trials.iloc[[3, 0, 5, 1, 4, 2]].reset_index(drop=True)
        event_times = np.array([0.5, 2.5, 4.5])

        out_sorted = rolling_reward_count_all_events(
            event_times, sorted_trials, _well_to_arm()
        )
        out_shuffled = rolling_reward_count_all_events(
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
    fabricated zero. Pre-fix ``rolling_reward_count_all_events`` returned
    zeros for the no-past-trials case while its sibling helpers
    correctly returned NaN, letting M3 (+ ``cumulative_reward``) treat
    before-trial events as valid neutral observations.
    """

    def test_event_before_first_trial_yields_nan_cumulative_reward(self):
        # User repro: event at 0.5s before the first trial ending
        # at 1.5s. Pre-fix returned all-zero cumulative_reward.
        event_times = np.array([0.5])
        trials = pd.DataFrame([{"end_time": 1.5, "to_well": 0, "is_reward": True}])
        out = rolling_reward_count_all_events(event_times, trials, _well_to_arm())
        assert np.all(np.isnan(out)), (
            f"cumulative_reward returned {out} for an event with no "
            "past trials — should be all-NaN to match the contract "
            "enforced by visit_recency_all_events / "
            "recent_reward_rate_all_events."
        )

    def test_three_helpers_agree_on_no_past_trials_case(self):
        event_times = np.array([0.5])
        trials = pd.DataFrame([{"end_time": 1.5, "to_well": 0, "is_reward": True}])
        w2a = _well_to_arm()
        vr = visit_recency_all_events(event_times, trials, w2a)
        cr = rolling_reward_count_all_events(event_times, trials, w2a)
        rrr = recent_reward_rate_all_events(event_times, trials, w2a)
        assert np.all(np.isnan(vr))
        assert np.all(np.isnan(cr))
        assert np.all(np.isnan(rrr))

    def test_post_trial_events_still_get_real_reward_counts(self):
        # Sanity: an event AFTER the first trial must still see the
        # real reward count (the no-past-trials NaN is gated on
        # ``cutoffs == 0``, not applied universally).
        event_times = np.array([0.5, 2.0])  # before, then after
        trials = pd.DataFrame([{"end_time": 1.5, "to_well": 0, "is_reward": True}])
        out = rolling_reward_count_all_events(event_times, trials, _well_to_arm())
        assert np.all(np.isnan(out[0]))
        # Event 1 is after a single rewarded trial to well 0 (arm 0).
        assert out[1, 0] == 1.0
        assert np.all(out[1, 1:] == 0.0)

    def test_destination_history_advances_at_arrival_but_reward_history_waits(self):
        """Arrival < query < pump exposes the destination, not the reward."""
        trials = pd.DataFrame(
            [
                {
                    "movement_start_time": 1.0,
                    "end_time": 3.0,
                    "destination_first_poke_in_time": 2.0,
                    # 0.2 s pump latency: within the refreshed distribution
                    # (max observed 0.2361 s) and under
                    # OMISSION_KNOWN_LATENCY_SEC, so
                    # the fixture exercises the arrival-vs-pump gap without
                    # tripping the latency guard in trial_outcome_times.
                    "reward_on_time": 2.2,
                    "outcome_known_time": 2.2,
                    "from_well": 0,
                    "to_well": 2,
                    "is_reward": True,
                    "is_patch_change": True,
                }
            ]
        )
        event_times = np.array([2.1, 2.2])
        vr = visit_recency_all_events(event_times, trials, _well_to_arm())
        cr = rolling_reward_count_all_events(event_times, trials, _well_to_arm())
        rrr = recent_reward_rate_all_events(event_times, trials, _well_to_arm())
        trial_rate = recent_trial_reward_rate(event_times, trials)
        lags = lagged_trial_rewards(event_times, trials, k=1)
        context = add_trial_context(pd.DataFrame(index=event_times), trials)

        # Destination/visit history advances at inclusive goal arrival.
        assert vr[0, 2] == 1.0
        assert context.loc[2.1, "last_visited_well"] == 2.0
        # Reward-bearing histories remain genuinely unknown before pump onset.
        assert np.isnan(cr[0]).all()
        assert np.isnan(rrr[0]).all()
        assert np.isnan(trial_rate[0])
        assert np.isnan(lags[0, 0])
        assert pd.isna(context.loc[2.1, "is_rewarded_outcome"])
        assert pd.isna(context.loc[2.1, "last_rewarded_well"])

        # Equality at pump onset is eligible across every reward-history path.
        assert cr[1, 2] == 1.0
        assert rrr[1, 2] == 1.0
        assert trial_rate[1] == 1.0
        assert lags[1, 0] == 1.0
        assert context.loc[2.2, "is_rewarded_outcome"] == 1.0
        assert context.loc[2.2, "last_rewarded_well"] == 2.0

    def test_arrival_clock_requires_the_destination_poke_in(self):
        """A trial frame without ``destination_first_poke_in_time`` is refused:
        every loaded trial frame carries it, and ``end_time`` (the final
        poke-out) is not the arrival."""
        trials = _toy_trials().drop(columns="destination_first_poke_in_time")
        with pytest.raises(ValueError, match="destination_first_poke_in_time"):
            trial_arrival_times(trials)

    @pytest.mark.parametrize("value", [np.nan, np.inf])
    def test_arrival_clock_refuses_a_missing_or_infinite_poke_in(self, value):
        """A trial whose destination poke-in is NaN or infinite is refused,
        not given its ``end_time``."""
        trials = _toy_trials()
        trials.loc[0, "destination_first_poke_in_time"] = value
        with pytest.raises(ValueError, match="destination_first_poke_in_time"):
            trial_arrival_times(trials)

    def test_zero_trials_returns_all_nan(self):
        # ``n_trials_total == 0`` early-return path: pre-fix returned
        # ``np.zeros``, post-fix returns NaN to match the sibling
        # helpers' contract for the same case.
        event_times = np.array([1.0, 2.0, 3.0])
        empty_trials = pd.DataFrame({"end_time": [], "to_well": [], "is_reward": []})
        out = rolling_reward_count_all_events(event_times, empty_trials, _well_to_arm())
        assert np.all(np.isnan(out))


class TestRewardRateArmDefinition:
    """Definition-level guards for the canonical reward_rate_arm_k10 column."""

    def test_k_window_truncates_at_k(self):
        """Only the last k visits count; k=10 and k=5 give different rates."""
        n = 12
        trials = pd.DataFrame(
            {
                "movement_start_time": np.arange(n, dtype=float),
                "end_time": np.arange(n, dtype=float) + 0.5,
                "from_well": [1] * n,
                "to_well": [0] * n,  # 12 visits, all to arm 0
                # rewarded on visits 3-7 (1-indexed); omitted elsewhere
                "is_reward": [
                    False,
                    False,
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                ],
                "is_patch_change": [False] * n,
            }
        )
        event_times = np.array([n + 0.5])
        w2a = _well_to_arm()
        r10 = recent_reward_rate_all_events(event_times, trials, w2a, k=10)
        r5 = recent_reward_rate_all_events(event_times, trials, w2a, k=5)
        # last 10 visits (3..12): 5 rewarded -> 0.5; last 5 (8..12): 0 -> 0.0
        assert r10[0, 0] == pytest.approx(0.5)
        assert r5[0, 0] == pytest.approx(0.0)

    def test_trunk_zero_vs_unvisited_leaf_encoded(self):
        """Trunk arms -> 0.0; never-visited leaf -> 0.0 too (mechanism-A encode).

        An unvisited leaf *with prior trials* is encoded 0.0 (no rewards
        observed on it yet), not NaN — only a no-trial-history event stays NaN. Here
        two trials precede the event at t=2.5, so arm 1 (never visited) is 0.0.
        """
        trials = pd.DataFrame(
            {
                "movement_start_time": [0.0, 1.0],
                "end_time": [0.5, 1.5],
                "from_well": [1, 1],
                "to_well": [0, 0],  # only arm 0 visited
                "is_reward": [True, False],
                "is_patch_change": [False, False],
            }
        )
        out = recent_reward_rate_all_events(
            np.array([2.5]), trials, _well_to_arm(), k=10
        )
        assert out.shape[1] == 9
        assert out[0, 0] == pytest.approx(0.5)  # visited leaf: 1/2 rewarded
        assert out[0, 1] == 0.0  # never-visited leaf, prior trials -> encoded 0.0
        for trunk in (6, 7, 8):
            assert out[0, trunk] == 0.0  # trunk arms structural 0.0
