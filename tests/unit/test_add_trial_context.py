"""Unit tests for ``add_trial_context``.

The function uses **in-progress trip semantics** to attach trial-derived
columns to per-timebin ``data_over_time`` rows. We want to be sure that
its ``searchsorted`` boundary handling, NaN propagation when fewer than
expected trials exist, and last-rewarded-well lookup all behave as the
docstring claims.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.hippocampus.replay_content.data_preparation import (
    add_trial_context,
)


def _make_dot(times: list[float]) -> pd.DataFrame:
    """Minimal ``data_over_time`` frame indexed by float times."""
    return pd.DataFrame(index=pd.Index(times, name="time"))


def _make_trials(rows: list[dict]) -> pd.DataFrame:
    """Build a trials frame from a list of {start_time, end_time, ...} dicts.

    Required keys (defaults shown):
        start_time, end_time, from_well, to_well, from_patch, to_patch,
        is_reward (False), is_patch_change (False)
    """
    defaults = {
        "from_patch": 1,
        "to_patch": 1,
        "is_reward": False,
        "is_patch_change": False,
    }
    return pd.DataFrame([{**defaults, **r} for r in rows])


# ── Basic correctness ──


class TestInProgressTripAssignment:
    """Each dot_time should be attributed to the latest trip whose
    start_time has occurred (the "in-progress" trip)."""

    def test_during_trip_one(self):
        trials = _make_trials(
            [
                {"start_time": 0.0, "end_time": 5.0, "from_well": 0, "to_well": 1},
                {"start_time": 6.0, "end_time": 10.0, "from_well": 1, "to_well": 2},
            ]
        )
        dot = _make_dot([2.5])
        out = add_trial_context(dot, trials)
        assert out["prev_trial_from_well"].iloc[0] == 0
        assert out["prev_trial_to_well"].iloc[0] == 1
        assert out["next_trial_to_well"].iloc[0] == 2

    def test_in_gap_between_trips(self):
        """Times in the gap between trip 0's end and trip 1's start are
        still 'on' trip 0 — the animal has arrived at trip 0's destination
        but the next trip hasn't formally begun."""
        trials = _make_trials(
            [
                {"start_time": 0.0, "end_time": 5.0, "from_well": 0, "to_well": 1},
                {"start_time": 6.0, "end_time": 10.0, "from_well": 1, "to_well": 2},
            ]
        )
        dot = _make_dot([5.5])  # gap between trips
        out = add_trial_context(dot, trials)
        assert out["prev_trial_from_well"].iloc[0] == 0
        assert out["prev_trial_to_well"].iloc[0] == 1
        assert out["next_trial_to_well"].iloc[0] == 2

    def test_at_exact_trial_start(self):
        """A timepoint exactly at trip i's start_time is attributed to
        trip i (side='right' in searchsorted means the new trip 'wins'
        at the boundary)."""
        trials = _make_trials(
            [
                {"start_time": 0.0, "end_time": 5.0, "from_well": 0, "to_well": 1},
                {"start_time": 6.0, "end_time": 10.0, "from_well": 1, "to_well": 2},
            ]
        )
        dot = _make_dot([6.0])
        out = add_trial_context(dot, trials)
        assert out["prev_trial_from_well"].iloc[0] == 1
        assert out["prev_trial_to_well"].iloc[0] == 2

    def test_during_last_trip_has_no_next(self):
        trials = _make_trials(
            [
                {"start_time": 0.0, "end_time": 5.0, "from_well": 0, "to_well": 1},
                {"start_time": 6.0, "end_time": 10.0, "from_well": 1, "to_well": 2},
            ]
        )
        dot = _make_dot([7.0])
        out = add_trial_context(dot, trials)
        assert np.isnan(out["next_trial_to_well"].iloc[0])
        assert np.isnan(out["next_trial_to_patch"].iloc[0])


class TestBeforeFirstTrial:
    """Times before any trip has started should propagate NaN."""

    def test_all_columns_nan(self):
        trials = _make_trials(
            [{"start_time": 5.0, "end_time": 10.0, "from_well": 0, "to_well": 1}]
        )
        dot = _make_dot([1.0])
        out = add_trial_context(dot, trials)
        for col in [
            "prev_trial_from_well",
            "prev_trial_to_well",
            "next_trial_to_well",
            "is_after_switch",
            "is_after_omission",
        ]:
            assert np.isnan(out[col].iloc[0]), f"{col} should be NaN before first trial"


class TestSingleTrialEpoch:
    """One-trial sessions: prev_* fields populated, next_* always NaN."""

    def test_single_trial(self):
        trials = _make_trials(
            [
                {
                    "start_time": 0.0,
                    "end_time": 5.0,
                    "from_well": 3,
                    "to_well": 4,
                    "from_patch": 2,
                    "to_patch": 3,
                    "is_reward": True,
                    "is_patch_change": True,
                }
            ]
        )
        dot = _make_dot([2.0, 4.0])
        out = add_trial_context(dot, trials)
        assert (out["prev_trial_from_well"] == 3).all()
        assert (out["prev_trial_to_well"] == 4).all()
        assert out["next_trial_to_well"].isna().all()
        assert (out["is_after_switch"] == 1.0).all()
        assert (out["is_after_omission"] == 0.0).all()  # is_reward=True


class TestLastRewardedWell:
    """Reward attribution uses end_times of rewarded trips, not start_times."""

    def test_no_rewarded_trials(self):
        trials = _make_trials(
            [{"start_time": 0.0, "end_time": 5.0, "from_well": 0, "to_well": 1}]
        )
        dot = _make_dot([3.0, 6.0])
        out = add_trial_context(dot, trials)
        assert out["last_rewarded_well"].isna().all()

    def test_reward_attributed_after_trip_end(self):
        """A rewarded trip's to_well becomes last_rewarded_well only after
        the trip ends — not while it is still in progress."""
        trials = _make_trials(
            [
                {
                    "start_time": 0.0,
                    "end_time": 5.0,
                    "from_well": 0,
                    "to_well": 1,
                    "is_reward": True,
                },
                {
                    "start_time": 6.0,
                    "end_time": 10.0,
                    "from_well": 1,
                    "to_well": 2,
                    "is_reward": False,
                },
            ]
        )
        # mid-trip 0: not yet rewarded
        # exactly at trip 0 end: rewarded
        # mid-trip 1: still last_rewarded = well 1
        dot = _make_dot([2.0, 5.0, 7.0])
        out = add_trial_context(dot, trials)
        assert np.isnan(out["last_rewarded_well"].iloc[0])
        assert out["last_rewarded_well"].iloc[1] == 1
        assert out["last_rewarded_well"].iloc[2] == 1

    def test_reward_updates_when_new_reward_occurs(self):
        trials = _make_trials(
            [
                {
                    "start_time": 0.0,
                    "end_time": 5.0,
                    "from_well": 0,
                    "to_well": 1,
                    "is_reward": True,
                },
                {
                    "start_time": 6.0,
                    "end_time": 10.0,
                    "from_well": 1,
                    "to_well": 4,
                    "is_reward": True,
                },
            ]
        )
        dot = _make_dot([3.0, 7.0, 11.0])
        out = add_trial_context(dot, trials)
        assert np.isnan(out["last_rewarded_well"].iloc[0])
        assert out["last_rewarded_well"].iloc[1] == 1
        assert out["last_rewarded_well"].iloc[2] == 4


class TestTaskFlags:
    """is_after_switch / is_after_omission reflect the current trip's state."""

    def test_omission_flag(self):
        trials = _make_trials(
            [
                {
                    "start_time": 0.0,
                    "end_time": 5.0,
                    "from_well": 0,
                    "to_well": 1,
                    "is_reward": False,
                },
                {
                    "start_time": 6.0,
                    "end_time": 10.0,
                    "from_well": 1,
                    "to_well": 2,
                    "is_reward": True,
                },
            ]
        )
        dot = _make_dot([2.0, 8.0])
        out = add_trial_context(dot, trials)
        assert out["is_after_omission"].iloc[0] == 1.0  # trip 0 was omitted
        assert out["is_after_omission"].iloc[1] == 0.0  # trip 1 rewarded

    def test_patch_change_flag(self):
        trials = _make_trials(
            [
                {
                    "start_time": 0.0,
                    "end_time": 5.0,
                    "from_well": 0,
                    "to_well": 1,
                    "from_patch": 1,
                    "to_patch": 1,
                    "is_patch_change": False,
                },
                {
                    "start_time": 6.0,
                    "end_time": 10.0,
                    "from_well": 1,
                    "to_well": 4,
                    "from_patch": 1,
                    "to_patch": 3,
                    "is_patch_change": True,
                },
            ]
        )
        dot = _make_dot([3.0, 7.0])
        out = add_trial_context(dot, trials)
        assert out["is_after_switch"].iloc[0] == 0.0
        assert out["is_after_switch"].iloc[1] == 1.0


class TestVectorizedMatchesPerTimepoint:
    """Calling the function on a long dot frame must give the same result
    as calling it once per row."""

    def test_consistency_across_many_times(self):
        rng = np.random.default_rng(0)
        n_trials = 30
        starts = np.cumsum(rng.uniform(1.0, 4.0, size=n_trials))
        trials = _make_trials(
            [
                {
                    "start_time": float(starts[i]),
                    "end_time": float(starts[i] + 1.5),
                    "from_well": int(rng.integers(0, 6)),
                    "to_well": int(rng.integers(0, 6)),
                    "is_reward": bool(rng.random() < 0.6),
                    "is_patch_change": bool(rng.random() < 0.3),
                }
                for i in range(n_trials)
            ]
        )

        times = sorted(rng.uniform(0.0, float(starts[-1]) + 5.0, size=200).tolist())
        dot_full = _make_dot(times)
        out_full = add_trial_context(dot_full, trials)

        for k in [0, 5, 50, 199]:
            single = add_trial_context(_make_dot([times[k]]), trials)
            for col in [
                "prev_trial_from_well",
                "prev_trial_to_well",
                "next_trial_to_well",
                "last_rewarded_well",
                "is_after_switch",
                "is_after_omission",
            ]:
                a = out_full[col].iloc[k]
                b = single[col].iloc[0]
                if pd.isna(a) and pd.isna(b):
                    continue
                assert a == b, f"{col} mismatch at t={times[k]}: vec={a} single={b}"


class TestNoTrials:
    """Empty trials frame.

    Documented finding: ``add_trial_context`` does **not** guard against
    zero-row trial frames — ``np.searchsorted`` returns -1 for every
    ``dot_time``, ``np.clip(-1, 0, -1)`` is undefined behavior on
    empty bounds, and indexing into a zero-length array raises
    ``IndexError``. The production caller (``prepare_session``) guards
    this case via ``if trials is None or len(trials) < 3: return None``,
    so the crash is unreachable in practice. We pin the current behavior
    here so a future refactor that loosens the upstream guard is forced
    to add a guard inside ``add_trial_context`` itself.
    """

    def test_empty_trials_with_required_columns(self):
        # Build an empty frame that still has the columns the function reads.
        cols = [
            "start_time",
            "end_time",
            "from_well",
            "to_well",
            "from_patch",
            "to_patch",
            "is_reward",
            "is_patch_change",
        ]
        trials = pd.DataFrame({c: pd.Series(dtype=float) for c in cols})
        # is_reward and is_patch_change need bool dtype for the boolean indexing
        # add_trial_context does on `~is_reward[current_clipped]`.
        trials["is_reward"] = trials["is_reward"].astype(bool)
        trials["is_patch_change"] = trials["is_patch_change"].astype(bool)
        dot = _make_dot([1.0, 2.0])
        with pytest.raises(IndexError):
            add_trial_context(dot, trials)


@pytest.mark.parametrize("side_case", ["start_eq_dot", "end_eq_dot"])
class TestBoundaryEquality:
    """Boundary conditions where dot_time exactly equals a trial timestamp."""

    def test_boundary(self, side_case):
        trials = _make_trials(
            [
                {
                    "start_time": 0.0,
                    "end_time": 5.0,
                    "from_well": 0,
                    "to_well": 1,
                    "is_reward": True,
                },
                {
                    "start_time": 5.0,  # back-to-back
                    "end_time": 10.0,
                    "from_well": 1,
                    "to_well": 2,
                },
            ]
        )
        if side_case == "start_eq_dot":
            t = 5.0  # exact start of trip 1 = exact end of trip 0
        else:
            t = 5.0  # same point in this back-to-back layout
        out = add_trial_context(_make_dot([t]), trials)
        # At the exact start of trip 1 (side='right'), we should be on trip 1.
        assert out["prev_trial_from_well"].iloc[0] == 1
        # last_rewarded uses side='right' on end_times → at t=5 (=trip 0 end),
        # the inequality `end_time <= t` holds, so reward is attributed.
        assert out["last_rewarded_well"].iloc[0] == 1
