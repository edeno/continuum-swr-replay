"""Tests for analysis/replay_models/choice/shuffles.py.

The three shuffle generators underpin every significance test in the
paper. A regression in any of them — a shift that doesn't shift, a
permutation that leaks across sessions, a marginal distribution that
silently changes — would inflate type-I error throughout the results
section without anyone noticing. Pin the invariants:

- per-event marginal preservation for ``circular_time_shuffle``
- single-bin no-op behavior
- non-identity shift (the original ordering is never returned)
- multi-session guard for ``stop_label_shuffle``
- non-identity behavior for ``next_arm_circular_shift``
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay_models.choice.shuffles import (
    circular_time_shuffle,
    next_arm_circular_shift,
    stop_label_shuffle,
)


class TestCircularTimeShuffle:
    """Within-event circular shift of ``decoded_arms``."""

    def test_per_event_marginal_arm_distribution_is_preserved(self):
        """For each event, the multiset of arm IDs is invariant — only
        the sequence is shuffled. This is what makes circular shift
        an ``arm_count_hard``-equivalent null: counts unchanged,
        order destroyed."""
        rng = np.random.default_rng(42)
        arms = rng.integers(0, 9, size=500)
        event_ids = np.repeat(np.arange(10), 50)

        shuffled = circular_time_shuffle(arms, event_ids, rng=np.random.default_rng(0))

        for ev in np.unique(event_ids):
            orig_counts = np.bincount(arms[event_ids == ev], minlength=9)
            shuf_counts = np.bincount(shuffled[event_ids == ev], minlength=9)
            np.testing.assert_array_equal(orig_counts, shuf_counts)

    def test_total_length_preserved(self):
        rng = np.random.default_rng(0)
        arms = rng.integers(0, 9, size=200)
        event_ids = np.repeat(np.arange(20), 10)

        shuffled = circular_time_shuffle(arms, event_ids, rng=rng)
        assert shuffled.shape == arms.shape

    def test_single_timebin_event_is_passed_through(self):
        """An event of length 1 has no non-identity shift available;
        the shuffle leaves it as-is rather than crashing on
        ``rng.integers(1, 1)``."""
        out = circular_time_shuffle(
            np.array([3]),
            np.array([0]),
            rng=np.random.default_rng(0),
        )
        assert out.tolist() == [3]

    def test_shift_is_never_zero_for_length_two_or_more(self):
        """Docstring contract: shift in ``{1, ..., n-1}``. Verifying
        across many seeds that a length-2 event with two distinct arms
        always swaps them — never returns the input."""
        for seed in range(50):
            rng = np.random.default_rng(seed)
            arms = np.array([0, 1])
            event_ids = np.zeros(2, dtype=int)
            out = circular_time_shuffle(arms, event_ids, rng=rng)
            assert not np.array_equal(out, arms), (
                f"seed {seed}: shift was 0 (identity) — should be in [1, n-1]"
            )

    def test_input_array_not_modified(self):
        """``circular_time_shuffle`` must not mutate the caller's array."""
        arms = np.array([0, 1, 2, 3, 4])
        event_ids = np.zeros(5, dtype=int)
        original = arms.copy()
        _ = circular_time_shuffle(arms, event_ids, rng=np.random.default_rng(0))
        np.testing.assert_array_equal(arms, original)

    def test_independent_events_shuffled_independently(self):
        """Two events with different arm distributions should each
        retain their own per-event marginal — the shuffle never
        leaks arm IDs across event boundaries."""
        arms = np.concatenate([np.zeros(10, dtype=int), np.ones(10, dtype=int)])
        event_ids = np.repeat([0, 1], 10)
        out = circular_time_shuffle(arms, event_ids, rng=np.random.default_rng(7))

        assert (out[event_ids == 0] == 0).all(), "event 0 leaked arm 1"
        assert (out[event_ids == 1] == 1).all(), "event 1 leaked arm 0"


class TestStopLabelShuffle:
    """Within-session permutation of behavioral labels across stops."""

    def _stops_from_dicts(self, rows):
        return pd.DataFrame(rows)

    def test_raises_on_multi_session_input(self):
        """Cross-session shuffling would swap labels between animals or
        epochs — never the intended null, but easy to do by accident
        if a caller passes the concatenated multi-session table."""
        stops = self._stops_from_dicts(
            [
                {
                    "animal": "a",
                    "session_date": 1,
                    "epoch": "02_r1",
                    "is_rewarded_outcome": True,
                },
                {
                    "animal": "a",
                    "session_date": 1,
                    "epoch": "02_r1",
                    "is_rewarded_outcome": False,
                },
                {
                    "animal": "b",
                    "session_date": 1,
                    "epoch": "02_r1",
                    "is_rewarded_outcome": True,
                },
            ]
        )
        with pytest.raises(ValueError, match="session"):
            stop_label_shuffle(stops, rng=np.random.default_rng(0))

    def test_marginal_preserved_for_shuffled_columns(self):
        """The set of label values per column is unchanged — only
        which stop has which label changes."""
        stops = self._stops_from_dicts(
            [
                {
                    "animal": "a",
                    "session_date": 1,
                    "epoch": "02_r1",
                    "is_rewarded_outcome": True,
                    "occupancy_well_arm": 0,
                },
                {
                    "animal": "a",
                    "session_date": 1,
                    "epoch": "02_r1",
                    "is_rewarded_outcome": True,
                    "occupancy_well_arm": 1,
                },
                {
                    "animal": "a",
                    "session_date": 1,
                    "epoch": "02_r1",
                    "is_rewarded_outcome": False,
                    "occupancy_well_arm": 2,
                },
                {
                    "animal": "a",
                    "session_date": 1,
                    "epoch": "02_r1",
                    "is_rewarded_outcome": False,
                    "occupancy_well_arm": 3,
                },
            ]
        )
        out = stop_label_shuffle(stops, rng=np.random.default_rng(11))

        assert sorted(out["is_rewarded_outcome"].tolist()) == sorted(
            stops["is_rewarded_outcome"].tolist()
        )
        assert sorted(out["occupancy_well_arm"].tolist()) == sorted(
            stops["occupancy_well_arm"].tolist()
        )

    def test_non_specified_columns_unchanged(self):
        """Only the columns in the ``columns`` argument are shuffled.
        Other columns (e.g. session keys) must remain row-aligned."""
        stops = self._stops_from_dicts(
            [
                {
                    "animal": "a",
                    "session_date": 1,
                    "epoch": "02_r1",
                    "stop_id": 100,
                    "is_rewarded_outcome": True,
                },
                {
                    "animal": "a",
                    "session_date": 1,
                    "epoch": "02_r1",
                    "stop_id": 200,
                    "is_rewarded_outcome": False,
                },
                {
                    "animal": "a",
                    "session_date": 1,
                    "epoch": "02_r1",
                    "stop_id": 300,
                    "is_rewarded_outcome": False,
                },
            ]
        )
        out = stop_label_shuffle(
            stops, columns=("is_rewarded_outcome",), rng=np.random.default_rng(0)
        )
        np.testing.assert_array_equal(out["stop_id"], stops["stop_id"])
        np.testing.assert_array_equal(out["animal"], stops["animal"])

    def test_input_table_not_modified(self):
        """Returns a copy; caller's frame is untouched."""
        stops = self._stops_from_dicts(
            [
                {
                    "animal": "a",
                    "session_date": 1,
                    "epoch": "02_r1",
                    "is_rewarded_outcome": True,
                },
                {
                    "animal": "a",
                    "session_date": 1,
                    "epoch": "02_r1",
                    "is_rewarded_outcome": False,
                },
            ]
        )
        before = stops.copy()
        _ = stop_label_shuffle(stops, rng=np.random.default_rng(0))
        pd.testing.assert_frame_equal(stops, before)

    def test_missing_columns_silently_skipped(self):
        """Columns named in ``columns`` but absent from the frame are
        skipped — typical when an analysis doesn't compute a particular
        contextual variable."""
        stops = self._stops_from_dicts(
            [
                {
                    "animal": "a",
                    "session_date": 1,
                    "epoch": "02_r1",
                    "occupancy_well_arm": 0,
                },
                {
                    "animal": "a",
                    "session_date": 1,
                    "epoch": "02_r1",
                    "occupancy_well_arm": 1,
                },
            ]
        )
        out = stop_label_shuffle(
            stops,
            columns=("occupancy_well_arm", "missing_col"),
            rng=np.random.default_rng(0),
        )
        assert "missing_col" not in out.columns
        assert sorted(out["occupancy_well_arm"].tolist()) == [0, 1]


class TestNextArmCircularShift:
    """Within-session circular shift of next_arm and post-hoc columns."""

    def _events(self, n, *, with_next_arm=True):
        rng = np.random.default_rng(0)
        rows = {
            "event_midpoint_s": np.arange(n, dtype=float),
        }
        if with_next_arm:
            rows["next_arm"] = rng.integers(0, 9, size=n)
            rows["next_patch"] = rng.integers(1, 4, size=n)
            rows["is_stay"] = rng.integers(0, 2, size=n)
        return pd.DataFrame(rows)

    def test_returns_unchanged_for_single_event(self):
        """No shift available for n=1; output equals input."""
        events = self._events(1)
        out = next_arm_circular_shift(events, rng=np.random.default_rng(0))
        pd.testing.assert_frame_equal(out, events)

    def test_marginal_preserved(self):
        events = self._events(20)
        out = next_arm_circular_shift(events, rng=np.random.default_rng(0))
        assert sorted(out["next_arm"].tolist()) == sorted(events["next_arm"].tolist())

    def test_non_identity_shift_for_n_at_least_two(self):
        """The shift is sampled from [1, n-1] so the output never
        equals the input ordering."""
        for seed in range(50):
            events = self._events(10)
            out = next_arm_circular_shift(events, rng=np.random.default_rng(seed))
            assert not np.array_equal(
                out["next_arm"].to_numpy(), events["next_arm"].to_numpy()
            ), f"seed {seed}: shift was 0 (identity)"

    def test_shifts_only_specified_columns(self):
        """Only the post-hoc columns roll; ``event_midpoint_s``
        stays sorted because the function sorts by it before shifting."""
        events = self._events(10)
        out = next_arm_circular_shift(events, rng=np.random.default_rng(13))
        np.testing.assert_array_equal(
            out["event_midpoint_s"].to_numpy(), events["event_midpoint_s"].to_numpy()
        )

    def test_input_table_not_modified(self):
        events = self._events(10)
        before = events.copy()
        _ = next_arm_circular_shift(events, rng=np.random.default_rng(0))
        pd.testing.assert_frame_equal(events, before)

    def test_missing_columns_silently_skipped(self):
        """A frame that lacks ``is_stay`` (for instance) shouldn't
        raise — only the present columns are shifted."""
        events = self._events(10, with_next_arm=True).drop(columns=["is_stay"])
        out = next_arm_circular_shift(events, rng=np.random.default_rng(0))
        assert "is_stay" not in out.columns

    def test_raises_on_multi_session_input(self):
        """Direct helper calls must not roll labels across sessions."""
        events = pd.DataFrame(
            {
                "animal": ["a", "a", "b", "b"],
                "session_date": [1, 1, 1, 1],
                "epoch": ["02_r1", "02_r1", "02_r1", "02_r1"],
                "event_midpoint_s": [0.0, 1.0, 0.5, 1.5],
                "next_arm": [0, 1, 2, 3],
            }
        )
        with pytest.raises(ValueError, match="sessions"):
            next_arm_circular_shift(events, rng=np.random.default_rng(0))

    def test_is_stay_recomputed_against_destination_current_patch(self):
        """Regression: ``is_stay`` was previously rolled alongside
        ``next_arm``/``next_patch``, preserving the source row's old
        ``current_patch`` reference. After the shift the destination
        row's ``current_patch`` is paired with the new ``next_patch``,
        so ``is_stay`` must be recomputed as
        ``current_patch == shifted_next_patch``.

        Probe: 3 events with mixed ``current_patch`` values where the
        rolled ``is_stay`` (source-row-derived) disagrees with the
        correct value computed from the destination row's
        ``current_patch`` and the just-shifted ``next_patch``.
        """
        events = pd.DataFrame(
            [
                {
                    "event_midpoint_s": 0.0,
                    "current_patch": 1.0,
                    "next_arm": 0,
                    "next_patch": 2.0,
                    "is_stay": 0.0,
                },
                {
                    "event_midpoint_s": 1.0,
                    "current_patch": 2.0,
                    "next_arm": 1,
                    "next_patch": 1.0,
                    "is_stay": 0.0,
                },
                {
                    "event_midpoint_s": 2.0,
                    "current_patch": 1.0,
                    "next_arm": 2,
                    "next_patch": 1.0,
                    "is_stay": 1.0,
                },
            ]
        )

        class _PinnedRng:
            def integers(self, low, high):  # noqa: ARG002
                return 1

        out = next_arm_circular_shift(events, rng=_PinnedRng())
        # After shift=1: next_patch becomes [1.0, 2.0, 1.0]; the
        # destination current_patch is [1.0, 2.0, 1.0]. So all three
        # rows match → is_stay should be all 1.0. Pre-fix the rolled
        # is_stay was [1.0, 0.0, 0.0].
        np.testing.assert_array_equal(
            out["is_stay"].to_numpy(), np.array([1.0, 1.0, 1.0])
        )
        # And the consistency invariant must hold:
        cp = out["current_patch"].to_numpy()
        npat = out["next_patch"].to_numpy()
        expected = (cp == npat).astype(float)
        np.testing.assert_array_equal(out["is_stay"].to_numpy(), expected)

    def test_is_stay_propagates_nan_when_current_or_next_patch_is_nan(self):
        """If either ``current_patch`` or the shifted ``next_patch``
        is NaN, ``is_stay`` must propagate NaN (matches the
        builders' contract)."""
        events = pd.DataFrame(
            [
                {
                    "event_midpoint_s": 0.0,
                    "current_patch": 1.0,
                    "next_arm": 0,
                    "next_patch": 2.0,
                    "is_stay": 0.0,
                },
                {
                    "event_midpoint_s": 1.0,
                    "current_patch": np.nan,
                    "next_arm": 1,
                    "next_patch": 1.0,
                    "is_stay": np.nan,
                },
            ]
        )

        class _PinnedRng:
            def integers(self, low, high):  # noqa: ARG002
                return 1

        out = next_arm_circular_shift(events, rng=_PinnedRng())
        # Row 1 has current_patch=NaN → is_stay=NaN.
        assert np.isnan(out["is_stay"].iloc[1])


@pytest.mark.parametrize(
    "shuffle_without_rng",
    [
        lambda: circular_time_shuffle(np.array([0, 1, 2]), np.array([0, 0, 0])),
        lambda: stop_label_shuffle(pd.DataFrame({"is_rewarded_outcome": [0, 1]})),
        lambda: next_arm_circular_shift(
            pd.DataFrame({"event_midpoint_s": [0.0, 1.0], "next_arm": [0, 1]})
        ),
    ],
    ids=["circular_time_shuffle", "stop_label_shuffle", "next_arm_circular_shift"],
)
def test_a_shuffle_needs_the_callers_generator(shuffle_without_rng):
    """No shuffle draws from an unseeded generator of its own."""
    with pytest.raises(TypeError, match="rng"):
        shuffle_without_rng()
