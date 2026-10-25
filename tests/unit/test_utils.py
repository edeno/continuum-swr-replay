"""Tests for data_loaders/utils.py functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.data_loaders.utils import get_labels


class TestGetLabels:
    """Tests for event labeling function."""

    def test_single_event(self):
        """Test labeling a single event."""
        # Create time index from 0 to 10 seconds
        time = pd.Index(np.linspace(0, 10, 101))

        # Single event from 3 to 7 seconds
        events = pd.DataFrame({"start_time": [3.0], "end_time": [7.0]})

        result = get_labels(events, time)

        # Check that event is labeled with 1
        assert result.loc[3.0:7.0, "event_number"].eq(1).all()

        # Check that times outside event are labeled 0
        assert result.loc[0.0:2.9, "event_number"].eq(0).all()
        assert result.loc[7.1:10.0, "event_number"].eq(0).all()

    def test_multiple_non_overlapping_events(self):
        """Test labeling multiple non-overlapping events."""
        time = pd.Index(np.linspace(0, 20, 201))

        # Three events
        events = pd.DataFrame(
            {
                "start_time": [2.0, 8.0, 15.0],
                "end_time": [4.0, 10.0, 17.0],
            }
        )

        result = get_labels(events, time)

        # Check each event has correct label (1, 2, 3)
        assert result.loc[2.0:4.0, "event_number"].eq(1).all()
        assert result.loc[8.0:10.0, "event_number"].eq(2).all()
        assert result.loc[15.0:17.0, "event_number"].eq(3).all()

        # Check gaps are labeled 0
        assert result.loc[0.0:1.9, "event_number"].eq(0).all()
        assert result.loc[4.1:7.9, "event_number"].eq(0).all()
        assert result.loc[10.1:14.9, "event_number"].eq(0).all()
        assert result.loc[17.1:20.0, "event_number"].eq(0).all()

    def test_empty_events(self):
        """Test with no events."""
        time = pd.Index(np.linspace(0, 10, 101))

        # Empty events DataFrame
        events = pd.DataFrame({"start_time": [], "end_time": []})

        result = get_labels(events, time)

        # All labels should be 0
        assert result["event_number"].eq(0).all()

    def test_event_at_time_boundaries(self):
        """Test event that spans entire time range."""
        time = pd.Index(np.linspace(0, 10, 11))

        # Event covers entire time range
        events = pd.DataFrame({"start_time": [0.0], "end_time": [10.0]})

        result = get_labels(events, time)

        # All time points should be labeled 1
        assert result["event_number"].eq(1).all()

    def test_labels_are_sequential(self):
        """Test that event labels are sequential integers starting from 1."""
        time = pd.Index(np.linspace(0, 50, 501))

        # Create 5 events
        events = pd.DataFrame(
            {
                "start_time": [1.0, 11.0, 21.0, 31.0, 41.0],
                "end_time": [3.0, 13.0, 23.0, 33.0, 43.0],
            }
        )

        result = get_labels(events, time)

        # Check that unique labels are [0, 1, 2, 3, 4, 5]
        unique_labels = sorted(result["event_number"].unique())
        assert unique_labels == [0, 1, 2, 3, 4, 5]

    def test_output_structure(self):
        """Test that output has correct structure."""
        time = pd.Index(np.linspace(0, 10, 101))
        events = pd.DataFrame({"start_time": [3.0], "end_time": [7.0]})

        result = get_labels(events, time)

        # Check DataFrame structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(time)
        assert result.columns.tolist() == ["event_number"]
        assert result.index.equals(time)
        assert result["event_number"].dtype == int

    def test_short_events(self):
        """Test labeling very short events."""
        time = pd.Index(np.arange(0, 10, 0.001))  # 1ms resolution

        # Very short events (10ms each)
        events = pd.DataFrame(
            {
                "start_time": [1.0, 5.0],
                "end_time": [1.01, 5.01],
            }
        )

        result = get_labels(events, time)

        # Count labeled time points for each event
        n_event1 = result.loc[1.0:1.01, "event_number"].eq(1).sum()
        n_event2 = result.loc[5.0:5.01, "event_number"].eq(2).sum()

        # Should have at least a few time points labeled
        assert n_event1 > 0
        assert n_event2 > 0

    def test_adjacent_events(self):
        """Test events that are adjacent (end of one = start of next)."""
        time = pd.Index(np.linspace(0, 10, 101))

        # Adjacent events
        events = pd.DataFrame(
            {
                "start_time": [2.0, 5.0],
                "end_time": [5.0, 8.0],
            }
        )

        result = get_labels(events, time)

        # Both events should be labeled
        # Note: t=5.0 may be labeled as either event 1 or 2 (edge case)
        assert result.loc[2.0:4.9, "event_number"].eq(1).all()
        assert result.loc[5.1:8.0, "event_number"].eq(2).all()


# Note: get_epoch_info() caching is validated through integration tests.
# Unit tests would require database access and are omitted.
# The @cache decorator is tested implicitly by verifying all integration
# tests pass with caching enabled.


# --- get_training_timepoints before it used interval_mask (frozen at ba2961a5) --
# The "no_ripple" mask was the complement of ``get_labels(...) > 0``. These are
# that branch and ``get_labels`` as they were, copied verbatim (docstrings
# dropped), so the closed ``interval_mask`` rule can be checked against them.


def _frozen_get_labels(times, time):
    event_labels = pd.DataFrame(0, index=time, columns=["event_number"], dtype=int)

    time_array = time.to_numpy()

    for i, (start, end) in enumerate(
        zip(times["start_time"], times["end_time"], strict=True), start=1
    ):
        start_idx = np.searchsorted(time_array, start, side="left")
        end_idx = np.searchsorted(time_array, end, side="right")

        event_labels.iloc[start_idx:end_idx, 0] = i

    return event_labels


def _frozen_not_ripple(data):
    ripple_labels = _frozen_get_labels(
        data["ripple_times"], data["position_info"].index
    )
    is_ripple = ripple_labels > 0
    not_ripple = ~np.asarray(is_ripple).squeeze()

    return not_ripple


class TestTrainingTimepointsMatchFrozen:
    """The closed ``interval_mask`` rule is the old ``get_labels > 0`` one."""

    @staticmethod
    def _assert_same(time, ripples):
        from continuum_swr_replay.data_loaders.utils import get_training_timepoints

        data = {
            "position_info": pd.DataFrame(index=pd.Index(time, name="time")),
            "ripple_times": ripples,
        }
        old = _frozen_not_ripple(data)
        new = get_training_timepoints(data, "no_ripple")
        assert new.dtype == old.dtype
        np.testing.assert_array_equal(new, old)
        return new

    def test_boundaries(self):
        """On-sample endpoints, touching and overlapping ripples, zero length,
        ripples between samples and outside the time range."""
        time = np.arange(0.0, 10.0, 0.5)
        ripples = pd.DataFrame(
            {
                "start_time": [1.0, 2.0, 2.4, 3.0, 4.2, 5.0, 6.1, -3.0, 9.4, 12.0, 8.0],
                "end_time": [
                    2.0,
                    2.4,
                    3.0,
                    3.0,
                    4.3,
                    5.5,
                    6.4,
                    -1.0,
                    9.6,
                    13.0,
                    7.0 + 1.5,
                ],
            }
        )
        new = self._assert_same(time, ripples)
        # a time on a ripple's start or end is a ripple time
        assert not new[time == 1.0][0] and not new[time == 5.5][0]
        # a ripple between two samples covers neither
        assert new[time == 6.0][0] and new[time == 6.5][0]

    def test_no_ripples_and_ripples_in_any_order(self):
        time = np.arange(0.0, 3.0, 0.002)
        self._assert_same(time, pd.DataFrame({"start_time": [], "end_time": []}))
        rng = np.random.default_rng(11)
        starts = rng.uniform(-0.5, 3.5, 300)
        ripples = pd.DataFrame(
            {"start_time": starts, "end_time": starts + rng.exponential(0.05, 300)}
        )
        self._assert_same(time, ripples)
        self._assert_same(time, ripples.sort_values("start_time"))
        on_grid = rng.choice(time, 100)
        self._assert_same(
            time,
            pd.DataFrame(
                {
                    "start_time": on_grid,
                    "end_time": on_grid + rng.choice(time, 100) % 0.1,
                }
            ),
        )

    def test_a_nan_bound_is_refused(self):
        """Old: a NaN start masked nothing, a NaN end masked to the epoch's end."""
        from continuum_swr_replay.data_loaders.utils import get_training_timepoints

        data = {
            "position_info": pd.DataFrame(index=pd.Index(np.arange(5.0))),
            "ripple_times": pd.DataFrame({"start_time": [1.0], "end_time": [np.nan]}),
        }
        with pytest.raises(ValueError, match="NaN bound"):
            get_training_timepoints(data, "no_ripple")
