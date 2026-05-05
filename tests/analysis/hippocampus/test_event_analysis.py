"""Unit tests for event analysis utilities.

Tests functions for filtering and characterizing detected events, including
speed-based filtering, event time extraction, and duration conversion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.hippocampus.event_analysis import (
    get_event_durations_ms,
    get_event_start_end_times,
    get_immobile_events,
)


class TestGetImmobileEvents:
    """Test filtering events by animal speed."""

    @pytest.fixture
    def mock_event_data(self) -> pd.DataFrame:
        """Create mock data with events and varying speeds."""
        # Create 20 time points at 500 Hz (40 ms total)
        time = np.arange(0.0, 0.04, 0.002)

        # Event 1: times 0-8 (speed: 3.0 at start, 5.0 at end, max 6.0)
        # Event 2: times 10-14 (speed: 2.0 at start/end, max 3.0)
        # Event 3: times 16-19 (speed: 5.0 at start/end, max 5.0)
        event_labels = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 0, 3, 3, 3, 3])
        speeds = np.array([
            # Event 1: start low, goes high
            3.0, 4.0, 5.0, 6.0, 5.5, 5.0, 4.5, 4.0, 5.0,
            # No event
            1.0,
            # Event 2: stays low
            2.0, 2.5, 3.0, 2.5, 2.0,
            # No event
            1.0,
            # Event 3: stays at 5.0
            5.0, 5.0, 5.0, 5.0,
        ])

        return pd.DataFrame(
            {
                "animal": ["mouse1"] * len(time),
                "day": [20200101] * len(time),
                "epoch": ["01_r1"] * len(time),
                "event": event_labels,
                "head_speed": speeds,
            },
            index=time,
        )

    def test_endpoints_method_filters_correctly(self, mock_event_data):
        """Endpoints method should filter based on start and end speeds."""
        result = get_immobile_events(
            mock_event_data, event_col="event", speed_threshold=4.0, method="endpoints"
        )

        # Event 1: start=3.0 (pass), end=5.0 (fail) -> excluded
        # Event 2: start=2.0 (pass), end=2.0 (pass) -> included
        # Event 3: start=5.0 (fail), end=5.0 (fail) -> excluded

        assert len(result) > 0, "Should have at least one event"
        event_ids = result["event"].unique()
        assert 2 in event_ids, "Event 2 should pass"
        assert 1 not in event_ids, "Event 1 should be filtered out (end too fast)"
        assert 3 not in event_ids, "Event 3 should be filtered out (endpoints too fast)"

    def test_max_method_filters_correctly(self, mock_event_data):
        """Max method should filter based on maximum speed during event."""
        result = get_immobile_events(
            mock_event_data, event_col="event", speed_threshold=4.0, method="max"
        )

        # Event 1: max=6.0 (fail) -> excluded
        # Event 2: max=3.0 (pass) -> included
        # Event 3: max=5.0 (fail) -> excluded

        event_ids = result["event"].unique()
        assert 2 in event_ids, "Event 2 should pass (max=3.0 < 4.0)"
        assert 1 not in event_ids, "Event 1 should be filtered (max=6.0 > 4.0)"
        assert 3 not in event_ids, "Event 3 should be filtered (max=5.0 > 4.0)"

    def test_mean_method_filters_correctly(self, mock_event_data):
        """Mean method should filter based on mean speed during event."""
        result = get_immobile_events(
            mock_event_data, event_col="event", speed_threshold=4.0, method="mean"
        )

        # Event 1: mean ≈ 4.67 (fail) -> excluded
        # Event 2: mean = 2.4 (pass) -> included
        # Event 3: mean = 5.0 (fail) -> excluded

        event_ids = result["event"].unique()
        assert 2 in event_ids, "Event 2 should pass (mean < 4.0)"
        assert 1 not in event_ids, "Event 1 should be filtered (mean > 4.0)"
        assert 3 not in event_ids, "Event 3 should be filtered (mean > 4.0)"

    def test_empty_data_returns_empty(self):
        """Empty input should return empty DataFrame."""
        data = pd.DataFrame(
            {
                "animal": [],
                "day": [],
                "epoch": [],
                "event": [],
                "head_speed": [],
            }
        )
        data.index = pd.Index([])

        result = get_immobile_events(data, event_col="event", speed_threshold=4.0)

        assert len(result) == 0
        assert isinstance(result, pd.DataFrame)

    def test_no_events_above_zero_returns_empty(self, mock_event_data):
        """Data with only event=0 (non-events) should return empty."""
        mock_event_data["event"] = 0  # All non-events

        result = get_immobile_events(mock_event_data, event_col="event", speed_threshold=4.0)

        assert len(result) == 0

    def test_invalid_speed_threshold_raises_error(self, mock_event_data):
        """Negative or zero speed threshold should raise ValueError."""
        with pytest.raises(ValueError, match="speed_threshold must be positive"):
            get_immobile_events(mock_event_data, event_col="event", speed_threshold=0.0)

        with pytest.raises(ValueError, match="speed_threshold must be positive"):
            get_immobile_events(mock_event_data, event_col="event", speed_threshold=-1.0)

    def test_invalid_method_raises_error(self, mock_event_data):
        """Invalid method should raise ValueError."""
        with pytest.raises(ValueError, match="method must be"):
            get_immobile_events(
                mock_event_data, event_col="event", speed_threshold=4.0, method="invalid"
            )

    def test_missing_event_col_raises_error(self, mock_event_data):
        """Missing event column should raise ValueError."""
        with pytest.raises(ValueError, match="event_col.*not found"):
            get_immobile_events(
                mock_event_data, event_col="nonexistent", speed_threshold=4.0
            )

    def test_missing_speed_col_raises_error(self, mock_event_data):
        """Missing speed column should raise ValueError."""
        with pytest.raises(ValueError, match="speed_col.*not found"):
            get_immobile_events(
                mock_event_data,
                event_col="event",
                speed_col="nonexistent",
                speed_threshold=4.0,
            )

    def test_custom_speed_col(self, mock_event_data):
        """Should accept custom speed column name."""
        mock_event_data["custom_speed"] = mock_event_data["head_speed"]

        result = get_immobile_events(
            mock_event_data,
            event_col="event",
            speed_col="custom_speed",
            speed_threshold=4.0,
            method="endpoints",
        )

        # Same as endpoints test
        event_ids = result["event"].unique()
        assert 2 in event_ids


class TestGetEventStartEndTimes:
    """Test extracting start/end times from event labels."""

    @pytest.fixture
    def labeled_events(self) -> pd.DataFrame:
        """Create labeled event data."""
        time = np.arange(0.0, 1.0, 0.01)  # 100 time points, 10 ms spacing
        # Event 1: indices 10-19 (10 points = 90 ms duration)
        # Event 2: indices 30-44 (15 points = 140 ms duration)
        # Event 3: indices 60-69 (10 points = 90 ms duration)
        event_labels = np.zeros(len(time), dtype=int)
        event_labels[10:20] = 1
        event_labels[30:45] = 2
        event_labels[60:70] = 3

        return pd.DataFrame(
            {
                "animal": ["mouse1"] * len(time),
                "day": [20200101] * len(time),
                "epoch": ["01_r1"] * len(time),
                "event": event_labels,
            },
            index=time,
        )

    def test_label_to_interval_conversion(self, labeled_events):
        """Should correctly convert event labels to start/end times."""
        result = get_event_start_end_times(labeled_events, event_col="event")

        # Should have 3 events
        assert len(result) == 3

        # Check event IDs
        assert set(result["event"]) == {1, 2, 3}

        # Check start times
        assert result[result["event"] == 1]["start_time"].iloc[0] == pytest.approx(0.10)
        assert result[result["event"] == 2]["start_time"].iloc[0] == pytest.approx(0.30)
        assert result[result["event"] == 3]["start_time"].iloc[0] == pytest.approx(0.60)

        # Check end times
        assert result[result["event"] == 1]["end_time"].iloc[0] == pytest.approx(0.19)
        assert result[result["event"] == 2]["end_time"].iloc[0] == pytest.approx(0.44)
        assert result[result["event"] == 3]["end_time"].iloc[0] == pytest.approx(0.69)

    def test_duration_calculation(self, labeled_events):
        """Should correctly calculate event durations."""
        result = get_event_start_end_times(labeled_events, event_col="event")

        # Event 1: 0.19 - 0.10 = 0.09 s
        # Event 2: 0.44 - 0.30 = 0.14 s
        # Event 3: 0.69 - 0.60 = 0.09 s

        assert result[result["event"] == 1]["duration"].iloc[0] == pytest.approx(0.09)
        assert result[result["event"] == 2]["duration"].iloc[0] == pytest.approx(0.14)
        assert result[result["event"] == 3]["duration"].iloc[0] == pytest.approx(0.09)

    def test_non_events_excluded(self, labeled_events):
        """Non-events (event_col == 0) should be excluded."""
        result = get_event_start_end_times(labeled_events, event_col="event")

        # Should not contain event ID 0
        assert 0 not in result["event"].values

    def test_empty_data_returns_empty(self):
        """Empty input should return empty DataFrame with correct columns."""
        data = pd.DataFrame(
            {"animal": [], "day": [], "epoch": [], "event": []}, index=pd.Index([])
        )

        result = get_event_start_end_times(data, event_col="event")

        assert len(result) == 0
        assert "start_time" in result.columns
        assert "end_time" in result.columns
        assert "duration" in result.columns
        assert "event" in result.columns

    def test_only_non_events_returns_empty(self, labeled_events):
        """Data with only event=0 should return empty DataFrame."""
        labeled_events["event"] = 0  # All non-events

        result = get_event_start_end_times(labeled_events, event_col="event")

        assert len(result) == 0
        assert "duration" in result.columns

    def test_missing_event_col_raises_error(self, labeled_events):
        """Missing event column should raise ValueError."""
        with pytest.raises(ValueError, match="event_col.*not found"):
            get_event_start_end_times(labeled_events, event_col="nonexistent")

    def test_named_time_index(self, labeled_events):
        """Should handle DataFrames whose index already has a name."""
        # Real ``data_over_time`` frames have index.name == "time", which
        # used to break the previous reset_index/agg pair that hard-coded
        # the column name "index".
        labeled_events.index.name = "time"

        result = get_event_start_end_times(labeled_events, event_col="event")

        assert len(result) == 3
        assert result[result["event"] == 1]["start_time"].iloc[0] == pytest.approx(0.10)
        assert result[result["event"] == 1]["end_time"].iloc[0] == pytest.approx(0.19)

    def test_custom_group_cols(self):
        """Should work with custom grouping columns."""
        time = np.arange(0.0, 0.1, 0.01)
        data = pd.DataFrame(
            {
                "session": ["A"] * len(time),
                "trial": [1] * len(time),
                "event": np.array([0, 0, 1, 1, 1, 0, 0, 2, 2, 0]),
            },
            index=time,
        )

        result = get_event_start_end_times(
            data, event_col="event", group_cols=["session", "trial"]
        )

        # Should have 2 events
        assert len(result) == 2
        assert "session" in result.columns
        assert "trial" in result.columns


class TestGetEventDurationsMs:
    """Test conversion of event durations from seconds to milliseconds."""

    def test_unit_conversion(self):
        """Should correctly convert seconds to milliseconds."""
        event_times = pd.DataFrame(
            {
                "event": [1, 2, 3],
                "start_time": [1.0, 2.0, 3.0],
                "end_time": [1.05, 2.12, 3.08],
                "duration": [0.05, 0.12, 0.08],  # in seconds
            }
        )

        result = get_event_durations_ms(event_times)

        # Check conversion: seconds * 1000 = milliseconds
        assert result.iloc[0] == pytest.approx(50.0)
        assert result.iloc[1] == pytest.approx(120.0)
        assert result.iloc[2] == pytest.approx(80.0)

    def test_missing_duration_column_raises_error(self):
        """Should raise ValueError if duration column is missing."""
        event_times = pd.DataFrame({"event": [1, 2], "start_time": [1.0, 2.0]})

        with pytest.raises(ValueError, match="duration"):
            get_event_durations_ms(event_times)

    def test_empty_dataframe_returns_empty_series(self):
        """Empty DataFrame should return empty Series."""
        event_times = pd.DataFrame({"duration": []})

        result = get_event_durations_ms(event_times)

        assert len(result) == 0
        assert isinstance(result, pd.Series)

    def test_preserves_series_properties(self):
        """Should preserve index and return Series."""
        event_times = pd.DataFrame(
            {"event": [1, 2, 3], "duration": [0.05, 0.12, 0.08]}, index=[10, 20, 30]
        )

        result = get_event_durations_ms(event_times)

        assert isinstance(result, pd.Series)
        # Index should be preserved from DataFrame
        assert list(result.index) == [10, 20, 30]
