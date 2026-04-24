"""Tests for data_loaders/utils.py functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.data_loaders.utils import get_labels, parse_nwb_filename


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
        events = pd.DataFrame({
            "start_time": [2.0, 8.0, 15.0],
            "end_time": [4.0, 10.0, 17.0],
        })

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
        events = pd.DataFrame({
            "start_time": [1.0, 11.0, 21.0, 31.0, 41.0],
            "end_time": [3.0, 13.0, 23.0, 33.0, 43.0],
        })

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
        events = pd.DataFrame({
            "start_time": [1.0, 5.0],
            "end_time": [1.01, 5.01],
        })

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
        events = pd.DataFrame({
            "start_time": [2.0, 5.0],
            "end_time": [5.0, 8.0],
        })

        result = get_labels(events, time)

        # Both events should be labeled
        # Note: t=5.0 may be labeled as either event 1 or 2 (edge case)
        assert result.loc[2.0:4.9, "event_number"].eq(1).all()
        assert result.loc[5.1:8.0, "event_number"].eq(2).all()


class TestParseNwbFilename:
    """Tests for NWB filename parsing."""

    def test_letters_only_animal_name(self):
        """Test parsing filename with letters-only animal name."""
        animal, date = parse_nwb_filename("chimi20200212_.nwb")
        assert animal == "chimi"
        assert date == 20200212

    def test_another_letters_only_animal(self):
        """Test parsing another letters-only animal name."""
        animal, date = parse_nwb_filename("peanut20201125_.nwb")
        assert animal == "peanut"
        assert date == 20201125

    def test_alphanumeric_animal_name(self):
        """Test parsing filename with alphanumeric animal name (e.g., j16)."""
        animal, date = parse_nwb_filename("j1620210706_.nwb")
        assert animal == "j16"
        assert date == 20210706

    def test_short_animal_name(self):
        """Test parsing filename with short animal name."""
        animal, date = parse_nwb_filename("wilbur20210315_.nwb")
        assert animal == "wilbur"
        assert date == 20210315

    def test_senor_animal_name(self):
        """Test parsing senor animal name."""
        animal, date = parse_nwb_filename("senor20201015_.nwb")
        assert animal == "senor"
        assert date == 20201015

    def test_invalid_format_missing_underscore(self):
        """Test that missing underscore raises ValueError."""
        with pytest.raises(ValueError, match="Invalid NWB filename format"):
            parse_nwb_filename("chimi20200212.nwb")

    def test_invalid_format_wrong_extension(self):
        """Test that wrong extension raises ValueError."""
        with pytest.raises(ValueError, match="Invalid NWB filename format"):
            parse_nwb_filename("chimi20200212_.txt")

    def test_invalid_format_short_date(self):
        """Test that date with fewer than 8 digits raises ValueError."""
        with pytest.raises(ValueError, match="Invalid NWB filename format"):
            parse_nwb_filename("chimi2020021_.nwb")

    def test_invalid_format_no_animal(self):
        """Test that filename with no animal name raises ValueError."""
        with pytest.raises(ValueError, match="Invalid NWB filename format"):
            parse_nwb_filename("20200212_.nwb")


# Note: get_epoch_info() caching is validated through integration tests.
# Unit tests would require database access and are omitted.
# The @cache decorator is tested implicitly by verifying all integration
# tests pass with caching enabled.
