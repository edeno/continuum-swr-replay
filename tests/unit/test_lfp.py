"""Unit tests for LFP data processing.

Tests the LFP filtering and ripple data processing functions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.data_loaders.lfp import filter_ripple_lfp_time


class TestRippleLFPFiltering:
    """Test temporal filtering of ripple LFP data to position bounds."""

    def test_filter_to_position_bounds(self):
        """Ripple LFP data outside position time range should be removed."""
        # Create sample ripple data with time index from 0 to 10 seconds
        time_index = pd.Index(np.arange(0.0, 10.0, 0.001), name="time")

        ripple_data = {
            "ripple_times": pd.DataFrame({"start_time": [1.0, 3.0], "end_time": [1.5, 3.5]}),
            "ripple_filtered_lfps": pd.DataFrame(
                np.random.RandomState(42).randn(len(time_index), 4),
                index=time_index,
                columns=["electrode1", "electrode2", "electrode3", "electrode4"],
            ),
            "ripple_lfps": pd.DataFrame(
                np.random.RandomState(42).randn(len(time_index), 4),
                index=time_index,
                columns=["electrode1", "electrode2", "electrode3", "electrode4"],
            ),
            "ripple_consensus_trace": pd.Series(
                np.random.RandomState(42).randn(len(time_index)),
                index=time_index,
                name="consensus_trace",
            ),
            "zscored_ripple_consensus_trace": pd.Series(
                np.random.RandomState(42).randn(len(time_index)),
                index=time_index,
                name="zscored_consensus_trace",
            ),
        }

        # Position time from 2.0 to 8.0 seconds
        position_time = np.arange(2.0, 8.0, 0.01)

        # Apply filter (returns new dict)
        ripple_data = filter_ripple_lfp_time(ripple_data, position_time)

        # Check that filtered data is within bounds
        assert ripple_data["ripple_filtered_lfps"].index.min() >= 2.0
        assert ripple_data["ripple_filtered_lfps"].index.max() <= 8.0

        assert ripple_data["ripple_lfps"].index.min() >= 2.0
        assert ripple_data["ripple_lfps"].index.max() <= 8.0

        assert ripple_data["ripple_consensus_trace"].index.min() >= 2.0
        assert ripple_data["ripple_consensus_trace"].index.max() <= 8.0

        assert ripple_data["zscored_ripple_consensus_trace"].index.min() >= 2.0
        assert ripple_data["zscored_ripple_consensus_trace"].index.max() <= 8.0

        # Check that ripple_times IS filtered: [1.0, 1.5] is before position
        # start (2.0), so only the [3.0, 3.5] event should remain
        assert len(ripple_data["ripple_times"]) == 1
        assert ripple_data["ripple_times"].iloc[0]["start_time"] == 3.0
        assert ripple_data["ripple_times"].iloc[0]["end_time"] == 3.5

    def test_ripple_straddling_start_boundary_is_dropped(self):
        """A ripple starting before position time should be dropped entirely."""
        time_index = pd.Index(np.arange(0.0, 10.0, 0.001), name="time")

        ripple_data = {
            "ripple_times": pd.DataFrame({
                "start_time": [1.5, 3.0],
                "end_time": [2.5, 3.5],
            }),
            "ripple_filtered_lfps": pd.DataFrame(
                np.zeros((len(time_index), 1)), index=time_index, columns=["e1"],
            ),
            "ripple_lfps": pd.DataFrame(
                np.zeros((len(time_index), 1)), index=time_index, columns=["e1"],
            ),
            "ripple_consensus_trace": pd.Series(
                np.zeros(len(time_index)), index=time_index, name="c",
            ),
            "zscored_ripple_consensus_trace": pd.Series(
                np.zeros(len(time_index)), index=time_index, name="zc",
            ),
        }

        # Position starts at 2.0 — the [1.5, 2.5] ripple straddles the boundary
        position_time = np.arange(2.0, 8.0, 0.01)
        result = filter_ripple_lfp_time(ripple_data, position_time)

        # Only the fully-contained [3.0, 3.5] ripple should survive
        assert len(result["ripple_times"]) == 1
        assert result["ripple_times"].iloc[0]["start_time"] == 3.0

    def test_ripple_straddling_end_boundary_is_dropped(self):
        """A ripple ending after position time should be dropped entirely."""
        time_index = pd.Index(np.arange(0.0, 10.0, 0.001), name="time")

        ripple_data = {
            "ripple_times": pd.DataFrame({
                "start_time": [3.0, 7.5],
                "end_time": [3.5, 8.5],
            }),
            "ripple_filtered_lfps": pd.DataFrame(
                np.zeros((len(time_index), 1)), index=time_index, columns=["e1"],
            ),
            "ripple_lfps": pd.DataFrame(
                np.zeros((len(time_index), 1)), index=time_index, columns=["e1"],
            ),
            "ripple_consensus_trace": pd.Series(
                np.zeros(len(time_index)), index=time_index, name="c",
            ),
            "zscored_ripple_consensus_trace": pd.Series(
                np.zeros(len(time_index)), index=time_index, name="zc",
            ),
        }

        # Position ends at 8.0 — the [7.5, 8.5] ripple straddles the boundary
        position_time = np.arange(2.0, 8.0, 0.01)
        result = filter_ripple_lfp_time(ripple_data, position_time)

        # Only the fully-contained [3.0, 3.5] ripple should survive
        assert len(result["ripple_times"]) == 1
        assert result["ripple_times"].iloc[0]["start_time"] == 3.0

    def test_empty_dataframes(self):
        """Empty ripple data should not crash."""
        # Create empty DataFrames and Series with proper structure
        empty_time_index = pd.Index([], dtype=np.float64, name="time")

        ripple_data = {
            "ripple_times": pd.DataFrame({"start_time": [], "end_time": []}),
            "ripple_filtered_lfps": pd.DataFrame(
                index=empty_time_index,
                columns=["electrode1"],
            ),
            "ripple_lfps": pd.DataFrame(
                index=empty_time_index,
                columns=["electrode1"],
            ),
            "ripple_consensus_trace": pd.Series(
                index=empty_time_index,
                dtype=np.float64,
                name="consensus_trace",
            ),
            "zscored_ripple_consensus_trace": pd.Series(
                index=empty_time_index,
                dtype=np.float64,
                name="zscored_consensus_trace",
            ),
        }

        position_time = np.arange(1.0, 5.0, 0.01)

        # Should not raise
        ripple_data = filter_ripple_lfp_time(ripple_data, position_time)

        # Should still be empty
        assert len(ripple_data["ripple_filtered_lfps"]) == 0
        assert len(ripple_data["ripple_lfps"]) == 0
        assert len(ripple_data["ripple_consensus_trace"]) == 0
        assert len(ripple_data["zscored_ripple_consensus_trace"]) == 0

    def test_boundary_conditions_exact_start_end(self):
        """Data at exact boundaries should be included (slice is inclusive)."""
        # Create data from 1.0 to 5.0
        time_index = pd.Index(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), name="time")

        ripple_data = {
            "ripple_times": pd.DataFrame({"start_time": [2.0], "end_time": [3.0]}),
            "ripple_filtered_lfps": pd.DataFrame(
                np.random.RandomState(42).randn(len(time_index), 2),
                index=time_index,
                columns=["electrode1", "electrode2"],
            ),
            "ripple_lfps": pd.DataFrame(
                np.random.RandomState(42).randn(len(time_index), 2),
                index=time_index,
                columns=["electrode1", "electrode2"],
            ),
            "ripple_consensus_trace": pd.Series(
                np.random.RandomState(42).randn(len(time_index)),
                index=time_index,
                name="consensus_trace",
            ),
            "zscored_ripple_consensus_trace": pd.Series(
                np.random.RandomState(42).randn(len(time_index)),
                index=time_index,
                name="zscored_consensus_trace",
            ),
        }

        # Position time from 2.0 to 4.0 (exact boundaries)
        position_time = np.array([2.0, 3.0, 4.0])

        ripple_data = filter_ripple_lfp_time(ripple_data, position_time)

        # Should include boundary times (pandas slice is inclusive on both ends)
        assert len(ripple_data["ripple_filtered_lfps"]) == 3
        np.testing.assert_array_equal(
            ripple_data["ripple_filtered_lfps"].index.to_numpy(),
            np.array([2.0, 3.0, 4.0])
        )

        assert len(ripple_data["ripple_consensus_trace"]) == 3
        np.testing.assert_array_equal(
            ripple_data["ripple_consensus_trace"].index.to_numpy(),
            np.array([2.0, 3.0, 4.0])
        )

    def test_all_data_outside_bounds(self):
        """If all ripple data outside position bounds, should return empty data."""
        # Ripple data from 0 to 1 second
        time_index = pd.Index(np.arange(0.0, 1.0, 0.01), name="time")

        ripple_data = {
            "ripple_times": pd.DataFrame({"start_time": [0.2], "end_time": [0.5]}),
            "ripple_filtered_lfps": pd.DataFrame(
                np.random.RandomState(42).randn(len(time_index), 2),
                index=time_index,
                columns=["electrode1", "electrode2"],
            ),
            "ripple_lfps": pd.DataFrame(
                np.random.RandomState(42).randn(len(time_index), 2),
                index=time_index,
                columns=["electrode1", "electrode2"],
            ),
            "ripple_consensus_trace": pd.Series(
                np.random.RandomState(42).randn(len(time_index)),
                index=time_index,
                name="consensus_trace",
            ),
            "zscored_ripple_consensus_trace": pd.Series(
                np.random.RandomState(42).randn(len(time_index)),
                index=time_index,
                name="zscored_consensus_trace",
            ),
        }

        # Position time from 5 to 10 seconds (completely outside ripple data)
        position_time = np.arange(5.0, 10.0, 0.01)

        ripple_data = filter_ripple_lfp_time(ripple_data, position_time)

        # Should have empty filtered data
        assert len(ripple_data["ripple_filtered_lfps"]) == 0
        assert len(ripple_data["ripple_lfps"]) == 0
        assert len(ripple_data["ripple_consensus_trace"]) == 0
        assert len(ripple_data["zscored_ripple_consensus_trace"]) == 0

    def test_no_data_filtered_all_within_bounds(self):
        """If all ripple data within position bounds, all should be preserved."""
        # Ripple data from 2.0 to 4.0
        time_index = pd.Index(np.arange(2.0, 4.0, 0.01), name="time")
        original_length = len(time_index)

        ripple_data = {
            "ripple_times": pd.DataFrame({"start_time": [2.5], "end_time": [3.5]}),
            "ripple_filtered_lfps": pd.DataFrame(
                np.random.RandomState(42).randn(original_length, 3),
                index=time_index,
                columns=["electrode1", "electrode2", "electrode3"],
            ),
            "ripple_lfps": pd.DataFrame(
                np.random.RandomState(42).randn(original_length, 3),
                index=time_index,
                columns=["electrode1", "electrode2", "electrode3"],
            ),
            "ripple_consensus_trace": pd.Series(
                np.random.RandomState(42).randn(original_length),
                index=time_index,
                name="consensus_trace",
            ),
            "zscored_ripple_consensus_trace": pd.Series(
                np.random.RandomState(42).randn(original_length),
                index=time_index,
                name="zscored_consensus_trace",
            ),
        }

        # Position time encompasses all ripple data (1.0 to 5.0)
        position_time = np.arange(1.0, 5.0, 0.01)

        ripple_data = filter_ripple_lfp_time(ripple_data, position_time)

        # All data should be preserved
        assert len(ripple_data["ripple_filtered_lfps"]) == original_length
        assert len(ripple_data["ripple_lfps"]) == original_length
        assert len(ripple_data["ripple_consensus_trace"]) == original_length
        assert len(ripple_data["zscored_ripple_consensus_trace"]) == original_length

    def test_preserves_data_structure(self):
        """Filtering should preserve DataFrame/Series structure and column names."""
        time_index = pd.Index(np.arange(0.0, 10.0, 0.01), name="time")

        ripple_data = {
            "ripple_times": pd.DataFrame({"start_time": [1.0], "end_time": [2.0]}),
            "ripple_filtered_lfps": pd.DataFrame(
                np.random.RandomState(42).randn(len(time_index), 3),
                index=time_index,
                columns=["CA1_01", "CA1_02", "CA1_03"],
            ),
            "ripple_lfps": pd.DataFrame(
                np.random.RandomState(42).randn(len(time_index), 3),
                index=time_index,
                columns=["CA1_01", "CA1_02", "CA1_03"],
            ),
            "ripple_consensus_trace": pd.Series(
                np.random.RandomState(42).randn(len(time_index)),
                index=time_index,
                name="consensus",
            ),
            "zscored_ripple_consensus_trace": pd.Series(
                np.random.RandomState(42).randn(len(time_index)),
                index=time_index,
                name="zscored",
            ),
        }

        position_time = np.arange(2.0, 5.0, 0.01)

        ripple_data = filter_ripple_lfp_time(ripple_data, position_time)

        # Check that column names are preserved
        assert list(ripple_data["ripple_filtered_lfps"].columns) == ["CA1_01", "CA1_02", "CA1_03"]
        assert list(ripple_data["ripple_lfps"].columns) == ["CA1_01", "CA1_02", "CA1_03"]

        # Check that Series names are preserved
        assert ripple_data["ripple_consensus_trace"].name == "consensus"
        assert ripple_data["zscored_ripple_consensus_trace"].name == "zscored"

        # Check that data types are preserved
        assert isinstance(ripple_data["ripple_filtered_lfps"], pd.DataFrame)
        assert isinstance(ripple_data["ripple_lfps"], pd.DataFrame)
        assert isinstance(ripple_data["ripple_consensus_trace"], pd.Series)
        assert isinstance(ripple_data["zscored_ripple_consensus_trace"], pd.Series)

    def test_returns_new_dict(self):
        """Function should return new dict (functional style)."""
        time_index = pd.Index(np.arange(0.0, 10.0, 0.1), name="time")

        original_ripple_data = {
            "ripple_times": pd.DataFrame({"start_time": [1.0], "end_time": [2.0]}),
            "ripple_filtered_lfps": pd.DataFrame(
                np.random.RandomState(42).randn(len(time_index), 2),
                index=time_index,
                columns=["e1", "e2"],
            ),
            "ripple_lfps": pd.DataFrame(
                np.random.RandomState(42).randn(len(time_index), 2),
                index=time_index,
                columns=["e1", "e2"],
            ),
            "ripple_consensus_trace": pd.Series(
                np.random.RandomState(42).randn(len(time_index)),
                index=time_index,
            ),
            "zscored_ripple_consensus_trace": pd.Series(
                np.random.RandomState(42).randn(len(time_index)),
                index=time_index,
            ),
        }

        # Store original length
        original_length = len(original_ripple_data["ripple_filtered_lfps"])

        position_time = np.arange(2.0, 5.0, 0.01)

        # Function returns new dict (functional style)
        result = filter_ripple_lfp_time(original_ripple_data, position_time)

        assert result is not None, "Function should return new dict"
        assert result is not original_ripple_data, "Should be a new dict object"
        assert len(result["ripple_filtered_lfps"]) < original_length, "Returned data should be filtered"
        assert len(original_ripple_data["ripple_filtered_lfps"]) == original_length, "Original data should be unchanged"
