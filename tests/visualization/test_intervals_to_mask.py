"""Tests for _intervals_to_mask utility function."""

from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.visualization.plots import _intervals_to_mask


class TestIntervalsToMask:
    def test_single_interval(self):
        intervals = pd.DataFrame({
            "start_time": [1.0],
            "end_time": [2.0],
        })
        time_idx = np.array([0.5, 1.0, 1.5, 2.0, 2.5])

        mask = _intervals_to_mask(intervals, time_idx)

        np.testing.assert_array_equal(mask, [False, True, True, True, False])

    def test_multiple_intervals(self):
        intervals = pd.DataFrame({
            "start_time": [1.0, 3.0],
            "end_time": [2.0, 4.0],
        })
        time_idx = np.array([0.5, 1.5, 2.5, 3.5, 4.5])

        mask = _intervals_to_mask(intervals, time_idx)

        np.testing.assert_array_equal(mask, [False, True, False, True, False])

    def test_empty_intervals(self):
        intervals = pd.DataFrame({"start_time": [], "end_time": []})
        time_idx = np.array([1.0, 2.0, 3.0])

        mask = _intervals_to_mask(intervals, time_idx)

        assert mask.shape == (3,)
        assert not mask.any()

    def test_empty_time_idx(self):
        intervals = pd.DataFrame({
            "start_time": [1.0],
            "end_time": [2.0],
        })
        time_idx = np.array([])

        mask = _intervals_to_mask(intervals, time_idx)

        assert mask.shape == (0,)

    def test_overlapping_intervals(self):
        intervals = pd.DataFrame({
            "start_time": [1.0, 1.5],
            "end_time": [2.0, 2.5],
        })
        time_idx = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

        mask = _intervals_to_mask(intervals, time_idx)

        np.testing.assert_array_equal(
            mask, [False, True, True, True, True, False]
        )

    def test_boundary_inclusive(self):
        """Start and end times should be inclusive."""
        intervals = pd.DataFrame({
            "start_time": [1.0],
            "end_time": [1.0],
        })
        time_idx = np.array([0.999, 1.0, 1.001])

        mask = _intervals_to_mask(intervals, time_idx)

        assert mask[1]  # exact match should be True
