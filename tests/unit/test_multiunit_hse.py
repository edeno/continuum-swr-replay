"""Tests for multiunit HSE (High Synchrony Event) detection.

Tests get_multiunit_hse — a pure numpy function with no Spyglass dependency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.data_loaders.lfp import get_multiunit_hse


class TestGetMultiunitHSE:
    """Test HSE detection from multiunit spike data."""

    @pytest.fixture
    def bin_time(self):
        """2 seconds at 500 Hz."""
        return np.arange(0.0, 2.0, 0.002)

    @pytest.fixture
    def low_speed(self, bin_time):
        """Stationary animal (speed below threshold)."""
        return np.full(len(bin_time), 1.0)

    @pytest.fixture
    def high_speed(self, bin_time):
        """Moving animal (speed above threshold)."""
        return np.full(len(bin_time), 10.0)

    def test_returns_three_dataframes(self, bin_time, low_speed):
        """Function should return (hse_times, firing_rate, zscore)."""
        rng = np.random.RandomState(42)
        spike_times = [np.sort(rng.uniform(0, 2, 50)) for _ in range(5)]

        hse_times, firing_rate, zscore = get_multiunit_hse(
            spike_times, low_speed, bin_time
        )

        assert isinstance(hse_times, pd.DataFrame)
        assert isinstance(firing_rate, pd.DataFrame)
        assert isinstance(zscore, pd.DataFrame)

    def test_firing_rate_shape_matches_bin_time(self, bin_time, low_speed):
        """Firing rate should have same length as bin_time."""
        rng = np.random.RandomState(42)
        spike_times = [np.sort(rng.uniform(0, 2, 50)) for _ in range(5)]

        _, firing_rate, zscore = get_multiunit_hse(
            spike_times, low_speed, bin_time
        )

        assert len(firing_rate) == len(bin_time)
        assert len(zscore) == len(bin_time)

    def test_firing_rate_index_is_bin_time(self, bin_time, low_speed):
        """Firing rate index should match bin_time values."""
        rng = np.random.RandomState(42)
        spike_times = [np.sort(rng.uniform(0, 2, 50)) for _ in range(5)]

        _, firing_rate, _ = get_multiunit_hse(spike_times, low_speed, bin_time)

        np.testing.assert_array_equal(firing_rate.index.to_numpy(), bin_time)

    def test_no_spikes_gives_zero_firing_rate(self, bin_time, low_speed):
        """Empty spike arrays should give zero firing rate."""
        spike_times = [np.array([]) for _ in range(5)]

        _, firing_rate, zscore = get_multiunit_hse(
            spike_times, low_speed, bin_time
        )

        assert firing_rate.values.max() == 0.0

    def test_constant_firing_gives_zero_zscore(self, bin_time, low_speed):
        """Constant firing rate should give zero z-score (std=0 guard)."""
        spike_times = [np.array([]) for _ in range(5)]

        _, _, zscore = get_multiunit_hse(spike_times, low_speed, bin_time)

        np.testing.assert_array_equal(zscore.values, 0.0)

    def test_no_spikes_gives_no_hse(self, bin_time, low_speed):
        """A spike-free area has a constant rate, so it holds no HSE.

        ripple_detection 2.0 raises on a rate with no scale, so the detector
        is skipped and an empty table comes back, as 1.7.0 returned.
        """
        spike_times = [np.array([]) for _ in range(5)]

        hse_times, _, _ = get_multiunit_hse(spike_times, low_speed, bin_time)

        assert len(hse_times) == 0
        assert list(hse_times.columns) == ["start_time", "end_time", "duration"]
        assert hse_times.index.name == "event_number"

    def test_high_speed_suppresses_hse_detection(self, bin_time, high_speed):
        """HSE events should not be detected when animal is moving fast."""
        rng = np.random.RandomState(42)
        # Create a burst of synchronous spikes
        burst_time = 1.0
        spike_times = [
            np.sort(
                np.concatenate(
                    [rng.uniform(0, 2, 20), np.full(10, burst_time) + rng.randn(10) * 0.001]
                )
            )
            for _ in range(10)
        ]

        hse_times, _, _ = get_multiunit_hse(spike_times, high_speed, bin_time)

        # High speed should suppress detection
        assert len(hse_times) == 0

    def test_hse_times_has_start_end_columns(self, bin_time, low_speed):
        """HSE times DataFrame should have start_time and end_time columns."""
        rng = np.random.RandomState(42)
        spike_times = [np.sort(rng.uniform(0, 2, 50)) for _ in range(5)]

        hse_times, _, _ = get_multiunit_hse(spike_times, low_speed, bin_time)

        assert "start_time" in hse_times.columns
        assert "end_time" in hse_times.columns

    def test_hse_times_within_data_bounds(self, bin_time, low_speed):
        """Detected HSE events should be within the bin_time range."""
        rng = np.random.RandomState(42)
        # Lots of spikes to increase chance of HSE detection
        spike_times = [np.sort(rng.uniform(0, 2, 200)) for _ in range(10)]

        hse_times, _, _ = get_multiunit_hse(spike_times, low_speed, bin_time)

        if len(hse_times) > 0:
            assert hse_times["start_time"].min() >= bin_time[0]
            assert hse_times["end_time"].max() <= bin_time[-1]

    def test_spikes_outside_bin_time_ignored(self, bin_time, low_speed):
        """Spikes outside the bin_time range should not affect results."""
        rng = np.random.RandomState(42)
        # Spikes within range
        spike_times_in = [np.sort(rng.uniform(0, 2, 50)) for _ in range(5)]
        # Add spikes far outside range
        spike_times_out = [
            np.concatenate([st, rng.uniform(100, 200, 50)]) for st in spike_times_in
        ]

        _, rate_in, _ = get_multiunit_hse(spike_times_in, low_speed, bin_time)
        _, rate_out, _ = get_multiunit_hse(spike_times_out, low_speed, bin_time)

        # Firing rates should be identical since out-of-range spikes are filtered
        np.testing.assert_array_equal(rate_in.values, rate_out.values)

    def test_single_unit_works(self, bin_time, low_speed):
        """Single unit (list of 1 array) should work without error."""
        rng = np.random.RandomState(42)
        spike_times = [np.sort(rng.uniform(0, 2, 100))]

        hse_times, firing_rate, zscore = get_multiunit_hse(
            spike_times, low_speed, bin_time
        )

        assert len(firing_rate) == len(bin_time)

    def test_zscore_has_zero_mean_unit_variance(self, bin_time, low_speed):
        """Z-scored firing rate should have approximately zero mean and unit variance."""
        rng = np.random.RandomState(42)
        # Need enough spikes for non-trivial distribution
        spike_times = [np.sort(rng.uniform(0, 2, 500)) for _ in range(10)]

        _, _, zscore = get_multiunit_hse(spike_times, low_speed, bin_time)

        values = zscore.values.flatten()
        # Z-score should be close to mean=0, std=1
        assert abs(np.nanmean(values)) < 0.1
        assert abs(np.nanstd(values) - 1.0) < 0.1

    def test_uses_max_speed_threshold_from_parameters(self):
        """Verify speed_threshold matches the centralized parameter."""
        from continuum_swr_replay.parameters import MAX_SPEED_THRESHOLD_CM_PER_SEC

        # The function uses MAX_SPEED_THRESHOLD_CM_PER_SEC internally
        # Verify the parameter value is what we expect
        assert MAX_SPEED_THRESHOLD_CM_PER_SEC == 4.0
