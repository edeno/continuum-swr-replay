"""Tests for spike count binning (get_spikecount_per_time_bin).

This function uses np.digitize + np.bincount, which is prone to off-by-one
errors at bin boundaries. These tests verify correctness against hand-computed
expected values.
"""

from __future__ import annotations

import numpy as np

from continuum_swr_replay.analysis.pfc.glm import get_spikecount_per_time_bin


class TestGetSpikecountPerTimeBin:
    """Test spike binning correctness."""

    def test_single_spike_per_bin(self):
        """One spike per bin should give counts of 1."""
        time = np.array([0.0, 1.0, 2.0, 3.0])
        # One spike in each bin: [0, 1), [1, 2), [2, 3]
        spike_times = np.array([0.5, 1.5, 2.5])

        counts = get_spikecount_per_time_bin(spike_times, time)

        assert len(counts) == len(time)
        # Spikes at 0.5, 1.5, 2.5 should land in bins 0, 1, 2
        assert counts[0] == 1  # [0.0, 1.0)
        assert counts[1] == 1  # [1.0, 2.0)
        assert counts[2] == 1  # [2.0, 3.0]

    def test_no_spikes_gives_zeros(self):
        """No spikes should give all-zero counts."""
        time = np.array([0.0, 1.0, 2.0, 3.0])
        spike_times = np.array([])

        counts = get_spikecount_per_time_bin(spike_times, time)

        assert len(counts) == len(time)
        np.testing.assert_array_equal(counts, 0)

    def test_multiple_spikes_in_same_bin(self):
        """Multiple spikes in same bin should be counted."""
        time = np.array([0.0, 1.0, 2.0])
        spike_times = np.array([0.1, 0.3, 0.5, 0.9])  # All in first bin

        counts = get_spikecount_per_time_bin(spike_times, time)

        assert counts[0] == 4
        assert counts[1] == 0

    def test_spike_at_bin_edge_goes_to_correct_bin(self):
        """Spike exactly at a bin edge should be assigned consistently."""
        time = np.array([0.0, 1.0, 2.0, 3.0])
        spike_times = np.array([1.0])  # Exactly at edge

        counts = get_spikecount_per_time_bin(spike_times, time)

        # np.digitize with time[1:-1] = [1.0, 2.0]:
        # spike at 1.0 gets index 1 (since digitize uses right=False by default)
        assert counts.sum() == 1, "Spike should be counted exactly once"
        assert counts[1] == 1, "Spike at edge 1.0 should land in bin index 1"

    def test_spikes_outside_range_are_excluded(self):
        """Spikes before start or after end of time should be excluded."""
        time = np.array([1.0, 2.0, 3.0])
        spike_times = np.array([-1.0, 0.5, 1.5, 3.5, 10.0])

        counts = get_spikecount_per_time_bin(spike_times, time)

        # Only spike at 1.5 is within [1.0, 3.0]
        assert counts.sum() == 1

    def test_output_length_matches_time_length(self):
        """Output array should always have same length as time array."""
        for n_bins in [2, 5, 10, 100]:
            time = np.linspace(0, 10, n_bins)
            spike_times = np.array([5.0])

            counts = get_spikecount_per_time_bin(spike_times, time)

            assert len(counts) == n_bins

    def test_total_count_equals_spikes_in_range(self):
        """Total spike count should equal number of spikes within time range."""
        time = np.arange(0.0, 10.0, 0.002)  # 500 Hz
        rng = np.random.RandomState(42)
        all_spikes = np.sort(rng.uniform(-5, 15, 200))
        in_range = (all_spikes >= time[0]) & (all_spikes <= time[-1])
        expected_total = in_range.sum()

        counts = get_spikecount_per_time_bin(all_spikes, time)

        assert counts.sum() == expected_total

    def test_high_frequency_regular_time(self):
        """Test with realistic 500 Hz time bins and known spike locations."""
        time = np.arange(0.0, 1.0, 0.002)  # 500 Hz, 1 second
        # Place spikes at known positions
        spike_times = np.array([0.001, 0.003, 0.005])

        counts = get_spikecount_per_time_bin(spike_times, time)

        assert len(counts) == len(time)
        assert counts.sum() == 3

    def test_spike_at_first_time_point(self):
        """Spike exactly at the first time point should be counted."""
        time = np.array([0.0, 1.0, 2.0])
        spike_times = np.array([0.0])

        counts = get_spikecount_per_time_bin(spike_times, time)

        assert counts.sum() == 1

    def test_spike_at_last_time_point(self):
        """Spike exactly at the last time point should be counted."""
        time = np.array([0.0, 1.0, 2.0])
        spike_times = np.array([2.0])

        counts = get_spikecount_per_time_bin(spike_times, time)

        assert counts.sum() == 1
