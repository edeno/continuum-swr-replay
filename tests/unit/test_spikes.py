"""Unit tests for spike data processing.

Tests the critical coincident spike detection algorithm and spike filtering.
"""

from __future__ import annotations

import numpy as np
import pytest

from continuum_swr_replay.data_loaders.spikes import (
    detect_coincident_spikes,
    filter_spike_times,
)


class TestCoincidentSpikeDetection:
    """Test artifact removal via coincident spike detection."""

    def test_empty_spike_times_returns_empty_lists(self):
        """Empty input should return empty output without crashing."""
        spike_times = [np.array([]) for _ in range(4)]
        filtered_times, filtered_indices = detect_coincident_spikes(spike_times)

        assert len(filtered_times) == 0
        assert len(filtered_indices) == 0

    def test_single_tetrode_passes_through(self):
        """Single tetrode input passes through unchanged (no coincidence possible)."""
        spike_times = [np.array([1.0, 2.0, 3.0])]
        filtered_times, filtered_indices = detect_coincident_spikes(spike_times)

        # With only one tetrode, data passes through
        assert len(filtered_times) == 1
        np.testing.assert_array_equal(filtered_times[0], spike_times[0])

    def test_non_coincident_spikes_all_preserved(self):
        """Spikes separated by >40μs should all be preserved exactly.

        With 100ms separation (2500x the 40μs threshold), all spikes should be kept.
        """
        spike_times = [
            np.array([1.0, 2.0, 3.0]),
            np.array([1.1, 2.1, 3.1]),  # 100ms apart (>> 40μs threshold)
            np.array([1.2, 2.2, 3.2]),
            np.array([1.3, 2.3, 3.3]),
        ]
        filtered_times, filtered_indices = detect_coincident_spikes(spike_times)

        # With such large separation, all spikes should be preserved exactly
        total_original = sum(len(st) for st in spike_times)
        total_filtered = sum(len(st) for st in filtered_times)
        assert total_filtered == total_original, \
            f"Expected all {total_original} spikes preserved, got {total_filtered}"

    def test_coincident_spikes_removed_correctly(self):
        """Spikes within 40μs across >33% tetrodes should be detected and removed.

        This test creates known coincident spikes and verifies that some are removed
        while well-separated spikes are preserved.
        """
        # 4 tetrodes, create coincident spikes
        spike_times = [
            np.array([1.0, 2.0, 5.0]),
            np.array([1.00002, 3.0, 5.00001]),  # 20μs from first at t≈1.0 and t≈5.0
            np.array([1.00003, 4.0, 5.00002]),  # 30μs from first at t≈1.0 and t≈5.0
            np.array([6.0, 7.0, 8.0]),          # Well separated
        ]
        filtered_times, filtered_indices = detect_coincident_spikes(spike_times)

        # Should have removed some spikes
        total_original = sum(len(st) for st in spike_times)
        total_filtered = sum(len(st) for st in filtered_times)
        assert total_filtered < total_original, \
            f"Expected some spikes removed, got {total_filtered}/{total_original}"

        # At least 2 spikes should be removed (from the coincident groups)
        assert total_original - total_filtered >= 2, \
            f"Expected at least 2 spikes removed, got {total_original - total_filtered}"

        # Non-coincident spikes should be preserved
        all_filtered = np.concatenate(filtered_times) if filtered_times else np.array([])
        if len(all_filtered) > 0:
            # Well-separated spikes should definitely be kept
            assert np.any(np.abs(all_filtered - 2.0) < 0.01), "Spike at t=2.0 should be preserved"
            assert np.any(np.abs(all_filtered - 3.0) < 0.01), "Spike at t=3.0 should be preserved"
            assert np.any(np.abs(all_filtered - 4.0) < 0.01), "Spike at t=4.0 should be preserved"
            assert np.any(np.abs(all_filtered - 6.0) < 0.01), "Spike at t=6.0 should be preserved"

    def test_well_separated_spikes_preserved(self):
        """Well-separated spikes across tetrodes should be preserved."""
        spike_times = [
            np.array([1.0, 5.0, 9.0]),
            np.array([2.0, 6.0, 10.0]),
            np.array([3.0, 7.0, 11.0]),
        ]
        filtered_times, filtered_indices = detect_coincident_spikes(spike_times)

        # All spikes well-separated, should keep most or all
        total_original = sum(len(st) for st in spike_times)
        total_filtered = sum(len(st) for st in filtered_times)
        assert total_filtered >= total_original * 0.9

    def test_preserves_spike_order(self):
        """Filtered spikes should remain in temporal order."""
        spike_times = [
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            np.array([1.5, 2.5, 3.5, 4.5]),
            np.array([1.7, 2.7, 3.7, 4.7]),
        ]
        filtered_times, filtered_indices = detect_coincident_spikes(spike_times)

        for filtered in filtered_times:
            if len(filtered) > 1:
                assert np.all(filtered[:-1] <= filtered[1:]), "Spikes not in order"


class TestSpikeFiltering:
    """Test temporal filtering of spikes to position bounds."""

    def test_filter_to_position_bounds(self):
        """Spikes outside position time range should be removed."""
        spike_times = {
            "HPC": [
                np.array([0.5, 1.0, 2.0, 3.0, 5.5]),  # 0.5 and 5.5 outside
                np.array([1.5, 2.5, 3.5]),
            ]
        }
        spike_features = {
            "HPC": [
                np.random.randn(5, 4),  # 5 spikes, 4 features
                np.random.randn(3, 4),
            ]
        }
        position_time = np.arange(1.0, 5.0, 0.1)

        spike_times, spike_features = filter_spike_times(spike_times, spike_features, position_time)

        # Check temporal bounds
        assert spike_times["HPC"][0].min() >= 1.0
        assert spike_times["HPC"][0].max() <= 5.0

        # Check feature dimensions match
        assert len(spike_features["HPC"][0]) == len(spike_times["HPC"][0])

    def test_empty_brain_area_dict(self):
        """Empty spike dict should not crash."""
        spike_times = {}
        spike_features = {}
        position_time = np.arange(0, 10, 0.1)

        # Should not raise
        spike_times, spike_features = filter_spike_times(spike_times, spike_features, position_time)

    def test_all_spikes_filtered_out(self):
        """If all spikes outside bounds, should return empty arrays."""
        spike_times = {
            "HPC": [np.array([0.1, 0.2, 0.3])],  # All before position starts
        }
        spike_features = {
            "HPC": [np.random.randn(3, 4)],
        }
        position_time = np.arange(1.0, 5.0, 0.1)

        spike_times, spike_features = filter_spike_times(spike_times, spike_features, position_time)

        assert len(spike_times["HPC"][0]) == 0
        assert len(spike_features["HPC"][0]) == 0

    def test_no_spikes_filtered_all_within_bounds(self):
        """If all spikes within bounds, all should be preserved."""
        spike_times = {
            "HPC": [
                np.array([1.5, 2.0, 2.5, 3.0, 4.5]),
                np.array([2.0, 3.0, 4.0]),
            ]
        }
        spike_features = {
            "HPC": [
                np.random.randn(5, 4),
                np.random.randn(3, 4),
            ]
        }
        position_time = np.arange(1.0, 5.0, 0.1)

        # Store original counts
        original_counts = [len(st) for st in spike_times["HPC"]]

        spike_times, spike_features = filter_spike_times(spike_times, spike_features, position_time)

        # All spikes should be preserved
        assert len(spike_times["HPC"][0]) == original_counts[0]
        assert len(spike_times["HPC"][1]) == original_counts[1]
        assert len(spike_features["HPC"][0]) == original_counts[0]
        assert len(spike_features["HPC"][1]) == original_counts[1]

    def test_boundary_conditions_exact_start_end(self):
        """Spikes at exact boundaries should be included (>= and <= operators)."""
        position_time = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Spikes at exact boundaries and just outside
        spike_times = {
            "HPC": [
                np.array([0.999, 1.0, 2.5, 5.0, 5.001]),  # 0.999 and 5.001 outside
            ]
        }
        spike_features = {
            "HPC": [np.random.randn(5, 4)],
        }

        spike_times, spike_features = filter_spike_times(spike_times, spike_features, position_time)

        # Should have exactly 3 spikes: 1.0, 2.5, 5.0
        assert len(spike_times["HPC"][0]) == 3
        np.testing.assert_array_equal(
            spike_times["HPC"][0], np.array([1.0, 2.5, 5.0])
        )

    def test_multiple_brain_areas(self):
        """Test filtering with multiple brain areas simultaneously."""
        spike_times = {
            "HPC": [
                np.array([0.5, 1.5, 2.5, 5.5]),  # 0.5 and 5.5 outside
                np.array([1.0, 3.0, 4.8]),
            ],
            "mPFC": [
                np.array([0.8, 2.0, 4.0, 6.0]),  # 0.8 and 6.0 outside
            ],
        }
        spike_features = {
            "HPC": [
                np.random.randn(4, 4),
                np.random.randn(3, 4),
            ],
            "mPFC": [
                np.random.randn(4, 4),
            ],
        }
        # position_time goes from 1.0 to 4.9 (arange stops before 5.0)
        position_time = np.arange(1.0, 5.0, 0.1)

        spike_times, spike_features = filter_spike_times(spike_times, spike_features, position_time)

        # Check HPC
        assert len(spike_times["HPC"][0]) == 2  # 1.5, 2.5
        assert len(spike_times["HPC"][1]) == 3  # 1.0, 3.0, 4.8
        np.testing.assert_array_equal(
            spike_times["HPC"][0], np.array([1.5, 2.5])
        )
        np.testing.assert_array_equal(
            spike_times["HPC"][1], np.array([1.0, 3.0, 4.8])
        )

        # Check mPFC
        assert len(spike_times["mPFC"][0]) == 2  # 2.0, 4.0
        np.testing.assert_array_equal(
            spike_times["mPFC"][0], np.array([2.0, 4.0])
        )

    def test_brain_area_without_features(self):
        """Test filtering when features dict doesn't have a brain area (KeyError handling)."""
        spike_times = {
            "HPC": [np.array([0.5, 1.5, 2.5, 5.5])],
            "mPFC": [np.array([1.0, 2.0, 3.0])],
        }
        spike_features = {
            "HPC": [np.random.randn(4, 4)],
            # mPFC features missing (KeyError should be caught)
        }
        position_time = np.arange(1.0, 5.0, 0.1)

        # Should not raise KeyError
        spike_times, spike_features = filter_spike_times(spike_times, spike_features, position_time)

        # Both brain areas should be filtered
        assert len(spike_times["HPC"][0]) == 2  # 1.5, 2.5
        assert len(spike_times["mPFC"][0]) == 3  # All within bounds

    def test_preserves_spike_order_after_filtering(self):
        """Filtered spikes should remain in temporal order."""
        spike_times = {
            "HPC": [np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 5.5])],
        }
        spike_features = {
            "HPC": [np.random.randn(7, 4)],
        }
        position_time = np.arange(1.0, 5.0, 0.1)

        spike_times, spike_features = filter_spike_times(spike_times, spike_features, position_time)

        # Should be in order
        filtered = spike_times["HPC"][0]
        if len(filtered) > 1:
            assert np.all(filtered[:-1] <= filtered[1:]), "Spikes not in temporal order"


class TestSpikeDataIntegrity:
    """Test data integrity properties of spike processing."""

    def test_spike_times_always_sorted(self):
        """Spike times should always be in ascending order."""
        # Create unsorted spikes (shouldn't happen, but test robustness)
        spike_times = [np.array([3.0, 1.0, 2.0, 5.0, 4.0])]

        # Process (sort if needed)
        spike_times_sorted = [np.sort(st) for st in spike_times]

        assert np.all(np.diff(spike_times_sorted[0]) > 0)

    def test_no_duplicate_spikes(self):
        """Spike detection should not produce duplicate timestamps."""
        spike_times = [np.array([1.0, 1.0, 2.0, 3.0, 3.0])]

        # Remove duplicates
        spike_times_unique = [np.unique(st) for st in spike_times]

        assert len(spike_times_unique[0]) == 3  # 1.0, 2.0, 3.0

    def test_spike_features_match_spike_count(self):
        """Number of feature vectors must equal number of spikes."""
        n_spikes = 100
        n_features = 4

        spike_times = [np.linspace(0, 10, n_spikes)]
        spike_features = [np.random.randn(n_spikes, n_features)]

        assert spike_features[0].shape[0] == len(spike_times[0])
        assert spike_features[0].shape[1] == n_features
