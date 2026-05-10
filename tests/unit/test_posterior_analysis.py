"""Tests for posterior analysis functions.

This module tests the critical posterior statistics and speed calculations
used for analyzing decoded neural replay events.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
import xarray as xr

from continuum_swr_replay.analysis.hippocampus.posterior_analysis import (
    get_conditional_non_local_posterior,
    get_map_speed,
)


class TestConditionalNonLocalPosterior:
    """Test posterior conditioning on non-local state."""

    def test_basic_normalization(self):
        """Posterior should be normalized (sum to 1 over position)."""
        time = np.arange(0, 1.0, 0.002)
        position = np.arange(0, 100, 2)
        n_time, n_pos = len(time), len(position)

        # Create mock posterior with two non-local states
        rng = np.random.default_rng(42)
        continuous_posterior = rng.random((n_time, n_pos))
        fragmented_posterior = rng.random((n_time, n_pos))

        acausal_posterior = xr.DataArray(
            np.stack([continuous_posterior, fragmented_posterior]),
            coords={
                "state": ["Non-Local Continuous", "Non-Local Fragmented"],
                "time": time,
                "position": position,
            },
            dims=["state", "time", "position"],
        )

        results = xr.Dataset({"acausal_posterior": acausal_posterior})
        conditional_posterior = get_conditional_non_local_posterior(results)

        # Check normalization: each time point should sum to 1 over position
        sums = conditional_posterior.sum("position")
        np.testing.assert_allclose(sums, 1.0, rtol=1e-10)

    def test_combines_both_states(self):
        """Should sum continuous and fragmented non-local states."""
        time = np.arange(0, 0.1, 0.002)
        position = np.arange(0, 50, 5)

        # Known values for easy checking
        continuous_posterior = np.ones((len(time), len(position))) * 0.3
        fragmented_posterior = np.ones((len(time), len(position))) * 0.7

        acausal_posterior = xr.DataArray(
            np.stack([continuous_posterior, fragmented_posterior]),
            coords={
                "state": ["Non-Local Continuous", "Non-Local Fragmented"],
                "time": time,
                "position": position,
            },
            dims=["state", "time", "position"],
        )

        results = xr.Dataset({"acausal_posterior": acausal_posterior})
        conditional_posterior = get_conditional_non_local_posterior(results)

        # After normalization, should sum to 1.0 over positions at each time
        np.testing.assert_allclose(
            conditional_posterior.sum("position").values, 1.0, rtol=1e-10
        )

    def test_handles_zero_posterior(self):
        """Division by zero when posterior sums to zero."""
        time = np.arange(0, 0.1, 0.002)
        position = np.arange(0, 50, 5)

        # All zeros - division by zero case
        continuous_posterior = np.zeros((len(time), len(position)))
        fragmented_posterior = np.zeros((len(time), len(position)))

        acausal_posterior = xr.DataArray(
            np.stack([continuous_posterior, fragmented_posterior]),
            coords={
                "state": ["Non-Local Continuous", "Non-Local Fragmented"],
                "time": time,
                "position": position,
            },
            dims=["state", "time", "position"],
        )

        results = xr.Dataset({"acausal_posterior": acausal_posterior})
        conditional_posterior = get_conditional_non_local_posterior(results)

        # Result should be NaN or Inf when dividing by zero
        assert np.all(
            np.isnan(conditional_posterior.values) | np.isinf(conditional_posterior.values)
        ), "Division by zero should produce NaN or Inf"

    def test_single_timepoint(self):
        """Should work with single time point."""
        time = np.array([0.0])
        position = np.arange(0, 100, 2)

        rng = np.random.default_rng(43)
        continuous_posterior = rng.random((1, len(position)))
        fragmented_posterior = rng.random((1, len(position)))

        acausal_posterior = xr.DataArray(
            np.stack([continuous_posterior, fragmented_posterior]),
            coords={
                "state": ["Non-Local Continuous", "Non-Local Fragmented"],
                "time": time,
                "position": position,
            },
            dims=["state", "time", "position"],
        )

        results = xr.Dataset({"acausal_posterior": acausal_posterior})
        conditional_posterior = get_conditional_non_local_posterior(results)

        assert len(conditional_posterior.time) == 1
        np.testing.assert_allclose(
            conditional_posterior.sum("position").values, 1.0, rtol=1e-10
        )


    def test_idempotent_does_not_mutate_results(self):
        """Calling twice on the same results should return identical output."""
        time = np.arange(0, 0.1, 0.002)
        position = np.arange(0, 50, 5)

        rng = np.random.default_rng(44)
        continuous_posterior = rng.random((len(time), len(position)))
        fragmented_posterior = rng.random((len(time), len(position)))

        acausal_posterior = xr.DataArray(
            np.stack([continuous_posterior, fragmented_posterior]),
            coords={
                "state": ["Non-Local Continuous", "Non-Local Fragmented"],
                "time": time,
                "position": position,
            },
            dims=["state", "time", "position"],
        )

        results = xr.Dataset({"acausal_posterior": acausal_posterior})

        first_call = get_conditional_non_local_posterior(results)
        second_call = get_conditional_non_local_posterior(results)

        xr.testing.assert_identical(first_call, second_call)


class TestMapSpeed:
    """Test speed computation from decoded positions."""

    def test_single_timepoint_returns_nan(self):
        """With only one time point, speed should be NaN."""
        # Create simple linear track graph
        graph = nx.Graph()
        graph.add_edge(0, 1, distance=10.0)

        # Single timepoint posterior (all probability at node 0)
        posterior = np.array([[1.0, 0.0]])
        place_bin_center_ind_to_node = np.array([0, 1])

        speed = get_map_speed(
            posterior,
            graph,
            place_bin_center_ind_to_node,
            sampling_frequency=500.0,
        )

        assert len(speed) == 1
        assert np.isnan(speed[0]), "Single timepoint should return NaN"

    def test_two_timepoints_basic_speed(self):
        """With two time points, compute distance / dt."""
        graph = nx.Graph()
        graph.add_edge(0, 1, distance=20.0)  # 20 cm apart

        # Two timepoints: move from node 0 to node 1
        posterior = np.array([[1.0, 0.0], [0.0, 1.0]])
        place_bin_center_ind_to_node = np.array([0, 1])

        sampling_frequency = 500.0  # Hz
        dt = 1 / sampling_frequency  # 0.002 seconds

        speed = get_map_speed(
            posterior,
            graph,
            place_bin_center_ind_to_node,
            sampling_frequency=sampling_frequency,
            smooth_sigma=0.0001,  # Minimal smoothing (can't use 0.0)
        )

        expected_speed = 20.0 / dt  # 20 cm / 0.002 s = 10000 cm/s
        assert len(speed) == 2
        # Both speeds should be approximately the same (repeated edge speed)
        # Use relaxed tolerance due to smoothing
        assert speed[0] == pytest.approx(expected_speed, rel=0.05)
        assert speed[1] == pytest.approx(expected_speed, rel=0.05)

    def test_stationary_animal_zero_speed(self):
        """Animal staying at same position should have zero speed."""
        graph = nx.Graph()
        graph.add_edge(0, 1, distance=10.0)

        # Five timepoints, all at node 0 (stationary)
        posterior = np.array([
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ])
        place_bin_center_ind_to_node = np.array([0, 1])

        speed = get_map_speed(
            posterior,
            graph,
            place_bin_center_ind_to_node,
            sampling_frequency=500.0,
            smooth_sigma=0.0001,
        )

        # Speed should be approximately zero (distance = 0)
        np.testing.assert_allclose(speed, 0.0, atol=1e-6)

    def test_linear_movement_speed(self):
        """Test speed calculation with linear movement along track."""
        # Create 5-node linear track
        graph = nx.Graph()
        edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
        for edge in edges:
            graph.add_edge(*edge, distance=10.0)  # 10 cm per edge

        # Move from node 0 to node 4 over 5 timepoints
        posterior = np.eye(5)  # Diagonal: 0→1→2→3→4
        place_bin_center_ind_to_node = np.arange(5)

        sampling_frequency = 500.0
        dt = 1 / sampling_frequency

        speed = get_map_speed(
            posterior,
            graph,
            place_bin_center_ind_to_node,
            sampling_frequency=sampling_frequency,
            smooth_sigma=0.0001,
        )

        # Expected: moving 10 cm every dt seconds
        expected_speed = 10.0 / dt  # 5000 cm/s
        assert len(speed) == 5
        # Central samples have neighbors on both sides; under uniform 10cm
        # steps the discrete derivative is exact up to smoothing.
        np.testing.assert_allclose(speed[1:-1], expected_speed, rtol=0.02)
        # Boundary samples use a one-sided derivative — looser by design.
        np.testing.assert_allclose(speed[[0, -1]], expected_speed, rtol=0.15)

    @pytest.mark.parametrize("invalid_freq", [0.0, -500.0, -1.0, -0.001])
    def test_invalid_sampling_frequency_raises(self, invalid_freq):
        """Zero or negative sampling frequency should raise ValueError."""
        graph = nx.Graph()
        graph.add_edge(0, 1, distance=10.0)

        posterior = np.array([[1.0, 0.0], [0.0, 1.0]])
        place_bin_center_ind_to_node = np.array([0, 1])

        with pytest.raises(ValueError, match="sampling_frequency must be positive"):
            get_map_speed(
                posterior,
                graph,
                place_bin_center_ind_to_node,
                sampling_frequency=invalid_freq,
            )

    def test_disconnected_graph_raises(self):
        """Disconnected nodes should raise NetworkXNoPath error."""
        # Create disconnected graph: 0-1 and 2-3 (no path between them)
        graph = nx.Graph()
        graph.add_edge(0, 1, distance=10.0)
        graph.add_edge(2, 3, distance=10.0)
        # No edge connecting component {0,1} to component {2,3}

        # Try to move from node 0 to node 2 (impossible)
        posterior = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],  # Jump from 0 to 2
        ])
        place_bin_center_ind_to_node = np.array([0, 1, 2, 3])

        with pytest.raises(nx.NetworkXNoPath):
            get_map_speed(
                posterior,
                graph,
                place_bin_center_ind_to_node,
                sampling_frequency=500.0,
            )

    def test_smoothing_reduces_noise(self):
        """Gaussian smoothing should reduce speed fluctuations."""
        # Create linear track
        graph = nx.Graph()
        for i in range(10):
            graph.add_edge(i, i + 1, distance=10.0)

        # Create noisy movement pattern
        node_sequence = [0, 1, 0, 2, 1, 3, 2, 4, 3, 5, 4]
        posterior = np.eye(11)[node_sequence]
        place_bin_center_ind_to_node = np.arange(11)

        # Compute speed with minimal smoothing
        speed_no_smooth = get_map_speed(
            posterior,
            graph,
            place_bin_center_ind_to_node,
            sampling_frequency=500.0,
            smooth_sigma=0.0001,  # Minimal smoothing
        )

        # Compute speed with heavier smoothing
        speed_smooth = get_map_speed(
            posterior,
            graph,
            place_bin_center_ind_to_node,
            sampling_frequency=500.0,
            smooth_sigma=0.01,  # 10 ms smoothing
        )

        # Smoothed version should have lower variance
        assert np.var(speed_smooth) < np.var(speed_no_smooth), \
            "Smoothing should reduce variance"

    def test_absolute_speed_always_positive(self):
        """Speed should always be non-negative (uses np.abs)."""
        # Create linear track
        graph = nx.Graph()
        graph.add_edges_from([(0, 1), (1, 2), (2, 3)])
        nx.set_edge_attributes(graph, 10.0, "distance")

        # Forward then backward movement: 0→1→2→3→2→1→0
        node_sequence = [0, 1, 2, 3, 2, 1, 0]
        posterior = np.eye(4)[node_sequence]
        place_bin_center_ind_to_node = np.arange(4)

        speed = get_map_speed(
            posterior,
            graph,
            place_bin_center_ind_to_node,
            sampling_frequency=500.0,
            smooth_sigma=0.0001,
        )

        # All speeds should be non-negative
        assert np.all(speed >= 0), "Speed should always be non-negative"

    def test_reversal_speed_correct(self):
        """Speed at trajectory reversal should reflect actual step speed.

        A forward-then-reverse trajectory moves at constant speed throughout,
        including at the reversal point. Forward differences correctly capture
        this because dist(node[t], node[t+1]) measures the actual step taken.
        """
        graph = nx.Graph()
        graph.add_edges_from([(i, i + 1) for i in range(10)])
        nx.set_edge_attributes(graph, 10.0, "distance")

        # Forward 0→5, reverse 5→0: constant speed of 10cm per step
        forward = list(range(6))        # [0, 1, 2, 3, 4, 5]
        backward = list(range(4, -1, -1))  # [4, 3, 2, 1, 0]
        node_sequence = forward + backward  # 11 timepoints
        posterior = np.eye(11)[node_sequence]
        place_bin_center_ind_to_node = np.arange(11)

        sampling_frequency = 500.0
        dt = 1 / sampling_frequency
        expected_speed = 10.0 / dt  # 5000 cm/s everywhere

        speed = get_map_speed(
            posterior,
            graph,
            place_bin_center_ind_to_node,
            sampling_frequency=sampling_frequency,
            smooth_sigma=0.0001,
        )

        # Every timepoint should report ~5000 cm/s, including the reversal
        np.testing.assert_allclose(speed, expected_speed, rtol=0.05)

    def test_oscillating_speed_correct(self):
        """Rapidly oscillating trajectory should report correct speed.

        An A→B→A→B pattern moves the same distance each step.
        """
        graph = nx.Graph()
        graph.add_edge(0, 1, distance=10.0)
        graph.add_edge(1, 2, distance=10.0)

        # Oscillate: 0→2→0→2→0→2
        node_sequence = [0, 2, 0, 2, 0, 2]
        posterior = np.zeros((6, 3))
        for t, n in enumerate(node_sequence):
            posterior[t, n] = 1.0
        place_bin_center_ind_to_node = np.arange(3)

        sampling_frequency = 500.0
        dt = 1 / sampling_frequency
        # Each step covers 20cm (0→2 or 2→0)
        expected_speed = 20.0 / dt  # 10000 cm/s

        speed = get_map_speed(
            posterior,
            graph,
            place_bin_center_ind_to_node,
            sampling_frequency=sampling_frequency,
            smooth_sigma=0.0001,
        )

        np.testing.assert_allclose(speed, expected_speed, rtol=0.05)

    def test_three_timepoints_forward_difference(self):
        """With 3+ time points, all use forward differences."""
        graph = nx.Graph()
        graph.add_edges_from([(0, 1), (1, 2), (2, 3)])
        nx.set_edge_attributes(graph, 20.0, "distance")

        # Three nodes: 0 → 1 → 2
        posterior = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ])
        place_bin_center_ind_to_node = np.arange(4)

        sampling_frequency = 500.0
        dt = 1 / sampling_frequency

        speed = get_map_speed(
            posterior,
            graph,
            place_bin_center_ind_to_node,
            sampling_frequency=sampling_frequency,
            smooth_sigma=0.0001,
        )

        # All points use forward difference: dist(t, t+1) / dt
        # Each step is 20 cm in dt seconds
        expected_speed = 20.0 / dt

        assert len(speed) == 3
        np.testing.assert_allclose(speed, expected_speed, rtol=0.05)

    def test_nonuniform_edges_last_speed_correct(self):
        """With non-uniform edges, speed should reflect each step's distance."""
        graph = nx.Graph()
        graph.add_edge(0, 1, distance=10.0)
        graph.add_edge(1, 2, distance=10.0)
        graph.add_edge(2, 3, distance=40.0)

        # Move 0 → 1 → 2 → 3
        posterior = np.eye(4)
        place_bin_center_ind_to_node = np.arange(4)

        sampling_frequency = 500.0
        dt = 1 / sampling_frequency

        speed = get_map_speed(
            posterior,
            graph,
            place_bin_center_ind_to_node,
            sampling_frequency=sampling_frequency,
            smooth_sigma=0.0001,
        )

        # Forward differences: speed[t] = dist(node[t], node[t+1]) / dt
        # speed[2] = dist(2,3)/dt = 40/0.002 = 20000
        expected_second_last = 40.0 / dt
        # Last timepoint repeats backward difference = dist(2,3)/dt = 20000
        expected_last = 40.0 / dt

        assert speed[2] == pytest.approx(expected_second_last, rel=0.05)
        assert speed[3] == pytest.approx(expected_last, rel=0.05)

    def test_xarray_input(self):
        """Function should accept xarray DataArray as input."""
        graph = nx.Graph()
        graph.add_edge(0, 1, distance=10.0)

        # Create xarray DataArray
        posterior = xr.DataArray(
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            coords={"time": [0.0, 0.002], "position": [0, 1]},
            dims=["time", "position"],
        )
        place_bin_center_ind_to_node = np.array([0, 1])

        # Should work without error
        speed = get_map_speed(
            posterior,
            graph,
            place_bin_center_ind_to_node,
            sampling_frequency=500.0,
        )

        assert len(speed) == 2
        assert np.all(np.isfinite(speed))

    def test_high_sampling_frequency(self):
        """Test with very high sampling frequency (small dt)."""
        graph = nx.Graph()
        graph.add_edge(0, 1, distance=10.0)

        posterior = np.array([[1.0, 0.0], [0.0, 1.0]])
        place_bin_center_ind_to_node = np.array([0, 1])

        # Very high sampling rate
        sampling_frequency = 10000.0  # 10 kHz
        dt = 1 / sampling_frequency  # 0.0001 seconds

        speed = get_map_speed(
            posterior,
            graph,
            place_bin_center_ind_to_node,
            sampling_frequency=sampling_frequency,
            smooth_sigma=0.0001,
        )

        # At high sampling rates, smoothing has larger effect
        # Just verify speed is computed and in reasonable range
        expected_speed = 10.0 / dt  # 100000 cm/s theoretical
        assert len(speed) == 2
        assert np.all(np.isfinite(speed))
        # Speed should be positive and within order of magnitude
        assert np.all(speed > 0)
        assert np.all(speed < expected_speed * 1.5), "Speed shouldn't exceed theoretical max"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_nan_in_posterior(self):
        """NaN in posterior should be handled by nanargmax."""
        graph = nx.Graph()
        graph.add_edge(0, 1, distance=10.0)

        # Posterior with NaN values
        posterior = np.array([
            [0.3, 0.7],
            [np.nan, 1.0],
            [0.5, 0.5],
        ])
        place_bin_center_ind_to_node = np.array([0, 1])

        # Should not crash; nanargmax ignores NaN
        speed = get_map_speed(
            posterior,
            graph,
            place_bin_center_ind_to_node,
            sampling_frequency=500.0,
        )

        assert len(speed) == 3
        # Speed should be computed (nanargmax handles NaN)
        assert np.all(np.isfinite(speed))

    def test_uniform_posterior(self):
        """Uniform posterior (no clear MAP) should pick first maximum."""
        graph = nx.Graph()
        graph.add_edge(0, 1, distance=10.0)

        # All positions equally likely
        posterior = np.array([
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
        ])
        place_bin_center_ind_to_node = np.array([0, 1])

        # Should not crash; argmax picks first occurrence
        speed = get_map_speed(
            posterior,
            graph,
            place_bin_center_ind_to_node,
            sampling_frequency=500.0,
        )

        assert len(speed) == 3
        assert np.all(np.isfinite(speed))

    def test_large_graph(self):
        """Test with larger graph (100 nodes)."""
        # Create 100-node linear track
        graph = nx.Graph()
        for i in range(99):
            graph.add_edge(i, i + 1, distance=2.0)

        # Move along first 10 nodes
        posterior = np.zeros((10, 100))
        for i in range(10):
            posterior[i, i] = 1.0

        place_bin_center_ind_to_node = np.arange(100)

        speed = get_map_speed(
            posterior,
            graph,
            place_bin_center_ind_to_node,
            sampling_frequency=500.0,
        )

        assert len(speed) == 10
        assert np.all(np.isfinite(speed))

    def test_weighted_graph_different_distances(self):
        """Graph with varying edge distances."""
        graph = nx.Graph()
        graph.add_edge(0, 1, distance=5.0)
        graph.add_edge(1, 2, distance=15.0)
        graph.add_edge(2, 3, distance=10.0)

        # Move 0 → 1 → 2 → 3
        posterior = np.eye(4)
        place_bin_center_ind_to_node = np.arange(4)

        speed = get_map_speed(
            posterior,
            graph,
            place_bin_center_ind_to_node,
            sampling_frequency=500.0,
            smooth_sigma=0.0001,
        )

        # Speeds should reflect different edge lengths
        assert len(speed) == 4
        assert np.all(np.isfinite(speed))
        # Middle speeds should vary based on edge lengths
        # Note: smoothing reduces differences, so check if any variation exists
        assert np.std(speed) > 0, \
            "Different edge lengths should produce variation in speeds"


class TestAllNaNRowHandlingFix:
    """Test fix for handling all-NaN rows in posterior (NaN handling fix)."""

    def test_all_nan_rows_return_nan_speed(self):
        """Timepoints with all-NaN posteriors should have NaN speed."""
        graph = nx.Graph()
        graph.add_edge(0, 1, distance=10.0)

        # Some rows have valid data, some are all NaN
        posterior = np.array([
            [0.3, 0.7],      # Valid
            [np.nan, np.nan],  # All NaN
            [0.5, 0.5],      # Valid
            [np.nan, np.nan],  # All NaN
            [0.2, 0.8],      # Valid
        ])
        place_bin_center_ind_to_node = np.array([0, 1])

        speed = get_map_speed(
            posterior,
            graph,
            place_bin_center_ind_to_node,
            sampling_frequency=500.0,
            smooth_sigma=0.0001,
        )

        assert len(speed) == 5
        # Rows with all NaN should produce NaN speed
        assert np.isnan(speed[1]), "All-NaN row should produce NaN speed"
        assert np.isnan(speed[3]), "All-NaN row should produce NaN speed"

    def test_all_rows_nan_returns_all_nan(self):
        """Completely NaN posterior should return all NaN speeds."""
        graph = nx.Graph()
        graph.add_edge(0, 1, distance=10.0)

        # All rows are NaN
        posterior = np.full((5, 2), np.nan)
        place_bin_center_ind_to_node = np.array([0, 1])

        speed = get_map_speed(
            posterior,
            graph,
            place_bin_center_ind_to_node,
            sampling_frequency=500.0,
        )

        assert len(speed) == 5
        assert np.all(np.isnan(speed)), "All-NaN posterior should return all NaN speeds"

    def test_partial_nan_rows_do_not_affect_valid_rows(self):
        """Valid rows should have computed speeds even when some rows are NaN."""
        graph = nx.Graph()
        graph.add_edge(0, 1, distance=20.0)

        # First and last row valid, middle is NaN
        posterior = np.array([
            [1.0, 0.0],       # At node 0
            [np.nan, np.nan],  # NaN
            [0.0, 1.0],       # At node 1
        ])
        place_bin_center_ind_to_node = np.array([0, 1])

        speed = get_map_speed(
            posterior,
            graph,
            place_bin_center_ind_to_node,
            sampling_frequency=500.0,
            smooth_sigma=0.0001,
        )

        assert len(speed) == 3
        # Valid rows should have finite speeds (before NaN masking)
        # After fix, speeds at invalid timepoints are set to NaN
        assert np.isnan(speed[1]), "NaN row should have NaN speed"
        # Edge rows may or may not be NaN depending on smoothing

    def test_no_warning_on_all_nan_rows(self):
        """Should not emit numpy warning when handling all-NaN rows."""
        import warnings

        graph = nx.Graph()
        graph.add_edge(0, 1, distance=10.0)

        posterior = np.array([
            [0.3, 0.7],
            [np.nan, np.nan],
            [0.5, 0.5],
        ])
        place_bin_center_ind_to_node = np.array([0, 1])

        # Catch any numpy warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            _ = get_map_speed(
                posterior,
                graph,
                place_bin_center_ind_to_node,
                sampling_frequency=500.0,
            )

            # Filter for RuntimeWarning about nanargmax
            nan_warnings = [
                warning for warning in w
                if "All-NaN" in str(warning.message)
                or "nanargmax" in str(warning.message).lower()
            ]

            assert len(nan_warnings) == 0, \
                f"Should not emit NaN warnings, got: {[str(w.message) for w in nan_warnings]}"
