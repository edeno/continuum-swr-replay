"""Property-based tests for mathematical invariants.

Uses the hypothesis library to generate random test data and verify that
mathematical properties hold for ALL valid inputs, not just specific examples.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from scipy.stats import zscore

from continuum_swr_replay.data_loaders.task_variables import (
    path_progress_for_run_segment,
)


# Custom strategies for scientific data
@st.composite
def positive_floats(draw, min_value=0.001, max_value=1000.0):
    """Generate positive floating point numbers."""
    return draw(st.floats(min_value=min_value, max_value=max_value))


@st.composite
def time_series_data(draw, min_length=2, max_length=1000):
    """Generate valid time series data."""
    n_points = draw(st.integers(min_value=min_length, max_value=max_length))
    # Generate monotonically increasing time
    time_diffs = draw(arrays(
        dtype=np.float64,
        shape=n_points - 1,
        elements=st.floats(min_value=0.001, max_value=1.0)
    ))
    time = np.concatenate([[0.0], np.cumsum(time_diffs)])

    # Generate data values
    values = draw(arrays(
        dtype=np.float64,
        shape=n_points,
        elements=st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False)
    ))

    return time, values


class TestNormalizationInvariants:
    """Test that normalization operations always produce valid outputs."""

    @given(
        data=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=10, max_value=100),
            elements=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_zscore_normalization_mean_zero(self, data):
        """Z-score normalization should produce mean ≈ 0 (for non-constant data)."""
        # Skip constant data (std = 0)
        if np.std(data) < 1e-10:
            return

        normalized = zscore(data, nan_policy='omit')

        # Mean should be very close to 0
        assert abs(np.mean(normalized)) < 1e-10, \
            f"Z-score mean should be ~0, got {np.mean(normalized)}"

    @given(
        data=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=10, max_value=100),
            elements=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_zscore_normalization_std_one(self, data):
        """Z-score normalization should produce std ≈ 1 (for non-constant data)."""
        # Skip constant data (std = 0)
        if np.std(data) < 1e-10:
            return

        normalized = zscore(data, nan_policy='omit')

        # Standard deviation should be very close to 1
        assert abs(np.std(normalized) - 1.0) < 1e-10, \
            f"Z-score std should be ~1, got {np.std(normalized)}"

    @given(
        values=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=2, max_value=100),
            elements=st.floats(min_value=0.001, max_value=1000.0, allow_nan=False, allow_infinity=False),
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_probability_normalization_sums_to_one(self, values):
        """Probability normalization should always sum to 1."""
        probabilities = values / values.sum()

        assert probabilities.sum() == pytest.approx(1.0, rel=1e-10), \
            f"Probabilities should sum to 1, got {probabilities.sum()}"

    @given(
        values=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=2, max_value=100),
            elements=st.floats(min_value=0.001, max_value=1000.0, allow_nan=False, allow_infinity=False),
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_probability_normalization_all_non_negative(self, values):
        """Normalized probabilities should all be non-negative."""
        probabilities = values / values.sum()

        assert np.all(probabilities >= 0.0), \
            "All probabilities should be non-negative"

    @given(
        values=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=2, max_value=100),
            elements=st.floats(min_value=0.001, max_value=1000.0, allow_nan=False, allow_infinity=False),
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_probability_normalization_bounds(self, values):
        """Each normalized probability should be in [0, 1]."""
        probabilities = values / values.sum()

        assert np.all(probabilities >= 0.0), "Probabilities should be >= 0"
        assert np.all(probabilities <= 1.0), "Probabilities should be <= 1"


class TestPathProgressInvariants:
    """Test that path progress always satisfies [0, 1] bounds."""

    @given(
        n_points=st.integers(min_value=5, max_value=200),
        start_distance=st.floats(min_value=0.0, max_value=50.0),
        end_distance=st.floats(min_value=0.1, max_value=100.0),
    )
    @settings(max_examples=50, deadline=None)
    def test_path_progress_always_in_zero_one(self, n_points, start_distance, end_distance):
        """Path progress must always be in [0, 1] range."""
        # Create simple linear track
        graph = nx.Graph()
        graph.add_edge(0, 1)
        graph.nodes[0]['pos'] = (0.0, 0.0)
        graph.nodes[1]['pos'] = (end_distance, 0.0)
        graph.edges[(0, 1)]['edge_id'] = 0
        graph.edges[(0, 1)]['distance'] = end_distance

        # Create position data
        time = np.linspace(0, 1.0, n_points)
        linear_position = np.linspace(start_distance, end_distance, n_points)

        pos_info = pd.DataFrame({
            'projected_x_position': linear_position,
            'projected_y_position': np.zeros(n_points),
            'track_segment_id': np.zeros(n_points, dtype=int),
        }, index=time)

        progress = path_progress_for_run_segment(pos_info, graph, start_node=0, end_node=1)

        # Must be in [0, 1]
        assert np.all(progress >= 0.0), f"Progress must be >= 0, got min={progress.min()}"
        assert np.all(progress <= 1.0), f"Progress must be <= 1, got max={progress.max()}"

    @given(
        n_points=st.integers(min_value=5, max_value=200),
    )
    @settings(max_examples=50, deadline=None)
    def test_path_progress_monotonic(self, n_points):
        """Path progress should be monotonically increasing (or constant)."""
        # Create simple linear track
        graph = nx.Graph()
        graph.add_edge(0, 1)
        graph.nodes[0]['pos'] = (0.0, 0.0)
        graph.nodes[1]['pos'] = (20.0, 0.0)
        graph.edges[(0, 1)]['edge_id'] = 0
        graph.edges[(0, 1)]['distance'] = 20.0

        # Create position data with monotonically increasing position
        time = np.linspace(0, 1.0, n_points)
        linear_position = np.linspace(0, 20, n_points)

        pos_info = pd.DataFrame({
            'projected_x_position': linear_position,
            'projected_y_position': np.zeros(n_points),
            'track_segment_id': np.zeros(n_points, dtype=int),
        }, index=time)

        progress = path_progress_for_run_segment(pos_info, graph, start_node=0, end_node=1)

        # Progress should be monotonic (allowing for small numerical errors)
        diffs = np.diff(progress)
        assert np.all(diffs >= -1e-10), \
            f"Progress should be monotonic, found negative diff: {diffs.min()}"


class TestGraphDistanceInvariants:
    """Test that graph distances satisfy mathematical properties."""

    @given(
        n_nodes=st.integers(min_value=3, max_value=10),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=30, deadline=None)
    def test_triangle_inequality(self, n_nodes, seed):
        """Graph distances must satisfy triangle inequality: d(a,c) <= d(a,b) + d(b,c)."""
        rng = np.random.RandomState(seed)

        # Create random connected graph
        graph = nx.complete_graph(n_nodes)

        # Assign random positive weights
        for edge in graph.edges():
            graph.edges[edge]['weight'] = rng.uniform(1.0, 10.0)

        # Test triangle inequality for random triplets
        for _ in range(min(10, n_nodes)):
            nodes = rng.choice(n_nodes, size=3, replace=False)
            a, b, c = nodes

            d_ac = nx.shortest_path_length(graph, a, c, weight='weight')
            d_ab = nx.shortest_path_length(graph, a, b, weight='weight')
            d_bc = nx.shortest_path_length(graph, b, c, weight='weight')

            assert d_ac <= d_ab + d_bc + 1e-10, \
                f"Triangle inequality violated: d({a},{c})={d_ac} > d({a},{b})+d({b},{c})={d_ab + d_bc}"

    @given(
        n_nodes=st.integers(min_value=2, max_value=10),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=30, deadline=None)
    def test_shortest_path_non_negative(self, n_nodes, seed):
        """Shortest path lengths must always be non-negative."""
        rng = np.random.RandomState(seed)

        # Create random connected graph
        graph = nx.complete_graph(n_nodes)

        # Assign random positive weights
        for edge in graph.edges():
            graph.edges[edge]['weight'] = rng.uniform(0.1, 10.0)

        # Check all pairs
        for source in range(n_nodes):
            for target in range(n_nodes):
                dist = nx.shortest_path_length(graph, source, target, weight='weight')
                assert dist >= 0.0, f"Distance must be non-negative, got {dist}"


class TestSortingInvariants:
    """Test that sorting operations preserve expected properties."""

    @given(
        data=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=2, max_value=100),
            elements=st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_sorted_array_monotonic(self, data):
        """Sorted arrays must be monotonically increasing."""
        sorted_data = np.sort(data)

        # Check monotonicity
        diffs = np.diff(sorted_data)
        assert np.all(diffs >= 0), \
            f"Sorted array should be monotonic, found negative diff: {diffs.min()}"

    @given(
        data=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=2, max_value=100),
            elements=st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_sorted_array_same_elements(self, data):
        """Sorting preserves all elements (just reorders)."""
        sorted_data = np.sort(data)

        # Should have same length
        assert len(sorted_data) == len(data)

        # Should have same elements (when sorted)
        np.testing.assert_array_equal(np.sort(data), np.sort(sorted_data))


class TestTemporalInvariants:
    """Test invariants for time-indexed data."""

    @given(time_data=time_series_data())
    @settings(max_examples=50, deadline=None)
    def test_time_always_monotonic(self, time_data):
        """Time indices must always be monotonically increasing."""
        time, _ = time_data

        # Check monotonicity
        diffs = np.diff(time)
        assert np.all(diffs > 0), \
            f"Time should be strictly increasing, found non-positive diff: {diffs.min()}"

    @given(time_data=time_series_data())
    @settings(max_examples=50, deadline=None)
    def test_time_unique(self, time_data):
        """Time indices should have no duplicates."""
        time, _ = time_data

        # Check uniqueness
        assert len(time) == len(np.unique(time)), \
            "Time indices should be unique (no duplicates)"

    @given(
        speed_values=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=2, max_value=100),
            elements=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_speed_always_non_negative(self, speed_values):
        """Speed must always be non-negative."""
        assert np.all(speed_values >= 0.0), \
            f"Speed must be non-negative, got min={speed_values.min()}"


class TestNumericStabilityInvariants:
    """Test numeric stability properties."""

    @given(
        values=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=2, max_value=100),
            elements=st.floats(min_value=1e-10, max_value=1e10, allow_nan=False, allow_infinity=False),
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_sum_order_independence(self, values):
        """Sum should be (approximately) independent of order."""
        forward_sum = np.sum(values)
        backward_sum = np.sum(values[::-1])

        # Should be very close (within floating point precision)
        assert forward_sum == pytest.approx(backward_sum, rel=1e-10), \
            f"Sum should be order-independent, got {forward_sum} vs {backward_sum}"

    @given(
        n_points=st.integers(min_value=2, max_value=100),
        start=st.floats(min_value=0.0, max_value=100.0),
        end=st.floats(min_value=0.1, max_value=200.0),
    )
    @settings(max_examples=100, deadline=None)
    def test_linspace_bounds(self, n_points, start, end):
        """np.linspace should always respect specified bounds."""
        # Skip if start >= end
        if start >= end:
            return

        values = np.linspace(start, end, n_points)

        assert values[0] == pytest.approx(start, rel=1e-10), \
            f"First value should be {start}, got {values[0]}"
        assert values[-1] == pytest.approx(end, rel=1e-10), \
            f"Last value should be {end}, got {values[-1]}"
        assert np.all(values >= start), \
            f"All values should be >= {start}"
        assert np.all(values <= end), \
            f"All values should be <= {end}"
