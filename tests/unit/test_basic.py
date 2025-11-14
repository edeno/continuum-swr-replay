"""Basic sanity tests that don't require heavy dependencies.

These tests verify basic Python/NumPy/Pandas operations work correctly.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest


class TestNumpyOperations:
    """Test basic NumPy operations used throughout the codebase."""

    def test_array_concatenation(self):
        """Test spike time concatenation logic."""
        spike_times = [
            np.array([1.0, 2.0, 3.0]),
            np.array([1.5, 2.5, 3.5]),
            np.array([2.0, 3.0, 4.0]),
        ]

        concatenated = np.concatenate(spike_times)
        assert len(concatenated) == 9
        assert concatenated.min() >= 1.0
        assert concatenated.max() <= 4.0

    def test_temporal_filtering(self):
        """Test temporal filtering logic."""
        spike_times = np.array([0.5, 1.0, 2.0, 3.0, 5.5])
        position_time_min, position_time_max = 1.0, 5.0

        is_in_bounds = np.logical_and(
            spike_times >= position_time_min,
            spike_times <= position_time_max,
        )
        filtered = spike_times[is_in_bounds]

        assert len(filtered) == 3  # 1.0, 2.0, 3.0
        assert filtered.min() >= position_time_min
        assert filtered.max() <= position_time_max

    def test_division_by_zero_protection(self):
        """Test division by zero handling in z-score calculation."""
        # Constant signal (std = 0)
        data = np.array([5.0, 5.0, 5.0, 5.0])

        std = data.std()
        if std == 0 or np.isnan(std):
            result = np.zeros_like(data)
        else:
            result = (data - data.mean()) / std

        # Should return zeros, not NaN
        np.testing.assert_array_equal(result, np.zeros(4))

    def test_spike_time_sorting(self):
        """Test that spike times remain sorted."""
        unsorted = np.array([3.0, 1.0, 2.0, 5.0, 4.0])
        sorted_spikes = np.sort(unsorted)

        assert np.all(np.diff(sorted_spikes) > 0)
        np.testing.assert_array_equal(sorted_spikes, np.array([1.0, 2.0, 3.0, 4.0, 5.0]))


class TestPandasOperations:
    """Test basic Pandas operations."""

    def test_time_indexed_dataframe(self):
        """Test time-indexed DataFrame operations."""
        time = np.arange(0.0, 10.0, 0.002)  # 500 Hz
        linear_position = np.linspace(0, 100, len(time))

        df = pd.DataFrame({
            'linear_position': linear_position,
            'speed': np.full(len(time), 10.0),
        }, index=time)

        assert df.index.is_monotonic_increasing
        assert df.index.min() == 0.0
        assert df.index.max() < 10.0

    def test_temporal_slicing(self):
        """Test temporal slicing with .loc."""
        time = np.arange(0.0, 10.0, 0.1)
        df = pd.DataFrame({'value': np.arange(len(time))}, index=time)

        # Slice from 2.0 to 5.0 seconds
        sliced = df.loc[2.0:5.0]

        assert sliced.index.min() >= 2.0
        assert sliced.index.max() <= 5.0
        assert len(sliced) > 0

    def test_event_duration_calculation(self):
        """Test event duration calculation."""
        events = pd.DataFrame({
            'start_time': [1.0, 3.5, 7.2],
            'end_time': [1.08, 3.65, 7.35],
        })

        events['duration'] = events['end_time'] - events['start_time']

        assert (events['duration'] > 0).all()
        assert events['duration'].iloc[0] == pytest.approx(0.08)


class TestNetworkxOperations:
    """Test NetworkX graph operations."""

    def test_simple_shortest_path(self):
        """Test shortest path calculation."""
        graph = nx.Graph()
        graph.add_edges_from([(0, 1), (1, 2), (2, 3)])

        path = nx.shortest_path(graph, source=0, target=3)

        assert path == [0, 1, 2, 3]
        assert len(path) == 4

    def test_shortest_path_length_with_weights(self):
        """Test weighted shortest path."""
        graph = nx.Graph()
        graph.add_edges_from([(0, 1), (1, 2), (0, 2)])

        # Assign weights
        nx.set_edge_attributes(graph, {(0, 1): 10.0, (1, 2): 10.0, (0, 2): 5.0}, 'distance')

        # Shortest path should use direct edge (0, 2)
        length = nx.shortest_path_length(graph, source=0, target=2, weight='distance')

        assert length == 5.0  # Direct path is shorter

    def test_graph_connectivity(self):
        """Test graph connectivity check."""
        # Connected graph
        graph1 = nx.Graph()
        graph1.add_edges_from([(0, 1), (1, 2)])
        assert nx.is_connected(graph1)

        # Disconnected graph
        graph2 = nx.Graph()
        graph2.add_edges_from([(0, 1), (2, 3)])
        assert not nx.is_connected(graph2)


class TestMathematicalInvariants:
    """Test mathematical properties that should always hold."""

    def test_path_progress_bounds(self):
        """Path progress normalization should give [0, 1] range."""
        dist = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
        max_dist = dist.max()

        if max_dist > 0:
            progress = dist / max_dist
        else:
            progress = np.zeros_like(dist)

        assert np.all(progress >= 0.0)
        assert np.all(progress <= 1.0)
        assert progress[0] == 0.0
        assert progress[-1] == 1.0

    def test_triangle_inequality(self):
        """Graph distances should satisfy triangle inequality."""
        graph = nx.complete_graph(4)

        # Assign random weights
        rng = np.random.RandomState(42)
        for edge in graph.edges():
            graph.edges[edge]['weight'] = rng.uniform(1, 10)

        # Test: dist(0,2) <= dist(0,1) + dist(1,2)
        d_02 = nx.shortest_path_length(graph, 0, 2, weight='weight')
        d_01 = nx.shortest_path_length(graph, 0, 1, weight='weight')
        d_12 = nx.shortest_path_length(graph, 1, 2, weight='weight')

        assert d_02 <= d_01 + d_12 + 1e-10

    def test_probability_normalization(self):
        """Probabilities must sum to 1."""
        # Create unnormalized probabilities
        values = np.array([1.5, 2.3, 0.8, 1.2])
        probabilities = values / values.sum()

        assert probabilities.sum() == pytest.approx(1.0)
        assert np.all(probabilities >= 0)


class TestDataIntegrity:
    """Test data integrity checks."""

    def test_no_nan_in_critical_data(self):
        """Critical columns should not have NaN."""
        position_df = pd.DataFrame({
            'linear_position': [10.0, 20.0, 30.0],
            'speed': [5.0, 10.0, 5.0],
            'extra': [1.0, np.nan, 3.0],  # OK to have NaN in non-critical
        })

        critical_columns = ['linear_position', 'speed']

        for col in critical_columns:
            assert not position_df[col].isna().any(), f"NaN found in {col}"

    def test_time_monotonicity(self):
        """Time index must be monotonically increasing."""
        time = np.arange(0, 10, 0.01)
        assert np.all(np.diff(time) > 0)

        # Check that it's strictly increasing
        df = pd.DataFrame({'value': np.arange(len(time))}, index=time)
        assert df.index.is_monotonic_increasing
        assert df.index.is_unique

    def test_speed_non_negative(self):
        """Speed must be non-negative."""
        speeds = np.array([0.0, 5.0, 10.0, 15.0, 0.0])

        assert np.all(speeds >= 0), "Negative speeds found"

    def test_duration_positive(self):
        """Event durations must be positive."""
        start_times = np.array([1.0, 3.5, 7.2])
        end_times = np.array([1.08, 3.65, 7.35])

        durations = end_times - start_times

        assert np.all(durations > 0), "Non-positive durations found"


class TestNaNInfHandling:
    """Test handling of NaN and Inf values in scientific computations."""

    def test_detect_nan_in_array(self):
        """NaN detection should work correctly."""
        data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])

        assert np.isnan(data).any(), "Should detect NaN"
        assert np.sum(np.isnan(data)) == 1, "Should find exactly one NaN"
        assert np.isnan(data[2]), "NaN should be at index 2"

    def test_detect_inf_in_array(self):
        """Inf detection should work correctly."""
        data = np.array([1.0, 2.0, np.inf, 4.0, -np.inf])

        assert np.isinf(data).any(), "Should detect Inf"
        assert np.sum(np.isinf(data)) == 2, "Should find two Inf values"
        assert np.isposinf(data[2]), "Should detect +Inf"
        assert np.isneginf(data[4]), "Should detect -Inf"

    def test_nan_propagation_in_operations(self):
        """NaN should propagate through arithmetic operations."""
        a = np.array([1.0, np.nan, 3.0])
        b = np.array([2.0, 2.0, 2.0])

        result = a + b
        assert np.isnan(result[1]), "NaN should propagate in addition"

        result = a * b
        assert np.isnan(result[1]), "NaN should propagate in multiplication"

        result = a / b
        assert np.isnan(result[1]), "NaN should propagate in division"

    @pytest.mark.parametrize(
        "operation,expected_check",
        [
            (lambda: np.inf + 5.0, np.isinf),
            (lambda: np.inf * 2.0, np.isinf),
            (lambda: np.inf * -2.0, np.isneginf),
            (lambda: np.inf / 2.0, np.isinf),
        ],
        ids=["inf_plus_finite", "inf_times_positive", "inf_times_negative", "inf_div_finite"]
    )
    def test_inf_arithmetic(self, operation, expected_check):
        """Inf arithmetic should follow IEEE 754 rules."""
        result = operation()
        assert expected_check(result), f"Operation {operation} failed IEEE 754 check"

    def test_finite_divided_by_inf_is_zero(self):
        """finite / Inf = 0 (IEEE 754 rule)."""
        assert 5.0 / np.inf == 0.0

    @pytest.mark.parametrize(
        "operation,error_state,description",
        [
            (lambda: np.array([0.0]) / np.array([0.0]), {'divide': 'ignore', 'invalid': 'ignore'}, "0/0"),
            (lambda: np.inf - np.inf, {'invalid': 'ignore'}, "Inf - Inf"),
            (lambda: np.inf / np.inf, {'invalid': 'ignore'}, "Inf / Inf"),
            (lambda: 0.0 * np.inf, {'invalid': 'ignore'}, "0 * Inf"),
        ],
        ids=["zero_div_zero", "inf_minus_inf", "inf_div_inf", "zero_times_inf"]
    )
    def test_nan_from_invalid_operations(self, operation, error_state, description):
        """Invalid operations should produce NaN."""
        with np.errstate(**error_state):
            result = operation()
            # Handle both scalar and array results
            if isinstance(result, np.ndarray):
                result = result[0]
            assert np.isnan(result), f"{description} should produce NaN"

    def test_nansum_ignores_nan(self):
        """np.nansum should ignore NaN values."""
        data = np.array([1.0, 2.0, np.nan, 4.0, np.nan])

        result = np.nansum(data)
        assert result == 7.0, "nansum should ignore NaN and sum to 7"

    def test_nanmean_ignores_nan(self):
        """np.nanmean should ignore NaN values."""
        data = np.array([1.0, 2.0, np.nan, 4.0])

        result = np.nanmean(data)
        assert result == pytest.approx(7.0 / 3.0), "nanmean should compute mean of non-NaN values"

    def test_isfinite_detects_both_nan_and_inf(self):
        """np.isfinite should detect both NaN and Inf."""
        data = np.array([1.0, np.nan, np.inf, 4.0, -np.inf])

        finite_mask = np.isfinite(data)
        assert finite_mask[0], "Regular number should be finite"
        assert not finite_mask[1], "NaN should not be finite"
        assert not finite_mask[2], "+Inf should not be finite"
        assert finite_mask[3], "Regular number should be finite"
        assert not finite_mask[4], "-Inf should not be finite"

    @pytest.mark.parametrize(
        "comparison,expected_result",
        [
            (lambda nan: nan == nan, False),
            (lambda nan: nan < 5.0, False),
            (lambda nan: nan > 5.0, False),
            (lambda nan: nan <= 5.0, False),
            (lambda nan: nan >= 5.0, False),
            (lambda nan: nan != nan, True),
        ],
        ids=["eq", "lt", "gt", "le", "ge", "ne"]
    )
    def test_nan_comparison_always_false(self, comparison, expected_result):
        """NaN comparisons should always return False (except !=)."""
        nan = np.nan
        assert comparison(nan) == expected_result

    def test_filtering_nan_and_inf(self):
        """Should be able to filter out NaN and Inf values."""
        data = np.array([1.0, np.nan, 3.0, np.inf, 5.0, -np.inf])

        # Filter to keep only finite values
        clean_data = data[np.isfinite(data)]

        assert len(clean_data) == 3, "Should have 3 finite values"
        np.testing.assert_array_equal(clean_data, np.array([1.0, 3.0, 5.0]))

    def test_replace_nan_with_value(self):
        """Should be able to replace NaN with a specific value."""
        data = np.array([1.0, np.nan, 3.0, np.nan, 5.0])

        # Replace NaN with 0
        cleaned = np.where(np.isnan(data), 0.0, data)

        expected = np.array([1.0, 0.0, 3.0, 0.0, 5.0])
        np.testing.assert_array_equal(cleaned, expected)

    def test_nan_inf_in_dataframe(self):
        """Pandas should handle NaN and Inf correctly."""
        df = pd.DataFrame({
            'a': [1.0, np.nan, 3.0],
            'b': [np.inf, 2.0, -np.inf],
            'c': [1.0, 2.0, 3.0],
        })

        # Check NaN detection
        assert df['a'].isna().sum() == 1, "Should detect 1 NaN in column 'a'"

        # Check Inf detection
        assert np.isinf(df['b']).sum() == 2, "Should detect 2 Inf in column 'b'"

        # Drop NaN rows
        clean_df = df.dropna()
        assert len(clean_df) == 2, "Should have 2 rows after dropping NaN"

    @pytest.mark.parametrize(
        "numerator,expected_check,description",
        [
            (1.0, np.isposinf, "+Inf"),
            (-1.0, np.isneginf, "-Inf"),
            (2.5, np.isposinf, "+Inf"),
            (-10.0, np.isneginf, "-Inf"),
        ],
        ids=["one_div_zero", "neg_one_div_zero", "positive_div_zero", "negative_div_zero"]
    )
    def test_division_by_zero_produces_inf(self, numerator, expected_check, description):
        """Division by zero should produce Inf (with warning suppression)."""
        # Use NumPy arrays to get IEEE 754 behavior instead of Python exception
        with np.errstate(divide='ignore'):
            result = np.array([numerator]) / np.array([0.0])
            assert expected_check(result[0]), f"{numerator}/0 should produce {description}"
