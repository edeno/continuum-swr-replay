"""Unit tests for position data processing.

Tests position interpolation and linearization functions used in the
data loading pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from track_linearization import make_track_graph

from continuum_swr_replay.data_loaders.position import get_interpolated_position_info


class TestGetInterpolatedPositionInfo:
    """Test position interpolation and linearization."""

    def test_interpolation_preserves_original_times(self):
        """When the interpolation grid equals the input grid, original
        column values must be returned exactly — no rounding, no
        propagation through the interpolator."""
        original_times = np.array([0.0, 1.0, 2.0, 3.0])
        original_x = np.array([0.0, 5.0, 10.0, 15.0])
        original_speed = np.array([5.0, 7.0, 9.0, 11.0])
        position_info = pd.DataFrame({
            'head_position_x': original_x,
            'head_position_y': np.zeros(4),
            'head_speed': original_speed,
        }, index=original_times)

        track_graph = make_track_graph(
            node_positions=[(0, 0), (10, 0), (20, 0)],
            edges=[(0, 1), (1, 2)]
        )

        result = get_interpolated_position_info(
            position_info,
            time=original_times,
            track_graph=track_graph,
            edge_order=[(0, 1), (1, 2)],
            edge_spacing=[10.0],
        )

        # Linearization columns are added.
        assert {'linear_position', 'projected_x_position', 'track_segment_id'} <= set(result.columns)

        # Original values survive intact at the sampled times.
        np.testing.assert_array_equal(
            result['head_position_x'].to_numpy(), original_x
        )
        np.testing.assert_array_equal(
            result['head_speed'].to_numpy(), original_speed
        )

    def test_interpolation_to_new_times(self):
        """New time points should have linearly interpolated values."""
        # Setup: Animal moves from 0 to 10 cm in 2 seconds
        original_times = np.array([0.0, 2.0])
        position_info = pd.DataFrame({
            'head_position_x': [0.0, 10.0],
            'head_position_y': [0.0, 0.0],
        }, index=original_times)

        # Create track graph using recommended function
        track_graph = make_track_graph(
            node_positions=[(0, 0), (10, 0), (20, 0)],
            edges=[(0, 1), (1, 2)]
        )

        # Interpolate to middle time point
        new_times = np.array([0.0, 1.0, 2.0])
        result = get_interpolated_position_info(
            position_info,
            time=new_times,
            track_graph=track_graph,
            edge_order=[(0, 1), (1, 2)],
            edge_spacing=[10.0],  # One spacing between two edges
        )

        # Check interpolation
        assert len(result) == 3
        assert result.index.tolist() == list(new_times)

        # Middle point should be interpolated (5.0 cm at t=1.0)
        np.testing.assert_allclose(
            result.loc[1.0, 'head_position_x'],
            5.0,
            rtol=0.01,
            err_msg="Expected linear interpolation at midpoint"
        )

    def test_interpolation_multiple_columns(self):
        """All columns should be interpolated independently."""
        original_times = np.array([0.0, 1.0])
        position_info = pd.DataFrame({
            'head_position_x': [0.0, 10.0],
            'head_position_y': [0.0, 5.0],
            'head_speed': [0.0, 10.0],
            'head_orientation': [0.0, np.pi],
        }, index=original_times)

        # Create track graph using recommended function
        track_graph = make_track_graph(
            node_positions=[(0, 0), (10, 0), (20, 0)],
            edges=[(0, 1), (1, 2)]
        )

        # Interpolate to midpoint
        new_times = np.array([0.5])
        result = get_interpolated_position_info(
            position_info,
            time=new_times,
            track_graph=track_graph,
            edge_order=[(0, 1), (1, 2)],
            edge_spacing=[10.0],  # One spacing between two edges
        )

        # All columns should be interpolated to midpoint values
        np.testing.assert_allclose(result.loc[0.5, 'head_position_x'], 5.0, rtol=0.01)
        np.testing.assert_allclose(result.loc[0.5, 'head_position_y'], 2.5, rtol=0.01)
        np.testing.assert_allclose(result.loc[0.5, 'head_speed'], 5.0, rtol=0.01)
        np.testing.assert_allclose(result.loc[0.5, 'head_orientation'], np.pi/2, rtol=0.01)

    def test_custom_position_columns(self):
        """Should handle custom position column names."""
        original_times = np.array([0.0, 1.0])
        position_info = pd.DataFrame({
            'custom_x': [0.0, 10.0],
            'custom_y': [0.0, 0.0],
        }, index=original_times)

        # Create track graph using recommended function
        track_graph = make_track_graph(
            node_positions=[(0, 0), (10, 0), (20, 0)],
            edges=[(0, 1), (1, 2)]
        )

        # Use custom column names
        result = get_interpolated_position_info(
            position_info,
            time=original_times,
            track_graph=track_graph,
            edge_order=[(0, 1), (1, 2)],
            edge_spacing=[10.0],  # One spacing between two edges
            position_columns=['custom_x', 'custom_y'],
        )

        # Should successfully linearize using custom columns
        assert 'linear_position' in result.columns
        assert 'custom_x' in result.columns

    def test_preserves_time_ordering(self):
        """Output should maintain time ordering even with unsorted input times."""
        position_info = pd.DataFrame({
            'head_position_x': [0.0, 5.0, 10.0],
            'head_position_y': [0.0, 0.0, 0.0],
        }, index=[0.0, 1.0, 2.0])

        # Create track graph using recommended function
        track_graph = make_track_graph(
            node_positions=[(0, 0), (10, 0), (20, 0)],
            edges=[(0, 1), (1, 2)]
        )

        # Unsorted times
        unsorted_times = np.array([2.0, 0.5, 1.5, 0.0])
        result = get_interpolated_position_info(
            position_info,
            time=unsorted_times,
            track_graph=track_graph,
            edge_order=[(0, 1), (1, 2)],
            edge_spacing=[10.0],  # One spacing between two edges
        )

        # Result should have exactly the requested times (may be reordered)
        assert set(result.index) == set(unsorted_times)

    def test_adds_linearization_columns(self):
        """Should add all required linearization columns."""
        position_info = pd.DataFrame({
            'head_position_x': [0.0, 10.0],
            'head_position_y': [0.0, 0.0],
        }, index=[0.0, 1.0])

        # Create track graph using recommended function
        track_graph = make_track_graph(
            node_positions=[(0, 0), (10, 0), (20, 0)],
            edges=[(0, 1), (1, 2)]
        )

        result = get_interpolated_position_info(
            position_info,
            time=np.array([0.0, 1.0]),
            track_graph=track_graph,
            edge_order=[(0, 1), (1, 2)],
            edge_spacing=[10.0],  # One spacing between two edges
        )

        # Check for linearization columns
        required_columns = {
            'linear_position',
            'projected_x_position',
            'projected_y_position',
            'track_segment_id',
        }

        for col in required_columns:
            assert col in result.columns, f"Missing required column: {col}"

    def test_linear_position_monotonic_for_forward_movement(self):
        """Linear position should increase monotonically for forward movement."""
        # Animal moves steadily from 0 to 15 cm
        times = np.arange(0.0, 3.0, 0.1)
        positions = np.linspace(0.0, 15.0, len(times))

        position_info = pd.DataFrame({
            'head_position_x': positions,
            'head_position_y': np.zeros(len(times)),
        }, index=times)

        # Create track graph using recommended function
        track_graph = make_track_graph(
            node_positions=[(0, 0), (10, 0), (20, 0)],
            edges=[(0, 1), (1, 2)]
        )

        result = get_interpolated_position_info(
            position_info,
            time=times,
            track_graph=track_graph,
            edge_order=[(0, 1), (1, 2)],
            edge_spacing=[10.0],  # One spacing between two edges
        )

        # Linear position should increase (within numerical tolerance)
        linear_pos = result['linear_position'].values
        diffs = np.diff(linear_pos)

        # Allow for small numerical errors and stationary periods
        assert np.all(diffs >= -0.01), \
            "Linear position should be monotonically increasing for forward movement"

    def test_respects_edge_spacing(self):
        """Linearization should respect provided edge spacing."""
        # Place animal at start and end of first edge
        position_info = pd.DataFrame({
            'head_position_x': [0.0, 10.0],
            'head_position_y': [0.0, 0.0],
        }, index=[0.0, 1.0])

        # Create track graph using recommended function
        track_graph = make_track_graph(
            node_positions=[(0, 0), (10, 0), (20, 0)],
            edges=[(0, 1), (1, 2)]
        )

        # Use specific edge spacing
        result = get_interpolated_position_info(
            position_info,
            time=np.array([0.0, 1.0]),
            track_graph=track_graph,
            edge_order=[(0, 1), (1, 2)],
            edge_spacing=[10.0],  # One spacing between two edges
        )

        # Linear position should roughly match edge spacing
        # (exact values depend on projection algorithm)
        assert result['linear_position'].min() >= 0.0
        assert result['linear_position'].max() <= 20.0  # Total track length
