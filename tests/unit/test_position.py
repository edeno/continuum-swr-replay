"""Unit tests for position data processing.

Tests position interpolation and linearization functions used in the
data loading pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from track_linearization import make_track_graph

from continuum_swr_replay.data_loaders.position import get_interpolated_position_info


# 3-node linear track shared by every test in this module; the production
# call signature requires both ``track_graph`` and the parallel
# ``edge_order`` + ``edge_spacing`` lists, so bundle them once.
_EDGE_ORDER = [(0, 1), (1, 2)]
_EDGE_SPACING = [10.0]


@pytest.fixture
def linear_3node_track():
    return make_track_graph(
        node_positions=[(0, 0), (10, 0), (20, 0)],
        edges=_EDGE_ORDER,
    )


def _interpolate(position_info, time, track_graph, **kwargs):
    """Thin wrapper that fills in the per-module edge_order / edge_spacing."""
    return get_interpolated_position_info(
        position_info,
        time=time,
        track_graph=track_graph,
        edge_order=_EDGE_ORDER,
        edge_spacing=_EDGE_SPACING,
        **kwargs,
    )


class TestGetInterpolatedPositionInfo:
    """Test position interpolation and linearization."""

    def test_interpolation_preserves_original_times(self, linear_3node_track):
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

        result = _interpolate(position_info, original_times, linear_3node_track)

        assert {'linear_position', 'projected_x_position', 'track_segment_id'} <= set(result.columns)
        np.testing.assert_array_equal(
            result['head_position_x'].to_numpy(), original_x
        )
        np.testing.assert_array_equal(
            result['head_speed'].to_numpy(), original_speed
        )

    def test_interpolation_to_new_times(self, linear_3node_track):
        """New time points should have linearly interpolated values."""
        original_times = np.array([0.0, 2.0])
        position_info = pd.DataFrame({
            'head_position_x': [0.0, 10.0],
            'head_position_y': [0.0, 0.0],
        }, index=original_times)

        new_times = np.array([0.0, 1.0, 2.0])
        result = _interpolate(position_info, new_times, linear_3node_track)

        assert len(result) == 3
        assert result.index.tolist() == list(new_times)
        # Midpoint at t=1.0: 5.0 cm under linear interpolation.
        np.testing.assert_allclose(
            result.loc[1.0, 'head_position_x'],
            5.0,
            rtol=0.01,
            err_msg="Expected linear interpolation at midpoint",
        )

    def test_interpolation_multiple_columns(self, linear_3node_track):
        """All columns should be interpolated independently."""
        original_times = np.array([0.0, 1.0])
        position_info = pd.DataFrame({
            'head_position_x': [0.0, 10.0],
            'head_position_y': [0.0, 5.0],
            'head_speed': [0.0, 10.0],
            'head_orientation': [0.0, np.pi],
        }, index=original_times)

        result = _interpolate(position_info, np.array([0.5]), linear_3node_track)

        np.testing.assert_allclose(result.loc[0.5, 'head_position_x'], 5.0, rtol=0.01)
        np.testing.assert_allclose(result.loc[0.5, 'head_position_y'], 2.5, rtol=0.01)
        np.testing.assert_allclose(result.loc[0.5, 'head_speed'], 5.0, rtol=0.01)
        np.testing.assert_allclose(result.loc[0.5, 'head_orientation'], np.pi / 2, rtol=0.01)

    def test_custom_position_columns(self, linear_3node_track):
        """Should handle custom position column names."""
        original_times = np.array([0.0, 1.0])
        position_info = pd.DataFrame({
            'custom_x': [0.0, 10.0],
            'custom_y': [0.0, 0.0],
        }, index=original_times)

        result = _interpolate(
            position_info, original_times, linear_3node_track,
            position_columns=['custom_x', 'custom_y'],
        )

        assert 'linear_position' in result.columns
        assert 'custom_x' in result.columns

    def test_preserves_time_ordering(self, linear_3node_track):
        """Output should maintain time ordering even with unsorted input times."""
        position_info = pd.DataFrame({
            'head_position_x': [0.0, 5.0, 10.0],
            'head_position_y': [0.0, 0.0, 0.0],
        }, index=[0.0, 1.0, 2.0])

        unsorted_times = np.array([2.0, 0.5, 1.5, 0.0])
        result = _interpolate(position_info, unsorted_times, linear_3node_track)

        assert set(result.index) == set(unsorted_times)

    def test_adds_linearization_columns(self, linear_3node_track):
        """Should add all required linearization columns."""
        position_info = pd.DataFrame({
            'head_position_x': [0.0, 10.0],
            'head_position_y': [0.0, 0.0],
        }, index=[0.0, 1.0])

        result = _interpolate(position_info, np.array([0.0, 1.0]), linear_3node_track)

        required_columns = {
            'linear_position',
            'projected_x_position',
            'projected_y_position',
            'track_segment_id',
        }
        for col in required_columns:
            assert col in result.columns, f"Missing required column: {col}"

    def test_linear_position_monotonic_for_forward_movement(self, linear_3node_track):
        """Linear position should increase monotonically for forward movement."""
        times = np.arange(0.0, 3.0, 0.1)
        positions = np.linspace(0.0, 15.0, len(times))

        position_info = pd.DataFrame({
            'head_position_x': positions,
            'head_position_y': np.zeros(len(times)),
        }, index=times)

        result = _interpolate(position_info, times, linear_3node_track)

        linear_pos = result['linear_position'].values
        diffs = np.diff(linear_pos)
        assert np.all(diffs >= -0.01), \
            "Linear position should be monotonically increasing for forward movement"

    def test_respects_edge_spacing(self, linear_3node_track):
        """Linearization output should fall within the total track length."""
        position_info = pd.DataFrame({
            'head_position_x': [0.0, 10.0],
            'head_position_y': [0.0, 0.0],
        }, index=[0.0, 1.0])

        result = _interpolate(position_info, np.array([0.0, 1.0]), linear_3node_track)

        assert result['linear_position'].min() >= 0.0
        assert result['linear_position'].max() <= 20.0  # Total track length
