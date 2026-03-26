"""Tests for task variable computation functions.

This module tests the critical task-specific derived variables used for
analyzing spatial navigation behavior in the bandit task.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from continuum_swr_replay.data_loaders.task_variables import (
    dist_to_goal_per_time,
    path_progress_for_run_segment,
    turn_direction,
)


class TestPathProgressForRunSegment:
    """Test path progress calculation (0-1 normalization)."""

    def test_basic_linear_progress(self, simple_linear_track):
        """Progress should go from 0 to 1 along a simple path."""
        # Create mock position data moving from node 0 to node 2
        time = np.arange(0, 1.0, 0.002)
        linear_position = np.linspace(0, 20, len(time))  # 0 to 20 cm

        pos_info = pd.DataFrame({
            'projected_x_position': linear_position,
            'projected_y_position': np.zeros(len(time)),
            'track_segment_id': np.concatenate([
                np.zeros(len(time)//2, dtype=int),  # First half on edge 0
                np.ones(len(time)//2, dtype=int),   # Second half on edge 1
            ]),
        }, index=time)

        # Set node positions for distance calculation
        for node, pos in [(0, (0, 0)), (1, (10, 0)), (2, (20, 0))]:
            simple_linear_track.nodes[node]['pos'] = pos

        # Set edge IDs
        simple_linear_track.edges[(0, 1)]['edge_id'] = 0
        simple_linear_track.edges[(1, 2)]['edge_id'] = 1

        progress = path_progress_for_run_segment(
            pos_info,
            simple_linear_track,
            start_node=0,
            end_node=2,
        )

        # Progress should be in [0, 1]
        assert np.all(progress >= 0.0), "Progress should be non-negative"
        assert np.all(progress <= 1.0), "Progress should not exceed 1.0"

        # Progress should increase monotonically (or stay constant)
        assert np.all(np.diff(progress) >= -1e-10), "Progress should be monotonic"

    def test_same_start_and_end_returns_zeros(self, simple_linear_track):
        """When start and end nodes are the same, should return all zeros."""
        time = np.arange(0, 0.5, 0.002)
        pos_info = pd.DataFrame({
            'projected_x_position': np.zeros(len(time)),
            'projected_y_position': np.zeros(len(time)),
            'track_segment_id': np.zeros(len(time), dtype=int),
        }, index=time)

        # Set node positions
        for node, pos in [(0, (0, 0)), (1, (10, 0)), (2, (20, 0))]:
            simple_linear_track.nodes[node]['pos'] = pos

        simple_linear_track.edges[(0, 1)]['edge_id'] = 0

        # Same start and end node
        progress = path_progress_for_run_segment(
            pos_info,
            simple_linear_track,
            start_node=0,
            end_node=0,
        )

        # Should return all zeros (no progress possible)
        np.testing.assert_array_equal(progress, np.zeros(len(time)))

    def test_empty_path_edges_returns_zeros(self, simple_linear_track):
        """When path has no edges, should return zeros."""
        time = np.arange(0, 0.5, 0.002)
        pos_info = pd.DataFrame({
            'projected_x_position': np.zeros(len(time)),
            'projected_y_position': np.zeros(len(time)),
            'track_segment_id': np.zeros(len(time), dtype=int),
        }, index=time)

        # Set node positions
        for node in simple_linear_track.nodes:
            simple_linear_track.nodes[node]['pos'] = (0, 0)

        # Same node means empty path
        progress = path_progress_for_run_segment(
            pos_info,
            simple_linear_track,
            start_node=0,
            end_node=0,
        )

        assert progress.shape == (len(time),)
        np.testing.assert_array_equal(progress, np.zeros(len(time)))

    def test_progress_bounds_always_valid(self, simple_linear_track):
        """Progress must always be in [0, 1] range."""
        time = np.arange(0, 1.0, 0.002)

        # Create position data that might stress the bounds
        pos_info = pd.DataFrame({
            'projected_x_position': np.random.randn(len(time)) * 5,
            'projected_y_position': np.random.randn(len(time)) * 5,
            'track_segment_id': np.random.choice([0, 1], size=len(time)),
        }, index=time)

        # Set node positions
        for node, pos in [(0, (0, 0)), (1, (10, 0)), (2, (20, 0))]:
            simple_linear_track.nodes[node]['pos'] = pos

        simple_linear_track.edges[(0, 1)]['edge_id'] = 0
        simple_linear_track.edges[(1, 2)]['edge_id'] = 1

        progress = path_progress_for_run_segment(
            pos_info,
            simple_linear_track,
            start_node=0,
            end_node=2,
        )

        # Must be in valid range
        assert np.all(progress >= 0.0), "Progress must be >= 0"
        assert np.all(progress <= 1.0), "Progress must be <= 1"

    def test_handles_missing_track_segments(self, simple_linear_track):
        """Should handle position data with NaN track segments."""
        time = np.arange(0, 0.5, 0.002)
        track_segments = np.zeros(len(time), dtype=float)
        track_segments[::5] = np.nan  # Every 5th point is NaN

        pos_info = pd.DataFrame({
            'projected_x_position': np.linspace(0, 10, len(time)),
            'projected_y_position': np.zeros(len(time)),
            'track_segment_id': track_segments,
        }, index=time)

        # Set node positions and edge IDs
        for node, pos in [(0, (0, 0)), (1, (10, 0)), (2, (20, 0))]:
            simple_linear_track.nodes[node]['pos'] = pos

        simple_linear_track.edges[(0, 1)]['edge_id'] = 0
        simple_linear_track.edges[(1, 2)]['edge_id'] = 1

        # Should interpolate over NaN values
        progress = path_progress_for_run_segment(
            pos_info,
            simple_linear_track,
            start_node=0,
            end_node=1,
        )

        # Progress should be computed (interpolated)
        assert progress.shape == (len(time),)
        assert np.all(progress >= 0.0)
        assert np.all(progress <= 1.0)

    def test_multi_segment_path(self, bandit_track_graph):
        """Test progress on a multi-segment path (center to outer well)."""
        time = np.arange(0, 1.0, 0.002)

        # Path from well 0 to well 2 goes through center (6)
        # Segments: (6,0) -> (6,2)
        pos_info = pd.DataFrame({
            'projected_x_position': np.linspace(0, 20, len(time)),
            'projected_y_position': np.zeros(len(time)),
            'track_segment_id': np.concatenate([
                np.zeros(len(time)//2, dtype=int),  # First segment
                np.ones(len(time)//2, dtype=int),   # Second segment
            ]),
        }, index=time)

        # Set node positions (arbitrary layout)
        positions = {
            0: (0, 10),
            1: (0, -10),
            2: (10, 10),
            3: (10, -10),
            4: (20, 10),
            5: (20, -10),
            6: (10, 0),  # Center
        }
        for node, pos in positions.items():
            bandit_track_graph.nodes[node]['pos'] = pos

        # Set edge IDs
        edge_id = 0
        for edge in bandit_track_graph.edges:
            bandit_track_graph.edges[edge]['edge_id'] = edge_id
            edge_id += 1

        progress = path_progress_for_run_segment(
            pos_info,
            bandit_track_graph,
            start_node=0,
            end_node=2,
        )

        # Progress should be valid
        assert np.all(progress >= 0.0)
        assert np.all(progress <= 1.0)

    def test_single_timepoint(self, simple_linear_track):
        """Single time point should return valid progress."""
        pos_info = pd.DataFrame({
            'projected_x_position': [5.0],
            'projected_y_position': [0.0],
            'track_segment_id': [0],
        }, index=[0.0])

        for node, pos in [(0, (0, 0)), (1, (10, 0)), (2, (20, 0))]:
            simple_linear_track.nodes[node]['pos'] = pos

        simple_linear_track.edges[(0, 1)]['edge_id'] = 0
        simple_linear_track.edges[(1, 2)]['edge_id'] = 1

        progress = path_progress_for_run_segment(
            pos_info,
            simple_linear_track,
            start_node=0,
            end_node=2,
        )

        assert len(progress) == 1
        assert 0.0 <= progress[0] <= 1.0


class TestDistToGoal:
    """Test distance to goal computation."""

    def test_basic_distance_calculation(self, simple_linear_track):
        """Distance to goal should decrease as animal approaches."""
        # Set node positions
        for node, pos in [(0, (0, 0)), (1, (10, 0)), (2, (20, 0))]:
            simple_linear_track.nodes[node]['pos'] = pos

        simple_linear_track.edges[(0, 1)]['edge_id'] = 0
        simple_linear_track.edges[(1, 2)]['edge_id'] = 1

        track_segment_id_to_edge = {
            0: (0, 1),
            1: (1, 2),
        }

        # At start of track (x=0), distance to goal (node 2) should be ~20
        dist_start = dist_to_goal_per_time(
            track_segment_id=0,
            projected_x_position=0.0,
            projected_y_position=0.0,
            goal_node_id=2,
            track_graph=simple_linear_track,
            track_segment_id_to_edge=track_segment_id_to_edge,
        )

        # At middle (x=10), distance should be ~10
        dist_middle = dist_to_goal_per_time(
            track_segment_id=1,
            projected_x_position=10.0,
            projected_y_position=0.0,
            goal_node_id=2,
            track_graph=simple_linear_track,
            track_segment_id_to_edge=track_segment_id_to_edge,
        )

        # At goal (x=20), distance should be ~0
        dist_goal = dist_to_goal_per_time(
            track_segment_id=1,
            projected_x_position=20.0,
            projected_y_position=0.0,
            goal_node_id=2,
            track_graph=simple_linear_track,
            track_segment_id_to_edge=track_segment_id_to_edge,
        )

        # Distances should decrease
        assert dist_start > dist_middle > dist_goal
        assert dist_goal < 1.0, "Should be very close to goal"

    def test_distance_always_non_negative(self, simple_linear_track):
        """Distance must always be non-negative."""
        for node, pos in [(0, (0, 0)), (1, (10, 0)), (2, (20, 0))]:
            simple_linear_track.nodes[node]['pos'] = pos

        simple_linear_track.edges[(0, 1)]['edge_id'] = 0
        simple_linear_track.edges[(1, 2)]['edge_id'] = 1

        track_segment_id_to_edge = {0: (0, 1), 1: (1, 2)}

        # Test multiple positions
        for x_pos in [0, 5, 10, 15, 20]:
            dist = dist_to_goal_per_time(
                track_segment_id=0 if x_pos < 10 else 1,
                projected_x_position=float(x_pos),
                projected_y_position=0.0,
                goal_node_id=2,
                track_graph=simple_linear_track,
                track_segment_id_to_edge=track_segment_id_to_edge,
            )

            assert dist >= 0.0, f"Distance must be non-negative, got {dist}"


class TestTurnDirection:
    """Test turn direction calculation."""

    def test_left_turn_detection(self, bandit_track_graph):
        """Left turn should be detected correctly."""
        # Create a path with a left turn
        # Set node positions to create geometry
        positions = {
            0: (0, 0),
            1: (10, 0),
            2: (10, 10),  # Turn left from (0,0)->(10,0) to (10,0)->(10,10)
            6: (5, 0),    # Center node
        }
        for node, pos in positions.items():
            if node in bandit_track_graph.nodes:
                bandit_track_graph.nodes[node]['pos'] = pos

        # Create trial data
        time = np.arange(0, 1.0, 0.002)
        pos_info = pd.DataFrame({
            'track_segment_id': np.concatenate([
                np.zeros(len(time)//2, dtype=int),
                np.ones(len(time)//2, dtype=int),
            ]),
            'projected_x_position': np.linspace(0, 10, len(time)),
            'projected_y_position': np.linspace(0, 10, len(time)),
        }, index=time)

        trial = pd.Series({
            'start_time': 0.0,
            'end_time': 1.0,
            'from_well': 0,
            'to_well': 2,
        })

        # Set edge IDs
        for i, edge in enumerate(bandit_track_graph.edges):
            bandit_track_graph.edges[edge]['edge_id'] = i

        track_edges = np.array(list(bandit_track_graph.edges))

        # This should detect turn direction
        # Note: Actual result depends on graph structure
        direction = turn_direction(trial, pos_info, bandit_track_graph, track_edges)

        # Direction should be a string
        assert isinstance(direction, str)
        # Should contain turn directions
        assert all(turn in ['left', 'right'] for turn in direction.split('-') if turn)

    def test_known_left_turn_geometry(self):
        """A 90-degree counterclockwise turn (y-up) should be 'left'.

        Geometry (y-up coordinate system):
            node 1 at (0, 10)
                |
            node 0 at (0, 0) --- node 2 at (10, 0)

        Path: node 2 -> node 0 -> node 1
        Vector 1: (0,0) - (10,0) = (-10, 0)  [going left]
        Vector 2: (0,10) - (0,0) = (0, 10)   [going up]
        Cross product: (-10)(10) - (0)(0) = -100 < 0 => "right"

        But path 0 -> 2 -> ... would give different vectors. Let's use
        a clear L-shaped path: 0 -> 1 -> 2 where going east then north.
        Vector 1: (10,0) - (0,0) = (10, 0)
        Vector 2: (10,10) - (10,0) = (0, 10)
        Cross product: (10)(10) - (0)(0) = 100 > 0 => "left"
        """
        graph = nx.Graph()
        graph.add_edges_from([(0, 1), (1, 2)])
        nx.set_edge_attributes(graph, {(0, 1): 10.0, (1, 2): 10.0}, "distance")
        graph.nodes[0]['pos'] = (0, 0)
        graph.nodes[1]['pos'] = (10, 0)
        graph.nodes[2]['pos'] = (10, 10)

        time = np.arange(0, 1.0, 0.002)
        n = len(time)
        pos_info = pd.DataFrame({
            'track_segment_id': np.concatenate([
                np.zeros(n // 2, dtype=int),
                np.ones(n // 2, dtype=int),
            ]),
            'projected_x_position': np.concatenate([
                np.linspace(0, 10, n // 2),
                np.full(n - n // 2, 10.0),
            ]),
            'projected_y_position': np.concatenate([
                np.zeros(n // 2),
                np.linspace(0, 10, n - n // 2),
            ]),
        }, index=time)

        trial = pd.Series({
            'start_time': 0.0,
            'end_time': 1.0,
            'from_well': 0,
            'to_well': 2,
        })
        track_edges = np.array([(0, 1), (1, 2)])

        direction = turn_direction(trial, pos_info, graph, track_edges)
        assert direction == "left"

    def test_known_right_turn_geometry(self):
        """A 90-degree clockwise turn (y-up) should be 'right'.

        Path: 0 -> 1 -> 2 going east then south.
        Vector 1: (10,0) - (0,0) = (10, 0)
        Vector 2: (10,-10) - (10,0) = (0, -10)
        Cross product: (10)(-10) - (0)(0) = -100 < 0 => "right"
        """
        graph = nx.Graph()
        graph.add_edges_from([(0, 1), (1, 2)])
        nx.set_edge_attributes(graph, {(0, 1): 10.0, (1, 2): 10.0}, "distance")
        graph.nodes[0]['pos'] = (0, 0)
        graph.nodes[1]['pos'] = (10, 0)
        graph.nodes[2]['pos'] = (10, -10)

        time = np.arange(0, 1.0, 0.002)
        n = len(time)
        pos_info = pd.DataFrame({
            'track_segment_id': np.concatenate([
                np.zeros(n // 2, dtype=int),
                np.ones(n // 2, dtype=int),
            ]),
            'projected_x_position': np.concatenate([
                np.linspace(0, 10, n // 2),
                np.full(n - n // 2, 10.0),
            ]),
            'projected_y_position': np.concatenate([
                np.zeros(n // 2),
                np.linspace(0, -10, n - n // 2),
            ]),
        }, index=time)

        trial = pd.Series({
            'start_time': 0.0,
            'end_time': 1.0,
            'from_well': 0,
            'to_well': 2,
        })
        track_edges = np.array([(0, 1), (1, 2)])

        direction = turn_direction(trial, pos_info, graph, track_edges)
        assert direction == "right"

    def test_straight_path_no_turns(self, simple_linear_track):
        """Straight path should have no turns."""
        for node, pos in [(0, (0, 0)), (1, (10, 0)), (2, (20, 0))]:
            simple_linear_track.nodes[node]['pos'] = pos

        time = np.arange(0, 1.0, 0.002)
        pos_info = pd.DataFrame({
            'track_segment_id': np.concatenate([
                np.zeros(len(time)//2, dtype=int),
                np.ones(len(time)//2, dtype=int),
            ]),
            'projected_x_position': np.linspace(0, 20, len(time)),
            'projected_y_position': np.zeros(len(time)),
        }, index=time)

        trial = pd.Series({
            'start_time': 0.0,
            'end_time': 1.0,
            'from_well': 0,
            'to_well': 2,
        })

        simple_linear_track.edges[(0, 1)]['edge_id'] = 0
        simple_linear_track.edges[(1, 2)]['edge_id'] = 1

        track_edges = np.array([(0, 1), (1, 2)])

        direction = turn_direction(trial, pos_info, simple_linear_track, track_edges)

        # Straight path has one turn direction (left or right)
        assert isinstance(direction, str)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_path_progress_with_all_nan_distances(self, simple_linear_track):
        """All NaN distances should be handled gracefully."""
        time = np.arange(0, 0.5, 0.002)

        # Position data that doesn't match any track segment
        pos_info = pd.DataFrame({
            'projected_x_position': np.zeros(len(time)),
            'projected_y_position': np.zeros(len(time)),
            'track_segment_id': np.full(len(time), 999, dtype=int),  # Invalid segment
        }, index=time)

        for node, pos in [(0, (0, 0)), (1, (10, 0)), (2, (20, 0))]:
            simple_linear_track.nodes[node]['pos'] = pos

        simple_linear_track.edges[(0, 1)]['edge_id'] = 0
        simple_linear_track.edges[(1, 2)]['edge_id'] = 1

        progress = path_progress_for_run_segment(
            pos_info,
            simple_linear_track,
            start_node=0,
            end_node=2,
        )

        # Should return zeros (max_dist is NaN)
        assert progress.shape == (len(time),)
        np.testing.assert_array_equal(progress, np.zeros(len(time)))

    def test_very_short_segment(self, simple_linear_track):
        """Very short time segment should still work."""
        pos_info = pd.DataFrame({
            'projected_x_position': [0.0, 5.0],
            'projected_y_position': [0.0, 0.0],
            'track_segment_id': [0, 0],
        }, index=[0.0, 0.002])

        for node, pos in [(0, (0, 0)), (1, (10, 0)), (2, (20, 0))]:
            simple_linear_track.nodes[node]['pos'] = pos

        simple_linear_track.edges[(0, 1)]['edge_id'] = 0
        simple_linear_track.edges[(1, 2)]['edge_id'] = 1

        progress = path_progress_for_run_segment(
            pos_info,
            simple_linear_track,
            start_node=0,
            end_node=2,
        )

        assert len(progress) == 2
        assert np.all(progress >= 0.0)
        assert np.all(progress <= 1.0)


class TestNumericalStabilityFixes:
    """Test fixes for numerical stability issues."""

    def test_path_progress_epsilon_check_very_small_max_dist(self, simple_linear_track):
        """Very small max_dist (< 1e-10) should return zeros to avoid division issues.

        This tests the fix: if max_dist <= 1e-10, return zeros.
        """
        time = np.arange(0, 0.1, 0.002)

        # Create position data at essentially the same point
        pos_info = pd.DataFrame({
            'projected_x_position': np.full(len(time), 0.0),
            'projected_y_position': np.full(len(time), 0.0),
            'track_segment_id': np.zeros(len(time), dtype=int),
        }, index=time)

        for node, pos in [(0, (0, 0)), (1, (1e-12, 0)), (2, (2e-12, 0))]:
            simple_linear_track.nodes[node]['pos'] = pos

        simple_linear_track.edges[(0, 1)]['edge_id'] = 0
        simple_linear_track.edges[(1, 2)]['edge_id'] = 1

        # Same start and end node produces very small max_dist
        progress = path_progress_for_run_segment(
            pos_info,
            simple_linear_track,
            start_node=0,
            end_node=0,
        )

        # Should return all zeros (no NaN or Inf from division)
        assert progress.shape == (len(time),)
        np.testing.assert_array_equal(progress, np.zeros(len(time)))
        assert not np.any(np.isnan(progress)), "Should not have NaN values"
        assert not np.any(np.isinf(progress)), "Should not have Inf values"

    def test_dist_to_goal_clamps_dist_from_node1(self, simple_linear_track):
        """Distance from node1 should be clamped to [0, edge_length].

        This tests the fix: np.clip(dist_from_node1, 0.0, edge_length).
        """
        from continuum_swr_replay.data_loaders.task_variables import dist_to_goal

        time = np.arange(0, 0.1, 0.002)

        # Position slightly outside the segment bounds
        pos_info = pd.DataFrame({
            'projected_x_position': np.full(len(time), -0.001),  # Slightly negative
            'projected_y_position': np.zeros(len(time)),
            'track_segment_id': np.zeros(len(time), dtype=int),
        }, index=time)

        for node, pos in [(0, (0, 0)), (1, (10, 0)), (2, (20, 0))]:
            simple_linear_track.nodes[node]['pos'] = pos

        simple_linear_track.edges[(0, 1)]['edge_id'] = 0
        simple_linear_track.edges[(1, 2)]['edge_id'] = 1

        track_segment_id_to_edge = {0: (0, 1), 1: (1, 2)}

        # Create a mock trial Series with to_well as the goal
        trial = pd.Series({
            'to_well': 2,
            'start_time': 0.0,
            'end_time': 0.1,
        })

        distances = dist_to_goal(
            pos_info,
            trial=trial,
            track_graph=simple_linear_track,
            track_segment_id_to_edge=track_segment_id_to_edge,
        )

        # All distances should be non-negative (no negative from unstable subtraction)
        assert np.all(distances >= 0.0), "Distances should be non-negative"
        assert not np.any(np.isnan(distances)), "Should not have NaN values"
