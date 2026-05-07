"""Property-based tests for project-level mathematical invariants.

Uses :mod:`hypothesis` to generate random inputs and verify properties
that downstream code relies on. Restricted to invariants that
actually exercise project code — generic numpy/scipy/networkx
smoke tests (z-score mean ≈ 0, sort monotonicity, triangle
inequality on random graphs, etc.) belong upstream, not here.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from continuum_swr_replay.data_loaders.task_variables import (
    path_progress_for_run_segment,
)


class TestPathProgressInvariants:
    """``path_progress_for_run_segment`` must always emit values in
    ``[0, 1]`` and never decrease as the animal advances along a
    monotone segment — downstream code (event labeling, run-segment
    weighting) treats it as a normalized run coordinate.
    """

    @given(
        n_points=st.integers(min_value=5, max_value=200),
        start_distance=st.floats(min_value=0.0, max_value=50.0),
        end_distance=st.floats(min_value=0.1, max_value=100.0),
    )
    @settings(max_examples=50, deadline=None)
    def test_path_progress_always_in_zero_one(
        self, n_points, start_distance, end_distance
    ):
        graph = nx.Graph()
        graph.add_edge(0, 1)
        graph.nodes[0]["pos"] = (0.0, 0.0)
        graph.nodes[1]["pos"] = (end_distance, 0.0)
        graph.edges[(0, 1)]["edge_id"] = 0
        graph.edges[(0, 1)]["distance"] = end_distance

        time = np.linspace(0, 1.0, n_points)
        linear_position = np.linspace(start_distance, end_distance, n_points)

        pos_info = pd.DataFrame(
            {
                "projected_x_position": linear_position,
                "projected_y_position": np.zeros(n_points),
                "track_segment_id": np.zeros(n_points, dtype=int),
            },
            index=time,
        )

        progress = path_progress_for_run_segment(
            pos_info, graph, start_node=0, end_node=1
        )

        assert np.all(progress >= 0.0), (
            f"Progress must be >= 0, got min={progress.min()}"
        )
        assert np.all(progress <= 1.0), (
            f"Progress must be <= 1, got max={progress.max()}"
        )

    @given(n_points=st.integers(min_value=5, max_value=200))
    @settings(max_examples=50, deadline=None)
    def test_path_progress_monotonic(self, n_points):
        graph = nx.Graph()
        graph.add_edge(0, 1)
        graph.nodes[0]["pos"] = (0.0, 0.0)
        graph.nodes[1]["pos"] = (20.0, 0.0)
        graph.edges[(0, 1)]["edge_id"] = 0
        graph.edges[(0, 1)]["distance"] = 20.0

        time = np.linspace(0, 1.0, n_points)
        linear_position = np.linspace(0, 20, n_points)

        pos_info = pd.DataFrame(
            {
                "projected_x_position": linear_position,
                "projected_y_position": np.zeros(n_points),
                "track_segment_id": np.zeros(n_points, dtype=int),
            },
            index=time,
        )

        progress = path_progress_for_run_segment(
            pos_info, graph, start_node=0, end_node=1
        )

        diffs = np.diff(progress)
        assert np.all(diffs >= -1e-10), (
            f"Progress should be monotonic, found negative diff: {diffs.min()}"
        )
