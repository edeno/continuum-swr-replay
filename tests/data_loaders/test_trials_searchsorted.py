"""Regression test for ``get_start_end_node_id`` interval-boundary indexing.

The previous implementation used ``searchsorted(..., side="right")``
without subtracting 1, so an interval boundary that fell exactly on a
position-frame timestamp resolved to the *next* row. For an interval
[0, 1] over a position frame at t=0 (well 0) and t=1 (well 1), this
mislabeled both endpoints as well 1 — silently dropping ``from_well``
information for runs whose start coincides with the position-frame
timebase.
"""

from __future__ import annotations

import networkx as nx
import pandas as pd

from continuum_swr_replay.data_loaders.trials import get_start_end_node_id


def _two_well_track_graph() -> nx.Graph:
    g = nx.Graph()
    g.add_node(0, pos=(0.0, 0.0))
    g.add_node(1, pos=(1.0, 0.0))
    g.add_edge(0, 1)
    return g


def _two_row_position_frame() -> pd.DataFrame:
    # Animal at well 0 at t=0, at well 1 at t=1.
    return pd.DataFrame(
        {
            "projected_x_position": [0.0, 1.0],
            "projected_y_position": [0.0, 0.0],
        },
        index=pd.Index([0.0, 1.0], name="time"),
    )


class TestStartEndNodeIdInterval:
    def test_exact_boundary_match_does_not_skip_to_next_row(self):
        """Interval [0, 1] must resolve from_well=0, to_well=1."""
        position_info = _two_row_position_frame()
        track_graph = _two_well_track_graph()
        times = pd.DataFrame({"start_time": [0.0], "end_time": [1.0]})

        result = get_start_end_node_id(
            position_info, track_graph, times, well_nodes=[0, 1]
        )

        assert int(result["from_well"][0]) == 0, (
            f"expected from_well=0, got {result['from_well'].tolist()}"
        )
        assert int(result["to_well"][0]) == 1, (
            f"expected to_well=1, got {result['to_well'].tolist()}"
        )

    def test_end_past_position_index_clamps_to_last_row(self):
        """``side='right' - 1`` clamped to ``[0, max_idx]`` keeps an
        end-time past the last position sample on the final row."""
        position_info = _two_row_position_frame()
        track_graph = _two_well_track_graph()
        times = pd.DataFrame({"start_time": [0.0], "end_time": [10.0]})

        result = get_start_end_node_id(
            position_info, track_graph, times, well_nodes=[0, 1]
        )

        assert int(result["from_well"][0]) == 0
        assert int(result["to_well"][0]) == 1

    def test_start_before_position_index_clamps_to_first_row(self):
        """An interval starting before the first position sample must clamp
        to row 0 (rather than wrapping via -1)."""
        position_info = _two_row_position_frame()
        track_graph = _two_well_track_graph()
        times = pd.DataFrame({"start_time": [-1.0], "end_time": [0.0]})

        result = get_start_end_node_id(
            position_info, track_graph, times, well_nodes=[0, 1]
        )

        assert int(result["from_well"][0]) == 0
        assert int(result["to_well"][0]) == 0
