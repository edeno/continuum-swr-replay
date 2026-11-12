"""Unit tests for trial and well visit construction.

Tests the logic for constructing behavioral trials from raw task events
(beam breaks, pump activations) and position data.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.data_loaders.constants import WELL_PATCH_MAPPING
from continuum_swr_replay.data_loaders.trials import (
    get_poke_in_poke_out_times,
    get_run_intervals,
    get_start_end_node_id,
)


class TestGetPokeInPokeOutTimes:
    """Test merging of beam break events into poke intervals."""

    def test_single_poke_event(self):
        """Single beam break should produce single poke interval."""
        beam_breaks = pd.DataFrame(
            {
                "start_time": [1.0],
                "end_time": [1.5],
                "event_name": ["Poke1"],
            }
        )

        result = get_poke_in_poke_out_times(beam_breaks)

        assert len(result) == 1
        assert result.iloc[0]["start_time"] == 1.0
        assert result.iloc[0]["end_time"] == 1.5
        assert result.iloc[0]["duration"] == pytest.approx(0.5)

    def test_merge_consecutive_events(self):
        """Beam breaks with short inter-event interval should be merged."""
        # Two events 0.5s apart (below 1.0s threshold)
        beam_breaks = pd.DataFrame(
            {
                "start_time": [1.0, 2.0],
                "end_time": [1.5, 2.5],
                "event_name": ["Poke1", "Poke1"],
            }
        )

        result = get_poke_in_poke_out_times(beam_breaks, max_inter_event_interval=1.0)

        # Should be merged into one poke
        assert len(result) == 1
        assert result.iloc[0]["start_time"] == 1.0
        assert result.iloc[0]["end_time"] == 2.5

    def test_separate_distant_events(self):
        """Beam breaks with long inter-event interval should remain separate."""
        # Two events 2.0s apart (exceeds 1.0s threshold)
        beam_breaks = pd.DataFrame(
            {
                "start_time": [1.0, 3.5],
                "end_time": [1.5, 4.0],
                "event_name": ["Poke1", "Poke1"],
            }
        )

        result = get_poke_in_poke_out_times(beam_breaks, max_inter_event_interval=1.0)

        # Should remain as two separate pokes
        assert len(result) == 2
        assert result.iloc[0]["start_time"] == 1.0
        assert result.iloc[1]["start_time"] == 3.5

    def test_custom_threshold(self):
        """Should respect custom max_inter_event_interval threshold."""
        beam_breaks = pd.DataFrame(
            {
                "start_time": [1.0, 2.0, 5.0],
                "end_time": [1.5, 2.5, 5.5],
                "event_name": ["Poke1", "Poke1", "Poke1"],
            }
        )

        # With threshold of 1.5s, first two should merge
        result = get_poke_in_poke_out_times(beam_breaks, max_inter_event_interval=1.5)

        assert len(result) == 2
        assert result.iloc[0]["end_time"] == 2.5  # Merged
        assert result.iloc[1]["start_time"] == 5.0  # Separate

    def test_multiple_event_names(self):
        """A positive gap preserves distinct physical bouts by default."""
        beam_breaks = pd.DataFrame(
            {
                "start_time": [1.0, 1.2, 3.0, 3.2],
                "end_time": [1.1, 1.3, 3.1, 3.3],
                "event_name": ["Poke1", "Poke1", "Poke2", "Poke2"],
            }
        )

        result = get_poke_in_poke_out_times(beam_breaks)

        assert len(result) == 4
        assert result["event_name"].tolist() == ["Poke1", "Poke1", "Poke2", "Poke2"]

    def test_preserves_temporal_ordering(self):
        """Result should be sorted by start_time."""
        beam_breaks = pd.DataFrame(
            {
                "start_time": [5.0, 1.0, 3.0],
                "end_time": [5.5, 1.5, 3.5],
                "event_name": ["Poke1", "Poke2", "Poke3"],
            }
        )

        result = get_poke_in_poke_out_times(beam_breaks)

        # Should be sorted
        assert result["start_time"].is_monotonic_increasing

    def test_computes_duration_correctly(self):
        """Duration should be end_time - start_time."""
        beam_breaks = pd.DataFrame(
            {
                "start_time": [1.0, 2.0],
                "end_time": [1.5, 3.0],
                "event_name": ["Poke1", "Poke1"],
            }
        )

        result = get_poke_in_poke_out_times(beam_breaks)

        assert result["duration"].tolist() == pytest.approx([0.5, 1.0])

    def test_unsorted_input_does_not_corrupt_pokes(self):
        """Regression: ``shift(-1)`` assumes chronological order
        within each ``event_name`` group. Pre-fix, an unsorted
        input produced negative-duration pokes (e.g.
        ``start_time=3.0, end_time=1.5, duration=-1.5``). The fix
        sorts each group by ``start_time`` before the
        inter-event-interval / cumsum step."""
        beam_breaks = pd.DataFrame(
            [
                {"event_name": "Poke1", "start_time": 3.0, "end_time": 3.5},
                {"event_name": "Poke1", "start_time": 1.0, "end_time": 1.5},
            ]
        )
        result = get_poke_in_poke_out_times(beam_breaks)
        assert (result["duration"] >= 0).all(), (
            f"negative-duration poke leaked: {result.to_dict('records')}"
        )
        # Two distinct pokes (gap > default max_inter_event_interval).
        assert len(result) == 2
        # Sorted result starts at 1.0.
        assert result.iloc[0]["start_time"] == pytest.approx(1.0)
        assert result.iloc[1]["start_time"] == pytest.approx(3.0)

    def test_empty_beam_breaks(self):
        """Should handle empty beam breaks DataFrame."""
        beam_breaks = pd.DataFrame(
            {
                "start_time": pd.Series([], dtype=float),
                "end_time": pd.Series([], dtype=float),
                "event_name": pd.Series([], dtype=str),
            }
        )

        result = get_poke_in_poke_out_times(beam_breaks)

        assert len(result) == 0
        assert "duration" in result.columns
        assert "event_name" in result.columns

    def test_three_way_merge(self):
        """Should merge three consecutive events if all within threshold."""
        beam_breaks = pd.DataFrame(
            {
                "start_time": [1.0, 1.6, 2.2],
                "end_time": [1.5, 2.0, 2.5],
                "event_name": ["Poke1", "Poke1", "Poke1"],
            }
        )

        result = get_poke_in_poke_out_times(beam_breaks, max_inter_event_interval=1.0)

        # All should merge into one
        assert len(result) == 1
        assert result.iloc[0]["start_time"] == 1.0
        assert result.iloc[0]["end_time"] == 2.5


class TestGetRunIntervals:
    """Test extraction of run intervals between pokes."""

    def test_single_run_between_two_pokes(self):
        """Two pokes should produce one run interval."""
        poke_times = pd.DataFrame(
            {
                "start_time": [1.0, 5.0],
                "end_time": [2.0, 6.0],
            }
        )

        result = get_run_intervals(poke_times)

        assert len(result) == 1
        assert result.iloc[0]["start_time"] == 2.0  # End of first poke
        assert result.iloc[0]["end_time"] == 5.0  # Start of second poke

    def test_multiple_runs(self):
        """Three pokes should produce two run intervals."""
        poke_times = pd.DataFrame(
            {
                "start_time": [1.0, 3.0, 5.0],
                "end_time": [1.5, 3.5, 5.5],
            }
        )

        result = get_run_intervals(poke_times)

        assert len(result) == 2
        # Run 1: between poke 1 and poke 2
        assert result.iloc[0]["start_time"] == 1.5
        assert result.iloc[0]["end_time"] == 3.0
        # Run 2: between poke 2 and poke 3
        assert result.iloc[1]["start_time"] == 3.5
        assert result.iloc[1]["end_time"] == 5.0

    def test_no_runs_from_single_poke(self):
        """Single poke should produce no run intervals."""
        poke_times = pd.DataFrame(
            {
                "start_time": [1.0],
                "end_time": [2.0],
            }
        )

        result = get_run_intervals(poke_times)

        assert len(result) == 0

    def test_computes_duration_correctly(self):
        """Duration should be end_time - start_time."""
        poke_times = pd.DataFrame(
            {
                "start_time": [1.0, 5.0],
                "end_time": [2.0, 6.0],
            }
        )

        result = get_run_intervals(poke_times)

        expected_duration = 5.0 - 2.0  # 3.0 seconds
        assert result.iloc[0]["duration"] == pytest.approx(expected_duration)

    def test_handles_empty_poke_times(self):
        """Should handle empty poke times DataFrame."""
        poke_times = pd.DataFrame(
            {
                "start_time": [],
                "end_time": [],
            }
        )

        result = get_run_intervals(poke_times)

        assert len(result) == 0

    def test_run_timing_sequence(self):
        """Run intervals should not overlap and be temporally ordered."""
        poke_times = pd.DataFrame(
            {
                "start_time": [1.0, 4.0, 7.0],
                "end_time": [2.0, 5.0, 8.0],
            }
        )

        result = get_run_intervals(poke_times)

        # Check non-overlapping
        for i in range(len(result) - 1):
            assert result.iloc[i]["end_time"] <= result.iloc[i + 1]["start_time"]

    def test_zero_duration_runs_handled(self):
        """Should handle case where pokes are back-to-back (zero duration run)."""
        poke_times = pd.DataFrame(
            {
                "start_time": [1.0, 2.0],
                "end_time": [2.0, 3.0],
            }
        )

        result = get_run_intervals(poke_times)

        assert len(result) == 1
        assert result.iloc[0]["duration"] == pytest.approx(0.0)


def _bandit_graph(well_positions: dict[int, tuple[float, float]]) -> nx.Graph:
    """The task's 10-node spider: wells 0-5, patch junctions 6-8, center 9.

    Wells missing from ``well_positions`` sit far away (y = 1000), so they are
    never nearest. Each patch junction lies between its two wells, nearer
    the first, so a junction is often the nearest node of all.
    """
    graph = nx.Graph()
    positions = {
        well: well_positions.get(well, (100.0 * well, 1000.0))
        for well in sorted(WELL_PATCH_MAPPING)
    }
    for junction, (well_a, well_b) in zip(
        (6, 7, 8), ((0, 1), (2, 3), (4, 5)), strict=True
    ):
        (xa, ya), (xb, yb) = positions[well_a], positions[well_b]
        positions[junction] = (xa + 0.4 * (xb - xa), ya + 0.4 * (yb - ya))
        graph.add_edge(well_a, junction)
        graph.add_edge(well_b, junction)
        graph.add_edge(junction, 9)
    positions[9] = (-50.0, -50.0)
    nx.set_node_attributes(graph, positions, "pos")
    return graph


class TestGetStartEndNodeID:
    """Test well node ID determination from position data."""

    def test_junction_is_never_a_well(self):
        """An animal standing on a patch junction maps to the nearest well."""
        track_graph = _bandit_graph({0: (0, 0), 1: (10, 0), 2: (20, 0)})
        assert track_graph.nodes[6]["pos"] == (4.0, 0.0)
        position_info = pd.DataFrame(
            {"projected_x_position": [4.0, 4.0], "projected_y_position": [0.0, 0.0]},
            index=[0.0, 1.0],
        )
        times = pd.DataFrame({"start_time": [0.0], "end_time": [1.0]})

        result = get_start_end_node_id(position_info, track_graph, times)

        assert result["from_well"][0] == 0
        assert result["to_well"][0] == 0

    def test_identifies_nearest_nodes(self):
        """Should identify nearest node based on projected position."""
        track_graph = _bandit_graph({0: (0, 0), 1: (10, 0), 2: (20, 0)})

        # Animal at position near node 1 with multiple time points
        position_info = pd.DataFrame(
            {
                "projected_x_position": [9.5, 9.6, 9.7],
                "projected_y_position": [0.0, 0.0, 0.0],
            },
            index=[0.0, 1.0, 2.0],
        )

        times = pd.DataFrame(
            {
                "start_time": [1.0],
                "end_time": [1.5],
            }
        )

        result = get_start_end_node_id(position_info, track_graph, times)

        # Should identify node 1 as nearest
        assert result["from_well"][0] == 1
        assert result["to_well"][0] == 1

    def test_different_start_and_end_nodes(self):
        """Should handle movement from one node to another."""
        track_graph = _bandit_graph({0: (0, 0), 1: (10, 0), 2: (20, 0)})

        # Animal moves from near node 0 to near node 2
        position_info = pd.DataFrame(
            {
                "projected_x_position": [0.5, 5.0, 10.0, 15.0, 19.5],
                "projected_y_position": [0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=[0.0, 1.0, 2.0, 3.0, 4.0],
        )

        times = pd.DataFrame(
            {
                "start_time": [0.0],
                "end_time": [4.0],
            }
        )

        result = get_start_end_node_id(position_info, track_graph, times)

        # Start near node 0, end near node 2
        assert result["from_well"][0] == 0
        assert result["to_well"][0] == 2

    def test_multiple_intervals(self):
        """Should handle multiple time intervals."""
        track_graph = _bandit_graph({0: (0, 0), 1: (10, 0), 2: (20, 0)})

        # More time points to avoid index errors
        position_info = pd.DataFrame(
            {
                "projected_x_position": [0.0, 5.0, 10.0, 15.0, 20.0],
                "projected_y_position": [0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=[0.0, 0.5, 1.0, 1.5, 2.0],
        )

        times = pd.DataFrame(
            {
                "start_time": [0.0, 1.0],
                "end_time": [0.8, 1.8],
            }
        )

        result = get_start_end_node_id(position_info, track_graph, times)

        # Two intervals
        assert len(result["from_well"]) == 2
        assert len(result["to_well"]) == 2

    def test_handles_2d_positions(self):
        """Should work with 2D position data."""
        track_graph = _bandit_graph({0: (0, 0), 1: (10, 0), 2: (20, 0)})
        track_graph.nodes[1]["pos"] = (10, 10)

        # More time points to avoid indexing issues
        position_info = pd.DataFrame(
            {
                "projected_x_position": [1.0, 5.0, 9.0, 12.0],
                "projected_y_position": [1.0, 5.0, 11.0, 10.0],
            },
            index=[0.0, 0.5, 1.0, 1.5],
        )

        times = pd.DataFrame(
            {
                "start_time": [0.0],
                "end_time": [1.2],
            }
        )

        result = get_start_end_node_id(position_info, track_graph, times)

        # Should identify nearest nodes in 2D
        assert result["from_well"][0] == 0  # Near (0,0)
        assert result["to_well"][0] == 1  # Near (10,10)

    def test_returns_correct_structure(self):
        """Should return dict with from_well and to_well arrays."""
        track_graph = _bandit_graph({0: (0, 0), 1: (10, 0), 2: (20, 0)})

        position_info = pd.DataFrame(
            {
                "projected_x_position": [0.0, 0.1, 0.2],
                "projected_y_position": [0.0, 0.0, 0.0],
            },
            index=[0.0, 0.5, 1.0],
        )

        times = pd.DataFrame(
            {
                "start_time": [0.0],
                "end_time": [0.8],
            }
        )

        result = get_start_end_node_id(position_info, track_graph, times)

        # Check structure
        assert "from_well" in result
        assert "to_well" in result
        assert isinstance(result["from_well"], np.ndarray)
        assert isinstance(result["to_well"], np.ndarray)

    def test_stationary_at_well(self):
        """Animal stationary at a well should have same from/to well."""
        track_graph = _bandit_graph({0: (0, 0), 1: (10, 0), 2: (20, 0)})

        # Stationary at node 1
        position_info = pd.DataFrame(
            {
                "projected_x_position": [10.0, 10.0, 10.0, 10.0],
                "projected_y_position": [0.0, 0.0, 0.0, 0.0],
            },
            index=[0.0, 1.0, 2.0, 3.0],
        )

        times = pd.DataFrame(
            {
                "start_time": [0.0],
                "end_time": [2.5],
            }
        )

        result = get_start_end_node_id(position_info, track_graph, times)

        # Should be at node 1 for both start and end
        assert result["from_well"][0] == 1
        assert result["to_well"][0] == 1
