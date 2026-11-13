"""Regression test for ``get_map_speed`` placeholder bleed.

When a posterior row is all-NaN the function uses MAP index 0 as a
placeholder, then computes forward-difference graph distance through
that placeholder, then masks only the invalid row. The masked row
zeros out, but the rows immediately *before* and *after* the NaN
gap inherited the placeholder's contribution: e.g. valid_node=1,
NaN, valid_node=1 used to return ``[10., NaN, 10.]`` even though
the two valid MAP positions are identical and the apparent motion
came entirely from node 0 being used as a stand-in.

Fix: invalidate every step whose endpoints touched a placeholder row,
not just the placeholder row itself.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from continuum_swr_replay.analysis.decode_quality.posterior_analysis import (
    get_map_speed,
)


def _two_node_graph() -> nx.Graph:
    g = nx.Graph()
    g.add_node(0, pos=(0.0, 0.0))
    g.add_node(1, pos=(10.0, 0.0))
    g.add_edge(0, 1, distance=10.0)
    g.add_edge(0, 0, distance=0.0)
    g.add_edge(1, 1, distance=0.0)
    return g


class TestGetMapSpeedPlaceholderBleed:
    def test_neighbors_of_all_nan_row_do_not_inherit_placeholder_motion(self):
        # Two place bins → nodes 0 and 1.
        place_bin_center_ind_to_node = np.array([0, 1])
        # Three timepoints: peak at bin 1, all-NaN, peak at bin 1.
        # All-NaN row uses MAP index 0 as placeholder ⇒ apparent
        # motion from node 1 → 0 → 1 ⇒ speed ~ 2 × 10 / dt without
        # the fix.
        posterior = np.array(
            [
                [0.1, 0.9],  # MAP = bin 1 (node 1)
                [np.nan, np.nan],
                [0.1, 0.9],  # MAP = bin 1 (node 1)
            ]
        )
        speed = get_map_speed(
            posterior,
            track_graph_with_bin_centers_edges=_two_node_graph(),
            place_bin_center_ind_to_node=place_bin_center_ind_to_node,
            sampling_frequency=1.0,
            smooth_sigma=1e-6,  # tiny, effectively no smoothing
        )
        # Both valid MAPs are at the same node; without contamination
        # we know nothing about motion across the NaN gap, so every
        # row is honestly NaN. Crucially: no row carries the
        # placeholder-induced 10 cm/s any more.
        assert np.isnan(speed[0]), f"row 0 leaked placeholder motion: {speed}"
        assert np.isnan(speed[1])
        assert np.isnan(speed[2]), f"row 2 leaked placeholder motion: {speed}"

    def test_step_into_nan_does_not_resurrect_via_interpolation(self):
        """User repro: MAP nodes ``0 → 1 → NaN → 1 → 1`` used to
        return ``[10.0, 6.6667, NaN, 0.0, 0.0]`` because the
        interpolate-smooth round trip refilled the row whose
        forward-difference touched the all-NaN frame. The post-fix
        flow re-applies the pre-interpolation NaN mask, so the
        contaminated row stays NaN."""
        place_bin_center_ind_to_node = np.array([0, 1])
        # Five timepoints. Index 2 is all-NaN (placeholder forces
        # ``has_valid_data[2] == False``); the surrounding rows
        # decode to nodes 0, 1, 1, 1.
        posterior = np.array(
            [
                [0.9, 0.1],  # node 0
                [0.1, 0.9],  # node 1
                [np.nan, np.nan],
                [0.1, 0.9],  # node 1
                [0.1, 0.9],  # node 1
            ]
        )
        speed = get_map_speed(
            posterior,
            track_graph_with_bin_centers_edges=_two_node_graph(),
            place_bin_center_ind_to_node=place_bin_center_ind_to_node,
            sampling_frequency=1.0,
            smooth_sigma=1e-6,
        )
        # The forward-diff at index 1 destination is the all-NaN
        # frame; that step must remain NaN. The forward-diff at
        # index 2 originates from the NaN frame; same.
        assert np.isnan(speed[1]), (
            f"speed[1] resurrected via interpolation: {speed.tolist()}"
        )
        assert np.isnan(speed[2]), (
            f"speed[2] (NaN frame itself) leaked: {speed.tolist()}"
        )
        # Indices 0 (step 0→1, valid) and 3, 4 (steps 3→4 and the
        # last-bin reuse) carry their honest values.
        assert speed[0] == 10.0  # 0 → 1: 10 cm
        assert speed[3] == 0.0  # 1 → 1
        assert speed[4] == 0.0  # backward-difference reuse of speed[3]

    def test_pure_valid_motion_still_computes_correctly(self):
        # Sanity check: when there are no NaN rows, motion is preserved.
        place_bin_center_ind_to_node = np.array([0, 1])
        posterior = np.array(
            [
                [0.9, 0.1],  # node 0
                [0.1, 0.9],  # node 1
                [0.9, 0.1],  # node 0
            ]
        )
        speed = get_map_speed(
            posterior,
            track_graph_with_bin_centers_edges=_two_node_graph(),
            place_bin_center_ind_to_node=place_bin_center_ind_to_node,
            sampling_frequency=1.0,
            smooth_sigma=1e-6,  # tiny, effectively no smoothing
        )
        # Each step covers 10 cm at 1 Hz ⇒ 10 cm/s.
        np.testing.assert_allclose(speed, np.array([10.0, 10.0, 10.0]))
