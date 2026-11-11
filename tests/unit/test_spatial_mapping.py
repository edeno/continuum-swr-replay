"""Tests for analysis/replay/spatial_mapping.py.

This module is the canonical pattern for junction-bin exclusion. The fix that makes this work is the
``real_mask = ei >= 0`` filter at line 47/87. Pin every aspect:

- junction-bin MAPs are routed to the nearest real-arm bin (never NaN
  unless the input is NaN)
- all-junction-bin environment returns NaN for every input
- patch mapping respects TRACK_SEGMENT_TO_PATCH
- arm_to_well returns NaN for trunk arms (6/7/8) and leaves (0-5)
- well_to_arm_set respects degree-1 well property on the real graph
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay.spatial_mapping import (
    arm_to_well,
    map_estimate_to_arm,
    map_estimate_to_patch,
    well_to_arm_set,
)
from continuum_swr_replay.data_loaders.constants import TRACK_SEGMENT_TO_PATCH

# ── Test environment fixture ──


class _StubEnv:
    """Minimal stand-in for non_local_detector's Environment.

    The real Environment exposes ``place_bin_centers_nodes_df_`` with at
    least ``linear_position`` and ``edge_id`` columns. Two real bins per
    arm (linear_position 0..17, edge_ids 0,0,1,1,...,8,8) plus three
    junction bins at linear_position 100, 101, 102 with edge_id = -1.
    """

    def __init__(self):
        arm_lp = np.arange(18, dtype=float)
        arm_eid = np.repeat(np.arange(9), 2)
        junc_lp = np.array([100.0, 101.0, 102.0])
        junc_eid = np.array([-1, -1, -1])
        self.place_bin_centers_nodes_df_ = pd.DataFrame(
            {
                "linear_position": np.concatenate([arm_lp, junc_lp]),
                "edge_id": np.concatenate([arm_eid, junc_eid]),
            }
        )


class _AllJunctionEnv:
    """Pathological env where every place bin is a junction bin
    (edge_id = -1). map_estimate_to_arm / map_estimate_to_patch must
    return NaN — there is no real arm to fall back to."""

    def __init__(self):
        self.place_bin_centers_nodes_df_ = pd.DataFrame(
            {
                "linear_position": np.array([100.0, 101.0, 102.0]),
                "edge_id": np.array([-1, -1, -1]),
            }
        )


def _bandit_graph() -> nx.Graph:
    """Real bandit topology: 6 leaf arms (0..5) + 3 trunk arms (6..8)."""
    g = nx.Graph()
    for u, v, eid in (
        (0, 6, 0),
        (1, 6, 1),
        (2, 7, 2),
        (3, 7, 3),
        (4, 8, 4),
        (5, 8, 5),
        (6, 9, 6),
        (7, 9, 7),
        (8, 9, 8),
    ):
        g.add_edge(u, v, edge_id=eid)
    return g


# ── map_estimate_to_arm ──


class TestMapEstimateToArm:
    def test_arm_bin_position_returns_arm_id(self):
        """A MAP estimate that lands on a real arm bin returns that
        arm's edge_id (0..8)."""
        env = _StubEnv()
        # linear_position 0 → first bin of arm 0; 4 → first bin of arm 2.
        out = map_estimate_to_arm(np.array([0.0, 4.0, 16.0]), env)
        np.testing.assert_array_equal(out, [0.0, 2.0, 8.0])

    def test_junction_bin_position_routes_to_nearest_real_arm(self):
        """A MAP estimate that happens to fall at a
        junction-bin linear_position (edge_id = -1) must NOT return
        edge_id=-1 or NaN. The nearest-neighbor lookup excludes
        junction bins, so the result is the closest real arm."""
        env = _StubEnv()
        # linear_position 100 is the first junction bin; nearest real
        # bin is at lp=17 (last bin of arm 8).
        out = map_estimate_to_arm(np.array([100.0]), env)
        assert not np.isnan(out[0])
        assert out[0] == 8.0  # nearest real arm

    def test_nan_input_yields_nan_output(self):
        env = _StubEnv()
        out = map_estimate_to_arm(np.array([0.0, np.nan, 4.0]), env)
        assert out[0] == 0.0
        assert np.isnan(out[1])
        assert out[2] == 2.0

    def test_all_junction_env_returns_nan_for_every_estimate(self):
        """Pathological environment where every bin is a junction.
        There is no real arm to fall back to, so every output is NaN
        (not 0, not -1, not silently the last junction)."""
        env = _AllJunctionEnv()
        out = map_estimate_to_arm(np.array([0.0, 50.0, 200.0]), env)
        assert np.isnan(out).all()

    def test_empty_input_returns_empty_output(self):
        env = _StubEnv()
        out = map_estimate_to_arm(np.array([], dtype=float), env)
        assert out.shape == (0,)


# ── map_estimate_to_patch ──


class TestMapEstimateToPatch:
    @pytest.mark.parametrize(
        "linear_position,expected_patch",
        [
            (0.0, 1),  # arm 0 → patch 1
            (3.0, 1),  # arm 1 → patch 1
            (4.0, 2),  # arm 2 → patch 2
            (7.0, 2),  # arm 3 → patch 2
            (8.0, 3),  # arm 4 → patch 3
            (11.0, 3),  # arm 5 → patch 3
            (12.0, 1),  # arm 6 (trunk for patch 1) → patch 1
            (14.0, 2),  # arm 7 (trunk for patch 2) → patch 2
            (16.0, 3),  # arm 8 (trunk for patch 3) → patch 3
        ],
    )
    def test_each_arm_maps_to_correct_patch(self, linear_position, expected_patch):
        """Both leaf and trunk arms map to their patch via
        TRACK_SEGMENT_TO_PATCH."""
        env = _StubEnv()
        out = map_estimate_to_patch(np.array([linear_position]), env)
        assert int(out[0]) == expected_patch

    def test_junction_bin_routes_to_nearest_real_patch(self):
        env = _StubEnv()
        out = map_estimate_to_patch(np.array([100.0]), env)
        # nearest real bin is lp=17 (arm 8) which is patch 3
        assert int(out[0]) == 3

    def test_nan_input_yields_nan_output(self):
        env = _StubEnv()
        out = map_estimate_to_patch(np.array([np.nan, 0.0]), env)
        assert np.isnan(out[0])
        assert out[1] == 1

    def test_all_junction_env_returns_nan(self):
        env = _AllJunctionEnv()
        out = map_estimate_to_patch(np.array([0.0, 50.0]), env)
        assert np.isnan(out).all()

    def test_patch_values_match_constants(self):
        """The patch IDs returned must come from
        TRACK_SEGMENT_TO_PATCH — no hardcoded duplicates."""
        env = _StubEnv()
        # Sample one bin per arm.
        bin_positions = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]
        arms = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        out = map_estimate_to_patch(np.array(bin_positions), env)
        for arm, patch in zip(arms, out, strict=True):
            assert int(patch) == TRACK_SEGMENT_TO_PATCH[arm]


# ── arm_to_well ──


class TestArmToWell:
    @pytest.mark.parametrize(
        "leaf_arm,expected_well",
        [
            (0, 0),
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 4),
            (5, 5),
        ],
    )
    def test_leaf_arms_return_their_well(self, leaf_arm, expected_well):
        """Leaf arms 0-5 each terminate at well 0-5 respectively."""
        graph = _bandit_graph()
        assert arm_to_well(leaf_arm, graph) == float(expected_well)

    @pytest.mark.parametrize("trunk_arm", [6, 7, 8])
    def test_trunk_arms_return_nan(self, trunk_arm):
        """Trunk arms (patch junction ↔ center junction) connect two
        non-well nodes — no well to return."""
        graph = _bandit_graph()
        assert np.isnan(arm_to_well(trunk_arm, graph))

    def test_nan_input_returns_nan(self):
        graph = _bandit_graph()
        assert np.isnan(arm_to_well(np.nan, graph))

    def test_unknown_arm_id_returns_nan(self):
        """An arm_id that doesn't match any edge_id in the graph
        returns NaN rather than raising."""
        graph = _bandit_graph()
        assert np.isnan(arm_to_well(99, graph))

    def test_custom_well_set_overrides_default(self):
        """If the caller declares a different well set (e.g. for a
        non-bandit graph), arm_to_well respects it."""
        graph = _bandit_graph()
        # Treat node 9 (center junction) as a "well" — then trunk arm 6
        # terminates at it.
        assert arm_to_well(6, graph, well_ids={9}) == 9.0


# ── well_to_arm_set ──


class TestWellToArmSet:
    @pytest.mark.parametrize(
        "well,expected_arms",
        [
            (0, {0}),
            (1, {1}),
            (2, {2}),
            (3, {3}),
            (4, {4}),
            (5, {5}),
        ],
    )
    def test_each_well_has_exactly_one_arm_on_real_graph(self, well, expected_arms):
        """Each well is degree-1 on the real
        bandit graph, so well_to_arm_set returns a set of size 1.
        This is what makes is_prev / is_next / is_last_rewarded each
        mark exactly one of the nine arms per event."""
        graph = _bandit_graph()
        assert well_to_arm_set(well, graph) == expected_arms

    def test_nan_input_returns_empty_set(self):
        graph = _bandit_graph()
        assert well_to_arm_set(np.nan, graph) == set()

    def test_returns_size_one_for_real_bandit_graph(self):
        """Verify the degree-1 property: every well has exactly one
        adjacent arm. This is the structural reason
        ``well_to_arm_set(prev_trial_to_well)`` collapses to
        ``{is_current arm}`` in the predictor logic."""
        graph = _bandit_graph()
        for well in range(6):
            arms = well_to_arm_set(well, graph)
            assert len(arms) == 1, f"well {well} has {len(arms)} arms (expected 1)"
