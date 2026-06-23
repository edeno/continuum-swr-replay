"""Unit tests for the per-event trajectory rollup families."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.replay.trajectory import (
    ahead_behind_summary,
    brain_state_means,
    trajectory_dynamics,
    uncertainty_extra,
)


class _EnvStub:
    """Minimal env exposing the two attributes the rollups read."""

    def __init__(self, n_bins: int = 5, edge_cm: float = 10.0):
        graph = nx.Graph()
        graph.add_edges_from((i, i + 1) for i in range(n_bins - 1))
        nx.set_edge_attributes(graph, edge_cm, "distance")
        self.track_graph_with_bin_centers_edges_ = graph
        self.place_bin_centers_nodes_df_ = pd.DataFrame(
            {
                "node_id": np.arange(n_bins),
                "linear_position": np.arange(n_bins) * edge_cm,
            }
        )


def test_trajectory_dynamics_forward_and_outback():
    env = _EnvStub()
    # event 1: forward 0->10->20 ; event 2: out-and-back 0->20->0
    event_id = np.array([1, 1, 1, 2, 2, 2])
    map_estimate = np.array([0.0, 10.0, 20.0, 0.0, 20.0, 0.0])
    step_distance = np.array([0.0, 10.0, 10.0, 0.0, 20.0, 20.0])
    decoded_arm = np.array([0.0, 0.0, 1.0, 6.0, 8.0, 6.0])  # arms across patches
    map_speed = np.array([5.0, 5.0, 5.0, 9.0, 9.0, 9.0])
    cont = np.array([0.9, 0.9, 0.1, 0.9, 0.9, 0.9])
    frag = 1.0 - cont
    baseline = np.full(6, 50.0)

    ids, out = trajectory_dynamics(
        event_id=event_id, map_estimate=map_estimate, step_distance=step_distance,
        map_speed=map_speed, decoded_arm=decoded_arm, continuous_prob=cont,
        fragmented_prob=frag, run_speed_baseline=baseline, env=env, place_bin_size=2.0,
    )
    np.testing.assert_array_equal(ids, [1, 2])

    # event 1: straight forward
    np.testing.assert_allclose(out["path_length"], [20.0, 40.0])
    np.testing.assert_allclose(out["net_displacement"], [20.0, 0.0])
    np.testing.assert_allclose(out["start_position"], [0.0, 0.0])
    np.testing.assert_allclose(out["end_position"], [20.0, 0.0])
    np.testing.assert_array_equal(out["start_arm"], [0, 6])
    np.testing.assert_array_equal(out["end_arm"], [1, 6])
    # tortuosity = path/net ; event 2 has net 0 -> NaN
    np.testing.assert_allclose(out["tortuosity"][0], 1.0)
    assert np.isnan(out["tortuosity"][1])
    # event 2 reverses once (out then back)
    np.testing.assert_array_equal(out["n_reversals"], [0, 1])
    # spatial extent: event1 spans 0..20 (20cm), event2 spans 0..20 (20cm)
    np.testing.assert_allclose(out["spatial_extent"], [20.0, 20.0])
    # patches: arms 0,1 -> patch 1 only; arms 6,8 -> patch1+patch3
    np.testing.assert_array_equal(out["n_patches_visited"], [1, 2])
    np.testing.assert_array_equal(out["crosses_trunk"], [False, True])
    # compression factor uses median map_speed / baseline
    np.testing.assert_allclose(out["compression_factor"], [5.0 / 50.0, 9.0 / 50.0])
    # terminates_at_well: end_arm 1 is a well (0-5); end_arm 6 is a trunk -> False
    np.testing.assert_array_equal(out["terminates_at_well"], [True, False])


def test_brain_state_means_and_fraction():
    event_id = np.array([1, 1, 2, 2])
    nonlocal_prob = np.array([0.9, 0.5, 0.95, 0.95])
    state_probs = {"local_prob_mean": np.array([0.1, 0.3, 0.0, 0.2])}
    extra = {"ripple_power": np.array([10.0, 20.0, 5.0, 7.0])}

    ids, out = brain_state_means(
        event_id=event_id, nonlocal_prob=nonlocal_prob,
        state_probs=state_probs, extra_means=extra, non_local_threshold=0.8,
    )
    np.testing.assert_array_equal(ids, [1, 2])
    np.testing.assert_allclose(out["local_prob_mean"], [0.2, 0.1])
    np.testing.assert_allclose(out["ripple_power"], [15.0, 6.0])
    # event1: 1 of 2 bins > 0.8 -> 0.5 ; event2: both -> 1.0
    np.testing.assert_allclose(out["nonlocal_fraction"], [0.5, 1.0])


def test_brain_state_means_skips_nan_signal_bins():
    """A NaN signal bin (LFP coverage gap) is excluded from the event mean."""
    event_id = np.array([1, 1, 2, 2])
    extra = {"ripple_power": np.array([10.0, np.nan, np.nan, np.nan])}
    ids, out = brain_state_means(
        event_id=event_id, state_probs={}, extra_means=extra,
    )
    np.testing.assert_array_equal(ids, [1, 2])
    # event1: mean of the one finite bin (10.0); event2: all NaN -> NaN
    np.testing.assert_allclose(out["ripple_power"][0], 10.0)
    assert np.isnan(out["ripple_power"][1])


def test_ahead_behind_direction_and_reach():
    event_id = np.array([1, 1, 1, 2, 2, 2])
    # event1 outbound (increasing), event2 inbound (decreasing)
    ab = np.array([0.0, 10.0, 20.0, 20.0, 10.0, 0.0])
    time = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
    decoded_arm = np.array([0.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    current_arm = np.array([0.0, 0.0, 0.0, 2.0, 2.0, 2.0])

    ids, out = ahead_behind_summary(
        event_id=event_id, ahead_behind_distance=ab,
        decoded_arm=decoded_arm, current_arm=current_arm, time=time,
    )
    np.testing.assert_array_equal(ids, [1, 2])
    np.testing.assert_allclose(out["ab_slope"], [10.0, -10.0])
    np.testing.assert_array_equal(out["is_outbound"], [True, False])
    np.testing.assert_allclose(out["max_ahead"], [20.0, 20.0])
    np.testing.assert_allclose(out["max_behind"], [0.0, 0.0])
    # event1: decoded != current at 2/3 bins ; event2: all equal -> 0
    np.testing.assert_allclose(out["remote_fraction"], [2 / 3, 0.0])


def test_uncertainty_extra_multimodal_and_hpd():
    # arms 0-1-2 chain: 0 and 2 are NOT adjacent
    adjacency = {0: {1}, 1: {0, 2}, 2: {1}}
    event_id = np.array([1, 1, 1])
    hpd = np.array([10.0, 20.0, 30.0])
    arm_posterior = np.array(
        [
            [0.5, 0.0, 0.5],  # HPD set {0,2} -> non-adjacent -> multimodal
            [0.6, 0.4, 0.0],  # HPD set {0,1} -> adjacent -> not
            [1.0, 0.0, 0.0],  # HPD set {0} -> not
        ]
    )
    ids, out = uncertainty_extra(
        event_id=event_id, hpd=hpd, arm_posterior=arm_posterior,
        arm_adjacency=adjacency, coverage=0.95,
    )
    np.testing.assert_array_equal(ids, [1])
    np.testing.assert_allclose(out["hpd_coverage_cm"], [20.0])
    np.testing.assert_allclose(out["multimodal_fraction"], [1 / 3])
