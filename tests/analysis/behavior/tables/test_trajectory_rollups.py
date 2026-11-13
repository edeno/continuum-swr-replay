"""Unit tests for the per-event trajectory rollup families."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.behavior.tables.trajectory import (
    _REQUIRED_DOT_COLUMNS,
    _SIGNAL_COLUMNS,
    _count_ab_reversals,
    ahead_behind_summary,
    brain_state_means,
    compute_trajectory_rollups,
    trajectory_dynamics,
    uncertainty_extra,
)

_NONLOCAL_STATE_COLS = {
    "local_prob_mean": "Local_prob",
    "nospike_prob_mean": "No-Spike_prob",
    "continuous_prob_mean": "Non-Local_Continuous_prob",
    "fragmented_prob_mean": "Non-Local_Fragmented_prob",
}


def test_count_ab_reversals():
    """Reversals on signed ahead/behind distance: branch-safe + hysteresis."""
    # Fewer than 3 finite points -> 0
    assert _count_ab_reversals(np.array([0.0, 5.0]), delta=0.0) == 0
    # Monotone outbound (would cross branches, but ab is geodesic) -> 0
    assert _count_ab_reversals(np.array([0.0, 10.0, 20.0, 35.0]), delta=2.0) == 0
    # Pass-through the animal (behind -> ahead) is monotone, not a turn -> 0
    assert _count_ab_reversals(np.array([-6.0, -2.0, 3.0, 7.0]), delta=2.0) == 0
    # Out-and-back -> 1
    assert _count_ab_reversals(np.array([0.0, 20.0, 0.0]), delta=2.0) == 1
    # Up, down, up -> 2
    assert _count_ab_reversals(np.array([0.0, 8.0, 0.0, 8.0]), delta=2.0) == 2
    # Hysteresis: a sub-delta dip is not a reversal ...
    assert _count_ab_reversals(np.array([0.0, 5.0, 4.0, 9.0]), delta=2.0) == 0
    # ... but at delta=0 the same dip is two strict sign changes
    assert _count_ab_reversals(np.array([0.0, 5.0, 4.0, 9.0]), delta=0.0) == 2
    # Flat steps are ignored even at delta=0
    assert _count_ab_reversals(np.array([0.0, 1.0, 1.0, 0.0]), delta=0.0) == 1
    # NaN bins are dropped before counting: [0, 5, 0] -> 1
    assert (
        _count_ab_reversals(np.array([0.0, np.nan, 5.0, np.nan, 0.0]), delta=1.0) == 1
    )


def test_terminates_at_well_proximity():
    """terminates_at_well needs the endpoint near a well node, not just a leaf arm."""
    env = _EnvStub()  # chain nodes 0..4 (linear 0..40); terminals 0 and 4 are wells
    # ev1 ends AT terminal node 4 on a leaf arm -> True
    # ev2 ends mid-chain (node 2) on a leaf arm -> False (well_radius = 4*2 = 8cm)
    event_id = np.array([1, 1, 2, 2])
    map_estimate = np.array([0.0, 40.0, 0.0, 20.0])
    decoded_arm = np.array([0.0, 0.0, 1.0, 1.0])  # leaf arms
    ahead_behind = np.array([0.0, 40.0, 0.0, 20.0])
    zeros = np.zeros(4)
    cont = np.full(4, 0.9)
    _, out = trajectory_dynamics(
        event_id=event_id,
        map_estimate=map_estimate,
        step_distance=zeros,
        map_speed=np.full(4, 5.0),
        decoded_arm=decoded_arm,
        continuous_prob=cont,
        fragmented_prob=1.0 - cont,
        run_speed_baseline=np.full(4, 50.0),
        ahead_behind_distance=ahead_behind,
        env=env,
        place_bin_size=2.0,
    )
    np.testing.assert_array_equal(out["terminates_at_well"], [True, False])


def test_is_multimodal_connectivity():
    """Multimodal = HPD arm set spans >1 connected component (not all-pairs)."""
    from continuum_swr_replay.analysis.behavior.tables.trajectory import _is_multimodal

    # chain 0-1-2 ; arm 3 isolated
    adjacency = {0: {1}, 1: {0, 2}, 2: {1}, 3: set()}
    assert _is_multimodal(np.array([0]), adjacency) is False  # single arm
    assert _is_multimodal(np.array([0, 1]), adjacency) is False  # adjacent
    # KEY FIX: connected chain 0-1-2 is unimodal (old all-pairs called it multimodal)
    assert _is_multimodal(np.array([0, 1, 2]), adjacency) is False
    # 0 and 2 without the linking arm 1 -> two components -> multimodal
    assert _is_multimodal(np.array([0, 2]), adjacency) is True
    # isolated arm 3 alongside the chain -> multimodal
    assert _is_multimodal(np.array([0, 1, 3]), adjacency) is True


def test_compute_rollups_fails_loud_on_missing_column():
    """A data_over_time schema that drifted from the rollups fails clearly."""
    cols = {
        c: np.zeros(2)
        for c in (
            *_REQUIRED_DOT_COLUMNS,
            *_SIGNAL_COLUMNS.values(),
            *_NONLOCAL_STATE_COLS.values(),
        )
    }
    nl = pd.DataFrame(cols).drop(columns=["hpd"])  # simulate a builder dropping hpd
    with pytest.raises(ValueError, match="hpd"):
        compute_trajectory_rollups(
            nl=nl,
            event_id=np.array([1, 1]),
            env=None,
            track_graph=None,
            state_prob_cols=_NONLOCAL_STATE_COLS,
            continuous_col="Non-Local_Continuous_prob",
            fragmented_col="Non-Local_Fragmented_prob",
            place_bin_size=2.0,
        )


class _EnvStub:
    """Minimal env exposing the two attributes the rollups read."""

    def __init__(self, n_bins: int = 5, edge_cm: float = 10.0):
        graph = nx.Graph()
        graph.add_edges_from((i, i + 1) for i in range(n_bins - 1))
        nx.set_edge_attributes(graph, edge_cm, "distance")
        self.track_graph_with_bin_centers_edges_ = graph
        # Mirrors how ``Environment`` populates this at fit time
        # (``get_track_grid``): all-pairs Dijkstra over the bin-centre graph.
        self.distance_between_nodes_ = dict(
            nx.all_pairs_dijkstra_path_length(graph, weight="distance")
        )
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
    # event 1: monotone outbound; event 2: out-and-back (signed ahead/behind)
    ahead_behind = np.array([0.0, 10.0, 20.0, 0.0, 20.0, 0.0])

    ids, out = trajectory_dynamics(
        event_id=event_id,
        map_estimate=map_estimate,
        step_distance=step_distance,
        map_speed=map_speed,
        decoded_arm=decoded_arm,
        continuous_prob=cont,
        fragmented_prob=frag,
        run_speed_baseline=baseline,
        ahead_behind_distance=ahead_behind,
        env=env,
        place_bin_size=2.0,
    )
    np.testing.assert_array_equal(ids, [1, 2])

    # event 1: straight forward
    np.testing.assert_allclose(out["path_length_cm"], [20.0, 40.0])
    np.testing.assert_allclose(out["net_displacement_cm"], [20.0, 0.0])
    np.testing.assert_allclose(out["replay_start_position_cm"], [0.0, 0.0])
    np.testing.assert_allclose(out["replay_end_position_cm"], [20.0, 0.0])
    np.testing.assert_array_equal(out["start_arm"], [0, 6])
    np.testing.assert_array_equal(out["end_arm"], [1, 6])
    # tortuosity = path/net ; event 2 has net 0 -> NaN
    np.testing.assert_allclose(out["tortuosity"][0], 1.0)
    assert np.isnan(out["tortuosity"][1])
    # event 2 reverses once (out then back)
    np.testing.assert_array_equal(out["n_reversals"], [0, 1])
    # spatial extent: event1 spans 0..20 (20cm), event2 spans 0..20 (20cm)
    np.testing.assert_allclose(out["spatial_extent_cm"], [20.0, 20.0])
    # patches: arms 0,1 -> patch 1 only; arms 6,8 -> patch1+patch3
    np.testing.assert_array_equal(out["n_patches_visited"], [1, 2])
    # pause_fraction: step_distance <= place_bin_size (2.0); ev1 [0,10,10]->1/3
    np.testing.assert_allclose(out["pause_fraction"], [1 / 3, 1 / 3])
    np.testing.assert_array_equal(out["crosses_trunk"], [False, True])
    # compression factor uses median map_speed / baseline
    np.testing.assert_allclose(out["compression_factor"], [5.0 / 50.0, 9.0 / 50.0])
    # terminates_at_well now needs endpoint proximity to a well node, not just a
    # leaf arm: ev1 ends mid-chain (node 2, 20cm from either terminal) and ev2 at
    # node 0 but on a trunk arm -> both False. (See test_terminates_at_well_proximity.)
    np.testing.assert_array_equal(out["terminates_at_well"], [False, False])


def test_brain_state_means_and_fraction():
    event_id = np.array([1, 1, 2, 2])
    nonlocal_prob = np.array([0.9, 0.5, 0.95, 0.95])
    state_probs = {"local_prob_mean": np.array([0.1, 0.3, 0.0, 0.2])}
    extra = {"ripple_power": np.array([10.0, 20.0, 5.0, 7.0])}

    ids, out = brain_state_means(
        event_id=event_id,
        nonlocal_prob=nonlocal_prob,
        state_probs=state_probs,
        extra_means=extra,
        non_local_threshold=0.8,
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
        event_id=event_id,
        state_probs={},
        extra_means=extra,
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
        event_id=event_id,
        ahead_behind_distance=ab,
        decoded_arm=decoded_arm,
        current_arm=current_arm,
        time=time,
    )
    np.testing.assert_array_equal(ids, [1, 2])
    np.testing.assert_allclose(out["ab_slope_cm_s"], [10.0, -10.0])
    np.testing.assert_array_equal(out["is_outbound"], [True, False])
    np.testing.assert_allclose(out["max_ahead_cm"], [20.0, 20.0])
    np.testing.assert_allclose(out["max_behind_cm"], [0.0, 0.0])
    # event1: decoded != current at 2/3 bins ; event2: all equal -> 0
    np.testing.assert_allclose(out["remote_fraction"], [2 / 3, 0.0])


def test_ahead_behind_is_outbound_nan_when_slope_undefined():
    """A single-bin event has undefined slope -> is_outbound NaN, not False."""
    ids, out = ahead_behind_summary(
        event_id=np.array([1]),
        ahead_behind_distance=np.array([3.0]),
        decoded_arm=np.array([1.0]),
        current_arm=np.array([0.0]),
        time=np.array([0.0]),
    )
    assert np.isnan(out["ab_slope_cm_s"][0])
    assert np.isnan(out["is_outbound"][0])


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
        event_id=event_id,
        hpd=hpd,
        arm_posterior=arm_posterior,
        arm_adjacency=adjacency,
        coverage=0.95,
    )
    np.testing.assert_array_equal(ids, [1])
    np.testing.assert_allclose(out["hpd_coverage_cm"], [20.0])
    np.testing.assert_allclose(out["multimodal_fraction"], [1 / 3])


def test_orchestrator_nospike_fraction_modal_state():
    """nospike_fraction = fraction of bins whose modal state is No-Spike."""
    env = _EnvStub()
    n = 3
    # bin 1 is No-Spike-modal; bins 0,2 are Local-modal
    nl = pd.DataFrame(
        {
            "map_estimate": [0.0, 10.0, 20.0],
            "step_distance": [0.0, 10.0, 10.0],
            "map_speed": [5.0, 5.0, 5.0],
            "decoded_arm": [0.0, 0.0, 1.0],
            "run_speed_baseline": [50.0] * n,
            "ahead_behind_distance": [1.0, 2.0, 3.0],
            "hpd": [10.0, 10.0, 10.0],
            "track_segment_id": [0.0, 0.0, 1.0],
            "SWR_zscore": [0.0] * n,
            "MUA_zscore": [0.0] * n,
            "theta_power_zscore": [0.0] * n,
            "Local_prob": [0.8, 0.05, 0.8],
            "No-Spike_prob": [0.1, 0.9, 0.1],
            "Non-Local_Continuous_prob": [0.05, 0.03, 0.05],
            "Non-Local_Fragmented_prob": [0.05, 0.02, 0.05],
        }
    )
    out = compute_trajectory_rollups(
        nl=nl,
        event_id=np.array([1, 1, 1]),
        env=env,
        track_graph=env.track_graph_with_bin_centers_edges_,
        state_prob_cols=_NONLOCAL_STATE_COLS,
        continuous_col="Non-Local_Continuous_prob",
        fragmented_col="Non-Local_Fragmented_prob",
        place_bin_size=2.0,
        nonlocal_prob=np.array([0.1, 0.05, 0.1]),
    )
    np.testing.assert_allclose(out["nospike_fraction"], [1 / 3])
    np.testing.assert_allclose(out["nospike_prob_mean"], [(0.1 + 0.9 + 0.1) / 3])
    # arm_posterior not supplied -> multimodal NaN, hpd_coverage from hpd column
    assert np.isnan(out["multimodal_fraction"][0])
    np.testing.assert_allclose(out["hpd_coverage_cm"], [10.0])


_CONTFRAG_STATE_COLS = {
    "continuous_prob_mean": "Continuous_prob",
    "fragmented_prob_mean": "Fragmented_prob",
}


def test_orchestrator_contfrag_2state_omits_nonlocal_only_columns():
    """ContFrag mapping (no nonlocal_prob, 2 states) must not emit 4-state cols."""
    env = _EnvStub()
    n = 3
    nl = pd.DataFrame(
        {
            "map_estimate": [0.0, 10.0, 20.0],
            "step_distance": [0.0, 10.0, 10.0],
            "map_speed": [5.0, 5.0, 5.0],
            "decoded_arm": [0.0, 0.0, 1.0],
            "run_speed_baseline": [50.0] * n,
            "ahead_behind_distance": [1.0, 2.0, 3.0],
            "hpd": [10.0] * n,
            "track_segment_id": [0.0, 0.0, 1.0],
            "SWR_zscore": [0.0] * n,
            "MUA_zscore": [0.0] * n,
            "theta_power_zscore": [0.0] * n,
            "Continuous_prob": [0.7, 0.6, 0.8],
            "Fragmented_prob": [0.3, 0.4, 0.2],
        }
    )
    out = compute_trajectory_rollups(
        nl=nl,
        event_id=np.array([1, 1, 1]),
        env=env,
        track_graph=env.track_graph_with_bin_centers_edges_,
        state_prob_cols=_CONTFRAG_STATE_COLS,
        continuous_col="Continuous_prob",
        fragmented_col="Fragmented_prob",
        place_bin_size=2.0,
        nonlocal_prob=None,  # ContFrag: both states non-local
    )
    # 4-state-only columns must be absent
    for col in (
        "nonlocal_fraction",
        "nospike_fraction",
        "local_prob_mean",
        "nospike_prob_mean",
    ):
        assert col not in out, f"{col} must not be produced for ContFrag"
    # continuous_prob_mean is produced and correct
    np.testing.assert_allclose(out["continuous_prob_mean"], [(0.7 + 0.6 + 0.8) / 3])
    # shared trajectory columns still present
    assert "path_length_cm" in out and "multimodal_fraction" in out
