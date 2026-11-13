"""Cross-scenario parity: shared trajectory columns match across both tables."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.contfrag.tables import (
    _CONTFRAG_TRAJECTORY_COLUMNS,
)
from continuum_swr_replay.analysis.contfrag.tables import (
    build_contfrag_event_arm_table as _build_contfrag_event_arm_table,
)
from continuum_swr_replay.analysis.contfrag.tables import (
    build_contfrag_event_table as _build_contfrag_event_table,
)
from continuum_swr_replay.analysis.event_tables._common import (
    _EVENT_TABLE_COLUMNS,
)
from continuum_swr_replay.analysis.event_tables.arm_patch_tables import (
    build_event_arm_table as _build_event_arm_table,
)
from continuum_swr_replay.analysis.event_tables.event_table import (
    build_event_table as _build_event_table,
)
from continuum_swr_replay.analysis.event_tables.trajectory import (
    NONLOCAL_ONLY_COLUMNS,
    SHARED_TRAJECTORY_COLUMNS,
)
from continuum_swr_replay.analysis.replay_models.model_specs import (
    RECENCY_SPLINE_COLUMNS,
)
from tests.support._port_experience_fixtures import port_experiences_from_trials


def _experiences(trials: pd.DataFrame) -> pd.DataFrame:
    return port_experiences_from_trials(trials, prepend_initial=True)


def build_event_table(dot, env, trials, graph, **kwargs):
    return _build_event_table(dot, env, trials, _experiences(trials), graph, **kwargs)


def build_event_arm_table(event_table, dot, env, trials, graph, **kwargs):
    return _build_event_arm_table(
        event_table, dot, env, trials, _experiences(trials), graph, **kwargs
    )


def build_contfrag_event_table(dot, env, trials, graph, **kwargs):
    return _build_contfrag_event_table(
        dot, env, trials, _experiences(trials), graph, **kwargs
    )


def build_contfrag_event_arm_table(event_table, dot, env, trials, graph, **kwargs):
    return _build_contfrag_event_arm_table(
        event_table, dot, env, trials, _experiences(trials), graph, **kwargs
    )


class _EnvStub:
    class _Df:
        _arm_lp = np.arange(18, dtype=float)
        _arm_eid = np.repeat(np.arange(9), 2).astype(int)
        d = {
            "linear_position": np.concatenate([_arm_lp, [100.0, 101.0, 102.0]]),
            "edge_id": np.concatenate([_arm_eid, [-1, -1, -1]]).astype(int),
            "node_id": np.arange(21),
        }

        def __getitem__(self, k):
            return self.d[k]

    @property
    def place_bin_centers_nodes_df_(self):
        return pd.DataFrame(self._Df.d)

    @property
    def track_graph_with_bin_centers_edges_(self):
        lp = self._Df.d["linear_position"]
        g = nx.Graph()
        for i in range(len(lp) - 1):
            g.add_edge(i, i + 1, distance=abs(lp[i + 1] - lp[i]))
        return g

    @property
    def distance_between_nodes_(self):
        # Mirrors how ``Environment`` populates this at fit time
        # (``get_track_grid``): all-pairs Dijkstra over the bin-centre graph.
        return dict(
            nx.all_pairs_dijkstra_path_length(
                self.track_graph_with_bin_centers_edges_, weight="distance"
            )
        )


def _bandit_graph() -> nx.Graph:
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
        g.add_edge(u, v, edge_id=eid, distance=1.0)
    return g


def _trials() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "start_time",
            "end_time",
            "to_well",
            "from_well",
            "from_patch",
            "to_patch",
            "is_reward",
            "is_patch_change",
        ]
    )


def _completed_trial() -> pd.DataFrame:
    """One fully known visit before the shared event window."""
    return pd.DataFrame(
        {
            "movement_start_time": [0.5],
            "destination_first_poke_in_time": [1.0],
            "end_time": [1.4],
            "outcome_known_time": [1.2],
            "from_well": [0],
            "to_well": [1],
            "from_patch": [1],
            "to_patch": [1],
            "is_reward": [True],
            "is_patch_change": [False],
        }
    )


_AUGMENTED = {
    "map_speed": 5.0,
    "step_distance": 1.0,
    "ahead_behind_distance": 3.0,
    "hpd": 12.0,
    "run_speed_baseline": 50.0,
    "ripple_power": 20.0,
    "SWR_zscore": 1.5,
    "mua_rate": 100.0,
    "MUA_zscore": 2.0,
    "theta_power": 0.3,
    "theta_power_zscore": 0.5,
}


def _nonlocal_dot() -> pd.DataFrame:
    idx = pd.Index(np.linspace(0, 10, 100), name="time")
    dot = pd.DataFrame(
        {
            "animal": "chimi",
            "day": 20200212,
            "epoch": "02_r1",
            "non_local_event": 0,
            "SWR_event": 0,
            "MUA_event": 0,
            "Non-Local_Continuous_prob": 0.0,
            "Non-Local_Fragmented_prob": 0.0,
            "Local_prob": 0.1,
            "No-Spike_prob": 0.0,
            "map_estimate": np.nan,
            "head_speed": 10.0,
            "track_segment_id": np.nan,
        },
        index=idx,
    )
    rows = dot.index[20:30]
    dot.loc[rows, "non_local_event"] = 1
    dot.loc[rows, "SWR_event"] = 1
    dot.loc[rows, "Non-Local_Continuous_prob"] = 0.9
    dot.loc[rows, "head_speed"] = 1.0
    dot.loc[rows, "map_estimate"] = 2.0
    dot.loc[rows, "track_segment_id"] = 1
    for col, val in _AUGMENTED.items():
        dot[col] = val
    return dot


def _contfrag_dot() -> pd.DataFrame:
    idx = pd.Index(np.linspace(0, 10, 100), name="time")
    dot = pd.DataFrame(
        {
            "animal": "chimi",
            "day": 20200212,
            "epoch": "02_r1",
            "SWR_event": 0,
            "MUA_event": 0,
            "Continuous_prob": 0.5,
            "Fragmented_prob": 0.5,
            "map_estimate": np.nan,
            "head_speed": 10.0,
            "track_segment_id": np.nan,
        },
        index=idx,
    )
    rows = dot.index[20:30]
    dot.loc[rows, "SWR_event"] = 1
    dot.loc[rows, "head_speed"] = 1.0
    dot.loc[rows, "map_estimate"] = 2.0
    dot.loc[rows, "track_segment_id"] = 1
    for col, val in _AUGMENTED.items():
        dot[col] = val
    return dot


def test_shared_columns_in_both_table_schemas():
    nl_cols = set(_EVENT_TABLE_COLUMNS)
    for col in SHARED_TRAJECTORY_COLUMNS:
        assert col in nl_cols, f"shared col {col} missing from Non-Local schema"
    for col in _CONTFRAG_TRAJECTORY_COLUMNS:
        assert col in nl_cols  # ContFrag's shared subset is a subset of Non-Local
    # Non-Local-only columns must NOT be promised by the ContFrag schema set.
    assert set(NONLOCAL_ONLY_COLUMNS).isdisjoint(set(_CONTFRAG_TRAJECTORY_COLUMNS))


def test_contfrag_wired_path_and_dtype_parity():
    env = _EnvStub()
    nl = build_event_table(_nonlocal_dot(), env, _trials(), _bandit_graph())
    cf = build_contfrag_event_table(_contfrag_dot(), env, _trials(), _bandit_graph())

    assert len(nl) == 1 and len(cf) == 1
    # shared (non-state-mean) columns present in both with matching dtype
    for col in _CONTFRAG_TRAJECTORY_COLUMNS:
        assert col in cf.columns and col in nl.columns
        assert nl[col].dtype == cf[col].dtype, (
            f"{col}: {nl[col].dtype} (NL) != {cf[col].dtype} (CF)"
        )
    # wiring carries real values through identically (same augmented constants)
    np.testing.assert_allclose(cf["ripple_power_z"], 1.5)  # from SWR_zscore
    np.testing.assert_allclose(nl["ripple_power_z"], 1.5)
    np.testing.assert_allclose(cf["compression_factor"], 0.1)
    np.testing.assert_allclose(nl["compression_factor"], 0.1)


def test_candidate_arm_covariates_are_identical_across_event_sources():
    """Canonical names have one derivation in Replay and ContFrag.

    Response counts remain source-specific; this test pins the complete shared
    predictor block on identical event windows and behavioral history.
    """
    env = _EnvStub()
    graph = _bandit_graph()
    trials = _completed_trial()
    nonlocal_dot = _nonlocal_dot()
    contfrag_dot = _contfrag_dot()
    nonlocal_events = build_event_table(nonlocal_dot, env, trials, graph)
    contfrag_events = build_contfrag_event_table(
        contfrag_dot,
        env,
        trials,
        graph,
        event_source="SWR",
    )
    nonlocal_arms = build_event_arm_table(
        nonlocal_events,
        nonlocal_dot,
        env,
        trials,
        graph,
    ).sort_values("candidate_arm")
    contfrag_arms = build_contfrag_event_arm_table(
        contfrag_events,
        contfrag_dot,
        env,
        trials,
        graph,
        event_source="SWR",
    ).sort_values("candidate_arm")

    shared = [
        "candidate_arm",
        "is_current_arm",
        "is_prev_arm",
        "is_next_arm",
        "is_last_rewarded_arm",
        "is_last_rewarded_noncurrent_arm",
        "arm_patch_id",
        "is_same_patch_as_current",
        "distance_to_candidate_arm_cm",
        "candidate_arm_distance_rank",
        "visit_recency_trials",
        "is_candidate_arm_never_visited",
        "visit_recency_trials_event_leaf_centered",
        *RECENCY_SPLINE_COLUMNS,
        "reward_count_candidate_arm_last10_known_trials",
        "reward_fraction_candidate_arm_last10_visits",
        "task_reward_probability_candidate_arm",
        "reward_belief_mean_candidate_arm_preoutcome",
        "is_current_arm_x_is_last_rewarded_arm",
        "time_since_stop_start_s_x_is_current_arm",
        "time_since_stop_start_s_x_is_prev_arm",
        "time_since_stop_start_s_x_is_next_arm",
        "swr_x_is_current_arm",
        "swr_x_is_prev_arm",
        "swr_x_is_next_arm",
        "swr_x_is_last_rewarded_arm",
        "swr_x_reward_fraction_candidate_arm_last10_visits",
        "swr_x_visit_recency_trials_event_leaf_centered",
        *[f"swr_x_{column}" for column in RECENCY_SPLINE_COLUMNS],
        "swr_x_is_candidate_arm_never_visited",
        "swr_x_is_last_rewarded_noncurrent_arm",
        "mua_x_is_current_arm",
        "mua_x_is_prev_arm",
        "mua_x_is_next_arm",
        "mua_x_is_last_rewarded_arm",
        "is_omitted_outcome_x_is_last_rewarded_arm",
    ]
    assert set(shared) <= set(nonlocal_arms) & set(contfrag_arms)
    np.testing.assert_allclose(
        nonlocal_arms[shared].to_numpy(dtype=float),
        contfrag_arms[shared].to_numpy(dtype=float),
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )
