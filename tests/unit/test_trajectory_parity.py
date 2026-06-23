"""Cross-scenario parity: shared trajectory columns match across both tables."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.contfrag.tables import (
    _CONTFRAG_TRAJECTORY_COLUMNS,
    build_contfrag_event_table,
)
from continuum_swr_replay.analysis.replay.tables import (
    _event_table_columns,
    build_event_table,
)
from continuum_swr_replay.analysis.replay.trajectory import (
    NONLOCAL_ONLY_COLUMNS,
    SHARED_TRAJECTORY_COLUMNS,
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


def _bandit_graph() -> nx.Graph:
    g = nx.Graph()
    for u, v, eid in ((0, 6, 0), (1, 6, 1), (2, 7, 2), (3, 7, 3), (4, 8, 4),
                      (5, 8, 5), (6, 9, 6), (7, 9, 7), (8, 9, 8)):
        g.add_edge(u, v, edge_id=eid, distance=1.0)
    return g


def _trials() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["start_time", "end_time", "to_well", "from_well",
                 "from_patch", "to_patch", "is_reward", "is_patch_change"]
    )


_AUGMENTED = {
    "map_speed": 5.0, "step_distance": 1.0, "ahead_behind_distance": 3.0,
    "hpd": 12.0, "run_speed_baseline": 50.0, "ripple_power": 20.0,
    "SWR_zscore": 1.5, "mua_rate": 100.0, "MUA_zscore": 2.0,
    "theta_power": 0.3, "theta_power_zscore": 0.5,
}


def _nonlocal_dot() -> pd.DataFrame:
    idx = pd.Index(np.linspace(0, 10, 100), name="time")
    dot = pd.DataFrame(
        {"animal": "chimi", "day": 20200212, "epoch": "02_r1",
         "non_local_event": 0, "SWR_event": 0, "MUA_event": 0,
         "Non-Local_Continuous_prob": 0.0, "Non-Local_Fragmented_prob": 0.0,
         "Local_prob": 0.1, "No-Spike_prob": 0.0,
         "map_estimate": np.nan, "head_speed": 10.0, "track_segment_id": np.nan},
        index=idx,
    )
    rows = dot.index[20:30]
    dot.loc[rows, "non_local_event"] = 1
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
        {"animal": "chimi", "day": 20200212, "epoch": "02_r1",
         "SWR_event": 0, "MUA_event": 0,
         "Continuous_prob": 0.5, "Fragmented_prob": 0.5,
         "map_estimate": np.nan, "head_speed": 10.0, "track_segment_id": np.nan},
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
    nl_cols = set(_event_table_columns())
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
