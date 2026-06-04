"""Regression test for stop-table bout-start trial-context leakage.

The replay/contfrag stop tables doc-claim "trial-context at bout
start", but the implementation used ``grouped.first()`` (default
``skipna=True``) and ``safe_first`` (drop NaN before .iloc[0]) to
extract those values. Both silently surface the *next* trial's
context for stops whose bout start fell before any trial began —
e.g. a bout from t=0..2 on a session whose first trial starts at
t=1 used to inherit trial 1's ``is_after_reward`` / ``current_arm``
even though t=0 had no trial active.

Fix: pass ``skipna=False`` to ``grouped.first()`` and replace
``safe_first`` with a NaN-strict ``_first_value`` for bout-start
extractions.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.replay.tables import build_stop_table


def _two_well_track_graph() -> nx.Graph:
    """Minimal degree-1 track graph: wells 0 and 1, single arm 0 between them."""
    g = nx.Graph()
    g.add_node(0, pos=(0.0, 0.0))
    g.add_node(1, pos=(1.0, 0.0))
    g.add_edge(0, 1, edge_id=0, distance=1.0)
    return g


def _bout_dot_frame() -> pd.DataFrame:
    """t=[0, 1, 2] all immobile. Trial context applied separately."""
    times = [0.0, 1.0, 2.0]
    return pd.DataFrame(
        {
            "animal": ["m"] * 3,
            "day": [20200101] * 3,
            "epoch": ["02_r1"] * 3,
            "head_speed": [0.0, 0.0, 0.0],
            "Non-Local_Continuous_prob": [0.0, 0.0, 0.0],
            "Non-Local_Fragmented_prob": [0.0, 0.0, 0.0],
            "non_local_event": [0, 0, 0],
            "SWR_event": [0, 0, 0],
            "MUA_event": [0, 0, 0],
            "track_segment_id": [np.nan, np.nan, np.nan],
            "map_estimate": [np.nan, np.nan, np.nan],
        },
        index=times,
    )


def _trials_starting_at_t1() -> pd.DataFrame:
    # Single rewarded trial that begins at t=1 — so t=0 (the bout
    # start) has no active trial context.
    return pd.DataFrame(
        {
            "start_time": [1.0],
            "end_time": [1.5],
            "from_well": [1],
            "to_well": [0],
            "is_reward": [True],
            "is_patch_change": [False],
        }
    )


class TestBoutStartContextNoLeak:
    def test_bout_starting_before_first_trial_yields_nan_context(self):
        dot = _bout_dot_frame()
        trials = _trials_starting_at_t1()
        graph = _two_well_track_graph()
        out = build_stop_table(
            dot, trials, graph, speed_threshold=4.0, nonlocal_threshold=0.5,
        )
        assert len(out) == 1
        row = out.iloc[0]
        # t=0 had no trial active — every trial-derived field must
        # be NaN, not inherited from the trial that started later.
        assert pd.isna(row["is_after_reward"]), (
            f"is_after_reward leaked: {row['is_after_reward']}"
        )
        assert pd.isna(row["reward_outcome"]), (
            f"reward_outcome leaked: {row['reward_outcome']}"
        )
        assert pd.isna(row["is_after_switch"]), (
            f"is_after_switch leaked: {row['is_after_switch']}"
        )
        assert pd.isna(row["current_arm"]), (
            f"current_arm leaked: {row['current_arm']}"
        )
