"""Tests for :mod:`continuum_swr_replay.analysis.replay.tables`.

Scope: correctness-at-the-boundary regression tests. Broader coverage
(invariant sweeps across all four tables) lives under Phase B3 of the
execution plan.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay.tables import (
    build_event_arm_table,
    build_event_table,
)


class _EnvStub:
    """Minimal ``Environment`` stand-in for the tests.

    Only exposes the single attribute that
    :func:`summaries.posterior_to_arm_marginal` and
    :func:`spatial_mapping.map_estimate_to_arm` touch:
    ``place_bin_centers_nodes_df_``.
    """

    class _Df:
        d = {"linear_position": np.array([0.0, 10.0]), "edge_id": np.array([0, 1])}

        def __getitem__(self, k):
            return self.d[k]

    place_bin_centers_nodes_df_ = _Df()


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


def _tiny_dot_with_two_events() -> pd.DataFrame:
    """100 timebins with two non-local events at indices 20-29 and 60-69."""
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
            "map_estimate": np.nan,
            "head_speed": 10.0,
            "track_segment_id": np.nan,
        },
        index=idx,
    )
    for ev_id, sl in [(1, slice(20, 30)), (2, slice(60, 70))]:
        rows = dot.index[sl]
        dot.loc[rows, "non_local_event"] = ev_id
        dot.loc[rows, "Non-Local_Continuous_prob"] = 0.9
        dot.loc[rows, "head_speed"] = 1.0
        dot.loc[rows, "map_estimate"] = float(ev_id)
        dot.loc[rows, "track_segment_id"] = int(ev_id)
    return dot


@pytest.fixture
def tiny_session():
    dot = _tiny_dot_with_two_events()
    env = _EnvStub()
    trials = pd.DataFrame(
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
    graph = _bandit_graph()
    return dot, env, trials, graph


def test_event_arm_table_respects_non_default_event_table_index(tiny_session):
    """``build_event_arm_table`` must tolerate any index on ``event_table``.

    Regression for a bug where the builder used the DataFrame's index
    label as a positional offset into the per-event predictor arrays.
    Worked on a ``RangeIndex(start=0)`` but silently corrupted the
    output for any frame produced by ``pd.concat`` across sessions
    (whose index is no longer zero-based).
    """
    dot, env, trials, graph = tiny_session
    et = build_event_table(dot, env, trials, graph)
    assert len(et) == 2

    # Shift the index to simulate concat of a prior session.
    shifted = et.copy()
    shifted.index = shifted.index + 500

    eat_shifted = build_event_arm_table(shifted, dot, env, trials, graph)
    eat_baseline = build_event_arm_table(et, dot, env, trials, graph)

    # Output must have 9 rows per event regardless of index.
    assert len(eat_shifted) == 2 * 9

    # Per-event per-arm predictor columns populated for every event.
    # Before the fix these were NaN (or IndexError) for everything
    # past the first session's rows.
    for col in ("dist_to_arm", "arm_dist_rank"):
        # ``dist_to_arm`` is NaN when current_arm is NaN — accept that,
        # but require the number of non-NaN rows to match the baseline.
        assert (
            eat_shifted[col].notna().sum() == eat_baseline[col].notna().sum()
        ), f"column {col} lost values when event_table had a non-zero-based index"
