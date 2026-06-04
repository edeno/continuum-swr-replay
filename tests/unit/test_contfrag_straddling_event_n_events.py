"""Regression: ContFrag stop-table ``n_events`` must count each event once.

ContFrag events come from external detectors (SWR/MUA), so a single event
label can span a brief mid-event movement that splits the surrounding
immobility into two bouts. ``build_contfrag_stop_table`` used to count
``n_events`` as ``nunique(event_col)`` per bout, double-counting such a
straddling event, while ``build_contfrag_event_table`` emits it as one row
(assigned to a single stop by ``event_start``). That broke the cross-table
invariant ``Σ stop_table.n_events == len(event_table)`` and made
``prepare_contfrag_replay_content_data.py`` refuse to write any output.

The fix assigns each event to the single bout of its ``event_start`` when
counting ``n_events`` — mirroring the event table — so a straddling event
is counted exactly once.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.contfrag.tables import (
    build_contfrag_event_table,
    build_contfrag_stop_table,
)


class _EnvStub:
    """Minimal Environment stand-in: 2 place bins per real arm (0-8)."""

    class _Df:
        _arm_lp = np.arange(18, dtype=float)
        _arm_eid = np.repeat(np.arange(9), 2).astype(int)
        d = {"linear_position": _arm_lp, "edge_id": _arm_eid}

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


def _empty_trials() -> pd.DataFrame:
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


def _straddling_dot() -> pd.DataFrame:
    """One SWR event (label 1) whose immobile timebins span two bouts.

    Layout (per sample): 5 mobile pad | 8 immobile (block A) | 4 mobile
    gap | 8 immobile (block B) | 5 mobile pad. ``SWR_event = 1`` over
    A+gap+B, so the in-event (SWR>0 & immobile) rows are A ∪ B — two
    distinct immobility bouts (the mobile gap splits them).
    """
    pad, blockA, gap, blockB = 5, 8, 4, 8
    total = pad + blockA + gap + blockB + pad
    idx = pd.Index(np.arange(total, dtype=float) * 0.01, name="time")

    dot = pd.DataFrame(
        {
            "animal": "test",
            "day": 1,
            "epoch": "02_r1",
            "SWR_event": 0,
            "MUA_event": 0,
            "Continuous_prob": 0.5,
            "Fragmented_prob": 0.5,
            "map_estimate": np.nan,
            "head_speed": 10.0,  # mobile by default
            "track_segment_id": np.nan,
        },
        index=idx,
    )

    a0, a1 = pad, pad + blockA
    b0, b1 = a1 + gap, a1 + gap + blockB

    # SWR label spans both blocks AND the mobile gap (single event).
    dot.iloc[a0:b1, dot.columns.get_loc("SWR_event")] = 1
    # Immobile only in the two blocks; the gap stays mobile -> 2 bouts.
    dot.iloc[a0:a1, dot.columns.get_loc("head_speed")] = 1.0
    dot.iloc[b0:b1, dot.columns.get_loc("head_speed")] = 1.0
    # Real-arm decoding on the in-event blocks (arm 0's first place bin).
    dot.iloc[a0:a1, dot.columns.get_loc("map_estimate")] = 0.0
    dot.iloc[b0:b1, dot.columns.get_loc("map_estimate")] = 0.0
    dot.iloc[a0:a1, dot.columns.get_loc("track_segment_id")] = 0.0
    dot.iloc[b0:b1, dot.columns.get_loc("track_segment_id")] = 0.0
    return dot


def test_straddling_event_counted_once_cross_table_invariant():
    dot = _straddling_dot()
    env, graph, trials = _EnvStub(), _bandit_graph(), _empty_trials()

    ev = build_contfrag_event_table(dot, env, trials, graph, event_source="SWR")
    stop = build_contfrag_stop_table(dot, trials, graph, event_source="SWR")

    # Precondition: the fixture really straddles — the single SWR event's
    # immobile timebins fall in two distinct immobility bouts.
    assert len(ev) == 1, f"expected exactly one event; got {len(ev)}"
    assert int(stop["is_swr_stop"].sum()) == 2, (
        "fixture should place the SWR event's immobile bins in two bouts"
    )

    # The invariant the prepare-script gate enforces.
    assert int(stop["n_events"].sum()) == len(ev) == 1, (
        f"Σ stop.n_events={int(stop['n_events'].sum())} != "
        f"len(event_table)={len(ev)} — straddling event double-counted."
    )
    # The event is attributed to exactly one bout (the one with its start).
    assert sorted(stop["n_events"].tolist()) == [0, 1]
