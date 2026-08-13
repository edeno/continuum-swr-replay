"""Regression: ContFrag uses the all-immobile (max) event-inclusion rule.

ContFrag events come from external detectors (SWR/MUA), so a single event
label can span real movement (an immobile → run → immobile sequence; for
MUA, up to ~23 s — see ``docs/known_issues/degenerate_mua_events.md``).
Such an event's immobile timebins fall in two immobility bouts, which under
the old per-timebin inclusion rule made the stop-table ``n_events``
double-count it relative to the (atomic) event table — breaking
``Σ stop_table.n_events == len(event_table)`` and making
``prepare_contfrag_replay_content_data.py`` refuse to write output.

The builders now include an event only if *every* timebin is immobile
(``_in_contfrag_event_mask``), so movement-spanning events are excluded
entirely (matching the model-detected non-local events' effective gating
and making the event-source comparison apples-to-apples). This removes the
straddle by construction and keeps the cross-table invariant.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.contfrag.tables import (
    build_contfrag_event_arm_table,
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


def _two_event_dot() -> pd.DataFrame:
    """Two SWR events: one movement-spanning, one fully immobile.

    Layout (per sample): 5 mobile pad |
      event 1: 6 immobile (A) | 4 mobile gap | 6 immobile (B) |
      5 mobile pad |
      event 2: 8 immobile |
      5 mobile pad.

    ``SWR_event = 1`` over A+gap+B (a single event spanning real movement —
    the old "straddler"); ``SWR_event = 2`` over the fully-immobile block.
    Under the all-immobile (max) rule, event 1 is excluded (it contains
    mobile bins) and event 2 is included exactly once.
    """
    pad, a, gap, b, ev2 = 5, 6, 4, 6, 8
    total = pad + a + gap + b + pad + ev2 + pad
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
    swr = dot.columns.get_loc("SWR_event")
    spd = dot.columns.get_loc("head_speed")
    mp = dot.columns.get_loc("map_estimate")
    ts = dot.columns.get_loc("track_segment_id")

    a0, a1 = pad, pad + a
    g1 = a1 + gap
    b0, b1 = g1, g1 + b
    e0, e1 = b1 + pad, b1 + pad + ev2

    # Event 1: spans A + mobile gap + B (movement-spanning).
    dot.iloc[a0:b1, swr] = 1
    dot.iloc[a0:a1, spd] = 1.0  # A immobile
    dot.iloc[b0:b1, spd] = 1.0  # B immobile; gap stays mobile (10.0)
    # Event 2: fully immobile block.
    dot.iloc[e0:e1, swr] = 2
    dot.iloc[e0:e1, spd] = 1.0
    # Real-arm decoding on every SWR-labelled bin (arm 0's first place bin).
    for lo, hi in ((a0, a1), (b0, b1), (e0, e1)):
        dot.iloc[lo:hi, mp] = 0.0
        dot.iloc[lo:hi, ts] = 0.0
    return dot


def test_movement_spanning_event_excluded_invariant_holds():
    dot = _two_event_dot()
    env, graph, trials = _EnvStub(), _bandit_graph(), _empty_trials()

    ev = build_contfrag_event_table(dot, env, trials, graph, event_source="SWR")
    stop = build_contfrag_stop_table(dot, trials, graph, event_source="SWR")

    # Precondition: event 1 genuinely spans movement (has mobile SWR bins),
    # so it is exactly the case the max rule must exclude.
    swr1 = dot[dot["SWR_event"] == 1]
    assert (swr1["head_speed"] >= 4.0).any(), "event 1 must contain a mobile bin"

    # Under all-immobile: only the fully-immobile event 2 survives.
    assert len(ev) == 1, f"expected only the immobile event; got {len(ev)}"
    assert int(ev["SWR_event"].iloc[0]) == 2
    assert 1 not in set(ev["SWR_event"]), "movement-spanning event 1 must be excluded"

    # Cross-table invariant holds: the excluded event is counted nowhere.
    assert int(stop["n_events"].sum()) == len(ev) == 1, (
        f"Σ stop.n_events={int(stop['n_events'].sum())} != len(event_table)={len(ev)}"
    )
    # Empty trial history is explicit rather than silently mapping the modal
    # physical arm onto a behavioral well. The same schema is present on both
    # canonical event-source tables.
    assert ev["location_state"].tolist() == ["unknown"]
    assert ev["at_well"].isna().all()
    assert ev["location_arm"].isna().all()
    assert ev["current_arm"].isna().all()
    assert ev["physical_arm"].tolist() == [0.0]
    for column in (
        "location_state",
        "at_well",
        "location_well",
        "location_arm",
        "physical_arm",
        "nearest_well_distance_cm_max",
    ):
        assert column in stop.columns
    assert stop["completed_trial_id"].isna().all()
    assert stop["completed_trial_arm"].isna().all()
    assert stop["recent_switch_rate"].isna().all()
    assert stop["leaky_q_value_diff"].isna().all()


@pytest.mark.parametrize(
    ("event_source", "event_col"),
    [("SWR", "SWR_event"), ("MUA", "MUA_event")],
)
@pytest.mark.parametrize("bad_speed", [4.0, np.nan, np.inf, -np.inf, 10.0])
def test_every_external_event_source_rejects_one_bad_bin_atomically(
    event_source, event_col, bad_speed
):
    """SWR and MUA share the exact finite, strict whole-event criterion."""
    dot = _two_event_dot()
    if event_col == "MUA_event":
        dot["MUA_event"] = dot["SWR_event"]
        dot["SWR_event"] = 0

    event_1 = dot[event_col].to_numpy() == 1
    dot.loc[event_1, "head_speed"] = 1.0
    middle = np.flatnonzero(event_1)[event_1.sum() // 2]
    dot.iloc[middle, dot.columns.get_loc("head_speed")] = bad_speed

    env, graph, trials = _EnvStub(), _bandit_graph(), _empty_trials()
    event = build_contfrag_event_table(
        dot, env, trials, graph, event_source=event_source
    )
    arm = build_contfrag_event_arm_table(
        event, dot, env, trials, graph, event_source=event_source
    )
    stop = build_contfrag_stop_table(dot, trials, graph, event_source=event_source)

    assert event[event_col].tolist() == [2]
    assert arm["event_id"].nunique() == 1
    assert len(arm) == 9
    assert int(stop["n_events"].sum()) == len(event) == 1


@pytest.mark.parametrize(
    ("mua_start_offset", "expected_overlap"),
    [(0, False), (-1, True)],
)
def test_cross_source_flags_use_strict_event_level_overlap(
    mua_start_offset, expected_overlap
):
    """SWR/MUA boundary contact is not positive-duration overlap."""
    dot = _two_event_dot()
    dot["MUA_event"] = 0
    swr_rows = np.flatnonzero(dot["SWR_event"].to_numpy() == 2)
    start = swr_rows[-1] + mua_start_offset
    stop = swr_rows[-1] + 4
    dot.iloc[start:stop, dot.columns.get_loc("MUA_event")] = 7
    dot.iloc[start:stop, dot.columns.get_loc("head_speed")] = 1.0

    env, graph, trials = _EnvStub(), _bandit_graph(), _empty_trials()
    swr_events = build_contfrag_event_table(dot, env, trials, graph, event_source="SWR")
    mua_events = build_contfrag_event_table(dot, env, trials, graph, event_source="MUA")

    swr_2 = swr_events.loc[swr_events["SWR_event"] == 2].iloc[0]
    mua_7 = mua_events.loc[mua_events["MUA_event"] == 7].iloc[0]
    assert bool(swr_2["is_mua"]) is expected_overlap
    assert bool(mua_7["is_swr"]) is expected_overlap
