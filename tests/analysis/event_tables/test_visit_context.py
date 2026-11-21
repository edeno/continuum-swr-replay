"""The well-visit label of events and stops (``event_tables/visit_context.py``).

Hand-built visits on a hand-placed track: three visits (wells 0, 2, 4), each
from its first poke-in (``arrival_time``) to its final poke-out
(``departure_time``), and windows placed by hand between, across and inside
them. Distances are in cm on the projected-position plane.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.constants import (
    VISIT_CONTEXTS,
    VISIT_JUNCTION_RADIUS_CM,
    VISIT_WELL_RADIUS_CM,
)
from continuum_swr_replay.analysis.event_tables import event_table as event_module
from continuum_swr_replay.analysis.event_tables import stop_table as stop_module
from continuum_swr_replay.analysis.event_tables.arm_patch_tables import (
    build_event_patch_table,
)
from continuum_swr_replay.analysis.event_tables.table_contracts import (
    require_visit_context_labels,
)
from continuum_swr_replay.analysis.event_tables.visit_context import (
    VISIT_CONTEXT_COLUMNS,
    visit_context_of_samples,
    window_visit_context,
)
from tests.analysis.event_tables.test_replay_tables import (
    _location_contract_session,
    build_event_arm_table,
    build_event_table,
    build_stop_table,
)

#: Node positions: center 9 at the origin, patch junctions 100 cm out, wells
#: 50 cm beyond their junction and 40 cm apart.
_POSITIONS = {
    9: (0.0, 0.0),
    6: (0.0, 100.0),
    7: (100.0, 0.0),
    8: (-100.0, 0.0),
    0: (-20.0, 150.0),
    1: (20.0, 150.0),
    2: (150.0, -20.0),
    3: (150.0, 20.0),
    4: (-150.0, -20.0),
    5: (-150.0, 20.0),
}
_EDGES = ((0, 6, 0), (1, 6, 1), (2, 7, 2), (3, 7, 3), (4, 8, 4), (5, 8, 5))
_EDGES += ((6, 9, 6), (7, 9, 7), (8, 9, 8))


def _graph() -> nx.Graph:
    graph = nx.Graph()
    for u, v, edge_id in _EDGES:
        graph.add_edge(u, v, edge_id=edge_id, distance=1.0)
    for node, pos in _POSITIONS.items():
        graph.nodes[node]["pos"] = pos
    return graph


def _experiences() -> pd.DataFrame:
    """Visits 0, 1, 2 at wells 0, 2, 4: [10, 20], [40, 50], [70, 80] s."""
    return pd.DataFrame(
        {
            "experience_id": [0, 1, 2],
            "movement_trial_number": pd.array([pd.NA, 0, 1], dtype="Int64"),
            "is_initial_experience": [True, False, False],
            "is_boundary_derived": [False, False, False],
            "outcome_observed": [True, True, True],
            "port": [0, 2, 4],
            "patch": [1, 2, 3],
            "arrival_time": [10.0, 40.0, 70.0],
            "departure_time": [20.0, 50.0, 80.0],
            "reward_on_time": [np.nan, 41.0, np.nan],
            "outcome_known_time": [10.3, 41.0, 70.3],
            "is_reward": [False, True, False],
        }
    )


def _offset(node: int, dx: float, dy: float = 0.0) -> tuple[float, float]:
    x, y = _POSITIONS[node]
    return (x + dx, y + dy)


#: (name, start, end, occupancy_state, first_sample_state, (x, y), segment,
#:  expected visit_context)
_WINDOWS = [
    (
        "first moment, at the first poke-in",
        10.0,
        11.0,
        "at_well",
        "at_well",
        _offset(0, 0),
        0,
        "in_visit",
    ),
    (
        "inside the second visit",
        42.0,
        43.0,
        "at_well",
        "at_well",
        _offset(2, 0),
        2,
        "in_visit",
    ),
    (
        "at the final poke-out, next to the well left",
        20.0,
        21.0,
        "in_transit",
        "in_transit",
        _offset(0, 0, -5.0),
        0,
        "after_visit_at_well",
    ),
    (
        "exactly at the well radius",
        22.0,
        23.0,
        "in_transit",
        "in_transit",
        _offset(0, 0, -VISIT_WELL_RADIUS_CM),
        0,
        "after_visit_at_well",
    ),
    (
        "just beyond the well radius",
        22.0,
        23.0,
        "in_transit",
        "in_transit",
        _offset(0, 0, -VISIT_WELL_RADIUS_CM - 0.01),
        0,
        "track_leaf_arm",
    ),
    (
        "next to the next well before its poke-in",
        25.0,
        26.0,
        "in_transit",
        "in_transit",
        _offset(2, -5.0),
        2,
        "before_visit_at_well",
    ),
    (
        "next to a well neither left nor next",
        27.0,
        28.0,
        "in_transit",
        "in_transit",
        _offset(5, 0, 3.0),
        5,
        "at_other_well",
    ),
    (
        "on a leaf arm",
        29.0,
        30.0,
        "in_transit",
        "in_transit",
        (10.0, 125.0),
        1,
        "track_leaf_arm",
    ),
    (
        "on a trunk arm",
        29.0,
        30.0,
        "in_transit",
        "in_transit",
        (50.0, 0.0),
        7,
        "track_trunk_arm",
    ),
    (
        "at a patch junction",
        29.0,
        30.0,
        "in_transit",
        "in_transit",
        _offset(6, 0, -VISIT_JUNCTION_RADIUS_CM),
        6,
        "track_patch_junction",
    ),
    (
        "at the center",
        29.0,
        30.0,
        "in_transit",
        "in_transit",
        (3.0, 3.0),
        8,
        "track_center",
    ),
    (
        "no position",
        29.0,
        30.0,
        "in_transit",
        "in_transit",
        (np.nan, np.nan),
        np.nan,
        "track_no_position",
    ),
    (
        "spans the final poke-out",
        18.0,
        22.0,
        "mixed",
        "at_well",
        _offset(0, 0),
        0,
        "spans_visit_end",
    ),
    (
        "spans a poke-in",
        38.0,
        41.0,
        "mixed",
        "in_transit",
        _offset(2, 0),
        2,
        "spans_visit_start",
    ),
    (
        "spans the first poke-in",
        8.0,
        11.0,
        "mixed",
        "unknown",
        _offset(0, 0),
        0,
        "spans_visit_start",
    ),
    (
        "before the first poke-in",
        2.0,
        3.0,
        "unknown",
        "unknown",
        _offset(0, 0, -30.0),
        0,
        "before_first_visit",
    ),
    (
        "after the last visit (last moments)",
        85.0,
        86.0,
        "in_transit",
        "in_transit",
        _offset(4, 4.0),
        4,
        "after_visit_at_well",
    ),
    (
        "no analyzed samples",
        np.nan,
        np.nan,
        "unknown",
        None,
        (np.nan, np.nan),
        np.nan,
        "no_analyzed_samples",
    ),
]


def _label(windows=_WINDOWS) -> pd.DataFrame:
    return window_visit_context(
        window_start=np.array([w[1] for w in windows]),
        window_end=np.array([w[2] for w in windows]),
        occupancy_state=np.array([w[3] for w in windows], dtype=object),
        first_sample_state=np.array([w[4] for w in windows], dtype=object),
        reference_position=np.array([w[5] for w in windows], dtype=float),
        reference_segment=np.array([w[6] for w in windows], dtype=float),
        port_experiences=_experiences(),
        track_graph=_graph(),
    )


def test_each_window_gets_its_hand_checked_context():
    labels = _label()
    assert list(labels.columns) == list(VISIT_CONTEXT_COLUMNS)
    got = dict(zip([w[0] for w in _WINDOWS], labels["visit_context"], strict=True))
    expected = {w[0]: w[7] for w in _WINDOWS}
    assert got == expected
    # Every value of the vocabulary is exercised.
    assert set(labels["visit_context"]) == set(VISIT_CONTEXTS)


def test_visit_ids_and_times_by_hand():
    labels = _label().set_axis([w[0] for w in _WINDOWS])

    def row(name):
        return labels.loc[name]

    first = row("first moment, at the first poke-in")
    assert first["well_visit_id"] == 0
    assert pd.isna(first["last_completed_well_visit_id"])
    assert np.isnan(first["time_since_last_poke_out_s"])
    assert np.isnan(first["time_until_next_poke_in_s"])
    assert row("inside the second visit")["well_visit_id"] == 1
    assert row("inside the second visit")["last_completed_well_visit_id"] == 0

    left = row("at the final poke-out, next to the well left")
    assert pd.isna(left["well_visit_id"])
    assert left["last_completed_well_visit_id"] == 0
    assert left["time_since_last_poke_out_s"] == 0.0  # starts at the poke-out
    assert left["time_until_next_poke_in_s"] == 19.0  # 40 - 21

    before_next = row("next to the next well before its poke-in")
    assert before_next["time_since_last_poke_out_s"] == 5.0
    assert before_next["time_until_next_poke_in_s"] == 14.0

    # A window across a boundary has no visit and no gap to a poke.
    for name in ("spans the final poke-out", "spans a poke-in"):
        assert pd.isna(row(name)["well_visit_id"])
        assert np.isnan(row(name)["time_since_last_poke_out_s"])
        assert np.isnan(row(name)["time_until_next_poke_in_s"])
    assert pd.isna(row("spans the final poke-out")["last_completed_well_visit_id"])
    assert row("spans a poke-in")["last_completed_well_visit_id"] == 0

    first_visit = row("before the first poke-in")
    assert pd.isna(first_visit["last_completed_well_visit_id"])
    assert np.isnan(first_visit["time_since_last_poke_out_s"])
    assert first_visit["time_until_next_poke_in_s"] == 7.0  # 10 - 3

    last = row("after the last visit (last moments)")
    assert last["last_completed_well_visit_id"] == 2
    assert last["time_since_last_poke_out_s"] == 5.0
    assert np.isnan(last["time_until_next_poke_in_s"])  # no later poke-in

    empty = row("no analyzed samples")
    assert empty[list(VISIT_CONTEXT_COLUMNS[1:])].isna().all()


def test_well_left_wins_over_the_next_well_and_rejects_unknown_states():
    # Move well 2 to 8 cm from well 0, so a point 4 cm from each is next to
    # both the well left (0) and the next well (2): it is "after".
    graph = _graph()
    graph.nodes[2]["pos"] = _offset(0, 8.0)
    labels = window_visit_context(
        window_start=np.array([25.0]),
        window_end=np.array([26.0]),
        occupancy_state=np.array(["in_transit"], dtype=object),
        first_sample_state=np.array(["in_transit"], dtype=object),
        reference_position=np.array([_offset(0, 4.0)]),
        reference_segment=np.array([0.0]),
        port_experiences=_experiences(),
        track_graph=graph,
    )
    assert labels["visit_context"].tolist() == ["after_visit_at_well"]
    with pytest.raises(ValueError, match="no visit context"):
        window_visit_context(
            window_start=np.array([25.0]),
            window_end=np.array([26.0]),
            occupancy_state=np.array(["moving"], dtype=object),
            first_sample_state=np.array(["in_transit"], dtype=object),
            reference_position=np.array([[0.0, 0.0]]),
            reference_segment=np.array([0.0]),
            port_experiences=_experiences(),
            track_graph=_graph(),
        )


def test_samples_give_times_first_state_and_middle_position():
    # Window 7: three samples; its middle (index 1) sits next to well 0.
    # Window 3: two samples; its middle is its first ((2 - 1) // 2 = 0).
    # Window 5 has no sample.
    samples = pd.DataFrame(
        {
            "occupancy_state": ["in_transit", "in_transit", "in_transit"]
            + ["in_transit", "in_transit"],
            "projected_x_position": [50.0, -20.0, 50.0, 0.0, 0.0],
            "projected_y_position": [0.0, 147.0, 0.0, 100.0, 50.0],
            "track_segment_id": [7.0, 0.0, 7.0, 6.0, 6.0],
        },
        index=pd.Index([21.0, 22.0, 23.0, 24.0, 25.0], name="time"),
    )
    labels = visit_context_of_samples(
        samples,
        np.array([7, 7, 7, 3, 3]),
        np.array([7, 3, 5]),
        np.array(["in_transit", "in_transit", "unknown"], dtype=object),
        _experiences(),
        _graph(),
    )
    assert labels["visit_context"].tolist() == [
        "after_visit_at_well",
        "track_patch_junction",
        "no_analyzed_samples",
    ]
    np.testing.assert_allclose(labels["time_since_last_poke_out_s"], [1.0, 4.0, np.nan])
    np.testing.assert_allclose(
        labels["time_until_next_poke_in_s"], [17.0, 15.0, np.nan]
    )


def test_builders_label_events_and_stops_and_carry_them_to_every_table():
    dot, env, trials, graph = _location_contract_session()
    events = build_event_table(dot, env, trials, graph).sort_values("non_local_event")
    stops = build_stop_table(dot, trials, graph).sort_values("stop_start")
    arms = build_event_arm_table(events, dot, env, trials, graph)
    patches = build_event_patch_table(events, arms)

    # Event 1 (2.0-2.4 s) is in transit after the initial visit at well 0,
    # 10 cm from it (the radius is inclusive); event 2 is inside visit 1;
    # event 3 starts in transit and crosses the 8.0 s poke-in.
    assert events["visit_context"].tolist() == [
        "after_visit_at_well",
        "in_visit",
        "spans_visit_start",
    ]
    assert str(events["well_visit_id"].dtype) == "Int64"
    assert events["well_visit_id"].isna().tolist() == [True, False, True]
    assert events["well_visit_id"].iloc[1] == 1
    assert stops["visit_context"].tolist() == events["visit_context"].tolist()
    for table in (arms, patches):
        by_event = table.groupby("event_id")[list(VISIT_CONTEXT_COLUMNS)].first()
        expected = events.set_index("event_id")[list(VISIT_CONTEXT_COLUMNS)]
        pd.testing.assert_frame_equal(
            by_event.sort_index(), expected.sort_index(), check_dtype=False
        )
    for table, name in ((events, "event"), (stops, "stop"), (arms, "event_arm")):
        require_visit_context_labels(table, name)


def test_the_label_changes_no_other_column(monkeypatch):
    """Building without the label gives the same other columns and rows."""
    dot, env, trials, graph = _location_contract_session()
    with_label = {
        "event": build_event_table(dot, env, trials, graph),
        "stop": build_stop_table(dot, trials, graph),
    }

    def _no_label(*args, **kwargs):
        windows = args[2]
        return pd.DataFrame(
            {column: [np.nan] * len(windows) for column in VISIT_CONTEXT_COLUMNS}
        )

    monkeypatch.setattr(event_module, "visit_context_of_samples", _no_label)
    monkeypatch.setattr(stop_module, "visit_context_of_samples", _no_label)
    without = {
        "event": build_event_table(dot, env, trials, graph),
        "stop": build_stop_table(dot, trials, graph),
    }
    for grain in with_label:
        other = [c for c in with_label[grain] if c not in VISIT_CONTEXT_COLUMNS]
        pd.testing.assert_frame_equal(
            with_label[grain][other], without[grain][other], check_exact=True
        )
    assert with_label["event"]["event_id"].tolist() == (
        without["event"]["event_id"].tolist()
    )


def test_contract_checks_vocabulary_and_agreement_with_occupancy():
    good = pd.DataFrame(
        {
            "occupancy_state": ["at_well", "in_transit"],
            "visit_context": ["in_visit", "track_center"],
        }
    )
    require_visit_context_labels(good, "t")
    require_visit_context_labels(good.drop(columns="visit_context"), "t")
    with pytest.raises(ValueError, match="unknown visit_context"):
        require_visit_context_labels(good.replace("track_center", "hallway"), "t")
    with pytest.raises(ValueError, match="not at_well"):
        require_visit_context_labels(
            good.assign(visit_context=["in_visit", "in_visit"]), "t"
        )
