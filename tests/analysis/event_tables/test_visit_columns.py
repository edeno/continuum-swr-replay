"""The well-visit columns of the stop, event and event-arm tables.

The visit functions of ``event_tables/visit_context.py`` (the visit a stop
overlaps and how it ends, the visit clocks of an event, the current arm of an
event that crosses the final poke-out) and the sample count of
``behavior/covariates/history/reward_history.py``, on hand-built visits, and
through the table builders on one hand-built session.

The session (:func:`visit_session`, 0.0 to 9.0 s in 0.1 s samples) has three
visits: the starting port at well 0 [0.0, 1.0), well 1 [3.0, 6.5) and well 2
[8.0, 9.0). The animal is immobile in five stops:

====  ===========  =========================================  ==============
stop  span (s)     where                                      ends by
====  ===========  =========================================  ==============
0     3.2 - 3.8    inside visit 1                             small movement
1     4.0 - 4.6    inside visit 1, holds event 1 (4.1 - 4.4)  small movement
2     6.0 - 6.8    crosses visit 1's final poke-out at 6.5,   leaves
                   holds event 2 (6.3 - 6.7)
3     7.0 - 7.4    between visits 1 and 2                     outside visit
4     7.7 - 8.5    crosses visit 2's first poke-in at 8.0     leaves
====  ===========  =========================================  ==============

Visit 0 has no stop.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.behavior.covariates.history import (
    log1p_known_visit_count_all_events,
)
from continuum_swr_replay.analysis.behavior.spatial_mapping import (
    build_well_to_arm_map,
)
from continuum_swr_replay.analysis.event_tables.arm_patch_tables import (
    build_event_patch_table,
)
from continuum_swr_replay.analysis.event_tables.visit_context import (
    EVENT_VISIT_COLUMNS,
    STOP_VISIT_COLUMNS,
    event_visit_columns,
    stop_visit_columns,
)
from tests.analysis.event_tables.test_replay_tables import (
    _bandit_graph,
    _EnvStub,
    build_event_arm_table,
    build_event_table,
    build_stop_table,
)
from tests.support._decode_frame_fixtures import with_trajectory_inputs
from tests.support._port_experience_fixtures import port_experiences_from_trials

#: (start, end) of each immobility stop (s).
_STOPS = ((3.2, 3.8), (4.0, 4.6), (6.0, 6.8), (7.0, 7.4), (7.7, 8.5))
#: (label, start, end, decoded position) of each non-local event (s).
_EVENTS = ((1, 4.1, 4.4, 2.0), (2, 6.3, 6.7, 6.0))


def _trials() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "movement_start_time": [1.0, 6.5],
            "destination_first_poke_in_time": [3.0, 8.0],
            "end_time": [6.5, 9.0],
            "from_well": [0, 1],
            "to_well": [1, 2],
            "is_reward": [False, True],
            "reward_on_time": [np.nan, 8.1],
            "outcome_known_time": [3.3, 8.1],
            "is_patch_change": [False, True],
        }
    )


def visit_session() -> tuple[pd.DataFrame, _EnvStub, pd.DataFrame, nx.Graph]:
    """The module's session: ``(dot, env, trials, track_graph)``."""
    times = np.round(np.arange(0.0, 9.01, 0.1), 6)
    dot = pd.DataFrame(
        {
            "animal": "test",
            "day": 1,
            "epoch": "02_r1",
            "non_local_event": 0,
            "SWR_event": 0,
            "MUA_event": 0,
            "Non-Local_Continuous_prob": 0.0,
            "Non-Local_Fragmented_prob": 0.0,
            "map_estimate": np.nan,
            "head_speed": 10.0,
            "track_segment_id": 1.0,
            "projected_x_position": 10.0,
            "projected_y_position": 0.0,
        },
        index=pd.Index(times, name="time"),
    )
    for start, end in _STOPS:
        dot.loc[(times >= start) & (times <= end), "head_speed"] = 1.0
    for label, start, end, position in _EVENTS:
        rows = (times >= start) & (times <= end)
        dot.loc[rows, "non_local_event"] = label
        dot.loc[rows, "Non-Local_Continuous_prob"] = 0.9
        dot.loc[rows, "map_estimate"] = position
    graph = _bandit_graph()
    for node in graph:
        graph.nodes[node]["pos"] = (float(node * 10), 0.0)
    return with_trajectory_inputs(dot), _EnvStub(), _trials(), graph


def _experiences() -> pd.DataFrame:
    return port_experiences_from_trials(_trials(), prepend_initial=True)


def _no_excluded_time() -> pd.DataFrame:
    return pd.DataFrame({"start_time": [], "end_time": []}, dtype=float)


@pytest.fixture(scope="module")
def tables() -> dict[str, pd.DataFrame]:
    dot, env, trials, graph = visit_session()
    events = build_event_table(dot, env, trials, graph).sort_values(
        "event_start_s", ignore_index=True
    )
    arms = build_event_arm_table(events, dot, env, trials, graph)
    return {
        "stop": build_stop_table(dot, trials, graph).sort_values(
            "stop_start", ignore_index=True
        ),
        "event": events,
        "event_arm": arms,
        "event_patch": build_event_patch_table(events, arms),
    }


def test_the_session_has_the_planned_visits_and_stops():
    experiences = _experiences()
    np.testing.assert_allclose(experiences["arrival_time"], [0.0, 3.0, 8.0])
    np.testing.assert_allclose(experiences["departure_time"], [1.0, 6.5, 9.0])
    dot, _, trials, graph = visit_session()
    stops = build_stop_table(dot, trials, graph).sort_values("stop_start")
    np.testing.assert_allclose(stops["stop_start"], [s for s, _ in _STOPS])
    np.testing.assert_allclose(stops["stop_end"], [e for _, e in _STOPS])


def test_stop_end_kind_on_synthetic_visits(tables):
    stops = tables["stop"]
    assert stops["visit_context"].tolist() == [
        "in_visit",
        "in_visit",
        "spans_visit_end",
        "after_visit_at_well",
        "spans_visit_start",
    ]
    assert stops["stop_end_kind"].tolist() == [
        "small_movement",
        "small_movement",
        "leaves",
        "outside_visit",
        "leaves",
    ]
    assert stops["overlapping_visit_id"].tolist() == [1, 1, 1, pd.NA, 2]
    assert stops["well_visit_id"].tolist() == [1, 1, 1, pd.NA, pd.NA]
    assert stops["visit_stretch_index"].tolist() == [0, 1, 2, pd.NA, 0]
    np.testing.assert_allclose(
        stops["visit_first_poke_in_s"], [3.0, 3.0, 3.0, np.nan, 8.0]
    )
    np.testing.assert_allclose(
        stops["visit_last_poke_out_s"], [6.5, 6.5, 6.5, np.nan, 9.0]
    )
    for column in ("overlapping_visit_id", "well_visit_id", "visit_stretch_index"):
        assert str(stops[column].dtype) == "Int64"


def test_well_visit_id_for_rows_starting_inside(tables):
    for name in ("stop", "event"):
        table = tables[name]
        starts_inside = table["visit_context"].isin(["in_visit", "spans_visit_end"])
        assert table.loc[starts_inside, "well_visit_id"].notna().all(), name
        assert table.loc[~starts_inside, "well_visit_id"].isna().all(), name
    assert tables["event"]["visit_context"].tolist() == ["in_visit", "spans_visit_end"]
    assert tables["event"]["well_visit_id"].tolist() == [1, 1]
    # A visit placed by its start agrees with the visit the stop overlaps.
    stops = tables["stop"]
    placed = stops["well_visit_id"].notna()
    assert (
        stops.loc[placed, "well_visit_id"] == stops.loc[placed, "overlapping_visit_id"]
    ).all()


def test_spans_visit_end_current_arm_is_the_visit_well(tables):
    events = tables["event"]
    crossing = events.iloc[1]
    assert crossing["occupancy_state"] == "mixed"
    assert np.isnan(crossing["occupancy_well_id"])
    # Visit 1 is at well 1, on arm 1 of patch 1; the next trip goes to well 2
    # (patch 2), so the animal does not stay.
    assert crossing["occupancy_well_arm"] == 1.0
    assert crossing["current_patch"] == 1.0
    assert crossing["is_stay"] == 0.0
    arms = tables["event_arm"]
    rows = arms["event_id"].eq(crossing["event_id"])
    assert arms.loc[rows, "is_current_arm"].tolist() == [
        float(arm == 1) for arm in range(9)
    ]
    patches = tables["event_patch"]
    rows = patches["event_id"].eq(crossing["event_id"])
    assert patches.loc[rows].sort_values("patch")["is_current_patch"].tolist() == [
        1.0,
        0.0,
        0.0,
    ]


def test_spans_visit_end_stop_takes_the_visit_well(tables):
    """A stop that starts inside a visit and crosses its final poke-out takes
    the visit's well, as its events do: stop 2 starts in visit 1 (well 1, arm
    1, patch 1) and the next trip goes to well 2 (patch 2). The stop that
    starts before a poke-in keeps its whole-window location (none)."""
    stops = tables["stop"]
    assert stops["occupancy_state"].tolist()[2] == "mixed"
    assert np.isnan(stops["occupancy_well_id"].iloc[2])
    np.testing.assert_array_equal(
        stops["occupancy_well_arm"], [1.0, 1.0, 1.0, np.nan, np.nan]
    )
    np.testing.assert_array_equal(
        stops["current_patch"], [1.0, 1.0, 1.0, np.nan, np.nan]
    )
    np.testing.assert_array_equal(stops["is_stay"], [0.0, 0.0, 0.0, np.nan, np.nan])
    # the stop and the event it holds agree
    event = tables["event"].iloc[1]
    stop = stops.loc[stops["stop_id"] == event["stop_id"]].iloc[0]
    for column in ("occupancy_well_arm", "current_patch", "is_stay"):
        assert stop[column] == event[column], column


def test_in_visit_immobile_seconds():
    starts = np.array([s for s, _ in _STOPS])
    ends = np.array([e for _, e in _STOPS])
    columns = stop_visit_columns(starts, ends, _experiences(), _no_excluded_time())
    # The crossing stop counts to the poke-out (6.0 to 6.5), the stop before
    # the first poke-in from it (8.0 to 8.5), the stop between visits nothing.
    np.testing.assert_allclose(
        columns["in_visit_immobile_s"], [0.6, 0.6, 0.5, 0.0, 0.5], atol=1e-12
    )
    excluded = pd.DataFrame({"start_time": [3.5, 6.4], "end_time": [3.6, 7.0]})
    columns = stop_visit_columns(starts, ends, _experiences(), excluded)
    np.testing.assert_allclose(
        columns["in_visit_immobile_s"], [0.5, 0.6, 0.4, 0.0, 0.5], atol=1e-12
    )
    assert list(columns.columns) == list(STOP_VISIT_COLUMNS)


def test_a_stop_across_two_visits_is_placed_in_the_later():
    experiences = _experiences()
    columns = stop_visit_columns(
        np.array([0.5]), np.array([3.5]), experiences, _no_excluded_time()
    )
    assert columns["overlapping_visit_id"].tolist() == [1]
    assert columns["stop_end_kind"].tolist() == ["leaves"]
    # 0.5 s in the starting visit and 0.5 s in visit 1.
    np.testing.assert_allclose(columns["in_visit_immobile_s"], [1.0])


def _stretches(columns: pd.DataFrame) -> list[int]:
    """The stretch indices, -1 where missing."""
    return columns["visit_stretch_index"].fillna(-1).astype(int).tolist()


def test_stretches_are_the_stops_with_analyzed_time_in_the_visit():
    """Only a stop with analyzed seconds inside its visit is one of the visit's
    stretches: a stop that ends on the first poke-in, or lies wholly in
    excluded time, is ``outside_visit`` with no stretch index, and does not
    take the visit's ``leaves``."""
    starts = np.array([2.0, 3.3, 4.0, 5.0])
    ends = np.array([3.0, 3.6, 4.5, 6.0])
    # stop 2 (4.0 - 4.5) lies wholly in excluded time
    excluded = pd.DataFrame({"start_time": [3.9], "end_time": [4.6]})
    columns = stop_visit_columns(starts, ends, _experiences(), excluded)
    np.testing.assert_allclose(columns["in_visit_immobile_s"], [0.0, 0.3, 0.0, 1.0])
    assert columns["overlapping_visit_id"].tolist() == [1, 1, 1, 1]
    assert _stretches(columns) == [-1, 0, -1, 1]
    assert columns["stop_end_kind"].tolist() == [
        "outside_visit",
        "small_movement",
        "outside_visit",
        "leaves",
    ]
    # the visit's last stop wholly in excluded time: the stop before it leaves
    excluded = pd.DataFrame({"start_time": [4.9], "end_time": [6.1]})
    columns = stop_visit_columns(starts[:2], ends[:2], _experiences(), excluded)
    assert _stretches(columns) == [-1, 0]
    columns = stop_visit_columns(
        np.array([3.3, 5.0]), np.array([3.6, 6.0]), _experiences(), excluded
    )
    assert _stretches(columns) == [0, -1]
    assert columns["stop_end_kind"].tolist() == ["leaves", "outside_visit"]


def test_a_stop_across_two_visits_with_no_time_in_the_later():
    """A stop placed in the later of two visits with no analyzed second in it
    is a stretch of neither: its seconds in the earlier visit still count in
    ``in_visit_immobile_s``."""
    columns = stop_visit_columns(
        np.array([0.5]), np.array([3.0]), _experiences(), _no_excluded_time()
    )
    assert columns["overlapping_visit_id"].tolist() == [1]
    np.testing.assert_allclose(columns["in_visit_immobile_s"], [0.5])
    assert _stretches(columns) == [-1]
    assert columns["stop_end_kind"].tolist() == ["outside_visit"]


def test_event_visit_clocks(tables):
    events = tables["event"]
    # Event 1 starts at 4.1 s in stop 1 (ends 4.6 s); event 2 at 6.3 s in
    # stop 2, which ends at 6.8 s, after the 6.5 s poke-out.
    np.testing.assert_allclose(events["time_since_visit_start_s"], [1.1, 3.3])
    np.testing.assert_allclose(events["time_until_visit_end_s"], [2.4, 0.2])
    np.testing.assert_allclose(events["time_until_stretch_end_s"], [0.5, 0.2])
    # The starting port and visit 1 have begun by both starts.
    assert events["n_visits_arrived"].tolist() == [2, 2]
    for name in ("event_arm", "event_patch"):
        by_event = tables[name].groupby("event_id")[list(EVENT_VISIT_COLUMNS)].first()
        pd.testing.assert_frame_equal(
            by_event.sort_index(),
            events.set_index("event_id")[list(EVENT_VISIT_COLUMNS)].sort_index(),
            check_dtype=False,
        )

    clocks = event_visit_columns(
        np.array([0.5, 2.0, 8.5]),
        np.array([0.8, 2.2, 8.9]),
        pd.array([0, pd.NA, 2], dtype="Int64"),
        _experiences(),
    )
    np.testing.assert_allclose(
        clocks["time_since_visit_start_s"], [0.5, np.nan, 0.5], atol=1e-12
    )
    np.testing.assert_allclose(
        clocks["time_until_visit_end_s"], [0.5, np.nan, 0.5], atol=1e-12
    )
    np.testing.assert_allclose(
        clocks["time_until_stretch_end_s"], [0.3, np.nan, 0.4], atol=1e-12
    )
    assert clocks["n_visits_arrived"].tolist() == [1, 1, 3]


def _hand_visits() -> pd.DataFrame:
    """Visits to wells 2, 0, 2, 2, 5, each outcome known 0.3 to 0.5 s in."""
    return pd.DataFrame(
        {
            "experience_id": [0, 1, 2, 3, 4],
            "movement_trial_number": pd.array([pd.NA, 0, 1, 2, 3], dtype="Int64"),
            "is_initial_experience": [True, False, False, False, False],
            "is_boundary_derived": [False, False, False, False, False],
            "outcome_observed": [True, True, True, True, True],
            "port": [2, 0, 2, 2, 5],
            "patch": [2, 1, 2, 2, 3],
            "arrival_time": [0.0, 10.0, 20.0, 30.0, 40.0],
            "departure_time": [5.0, 15.0, 25.0, 35.0, 45.0],
            "reward_on_time": [np.nan, 10.5, np.nan, 30.5, np.nan],
            "outcome_known_time": [0.0, 10.5, 20.3, 30.5, 40.3],
            "is_reward": [False, True, False, True, False],
        }
    )


def test_sample_count_and_trial_number():
    graph = _bandit_graph()
    times = np.array([1.0, 12.0, 31.0, 50.0])
    counts = log1p_known_visit_count_all_events(
        times, _hand_visits(), build_well_to_arm_map(graph)
    )
    # Known visits by arm (wells 0-5 on arms 0-5): at 1 s, well 2 once; at
    # 12 s, wells 2 and 0 once; at 31 s, well 2 three times and well 0 once;
    # at 50 s, also well 5 once.
    expected = np.zeros((4, 9))
    expected[0, 2] = 1
    expected[1, [0, 2]] = 1
    expected[2, 0], expected[2, 2] = 1, 3
    expected[3, 0], expected[3, 2], expected[3, 5] = 1, 3, 1
    np.testing.assert_allclose(counts, np.log1p(expected))
    clocks = event_visit_columns(
        times, times + 1.0, pd.array([pd.NA] * 4, dtype="Int64"), _hand_visits()
    )
    assert clocks["n_visits_arrived"].tolist() == [1, 2, 4, 5]


def test_sample_count_uses_known_outcomes_only(tables):
    graph = _bandit_graph()
    well_to_arm = build_well_to_arm_map(graph)
    # Visit 3 (well 2) arrives at 30.0 s; its outcome is known at 30.5 s.
    counts = log1p_known_visit_count_all_events(
        np.array([30.2, 30.5]), _hand_visits(), well_to_arm
    )
    np.testing.assert_allclose(counts[:, 2], np.log1p([2, 3]))
    # Trunk arms are no reward site: zero.
    np.testing.assert_array_equal(counts[:, 6:], 0.0)

    # In the session, both events start after the starting port's outcome
    # (known at 1.0 s) and visit 1's (known at 3.3 s): one known visit each
    # to wells 0 and 1.
    arms = tables["event_arm"]
    by_arm = arms.pivot(
        index="event_id",
        columns="candidate_arm",
        values="log1p_n_known_visits_candidate_arm",
    )
    expected = np.zeros(9)
    expected[[0, 1]] = np.log1p(1)
    for row in by_arm.to_numpy():
        np.testing.assert_allclose(row, expected)
