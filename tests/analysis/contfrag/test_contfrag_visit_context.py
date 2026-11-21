"""The ContFrag tables carry the well-visit label (``event_tables/visit_context.py``).

The session is the non-local tables' location fixture
(``test_replay_tables._location_contract_session``) with its three events
copied into the SWR and MUA labels: one in transit 10 cm from the well just
left, one inside a well visit, one that starts in transit and crosses a
poke-in. Each event is its own immobility bout, so the stops carry the same
three labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.contfrag import tables as contfrag_tables
from continuum_swr_replay.analysis.contfrag.prepare_data import (
    contfrag_event_patch_table,
)
from continuum_swr_replay.analysis.event_tables.table_contracts import (
    require_visit_context_labels,
)
from continuum_swr_replay.analysis.event_tables.visit_context import (
    EVENT_VISIT_COLUMNS,
    STOP_VISIT_COLUMNS,
    VISIT_CONTEXT_COLUMNS,
)
from tests.analysis.event_tables.test_replay_tables import (
    _location_contract_session,
    _tiny_posterior,
    build_event_arm_table,
    build_event_table,
    build_stop_table,
)
from tests.analysis.event_tables.test_visit_columns import visit_session
from tests.support._port_experience_fixtures import port_experiences_from_trials

_EXPECTED = ["after_visit_at_well", "in_visit", "spans_visit_start"]


def _contfrag_tables(event_source: str) -> dict[str, pd.DataFrame]:
    """The four ContFrag tables of the fixture session, rows in time order."""
    dot, env, trials, graph = _location_contract_session()
    dot["SWR_event"] = dot["non_local_event"]
    dot["MUA_event"] = dot["non_local_event"]
    dot["Continuous_prob"] = 0.7
    dot["Fragmented_prob"] = 0.3
    experiences = port_experiences_from_trials(trials, prepend_initial=True)
    posterior = _tiny_posterior(dot)
    event = contfrag_tables.build_contfrag_event_table(
        dot,
        env,
        trials,
        experiences,
        graph,
        event_source=event_source,
        conditional_posterior=posterior,
    )
    arm = contfrag_tables.build_contfrag_event_arm_table(
        event,
        dot,
        env,
        trials,
        experiences,
        graph,
        event_source=event_source,
        conditional_posterior=posterior,
    )
    stop = contfrag_tables.build_contfrag_stop_table(
        dot, trials, experiences, graph, event_source=event_source
    )
    patch = contfrag_event_patch_table(
        event, arm, event_source=event_source, brain_area="HPC"
    )
    return {
        "event": event.sort_values("event_start_s", ignore_index=True),
        "event_arm": arm,
        "stop": stop.sort_values("stop_start", ignore_index=True),
        "event_patch": patch,
    }


@pytest.mark.parametrize("event_source", ["SWR", "MUA"])
def test_events_and_stops_are_labeled_by_hand(event_source):
    tables = _contfrag_tables(event_source)
    events, stops = tables["event"], tables["stop"]

    assert events["visit_context"].tolist() == _EXPECTED
    assert events["occupancy_state"].tolist() == ["in_transit", "at_well", "mixed"]
    assert str(events["well_visit_id"].dtype) == "Int64"
    assert events["well_visit_id"].isna().tolist() == [True, False, True]
    assert events["well_visit_id"].iloc[1] == 1
    # Event 1 starts at 2.0 s, 1 s after the initial visit's final poke-out
    # at 1.0 s; its last sample is 2.3 s (the grid's 2.4 lies just above the
    # fixture's 2.4 s bound), 0.7 s before the 3.0 s poke-in.
    first = events.iloc[0]
    assert first["event_end_s"] == pytest.approx(2.3)
    assert first["last_completed_well_visit_id"] == 0
    assert first["time_since_last_poke_out_s"] == pytest.approx(1.0)
    assert first["time_until_next_poke_in_s"] == pytest.approx(0.7)
    for row in (1, 2):
        assert np.isnan(events["time_since_last_poke_out_s"].iloc[row])
        assert np.isnan(events["time_until_next_poke_in_s"].iloc[row])

    assert stops["visit_context"].tolist() == _EXPECTED
    for name, table in tables.items():
        require_visit_context_labels(table, name)


@pytest.mark.parametrize("event_source", ["SWR", "MUA"])
def test_the_labels_are_the_nonlocal_tables_on_the_same_session(event_source):
    tables = _contfrag_tables(event_source)
    dot, env, trials, graph = _location_contract_session()
    nonlocal_events = build_event_table(dot, env, trials, graph).sort_values(
        "event_start_s", ignore_index=True
    )
    nonlocal_stops = build_stop_table(dot, trials, graph).sort_values(
        "stop_start", ignore_index=True
    )
    columns = list(VISIT_CONTEXT_COLUMNS)

    pd.testing.assert_frame_equal(
        tables["event"][columns], nonlocal_events[columns], check_exact=True
    )
    pd.testing.assert_frame_equal(
        tables["stop"][columns], nonlocal_stops[columns], check_exact=True
    )


@pytest.mark.parametrize("event_source", ["SWR", "MUA"])
def test_the_labels_reach_the_arm_and_patch_tables(event_source):
    tables = _contfrag_tables(event_source)
    expected = tables["event"].set_index("event_id")[list(VISIT_CONTEXT_COLUMNS)]
    for name in ("event_arm", "event_patch"):
        by_event = tables[name].groupby("event_id")[list(VISIT_CONTEXT_COLUMNS)].first()
        pd.testing.assert_frame_equal(
            by_event.sort_index(), expected.sort_index(), check_exact=True
        )


@pytest.mark.parametrize("event_source", ["SWR", "MUA"])
def test_the_label_sits_after_the_location_columns(event_source):
    tables = _contfrag_tables(event_source)
    for name in ("event", "event_arm", "stop"):
        columns = list(tables[name].columns)
        start = columns.index("nearest_well_distance_cm_max") + 1
        assert columns[start : start + len(VISIT_CONTEXT_COLUMNS)] == list(
            VISIT_CONTEXT_COLUMNS
        ), name


#: The visit clocks are read from the visit the label places an event in.
_LABEL_DERIVED_CLOCKS = (
    "time_since_visit_start_s",
    "time_until_visit_end_s",
    "time_until_stretch_end_s",
)


@pytest.mark.parametrize("event_source", ["SWR", "MUA"])
def test_the_label_changes_no_other_column(event_source, monkeypatch):
    """A stand-in labeler gives the same other columns, rows and ids."""
    with_label = _contfrag_tables(event_source)

    def _stand_in(samples, sample_window, windows, occupancy_state, *args):
        at_well = np.asarray(occupancy_state, dtype=object) == "at_well"
        frame = pd.DataFrame(
            {column: [np.nan] * len(windows) for column in VISIT_CONTEXT_COLUMNS}
        )
        frame["visit_context"] = np.where(at_well, "in_visit", "track_no_position")
        return frame

    monkeypatch.setattr(contfrag_tables, "visit_context_of_samples", _stand_in)
    without = _contfrag_tables(event_source)
    assert with_label["event"]["visit_context"].tolist() != (
        without["event"]["visit_context"].tolist()
    )
    for name, table in with_label.items():
        other = [
            column
            for column in table
            if column not in (*VISIT_CONTEXT_COLUMNS, *_LABEL_DERIVED_CLOCKS)
        ]
        assert list(without[name].columns) == list(table.columns)
        pd.testing.assert_frame_equal(
            table[other], without[name][other], check_exact=True
        )


def test_the_contract_refuses_a_table_without_the_label():
    table = pd.DataFrame(
        {"occupancy_state": ["at_well"], "visit_context": ["in_visit"]}
    )
    require_visit_context_labels(table, "t")
    for column in ("visit_context", "occupancy_state"):
        with pytest.raises(ValueError, match=column):
            require_visit_context_labels(table.drop(columns=column), "t")
    with pytest.raises(ValueError, match="visit_context"):
        require_visit_context_labels(table.drop(columns="visit_context").iloc[:0], "t")


@pytest.mark.parametrize("event_source", ["SWR", "MUA"])
def test_the_visit_columns_are_the_nonlocal_tables_on_the_same_session(event_source):
    """Same stop visit columns, visit clocks, current arm and sample count.

    The session (``test_visit_columns.visit_session``) has an event inside a
    visit and one that crosses the final poke-out.
    """
    dot, env, trials, graph = visit_session()
    nonlocal_events = build_event_table(dot, env, trials, graph).sort_values(
        "event_start_s", ignore_index=True
    )
    nonlocal_arms = build_event_arm_table(nonlocal_events, dot, env, trials, graph)
    nonlocal_stops = build_stop_table(dot, trials, graph).sort_values(
        "stop_start", ignore_index=True
    )
    dot["SWR_event"] = dot["non_local_event"]
    dot["MUA_event"] = dot["non_local_event"]
    dot["Continuous_prob"] = 0.7
    dot["Fragmented_prob"] = 0.3
    experiences = port_experiences_from_trials(trials, prepend_initial=True)
    events = contfrag_tables.build_contfrag_event_table(
        dot, env, trials, experiences, graph, event_source=event_source
    ).sort_values("event_start_s", ignore_index=True)
    arms = contfrag_tables.build_contfrag_event_arm_table(
        events, dot, env, trials, experiences, graph, event_source=event_source
    )
    stops = contfrag_tables.build_contfrag_stop_table(
        dot, trials, experiences, graph, event_source=event_source
    ).sort_values("stop_start", ignore_index=True)

    assert events["visit_context"].tolist() == ["in_visit", "spans_visit_end"]
    event_columns = [
        *EVENT_VISIT_COLUMNS,
        "well_visit_id",
        "occupancy_well_arm",
        "current_patch",
        "is_stay",
    ]
    pd.testing.assert_frame_equal(
        events[event_columns], nonlocal_events[event_columns], check_exact=True
    )
    pd.testing.assert_frame_equal(
        stops[list(STOP_VISIT_COLUMNS)],
        nonlocal_stops[list(STOP_VISIT_COLUMNS)],
        check_exact=True,
    )
    arm_columns = [
        "candidate_arm",
        "is_current_arm",
        "log1p_n_known_visits_candidate_arm",
        *EVENT_VISIT_COLUMNS,
    ]
    pd.testing.assert_frame_equal(
        arms[arm_columns].reset_index(drop=True),
        nonlocal_arms[arm_columns].reset_index(drop=True),
        check_exact=True,
    )
