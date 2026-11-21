"""Selecting rows by their well-visit label (``cohort_selection``)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.constants import (
    STARTS_IN_WELL_VISIT,
    VISIT_CONTEXTS,
)
from continuum_swr_replay.analysis.event_definition.cohort_selection import (
    count_visit_context,
    select_events_in,
    select_visit_context,
)


def _events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": [11, 12, 13, 14, 15],
            "visit_context": [
                "in_visit",
                "after_visit_at_well",
                "in_visit",
                "track_center",
                "spans_visit_end",
            ],
        },
        index=[5, 6, 7, 8, 9],
    )


def test_default_keeps_the_rows_that_start_inside_a_visit():
    events = _events()
    selected = select_visit_context(events)
    # in_visit (11, 13) and spans_visit_end (15) are kept; the rows that
    # start outside a visit (12, 14) are dropped.
    assert selected["event_id"].tolist() == [11, 13, 15]
    assert selected.index.tolist() == [0, 1, 2]
    assert set(STARTS_IN_WELL_VISIT) == {"in_visit", "spans_visit_end"}
    assert len(events) == 5  # the input is not changed


def test_a_caller_can_keep_only_the_rows_wholly_inside_a_visit():
    assert select_visit_context(_events(), "in_visit")["event_id"].tolist() == [
        11,
        13,
    ]
    counts = count_visit_context(_events(), "in_visit")
    assert counts.loc[counts["kept"]].index.tolist() == ["in_visit"]


def test_named_contexts_are_kept_together():
    selected = select_visit_context(_events(), ["in_visit", "after_visit_at_well"])
    assert selected["event_id"].tolist() == [11, 12, 13]
    assert select_visit_context(_events(), "track_center")["event_id"].tolist() == [14]


def test_refuses_a_table_without_the_label_or_an_unknown_label():
    with pytest.raises(ValueError, match="visit_context"):
        select_visit_context(_events().drop(columns="visit_context"))
    with pytest.raises(ValueError, match="unknown visit_context"):
        select_visit_context(_events(), ["in_visit", "hallway"])
    with pytest.raises(ValueError, match="unknown visit_context"):
        select_visit_context(_events(), [])


def test_counts_every_label_kept_and_dropped():
    counts = count_visit_context(_events())
    assert counts.index.tolist() == list(VISIT_CONTEXTS)
    assert counts.loc["in_visit", "n"] == 2
    assert counts.loc["after_visit_at_well", "n"] == 1
    assert counts.loc["before_first_visit", "n"] == 0
    assert counts["n"].sum() == 5
    assert counts.loc[counts["kept"], "n"].sum() == len(select_visit_context(_events()))
    assert counts.loc[counts["kept"]].index.tolist() == [
        "in_visit",
        "spans_visit_end",
    ]


def test_counts_units_on_a_candidate_row_table():
    # Two candidate rows per event: rows double, events do not.
    long = _events().loc[_events().index.repeat(2)]
    by_row = count_visit_context(long, ["in_visit", "track_center"])
    by_event = count_visit_context(
        long, ["in_visit", "track_center"], unit_col="event_id"
    )
    assert by_row.loc["in_visit", "n"] == 4
    assert by_event.loc["in_visit", "n"] == 2
    assert by_event.loc["track_center", "kept"]
    with pytest.raises(ValueError, match="stop_id"):
        count_visit_context(long, unit_col="stop_id")


def test_select_events_in_keeps_every_row_of_a_selected_event():
    # An event row and its matched control row share one event_id.
    table = pd.DataFrame(
        {
            "event_id": [11, 11, 12, 12, 13, 15, 99],
            "kind": ["event", "control"] * 2 + ["event"] * 3,
        },
        index=[3, 4, 5, 6, 7, 8, 9],
    )
    selected = select_events_in(table, select_visit_context(_events())["event_id"])
    assert selected["event_id"].tolist() == [11, 11, 13, 15]
    assert selected["kind"].tolist() == ["event", "control", "event", "event"]
    assert selected.index.tolist() == [0, 1, 2, 3]
    assert len(table) == 7  # the input is not changed
    assert select_events_in(table, []).empty


def test_select_events_in_refuses_a_table_without_event_id():
    with pytest.raises(ValueError, match="event_id"):
        select_events_in(pd.DataFrame({"stop_id": [1]}), [1])


def test_an_output_not_made_over_the_subset_is_named():
    from continuum_swr_replay.analysis.event_definition.cohort_selection import (
        ANALYZED_SUBSET,
        subset_staleness,
    )

    assert subset_staleness({"subset": ANALYZED_SUBSET}, "out") is None
    for manifest in ({}, {"subset": "every event"}):
        note = subset_staleness(manifest, "out")
        assert note.startswith("out was not made over")


def _timed_events() -> pd.DataFrame:
    """Two epochs' events: a and b each with one event inside its visit
    (0 to 10 s) and one after it."""
    return pd.DataFrame(
        {
            "animal": ["chimi"] * 4,
            "session_date": [20200212] * 4,
            "epoch": ["02_r1", "02_r1", "04_r2", "04_r2"],
            "event_start_s": [1.0, 12.0, 9.5, 10.0],
            "visit_context": [
                "in_visit",
                "after_visit_at_well",
                "spans_visit_end",
                "track_center",
            ],
        }
    )


_VISITS = {
    "chimi20200212_02_r1_HPC": (np.array([0.0]), np.array([10.0])),
    "chimi20200212_04_r2_HPC": (np.array([0.0]), np.array([10.0])),
}


def test_labels_that_agree_with_the_visits_pass():
    from continuum_swr_replay.analysis.event_definition.cohort_selection import (
        check_visit_context_starts,
    )

    check_visit_context_starts(_timed_events(), _VISITS)


def test_a_label_that_disagrees_with_the_visits_is_refused_with_the_count():
    from continuum_swr_replay.analysis.event_definition.cohort_selection import (
        check_visit_context_starts,
    )

    events = _timed_events()
    # the departure is not inside the visit: an event starting there is outside
    events.loc[3, "visit_context"] = "in_visit"
    events.loc[1, "visit_context"] = "spans_visit_end"
    with pytest.raises(ValueError, match="2 of 4 events.*12.0 s.*outside"):
        check_visit_context_starts(events, _VISITS)
    with pytest.raises(ValueError, match="no well visits for 1 epochs"):
        check_visit_context_starts(
            _timed_events(),
            {"chimi20200212_02_r1_HPC": _VISITS["chimi20200212_02_r1_HPC"]},
        )
