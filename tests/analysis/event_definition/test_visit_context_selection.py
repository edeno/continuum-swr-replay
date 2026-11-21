"""Selecting rows by their well-visit label (``cohort_selection``)."""

from __future__ import annotations

import pandas as pd
import pytest

from continuum_swr_replay.analysis.constants import VISIT_CONTEXTS
from continuum_swr_replay.analysis.event_definition.cohort_selection import (
    count_visit_context,
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


def test_default_keeps_only_rows_inside_a_visit():
    events = _events()
    selected = select_visit_context(events)
    assert selected["event_id"].tolist() == [11, 13]
    assert selected.index.tolist() == [0, 1]
    assert len(events) == 5  # the input is not changed


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
    assert counts["kept"].sum() == 1


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
