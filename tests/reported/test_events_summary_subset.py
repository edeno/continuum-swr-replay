"""The events summary reads the events that start inside a well visit, and counts
every well-visit label, kept and left out."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.constants import VISIT_CONTEXTS
from continuum_swr_replay.analysis.event_definition.cohort_selection import (
    ANALYZED_SUBSET,
    count_visit_context,
    select_visit_context,
)
from continuum_swr_replay.reported import events as builder
from tests.reported.test_reported_events_values import _by_key, _events


def _labeled() -> pd.DataFrame:
    """The thirteen events of the builders' tests, all starting inside a visit,
    and one more event of every label, each an SWR-partial event of animal c,
    which would change every class count and share if it were read."""
    inside = _events().assign(
        event_id=np.arange(13, dtype="uint64"),
        visit_context=["in_visit"] * 12 + ["spans_visit_end"],
    )
    extra = inside.iloc[[12] * len(VISIT_CONTEXTS)].assign(
        event_id=np.arange(100, 100 + len(VISIT_CONTEXTS), dtype="uint64"),
        visit_context=list(VISIT_CONTEXTS),
    )
    return pd.concat([inside, extra], ignore_index=True)


def test_the_label_counts_are_count_visit_context():
    table = _labeled()
    values = _by_key(builder.visit_context_values(table, "s"))
    counts = count_visit_context(table)
    for label in VISIT_CONTEXTS:
        assert values[f"events_label_{label}"].value == counts.loc[label, "n"]
    assert values["events_label_in_visit"].value == 13
    assert values["events_label_spans_visit_end"].value == 2
    assert values["events_label_track_center"].value == 1
    assert values["events_in_subset"].value == 15
    assert values["events_left_out"].value == len(VISIT_CONTEXTS) - 2
    # the extra events are SWR-partial: every left-out one is an SWR event
    assert values["swr_events_in_subset"].value == 6 + 2
    assert values["swr_events_left_out"].value == len(VISIT_CONTEXTS) - 2
    assert values["non_swr_events_in_subset"].value == 7
    assert values["non_swr_events_left_out"].value == 0
    assert "kept" in values["events_label_in_visit"].description
    assert "left out" in values["events_label_track_leaf_arm"].description


class _Read(Exception):
    """Stops the summary after the event table's values."""


def test_summary_reads_the_selected_events(monkeypatch):
    table = _labeled()
    seen: dict[str, pd.DataFrame] = {}

    def stop(events, folder, n_epochs):
        seen["extras"] = events
        raise _Read

    monkeypatch.setattr(builder, "read_swr_table", lambda path: table)
    monkeypatch.setattr(
        builder, "file_record", lambda path, root: {"path": "p", "sha256": "0"}
    )
    monkeypatch.setattr(builder, "with_start_and_rest_theta", stop)
    # (the state shares are not columns of these events)
    monkeypatch.setattr(builder, "state_composition_values", lambda events, s: [])
    added = []
    monkeypatch.setattr(
        builder.Summary, "add", lambda self, *values: added.extend(values)
    )
    with pytest.raises(_Read):
        builder.summarize(builder.default_inputs(), "by")

    selected = select_visit_context(table)
    assert seen["extras"]["event_id"].tolist() == selected["event_id"].tolist()
    values = _by_key(added)
    # the class counts of the selected rows: the thirteen and the two extra
    # events labeled in_visit and spans_visit_end (both SWR-partial)
    assert values["nonlocal_events"].value == 15
    assert values["nonlocal_events_non_swr"].value == 7
    assert values["nonlocal_events_swr_partial"].value == 6
    assert values["events_label_before_first_visit"].value == 1
    for key in ("nonlocal_events", "swr_overlapping_events"):
        assert values[key].description.endswith(f"; over {builder.EVENTS_SUBSET}")
    assert "; over" not in values["events_label_in_visit"].description


def test_content_bins_are_read_for_the_selected_events_only(tmp_path: Path):
    events = select_visit_context(_labeled())
    rows = []
    for event_id in [*events["event_id"], 100 + 5]:  # 105 is not selected
        rows += [{"event_id": event_id, "epoch_id": "e", "arm": 1}] * 2
    pd.DataFrame(rows).to_parquet(tmp_path / "e.parquet")

    bins, _ = builder.content_along_arms_bins(events, tmp_path, 1, ["arm"])

    assert set(bins["event_id"]) == set(events["event_id"])
    assert len(bins) == 2 * len(events)
    with pytest.raises(ValueError, match="do not cover"):
        builder.content_along_arms_bins(
            events.iloc[:-1].assign(
                event_id=lambda t: t["event_id"].where(t.index > 0, 999)
            ),
            tmp_path,
            1,
            ["arm"],
        )


def test_an_output_not_made_over_the_subset_is_pending():
    assert builder.subset_staleness({"subset": ANALYZED_SUBSET}, "out") is None
    assert "rerun it" in builder.subset_staleness({}, "out")
    assert "rerun it" in builder.subset_staleness({"subset": "every event"}, "out")
