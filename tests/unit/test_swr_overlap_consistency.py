"""Regression test: the two SWR-overlap classifiers must produce the
same count distribution for any interval geometry.

``classify_swr_overlap`` (event_detection.py) works on per-event
intervals and asks: does any single SWR fully span each event.
``compute_swr_overlap_class_counts`` (data_aggregation.py) works on
per-timebin data and asks: does every timebin of the event overlap
some SWR.

These differ on the ``multi_swr_no_gap`` geometry — an event tiled by
two consecutive SWRs with no gap between them. The event-level
classifier calls that "partial" (no single SWR spans the event); the
timebin-level classifier calls it "contained" (every timebin is
covered by some SWR). Both functions feed Figure 2, so the
inconsistency showed up as disagreement between the example panel
and the population bar chart on events with this geometry.

After Wave 2 Task 5 reunifies the two functions around the event-level
definition, both produce the same counts for every input.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.hippocampus.data_aggregation import (
    compute_swr_overlap_class_counts,
)
from continuum_swr_replay.analysis.hippocampus.event_detection import (
    classify_swr_overlap,
)


def _build_data_over_time(
    events: pd.DataFrame,
    swrs: pd.DataFrame,
    t_start: float = -1.0,
    t_end: float = 5.0,
    dt: float = 0.001,
) -> pd.DataFrame:
    """Render interval tables as a per-timebin frame.

    Produces the schema consumed by
    :func:`compute_swr_overlap_class_counts`: ``time``,
    ``non_local_event`` (0 outside events, 1..N inside),
    ``SWR_event`` (0 outside SWRs, 1..M inside), ``head_speed``
    (all zeros so every event passes the immobility filter), and the
    ``animal`` / ``day`` / ``epoch`` grouping columns.
    """
    time = np.arange(t_start, t_end, dt)
    non_local_event = np.zeros(len(time), dtype=int)
    swr_event = np.zeros(len(time), dtype=int)

    for i, ev in events.reset_index(drop=True).iterrows():
        mask = (time >= ev["start_time"]) & (time < ev["end_time"])
        non_local_event[mask] = i + 1
    for j, swr in swrs.reset_index(drop=True).iterrows():
        mask = (time >= swr["start_time"]) & (time < swr["end_time"])
        swr_event[mask] = j + 1

    return pd.DataFrame(
        {
            "time": time,
            "non_local_event": non_local_event,
            "SWR_event": swr_event,
            "head_speed": np.zeros(len(time)),
            "animal": "test",
            "day": 1,
            "epoch": 1,
        }
    )


def _counts_from_events(events_df: pd.DataFrame, swrs_df: pd.DataFrame) -> dict:
    """Per-class counts from the event-level classifier."""
    classified = classify_swr_overlap(events_df, swrs_df)
    return classified["overlap_class"].value_counts().to_dict()


def _counts_from_timebins(events_df: pd.DataFrame, swrs_df: pd.DataFrame) -> dict:
    """Per-class counts from the timebin-level classifier, stripped of
    the ``total`` key and zero entries so the shape matches the
    event-level counts dict."""
    dot = _build_data_over_time(events_df, swrs_df)
    raw = compute_swr_overlap_class_counts(dot)
    return {k: v for k, v in raw.items() if k != "total" and v > 0}


@pytest.mark.parametrize(
    "label,events,swrs",
    [
        (
            "swr_contains_event",
            [(1.0, 2.0)],
            [(0.5, 2.5)],
        ),
        (
            "event_contains_swr",
            [(0.5, 2.5)],
            [(1.0, 2.0)],
        ),
        (
            "single_swr_partial",
            [(1.0, 2.0)],
            [(1.5, 2.5)],
        ),
        (
            "multi_swr_no_gap",  # the disagreement case
            [(1.0, 2.0)],
            [(0.9, 1.5), (1.5, 2.1)],
        ),
        (
            "no_overlap",
            [(1.0, 2.0)],
            [(3.0, 4.0)],
        ),
        (
            "mixed_batch",
            [(1.0, 2.0), (2.5, 3.0), (3.5, 4.5)],
            [(0.5, 2.5), (2.6, 2.8), (5.0, 5.5)],
        ),
    ],
)
def test_event_and_timebin_classifiers_agree(label, events, swrs):
    """Both classifiers must produce the same count distribution."""
    events_df = pd.DataFrame(events, columns=["start_time", "end_time"])
    swrs_df = pd.DataFrame(swrs, columns=["start_time", "end_time"])

    event_counts = _counts_from_events(events_df, swrs_df)
    timebin_counts = _counts_from_timebins(events_df, swrs_df)

    assert event_counts == timebin_counts, (
        f"[{label}] event-level={event_counts}, "
        f"timebin-level={timebin_counts}"
    )
