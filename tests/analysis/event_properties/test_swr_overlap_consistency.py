"""Regression test: the table-driven overlap counts agree with the
direct event-level classifier for every interval geometry.

``classify_swr_overlap`` (events/nonlocal_detection.py) classifies per-event
intervals directly. ``compute_swr_overlap_class_counts``
(data_aggregation.py) now tallies the ``overlap_class`` column of
``build_classified_nonlocal_event_table``, which itself classifies via
``classify_swr_overlap_geometry`` — the same event-level,
strict-positive-duration rule. Both therefore use a single overlap
convention and must produce identical counts for any input, including the
``multi_swr_no_gap`` geometry (an event tiled by two consecutive SWRs)
that the old timebin-level counter disagreed on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_properties.event_statistics import (
    compute_swr_overlap_class_counts,
)
from continuum_swr_replay.analysis.events.classified_table import (
    build_classified_nonlocal_event_table,
)
from continuum_swr_replay.analysis.events.nonlocal_detection import (
    classify_swr_overlap,
)


def _build_data_over_time(
    events: pd.DataFrame,
    swrs: pd.DataFrame,
    t_start: float = -1.0,
    t_end: float = 5.0,
    dt: float = 0.001,
) -> pd.DataFrame:
    """Render interval tables as a time-indexed per-timebin frame.

    Produces the schema consumed by
    :func:`build_classified_nonlocal_event_table`: a ``time`` index,
    ``non_local_event`` (0 outside events, 1..N inside), ``SWR_event``
    (0 outside SWRs, 1..M inside), ``head_speed`` (all zeros so every
    event passes the whole-event immobility filter), the two non-local
    probability columns, and the ``animal`` / ``day`` / ``epoch``
    grouping columns.
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
            "non_local_event": non_local_event,
            "SWR_event": swr_event,
            "head_speed": np.zeros(len(time)),
            # peak_nonlocal_prob is unused by the counter, but the builder
            # requires the columns; mark event timebins as high-prob.
            "Non-Local_Continuous_prob": (non_local_event > 0).astype(float),
            "Non-Local_Fragmented_prob": np.zeros(len(time)),
            "animal": "test",
            "day": 1,
            "epoch": 1,
        },
        index=pd.Index(time, name="time"),
    )


def _counts_from_events(events_df: pd.DataFrame, swrs_df: pd.DataFrame) -> dict:
    """Per-class counts from the direct event-level classifier."""
    classified = classify_swr_overlap(events_df, swrs_df)
    return classified["overlap_class"].value_counts().to_dict()


def _counts_from_table(events_df: pd.DataFrame, swrs_df: pd.DataFrame) -> dict:
    """Per-class counts from the table-driven counter, stripped of the
    ``total`` key and zero entries so the shape matches the event-level
    counts dict."""
    dot = _build_data_over_time(events_df, swrs_df)
    classified = build_classified_nonlocal_event_table(dot)
    raw = compute_swr_overlap_class_counts(classified)
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
            "multi_swr_no_gap",  # the historical disagreement case
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
def test_event_and_table_counts_agree(label, events, swrs):
    """The direct classifier and the table-driven counter must agree."""
    events_df = pd.DataFrame(events, columns=["start_time", "end_time"])
    swrs_df = pd.DataFrame(swrs, columns=["start_time", "end_time"])

    event_counts = _counts_from_events(events_df, swrs_df)
    table_counts = _counts_from_table(events_df, swrs_df)

    assert event_counts == table_counts, (
        f"[{label}] event-level={event_counts}, table-driven={table_counts}"
    )
