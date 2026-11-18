"""Tests for event classification functions.

Tests get_event_types which classifies non-local events by their overlap
with SWR and MUA events.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.detection.nonlocal_events import (
    get_event_types,
)


def _make_data_over_time(
    n_time: int = 1000,
    n_non_local: int = 3,
    n_swr: int = 2,
    n_mua: int = 2,
    overlap_nl_swr: bool = True,
) -> pd.DataFrame:
    """Create synthetic data_over_time DataFrame for testing."""
    rng = np.random.default_rng(42)

    df = pd.DataFrame(
        {
            "head_speed": rng.uniform(0, 2, n_time),  # all immobile
            "non_local_event": np.zeros(n_time, dtype=int),
            "SWR_event": np.zeros(n_time, dtype=int),
            "MUA_event": np.zeros(n_time, dtype=int),
            "animal": "test",
            "day": 20200101,
            "epoch": "02_r1",
        }
    )

    # Place non-local events
    event_starts = [100, 300, 600]
    for i, start in enumerate(event_starts[:n_non_local]):
        df.loc[start : start + 20, "non_local_event"] = i + 1

    df["Non-Local_Continuous_prob"] = np.where(df["non_local_event"] > 0, 0.9, 0.1)
    df["Non-Local_Fragmented_prob"] = 0.0

    # Place SWR events (overlapping with first non-local)
    if n_swr > 0:
        if overlap_nl_swr:
            df.loc[105:115, "SWR_event"] = 1  # overlaps with NL event 1
        else:
            df.loc[200:210, "SWR_event"] = 1  # no overlap
    if n_swr > 1:
        df.loc[500:510, "SWR_event"] = 2  # no overlap with NL

    # Place MUA events (overlapping with second non-local)
    if n_mua > 0:
        df.loc[305:315, "MUA_event"] = 1  # overlaps with NL event 2
    if n_mua > 1:
        df.loc[800:810, "MUA_event"] = 2  # no overlap with NL

    # The labels were placed by row number; a decode frame's index is its time.
    df.index = pd.Index(np.arange(n_time) / 500.0, name="time")
    return df


class TestGetEventTypes:
    def test_returns_expected_keys(self):
        df = _make_data_over_time()
        result = get_event_types(df)

        expected_keys = {
            "non_local_event_immobile",
            "SWR_event_events",
            "MUA_event_events",
            "non_local_event_immobile_overlap_MUA_event",
            "non_local_event_immobile_overlap_SWR_event",
            "non_local_event_immobile_no_overlap",
            "non_local_event_immobile_no_overlap_MUA_event",
            "non_local_event_immobile_no_overlap_SWR_event",
            "SWR_event_events_no_non_local_event",
            "MUA_event_events_no_non_local_event",
        }
        assert set(result.keys()) == expected_keys

    def test_overlap_swr_detects_overlapping_events(self):
        df = _make_data_over_time(overlap_nl_swr=True)
        result = get_event_types(df)

        # Event 1 overlaps with SWR
        overlap_events = result["non_local_event_immobile_overlap_SWR_event"][
            "non_local_event"
        ].unique()
        assert 1 in overlap_events

    def test_no_overlap_excludes_overlapping_events(self):
        df = _make_data_over_time(overlap_nl_swr=True)
        result = get_event_types(df)

        # Event 3 (at index 600) has no SWR or MUA overlap
        no_overlap = result["non_local_event_immobile_no_overlap"]
        no_overlap_events = no_overlap["non_local_event"].unique()
        assert 3 in no_overlap_events
        assert 1 not in no_overlap_events  # overlaps with SWR
        assert 2 not in no_overlap_events  # overlaps with MUA

    def test_all_events_accounted_for(self):
        df = _make_data_over_time()
        result = get_event_types(df)

        all_nl = result["non_local_event_immobile"]["non_local_event"].unique()
        overlap_swr = set(
            result["non_local_event_immobile_overlap_SWR_event"][
                "non_local_event"
            ].unique()
        )
        no_overlap_swr = set(
            result["non_local_event_immobile_no_overlap_SWR_event"][
                "non_local_event"
            ].unique()
        )

        # Every non-local event should be in exactly one of overlap/no_overlap
        for event_id in all_nl:
            in_overlap = event_id in overlap_swr
            in_no_overlap = event_id in no_overlap_swr
            assert in_overlap != in_no_overlap, (
                f"Event {event_id} should be in exactly one group"
            )

    def test_swr_events_no_non_local(self):
        df = _make_data_over_time(n_swr=2)
        result = get_event_types(df)

        # SWR event 2 (at index 500) has no non-local overlap
        swr_no_nl = result["SWR_event_events_no_non_local_event"]
        assert 2 in swr_no_nl["SWR_event"].unique()

    def test_no_events_returns_empty_dataframes(self):
        df = _make_data_over_time(n_non_local=0, n_swr=0, n_mua=0)
        # All events are zero, so after filtering head_speed, immobile events empty
        result = get_event_types(df)

        assert len(result["non_local_event_immobile"]) == 0
        assert len(result["SWR_event_events"]) == 0

    def test_boundary_touch_is_no_overlap(self):
        """Regression: events whose intervals touch at a single
        boundary timestamp share zero positive duration. The previous
        label-based ``(g[other_col] > 0).any()`` filter counted that
        as overlap because ``get_labels`` writes inclusive bin labels.
        Strict positive-duration semantics must classify it as no
        overlap (matches ``classify_swr_overlap`` /
        ``get_overlap_times``)."""
        # NL labels [0.0, 1.0]; SWR labels [1.0, 2.0]; share t=1.0.
        df = pd.DataFrame(
            {
                "animal": ["a"] * 5,
                "day": [1] * 5,
                "epoch": ["e1"] * 5,
                "non_local_event": [1, 1, 1, 0, 0],
                "SWR_event": [0, 0, 2, 2, 2],
                "MUA_event": [0, 0, 0, 0, 0],
                "head_speed": [0.0] * 5,
                "Non-Local_Continuous_prob": [0.9, 0.9, 0.9, 0.1, 0.1],
                "Non-Local_Fragmented_prob": [0.0] * 5,
            },
            index=[0.0, 0.5, 1.0, 1.5, 2.0],
        )
        result = get_event_types(df)
        assert len(result["non_local_event_immobile_overlap_SWR_event"]) == 0, (
            "boundary-touch SWR was classified as overlap"
        )
        # All NL timebins fall into no_overlap_SWR (full event = 3 bins).
        assert len(result["non_local_event_immobile_no_overlap_SWR_event"]) == 3
        # Other-detector "events_no_non_local_event" should also flip:
        # the SWR has no positive-duration overlap with the NL event,
        # so it's NOT excluded from the SWR-no-NL bucket.
        assert 2 in (
            result["SWR_event_events_no_non_local_event"]["SWR_event"].unique()
        )
