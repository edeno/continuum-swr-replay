"""The per-stop SWR / no-SWR split of the non-local event count."""

from __future__ import annotations

import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_tables.stop_table import (
    add_nonlocal_swr_split,
)


def _tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Three stops with 2, 1 and 0 events; stop 1 has one SWR event."""
    stops = pd.DataFrame({"stop_id": [1, 2, 3], "n_selected_events": [2, 1, 0]})
    events = pd.DataFrame({"stop_id": [1, 1, 2], "is_swr": [True, False, False]})
    return stops, events


def test_split_counts_each_stop_by_hand():
    stops, events = _tables()
    out = add_nonlocal_swr_split(stops, events)
    assert out["n_nonlocal_swr_events"].tolist() == [1, 0, 0]
    assert out["n_nonlocal_non_swr_events"].tolist() == [1, 1, 0]
    assert (
        out["n_nonlocal_swr_events"] + out["n_nonlocal_non_swr_events"]
    ).tolist() == stops["n_selected_events"].tolist()


def test_split_is_recomputed_on_a_stop_table_that_already_carries_it():
    """The shard assembly calls it on stop tables whose shards added the split."""
    stops, events = _tables()
    once = add_nonlocal_swr_split(stops, events)
    stale = once.assign(n_nonlocal_swr_events=99, n_nonlocal_non_swr_events=99)
    twice = add_nonlocal_swr_split(stale, events)
    pd.testing.assert_frame_equal(twice, once)


def test_split_refuses_an_event_whose_stop_is_absent():
    stops, events = _tables()
    with pytest.raises(ValueError, match="absent from stop_table"):
        add_nonlocal_swr_split(stops.iloc[:1], events)
