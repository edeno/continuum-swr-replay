"""The share of events with no SWR among the events with a resolved place."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_properties.resolved_content import (
    RESOLVED_WIDTH_CM,
    event_widths,
    resolved_share_statistics,
)


def _events_and_windows() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Animal a, epochs e1 and e2; animal b, epoch e1. By event: (animal,
    epoch, overlap class, width cm).

    a e1: non_swr 10, swr_contained 300, non_swr 50, swr_partial 20
    a e2: non_swr 250, swr_partial 30
    b e1: non_swr 10, non_swr 400, swr_contained 10, and one event without a
    width (no window).

    Resolved (< 200 cm): a e1 has 2 of 3 non-SWR, a e2 0 of 1 -> a's share
    (2/3 + 0) / 2 = 1/3; b e1 has 1 of 2 -> 1/2; the mean 5/12. Wide: a e1 0
    of 1, a e2 1 of 1 -> 1/2; b e1 1 of 1 -> 1; the mean 3/4. The share of
    events resolved: a e1 3/4, a e2 1/2 -> 5/8; b e1 2/3; the mean 31/48."""
    rows = [
        ("a", "e1", "non_swr", 10.0),
        ("a", "e1", "swr_contained", 300.0),
        ("a", "e1", "non_swr", 50.0),
        ("a", "e1", "swr_partial", 20.0),
        ("a", "e2", "non_swr", 250.0),
        ("a", "e2", "swr_partial", 30.0),
        ("b", "e1", "non_swr", 10.0),
        ("b", "e1", "non_swr", 400.0),
        ("b", "e1", "swr_contained", 10.0),
        ("b", "e1", "non_swr", np.nan),
    ]
    frame = pd.DataFrame(rows, columns=["animal", "epoch", "overlap_class", "hpd_cm"])
    events = pd.DataFrame(
        {
            "event_id": range(len(frame)),
            "animal": frame["animal"],
            "session_date": "d1",
            "epoch": frame["epoch"],
            "overlap_class": frame["overlap_class"],
            "whole_event_speed_eligible": True,
        }
    )
    windows = pd.concat(
        [
            pd.DataFrame({"event_id": events["event_id"], "kind": "event",
                          "hpd_cm": frame["hpd_cm"]}).dropna(),
            pd.DataFrame({"event_id": events["event_id"], "kind": "control",
                          "hpd_cm": 1000.0}),
        ],
        ignore_index=True,
    )  # fmt: skip
    return events, windows


def test_the_widths_are_the_events_windows_only():
    events, windows = _events_and_windows()
    width = event_widths(events, windows)
    assert width.iloc[1] == 300.0 and np.isnan(width.iloc[9])
    with pytest.raises(ValueError, match="two event windows"):
        event_widths(events, pd.concat([windows, windows.iloc[[0]]]))


def test_the_share_with_no_swr_among_the_resolved_and_the_wide_events():
    assert RESOLVED_WIDTH_CM == 200.0
    statistics = resolved_share_statistics(*_events_and_windows())
    resolved, wide = statistics["resolved"], statistics["wide"]
    assert resolved["non_swr_fraction_animal_mean"] == pytest.approx(5 / 12)
    assert resolved["non_swr_fraction_animal_min"] == pytest.approx(1 / 3)
    assert resolved["non_swr_fraction_animal_max"] == pytest.approx(1 / 2)
    assert resolved["n_events"] == 6
    assert wide["non_swr_fraction_animal_mean"] == pytest.approx(3 / 4)
    assert wide["n_events"] == 3
    assert statistics["resolved_share_animal_mean"] == pytest.approx(31 / 48)
    assert statistics["resolved_share_animal_min"] == pytest.approx(5 / 8)
    assert statistics["resolved_share_animal_max"] == pytest.approx(2 / 3)
    assert statistics["n_without_width"] == 1
