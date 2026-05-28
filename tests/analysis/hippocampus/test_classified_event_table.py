"""Tests for the canonical per-non-local-event classification table.

Covers ``build_classified_nonlocal_event_table`` and its
``intervals_from_labels`` helper in
``continuum_swr_replay.analysis.hippocampus.event_table``.

The shared fixture synthesises a ``data_over_time``-shaped frame sampled
at 500 Hz with one event per interval-geometry case, plus a second
session, so overlap classification, matched-SWR pairing, per-event speed
summaries, peak non-local probability, event-id stability, and
cross-session isolation can each be asserted against known inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.hippocampus.event_table import (
    build_classified_nonlocal_event_table,
    intervals_from_labels,
)

DT = 1.0 / 500.0  # 500 Hz sampling
SPEED_THRESHOLD = 4.0

# Session 1 event windows (inclusive sample ranges) and the SWR windows
# arranged to realise each interval-geometry case. SWR label ids are
# deliberately distinct from the event ids (11+ vs 1-8) so a matched-SWR
# assertion cannot pass by accidentally returning the event id.
#
#   ev1 [100,110] ⊂ swr11 [95,120]        -> swr_contained (matched swr 11)
#   ev2 [200,220] ∩ swr12 [210,230]       -> single_swr_partial (matched swr 12)
#   ev3 [300,340] ⊃ swr13 [310,330]       -> event_contains_swr (matched swr 13)
#   ev4 [400,420] disjoint                -> non_swr
#   ev5 [500,520], swr14 ends at 500      -> non_swr (boundary touch)
#   ev6 [600,620] fast middle, slow ends  -> immobile-endpoints flag True
#   ev7 [700,700] single timebin          -> iloc[0] == iloc[-1] collapse
#   ev8 [900,940] ∩ swr15 [890,910] & swr16 [930,950] (gap inside)
#       -> multi_swr_gap, n_swrs_overlapping == 2, matched earliest = swr 15
_EVENTS_1 = {
    1: (100, 110),
    2: (200, 220),
    3: (300, 340),
    4: (400, 420),
    5: (500, 520),
    6: (600, 620),
    7: (700, 700),
    8: (900, 940),
}
_SWRS_1 = {
    11: (95, 120),
    12: (210, 230),
    13: (310, 330),
    14: (480, 500),
    15: (890, 910),
    16: (930, 950),
}


def _blank_session(animal: str, day: int, epoch: str, n: int) -> pd.DataFrame:
    """Zeroed ``dot``-shaped session frame with a time index at 500 Hz."""
    time = np.arange(n) * DT
    frame = pd.DataFrame(
        {
            "animal": animal,
            "day": day,
            "epoch": epoch,
            "non_local_event": np.zeros(n, dtype=int),
            "SWR_event": np.zeros(n, dtype=int),
            "head_speed": np.full(n, 1.0),  # immobile by default
            "Non-Local_Continuous_prob": np.full(n, 0.1),
            "Non-Local_Fragmented_prob": np.zeros(n),
        },
        index=pd.Index(time, name="time"),
    )
    return frame


def _label(frame: pd.DataFrame, col: str, windows: dict[int, tuple[int, int]]) -> None:
    """Stamp inclusive sample windows with their integer label id."""
    arr = frame[col].to_numpy()
    for label_id, (lo, hi) in windows.items():
        arr[lo : hi + 1] = label_id
    frame[col] = arr


@pytest.fixture
def dot() -> pd.DataFrame:
    """Two-session ``dot`` frame covering every geometry case."""
    s1 = _blank_session("m", 1, "01_r1", n=1000)
    _label(s1, "non_local_event", _EVENTS_1)
    _label(s1, "SWR_event", _SWRS_1)

    # ev6: fast middle, immobile endpoints.
    speed = s1["head_speed"].to_numpy()
    speed[605:616] = 10.0  # interior of [600, 620]
    s1["head_speed"] = speed

    # ev1: a known non-local-prob trajectory peaking inside the window.
    cont = s1["Non-Local_Continuous_prob"].to_numpy()
    frag = s1["Non-Local_Fragmented_prob"].to_numpy()
    cont[100:111] = np.linspace(0.30, 0.70, 11)  # peak 0.70 at sample 110
    frag[100:111] = 0.20
    s1["Non-Local_Continuous_prob"] = cont
    s1["Non-Local_Fragmented_prob"] = frag

    # Session 2 shares Session 1's time range; its single event sits where
    # Session 1 carries swr11. With correct per-session scoping it must be
    # ``non_swr``; a leak would label it ``swr_contained``.
    s2 = _blank_session("m", 1, "02_r2", n=200)
    _label(s2, "non_local_event", {1: (100, 110)})

    return pd.concat([s1, s2])


def _row(table: pd.DataFrame, epoch: str, event: int) -> pd.Series:
    """Single classified-table row for one (epoch, non_local_event)."""
    mask = (table["epoch"] == epoch) & (table["non_local_event"] == event)
    matched = table.loc[mask]
    assert len(matched) == 1, f"expected exactly one row for {epoch}/{event}"
    return matched.iloc[0]


def test_swr_contained_event(dot):
    table = build_classified_nonlocal_event_table(dot)
    ev = _row(table, "01_r1", 1)
    assert ev["overlap_class"] == "swr_contained"
    assert ev["is_swr"]
    assert ev["matched_swr_start"] <= ev["event_start"]
    assert ev["matched_swr_end"] >= ev["event_end"]
    assert ev["matched_swr_event"] == 11  # SWR label, not the event id


def test_swr_partial_single(dot):
    table = build_classified_nonlocal_event_table(dot)
    ev = _row(table, "01_r1", 2)
    assert ev["overlap_class"] == "swr_partial"
    assert ev["overlap_subclass"] == "single_swr_partial"
    assert ev["n_swrs_overlapping"] == 1
    assert ev["matched_swr_event"] == 12
    assert ev["matched_swr_start"] == pytest.approx(210 * DT)
    assert ev["matched_swr_end"] == pytest.approx(230 * DT)


def test_event_contains_swr(dot):
    table = build_classified_nonlocal_event_table(dot)
    ev = _row(table, "01_r1", 3)
    assert ev["overlap_class"] == "swr_partial"
    assert ev["overlap_subclass"] == "event_contains_swr"
    assert ev["matched_swr_event"] == 13


def test_multiple_swr_overlap_picks_earliest(dot):
    # ev8 overlaps swr15 [890,910] and swr16 [930,950] with a gap inside.
    table = build_classified_nonlocal_event_table(dot)
    ev = _row(table, "01_r1", 8)
    assert ev["overlap_class"] == "swr_partial"
    assert ev["overlap_subclass"] == "multi_swr_gap"
    assert ev["n_swrs_overlapping"] == 2
    # Earliest-by-start tie-break: swr15 (start 890) precedes swr16 (930).
    assert ev["matched_swr_event"] == 15
    assert ev["matched_swr_start"] == pytest.approx(890 * DT)


def test_non_swr(dot):
    table = build_classified_nonlocal_event_table(dot)
    ev = _row(table, "01_r1", 4)
    assert ev["overlap_class"] == "non_swr"
    assert not ev["is_swr"]
    assert pd.isna(ev["matched_swr_event"])
    assert np.isnan(ev["matched_swr_start"])
    assert np.isnan(ev["matched_swr_end"])


def test_boundary_touch_is_non_swr(dot):
    # swr4 ends exactly at ev5's start; zero positive-duration overlap.
    table = build_classified_nonlocal_event_table(dot)
    ev = _row(table, "01_r1", 5)
    assert ev["event_start"] == pytest.approx(500 * DT)
    assert ev["overlap_class"] == "non_swr"
    assert not ev["is_swr"]


def test_event_id_stable(dot):
    first = build_classified_nonlocal_event_table(dot)
    second = build_classified_nonlocal_event_table(dot)
    pd.testing.assert_series_equal(first["event_id"], second["event_id"])


def test_event_id_unique_within_session(dot):
    table = build_classified_nonlocal_event_table(dot)
    # validate_event_ids runs inside the builder; uniqueness must hold.
    assert table["event_id"].nunique() == len(table)


def test_peak_prob(dot):
    table = build_classified_nonlocal_event_table(dot)
    ev = _row(table, "01_r1", 1)
    s1_event1 = dot.loc[(dot["epoch"] == "01_r1") & (dot["non_local_event"] == 1)]
    expected = (
        s1_event1["Non-Local_Continuous_prob"] + s1_event1["Non-Local_Fragmented_prob"]
    ).max()
    assert ev["peak_nonlocal_prob"] == pytest.approx(expected)
    # Sanity: the peak is the within-window max (0.70 + 0.20), not a global.
    assert ev["peak_nonlocal_prob"] == pytest.approx(0.90)


def test_max_speed_vs_endpoints(dot):
    table = build_classified_nonlocal_event_table(dot, speed_threshold=SPEED_THRESHOLD)
    ev = _row(table, "01_r1", 6)
    assert ev["max_speed"] > SPEED_THRESHOLD
    assert ev["start_speed"] < SPEED_THRESHOLD
    assert ev["end_speed"] < SPEED_THRESHOLD
    assert ev["endpoints_speed_below_thresh"]
    # mean_speed is the within-window mean (not median/max/sum); derive the
    # expected value independently from the fixture's speed profile.
    s1_event6 = dot.loc[(dot["epoch"] == "01_r1") & (dot["non_local_event"] == 6)]
    assert ev["mean_speed"] == pytest.approx(s1_event6["head_speed"].mean())
    assert ev["mean_speed"] != pytest.approx(ev["max_speed"])  # genuinely averaged


def test_timing_columns(dot):
    table = build_classified_nonlocal_event_table(dot)
    ev = _row(table, "01_r1", 1)  # samples [100, 110] -> 11 bins
    assert ev["event_start"] == pytest.approx(100 * DT)
    assert ev["event_end"] == pytest.approx(110 * DT)
    assert ev["event_duration"] == pytest.approx(10 * DT)
    # Midpoint is times[len(times) // 2] -> 11 bins picks index 5 = sample 105.
    assert ev["event_midpoint_time"] == pytest.approx(105 * DT)


def test_single_timebin_event_collapse(dot):
    table = build_classified_nonlocal_event_table(dot)
    ev = _row(table, "01_r1", 7)
    assert ev["event_start"] == ev["event_end"]
    assert ev["event_duration"] == pytest.approx(0.0)
    assert ev["max_speed"] == ev["mean_speed"] == ev["start_speed"] == ev["end_speed"]


def test_multiple_sessions_no_leak(dot):
    table = build_classified_nonlocal_event_table(dot)
    # Session 2's only event coincides in time with Session 1's swr11 but
    # Session 2 has no SWR of its own.
    ev = _row(table, "02_r2", 1)
    assert ev["overlap_class"] == "non_swr"
    assert not ev["is_swr"]
    assert pd.isna(ev["matched_swr_event"])


def test_empty_session():
    n = 50
    frame = _blank_session("m", 1, "01_r1", n=n)  # all-zero event labels
    table = build_classified_nonlocal_event_table(frame)
    assert table.empty
    expected_cols = build_classified_nonlocal_event_table(
        _make_minimal_event_frame()
    ).columns.tolist()
    assert table.columns.tolist() == expected_cols


def _make_minimal_event_frame() -> pd.DataFrame:
    """Tiny single-event frame used to read off the canonical column order."""
    frame = _blank_session("m", 1, "01_r1", n=20)
    _label(frame, "non_local_event", {1: (5, 8)})
    return frame


def test_no_overlapping_pairs_schema():
    # Events exist; no SWR labels anywhere -> every event is non_swr and
    # the matched-SWR columns carry the right NA dtypes.
    frame = _blank_session("m", 1, "01_r1", n=200)
    _label(frame, "non_local_event", {1: (10, 20), 2: (50, 60)})
    table = build_classified_nonlocal_event_table(frame)

    assert len(table) == 2
    assert (table["overlap_class"] == "non_swr").all()
    assert not table["is_swr"].any()
    assert str(table["matched_swr_event"].dtype) == "Int64"
    assert table["matched_swr_event"].isna().all()
    assert table["matched_swr_start"].dtype == np.float64
    assert table["matched_swr_end"].dtype == np.float64
    assert table["matched_swr_start"].isna().all()
    assert table["matched_swr_end"].isna().all()


def test_intervals_from_labels_accepts_index_time_and_retains_label():
    frame = _blank_session("m", 1, "01_r1", n=100)
    _label(frame, "SWR_event", {3: (10, 20), 7: (40, 55)})
    intervals = intervals_from_labels(frame, "SWR_event")

    assert list(intervals.columns) == ["SWR_event", "start_time", "end_time"]
    assert set(intervals["SWR_event"]) == {3, 7}
    swr3 = intervals.loc[intervals["SWR_event"] == 3].iloc[0]
    assert swr3["start_time"] == pytest.approx(10 * DT)
    assert swr3["end_time"] == pytest.approx(20 * DT)


def test_intervals_from_labels_accepts_time_column():
    n = 100
    frame = _blank_session("m", 1, "01_r1", n=n).reset_index()  # "time" column
    assert "time" in frame.columns
    _label(frame, "SWR_event", {1: (5, 15)})
    intervals = intervals_from_labels(frame, "SWR_event")

    assert list(intervals.columns) == ["SWR_event", "start_time", "end_time"]
    row = intervals.iloc[0]
    assert row["SWR_event"] == 1
    assert row["start_time"] == pytest.approx(5 * DT)
    assert row["end_time"] == pytest.approx(15 * DT)


def test_intervals_from_labels_empty_returns_schema():
    frame = _blank_session("m", 1, "01_r1", n=50)  # no positive SWR labels
    intervals = intervals_from_labels(frame, "SWR_event")
    assert intervals.empty
    assert list(intervals.columns) == ["SWR_event", "start_time", "end_time"]


def test_missing_required_column_raises():
    frame = _blank_session("m", 1, "01_r1", n=50)
    _label(frame, "non_local_event", {1: (5, 8)})
    frame = frame.drop(columns=["Non-Local_Fragmented_prob"])
    with pytest.raises(KeyError, match="Non-Local_Fragmented_prob"):
        build_classified_nonlocal_event_table(frame)
