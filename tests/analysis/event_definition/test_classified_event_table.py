"""Tests for the canonical per-non-local-event classification table.

Covers ``build_classified_nonlocal_event_table`` and its
``intervals_from_labels`` helper in
``continuum_swr_replay.analysis.event_definition.classified_table``.

The shared fixture synthesizes a ``data_over_time``-shaped frame sampled
at 500 Hz with one event per interval-geometry case, plus a second
session, so overlap classification, matched-SWR pairing, per-event speed
summaries, peak non-local probability, event-id stability, and
cross-session isolation can each be asserted against known inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_definition.classified_table import (
    assert_drawn_class_matches_ripples,
    build_classified_nonlocal_event_table,
    intervals_from_labels,
    match_swr,
)
from continuum_swr_replay.analysis.event_definition.immobility import (
    EventDefinitionError,
)

DT = 1.0 / 500.0  # 500 Hz sampling
SPEED_THRESHOLD = 4.0
EXPECTED_CLASSIFIED_COLUMNS = [
    "animal",
    "day",
    "epoch",
    "non_local_event",
    "event_id",
    "event_start",
    "event_end",
    "event_duration",
    "event_midpoint_time",
    "animal_max_speed_cm_s",
    "animal_mean_speed_cm_s",
    "animal_start_speed_cm_s",
    "animal_end_speed_cm_s",
    "whole_event_speed_eligible",
    "endpoints_speed_below_thresh",
    "peak_nonlocal_prob",
    "is_swr",
    "overlap_class",
    "overlap_subclass",
    "n_swrs_overlapping",
    "matched_swr_event",
    "matched_swr_start",
    "matched_swr_end",
]

# Session 1 event windows (inclusive sample ranges) and the SWR windows
# arranged to realize each interval-geometry case. SWR label ids are
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
    if col == "non_local_event":
        frame.loc[frame[col] > 0, "Non-Local_Continuous_prob"] = 0.9


@pytest.fixture
def dot() -> pd.DataFrame:
    """Two-session ``dot`` frame covering every geometry case."""
    s1 = _blank_session("m", 1, "01_r1", n=1000)
    _label(s1, "non_local_event", _EVENTS_1)
    _label(s1, "SWR_event", _SWRS_1)
    s1.loc[s1["non_local_event"] > 0, "Non-Local_Continuous_prob"] = 0.9

    # ev6: fast middle, immobile endpoints.
    speed = s1["head_speed"].to_numpy()
    speed[605:616] = 10.0  # interior of [600, 620]
    s1["head_speed"] = speed

    # ev1: a known non-local-prob trajectory peaking inside the window.
    cont = s1["Non-Local_Continuous_prob"].to_numpy()
    frag = s1["Non-Local_Fragmented_prob"].to_numpy()
    cont[100:111] = np.linspace(0.61, 0.70, 11)  # total stays > .8
    frag[100:111] = 0.20
    s1["Non-Local_Continuous_prob"] = cont
    s1["Non-Local_Fragmented_prob"] = frag

    # Session 2 shares Session 1's time range; its single event sits where
    # Session 1 carries swr11. With correct per-session scoping it must be
    # ``non_swr``; a leak would label it ``swr_contained``.
    s2 = _blank_session("m", 1, "02_r2", n=200)
    _label(s2, "non_local_event", {1: (100, 110)})
    s2.loc[s2["non_local_event"] > 0, "Non-Local_Continuous_prob"] = 0.9

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
    table = build_classified_nonlocal_event_table(dot)
    ev = _row(table, "01_r1", 6)
    assert ev["animal_max_speed_cm_s"] > SPEED_THRESHOLD
    assert ev["animal_start_speed_cm_s"] < SPEED_THRESHOLD
    assert ev["animal_end_speed_cm_s"] < SPEED_THRESHOLD
    assert not ev["whole_event_speed_eligible"]
    # Retained as a diagnostic only; endpoints no longer define the cohort.
    assert ev["endpoints_speed_below_thresh"]
    # mean_speed is the within-window mean (not median/max/sum); derive the
    # expected value independently from the fixture's speed profile.
    s1_event6 = dot.loc[(dot["epoch"] == "01_r1") & (dot["non_local_event"] == 6)]
    assert ev["animal_mean_speed_cm_s"] == pytest.approx(s1_event6["head_speed"].mean())
    assert ev["animal_mean_speed_cm_s"] != pytest.approx(
        ev["animal_max_speed_cm_s"]
    )  # genuinely averaged


def test_timing_columns(dot):
    table = build_classified_nonlocal_event_table(dot)
    ev = _row(table, "01_r1", 1)  # samples [100, 110] -> 11 bins
    assert ev["event_start"] == pytest.approx(100 * DT)
    assert ev["event_end"] == pytest.approx(110 * DT)
    assert ev["event_duration"] == pytest.approx(
        11 * DT
    )  # (n_bins) * DT, incl. final bin
    # Midpoint is times[len(times) // 2] -> 11 bins picks index 5 = sample 105.
    assert ev["event_midpoint_time"] == pytest.approx(105 * DT)


def test_single_timebin_event_collapse(dot):
    table = build_classified_nonlocal_event_table(dot)
    ev = _row(table, "01_r1", 7)
    assert ev["event_start"] == ev["event_end"]
    assert ev["event_duration"] == pytest.approx(
        DT
    )  # single-bin event spans one bin width
    assert (
        ev["animal_max_speed_cm_s"]
        == ev["animal_mean_speed_cm_s"]
        == ev["animal_start_speed_cm_s"]
        == ev["animal_end_speed_cm_s"]
    )


def test_multiple_sessions_no_leak(dot):
    table = build_classified_nonlocal_event_table(dot)
    # Session 2's only event coincides in time with Session 1's swr11 but
    # Session 2 has no SWR of its own.
    ev = _row(table, "02_r2", 1)
    assert ev["overlap_class"] == "non_swr"
    assert not ev["is_swr"]
    assert pd.isna(ev["matched_swr_event"])


def test_interleaved_sessions_with_reused_times_match_contiguous_input(dot):
    """One-time session grouping must preserve row and time alignment."""
    expected = build_classified_nonlocal_event_table(dot)
    interleaved = (
        dot.assign(
            _session_row=dot.groupby(
                ["animal", "day", "epoch"],
                sort=False,
            ).cumcount()
        )
        .sort_values("_session_row", kind="stable")
        .drop(columns="_session_row")
    )

    actual = build_classified_nonlocal_event_table(interleaved)
    sort_cols = ["animal", "day", "epoch", "non_local_event"]
    expected = expected.sort_values(sort_cols).reset_index(drop=True)
    actual = actual.sort_values(sort_cols).reset_index(drop=True)

    pd.testing.assert_frame_equal(actual, expected)


def test_empty_session():
    n = 50
    frame = _blank_session("m", 1, "01_r1", n=n)  # all-zero event labels
    table = build_classified_nonlocal_event_table(frame)
    assert table.empty
    assert table.columns.tolist() == EXPECTED_CLASSIFIED_COLUMNS


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


def test_intervals_from_labels_refuses_a_time_column():
    # A stray column that disagrees with the index would be a second time axis.
    frame = _blank_session("m", 1, "01_r1", n=100)
    frame["time"] = frame.index.to_numpy() + 10.0
    _label(frame, "SWR_event", {1: (5, 15)})

    with pytest.raises(ValueError, match="'time' column.*set the time index"):
        intervals_from_labels(frame, "SWR_event")


@pytest.mark.parametrize(
    ("lose_the_axis", "message"),
    [
        (lambda f: f.reset_index(drop=True), "RangeIndex"),
        (lambda f: f.set_axis(np.arange(len(f))), "dtype int64"),
        (lambda f: f.iloc[::-1], "strictly increasing"),
        (lambda f: pd.concat([f.iloc[:10], f.iloc[9:]]), "strictly increasing"),
    ],
    ids=["range_index", "integer_index", "decreasing", "repeated_time"],
)
def test_intervals_from_labels_refuses_a_frame_without_its_time_axis(
    lose_the_axis, message
):
    # Row numbers read as seconds would give plausible, wrong bounds.
    frame = _blank_session("m", 1, "01_r1", n=100)
    _label(frame, "SWR_event", {1: (5, 15)})

    with pytest.raises(ValueError, match=message):
        intervals_from_labels(lose_the_axis(frame), "SWR_event")


def test_intervals_from_labels_empty_returns_schema():
    frame = _blank_session("m", 1, "01_r1", n=50)  # no positive SWR labels
    intervals = intervals_from_labels(frame, "SWR_event")
    assert intervals.empty
    assert list(intervals.columns) == ["SWR_event", "start_time", "end_time"]


def test_missing_required_column_raises():
    frame = _blank_session("m", 1, "01_r1", n=50)
    _label(frame, "non_local_event", {1: (5, 8)})
    frame = frame.drop(columns=["Non-Local_Fragmented_prob"])
    with pytest.raises(ValueError, match="Non-Local_Fragmented_prob"):
        build_classified_nonlocal_event_table(frame)


def test_invalid_saved_nonlocal_probability_fails_loud(dot):
    bad = dot.copy()
    event_row = (bad["epoch"] == "01_r1") & (bad["non_local_event"] == 2)
    bad.loc[event_row, "Non-Local_Continuous_prob"] = 0.9
    bad.loc[event_row, "Non-Local_Fragmented_prob"] = 0.0
    bad.loc[bad.index[event_row][3], "Non-Local_Continuous_prob"] = 0.5

    with pytest.raises(EventDefinitionError, match="probability-threshold contract"):
        build_classified_nonlocal_event_table(bad)


# --- Downstream consumers: parity with get_event_types, table-driven counts,
#     and the matched-SWR columns the timing offsets are taken from. ---

from continuum_swr_replay.analysis.detection.nonlocal_events import (  # noqa: E402
    get_event_types,
)
from continuum_swr_replay.analysis.event_properties.event_statistics import (  # noqa: E402
    compute_swr_overlap_class_counts,
)


def _key_set(frame: pd.DataFrame) -> set:
    return set(
        zip(
            frame["animal"],
            frame["day"],
            frame["epoch"],
            frame["non_local_event"],
            strict=True,
        )
    )


def test_event_types_matches_classified_overlap(dot):
    """get_event_types' SWR-overlap split must equal the classified table's
    is_swr split for immobile events: Figure 2's panel c is drawn from the
    first, and panel a's examples and the class counts that the canonical event
    table (panels b and d) is checked against come from the second."""
    classified = build_classified_nonlocal_event_table(dot)
    event_types = get_event_types(dot, event_col="non_local_event")

    overlap_keys = _key_set(
        event_types["non_local_event_immobile_overlap_SWR_event"].reset_index()
    )
    classified_overlap_keys = _key_set(
        classified[
            classified["whole_event_speed_eligible"]
            & (classified["overlap_class"] != "non_swr")
        ]
    )
    assert overlap_keys == classified_overlap_keys, (
        f"Drift: event_types-only={overlap_keys - classified_overlap_keys}, "
        f"classified-only={classified_overlap_keys - overlap_keys}"
    )


def test_compute_swr_overlap_class_counts(dot):
    classified = build_classified_nonlocal_event_table(dot)
    counts = compute_swr_overlap_class_counts(classified)
    # session 1: ev1 contained; ev2/ev3/ev8 partial; ev4/5/6/7 non_swr.
    # session 2: ev1 non_swr. Event 6 has slow endpoints but a fast interior,
    # so the canonical whole-event criterion excludes it from this tally.
    assert counts == {
        "swr_contained": 1,
        "swr_partial": 3,
        "non_swr": 4,
        "total": 8,
    }


def test_swr_overlap_class_counts_empty():
    empty = build_classified_nonlocal_event_table(_blank_session("m", 1, "01_r1", n=20))
    assert empty.empty
    assert compute_swr_overlap_class_counts(empty) == {
        "swr_contained": 0,
        "swr_partial": 0,
        "non_swr": 0,
        "total": 0,
    }


def test_panel_b_full_swr_intervals_not_clipped_to_nl_rows():
    # SWR [40,80] strictly spans event [50,60] and carries SWR-only timebins
    # outside the event window. The classified table derives SWR intervals
    # from the full session labels, so the event is swr_contained — a counter
    # that clipped SWR bounds to NL-event rows could miss this.
    frame = _blank_session("m", 1, "01_r1", n=200)
    _label(frame, "non_local_event", {1: (50, 60)})
    _label(frame, "SWR_event", {1: (40, 80)})
    classified = build_classified_nonlocal_event_table(frame)
    ev = _row(classified, "01_r1", 1)
    assert ev["overlap_class"] == "swr_contained"
    assert compute_swr_overlap_class_counts(classified)["swr_contained"] == 1


def test_timing_offsets_via_columns(dot):
    classified = build_classified_nonlocal_event_table(dot)
    # ev2 [200,220] partial with swr12 [210,230]: NL-minus-SWR offsets.
    ev2 = _row(classified, "01_r1", 2)
    assert ev2["event_start"] - ev2["matched_swr_start"] == pytest.approx(
        (200 - 210) * DT
    )
    assert ev2["event_end"] - ev2["matched_swr_end"] == pytest.approx((220 - 230) * DT)


def test_assert_drawn_matches_label_fires_on_boundary_mismatch():
    # Column header says swr_contained, but the drawn SWR span (from another
    # population) ends inside the event (-> only partial). The guard must raise.
    chosen = {"swr_contained": pd.Series({"start_time": 1.0, "end_time": 2.0})}
    ripple_times = pd.DataFrame({"start_time": [0.5], "end_time": [1.5]})
    with pytest.raises(AssertionError, match="contradict"):
        assert_drawn_class_matches_ripples(chosen, ripple_times)


def test_assert_drawn_matches_label_passes_when_consistent():
    # Ripple fully spans the event -> drawn class swr_contained == header.
    chosen = {"swr_contained": pd.Series({"start_time": 1.0, "end_time": 2.0})}
    ripple_times = pd.DataFrame({"start_time": [0.5], "end_time": [2.5]})
    assert_drawn_class_matches_ripples(chosen, ripple_times)  # must not raise


def test_events_and_swrs_are_read_on_the_one_time_index():
    # Event timing and the SWR intervals both read the index.
    frame = _blank_session("m", 1, "01_r1", n=400)
    _label(frame, "non_local_event", {1: (100, 110)})
    _label(frame, "SWR_event", {1: (95, 120)})
    table = build_classified_nonlocal_event_table(frame)
    ev = _row(table, "01_r1", 1)
    assert ev["overlap_class"] == "swr_contained"
    assert ev["event_start"] == pytest.approx(100 * DT)
    assert ev["matched_swr_start"] == pytest.approx(95 * DT)

    # A stray "time" column that disagrees with it (here a reversed copy)
    # would be a second axis: read as the axis, it would reverse the event's
    # bounds and leave it non_swr. It is refused, not ignored.
    frame["time"] = frame.index.to_numpy()[::-1]
    with pytest.raises(ValueError, match="'time' column"):
        build_classified_nonlocal_event_table(frame)


def test_zero_duration_event_inside_swr_is_non_swr():
    # A point event [1.0, 1.0] strictly inside an SWR has no positive
    # duration to overlap; the strict convention classifies it non_swr.
    from continuum_swr_replay.analysis.event_definition.nonlocal_event_rule import (
        classify_swr_overlap,
        classify_swr_overlap_geometry,
    )

    events = pd.DataFrame({"start_time": [1.0], "end_time": [1.0]})
    swrs = pd.DataFrame({"start_time": [0.5], "end_time": [1.5]})
    assert classify_swr_overlap(events, swrs)["overlap_class"].iloc[0] == "non_swr"
    geo = classify_swr_overlap_geometry(events, swrs)
    assert geo["overlap_class"].iloc[0] == "non_swr"
    assert geo["overlap_subclass"].iloc[0] == "no_overlap"


def test_match_swr_ignores_earlier_point_interval():
    """Representative matching uses the classifier's duration-valid SWR set."""
    matched = match_swr(
        "swr_partial",
        0.0,
        2.0,
        np.array([1, 2]),
        np.array([0.5, 1.5]),
        np.array([0.5, 2.5]),
    )

    assert matched == (2, 1.5, 2.5)
