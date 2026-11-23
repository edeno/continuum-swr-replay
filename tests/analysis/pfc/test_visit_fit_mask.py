"""The mPFC fit bins and events on the well-visit subset, on a hand-made session.

The session has 100 bins of 0.1 s, four stops and two well visits:

- stop A [1.0, 3.0], starting before the first poke-in at 1.5 (``spans_visit_start``);
- stop B [4.0, 6.0], running past the final poke-out at 5.0 (``spans_visit_end``);
- stop C [7.0, 9.0], wholly inside the second visit [6.8, 9.5) (``in_visit``), with
  two moving bins (8.5 and 8.6 s);
- stop D [9.6, 9.9], on the track (``track_leaf_arm``).

The stop cohort the models were fit on keeps A, B and C whole (``at_well`` is
``at_well`` or ``mixed`` at the stop grain); the in-visit rule keeps only their
bins inside a visit. The first two tests record what the stop cohort did on this
case; the rest test the in-visit rule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_definition.cohort_selection import (
    events_in_stops,
)
from continuum_swr_replay.analysis.pfc.event_sets import (
    build_session_fit_mask,
    build_visit_event_sets,
    build_visit_fit_mask,
    visit_bin_labels,
)

TIME = np.arange(100) / 10.0
ARRIVALS = np.array([1.5, 6.8])
DEPARTURES = np.array([5.0, 9.5])


@pytest.fixture
def position_info() -> pd.DataFrame:
    speed = np.zeros(TIME.size)
    speed[[85, 86]] = 10.0  # moving inside stop C
    return pd.DataFrame({"head_speed": speed}, index=pd.Index(TIME, name="time"))


@pytest.fixture
def stops() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stop_id": ["A", "B", "C", "D"],
            "stop_start": [1.0, 4.0, 7.0, 9.6],
            "stop_end": [3.0, 6.0, 9.0, 9.9],
            "occupancy_state": ["mixed", "mixed", "at_well", "in_transit"],
            "visit_context": [
                "spans_visit_start",
                "spans_visit_end",
                "in_visit",
                "track_leaf_arm",
            ],
        }
    )


@pytest.fixture
def events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": ["e1", "e2", "e3", "e4", "e5", "e6"],
            "start_time": [1.2, 2.0, 4.95, 5.5, 7.5, 9.7],
            "end_time": [1.3, 2.1, 5.2, 5.6, 7.6, 9.75],
            "is_swr": [False, True, False, True, False, True],
            "visit_context": [
                "spans_visit_start",
                "in_visit",
                "spans_visit_end",
                "after_visit_at_well",
                "in_visit",
                "track_leaf_arm",
            ],
        }
    )


@pytest.fixture
def swr_intervals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "start_time": [1.3, 2.5, 5.4, 7.2, 9.65],
            "end_time": [1.35, 2.55, 5.45, 7.25, 9.7],
        }
    )


# --- the stop cohort, recorded on this case ------------------------------------------


def test_the_stop_cohort_fit_every_immobile_bin_of_its_stops(position_info, stops):
    cohort = stops[stops["occupancy_state"].isin(["at_well", "mixed"])]
    _, fit_mask, counts = build_session_fit_mask(
        position_info,
        cohort_stops=cohort,
        stop_cohort="at_well",
        excluded_intervals=pd.DataFrame({"start_time": [8.0], "end_time": [8.0]}),
        basis_half_window_s=0.25,
    )
    # A 21 bins, B 21, C 21 less 2 moving; 7.8 to 8.2 s reach excluded time.
    assert counts["n_bins_fit"] == 61 - 5 == int(fit_mask.sum())
    assert counts["n_bins_excluded"] == 5


def test_the_stop_cohort_kept_every_event_starting_in_its_stops(
    stops, events, swr_intervals
):
    cohort = stops[stops["occupancy_state"].isin(["at_well", "mixed"])]
    kept = events_in_stops(events["start_time"].to_numpy(), cohort)
    assert events.loc[kept, "event_id"].tolist() == ["e1", "e2", "e3", "e4", "e5"]
    assert events_in_stops(swr_intervals["start_time"].to_numpy(), cohort).sum() == 4


# --- the in-visit rule ----------------------------------------------------------------


def test_bins_carry_the_visit_label_of_their_time(position_info, stops):
    labels = visit_bin_labels(TIME, stops, ARRIVALS, DEPARTURES)
    by_label = labels.groupby("visit_context")["time"].agg(["min", "max", "size"])
    # A before the poke-in keeps its stop's label; B after the final poke-out
    # is after the visit; no bin is spans_visit_end.
    assert by_label.loc["spans_visit_start"].tolist() == [1.0, 1.4, 5]
    assert by_label.loc["after_visit_at_well"].tolist() == [5.0, 6.0, 11]
    assert by_label.loc["track_leaf_arm"].tolist() == [9.6, 9.9, 4]
    assert by_label.loc["in_visit", "size"] == 16 + 10 + 21
    assert "spans_visit_end" not in by_label.index


def test_fit_mask_is_the_immobile_in_visit_bins(position_info, stops):
    _, fit_mask, counts = build_visit_fit_mask(
        position_info,
        stops=stops,
        arrival_times=ARRIVALS,
        departure_times=DEPARTURES,
    )
    expected = np.zeros(TIME.size, bool)
    expected[15:31] = True  # A from the poke-in at 1.5 s to 3.0 s
    expected[40:50] = True  # B up to the poke-out at 5.0 s (not included)
    expected[70:91] = True  # C whole
    expected[[85, 86]] = False  # moving
    np.testing.assert_array_equal(fit_mask.to_numpy(), expected)
    assert counts["n_bins_fit"] == 45
    assert counts["n_bins_excluded"] == 0
    # The labels of the immobile bins in stops, kept and dropped, by label.
    assert counts["bin_visit_context"]["in_visit"] == {"n": 45, "kept": True}
    assert counts["bin_visit_context"]["spans_visit_start"] == {"n": 5, "kept": False}
    assert counts["bin_visit_context"]["after_visit_at_well"] == {
        "n": 11,
        "kept": False,
    }
    assert counts["bin_visit_context"]["track_leaf_arm"] == {"n": 4, "kept": False}
    assert counts["bin_visit_context"]["spans_visit_end"] == {"n": 0, "kept": True}


def test_excluded_time_still_removes_the_bins_its_basis_reaches(position_info, stops):
    _, fit_mask, counts = build_visit_fit_mask(
        position_info,
        stops=stops,
        arrival_times=ARRIVALS,
        departure_times=DEPARTURES,
        excluded_intervals=pd.DataFrame({"start_time": [8.0], "end_time": [8.0]}),
        basis_half_window_s=0.25,
    )
    assert counts["n_bins_excluded"] == 5
    assert counts["n_bins_fit"] == 40 == int(fit_mask.sum())
    assert not fit_mask.loc[7.8:8.2].any()


def test_events_are_those_that_start_inside_a_visit(events, swr_intervals):
    event_times, counts = build_visit_event_sets(
        events,
        swr_intervals,
        arrival_times=ARRIVALS,
        departure_times=DEPARTURES,
        position_time=pd.Index(TIME),
    )
    assert event_times["non_local"]["event_id"].tolist() == ["e2", "e3", "e5"]
    assert event_times["non_local_SWR"]["event_id"].tolist() == ["e2"]
    assert event_times["non_local_no_SWR"]["event_id"].tolist() == ["e3", "e5"]
    # An SWR counts when its first sample is inside a visit.
    assert event_times["SWR"]["start_time"].tolist() == [2.5, 7.2]
    assert counts["n_nonlocal_before"] == 6
    assert counts["n_nonlocal_after"] == 3
    assert counts["n_ripple_before"] == 5
    assert counts["n_ripple_after"] == 2
    assert counts["event_visit_context"]["in_visit"] == {"n": 2, "kept": True}
    assert counts["event_visit_context"]["spans_visit_end"] == {"n": 1, "kept": True}
    assert counts["event_visit_context"]["after_visit_at_well"] == {
        "n": 1,
        "kept": False,
    }


def test_events_without_a_visit_label_are_refused(events, swr_intervals):
    with pytest.raises(ValueError, match="visit_context"):
        build_visit_event_sets(
            events.drop(columns="visit_context"),
            swr_intervals,
            arrival_times=ARRIVALS,
            departure_times=DEPARTURES,
            position_time=pd.Index(TIME),
        )
