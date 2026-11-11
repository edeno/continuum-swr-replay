"""The stop- and onset-table statistics, on tables small enough to check by hand."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay import per_bin_design
from continuum_swr_replay.analysis.replay import stop_statistics as stats

NAN = np.nan


def test_stop_cohort_counts_divide_the_stops_by_occupancy_and_stay():
    stops = pd.DataFrame(
        {
            "occupancy_state": [
                "at_well",
                "at_well",
                "mixed",
                "mixed",
                "in_transit",
                "in_transit",
                "in_transit",
                None,
            ],
            "is_stay": [1.0, NAN, 0.0, NAN, 1.0, NAN, 0.0, 1.0],
        }
    )
    assert stats.stop_cohort_counts(stops) == {
        "n_stops": 8,
        "n_well_anchored": 4,
        "n_not_anchored": 4,
        # two in-transit stops and the stop with no state
        "n_not_anchored_with_stay": 3,
        "n_in_transit": 3,
        "n_mixed": 2,
        "n_mixed_with_stay": 1,
        "n_at_well_with_stay": 1,
    }


def test_dwell_statistics_compare_the_analysed_immobility_by_outcome():
    # The first rewarded stop spans 20 s with 18 s of excluded time inside it,
    # and the first omitted stop is one sample long: both count as they are.
    stops = pd.DataFrame(
        {
            "is_rewarded_outcome": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, NAN],
            "stop_start": [0.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
            "stop_end": [20.0, 34.0, 49.0, 50.0, 62.0, 73.0, 87.0, 190.0],
            "immobility_duration_s": [2.0, 4.0, 9.0, 0.0, 2.0, 3.0, 7.0, 100.0],
        }
    )
    assert stats.dwell_statistics(stops) == {
        "mean_rewarded_s": 5.0,  # of 2, 4 and 9 (the spans' mean is 11)
        "mean_omitted_s": 3.0,  # of 0, 2, 3 and 7
        "mean_ratio": pytest.approx(5 / 3),
        "median_rewarded_s": 4.0,
        "median_omitted_s": 2.5,
        "median_ratio": 1.6,
    }


def test_still_immobile_is_measured_from_the_start_of_the_stop():
    # Spans of 1, 2, 3 and 4 s. The second and the fourth stop hold excluded
    # time, which is not taken out: they are still immobile at 2 s and at 4 s.
    stops = pd.DataFrame(
        {
            "stop_start": [0.0, 10.0, 20.0, 30.0],
            "stop_end": [1.0, 12.0, 23.0, 34.0],
            "immobility_duration_s": [1.0, 1.5, 3.0, 2.5],
        }
    )
    # a stop that lasts exactly that long counts
    assert stats.still_immobile_fraction(stops, 2) == 0.75
    assert stats.still_immobile_fraction(stops, 4) == 0.25


def _boundary_stops() -> pd.DataFrame:
    """Eight stops in two sessions, rows out of time order.

    Session a: stop 1 learns its outcome 1 s in (shifted). Stop 2 learns it
    before it starts, and contains stop 3's arrival (22 s). Stop 3 learns it
    as it ends, 0.5 s in: its bins end before the boundary. Stop 4 learns it
    5 s in, past the 4 s the design keeps. Stop 5 learns it 1 s in but
    contains stop 6's arrival (52 s). Stop 6 has no outcome time.
    Session b: stop 7 learns its outcome 1 s in (shifted); stop 8 before it starts.
    """
    stops = pd.DataFrame(
        {
            "animal": ["a"] * 6 + ["b"] * 2,
            "session_date": 1,
            "epoch": "02_r1",
            "stop_id": [1, 2, 3, 4, 5, 6, 7, 8],
            "stop_start": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 5.0, 70.0],
            "stop_end": [13.0, 23.0, 30.5, 46.0, 53.0, 63.0, 8.0, 73.0],
            "outcome_known_time_s": [11.0, 19.5, 30.5, 45.0, 51.0, NAN, 6.0, 69.5],
            "last_well_arrival_time_s": [9.0, 19.0, 22.0, 39.0, 49.0, 52.0, 4.0, 69.0],
            "occupancy_state": "at_well",
        }
    )
    stops["immobility_duration_s"] = stops["stop_end"] - stops["stop_start"]
    return stops.iloc[[5, 2, 7, 0, 4, 1, 6, 3]].reset_index(drop=True)


def test_the_outcome_boundary_shift_needs_a_crossing_no_arrival_and_a_bin(
    monkeypatch,
):
    monkeypatch.setattr(
        per_bin_design,
        "excluded_intervals_by_session",
        lambda sessions: per_bin_design.NO_EXCLUDED_TIME,
    )
    assert stats.outcome_boundary_statistics(_boundary_stops()) == {
        "n_crossing": 5,  # stops 1, 3, 4, 5 and 7
        "crossing_fraction": 0.625,
        "n_shifted": 2,  # stops 1 and 7
        "shifted_fraction": 0.25,
        "n_spanning_arrival": 2,  # stops 2 and 5
        "n_spanning_arrival_crossing": 1,  # stop 5
    }


def test_onset_time_statistics_have_one_row_per_event_type():
    onsets = pd.DataFrame(
        {
            "event_type": ["non_local"] * 5 + ["SWR"] * 3,
            "t_since_stop_onset": [0.5, 1.5, 4.0, 6.0, NAN, 1.0, 2.0, 9.0],
        }
    )
    table = stats.onset_time_statistics(onsets, 4.0)
    assert table.index.name == "event_type"
    # an onset at the window's end is not within it; the unknown one is left out
    assert table.loc["non_local"].to_dict() == {
        "within_window_fraction": 0.5,
        "median_s": 2.75,
    }
    assert table.loc["SWR"].to_dict() == {
        "within_window_fraction": pytest.approx(2 / 3),
        "median_s": 2.0,
    }


def test_reward_history_incomplete_shares_of_stops_and_of_onsets():
    stops = pd.DataFrame(
        {"stop_id": [1, 2, 3, 4, 5], "reward_history_incomplete": [1, 0, 0, 1, 0]}
    )
    onsets = pd.DataFrame(
        {
            "event_type": ["non_local"] * 5 + ["SWR"] * 2,
            "stop_id": [1, 1, 2, 3, 4, 2, 2],
        }
    )
    assert stats.reward_history_incomplete_statistics(stops, onsets, "non_local") == {
        "stop_fraction": 0.4,
        "onset_fraction": 0.6,
    }
    assert (
        stats.reward_history_incomplete_statistics(stops, onsets, "SWR")[
            "onset_fraction"
        ]
        == 0.0
    )
