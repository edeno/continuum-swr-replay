"""The behavior summary's stop- and onset-table values, on tables small enough
to check by hand."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.reported.behavior.stops import stop_table_values

NAN = np.nan


def _stops() -> pd.DataFrame:
    """Eight stops of one session; the first six are well-anchored.

    Spans 3, 5, 8, 1, 4.5 and 0.5 s; stops 2, 5 and 6 hold excluded time, so
    their analyzed immobility is shorter. Stops 1 and 4 learn their outcome
    inside them, within the 4 s the design keeps. Stop 3 learns it 5 s in.
    Stop 5 learns it 1 s in, but contains stop 6's arrival (52 s). Stop 2
    contains stop 3's arrival (24 s) and learned its outcome before it began.
    """
    return pd.DataFrame(
        {
            "animal": "a",
            "session_date": 1,
            "epoch": "02_r1",
            "stop_id": [1, 2, 3, 4, 5, 6, 7, 8],
            "occupancy_state": ["at_well"] * 4
            + ["mixed", "at_well", "in_transit", "in_transit"],
            "is_rewarded_outcome": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, NAN, 1.0],
            "stop_start": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
            "stop_end": [13.0, 25.0, 38.0, 41.0, 54.5, 60.5, 72.0, 86.0],
            "immobility_duration_s": [3.0, 1.0, 8.0, 1.0, 4.0, 0.25, 2.0, 6.0],
            "outcome_known_time_s": [11.0, 19.0, 35.0, 40.5, 51.0, 59.0, NAN, NAN],
            "last_well_arrival_time_s": [9.0, 18.0, 24.0, 39.0, 48.0, 52.0, NAN, NAN],
            "is_stay": [1.0, 0.0, 1.0, 0.0, NAN, 1.0, NAN, NAN],
            "reward_history_incomplete": [1, 0, 0, 0, 0, 1, 0, 0],
        }
    )


def _onsets() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_type": ["non_local"] * 5 + ["SWR"] * 2 + ["MUA"] * 4,
            "stop_id": [1, 1, 2, 3, 6, 2, 3, 3, 3, 3, 5],
            "t_since_stop_onset": [0.5, 1.5, 3.0, 5.0, 0.2, 1.0, 6.0]
            + [1.0, 2.0, 7.0, 0.1],
        }
    )


def test_stop_table_values_name_each_statistic():
    values = {v.key: v for v in stop_table_values(_stops(), _onsets(), "x")}
    read = {key: value.value for key, value in values.items()}
    # analyzed immobility of the well-anchored stops, rewarded: 3, 1 and 8;
    # omitted: 1, 4 and 0.25 (the rewarded in-transit stop is not one of them)
    assert read["stop_duration_mean_rewarded"] == 4.0
    assert read["stop_duration_mean_unrewarded"] == 1.75
    assert read["stop_duration_mean_ratio"] == pytest.approx(16 / 7)
    assert read["stop_duration_median_rewarded"] == 3.0
    assert read["stop_duration_median_unrewarded"] == 1.0
    assert read["stop_duration_median_ratio"] == 3.0
    # spans of the six well-anchored stops: 3, 5, 8, 1, 4.5 and 0.5
    assert read["stop_survival_two_s"] == pytest.approx(4 / 6)
    assert read["stop_survival_four_s"] == 0.5
    assert "2 s after they began" in values["stop_survival_two_s"].description
    assert "4 s after they began" in values["stop_survival_four_s"].description
    assert read["stops_in_transit"] == 2
    assert read["stops_crossing_outcome"] == 4  # stops 1, 3, 4 and 5
    assert read["stops_crossing_outcome_fraction"] == 0.5
    assert read["stops_outcome_shifted"] == 2  # stops 1 and 4
    assert read["stops_outcome_shifted_fraction"] == 0.25
    assert read["stops_spanning_arrival"] == 2  # stops 2 and 5
    assert read["stops_spanning_arrival_crossing"] == 1  # stop 5
    # onsets earlier than 4 s: 4 of 5 non-local, 1 of 2 SWR, 3 of 4 MUA
    assert read["onsets_within_tmax_fraction_non_local"] == 0.8
    assert read["onsets_within_tmax_fraction_swr"] == 0.5
    assert read["onsets_within_tmax_fraction_mua"] == 0.75
    assert read["stops_reward_history_incomplete_fraction"] == 0.25  # stops 1 and 6
    assert read["onsets_reward_history_incomplete_fraction_non_local"] == 0.6
    assert read["onset_median_s_non_local"] == 1.5
    assert {v.function for v in values.values()} == {"stop_table_values"}
