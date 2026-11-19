"""A stop's location, segment and distance summaries leave excluded time out.

Excluded time (the artifact mask widened by 0.3 s) is not analyzed, so the
per-stop summaries over the stop's rows (``occupancy_state``, ``at_well``,
``occupancy_well_id``, ``modal_observed_track_segment_id`` and
``nearest_well_distance_cm_min/mean/max``, and the columns derived from them)
read only the rows outside it, in both stop builders:

- a stop partly in excluded time is summarized from its outside rows;
- a stop wholly in excluded time is ``unknown``, with no well, segment or
  distance;
- a stop outside excluded time is what it was;
- the NonLocal and ContFrag builders agree, bout for bout.

The frame, graph and artifact run are ``test_artifact_exposure.py``'s: the mask
is [2, 5] and [15.85, 16.05], excluded time [1.7, 5.3] and [15.55, 16.35].
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.contfrag.tables import build_contfrag_stop_table
from continuum_swr_replay.analysis.event_definition import analyzed_time
from continuum_swr_replay.analysis.event_tables.stop_table import build_stop_table
from tests.analysis.event_definition.test_artifact_exposure import (
    CORES,
    EPOCH_ID,
    TIME,
    _graph,
)
from tests.analysis.event_definition.test_artifact_exposure import _dot as _exposure_dot
from tests.support._port_experience_fixtures import port_experiences_from_trials
from tests.support._swr_inventory_fixtures import use_artifact_run, write_artifact_run

STOPS = [(0.2, 1.0), (1.8, 2.4), (3.0, 7.0)]
"""A outside excluded time, B wholly in [1.7, 5.3], C partly: [3.0, 5.3] in it
and (5.3, 7.0] outside."""

LOCATION_COLUMNS = [
    "occupancy_state",
    "at_well",
    "occupancy_well_id",
    "modal_observed_track_segment_id",
    "nearest_well_distance_cm_min",
    "nearest_well_distance_cm_mean",
    "nearest_well_distance_cm_max",
    "occupancy_well_arm",
    "current_patch",
    "is_stay",
]

SEGMENT_CHANGE_S = 5.25
"""Before it the animal's segment is 6 and it is 50 cm from a well; after it the
segment is 0 and it is 2 cm and more from one. So stop C's rows in excluded time
outnumber its rows outside, and its mode over all rows is 6."""


def _trials() -> pd.DataFrame:
    """One trip: the animal leaves well 1 at 1.1 s and reaches well 0 at 5.2 s,
    inside excluded time, and leaves it at 7.5 s."""
    return pd.DataFrame(
        {
            "movement_start_time": [1.1],
            "destination_first_poke_in_time": [5.2],
            "end_time": [7.5],
            "from_well": [1],
            "to_well": [0],
            "is_reward": [True],
            "is_patch_change": [False],
            "outcome_known_time": [5.7],
        }
    )


def _dot() -> pd.DataFrame:
    dot = _exposure_dot()
    immobile = np.zeros(TIME.size, dtype=bool)
    for start, end in STOPS:
        immobile |= (TIME >= start) & (TIME <= end)
    dot["head_speed"] = np.where(immobile, 0.0, 10.0)
    before = TIME < SEGMENT_CHANGE_S
    dot["track_segment_id"] = np.where(before, 6.0, 0.0)
    dot["projected_x_position"] = 0.0
    dot["projected_y_position"] = np.where(
        before, 50.0, 2.0 + (TIME - SEGMENT_CHANGE_S)
    )
    return dot


def _tables() -> dict[str, pd.DataFrame]:
    trials = _trials()
    experiences = port_experiences_from_trials(trials, prepend_initial=True)
    return {
        "nonlocal": build_stop_table(_dot(), trials, experiences, _graph()),
        "contfrag": build_contfrag_stop_table(_dot(), trials, experiences, _graph()),
    }


def _outside_rows_of_stop_c() -> np.ndarray:
    """By hand: stop C's rows outside the closed excluded intervals."""
    excluded = np.zeros(TIME.size, dtype=bool)
    for start, end in analyzed_time.excluded_intervals(EPOCH_ID).to_numpy():
        excluded |= (TIME >= start) & (TIME <= end)
    return (TIME >= 3.0) & (TIME <= 7.0) & ~excluded


@pytest.fixture
def tables(tmp_path, monkeypatch) -> dict[str, dict[str, pd.DataFrame]]:
    """Both tables built with no artifact run (``"all_rows"``: every stop
    summarized over all its rows), then with the exposure test's cores
    (``"masked"``). The source stays configured for the test."""
    all_rows = _tables()
    directory = write_artifact_run(tmp_path, {EPOCH_ID: CORES})
    use_artifact_run(monkeypatch, directory)
    return {"all_rows": all_rows, "masked": _tables()}


@pytest.mark.parametrize("table", ["nonlocal", "contfrag"])
class TestTheStopsLocation:
    def test_the_fixture_has_the_three_stops(self, tables, table):
        stops = tables["masked"][table]

        np.testing.assert_allclose(stops["stop_start"], [s for s, _ in STOPS])
        np.testing.assert_allclose(stops["stop_end"], [e for _, e in STOPS])
        np.testing.assert_allclose(
            stops["immobility_duration_s"], [0.8, 0.0, 7.0 - 5.3], atol=1e-9
        )

    def test_a_stop_partly_in_excluded_time_reads_its_outside_rows(self, tables, table):
        stop = tables["masked"][table].iloc[2]
        outside = _outside_rows_of_stop_c()
        distance = (2.0 + (TIME - SEGMENT_CHANGE_S))[outside]
        assert 0 < outside.sum() < ((TIME >= 3.0) & (TIME <= 7.0)).sum() / 2

        assert stop["occupancy_state"] == "at_well"
        assert bool(stop["at_well"])
        assert stop["occupancy_well_id"] == 0.0
        assert stop["modal_observed_track_segment_id"] == 0.0
        for statistic in ("min", "mean", "max"):
            assert stop[f"nearest_well_distance_cm_{statistic}"] == pytest.approx(
                getattr(distance, statistic)(), rel=1e-12
            )
        assert stop["occupancy_well_arm"] == 0.0
        assert stop["current_patch"] == 1.0
        # over all its rows the stop reads otherwise: the case discriminates
        everything = tables["all_rows"][table].iloc[2]
        assert everything["occupancy_state"] == "mixed"
        assert everything["modal_observed_track_segment_id"] == 6.0
        assert everything["nearest_well_distance_cm_min"] == 2.0
        assert everything["nearest_well_distance_cm_max"] == 50.0
        assert np.isnan(everything["occupancy_well_arm"])

    def test_a_stop_wholly_in_excluded_time_is_unknown(self, tables, table):
        stop = tables["masked"][table].iloc[1]
        # over all its rows it was a stop in transit
        assert tables["all_rows"][table].iloc[1]["occupancy_state"] == "in_transit"

        assert stop["occupancy_state"] == "unknown"
        assert pd.isna(stop["at_well"])
        for column in LOCATION_COLUMNS[2:]:
            assert np.isnan(stop[column]), column

    def test_a_stop_outside_excluded_time_is_unchanged(self, tables, table):
        stops = tables["masked"][table]

        pd.testing.assert_frame_equal(
            stops.iloc[[0]], tables["all_rows"][table].iloc[[0]], check_exact=True
        )
        assert stops.iloc[0]["occupancy_state"] == "at_well"
        assert stops.iloc[0]["occupancy_well_id"] == 1.0

    def test_the_bout_and_its_id_are_unchanged(self, tables, table):
        columns = ["stop_idx", "stop_start", "stop_end", "stop_id"]

        pd.testing.assert_frame_equal(
            tables["masked"][table][columns], tables["all_rows"][table][columns]
        )


def test_the_two_builders_agree_bout_for_bout(tables):
    masked = tables["masked"]

    pd.testing.assert_frame_equal(
        masked["nonlocal"][LOCATION_COLUMNS],
        masked["contfrag"][LOCATION_COLUMNS],
        check_exact=True,
    )
