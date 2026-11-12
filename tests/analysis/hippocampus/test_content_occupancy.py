"""The content scan's well positions and per-bin track distances."""

from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.hippocampus import content_occupancy
from tests.support._excluded_time_frames import unfiltered


def test_the_well_positions_leave_excluded_time_out():
    """Each leaf arm's well is the median immobile position on it. On arm 0,
    100 excluded immobile bins at 50 cm would move it from 10 cm."""
    arms = np.repeat(np.arange(6), 60)
    linear = 10.0 + 20.0 * arms
    analysed = pd.DataFrame(
        {
            "head_speed": 1.0,
            "track_segment_id": arms,
            "linear_position": linear,
            "in_excluded_time": False,
        }
    )
    excluded = pd.DataFrame(
        {
            "head_speed": 1.0,
            "track_segment_id": 0,
            "linear_position": np.full(100, 50.0),
            "in_excluded_time": True,
        }
    )
    frame = pd.concat([analysed, excluded], ignore_index=True)

    wells = content_occupancy.well_positions(frame)

    assert wells == [10.0, 30.0, 50.0, 70.0, 90.0, 110.0]
    assert wells == content_occupancy.well_positions(analysed)
    assert content_occupancy.well_positions(unfiltered(frame))[0] == 50.0


def test_an_arm_needs_more_than_the_minimum_of_immobile_bins_at_running_speed_excluded():
    """Arm 0 has 51 immobile bins (a well), arm 1 has 50 (none); 4 cm/s is not
    immobile, and a non-finite position is not counted."""
    n = content_occupancy.MIN_IMMOBILE_BINS
    frame = pd.DataFrame(
        {
            "head_speed": [1.0] * (n + 1) + [1.0] * n + [4.0] * 10 + [1.0] * 5,
            "track_segment_id": [0] * (n + 1) + [1] * n + [1] * 10 + [1] * 5,
            "linear_position": [5.0] * (n + 1) + [7.0] * n + [7.0] * 10 + [np.nan] * 5,
            "in_excluded_time": False,
        }
    )
    assert content_occupancy.well_positions(frame) == [5.0]


class _LineEnvironment:
    """Place bins at 0, 10, ..., 100 cm on one line; track distance is |a - b|.
    Bin 5 (50 cm) is not interior."""

    def __init__(self) -> None:
        self.place_bin_centers_ = np.arange(0.0, 101.0, 10.0)[:, None]
        self.is_track_interior_ = np.arange(11) != 5

    def get_distances_to_interior_bins(self, points: np.ndarray) -> np.ndarray:
        interior = self.place_bin_centers_[self.is_track_interior_, 0]
        return np.abs(points[:, 0][:, None] - interior[None, :])


def _frame(well_arms: int) -> pd.DataFrame:
    """60 immobile bins on each of ``well_arms`` leaf arms placing wells at 0,
    20, 40, ... cm, then six bins of two non-local events: event 1 (bins 0-2)
    overlaps an SWR in its last two bins, so all its bins are SWR bins; event 2
    (bins 3-5) does not."""
    rest = pd.DataFrame(
        {
            "head_speed": 1.0,
            "track_segment_id": np.repeat(np.arange(well_arms), 60),
            "linear_position": 20.0 * np.repeat(np.arange(well_arms), 60),
            "non_local_event": 0,
            "SWR_event": 0,
            "map_estimate": np.nan,
            "ahead_behind_distance": np.nan,
        }
    )
    events = pd.DataFrame(
        {
            "head_speed": 1.0,
            "track_segment_id": 8,
            "linear_position": 90.0,
            "non_local_event": [1, 1, 1, 2, 2, 2],
            "SWR_event": [0, 1, 1, 0, 0, 0],
            "map_estimate": [12.0, 31.0, 49.0, 99.0, np.nan, 66.0],
            "ahead_behind_distance": [-78.0, -59.0, -41.0, 9.0, 0.0, np.inf],
        }
    )
    frame = pd.concat([rest, events], ignore_index=True)
    frame.index = 0.002 * np.arange(len(frame))
    return frame.assign(animal="a", day=1, epoch="02_r1", in_excluded_time=False)


def test_content_distances_measure_each_event_bin_to_the_animal_and_the_nearest_well():
    d_animal, d_well, is_swr = content_occupancy.content_distances(
        _frame(4), _LineEnvironment()
    )
    # wells at 0, 20, 40, 60 cm; content bins: 12 -> 10, 31 -> 30, 49 -> 40
    # (50 is not interior), 99 -> 100; the bin with no MAP and the bin with an
    # infinite distance to the animal are left out
    np.testing.assert_array_equal(d_animal, [78.0, 59.0, 41.0, 9.0])
    np.testing.assert_array_equal(d_well, [10.0, 10.0, 0.0, 40.0])
    np.testing.assert_array_equal(is_swr, [True, True, True, False])


def test_a_session_with_too_few_wells_has_no_distances():
    assert content_occupancy.MIN_WELLS == 4
    assert content_occupancy.content_distances(_frame(3), _LineEnvironment()) is None
