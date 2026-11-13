"""The decode-quality summaries leave excluded time out (task 26b, F2).

``analysed_time.with_swr_source`` marks each bin of a read frame with
``in_excluded_time``. Every ``compute_*`` in ``quality_metrics`` requires that
column. A point-wise statistic gives the same answer as on the frame with those
rows removed. A run-labelling function (collapse episodes, non-local runs during
movement, running bouts) treats the excluded bins as a gap, so a run on either
side of excluded time stays two runs: removing the rows would join them.

The frames are ``tests/support/_excluded_time_frames.py``'s.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.hippocampus import quality_metrics as qm
from tests.support._excluded_time_frames import (
    FS,
    as_gap,
    rows_removed,
    two_sessions,
    unfiltered,
)

POINT_WISE = {
    "compute_local_decoding_error": {},  # B1
    "compute_calibration_curve": {},  # B6
    "compute_behavior_state_breakdown": {},  # C2
    "compute_running_error_tail": {},  # C4
    "compute_error_by_track_segment": {},  # C5
    "compute_no_spike_firing_correspondence": {},  # C6
}

RUN_LABELLING = {
    "compute_state_fraction_during_movement": {},  # B4
    "compute_non_local_during_movement": {},  # C1
    "compute_bout_nonlocal_dominance": {"min_bout_duration_sec": 0.1},  # C3
    "compute_local_state_collapse_details": {},  # C8
}

EVERY_COMPUTE = [
    *POINT_WISE,
    *RUN_LABELLING,
    "compute_non_local_event_context",  # C7
    "compute_session_quality_summary",
]


def test_every_compute_function_is_covered():
    names = {name for name in vars(qm) if name.startswith("compute_")}

    assert names == set(EVERY_COMPUTE)


@pytest.mark.parametrize("name", POINT_WISE)
def test_a_point_wise_summary_equals_the_summary_without_the_excluded_rows(name):
    function = getattr(qm, name)
    frame = two_sessions()

    result = function(frame, **POINT_WISE[name])

    pd.testing.assert_frame_equal(result, function(rows_removed(frame)))
    # the excluded rows would change it
    assert not result.equals(function(unfiltered(frame)))


@pytest.mark.parametrize("name", RUN_LABELLING)
def test_a_run_labelling_summary_treats_excluded_time_as_a_gap(name):
    function = getattr(qm, name)
    frame = two_sessions()
    kwargs = RUN_LABELLING[name]

    result = function(frame, **kwargs)

    pd.testing.assert_frame_equal(result, function(as_gap(frame), **kwargs))
    # Counting the excluded bins, or removing the rows (which joins the runs on
    # either side), would both change it.
    assert not result.equals(function(unfiltered(frame), **kwargs))
    assert not result.equals(function(rows_removed(frame), **kwargs))


def test_a_collapse_run_across_excluded_time_is_two_episodes():
    """B4: the analysed parts of the collapse run, 380-399 and 500-519."""
    result = qm.compute_state_fraction_during_movement(two_sessions()).set_index(
        "animal"
    )

    assert result.loc["a", "n_collapse_episodes"] == 2
    assert result.loc["a", "max_collapse_duration_sec"] == pytest.approx(20 / FS)
    assert result.loc["a", "n_moving_bins"] == 700 - 150
    assert result.loc["b", "n_collapse_episodes"] == 1
    assert result.loc["b", "max_collapse_duration_sec"] == pytest.approx(140 / FS)


def test_the_non_local_runs_during_movement_split_at_excluded_time():
    """C1: two 20-bin runs over 550 analysed moving bins."""
    result = qm.compute_non_local_during_movement(two_sessions()).set_index("animal")

    assert result.loc["a", "n_nonlocal_movement_events"] == 2
    assert result.loc["a", "nonlocal_movement_fraction"] == pytest.approx(40 / 550)
    assert result.loc["a", "nonlocal_movement_rate_per_min"] == pytest.approx(
        2 / (550 / FS / 60.0)
    )


def test_the_running_bouts_split_at_excluded_time():
    """C3: bins 0-199, 250-399 and 500-699."""
    result = qm.compute_bout_nonlocal_dominance(
        two_sessions(), min_bout_duration_sec=0.1
    ).set_index("animal")

    assert result.loc["a", "n_running_bouts"] == 3
    assert result.loc["b", "n_running_bouts"] == 1


def test_the_collapse_details_take_the_session_median_hpd_outside_excluded_time():
    """C8: the collapse's hpd, 3, lies between the analysed bins' 5 and the
    excluded bins' 1, and the excluded bins are most of the session."""
    n = 100
    bins = np.arange(n)
    frame = pd.DataFrame(
        {
            "animal": "a",
            "day": 1,
            "epoch": "02_r1",
            "head_speed": 10.0,
            "Local_prob": np.where(bins < 10, 0.1, 0.9),
            "Non-Local_Continuous_prob": 0.0,
            "Non-Local_Fragmented_prob": 0.0,
            "ahead_behind_distance": 0.0,
            "hpd": np.select([bins < 10, bins < 40], [3.0, 5.0], 1.0),
            "in_excluded_time": bins >= 40,
        },
        index=bins / FS,
    )

    details = qm.compute_local_state_collapse_details(frame)

    assert details["classification"].tolist() == ["confidently_wrong"]
    counted = qm.compute_local_state_collapse_details(unfiltered(frame))
    assert counted["classification"].tolist() == ["uncertain"]


def test_the_event_context_inherits_the_rule_from_the_labels():
    """C7 is event-based. The read zeroes every event label in excluded time
    and renormalizes the theta z it classifies on (task 26a), so rows in
    excluded time hold no event and do not change it."""
    frame = two_sessions()

    result = qm.compute_non_local_event_context(frame)

    pd.testing.assert_frame_equal(
        result, qm.compute_non_local_event_context(rows_removed(frame))
    )
    assert result["n_events"].tolist() == [1, 1]


@pytest.mark.parametrize("name", EVERY_COMPUTE)
def test_a_frame_not_read_through_the_source_is_refused(name):
    frame = two_sessions().drop(columns="in_excluded_time")

    with pytest.raises(ValueError, match="read_dot_with_swr_source.*in_excluded_time"):
        getattr(qm, name)(frame)


def test_a_column_that_is_not_boolean_is_refused():
    """A concat with a frame read another way leaves NaN in the column, which
    a cast to bool would read as excluded."""
    frame = two_sessions()
    frame["in_excluded_time"] = frame["in_excluded_time"].astype(object)
    frame.iloc[:10, frame.columns.get_loc("in_excluded_time")] = np.nan

    with pytest.raises(TypeError, match="in_excluded_time"):
        qm.compute_local_decoding_error(frame)
