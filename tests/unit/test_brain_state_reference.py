"""brain_state_reference: running and quiet-rest levels of a decode frame's z-scores."""

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.hippocampus.brain_state_reference import (
    brain_state_reference,
)


def frame() -> pd.DataFrame:
    # 10 running bins (theta 2), 10 quiet rest bins (theta -1), 5 rest bins in an
    # event (theta 5), 5 rest bins in excluded time (theta 9), 5 bins at 6 cm/s
    speed = np.r_[np.full(10, 20.0), np.full(20, 1.0), np.full(5, 6.0)]
    theta = np.r_[
        np.full(10, 2.0),
        np.full(10, -1.0),
        np.full(5, 5.0),
        np.full(5, 9.0),
        np.zeros(5),
    ]
    event = np.r_[np.zeros(20), np.ones(5), np.zeros(10)].astype(int)
    excluded = np.r_[np.zeros(25), np.ones(5), np.zeros(5)].astype(bool)
    return pd.DataFrame(
        {
            "head_speed": speed,
            "in_excluded_time": excluded,
            "non_local_event": event,
            "SWR_event": np.zeros(35, int),
            "MUA_event": np.zeros(35, int),
            "theta_power_zscore": theta,
            "SWR_zscore": -theta,
            "MUA_zscore": np.zeros(35),
        }
    )


def test_running_and_rest_exclude_events_excluded_time_and_middle_speeds():
    ref = brain_state_reference(frame())
    assert ref["n_running_bins"] == 10
    assert ref["n_rest_bins"] == 10
    assert ref["theta_running_median"] == 2.0
    assert ref["theta_rest_median"] == -1.0
    assert ref["ripple_rest_mean"] == 1.0


def test_a_regime_with_no_bin_reads_nan():
    dot = frame().assign(head_speed=1.0)
    ref = brain_state_reference(dot)
    assert ref["n_running_bins"] == 0
    assert np.isnan(ref["theta_running_median"])
