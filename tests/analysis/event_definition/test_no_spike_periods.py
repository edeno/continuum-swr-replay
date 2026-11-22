"""No-Spike periods: the non-local events' rule on the No-Spike state."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_definition.no_spike_periods import (
    level_outside_periods,
    no_spike_periods,
)

STEP = 0.002
"""The decode grid's step (s), 500 Hz."""


def _frame() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """200 samples from t = 0, one well visit over [0, 0.3) s. P(No-Spike) is 0.9
    in five runs, 0.5 elsewhere:

    - samples 10-19 (10 samples): kept;
    - samples 30-34 (5 samples): shorter than 7 samples, not a period;
    - samples 50-59: the animal moves at sample 55 (5 cm/s), dropped;
    - samples 80-89: sample 85 is in excluded time, dropped;
    - samples 170-179: starts at 0.34 s, after the visit, dropped.

    MUA is 1 everywhere but 5 inside the runs."""
    time = np.arange(200) * STEP
    probability = np.full(200, 0.5)
    mua = np.ones(200)
    for start, n in ((10, 10), (30, 5), (50, 10), (80, 10), (170, 10)):
        probability[start : start + n] = 0.9
        mua[start : start + n] = 5.0
    speed = np.zeros(200)
    speed[55] = 5.0
    excluded = np.zeros(200, dtype=bool)
    excluded[85] = True
    frame = pd.DataFrame(
        {
            "No-Spike_prob": probability,
            "head_speed": speed,
            "in_excluded_time": excluded,
            "MUA_zscore": mua,
        },
        index=pd.Index(time, name="time"),
    )
    return frame, np.array([0.0]), np.array([0.3])


def test_a_period_is_the_event_rule_on_the_no_spike_state_in_a_visit():
    frame, arrival, departure = _frame()
    periods = no_spike_periods(frame, arrival, departure)
    assert len(periods) == 1
    assert periods.loc[0, "start_time"] == pytest.approx(10 * STEP)
    assert periods.loc[0, "end_time"] == pytest.approx(19 * STEP)
    assert periods.loc[0, "duration"] == pytest.approx(9 * STEP)
    assert list(periods.columns) == ["start_time", "end_time", "duration"]


def test_the_threshold_is_strictly_above():
    frame, arrival, departure = _frame()
    frame.loc[frame.index[10:20], "No-Spike_prob"] = 0.8
    assert no_spike_periods(frame, arrival, departure).empty


def test_the_level_outside_periods_is_over_immobile_analyzed_time_in_a_visit():
    """Outside the one kept period the MUA is 1 in most of the visit, 5 in the
    runs that are not periods; samples after the visit, in excluded time or
    moving do not count."""
    frame, arrival, departure = _frame()
    periods = no_spike_periods(frame, arrival, departure)
    # inside the visit (150 samples), outside the period (10) and the excluded
    # and moving samples (2): 138 samples, of which 5 + 9 + 9 are 5, the rest 1
    level = level_outside_periods(frame, periods, arrival, departure, "MUA_zscore")
    assert level == pytest.approx(1.0)
    frame["MUA_zscore"] = np.where(frame.index < 0.3, 2.0, 9.0)
    assert level_outside_periods(
        frame, periods, arrival, departure, "MUA_zscore"
    ) == pytest.approx(2.0)


def test_a_frame_without_a_column_is_refused():
    frame, arrival, departure = _frame()
    with pytest.raises(ValueError, match="in_excluded_time"):
        no_spike_periods(frame.drop(columns="in_excluded_time"), arrival, departure)
