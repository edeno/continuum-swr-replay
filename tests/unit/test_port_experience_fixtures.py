"""Tests for the shared port-experience fixture adapter.

The adapter turns a hand-written movement-trial frame into the experience
stream the table builders consume, and nearly every builder test runs on its
output. If it clocks an omission differently from the pipeline, the whole suite
quietly tests a task the animal never performed -- so its outcome clock is
pinned here rather than left implicit in whichever test happens to read it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.constants import OMISSION_KNOWN_LATENCY_SEC
from tests.unit._port_experience_fixtures import port_experiences_from_trials

_OMISSION_ARRIVAL = 1.0
_OMISSION_DEPARTURE = 3.0
_REWARD_ARRIVAL = 5.0
_PUMP_TIME = 5.1


def _trials_with_reward_on_time_only() -> pd.DataFrame:
    """One omission and one rewarded trip, with no explicit outcome clock."""
    return pd.DataFrame(
        {
            "start_time": [0.0, 3.0],
            "goal_arrival_time": [_OMISSION_ARRIVAL, _REWARD_ARRIVAL],
            "end_time": [_OMISSION_DEPARTURE, 7.0],
            "reward_on_time": [np.nan, _PUMP_TIME],
            "from_well": [1, 0],
            "to_well": [0, 2],
            "is_reward": [False, True],
            "is_patch_change": [False, True],
        }
    )


def test_omission_clock_is_the_latency_boundary_not_the_departure():
    experiences = port_experiences_from_trials(
        _trials_with_reward_on_time_only(), prepend_initial=False
    )

    omission = experiences.iloc[0]
    assert not bool(omission["is_reward"])
    assert omission["outcome_known_time"] == pytest.approx(
        _OMISSION_ARRIVAL + OMISSION_KNOWN_LATENCY_SEC
    )
    # The departure clock this adapter used to carry is 1.7 s later.
    assert omission["departure_time"] == pytest.approx(_OMISSION_DEPARTURE)
    assert np.isnan(omission["reward_on_time"])


def test_reward_clock_is_the_recorded_pump_onset():
    experiences = port_experiences_from_trials(
        _trials_with_reward_on_time_only(), prepend_initial=False
    )

    reward = experiences.iloc[1]
    assert bool(reward["is_reward"])
    assert reward["outcome_known_time"] == pytest.approx(_PUMP_TIME)
    assert reward["reward_on_time"] == pytest.approx(_PUMP_TIME)
