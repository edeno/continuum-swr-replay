"""Mechanism-A encode: an unvisited leaf's reward rate is 0.0 (not NaN)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.behavior.covariates import (
    recent_reward_rate_all_events as _recent_reward_rate_all_events,
)
from continuum_swr_replay.analysis.behavior.covariates import trial_outcome_times
from tests.support._port_experience_fixtures import port_experiences_from_trials


def recent_reward_rate_all_events(times, trials, well_to_arm, k=10):
    return _recent_reward_rate_all_events(
        times,
        port_experiences_from_trials(trials, prepend_initial=False),
        well_to_arm,
        k,
    )


# 6 leaf wells -> arms 0..5; trunk arms 6,7,8 are never trial destinations.
_WELL_TO_ARM = {w: w for w in range(6)}


def _trials():
    n = 6
    return pd.DataFrame(
        {
            "movement_start_time": np.arange(n, dtype=float),
            "end_time": np.arange(n, dtype=float) + 0.5,
            "from_well": [0, 1, 2, 3, 4, 5],
            "to_well": [1, 2, 3, 4, 5, 0],  # arm 1 visited first, then 2,3,4,5,0
            "is_reward": [True, False, True, False, True, False],
            "is_patch_change": [False, True, False, True, False, True],
        }
    )


def test_unvisited_leaf_encoded_zero():
    trials = _trials()
    cutoffs = trial_outcome_times(trials)
    # An event just after ONLY the first trial (arm 1 visited, rewarded); a no-past event.
    after_first = float((cutoffs[0] + cutoffs[1]) / 2)
    before_any = float(cutoffs[0] - 1.0)
    ev = np.array([after_first, before_any])

    rr = recent_reward_rate_all_events(ev, trials, _WELL_TO_ARM, k=5)

    # Row 0 (after first trial): arm 1 visited+rewarded; arms 2-5 & 0 unvisited; trunk 6-8.
    assert rr[0, 1] == 1.0  # visited, rewarded
    for a in (0, 2, 3, 4, 5):
        assert rr[0, a] == 0.0, f"unvisited leaf {a} must ENCODE 0.0, not NaN"
    for a in (6, 7, 8):
        assert rr[0, a] == 0.0  # trunk: structural 0

    # Row 1 (no prior trials): everything NaN (so no NEW events drop vs the old behavior).
    assert np.isnan(rr[1]).all()


def test_no_trials_all_nan():
    empty = pd.DataFrame(
        columns=["start_time", "end_time", "from_well", "to_well", "is_reward"]
    )
    ev = np.array([1.0, 2.0])
    rr = recent_reward_rate_all_events(ev, empty, _WELL_TO_ARM, k=5)
    assert np.isnan(rr).all()
