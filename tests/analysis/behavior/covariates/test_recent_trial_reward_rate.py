"""Tests for ``recent_trial_reward_rate`` — the stop-level reward-history
predictor for the rate model (R2). Arm-agnostic: the denominator is
trials, not visits to a particular arm.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.behavior.covariates import (
    recent_trial_reward_rate as _recent_trial_reward_rate,
)
from tests.support._port_experience_fixtures import port_experiences_from_trials


def recent_trial_reward_rate(times, trials, k=10):
    return _recent_trial_reward_rate(
        times,
        port_experiences_from_trials(trials, prepend_initial=False),
        k,
    )


def _trials(end_times, is_reward) -> pd.DataFrame:
    end = np.asarray(end_times, dtype=float)
    return pd.DataFrame(
        {
            "movement_start_time": end - 0.5,
            "end_time": end,
            "is_reward": np.asarray(is_reward),
        }
    )


def test_nan_before_first_trial():
    trials = _trials([1, 2, 3], [1, 0, 1])
    out = recent_trial_reward_rate(np.array([0.5]), trials, k=10)
    assert np.isnan(out[0])


def test_rate_is_rewarded_fraction():
    trials = _trials([1, 2, 3, 4], [1, 0, 1, 1])
    out = recent_trial_reward_rate(np.array([5.0]), trials, k=10)
    assert out[0] == 0.75


def test_window_truncates_to_last_k():
    trials = _trials([1, 2, 3, 4, 5], [1, 1, 0, 0, 0])
    # k=2 after all five trials -> last two outcomes [0, 0] -> 0.0
    out = recent_trial_reward_rate(np.array([6.0]), trials, k=2)
    assert out[0] == 0.0


def test_partial_window_before_k_trials():
    trials = _trials([1, 2, 3], [1, 0, 1])
    # only the first trial has elapsed -> mean([1]) == 1.0
    out = recent_trial_reward_rate(np.array([1.5]), trials, k=10)
    assert out[0] == 1.0


def test_empty_trials_all_nan():
    out = recent_trial_reward_rate(np.array([1.0, 2.0]), _trials([], []))
    assert np.isnan(out).all()


def test_vectorized_multiple_queries():
    trials = _trials([1, 2, 3, 4], [1, 0, 0, 1])
    out = recent_trial_reward_rate(np.array([0.5, 2.5, 5.0]), trials, k=10)
    assert np.isnan(out[0])
    assert out[1] == 0.5  # after trials 1,2 -> [1, 0]
    assert out[2] == 0.5  # after all four -> [1, 0, 0, 1]


def test_unsorted_trials_give_same_result():
    trials = _trials([1, 2, 3, 4], [1, 0, 0, 1])
    shuffled = trials.iloc[[2, 0, 3, 1]].reset_index(drop=True)
    q = np.array([0.5, 2.5, 5.0])
    np.testing.assert_array_equal(
        recent_trial_reward_rate(q, trials, k=10),
        recent_trial_reward_rate(q, shuffled, k=10),
    )
