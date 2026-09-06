"""Loop-vs-vectorized equivalence guard for ``recent_reward_rate_all_events``.

The vectorized rewrite must equal the original nested-loop implementation
*exactly* on every code path. The loop form is kept here as the reference
oracle so the vectorization can never silently drift; this test gates the
removal of the loop from the production module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay.covariates.history import (
    recent_reward_rate_all_events,
)
from continuum_swr_replay.analysis.replay.covariates.history._common import (
    _prepare_port_experiences,
)
from continuum_swr_replay.analysis.replay.covariates.history.reward_history import (
    _observed_port_outcomes,
)
from continuum_swr_replay.data_loaders.constants import ALL_ARMS
from tests.unit._port_experience_fixtures import port_experiences_from_trials

WELL_TO_ARM = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}


def _recent_reward_rate_loop(
    event_times: np.ndarray,
    port_experiences: pd.DataFrame,
    well_to_arm: dict[int, int],
    k: int = 10,
) -> np.ndarray:
    """Pre-vectorization nested-loop reference oracle (verbatim).

    Kept only in the test as the ground truth the production vectorized form
    must match bit-for-bit.
    """
    n_arms = len(ALL_ARMS)
    n_events = len(event_times)
    experiences = _prepare_port_experiences(port_experiences)
    experiences = _observed_port_outcomes(experiences)
    n_experiences = len(experiences)
    reachable_arms = set(well_to_arm.values())

    if n_experiences == 0:
        return np.full((n_events, n_arms), np.nan)

    experiences = experiences.sort_values("outcome_known_time", kind="stable")
    cutoff_times = experiences["outcome_known_time"].to_numpy(dtype=float)
    ports = experiences["port"].to_numpy(dtype=np.int64)
    is_reward = experiences["is_reward"].to_numpy(dtype=float)

    arm_visits: list[np.ndarray] = []
    arm_rewards: list[np.ndarray] = []
    for arm in ALL_ARMS:
        vis_idx = []
        vis_rew = []
        for ti in range(n_experiences):
            aid = well_to_arm.get(int(ports[ti]))
            if aid == arm:
                vis_idx.append(ti)
                vis_rew.append(is_reward[ti])
        arm_visits.append(np.array(vis_idx, dtype=np.intp))
        arm_rewards.append(np.array(vis_rew, dtype=float))

    cutoffs = np.searchsorted(cutoff_times, event_times, side="right")
    no_past = cutoffs == 0
    out = np.full((n_events, n_arms), np.nan)

    for j in range(n_arms):
        if ALL_ARMS[j] not in reachable_arms:
            out[~no_past, j] = 0.0
            continue
        visits = arm_visits[j]
        rewards = arm_rewards[j]
        if visits.size == 0:
            out[~no_past, j] = 0.0
            continue
        n_vis_before = np.searchsorted(visits, cutoffs, side="left")
        for i in range(n_events):
            if no_past[i]:
                continue
            nv = n_vis_before[i]
            if nv == 0:
                out[i, j] = 0.0
                continue
            start = max(0, nv - k)
            out[i, j] = rewards[start:nv].mean()

    return out


def _trials(to_wells, rewards, arrivals=None) -> pd.DataFrame:
    """Toy movement trials with explicit destinations, rewards, and arrivals."""
    n = len(to_wells)
    if arrivals is None:
        arrivals = np.arange(n, dtype=float) * 10.0 + 8.0
    arrivals = np.asarray(arrivals, dtype=float)
    return pd.DataFrame(
        {
            "start_time": arrivals - 8.0,
            "end_time": arrivals + 1.0,
            "goal_arrival_time": arrivals,
            "from_well": 0,
            "to_well": list(to_wells),
            "is_reward": list(rewards),
            "is_patch_change": False,
        }
    )


# (label, trials, event_times, k) — covers no-history, < k, the cutoff
# boundary, all-omission, all-reward, multi-arm, and both k values.
#
# NB: exact ties *between two experiences'* outcome_known_time cannot occur in
# valid port-experience data — an outcome time is bounded within its own dwell
# and dwells are strictly ordered and non-overlapping (_prepare_port_experiences
# enforces this), so no two outcomes coincide. The boundary case below (events
# landing exactly on an outcome time) is the reachable searchsorted edge; and
# ties could not diverge the two forms regardless, since both consume the same
# stable-sorted cutoff array.
_CASES = [
    ("empty_history", pd.DataFrame(), np.array([5.0, 50.0]), 10),
    (
        "fewer_than_k",
        _trials([0, 0, 0], [True, False, True]),
        np.array([5.0, 12.0, 22.0, 100.0]),
        10,
    ),
    (
        "fewer_than_k_small_k",
        _trials([0, 0, 0], [True, False, True]),
        np.array([5.0, 12.0, 22.0, 100.0]),
        2,
    ),
    (
        # Events landing exactly on outcome_known_time values exercise the
        # searchsorted(side="right") cutoff boundary shared by both forms.
        "events_on_outcome_boundary",
        _trials([0, 0, 1, 0], [True, False, True, True]),
        np.array([9.0, 19.0, 29.0, 39.0, 100.0]),
        10,
    ),
    (
        "all_omission",
        _trials([0, 1, 2, 0], [False, False, False, False]),
        np.array([1.0, 15.0, 25.0, 45.0]),
        10,
    ),
    (
        "all_reward",
        _trials([0, 1, 2, 0], [True, True, True, True]),
        np.array([1.0, 15.0, 25.0, 45.0]),
        10,
    ),
    (
        "multi_arm_mixed",
        _trials(
            [0, 1, 0, 2, 1, 0], [True, False, True, True, False, False]
        ),
        np.array([1.0, 12.0, 22.0, 32.0, 42.0, 52.0, 100.0]),
        3,
    ),
    (
        "multi_arm_long_window",
        _trials(
            [0, 1, 0, 2, 1, 0], [True, False, True, True, False, False]
        ),
        np.array([1.0, 12.0, 22.0, 32.0, 42.0, 52.0, 100.0]),
        10,
    ),
]


@pytest.mark.parametrize("label,trials,event_times,k", _CASES, ids=[c[0] for c in _CASES])
def test_recent_reward_rate_vectorized_equals_loop(label, trials, event_times, k):
    """Vectorized production form == nested-loop reference, exactly (NaN-aware)."""
    pe = port_experiences_from_trials(trials, prepend_initial=False)
    vectorized = recent_reward_rate_all_events(event_times, pe, WELL_TO_ARM, k=k)
    loop = _recent_reward_rate_loop(event_times, pe, WELL_TO_ARM, k=k)
    # assert_array_equal treats NaNs in matching positions as equal.
    np.testing.assert_array_equal(vectorized, loop, err_msg=f"case={label}")
