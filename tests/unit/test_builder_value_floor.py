"""Value-level floor for the covariate builders.

The schema and timing gates check a covariate's name and availability window,
not its value. These tests pin the value at the exact places a wrong operation
would slip through -- window edges, searchsorted sides, join types, sign/offset
in reward and recency arithmetic -- so a builder change that alters a covariate
value fails a named test at the line instead of surfacing as an opaque table
diff. The starting coverage and the survivor list this file closes are in
docs/design/builder_test_floor.md.

Each `side=`/boundary test is constructed so the query sits exactly on a
cutoff: that is the only input where `searchsorted(side="right")` and
`side="left"` differ, so the assertion distinguishes them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay.covariates.history import (
    attach_arm_uncertainty,
    lagged_trial_rewards,
    recent_trial_reward_rate,
    rolling_reward_count_all_events,
    visit_recency_and_never_visited_all_events,
)
from tests.unit._port_experience_fixtures import port_experiences_from_trials

_WELL_TO_ARM = {well: well for well in range(6)}  # each leaf well -> its own arm


def _trials(end_times, is_reward, to_well=None) -> pd.DataFrame:
    end = np.asarray(end_times, dtype=float)
    data = {
        "movement_start_time": end - 0.5,
        "end_time": end,
        "is_reward": np.asarray(is_reward),
    }
    if to_well is not None:
        data["to_well"] = np.asarray(to_well)
    return pd.DataFrame(data)


# ── searchsorted side= : the window-edge semantics (convention A) ──


def test_recent_trial_reward_rate_includes_experience_on_cutoff():
    """side='right' counts the experience whose outcome is known exactly at the
    query time; side='left' would drop it. (history/ recent_trial_reward_rate)"""
    pe = port_experiences_from_trials(
        _trials([1.0, 2.0, 3.0], [1, 0, 1]), prepend_initial=False
    )
    out = recent_trial_reward_rate(np.array([2.0]), pe, k=10)
    assert out[0] == 0.5  # mean([1, 0]); a 'left' flip would give 1.0


def test_lagged_trial_rewards_lag1_is_current_outcome_on_cutoff():
    """On the cutoff, lag1 is the current outcome (side='right'); a 'left' flip
    shifts every lag back one. (history/ lagged_trial_rewards)"""
    pe = port_experiences_from_trials(
        _trials([1.0, 2.0, 3.0], [1, 0, 1]), prepend_initial=False
    )
    out = lagged_trial_rewards(np.array([2.0]), pe, k=3)
    assert out[0, 0] == 0.0  # lag1 = the t=2 outcome (r=0); 'left' -> 1.0
    assert out[0, 1] == 1.0  # lag2 = the t=1 outcome (r=1)
    assert np.isnan(out[0, 2])  # only two outcomes are known by t=2


def test_rolling_reward_count_counts_arm_reward_on_cutoff():
    """The current arm's reward at the cutoff is inside the window (side='right').
    (history/ rolling_reward_count_all_events)"""
    pe = port_experiences_from_trials(
        _trials([1.0, 2.0], [1, 1], to_well=[0, 1]), prepend_initial=False
    )
    out = rolling_reward_count_all_events(np.array([2.0]), pe, _WELL_TO_ARM, window=10)
    assert out[0, 1] == 1.0  # arm 1 (well 1) reward at t=2 counted; 'left' -> 0
    assert out[0, 0] == 1.0  # arm 0 reward at t=1 counted


def test_visit_recency_counts_arrival_on_cutoff():
    """An arrival exactly at the event time marks the arm visited (side='right').
    (history/ visit_recency_and_never_visited_all_events)"""
    pe = port_experiences_from_trials(
        _trials([1.0, 2.0], [0, 0], to_well=[0, 1]), prepend_initial=False
    )
    _recency, never = visit_recency_and_never_visited_all_events(
        np.array([2.0]), pe, _WELL_TO_ARM
    )
    assert never[0, 1] == 0.0  # well 1 arrived at t=2 -> visited; 'left' -> 1.0


# ── merge how='left' : missing stops survive as NaN, not dropped ──


def _contingency(stop_ids) -> pd.DataFrame:
    data = {"stop_id": np.asarray(stop_ids)}
    for port in range(6):
        data[f"reward_belief_variance_port_{port}"] = np.full(len(stop_ids), 0.1)
    return pd.DataFrame(data)


def test_attach_arm_uncertainty_left_join_keeps_missing_stop_as_nan():
    """A query stop absent from the contingency survives the left join as NaN;
    a right-join flip would drop it and misalign every row.
    (history/ attach_arm_uncertainty)"""
    arm = np.array([0, 0])
    stop_id = np.array([1, 2])  # stop 2 is not in the contingency
    out = attach_arm_uncertainty(arm, stop_id, _contingency([1]))
    assert len(out) == 2  # left join preserves both query rows
    assert out[0] == 0.1
    assert np.isnan(out[1])


def test_attach_arm_uncertainty_trunk_arm_takes_zero_not_a_leaf_value():
    """Trunk arms (6-8) are not reward sites: present stops take 0.0, never a
    leaf variance. `arm < len(WELL_ARMS)` must stay strict `<`.
    (history/ attach_arm_uncertainty leaf/trunk split)"""
    out = attach_arm_uncertainty(np.array([0, 6]), np.array([1, 1]), _contingency([1]))
    assert out[0] == 0.1  # leaf arm 0
    assert out[1] == 0.0  # trunk arm 6 present in contingency -> 0.0, not 0.1


def test_attach_arm_uncertainty_rejects_out_of_range_variance():
    """A Beta posterior variance outside [0, 0.25] is impossible; the guard must
    fire. (history/ attach_arm_uncertainty range check)"""
    bad = _contingency([1])
    bad.loc[0, "reward_belief_variance_port_0"] = 0.9  # > 0.25
    with pytest.raises(ValueError, match="variance"):
        attach_arm_uncertainty(np.array([0]), np.array([1]), bad)


# ── window denominator and input-integrity guards ──


def test_recent_trial_reward_rate_denominator_is_window_size():
    """With more experiences than k, the rate divides by the window size k, not
    the total count. (history/ recent_trial_reward_rate n_in_window)"""
    pe = port_experiences_from_trials(
        _trials([1.0, 2.0, 3.0, 4.0, 5.0], [0, 0, 0, 1, 1]), prepend_initial=False
    )
    out = recent_trial_reward_rate(np.array([6.0]), pe, k=2)
    assert out[0] == 1.0  # last two are rewarded -> 2/2; a wrong denominator gives 0.25


def _hand_experiences(arrival, outcome, is_reward, ports) -> pd.DataFrame:
    n = len(arrival)
    return pd.DataFrame(
        {
            "experience_id": np.arange(n),
            "movement_trial_number": pd.array(range(n), dtype="Int64"),
            "is_initial_experience": [False] * n,
            "is_boundary_derived": [False] * n,
            "outcome_observed": [True] * n,
            "port": np.asarray(ports, dtype=int),
            "patch": [1 + p // 2 for p in ports],
            "arrival_time": np.asarray(arrival, dtype=float),
            "departure_time": np.asarray(arrival, dtype=float) + 0.1,
            "reward_on_time": np.where(
                np.asarray(is_reward, dtype=bool),
                np.asarray(outcome, dtype=float),
                np.nan,
            ),
            "outcome_known_time": np.asarray(outcome, dtype=float),
            "is_reward": np.asarray(is_reward, dtype=bool),
        }
    )


def test_history_rejects_nonmonotonic_arrival():
    """Arrivals must be strictly increasing; the integrity guard must fire.
    (history/ _prepare_port_experiences)"""
    bad = _hand_experiences([2.0, 1.0], [2.0, 3.0], [1, 0], [0, 1])
    with pytest.raises(ValueError):
        recent_trial_reward_rate(np.array([5.0]), bad, k=10)


# ── I/O purity: covariate builders are functions of their frames, not disk ──


def test_history_builders_are_io_pure(monkeypatch):
    """No covariate builder reads a table from disk; the only disk read is the
    intentional load_contingency loader, which callers invoke once at the top and
    pass the frame down. This pins that contract so a future builder cannot
    silently reintroduce a hidden read.
    """
    # ``read_table`` is the single intentional disk read; after the history
    # split it lives only in the value_block submodule (load_contingency). The
    # reward/recency/value builders import no reader at all, so they cannot read.
    from continuum_swr_replay.analysis.replay.covariates.history import (
        value_block as value_mod,
    )

    def _explode(*args, **kwargs):
        raise AssertionError("a covariate builder read a table from disk")

    monkeypatch.setattr(value_mod, "read_table", _explode)

    pe = port_experiences_from_trials(
        _trials([1.0, 2.0], [1, 0]), prepend_initial=False
    )
    # A representative spread of builders, all given only in-memory frames.
    assert recent_trial_reward_rate(np.array([3.0]), pe, k=10)[0] == 0.5
    assert lagged_trial_rewards(np.array([3.0]), pe, k=2).shape == (1, 2)
    out = attach_arm_uncertainty(np.array([0]), np.array([1]), _contingency([1]))
    assert out[0] == 0.1
