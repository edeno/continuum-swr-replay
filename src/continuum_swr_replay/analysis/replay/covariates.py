"""Behavioral covariate definitions for the replay rate / content / choice models.

Single home for the per-event / per-stop / per-arm behavioral covariates (reward
history, visit recency, model-free uncertainty) plus the local fitted-alpha
leaky-Q learner (``leaky_q_value_diff``). Note: the per-arm value block
(``attach_contingency_value_block`` -> ``p_reward_arm`` / ``q_arm``) is sourced
from ``qvals_table`` -- ``q_arm`` is the beta-model belief (Q), NOT leaky-Q; the
only leaky-Q construct in this file is the local ``leaky_q_value_diff``.
Implements the constructs specified in
``docs/covariate_dictionary.md``. These definitions are re-exported from
``_helpers`` (a shim) so existing call sites keep resolving.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from continuum_swr_replay.data_loaders.constants import (
    ALL_ARMS,
    WELL_ARMS,
    WELL_PATCH_MAPPING,
)

from ._helpers import _trial_cutoff_times, build_well_to_arm_map

_SESSION_COLS = ["animal", "day", "epoch"]
_P_REWARD_PORT_COLS = [f"p_reward_port{i}" for i in range(6)]
_Q_PORT_COLS = [f"q_port{i}" for i in range(6)]
# Per-leaf beta-model posterior variance (qvals_table betadist_var{1..6}); the
# variance counterpart of _Q_PORT_COLS (mean). betadist_var{i} indexes leaf port
# i-1 (1-indexed columns), so leaf arm j -> _BETADIST_VAR_COLS[j] = betadist_var{j+1}.
_BETADIST_VAR_COLS = [f"betadist_var{i}" for i in range(1, 7)]


UNCERTAINTY_COVARIATES = (
    "omission_streak",
    "recent_switch_rate",
    "wsls_deviation",
    "leaky_q_value_diff",
)


def center_visit_recency_within_event(recency: np.ndarray) -> np.ndarray:
    """Center leaf-arm recency within each event without using future events.

    Only the six reward-site arms participate in the mean. Structural trunk
    values remain zero once history exists, and an all-unknown event remains
    all-NaN. This retains a relative-recency predictor while guaranteeing that
    appending later events cannot change an earlier row.
    """
    values = np.asarray(recency, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(ALL_ARMS):
        raise ValueError(
            "recency must have shape (n_events, len(ALL_ARMS)); "
            f"got {values.shape}"
        )
    output = values.copy()
    arm_positions = {arm: index for index, arm in enumerate(ALL_ARMS)}
    leaf_positions = np.asarray([arm_positions[arm] for arm in WELL_ARMS])
    leaf_values = values[:, leaf_positions]
    finite = np.isfinite(leaf_values)
    counts = finite.sum(axis=1)
    totals = np.where(finite, leaf_values, 0.0).sum(axis=1)
    means = np.divide(
        totals,
        counts,
        out=np.full(len(values), np.nan),
        where=counts > 0,
    )
    output[:, leaf_positions] = leaf_values - means[:, None]
    return output


def visit_recency_all_events(
    event_times: np.ndarray,
    trials: pd.DataFrame,
    well_to_arm: dict[int, int],
) -> np.ndarray:
    """Completed trials since each arm was last visited, for all events.

    A reachable arm not yet visited at an event receives
    ``n_completed_trials + 1``. This right-censored value depends only on
    history available at the event; appending later trials cannot change it.
    Trunk/non-reachable arms are structural ``0`` once history exists, and all
    arms are ``NaN`` before the first completed trial.

    """
    recency, _ = _visit_recency_and_never_visited_all_events(
        event_times, trials, well_to_arm
    )
    return recency


def arm_never_visited_all_events(
    event_times: np.ndarray,
    trials: pd.DataFrame,
    well_to_arm: dict[int, int],
) -> np.ndarray:
    """Indicate arms not visited before each event.

    Returns a float ``(n_events, len(ALL_ARMS))`` array so unknown history can
    remain ``NaN``. With completed history, reachable never-visited arms are
    ``1``, visited arms are ``0``, and trunk/non-reachable arms are structural
    ``0``.
    """
    _, never_visited = _visit_recency_and_never_visited_all_events(
        event_times, trials, well_to_arm
    )
    return never_visited


def _visit_recency_and_never_visited_all_events(
    event_times: np.ndarray,
    trials: pd.DataFrame,
    well_to_arm: dict[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute time-local recency and novelty from one shared history pass."""
    n_events = len(event_times)
    n_arms = len(ALL_ARMS)
    n_trials_total = len(trials)
    recency = np.full((n_events, n_arms), np.nan)
    never_visited = np.full((n_events, n_arms), np.nan)

    if n_trials_total == 0:
        return recency, never_visited

    arm_idx = {arm: j for j, arm in enumerate(ALL_ARMS)}
    mapped_wells = set(well_to_arm)
    expected_wells = set(WELL_PATCH_MAPPING)
    if mapped_wells != expected_wells:
        missing = sorted(expected_wells - mapped_wells)
        unexpected = sorted(mapped_wells - expected_wells)
        raise ValueError(
            "well_to_arm must cover every canonical reward-site well exactly; "
            f"missing={missing}, unexpected={unexpected}"
        )
    mapped_leaf_arms = set(well_to_arm.values())
    expected_leaf_arms = set(WELL_ARMS)
    if mapped_leaf_arms != expected_leaf_arms:
        missing = sorted(expected_leaf_arms - mapped_leaf_arms)
        unexpected = sorted(mapped_leaf_arms - expected_leaf_arms)
        raise ValueError(
            "well_to_arm must cover every canonical reward-site leaf arm "
            f"exactly; missing={missing}, unexpected={unexpected}"
        )
    if len(well_to_arm) != len(expected_leaf_arms):
        raise ValueError(
            "well_to_arm must map exactly one well to each canonical leaf arm"
        )
    reachable_arms = expected_leaf_arms

    # ``np.searchsorted`` assumes chronological cutoff times. A visit becomes
    # available at goal arrival (or the per-row end-time fallback).
    trials = trials.assign(_cutoff=_trial_cutoff_times(trials))
    trials = trials.sort_values("_cutoff").reset_index(drop=True)
    cutoff_times = trials["_cutoff"].to_numpy()
    to_wells = trials["to_well"].to_numpy()
    cutoffs = np.searchsorted(cutoff_times, event_times, side="right")
    has_history = cutoffs > 0

    visit_lists: list[list[int]] = [[] for _ in ALL_ARMS]
    for trial_idx, well in enumerate(to_wells):
        if pd.isna(well):
            continue
        arm = well_to_arm.get(int(well))
        if arm in arm_idx:
            visit_lists[arm_idx[arm]].append(trial_idx)

    for j, arm in enumerate(ALL_ARMS):
        if arm not in reachable_arms:
            recency[has_history, j] = 0.0
            never_visited[has_history, j] = 0.0
            continue

        # Right-censored fill for a reachable arm with no visit yet. Unlike a
        # session-wide default, this can be known at the event time.
        recency[has_history, j] = cutoffs[has_history] + 1.0
        never_visited[has_history, j] = 1.0

        visits = np.asarray(visit_lists[j], dtype=np.intp)
        if visits.size == 0:
            continue
        n_visits = np.searchsorted(visits, cutoffs, side="left")
        was_visited = has_history & (n_visits > 0)
        last_visit = visits[np.clip(n_visits[was_visited] - 1, 0, None)]
        recency[was_visited, j] = cutoffs[was_visited] - last_visit
        never_visited[was_visited, j] = 0.0

    return recency, never_visited


def cumulative_reward_all_events(
    event_times: np.ndarray,
    trials: pd.DataFrame,
    well_to_arm: dict[int, int],
    window: int = 10,
) -> np.ndarray:
    """Reward count per arm over the last ``window`` trials, for all events.

    Despite the name (and the ``cumulative_reward`` column it fills) this is a
    **rolling k=10 window**, not a running total: the count is
    ``cumsum[t] - cumsum[t - window]``, so a reward more than ``window`` trials
    back contributes nothing. The name is kept because the column ships in
    ``event_arm_table.pkl`` and is named in ``model_specs``; renaming it is an
    artifact-invalidating migration, not a cleanup.

    ``NaN`` where no trial has completed yet -- unknown history, not zero
    reward.

    Returns ``(n_events, n_arms)`` array.
    """
    n_arms = len(ALL_ARMS)
    arm_idx = {a: j for j, a in enumerate(ALL_ARMS)}
    n_events = len(event_times)
    n_trials_total = len(trials)

    if n_trials_total == 0:
        # No completed trials → unknown history, not zero reward
        # (matches the sibling helpers).
        return np.full((n_events, n_arms), np.nan)

    # Sort defensively by the goal-arrival cutoff (see
    # ``visit_recency_all_events`` and ``_trial_cutoff_times``).
    trials = trials.assign(_cutoff=_trial_cutoff_times(trials))
    trials = trials.sort_values("_cutoff").reset_index(drop=True)
    cutoff_times = trials["_cutoff"].to_numpy()
    to_wells = trials["to_well"].to_numpy()
    is_reward = trials["is_reward"].to_numpy().astype(bool)

    # Build per-arm cumulative reward arrays (cumsum over trial order)
    # reward_at[j][ti] = 1 if trial ti visits arm j and is rewarded
    reward_per_arm = np.zeros((n_trials_total, n_arms), dtype=float)
    for ti in range(n_trials_total):
        if not is_reward[ti]:
            continue
        w = to_wells[ti]
        if np.isnan(w):
            continue
        arm_id = well_to_arm.get(int(w))
        if arm_id is not None and arm_id in arm_idx:
            reward_per_arm[ti, arm_idx[arm_id]] = 1.0

    # cumsum with a leading zero row for easy slicing
    cum = np.zeros((n_trials_total + 1, n_arms), dtype=float)
    np.cumsum(reward_per_arm, axis=0, out=cum[1:])

    cutoffs = np.searchsorted(cutoff_times, event_times, side="right")
    window_starts = np.maximum(cutoffs - window, 0)

    out = cum[cutoffs] - cum[window_starts]

    # No-past-trials → NaN, matching the sibling helpers' contract.
    no_past = cutoffs == 0
    if no_past.any():
        out[no_past] = np.nan
    return out


def recent_reward_rate_all_events(
    event_times: np.ndarray,
    trials: pd.DataFrame,
    well_to_arm: dict[int, int],
    k: int = 5,
) -> np.ndarray:
    """Fraction of last ``k`` visits to each arm that were rewarded, for all events.

    Returns ``(n_events, n_arms)`` array.

    Trunk arms (graph edges that touch no well — never a trial
    destination, so never a reward site) are filled with ``0.0``
    rather than ``NaN``: their reward rate is structurally zero, not
    unknown. This matters because the conditional-Poisson fitter
    drops an entire event if *any* of its nine arm rows is NaN, and
    every event has three trunk-arm rows — leaving them NaN would
    annihilate the M3+ event set.

    A leaf arm not yet visited in a session *with* prior trials is filled
    ``0.0`` (no rewards observed on it yet), retaining the novel-arm /
    early-session regime. Only no-trial-history rows stay ``NaN``. Models that
    use recency also carry ``arm_never_visited``, separating this censored
    state from an observed zero reward rate.
    """
    n_arms = len(ALL_ARMS)
    n_events = len(event_times)
    n_trials_total = len(trials)
    # Arms reachable as a trial destination = leaf arms (values of the
    # well→arm map). Anything else is a trunk arm.
    reachable_arms = set(well_to_arm.values())

    if n_trials_total == 0:
        return np.full((n_events, n_arms), np.nan)

    # Sort defensively by the goal-arrival cutoff (see
    # ``visit_recency_all_events`` and ``_trial_cutoff_times``).
    trials = trials.assign(_cutoff=_trial_cutoff_times(trials))
    trials = trials.sort_values("_cutoff").reset_index(drop=True)
    cutoff_times = trials["_cutoff"].to_numpy()
    to_wells = trials["to_well"].to_numpy()
    is_reward = trials["is_reward"].to_numpy().astype(float)

    # Per-arm: arrays of visit trial indices and corresponding rewards
    arm_visits: list[np.ndarray] = []
    arm_rewards: list[np.ndarray] = []
    for arm in ALL_ARMS:
        vis_idx = []
        vis_rew = []
        for ti in range(n_trials_total):
            w = to_wells[ti]
            if np.isnan(w):
                continue
            aid = well_to_arm.get(int(w))
            if aid == arm:
                vis_idx.append(ti)
                vis_rew.append(is_reward[ti])
        arm_visits.append(np.array(vis_idx, dtype=np.intp))
        arm_rewards.append(np.array(vis_rew, dtype=float))

    cutoffs = np.searchsorted(cutoff_times, event_times, side="right")

    # Events with no past trials -> NaN
    no_past = cutoffs == 0
    out = np.full((n_events, n_arms), np.nan)

    for j in range(n_arms):
        # Trunk arm: reward rate is structurally 0 wherever there is
        # any trial history. ``no_past`` rows stay NaN.
        if ALL_ARMS[j] not in reachable_arms:
            out[~no_past, j] = 0.0
            continue
        visits = arm_visits[j]
        rewards = arm_rewards[j]
        if visits.size == 0:
            # ENCODE: leaf never visited in this session, but there ARE prior trials ->
            # 0.0 (no rewards observed yet), not NaN. no_past rows stay NaN.
            out[~no_past, j] = 0.0
            continue
        # n_vis_before[i] = number of visits to this arm before cutoff
        n_vis_before = np.searchsorted(visits, cutoffs, side="left")
        for i in range(n_events):
            if no_past[i]:
                continue
            nv = n_vis_before[i]
            if nv == 0:
                # ENCODE: leaf not yet visited BEFORE this event -> 0.0, not a dropped NaN.
                out[i, j] = 0.0
                continue
            start = max(0, nv - k)
            out[i, j] = rewards[start:nv].mean()

    return out


def recent_trial_reward_rate(
    query_times: np.ndarray,
    trials: pd.DataFrame,
    k: int = 10,
) -> np.ndarray:
    """Rewarded fraction of the most recent ``k`` trials before each query time.

    The arm-agnostic counterpart of :func:`recent_reward_rate_all_events`:
    the denominator is trials, not visits to one arm. A trial is "before"
    a query time if its goal-arrival cutoff (see :func:`_trial_cutoff_times`)
    is at or before that time.

    Parameters
    ----------
    query_times : np.ndarray, shape (n_query,)
        Times (s) at which to evaluate the recent reward rate — e.g.
        immobility-bout start times for the stop-level rate model.
    trials : pd.DataFrame
        Must carry ``is_reward`` plus the columns ``_trial_cutoff_times``
        consumes.
    k : int
        Window length in trials.

    Returns
    -------
    np.ndarray, shape (n_query,)
        Rewarded fraction of the last ``k`` trials; ``NaN`` for query
        times that precede every trial cutoff.
    """
    query_times = np.asarray(query_times, dtype=float)
    if len(trials) == 0:
        return np.full(query_times.size, np.nan)

    # Sort by the goal-arrival cutoff so ``searchsorted`` is valid; see
    # ``_trial_cutoff_times`` for why this cutoff (not ``end_time``).
    cutoff_times = np.sort(_trial_cutoff_times(trials))
    is_reward = (
        trials.assign(_cutoff=_trial_cutoff_times(trials))
        .sort_values("_cutoff")["is_reward"]
        .to_numpy()
        .astype(float)
    )
    cum = np.concatenate([[0.0], np.cumsum(is_reward)])

    counts = np.searchsorted(cutoff_times, query_times, side="right")
    window_starts = np.maximum(counts - k, 0)
    n_in_window = counts - window_starts
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = (cum[counts] - cum[window_starts]) / n_in_window
    rate[counts == 0] = np.nan
    return rate


def lagged_trial_rewards(
    query_times: np.ndarray,
    trials: pd.DataFrame,
    k: int = 10,
) -> np.ndarray:
    """Reward outcome of each of the last ``k`` trials before each query time.

    The per-lag companion to :func:`recent_trial_reward_rate` (which averages
    the last ``k``): same trial-cutoff mapping, but returns the individual
    outcomes so a rate model can estimate the reward *kernel* across trials —
    ``reward_lag{j}`` = ``is_reward`` of the j-th most recent trial. ``lag1`` is
    the immediate previous trial (the dwell-confounded one); ``lag>=2`` are
    earlier trials and so do not set the current stop's duration (dwell-clean).

    Parameters
    ----------
    query_times : np.ndarray, shape (n_query,)
        Times (s) at which to evaluate — immobility-bout start times.
    trials : pd.DataFrame
        Must carry ``is_reward`` plus what :func:`_trial_cutoff_times` needs.
    k : int
        Number of lags.

    Returns
    -------
    np.ndarray, shape (n_query, k)
        Column ``j-1`` = reward (0/1) of the j-th most recent trial; ``NaN``
        where fewer than ``j`` trials precede the query time.
    """
    query_times = np.asarray(query_times, dtype=float)
    out = np.full((query_times.size, k), np.nan)
    if len(trials) == 0:
        return out
    cutoff = _trial_cutoff_times(trials)
    order = np.argsort(cutoff)
    cutoff_sorted = cutoff[order]
    is_reward = trials["is_reward"].to_numpy().astype(float)[order]
    counts = np.searchsorted(cutoff_sorted, query_times, side="right")
    for j in range(1, k + 1):
        idx = counts - j
        valid = idx >= 0
        out[valid, j - 1] = is_reward[idx[valid]]
    return out


def _leaky_q_value_diff(
    current_arm: np.ndarray,
    reward_outcome: np.ndarray,
    q_alpha: float,
) -> np.ndarray:
    """Per-stop experienced-value choice margin from a leaky per-arm Q.

    For each arm ``a`` a leaky reward-rate is updated in stop order
    ``Q_a <- (1 - q_alpha) Q_a + q_alpha * reward_outcome`` (init 0.5 for all
    arms). The covariate at a stop is the gap between the best and second-best
    arm value ``sort(Q)[-1] - sort(Q)[-2]`` evaluated **before** that stop's
    update (so it is the experienced choice margin the animal faced going in).

    Parameters
    ----------
    current_arm : np.ndarray, shape (n_stops,)
        Arm 0..5 visited at each stop (in stop order). A ``NaN`` arm still
        yields the current pre-update Q-spread margin (a real value -- the
        choice margin does not depend on which arm is visited) but does not
        update any arm's ``Q`` (the unknown arm cannot be credited).
    reward_outcome : np.ndarray, shape (n_stops,)
        Binary reward (0/1) at each stop.
    q_alpha : float
        Leaky-integration learning rate.

    Returns
    -------
    np.ndarray, shape (n_stops,)
        Pre-update best-minus-second-best arm value at each stop.
    """
    q = np.full(6, 0.5)
    out = np.full(current_arm.size, np.nan)
    for i in range(current_arm.size):
        ordered = np.sort(q)
        out[i] = ordered[-1] - ordered[-2]
        arm = current_arm[i]
        if not np.isnan(arm):
            a = int(arm)
            q[a] = (1.0 - q_alpha) * q[a] + q_alpha * reward_outcome[i]
    return out


def _omission_streak(reward_lag_matrix: np.ndarray) -> np.ndarray:
    """Count leading consecutive omissions in ``reward_lag1, reward_lag2, ...``.

    Parameters
    ----------
    reward_lag_matrix : np.ndarray, shape (n_stops, n_lags)
        Column ``j`` is ``reward_lag{j+1}`` (1 = reward, 0 = omission, NaN =
        no such prior trial).

    Returns
    -------
    np.ndarray, shape (n_stops,)
        Number of leading 0s before the first 1; a ``NaN`` lag terminates the
        streak (counts the omissions seen before it).
    """
    n_stops = reward_lag_matrix.shape[0]
    out = np.zeros(n_stops)
    for i in range(n_stops):
        streak = 0
        for val in reward_lag_matrix[i]:
            if np.isnan(val) or val == 1.0:
                break
            streak += 1
        out[i] = streak
    return out


def _recent_switch_rate_per_visit(
    stops: pd.DataFrame, switch_window: int
) -> np.ndarray:
    """Roll completed switch outcomes at visit grain, then broadcast to bouts.

    The consecutive ``(prev_arm, current_arm)`` pair identifies the most
    recently completed trip in canonical stop tables and is constant across
    all bouts with that completed-choice context.  ``is_after_switch`` is
    therefore known at every bout start, unlike the same row's retrospective
    ``is_stay``.
    """
    prev_arm = stops["prev_arm"].to_numpy(dtype=float)
    current_arm = stops["current_arm"].to_numpy(dtype=float)
    is_after_switch = stops["is_after_switch"].to_numpy(dtype=float)
    if len(stops) == 0:
        return np.array([], dtype=float)

    same_prev = (prev_arm[1:] == prev_arm[:-1]) | (
        np.isnan(prev_arm[1:]) & np.isnan(prev_arm[:-1])
    )
    same_current = (current_arm[1:] == current_arm[:-1]) | (
        np.isnan(current_arm[1:]) & np.isnan(current_arm[:-1])
    )
    same_visit = same_prev & same_current
    visit_id = np.r_[0, np.cumsum(~same_visit)]
    n_visits = int(visit_id[-1]) + 1
    visit_switch = np.full(n_visits, np.nan)
    for visit in range(n_visits):
        values = is_after_switch[visit_id == visit]
        finite = values[np.isfinite(values)]
        if finite.size:
            if finite.size != values.size or not np.all(finite == finite[0]):
                raise ValueError(
                    "is_after_switch must be constant within each completed "
                    "visit (consecutive rows with the same prev/current arms)"
                )
            visit_switch[visit] = finite[0]

    # The trip leading into the first observed visit can predate the recorded
    # session.  Treat that carry-in state as unknown even if a stale source
    # supplied a numeric label.
    visit_switch[0] = np.nan

    visit_rate = (
        pd.Series(visit_switch)
        .rolling(switch_window, min_periods=1)
        .mean()
        .to_numpy()
    )
    return visit_rate[visit_id]


def add_uncertainty_covariates(
    stop_table: pd.DataFrame,
    *,
    switch_window: int = 5,
    q_alpha: float = 0.3,
) -> pd.DataFrame:
    """Add four model-free behavioral-uncertainty covariates per stop.

    Computed from existing ``stop_table`` columns only (no new data), grouped by
    session ``(animal, day, epoch)`` and ordered by ``stop_idx`` within each
    session. These are the experienced/subjective complement to the nominal
    contingency values; both enter the rate model so nominal and experienced
    value are dissociable. The input is not mutated; the returned frame keeps
    the input's index and row count.

    The four covariates:

    - ``omission_streak``: leading consecutive omissions = count of leading 0s
      in ``reward_lag1, reward_lag2, ...`` (stop at the first 1; a ``NaN`` lag
      terminates the streak).
    - ``recent_switch_rate``: rolling mean of ``is_after_switch`` over the last
      ``switch_window`` completed visits. Multiple stop/bout rows sharing one
      completed-choice context receive the same value; the current visit's
      eventual ``is_stay`` is never used.
    - ``wsls_deviation``: 1.0 (deviated from win-stay/lose-shift) unless the stop
      is ``(is_after_reward & is_stay)`` or ``(is_after_omission & ~is_stay)``,
      in which case 0.0. ``NaN`` when neither ``is_after_reward`` nor
      ``is_after_omission`` is set (no defined prior outcome, e.g. the first
      stop of a session) -- such rows are dropped at fit time rather than
      counted as a deviation.
    - ``leaky_q_value_diff``: best-minus-second-best per-arm leaky reward-rate
      (init 0.5, rate ``q_alpha``) evaluated **before** the stop's update -- the
      experienced-value choice margin.

    ``q_alpha`` and ``switch_window`` are free parameters (a fitted RL model
    would learn them); defaults ``0.3`` / ``5``. Report sensitivity.

    Parameters
    ----------
    stop_table : pd.DataFrame
        Per-stop table. Must carry ``animal``, ``day``, ``epoch``, ``stop_idx``,
        ``prev_arm``, ``is_after_switch``, ``is_stay``, ``is_after_reward``,
        ``is_after_omission``, ``reward_outcome``, ``current_arm``, and
        ``reward_lag1..`` columns.
    switch_window : int
        Rolling window (in completed visits) for ``recent_switch_rate``.
    q_alpha : float
        Leaky-integration learning rate for ``leaky_q_value_diff``.

    Returns
    -------
    pd.DataFrame
        Copy of ``stop_table`` with the four covariate columns added.
    """
    out = stop_table.copy()
    lag_cols = sorted(
        (c for c in out.columns if c.startswith("reward_lag")),
        key=lambda c: int(c.removeprefix("reward_lag")),
    )

    omission = pd.Series(np.nan, index=out.index, dtype=float)
    switch_rate = pd.Series(np.nan, index=out.index, dtype=float)
    wsls = pd.Series(np.nan, index=out.index, dtype=float)
    leaky = pd.Series(np.nan, index=out.index, dtype=float)

    for _, idx in out.groupby(_SESSION_COLS, sort=False).groups.items():
        g = out.loc[idx].sort_values("stop_idx")
        pos = g.index

        if lag_cols:
            omission.loc[pos] = _omission_streak(g[lag_cols].to_numpy(dtype=float))
        else:
            omission.loc[pos] = 0.0

        switch_rate.loc[pos] = _recent_switch_rate_per_visit(g, switch_window)

        after_reward = g["is_after_reward"].to_numpy(dtype=float)
        after_omission = g["is_after_omission"].to_numpy(dtype=float)
        stay = g["is_stay"].to_numpy(dtype=float)
        win_stay = (after_reward == 1.0) & (stay == 1.0)
        lose_shift = (after_omission == 1.0) & (stay == 0.0)
        wsls_vals = np.where(win_stay | lose_shift, 0.0, 1.0)
        # No defined prior outcome (neither after-reward nor after-omission set,
        # e.g. the first stop of a session) -> NaN, so the row drops at fit time
        # instead of being counted as a deviation.
        no_prior = (after_reward != 1.0) & (after_omission != 1.0)
        wsls_vals[no_prior] = np.nan
        wsls.loc[pos] = wsls_vals

        leaky.loc[pos] = _leaky_q_value_diff(
            g["current_arm"].to_numpy(dtype=float),
            g["reward_outcome"].to_numpy(dtype=float),
            q_alpha,
        )

    out["omission_streak"] = omission
    out["recent_switch_rate"] = switch_rate
    out["wsls_deviation"] = wsls
    out["leaky_q_value_diff"] = leaky
    return out


def visit_recency_per_arm(
    event_time: float,
    trials: pd.DataFrame,
    track_graph: nx.Graph,
) -> dict[int, float]:
    """Trials since each arm was last visited."""
    w2a = build_well_to_arm_map(track_graph)
    row = visit_recency_all_events(np.array([event_time]), trials, w2a)
    return {arm: float(row[0, j]) for j, arm in enumerate(ALL_ARMS)}


def arm_never_visited_per_arm(
    event_time: float,
    trials: pd.DataFrame,
    track_graph: nx.Graph,
) -> dict[int, float]:
    """Indicate which reachable arms were unvisited before one event."""
    w2a = build_well_to_arm_map(track_graph)
    row = arm_never_visited_all_events(np.array([event_time]), trials, w2a)
    return {arm: float(row[0, j]) for j, arm in enumerate(ALL_ARMS)}


def cumulative_reward_per_arm(
    event_time: float,
    trials: pd.DataFrame,
    track_graph: nx.Graph,
    window: int = 10,
) -> dict[int, float]:
    """Reward count at each arm over the last ``window`` trials."""
    w2a = build_well_to_arm_map(track_graph)
    row = cumulative_reward_all_events(
        np.array([event_time]), trials, w2a, window=window
    )
    return {arm: float(row[0, j]) for j, arm in enumerate(ALL_ARMS)}


def recent_reward_rate_per_arm(
    event_time: float,
    trials: pd.DataFrame,
    track_graph: nx.Graph,
    k: int = 5,
) -> dict[int, float]:
    """Fraction of the last ``k`` visits to each arm that were rewarded."""
    w2a = build_well_to_arm_map(track_graph)
    row = recent_reward_rate_all_events(np.array([event_time]), trials, w2a, k=k)
    return {arm: float(row[0, j]) for j, arm in enumerate(ALL_ARMS)}


def _q_chosen(
    arm: np.ndarray,
    reward: np.ndarray,
    grp: np.ndarray,
    alpha: float,
    init: float = 0.5,
) -> np.ndarray:
    """Per-leaf delta-rule Q value of the chosen leaf, pre-update.

    Q[a] += alpha*(reward - Q[a]); Q resets to ``init`` at each new group (session),
    so within a session it updates continuously across contingency blocks -- the
    within-block-updating subjective value the block-constant nominal value lacks.

    The chosen-arm column of :func:`q_all_arm` by construction, so it delegates
    rather than re-implement the delta-rule rollout.
    """
    return q_all_arm(arm, reward, grp, alpha, init)[np.arange(len(arm)), arm]


def q_all_arm(
    arm: np.ndarray,
    reward: np.ndarray,
    grp: np.ndarray,
    alpha: float,
    init: float = 0.5,
) -> np.ndarray:
    """Per-leaf delta-rule Q **level of every arm**, pre-update, for all trials.

    The canonical leaky delta-rule rollout (``Q[a] += alpha*(reward - Q[a])``,
    ``Q`` reset to ``init`` at each new group), snapshotting the full 6-arm value
    vector *before* each trial's update. ``q_current`` is the chosen-arm column
    (``[i, arm[i]]``, which :func:`_q_chosen` returns) and ``q_arm`` is the
    per-arm value.

    Parameters
    ----------
    arm : np.ndarray, shape (n_trials,)
        Chosen leaf arm 0..5 at each trial, in trial order within group.
    reward : np.ndarray, shape (n_trials,)
        Binary reward (0/1) at each trial.
    grp : np.ndarray, shape (n_trials,)
        Group label (session); ``Q`` resets to ``init`` at each new group.
    alpha : float
        Leaky delta-rule learning rate.
    init : float
        Initial Q for every arm at the start of each group.

    Returns
    -------
    np.ndarray, shape (n_trials, 6)
        Row ``i`` is the pre-update Q value of each of the 6 leaf arms at trial ``i``.
    """
    out = np.empty((len(arm), 6))
    Q, prev = None, None
    for i in range(len(arm)):
        if grp[i] != prev:
            Q, prev = np.full(6, init), grp[i]
        out[i] = Q  # numpy row-assignment copies values, so the later in-place
        a = arm[i]  # update of Q does not alias the stored snapshot
        Q[a] += alpha * (reward[i] - Q[a])
    return out


def _fit_alpha(arm: np.ndarray, reward: np.ndarray, grp: np.ndarray) -> float:
    """alpha (per animal) that makes q_chosen best predict reward (Bernoulli NLL)."""

    def nll(a: float) -> float:
        p = np.clip(_q_chosen(arm, reward, grp, a), 1e-4, 1 - 1e-4)
        return -float(np.sum(reward * np.log(p) + (1 - reward) * np.log(1 - p)))

    return float(minimize_scalar(nll, bounds=(0.01, 0.99), method="bounded").x)


def load_contingency(path: Path) -> pd.DataFrame | None:
    """Load the per-stop value artifact (``qvals_table``) for the value block.

    Shared by the NonLocal (``prepare_data``) and ContFrag prep entry points.
    ``qvals_table`` is the *sole* source for ``p_reward_arm`` / ``q_arm``, so its
    absence must degrade visibly, not silently: present -> returns the frame and
    logs the row count; absent -> returns ``None`` and logs a warning (the
    builders then emit those columns as NaN, dropping uniformly at fit time).
    """
    path = Path(path)
    if path.exists():
        cont = pd.read_pickle(path)
        logging.info(
            "loaded qvals_table (%d stop rows) for p_reward_arm / q_arm", len(cont)
        )
        return cont
    logging.warning(
        "no qvals_table at %s — p_reward_arm / q_arm will be NaN", path
    )
    return None


def attach_contingency_value_block(
    arm: np.ndarray,
    stop_id: np.ndarray,
    contingency: pd.DataFrame | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-(event, arm) nominal p(R) and model value (Q) from the contingency.

    Single source for the per-arm value block shared by the NonLocal
    (``tables.py``) and ContFrag (``contfrag/tables.py``) event-arm builders
    (``docs/covariate_dictionary.md`` D2/D3). Both builders previously inlined
    this join, which let the join key silently diverge.

    Parameters
    ----------
    arm : np.ndarray, shape (n_rows,)
        Arm id (0-8) for each (event, arm) row.
    stop_id : np.ndarray, shape (n_rows,)
        Join key matching ``contingency["stop_id"]``. The caller supplies the
        key that matches the artifact: the NonLocal pipeline passes the event-arm
        table's own ``stop_id``; ContFrag passes the *plain*
        ``hash_stop_id(animal, day, epoch, stop_idx)`` because its own
        ``stop_id`` mixes in ``brain_area`` / ``event_source`` and so matches
        nothing in the (NonLocal-keyed) contingency.
    contingency : pd.DataFrame | None
        Per-stop artifact with ``stop_id`` + ``p_reward_port{0..5}`` +
        ``q_port{0..5}``. ``None`` (no artifact supplied) yields all-NaN. A
        supplied frame that is *missing* those columns raises ``ValueError``:
        ``qvals_table`` is the sole source, so a stale/partial artifact must
        fail loudly rather than silently blanking the whole value block.

    Returns
    -------
    p_reward_arm, q_arm : np.ndarray, each shape (n_rows,)
        Leaf arm ``j`` takes ``p_reward_port{j}`` / ``q_port{j}``; trunk arms
        (6-8) are not reward sites and take ``0.0`` where the stop is present in
        the contingency, ``NaN`` where absent — so a stop with no contingency row
        drops uniformly (all nine arms NaN) at fit time rather than contributing
        a spurious 0.0.

    Raises
    ------
    ValueError
        If ``contingency`` is supplied but lacks the ``p_reward_port{0..5}`` /
        ``q_port{0..5}`` columns (e.g. an old artifact from before the per-arm
        migration).
    """
    arm = np.asarray(arm)
    n_rows = len(arm)
    if contingency is None:
        return np.full(n_rows, np.nan), np.full(n_rows, np.nan)
    missing = {*_P_REWARD_PORT_COLS, *_Q_PORT_COLS} - set(contingency.columns)
    if missing:
        raise ValueError(
            "contingency is missing per-arm value columns "
            f"{sorted(missing)}; qvals_table is the sole source and must carry "
            "p_reward_port{0..5} / q_port{0..5} (regenerate with "
            "`python -m continuum_swr_replay.analysis.behavior.qvals_table --write`)"
        )

    n_unique = contingency["stop_id"].nunique()
    if n_unique != len(contingency):
        warnings.warn(
            f"contingency has {len(contingency) - n_unique} duplicate stop_id "
            "row(s); keeping the first occurrence per stop_id. If duplicate "
            "rows carry different port/q values this silently picks one.",
            stacklevel=2,
        )
    cont = contingency.drop_duplicates("stop_id")[
        ["stop_id", *_P_REWARD_PORT_COLS, *_Q_PORT_COLS]
    ]
    merged = pd.DataFrame({"stop_id": np.asarray(stop_id), "arm": arm}).merge(
        cont, on="stop_id", how="left"
    )
    port_mat = merged[_P_REWARD_PORT_COLS].to_numpy(dtype=float)
    q_mat = merged[_Q_PORT_COLS].to_numpy(dtype=float)
    leaf = arm < 6
    leaf_idx = np.clip(arm, 0, 5).astype(int)
    row = np.arange(n_rows)
    present = ~np.isnan(port_mat[:, 0])  # stop_id had a contingency row
    trunk_fill = np.where(present, 0.0, np.nan)
    p_reward_arm = np.where(leaf, port_mat[row, leaf_idx], trunk_fill)
    q_arm = np.where(leaf, q_mat[row, leaf_idx], trunk_fill)
    return p_reward_arm, q_arm


def attach_arm_uncertainty(
    arm: np.ndarray,
    stop_id: np.ndarray,
    contingency: pd.DataFrame | None,
) -> np.ndarray:
    """Per-arm beta-model belief uncertainty ``arm_uncertainty``.

    The variance counterpart of the beta value ``q_arm``: both come from the SAME
    Beta(alpha, beta) posterior (mean -> ``q_arm``, variance -> ``arm_uncertainty``),
    so a value+uncertainty model stays construct-coherent. Built
    with the SAME leaf->arm mapping as :func:`attach_contingency_value_block`,
    sourced from ``betadist_var{1..6}`` (the per-leaf Beta posterior variance
    ``alpha*beta / ((alpha+beta)^2 (alpha+beta+1))``). Leaf arm ``j`` takes
    ``betadist_var{j+1}``; trunk arms (6-8) are not reward sites and take ``0.0``
    where the stop is present in the contingency and ``NaN`` where absent, so a
    stop missing the value block drops uniformly across all nine arms at fit time
    (matching ``q_arm``).

    Parameters
    ----------
    arm : np.ndarray, shape (n_rows,)
        Arm id per row (0-5 leaf, 6-8 trunk).
    stop_id : np.ndarray, shape (n_rows,)
        Stop id per row, keyed into ``contingency``.
    contingency : pd.DataFrame or None
        ``qvals_table`` rows carrying ``betadist_var{1..6}``; ``None`` -> all NaN.

    Returns
    -------
    arm_uncertainty : np.ndarray, shape (n_rows,)
    """
    arm = np.asarray(arm)
    n_rows = len(arm)
    if contingency is None:
        return np.full(n_rows, np.nan)
    missing = set(_BETADIST_VAR_COLS) - set(contingency.columns)
    if missing:
        raise ValueError(
            "contingency is missing per-leaf beta-variance columns "
            f"{sorted(missing)}; regenerate qvals_table with `python -m "
            "continuum_swr_replay.analysis.behavior.qvals_table --write` "
            "(carries betadist_var{1..6})."
        )
    cont = contingency.drop_duplicates("stop_id")[["stop_id", *_BETADIST_VAR_COLS]]
    merged = pd.DataFrame({"stop_id": np.asarray(stop_id), "arm": arm}).merge(
        cont, on="stop_id", how="left"
    )
    var_mat = merged[_BETADIST_VAR_COLS].to_numpy(dtype=float)
    leaf = arm < 6
    leaf_idx = np.clip(arm, 0, 5).astype(int)
    row = np.arange(n_rows)
    present = ~np.isnan(var_mat[:, 0])  # stop_id had a value row
    trunk_fill = np.where(present, 0.0, np.nan)
    return np.where(leaf, var_mat[row, leaf_idx], trunk_fill)
