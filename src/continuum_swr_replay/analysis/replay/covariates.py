"""Behavioral covariate definitions for the replay rate / content / choice models.

Single home for the per-event / per-stop / per-arm behavioral covariates (reward
history, visit recency, leaky-Q value, model-free uncertainty) plus the
fitted-alpha leaky-Q learner. Implements the constructs specified in
``docs/covariate_dictionary.md``. These definitions are re-exported from
``_helpers`` (a shim) so existing call sites keep resolving.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from continuum_swr_replay.data_loaders.constants import TRACK_SEGMENT_TO_PATCH

from ._helpers import _trial_cutoff_times, build_well_to_arm_map

ALL_ARMS = sorted(TRACK_SEGMENT_TO_PATCH.keys())
_SESSION_COLS = ["animal", "day", "epoch"]


UNCERTAINTY_COVARIATES = (
    "omission_streak",
    "recent_switch_rate",
    "wsls_deviation",
    "leaky_q_value_diff",
)


def visit_recency_all_events(
    event_times: np.ndarray,
    trials: pd.DataFrame,
    well_to_arm: dict[int, int],
    n_default: int | None = None,
) -> np.ndarray:
    """Trials-ago since each arm was last visited, for all events.

    Returns ``(n_events, n_arms)`` array with the same semantics as
    :func:`visit_recency_per_arm` but computed in a single vectorized
    pass.
    """
    n_arms = len(ALL_ARMS)
    arm_idx = {a: j for j, a in enumerate(ALL_ARMS)}
    n_events = len(event_times)
    n_trials_total = len(trials)

    if n_default is None:
        n_default = n_trials_total + 1
    default_f = float(n_default)

    if n_trials_total == 0:
        return np.full((n_events, n_arms), np.nan)

    # ``np.searchsorted`` below assumes the cutoff times are non-decreasing.
    # Sort defensively so chronological-order assumptions hold for any
    # caller (Spyglass-loaded frames are not guaranteed to be sorted).
    # Use ``goal_arrival_time`` (with ``end_time`` fallback) as the
    # cutoff so this helper aligns with ``add_trial_context`` —
    # see ``_trial_cutoff_times`` for the rationale.
    trials = trials.assign(_cutoff=_trial_cutoff_times(trials))
    trials = trials.sort_values("_cutoff").reset_index(drop=True)
    cutoff_times = trials["_cutoff"].to_numpy()
    to_wells = trials["to_well"].to_numpy()

    # For each trial, record which arm was visited (to_well = destination).
    # Uses to_well only, consistent with cumulative_reward_all_events.
    # A "visit" is arriving at a well, not departing from one.
    arm_visit_indices: list[np.ndarray] = [np.empty(0, dtype=np.intp) for _ in ALL_ARMS]
    for ti in range(n_trials_total):
        w = to_wells[ti]
        if np.isnan(w):
            continue
        arm_id = well_to_arm.get(int(w))
        if arm_id is not None and arm_id in arm_idx:
            j = arm_idx[arm_id]
            arm_visit_indices[j] = np.append(arm_visit_indices[j], ti)

    # cutoff[i] = number of trials with cutoff_time <= event_times[i]
    cutoffs = np.searchsorted(cutoff_times, event_times, side="right")

    out = np.full((n_events, n_arms), default_f)

    # Events with no past trials -> NaN
    no_past = cutoffs == 0
    out[no_past] = np.nan

    for j in range(n_arms):
        visits = arm_visit_indices[j]
        if visits.size == 0:
            continue
        # For each event, find how many of this arm's visits are before cutoff
        n_vis = np.searchsorted(visits, cutoffs, side="left")
        # The last visit before cutoff is at visits[n_vis - 1]
        has_visit = (~no_past) & (n_vis > 0)
        last_visit_trial_idx = np.where(
            has_visit, visits[np.clip(n_vis - 1, 0, visits.size - 1)], -1
        )
        # trials_ago = cutoff - last_visit_trial_idx
        trials_ago = np.where(has_visit, cutoffs - last_visit_trial_idx, default_f)
        out[has_visit, j] = trials_ago[has_visit].astype(float)

    return out


def cumulative_reward_all_events(
    event_times: np.ndarray,
    trials: pd.DataFrame,
    well_to_arm: dict[int, int],
    window: int = 10,
) -> np.ndarray:
    """Reward count per arm over the last ``window`` trials, for all events.

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
    annihilate the M3+ event set. A leaf arm that simply has not been
    visited yet in a session *with* prior trials stays ``NaN`` (a
    genuine unknown); the no-trial-history rows stay ``NaN`` too,
    matching the sibling helpers' contract.
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
            continue
        # n_vis_before[i] = number of visits to this arm before cutoff
        n_vis_before = np.searchsorted(visits, cutoffs, side="left")
        for i in range(n_events):
            if no_past[i]:
                continue
            nv = n_vis_before[i]
            if nv == 0:
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
    - ``recent_switch_rate``: ``(1 - is_stay).shift(1).rolling(switch_window,
      min_periods=1).mean()`` within session (prior stops only -> no leakage of
      the current stop's choice).
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
        ``is_stay``, ``is_after_reward``, ``is_after_omission``,
        ``reward_outcome``, ``current_arm``, and ``reward_lag1..`` columns.
    switch_window : int
        Rolling window (in stops) for ``recent_switch_rate``.
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

        is_switch = 1.0 - g["is_stay"].to_numpy(dtype=float)
        switch_rate.loc[pos] = (
            pd.Series(is_switch, index=pos)
            .shift(1)
            .rolling(switch_window, min_periods=1)
            .mean()
            .to_numpy()
        )

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
    n_default: int | None = None,
) -> dict[int, float]:
    """Trials since each arm was last visited."""
    w2a = build_well_to_arm_map(track_graph)
    row = visit_recency_all_events(
        np.array([event_time]), trials, w2a, n_default=n_default
    )
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
    """
    qc = np.empty(len(arm))
    Q, prev = None, None
    for i in range(len(arm)):
        if grp[i] != prev:
            Q, prev = np.full(6, init), grp[i]
        a = arm[i]
        qc[i] = Q[a]
        Q[a] += alpha * (reward[i] - Q[a])
    return qc


def q_all_arm(
    arm: np.ndarray,
    reward: np.ndarray,
    grp: np.ndarray,
    alpha: float,
    init: float = 0.5,
) -> np.ndarray:
    """Per-leaf delta-rule Q **level of every arm**, pre-update, for all trials.

    The all-arm companion to :func:`_q_chosen`: same leaky delta-rule
    (``Q[a] += alpha*(reward - Q[a])``, ``Q`` reset to ``init`` at each new group),
    but snapshots the full 6-arm value vector *before* each trial's update rather
    than only the chosen arm. ``q_all_arm(...)[i, arm[i]]`` equals
    ``_q_chosen(...)[i]`` by construction, so ``q_current`` is the chosen-arm
    column of this array and ``q_arm`` is the per-arm value.

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
