"""Behavioral covariate definitions for the replay rate / content / choice models.

Single home for the per-event / per-stop / per-arm behavioral covariates (reward
history, visit recency, model-free uncertainty) plus the fixed-alpha local
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
from continuum_swr_replay.table_io import read_table, table_exists

from ._helpers import (
    _trial_arrival_times,
    _trial_outcome_times,
    build_well_to_arm_map,
)
from .model_specs import (
    N_REWARD_LAGS,
    RECENCY_SPLINE_BOUNDS,
    RECENCY_SPLINE_KNOTS,
)

_SESSION_COLS = ["animal", "day", "epoch"]
_P_REWARD_PORT_COLS = [f"p_reward_port{i}" for i in range(6)]
_Q_PORT_COLS = [f"q_port{i}" for i in range(6)]
# Per-leaf beta-model posterior variance (qvals_table betadist_var{1..6}); the
# variance counterpart of _Q_PORT_COLS (mean). betadist_var{i} indexes leaf port
# i-1 (1-indexed columns), so leaf arm j -> _BETADIST_VAR_COLS[j] = betadist_var{j+1}.
_BETADIST_VAR_COLS = [f"betadist_var{i}" for i in range(1, 7)]

# Canonical paper-analysis settings. These covariates are computed from the
# complete completion-ordered trial stream before it is reduced to stop rows;
# they cannot be reconstructed correctly from a stop table with missing visits.
CANONICAL_SWITCH_WINDOW = 5
CANONICAL_LEAKY_Q_ALPHA = 0.3


UNCERTAINTY_COVARIATES = (
    "omission_streak",
    "recent_switch_rate",
    "wsls_deviation",
    "leaky_q_value_diff",
)


def center_visit_recency_within_event(
    recency: np.ndarray,
    never_visited: np.ndarray,
    *,
    log_scale: bool = False,
) -> np.ndarray:
    """Center leaf-arm recency within each event without using future events.

    Only *visited* reward-site arms participate in the mean. A never-visited
    arm carries the right-censoring fill (``n_completed + 1``), which is a
    property of the censoring rule rather than a measured recency; letting it
    into the mean would displace every other arm in the event by an arbitrary
    amount. In a session with one durably avoided well the fill can exceed the
    visited arms by two orders of magnitude, so the resulting coefficient would
    be estimated off the censoring rule instead of off recency.
    ``arm_never_visited`` carries the censored state on its own.

    Structural trunk values remain zero once history exists, and an event with
    no visited leaf arm remains all-NaN. Guarantees that appending later
    events cannot change an earlier row.

    Parameters
    ----------
    recency : np.ndarray, shape (n_events, len(ALL_ARMS))
        Per-event per-arm visit recency.
    never_visited : np.ndarray, shape (n_events, len(ALL_ARMS))
        1.0 where the arm has never been visited by this event, else 0.0.
        NaN is treated as not-censored, matching ``recency``'s own NaN.
    log_scale : bool
        Apply ``log1p`` to the raw recency *before* centering. Recency is a
        trial count spanning 1 to ~200, and it is deterministic on the two arms
        that matter -- exactly 1 on the current arm and exactly 2 on the
        previous arm -- so a linear term forces the current-versus-previous
        contrast to one unit while the data show far more. Within event, among
        the arms that are neither current nor previous, the relationship with
        replay share is monotone (median within-event r = +0.24, positive in
        70% of events), so a monotone transform is the right shape; a spline
        is not needed. Transforming before centering keeps the censoring rule
        below unchanged -- the mean is still taken over visited leaves only.

    Returns
    -------
    np.ndarray, shape (n_events, len(ALL_ARMS))
        Recency with visited leaf arms centered within each event.
    """
    values = np.asarray(recency, dtype=float)
    censored = np.asarray(never_visited, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(ALL_ARMS):
        raise ValueError(
            f"recency must have shape (n_events, len(ALL_ARMS)); got {values.shape}"
        )
    if censored.shape != values.shape:
        raise ValueError(
            "never_visited must match recency shape "
            f"{values.shape}; got {censored.shape}"
        )
    if log_scale:
        values = np.log1p(values)
    output = values.copy()
    arm_positions = {arm: index for index, arm in enumerate(ALL_ARMS)}
    leaf_positions = np.asarray([arm_positions[arm] for arm in WELL_ARMS])
    leaf_values = values[:, leaf_positions]
    visited = np.isfinite(leaf_values) & (censored[:, leaf_positions] != 1.0)
    counts = visited.sum(axis=1)
    totals = np.where(visited, leaf_values, 0.0).sum(axis=1)
    means = np.divide(
        totals,
        counts,
        out=np.full(len(values), np.nan),
        where=counts > 0,
    )
    output[:, leaf_positions] = leaf_values - means[:, None]
    return output


def spline_visit_recency_within_event(
    recency: np.ndarray,
    never_visited: np.ndarray,
    is_current: np.ndarray,
    is_prev: np.ndarray,
    arm: np.ndarray,
    event_id: np.ndarray,
) -> np.ndarray:
    """Event-centered natural-spline basis of recency, on its identified support.

    The basis is built on the raw trial-count scale (see the knot constants in
    ``model_specs``), so a coefficient reads per unit of the basis at knots that
    are themselves trial counts.

    All inputs are flat per-(event, arm) row arrays of equal length. Returns
    ``(n_rows, len(RECENCY_SPLINE_COLUMNS))``, zero outside the masked support so
    the columns carry no weight on arms whose recency is already determined.

    The within-event mean is taken over the masked rows only, mirroring
    :func:`center_visit_recency_within_event`: an arm outside the support must
    not displace the arms inside it.
    """
    from patsy import dmatrix

    values = np.asarray(recency, dtype=float)
    mask = (
        (np.asarray(arm) < len(WELL_ARMS))
        & (np.asarray(never_visited, dtype=float) == 0.0)
        & (np.asarray(is_current, dtype=float) == 0.0)
        & (np.asarray(is_prev, dtype=float) == 0.0)
        & np.isfinite(values)
    )
    basis = np.asarray(
        dmatrix(
            "cr(x, knots=k, lower_bound=lo, upper_bound=hi) - 1",
            {
                "x": np.where(np.isfinite(values), values, RECENCY_SPLINE_BOUNDS[0]),
                "k": list(RECENCY_SPLINE_KNOTS),
                "lo": RECENCY_SPLINE_BOUNDS[0],
                "hi": RECENCY_SPLINE_BOUNDS[1],
            },
            return_type="dataframe",
        )
    )[:, 1:]
    out = basis * mask[:, None]
    events = pd.Series(np.asarray(event_id))
    for column in range(out.shape[1]):
        masked = pd.Series(np.where(mask, out[:, column], np.nan))
        means = masked.groupby(events).transform("mean").fillna(0.0).to_numpy()
        out[:, column] = np.where(mask, out[:, column] - means, 0.0)
    return out


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
    recency, _ = visit_recency_and_never_visited_all_events(
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
    _, never_visited = visit_recency_and_never_visited_all_events(
        event_times, trials, well_to_arm
    )
    return never_visited


def visit_recency_and_never_visited_all_events(
    event_times: np.ndarray,
    trials: pd.DataFrame,
    well_to_arm: dict[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute time-local recency and novelty from one shared history pass.

    This is the batched source of truth for the two coupled visit-history
    covariates.  Both outputs use the same arrival-time cutoffs, arm mapping,
    and right-censoring rule, so callers that need both should use this helper
    rather than scanning the trial stream once per output.

    Returns
    -------
    recency, never_visited : tuple[np.ndarray, np.ndarray]
        Two float arrays with shape ``(n_events, len(ALL_ARMS))``.  Rows before
        any completed arrival are all-NaN; after history begins, trunk arms are
        structural zero and unvisited leaf arms carry the time-local censored
        recency plus a novelty indicator of one.
    """
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
    trials = trials.assign(_cutoff=_trial_arrival_times(trials))
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

    # Reward-bearing history advances at outcome availability, not destination
    # arrival. This prevents an event during the arrival-to-pump/poke-out gap
    # from seeing the current trial's eventual reward.
    trials = trials.assign(_cutoff=_trial_outcome_times(trials))
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

    # Visits are spatially known at arrival, but their reward values enter this
    # statistic only when the outcome is observable.
    trials = trials.assign(_cutoff=_trial_outcome_times(trials))
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
    a query time if its reward/omission availability time (see
    :func:`_trial_outcome_times`) is at or before that time.

    Parameters
    ----------
    query_times : np.ndarray, shape (n_query,)
        Times (s) at which to evaluate the recent reward rate — e.g.
        immobility-bout start times for the stop-level rate model.
    trials : pd.DataFrame
        Must carry ``is_reward`` plus the columns ``_trial_outcome_times``
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

    # Sort by outcome availability so ``searchsorted`` is valid.
    cutoff_times = np.sort(_trial_outcome_times(trials))
    is_reward = (
        trials.assign(_cutoff=_trial_outcome_times(trials))
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
        Must carry ``is_reward`` plus what :func:`_trial_outcome_times` needs.
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
    cutoff = _trial_outcome_times(trials)
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
    """Pre-update experienced-value margin over an ordered visit sequence.

    For each arm ``a`` a leaky reward-rate is updated once per input row
    ``Q_a <- (1 - q_alpha) Q_a + q_alpha * reward_outcome`` (init 0.5 for all
    arms). The returned value is the best-minus-second-best gap evaluated
    **before** that row's update. Canonical paper tables call this on the full
    completion-ordered trial stream, never on the reduced stop sequence.

    Parameters
    ----------
    current_arm : np.ndarray, shape (n_visits,)
        Arm/well 0..5 visited at each completed visit. A ``NaN`` arm still
        yields the current pre-update Q-spread margin (a real value -- the
        choice margin does not depend on which arm is visited) but does not
        update any arm's ``Q`` (the unknown arm cannot be credited).
    reward_outcome : np.ndarray, shape (n_visits,)
        Binary reward (0/1) at each completed visit.
    q_alpha : float
        Leaky-integration learning rate.

    Returns
    -------
    np.ndarray, shape (n_visits,)
        Pre-update best-minus-second-best arm value at each visit.
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


def canonical_trial_uncertainty_history(
    completed_well: np.ndarray,
    reward_outcome: np.ndarray,
    is_after_switch: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute canonical uncertainty history over every completed trial.

    The switch rate at trial ``j`` is the rolling mean through ``j`` because
    that trip's switch outcome is known at its completion. The leaky-Q margin
    at ``j`` is evaluated immediately before trial ``j`` updates its visited
    well's value; every completed trial then updates exactly once, whether or
    not that visit later contributes an immobility-bout row.
    """
    well = np.asarray(completed_well, dtype=float)
    reward = np.asarray(reward_outcome, dtype=float)
    switched = np.asarray(is_after_switch, dtype=float)
    if well.ndim != 1 or reward.ndim != 1 or switched.ndim != 1:
        raise ValueError("completed-trial uncertainty inputs must be one-dimensional")
    if not (len(well) == len(reward) == len(switched)):
        raise ValueError("completed-trial uncertainty inputs must have equal lengths")
    if not np.isfinite(well).all() or not np.equal(well, np.floor(well)).all():
        raise ValueError("completed well IDs must be finite integers")
    if not np.isin(well, tuple(WELL_PATCH_MAPPING)).all():
        raise ValueError(
            "completed well IDs must be canonical task wells "
            f"{tuple(WELL_PATCH_MAPPING)}"
        )
    for name, values in (("reward_outcome", reward), ("is_after_switch", switched)):
        if not np.isfinite(values).all() or not np.isin(values, (0.0, 1.0)).all():
            raise ValueError(f"{name} must contain finite binary values")

    recent_switch_rate = (
        pd.Series(switched)
        .rolling(CANONICAL_SWITCH_WINDOW, min_periods=1)
        .mean()
        .to_numpy()
    )
    leaky_q_value_diff = _leaky_q_value_diff(
        well,
        reward,
        CANONICAL_LEAKY_Q_ALPHA,
    )
    return recent_switch_rate, leaky_q_value_diff


def _completed_visit_context(stops: pd.DataFrame) -> pd.DataFrame:
    """Validate and collapse stop rows to explicitly identified visits.

    ``completed_trial_id`` is the zero-based completion-order ID regenerated
    by :func:`add_trial_context`. Stop tables need not contain every completed
    trial: a visit with no immobility bout has no stop row, so ID gaps are
    valid. IDs must nevertheless be nondecreasing in stop order, nullable rows
    may occur only before the first observed ID (true carry-in), and every row
    sharing an ID must agree on its completed-trial arm, outcome, and switch
    status.

    The returned sequence contains only visits represented in ``stops``. It is
    not a reconstruction of completed visits with no stop row; computing such
    a full sequence would require the original trials table, which this
    stop-table-only helper deliberately does not accept.
    """
    required = (
        "completed_trial_id",
        "completed_trial_arm",
        "reward_outcome",
        "is_after_switch",
        "recent_switch_rate",
        "leaky_q_value_diff",
    )
    missing = [column for column in required if column not in stops]
    if missing:
        raise KeyError(
            "visit-grain uncertainty covariates require explicit completed "
            f"trial context columns {missing}; regenerate stop_table.pkl"
        )
    if len(stops) == 0:
        return pd.DataFrame(columns=required)

    raw_ids = stops["completed_trial_id"]
    try:
        visit_ids = pd.array(raw_ids, dtype="Int64")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "completed_trial_id must contain nullable integral completion-order IDs"
        ) from exc
    finite_ids = visit_ids[~visit_ids.isna()].to_numpy(dtype=np.int64)
    if len(finite_ids) and (finite_ids < 0).any():
        raise ValueError("completed_trial_id must be nonnegative")
    if len(finite_ids) and np.any(np.diff(finite_ids) < 0):
        raise ValueError("completed_trial_id must be nondecreasing in stop order")
    if len(finite_ids):
        first_observed = int(np.flatnonzero(~visit_ids.isna())[0])
        if visit_ids[first_observed:].isna().any():
            raise ValueError(
                "nullable completed_trial_id carry-in rows may occur only before "
                "the first observed completed trial"
            )

    carry_in = visit_ids.isna()
    arrival_history_columns = (
        "completed_trial_arm",
        "is_after_switch",
        "recent_switch_rate",
        "leaky_q_value_diff",
    )
    history_columns = (*arrival_history_columns, "reward_outcome")
    if stops.loc[carry_in, list(history_columns)].notna().to_numpy().any():
        raise ValueError(
            "carry-in rows without a completed_trial_id must have unknown "
            "completed-trial context and canonical history covariates"
        )

    working = stops.loc[~carry_in, list(required)].copy()
    working["completed_trial_id"] = finite_ids
    if working.empty:
        return working

    for column in history_columns:
        values = pd.to_numeric(working[column], errors="coerce")
        raw_finite = working[column].notna()
        if (raw_finite & values.isna()).any():
            raise ValueError(f"{column} must be numeric within completed trials")
        working[column] = values.astype(float)
        if column == "reward_outcome":
            # An arrived trip can legitimately have pre-outcome bouts (NaN)
            # followed by post-outcome bouts (one binary value). Its realized
            # value may never change, and availability may never regress from
            # finite back to missing as stop time advances.
            grouped_outcome = working.groupby("completed_trial_id", sort=False)[column]
            inconsistent = grouped_outcome.apply(
                lambda group: group.dropna().nunique() > 1
            )
            regressed = grouped_outcome.apply(
                lambda group: bool((group.notna().cummax() & group.isna()).any())
            )
            inconsistent = inconsistent | regressed
        else:
            inconsistent = working.groupby("completed_trial_id", sort=False)[
                column
            ].apply(lambda group: group.nunique(dropna=False) != 1)
        if inconsistent.any():
            bad = tuple(inconsistent[inconsistent].index[:10])
            raise ValueError(
                f"{column} must be constant within each completed_trial_id; "
                f"invalid ID(s): {bad}"
            )

    visits = working.drop_duplicates("completed_trial_id", keep="first").reset_index(
        drop=True
    )
    for column in arrival_history_columns:
        if not np.isfinite(visits[column].to_numpy(dtype=float)).all():
            raise ValueError(
                f"{column} must be finite for every observed completed_trial_id"
            )
    arm = visits["completed_trial_arm"].to_numpy(dtype=float)
    if not np.equal(arm, np.floor(arm)).all() or not np.isin(arm, WELL_ARMS).all():
        raise ValueError(
            f"completed_trial_arm must be a canonical leaf arm {tuple(WELL_ARMS)}"
        )
    for column in ("is_after_switch",):
        values = visits[column].to_numpy(dtype=float)
        if not np.isin(values, (0.0, 1.0)).all():
            raise ValueError(f"{column} must be binary within completed trials")
    reward_values = working["reward_outcome"].dropna().to_numpy(dtype=float)
    if not np.isin(reward_values, (0.0, 1.0)).all():
        raise ValueError("reward_outcome must be binary whenever it is known")
    for column in ("recent_switch_rate", "leaky_q_value_diff"):
        values = visits[column].to_numpy(dtype=float)
        if ((values < 0.0) | (values > 1.0)).any():
            raise ValueError(f"{column} must lie in [0, 1]")
    return visits


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
        streak (counts the omissions seen before it). ``NaN`` when the most
        recent outcome itself is unresolved -- with no known outcome to start
        from, a count of leading omissions is undefined, and returning 0 would
        be indistinguishable from "the last trial was rewarded".
    """
    reward_lag_matrix = np.asarray(reward_lag_matrix, dtype=float)
    if reward_lag_matrix.ndim != 2:
        raise ValueError("reward_lag_matrix must be two-dimensional")
    finite = reward_lag_matrix[np.isfinite(reward_lag_matrix)]
    if not np.isin(finite, (0.0, 1.0)).all():
        raise ValueError("reward lag values must be binary 0/1 or NaN")
    n_stops = reward_lag_matrix.shape[0]
    out = np.zeros(n_stops)
    for i in range(n_stops):
        streak = 0
        for val in reward_lag_matrix[i]:
            if np.isnan(val) or val == 1.0:
                break
            streak += 1
        out[i] = streak
    # No resolved most-recent outcome -> the streak has nothing to count from.
    # Without this the pre-first-outcome stops of every session land in the
    # streak==0 bucket alongside genuinely just-rewarded stops.
    if n_stops:
        out[~np.isfinite(reward_lag_matrix[:, 0])] = np.nan
    return out


def add_uncertainty_covariates(
    stop_table: pd.DataFrame,
    *,
    switch_window: int = 5,
    q_alpha: float = 0.3,
) -> pd.DataFrame:
    """Complete the four model-free behavioral-uncertainty covariates per stop.

    Only two columns are *added* -- ``omission_streak`` and ``wsls_deviation``,
    the stop-local summaries. The two visit-history covariates
    (``recent_switch_rate``, ``leaky_q_value_diff``) must already be present:
    they are computed from the full completion-ordered trial stream and are
    here validated and passed through unchanged, never recomputed. A reduced
    stop table cannot reconstruct history across completed visits that contain
    no immobility bout, so missing canonical columns fail loudly.
    The input is not mutated; the returned frame keeps its index and row count.

    The four covariates:

    - ``omission_streak``: leading consecutive omissions = count of leading 0s
      in ``reward_lag1, reward_lag2, ...`` (stop at the first 1; a ``NaN`` lag
      terminates the streak).
    - ``recent_switch_rate``: canonical rolling mean of ``is_after_switch`` over
      the last five completed visits in the full trial stream, through the
      identified completed trial. It is validated and retained, not recomputed.
    - ``wsls_deviation``: 1.0 (deviated from win-stay/lose-shift) unless the stop
      is ``(rewarded & is_stay)`` or ``(omitted & ~is_stay)``, in which case
      0.0. Uses ``completed_trial_reward``, the trip's *realized* outcome, not
      the bout-start-gated ``is_after_reward`` -- an at-well bout starts at
      arrival, before the outcome exists, so the gated form was NaN at every
      at-well stop. ``NaN`` where no trip has completed or the outbound choice
      is unresolved -- such rows are dropped at fit time rather than counted as
      a deviation. Descriptive only: it embeds the eventual choice.
    - ``leaky_q_value_diff``: canonical best-minus-second-best leaky-Q margin
      (init 0.5, alpha 0.3) immediately before the identified completed trial's
      update. It is validated and retained, not recomputed.

    Noncanonical ``q_alpha`` or ``switch_window`` requests fail: sensitivity
    values require recomputation from the complete trials table, not this
    reduced stop table.

    Parameters
    ----------
    stop_table : pd.DataFrame
        Per-stop table. Must carry ``animal``, ``day``, ``epoch``, ``stop_idx``,
        ``completed_trial_id``, ``completed_trial_arm``, ``is_after_switch``,
        ``is_stay``, ``completed_trial_reward``,
        ``reward_outcome``, ``recent_switch_rate``, ``leaky_q_value_diff``, and
        ``reward_lag1..`` columns.
    switch_window : int
        Must equal the canonical value 5.
    q_alpha : float
        Must equal the canonical value 0.3.

    Returns
    -------
    pd.DataFrame
        Copy of ``stop_table`` with canonical upstream history validated and
        the stop-local ``omission_streak`` and ``wsls_deviation`` columns added.
    """
    if switch_window != CANONICAL_SWITCH_WINDOW:
        raise ValueError(
            "noncanonical switch_window requires recomputation from the full "
            "trials table; a stop table cannot recover visits without stops"
        )
    if not np.isfinite(q_alpha) or q_alpha != CANONICAL_LEAKY_Q_ALPHA:
        raise ValueError(
            "noncanonical q_alpha requires recomputation from the full trials "
            "table; a stop table cannot recover visits without stops"
        )

    required = (
        *_SESSION_COLS,
        "stop_idx",
        "completed_trial_id",
        "completed_trial_arm",
        "is_after_switch",
        # wsls_deviation reads the *realized* outcome, not the gated flags --
        # validate the column actually consumed. The gated trio is retained
        # here only because a stop table lacking them predates the outcome
        # clock entirely and should fail this check too.
        "completed_trial_reward",
        "is_after_reward",
        "is_after_omission",
        "reward_outcome",
        "is_stay",
        "recent_switch_rate",
        "leaky_q_value_diff",
    )
    missing = [column for column in required if column not in stop_table]
    if missing:
        raise KeyError(
            "uncertainty covariates require canonical stop-table columns "
            f"{missing}; regenerate stop_table.pkl"
        )

    out = stop_table.copy()
    expected_lag_cols = [f"reward_lag{j}" for j in range(1, N_REWARD_LAGS + 1)]
    observed_lag_cols = [c for c in out.columns if c.startswith("reward_lag")]
    duplicate_lags = sorted(
        {column for column in observed_lag_cols if observed_lag_cols.count(column) > 1}
    )
    if set(observed_lag_cols) != set(expected_lag_cols) or duplicate_lags:
        missing_lags = [c for c in expected_lag_cols if c not in observed_lag_cols]
        unexpected_lags = [c for c in observed_lag_cols if c not in expected_lag_cols]
        raise KeyError(
            "omission_streak requires the exact contiguous canonical reward-lag "
            f"schema {expected_lag_cols}; missing={missing_lags}, "
            f"unexpected={unexpected_lags}, duplicates={duplicate_lags}"
        )
    raw_lags = out[expected_lag_cols]
    numeric_lags = raw_lags.apply(pd.to_numeric, errors="coerce")
    if (raw_lags.notna() & numeric_lags.isna()).to_numpy().any():
        raise ValueError("reward lag values must be numeric binary 0/1 or NaN")
    finite_lags = numeric_lags.to_numpy(dtype=float)
    finite_values = finite_lags[np.isfinite(finite_lags)]
    if not np.isin(finite_values, (0.0, 1.0)).all():
        raise ValueError("reward lag values must be binary 0/1 or NaN")
    out.loc[:, expected_lag_cols] = numeric_lags

    omission = pd.Series(np.nan, index=out.index, dtype=float)
    wsls = pd.Series(np.nan, index=out.index, dtype=float)

    for _, idx in out.groupby(_SESSION_COLS, sort=False).groups.items():
        g = out.loc[idx].sort_values("stop_idx")
        pos = g.index
        _completed_visit_context(g)

        omission.loc[pos] = _omission_streak(g[expected_lag_cols].to_numpy(dtype=float))

        # Read the trip's *realized* outcome, not the bout-start-gated flags.
        # A bout at a well begins at arrival, which precedes reward onset (and
        # poke-out on an omission), so ``is_after_reward`` / ``is_after_omission``
        # are missing at essentially every at-well stop and gating here left the
        # covariate NaN wherever it means anything. Win-stay/lose-shift is by
        # definition about the outcome the animal just experienced, paired with
        # the choice it then makes -- and that choice, ``is_stay``, is future
        # information regardless, so this construct is descriptive-only however
        # the outcome is clocked. See covariate_timing.FUTURE_REFERENCING.
        outcome = g["completed_trial_reward"].to_numpy(dtype=float)
        stay = g["is_stay"].to_numpy(dtype=float)
        win_stay = (outcome == 1.0) & (stay == 1.0)
        lose_shift = (outcome == 0.0) & (stay == 0.0)
        wsls_vals = np.where(win_stay | lose_shift, 0.0, 1.0)
        # No completed trip (e.g. before the session's first arrival) or no
        # resolved outbound choice -> NaN, so the row drops at fit time instead
        # of being counted as a deviation.
        undefined = ~np.isfinite(outcome) | ~np.isfinite(stay)
        wsls_vals[undefined] = np.nan
        wsls.loc[pos] = wsls_vals

    out["omission_streak"] = omission
    out["wsls_deviation"] = wsls
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
    if table_exists(path):
        cont = read_table(path)
        logging.info(
            "loaded qvals_table (%d stop rows) for p_reward_arm / q_arm", len(cont)
        )
        return cont
    logging.warning("no qvals_table at %s — p_reward_arm / q_arm will be NaN", path)
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
