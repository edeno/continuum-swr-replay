"""Reward-history covariates: rolling reward counts and recent reward rates over an arm's last visits, plus stop-level reward histories."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.replay._helpers import (
    build_well_to_arm_map,
    raised_cosine_history_basis,
    # Re-exported: the projection is a pure array primitive shared with the
    # per-bin design builder, which must not import this package.
    reward_basis_from_lags,
)
from continuum_swr_replay.analysis.replay.model_specs import (
    REWARD_HISTORY_N_BASIS,
    REWARD_HISTORY_N_LAGS,
)
from continuum_swr_replay.data_loaders.constants import (
    ALL_ARMS,
)

from ._common import _prepare_port_experiences


def _observed_port_outcomes(experiences: pd.DataFrame) -> pd.DataFrame:
    """Select the outcome-bearing subset without changing experience order."""
    return experiences.loc[
        experiences["outcome_observed"].to_numpy(dtype=bool)
    ].reset_index(drop=True)


def rolling_reward_count_all_events(
    event_times: np.ndarray,
    port_experiences: pd.DataFrame,
    well_to_arm: dict[int, int],
    window: int = 10,
) -> np.ndarray:
    """Rewards per arm over the last ``window`` known port outcomes.

    A **rolling k=10 window**, not a running total: the count is
    ``cumsum[t] - cumsum[t - window]``, so a reward more than ``window`` trials
    back contributes nothing. It fills the persisted column
    ``reward_count_candidate_arm_last10_known_trials`` (shipped in
    ``event_arm_table.pkl`` and consumed by the choice-prediction model); the column name is
    frozen even though the function name now describes the window.

    ``NaN`` where no trial has completed yet -- unknown history, not zero
    reward.

    Returns ``(n_events, n_arms)`` array.
    """
    n_arms = len(ALL_ARMS)
    arm_idx = {a: j for j, a in enumerate(ALL_ARMS)}
    n_events = len(event_times)
    experiences = _prepare_port_experiences(port_experiences)
    experiences = _observed_port_outcomes(experiences)
    n_experiences = len(experiences)

    if n_experiences == 0:
        # No observed port experience → unknown history, not zero reward
        # (matches the sibling helpers).
        return np.full((n_events, n_arms), np.nan)

    # Reward-bearing history advances at outcome availability, not destination
    # arrival. This prevents an event during the arrival-to-feedback gap
    # from seeing the current trial's eventual reward.
    experiences = experiences.sort_values("outcome_known_time", kind="stable")
    cutoff_times = experiences["outcome_known_time"].to_numpy(dtype=float)
    ports = experiences["port"].to_numpy(dtype=np.int64)
    is_reward = experiences["is_reward"].to_numpy(dtype=bool)

    # Build per-arm cumulative reward arrays over port-experience order.
    reward_per_arm = np.zeros((n_experiences, n_arms), dtype=float)
    for ti in range(n_experiences):
        if not is_reward[ti]:
            continue
        arm_id = well_to_arm.get(int(ports[ti]))
        if arm_id is not None and arm_id in arm_idx:
            reward_per_arm[ti, arm_idx[arm_id]] = 1.0

    # cumsum with a leading zero row for easy slicing
    cum = np.zeros((n_experiences + 1, n_arms), dtype=float)
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
    port_experiences: pd.DataFrame,
    well_to_arm: dict[int, int],
    k: int = 10,
) -> np.ndarray:
    """Fraction of last ``k`` visits to each arm that were rewarded, for all events.

    The default ``k=10`` matches the canonical covariate
    ``reward_fraction_candidate_arm_last10_visits`` and the sibling helpers; the
    production caller passes ``k=10`` explicitly.

    Returns ``(n_events, n_arms)`` array.

    Trunk arms (graph edges that touch no well — never a trial
    destination, so never a reward site) are filled with ``0.0``
    rather than ``NaN``: their reward rate is structurally zero, not
    unknown. This matters because the conditional-Poisson fitter
    drops an entire event if *any* of its nine arm rows is NaN, and
    every event has three trunk-arm rows — leaving them NaN would
    annihilate the M3+ event set.

    A leaf arm not yet visited in a session *with* prior port experiences is filled
    ``0.0`` (no rewards observed on it yet), retaining the novel-arm /
    early-session regime. Only rows with no known port outcome stay ``NaN``. Models that
    use recency also carry ``arm_never_visited``, separating this censored
    state from an observed zero reward rate.
    """
    n_arms = len(ALL_ARMS)
    n_events = len(event_times)
    experiences = _prepare_port_experiences(port_experiences)
    experiences = _observed_port_outcomes(experiences)
    n_experiences = len(experiences)
    # Arms reachable as a trial destination = leaf arms (values of the
    # well→arm map). Anything else is a trunk arm.
    reachable_arms = set(well_to_arm.values())

    if n_experiences == 0:
        return np.full((n_events, n_arms), np.nan)

    # Visits are spatially known at arrival, but their reward values enter this
    # statistic only when the outcome is observable.
    experiences = experiences.sort_values("outcome_known_time", kind="stable")
    cutoff_times = experiences["outcome_known_time"].to_numpy(dtype=float)
    ports = experiences["port"].to_numpy(dtype=np.int64)
    is_reward = experiences["is_reward"].to_numpy(dtype=float)

    # Map each port experience to its leaf arm once (-1 for a non-leaf port).
    arm_of_experience = np.array(
        [well_to_arm.get(int(p), -1) for p in ports], dtype=np.int64
    )

    cutoffs = np.searchsorted(cutoff_times, event_times, side="right")

    # Events with no physically known port outcome -> NaN.
    no_past = cutoffs == 0
    out = np.full((n_events, n_arms), np.nan)

    for j in range(n_arms):
        # Trunk arm: reward rate is structurally 0 wherever there is
        # any trial history. ``no_past`` rows stay NaN.
        if ALL_ARMS[j] not in reachable_arms:
            out[~no_past, j] = 0.0
            continue
        visits = np.flatnonzero(arm_of_experience == ALL_ARMS[j])
        if visits.size == 0:
            # ENCODE: leaf never visited, but there ARE prior port experiences ->
            # 0.0 (no rewards observed yet), not NaN. no_past rows stay NaN.
            out[~no_past, j] = 0.0
            continue
        # n_vis_before[i] = number of visits to this arm before cutoff. The last
        # ``k`` of those are ``rewards[start:nv]``; its mean is the window sum
        # over the window count, computed from a cumsum with a leading zero so
        # 0/1 rewards give the mean exactly (integer sums are exact in float).
        rewards = is_reward[visits]
        cum = np.concatenate(([0.0], np.cumsum(rewards)))
        n_vis_before = np.searchsorted(visits, cutoffs, side="left")
        start = np.maximum(n_vis_before - k, 0)
        count = n_vis_before - start
        with np.errstate(invalid="ignore", divide="ignore"):
            rate = (cum[n_vis_before] - cum[start]) / count
        # ENCODE: leaf not yet visited BEFORE this event -> 0.0, not a dropped
        # NaN; ``no_past`` events stay NaN (they precede all known outcomes).
        col = np.where(n_vis_before == 0, 0.0, rate)
        col[no_past] = np.nan
        out[:, j] = col

    return out


# ── Stop-level known-outcome histories ──


def recent_trial_reward_rate(
    query_times: np.ndarray,
    port_experiences: pd.DataFrame,
    k: int = 10,
) -> np.ndarray:
    """Rewarded fraction of the most recent ``k`` known port experiences.

    The arm-agnostic counterpart of :func:`recent_reward_rate_all_events`:
    the denominator is all port experiences, not visits to one arm. An
    experience is "before"
    a query time if its reward/omission availability time (see
    ``outcome_known_time``) is at or before that time.

    Parameters
    ----------
    query_times : np.ndarray, shape (n_query,)
        Times (s) at which to evaluate the recent reward rate — e.g.
        immobility-bout start times for the stop-level rate model.
    port_experiences : pd.DataFrame
        Canonical experience stream with physical outcome clocks.
    k : int
        Window length in port experiences.

    Returns
    -------
    np.ndarray, shape (n_query,)
        Rewarded fraction of the last ``k`` port experiences; ``NaN`` for
        query times that precede every outcome-availability cutoff.
    """
    query_times = np.asarray(query_times, dtype=float)
    experiences = _prepare_port_experiences(port_experiences)
    experiences = _observed_port_outcomes(experiences)
    if len(experiences) == 0:
        return np.full(query_times.size, np.nan)

    experiences = experiences.sort_values("outcome_known_time", kind="stable")
    cutoff_times = experiences["outcome_known_time"].to_numpy(dtype=float)
    is_reward = experiences["is_reward"].to_numpy(dtype=float)
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
    port_experiences: pd.DataFrame,
    k: int = 10,
) -> np.ndarray:
    """Reward outcome of each of the last ``k`` known port experiences.

    The per-lag companion to :func:`recent_trial_reward_rate` (which averages
    the last ``k``): same trial-cutoff mapping, but returns the individual
    outcomes so a rate model can estimate the reward *kernel* across port
    experiences — ``known_reward_outcome_lag_{j}`` is the j-th most recent
    physically known port outcome. ``lag1`` is the most recent known outcome;
    it may be the current destination or the preceding experience depending on
    whether feedback was available at stop start.

    Parameters
    ----------
    query_times : np.ndarray, shape (n_query,)
        Times (s) at which to evaluate — immobility-bout start times.
    port_experiences : pd.DataFrame
        Canonical experience stream with physical outcome clocks.
    k : int
        Number of port-experience lags.

    Returns
    -------
    np.ndarray, shape (n_query, k)
        Column ``j-1`` = reward (0/1) of the j-th most recent known port
        outcome; ``NaN`` where fewer than ``j`` outcomes precede the query.
    """
    query_times = np.asarray(query_times, dtype=float)
    out = np.full((query_times.size, k), np.nan)
    experiences = _prepare_port_experiences(port_experiences)
    experiences = _observed_port_outcomes(experiences)
    if len(experiences) == 0:
        return out
    cutoff = experiences["outcome_known_time"].to_numpy(dtype=float)
    order = np.argsort(cutoff)
    cutoff_sorted = cutoff[order]
    is_reward = experiences["is_reward"].to_numpy(dtype=float)[order]
    counts = np.searchsorted(cutoff_sorted, query_times, side="right")
    for j in range(1, k + 1):
        idx = counts - j
        valid = idx >= 0
        out[valid, j - 1] = is_reward[idx[valid]]
    return out


def reward_history_basis_features(
    query_times: np.ndarray,
    port_experiences: pd.DataFrame,
    *,
    n_lags: int = REWARD_HISTORY_N_LAGS,
    n_basis: int = REWARD_HISTORY_N_BASIS,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth reward-history features over the last ``n_lags`` known outcomes.

    The regularized replacement for the ``n_lags`` collinear per-lag dummies of
    :func:`lagged_trial_rewards`: the same outcome clock and the same lag
    vector, projected onto log-spaced raised-cosine bumps. Each feature is a
    weighted reward fraction in ``[0, 1]``, so its coefficient is per unit rate.
    Bump 0 is the immediate component; later bumps average progressively older
    outcomes.

    Parameters
    ----------
    query_times : np.ndarray, shape (n_query,)
        Times (s) at which to evaluate -- immobility-bout start times.
    port_experiences : pd.DataFrame
        Canonical experience stream with physical outcome clocks.
    n_lags : int
        Number of port-experience lags spanned.
    n_basis : int
        Number of raised-cosine bumps.

    Returns
    -------
    features : np.ndarray, shape (n_query, n_basis)
        Normalized reward-history features. Entries whose bump has no known lag
        in its support are zero-filled (they are undefined, not reward-free).
    incomplete : np.ndarray, shape (n_query,)
        ``1.0`` where any feature was undefined and zero-filled -- fewer than
        two known outcomes, since the slowest bump carries no weight at lag 1 --
        else ``0.0``. A structural-missingness indicator, not a behavioral
        quantity: it enters rate models as a nuisance main effect so the
        zero-filled rows do not masquerade as reward-free history.
    """
    lags = lagged_trial_rewards(query_times, port_experiences, k=n_lags)
    raw = reward_basis_from_lags(lags, raised_cosine_history_basis(n_lags, n_basis))
    incomplete = np.isnan(raw).any(axis=1).astype(float)
    return np.nan_to_num(raw, nan=0.0), incomplete


def rolling_reward_count_per_arm(
    event_time: float,
    port_experiences: pd.DataFrame,
    track_graph: nx.Graph,
    window: int = 10,
) -> dict[int, float]:
    """Rewards at each arm over the last ``window`` known port outcomes."""
    w2a = build_well_to_arm_map(track_graph)
    row = rolling_reward_count_all_events(
        np.array([event_time]), port_experiences, w2a, window=window
    )
    return {arm: float(row[0, j]) for j, arm in enumerate(ALL_ARMS)}


def recent_reward_rate_per_arm(
    event_time: float,
    port_experiences: pd.DataFrame,
    track_graph: nx.Graph,
    k: int = 10,
) -> dict[int, float]:
    """Fraction of the last ``k`` visits to each arm that were rewarded."""
    w2a = build_well_to_arm_map(track_graph)
    row = recent_reward_rate_all_events(
        np.array([event_time]), port_experiences, w2a, k=k
    )
    return {arm: float(row[0, j]) for j, arm in enumerate(ALL_ARMS)}
