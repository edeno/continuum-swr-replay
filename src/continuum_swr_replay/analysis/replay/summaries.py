"""Per-event replay summaries from decoder outputs.

Three response variables for the conditional Poisson content model, plus
their companion length measures:

==============================  =============================  ======================
Response                        Length measure                 Plan name
==============================  =============================  ======================
``arm_count_hard`` (int)        ``n_nonlocal_bins`` (int)      primary analysis
``arm_count_soft`` (float)      ``nonlocal_weight_sum``        robustness
``arm_posterior_mass`` (float)  ``posterior_mass_total``       robustness
==============================  =============================  ======================

Inputs are assumed to be **pre-filtered to in-event timebins** — the
caller selects rows where ``non_local_event > 0`` and passes flat
per-timebin arrays plus an ``event_id`` label column. This keeps the
module independent of any particular event-segmentation convention and
avoids threading a "filter-to-events" predicate through every function.

Every function returns ``(sorted_unique_event_ids, per_event_array)``.
Event IDs need not be contiguous; the caller is expected to align the
returned ordering with their own event table.

Design notes
------------
**Arm-posterior marginalization.** The conditional non-local posterior
lives on per-position place bins. ``non_local_detector`` interleaves
``edge_id == -1`` junction placeholder bins between real arms; those
bins carry no physical position and must not be counted toward any arm.
``posterior_to_arm_marginal`` sums posterior mass across real bins per
arm, zeroing junction bins, then renormalizes each timebin's arm
distribution to sum to 1 over real arms. This matches the
nearest-neighbor-over-real-bins convention used by
``spatial_mapping.map_estimate_to_arm`` for MAP collapse, and makes
``posterior_mass_total_e == nonlocal_weight_sum_e`` for every event by
construction.

The analysis plan's Section 1.2 shows ``posterior_mass_total_e ≤
nonlocal_weight_sum_e`` with a strict ≤, reflecting the alternative
convention where junction mass is "lost" rather than redistributed. We
deliberately chose renormalization here for internal consistency with
the MAP collapse; ``posterior_mass_total`` is therefore identically
equal to ``nonlocal_weight_sum`` and is retained as a separate symbol
only so callers can address "the offset for the ``arm_posterior_mass``
response" by name.

**Vectorization.** All summary accumulators use ``np.add.at`` or
``np.bincount`` over a dense (event × arm) target, so cost is O(T) in
total timebins regardless of event count. No Python loops over events.
"""

from __future__ import annotations

import numpy as np

# ── Posterior collapse to arms ──


def posterior_to_arm_marginal(
    conditional_posterior: np.ndarray,
    edge_id: np.ndarray,
    n_arms: int | None = None,
) -> np.ndarray:
    """Collapse per-position-bin posterior to per-arm posterior.

    Junction bins (``edge_id == -1``) contribute zero mass; each
    timebin's arm distribution is renormalized to sum to 1 over real
    arms.

    Parameters
    ----------
    conditional_posterior : np.ndarray
        Shape ``(n_timebins, n_position_bins)``. Each row is a proper
        probability distribution over place bins conditioned on the
        non-local state.
    edge_id : np.ndarray
        Shape ``(n_position_bins,)``. Integer edge label per place bin;
        ``-1`` marks junction / placeholder bins (not a real arm).
    n_arms : int, optional
        Number of real arms. If ``None``, inferred as ``edge_id.max() + 1``.

    Returns
    -------
    np.ndarray
        Shape ``(n_timebins, n_arms)``; each row sums to 1 up to
        floating-point tolerance for timebins with any real-arm mass.
        Timebins whose posterior is entirely on junction bins (which
        should never happen given the canonical handling — see CLAUDE.md)
        return a uniform ``1/n_arms`` row.
    """
    conditional_posterior = np.asarray(conditional_posterior, dtype=float)
    edge_id = np.asarray(edge_id, dtype=int)

    if conditional_posterior.shape[1] != edge_id.shape[0]:
        raise ValueError(
            f"posterior shape {conditional_posterior.shape} does not match "
            f"edge_id length {edge_id.shape[0]}"
        )

    is_real = edge_id >= 0
    if not is_real.any():
        raise ValueError(
            "No real arm bins found — every edge_id is -1 (junction). "
            "Check that the environment's place_bin_centers_nodes_df_ "
            "contains at least one real edge."
        )
    if n_arms is None:
        n_arms = int(edge_id[is_real].max()) + 1
    if n_arms <= 0:
        raise ValueError(f"n_arms must be positive; got {n_arms}")

    n_timebins = conditional_posterior.shape[0]
    arm_posterior = np.zeros((n_timebins, n_arms), dtype=float)
    real_edges = edge_id[is_real]
    real_mass = conditional_posterior[:, is_real]  # (T, n_real_bins)

    # Explicit per-arm scatter. Iterating over arms (not timebins) keeps
    # this O(n_arms · T); n_arms = 9 in the bandit task so the cost is
    # negligible. Prefer this to ``np.add.at(arm_posterior.T, ...)`` —
    # that form is correct but relies on transposed-view mutation
    # semantics that are opaque to read.
    for arm_idx in range(n_arms):
        bins_in_arm = real_edges == arm_idx
        if bins_in_arm.any():
            arm_posterior[:, arm_idx] = real_mass[:, bins_in_arm].sum(axis=1)

    row_sum = arm_posterior.sum(axis=1, keepdims=True)
    zero_rows = row_sum.squeeze(1) == 0
    # Avoid divide-by-zero for entirely-junction rows; leave as zeros, fix below.
    safe_sum = np.where(row_sum == 0, 1.0, row_sum)
    arm_posterior = arm_posterior / safe_sum
    if zero_rows.any():
        arm_posterior[zero_rows, :] = 1.0 / n_arms

    return arm_posterior


def arm_map_from_posterior(arm_posterior: np.ndarray) -> np.ndarray:
    """Per-timebin argmax arm from arm-marginal posterior.

    Ties broken by lowest arm index via ``np.argmax``.
    """
    return np.argmax(np.asarray(arm_posterior), axis=1).astype(np.int64)


# ── Event-level aggregation helper ──


def _group_by_event(
    event_id: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (unique_sorted_ids, inverse_indices_into_unique, n_events).

    Wrap ``np.unique(return_inverse=True)`` once; reused by every
    aggregator to avoid recomputing the group mapping.
    """
    unique, inverse = np.unique(np.asarray(event_id), return_inverse=True)
    return unique, inverse.astype(np.int64), unique.shape[0]


# ── Per-event length measures ──


def n_nonlocal_bins(
    nonlocal_prob: np.ndarray,
    event_id: np.ndarray,
    *,
    threshold: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """Hard count of above-threshold timebins per event.

    Parameters
    ----------
    nonlocal_prob : np.ndarray, shape (T,)
        Smoothed non-local state posterior ``w_t`` per in-event timebin.
    event_id : np.ndarray, shape (T,)
        Event label per timebin.
    threshold : float
        ``w_t > threshold`` timebins contribute 1 to the count.

    Returns
    -------
    event_ids : np.ndarray, shape (n_events,)
    n_bins : np.ndarray, shape (n_events,), int
    """
    nonlocal_prob = np.asarray(nonlocal_prob, dtype=float)
    unique, inv, n_events = _group_by_event(event_id)
    above = (nonlocal_prob > threshold).astype(np.int64)
    counts = np.bincount(inv, weights=above, minlength=n_events).astype(np.int64)
    return unique, counts


def nonlocal_weight_sum(
    nonlocal_prob: np.ndarray,
    event_id: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Soft per-event sum of ``w_t``.

    Returns ``(event_ids, sums_float)``.
    """
    nonlocal_prob = np.asarray(nonlocal_prob, dtype=float)
    unique, inv, n_events = _group_by_event(event_id)
    sums = np.bincount(inv, weights=nonlocal_prob, minlength=n_events).astype(float)
    return unique, sums


def posterior_mass_total(
    nonlocal_prob: np.ndarray,
    event_id: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-event total posterior mass, for the ``arm_posterior_mass`` offset.

    Under the renormalization convention in ``posterior_to_arm_marginal``,
    this is identically equal to ``nonlocal_weight_sum``. The analysis
    plan (Section 1.2) notes ``posterior_mass_total_e ≤ nonlocal_weight_sum_e``
    assuming the unnormalized "junction mass is lost" convention;
    because we renormalize instead, equality holds and the distinction
    collapses. Exposed as a separate function so downstream code can
    address the ``arm_posterior_mass`` offset by name.
    """
    return nonlocal_weight_sum(nonlocal_prob, event_id)


# ── Per-event arm summaries ──


def arm_count_hard(
    nonlocal_prob: np.ndarray,
    arm_map: np.ndarray,
    event_id: np.ndarray,
    n_arms: int = 9,
    *,
    threshold: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """Hard MAP count per (event, arm).

    For each timebin with ``w_t > threshold``, increment the count for
    that timebin's event and MAP arm by 1.

    Returns
    -------
    event_ids : np.ndarray, shape (n_events,)
    counts : np.ndarray, shape (n_events, n_arms), int
    """
    nonlocal_prob = np.asarray(nonlocal_prob, dtype=float)
    arm_map = np.asarray(arm_map, dtype=np.int64)
    unique, inv, n_events = _group_by_event(event_id)

    mask = nonlocal_prob > threshold
    counts = np.zeros((n_events, n_arms), dtype=np.int64)
    if mask.any():
        np.add.at(counts, (inv[mask], arm_map[mask]), 1)
    return unique, counts


def arm_count_soft(
    nonlocal_prob: np.ndarray,
    arm_map: np.ndarray,
    event_id: np.ndarray,
    n_arms: int = 9,
) -> tuple[np.ndarray, np.ndarray]:
    """Soft MAP count per (event, arm), weighted by ``w_t``.

    Same MAP-arm assignment as :func:`arm_count_hard`; each timebin
    contributes ``w_t`` (not 1) to its MAP arm's bucket.

    Returns
    -------
    event_ids : np.ndarray, shape (n_events,)
    counts : np.ndarray, shape (n_events, n_arms), float
    """
    nonlocal_prob = np.asarray(nonlocal_prob, dtype=float)
    arm_map = np.asarray(arm_map, dtype=np.int64)
    unique, inv, n_events = _group_by_event(event_id)

    counts = np.zeros((n_events, n_arms), dtype=float)
    np.add.at(counts, (inv, arm_map), nonlocal_prob)
    return unique, counts


def arm_posterior_mass(
    nonlocal_prob: np.ndarray,
    arm_posterior: np.ndarray,
    event_id: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Full w_t-weighted arm-marginal posterior mass per (event, arm).

    ``m_{ea} = Σ_{t ∈ e} w_t · P(arm = a | spikes_t, non-local)``

    Uses the arm-marginal posterior (junction bins already excluded and
    renormalized).

    Parameters
    ----------
    nonlocal_prob : np.ndarray, shape (T,)
    arm_posterior : np.ndarray, shape (T, n_arms)
    event_id : np.ndarray, shape (T,)

    Returns
    -------
    event_ids : np.ndarray, shape (n_events,)
    mass : np.ndarray, shape (n_events, n_arms), float
    """
    nonlocal_prob = np.asarray(nonlocal_prob, dtype=float)
    arm_posterior = np.asarray(arm_posterior, dtype=float)
    unique, inv, n_events = _group_by_event(event_id)

    n_arms = arm_posterior.shape[1]
    mass = np.zeros((n_events, n_arms), dtype=float)
    weighted = nonlocal_prob[:, None] * arm_posterior  # (T, n_arms)

    # Iterate over arms, vectorized over timebins. n_arms = 9 in the
    # bandit task so the outer loop cost is negligible; the hot axis
    # is the per-arm bincount over T.
    for a in range(n_arms):
        mass[:, a] = np.bincount(inv, weights=weighted[:, a], minlength=n_events)
    return unique, mass
