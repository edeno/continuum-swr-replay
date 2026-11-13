"""Per-event replay summaries from decoder outputs.

Three response variables for the conditional Poisson content model, plus
their companion length measures:

==============================  =============================  ======================
Response                        Length measure                 Role
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

The alternative convention — treating junction mass as "lost" rather
than redistributed — would give ``posterior_mass_total_e ≤
nonlocal_weight_sum_e`` with a strict ≤. We
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

from ...parameters import DEFAULT_NON_LOCAL_THRESHOLD

PROBABILITY_ROW_SUM_ATOL = 1e-5

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
        Shape ``(n_timebins, n_arms)``; every row sums to 1 up to
        floating-point tolerance.

    Raises
    ------
    ValueError
        If the posterior is not a finite, nonnegative probability matrix on
        real-arm bins; if edge labels are not one-dimensional integer IDs in
        ``{-1, 0, ..., n_arms - 1}``; or if any timebin's posterior lies
        entirely on junction bins. Junction-bin entries may be ``NaN`` because
        decoder artifacts use those values for structural placeholders.
    """
    raw_posterior = np.asarray(conditional_posterior)
    if raw_posterior.ndim != 2:
        raise ValueError(
            "conditional_posterior must be 2-D with shape "
            f"(n_timebins, n_position_bins); got {raw_posterior.shape}"
        )
    if np.iscomplexobj(raw_posterior):
        raise ValueError("conditional_posterior probabilities must be real-valued")
    try:
        conditional_posterior = raw_posterior.astype(float, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("conditional_posterior probabilities must be numeric") from exc

    raw_edge_id = np.asarray(edge_id)
    if raw_edge_id.ndim != 1:
        raise ValueError(f"edge_id must be 1-D; got shape {raw_edge_id.shape}")
    if np.iscomplexobj(raw_edge_id):
        raise ValueError("edge_id values must be finite integers")
    try:
        edge_id_float = raw_edge_id.astype(float, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("edge_id values must be finite integers") from exc
    if (
        not np.isfinite(edge_id_float).all()
        or not np.equal(edge_id_float, np.floor(edge_id_float)).all()
    ):
        raise ValueError("edge_id values must be finite integers")
    edge_id = edge_id_float.astype(np.int64, copy=False)

    if conditional_posterior.shape[1] != edge_id.shape[0]:
        raise ValueError(
            f"posterior shape {conditional_posterior.shape} does not match "
            f"edge_id length {edge_id.shape[0]}"
        )

    invalid_negative = edge_id < -1
    if invalid_negative.any():
        invalid = np.unique(edge_id[invalid_negative]).tolist()
        raise ValueError(
            "edge_id values must use -1 for junctions and nonnegative integers "
            f"for real arms; got invalid IDs {invalid}"
        )

    if n_arms is not None and (
        isinstance(n_arms, (bool, np.bool_))
        or not isinstance(n_arms, (int, np.integer))
        or int(n_arms) <= 0
    ):
        raise ValueError(f"n_arms must be a positive integer; got {n_arms!r}")

    is_real = edge_id >= 0
    if not is_real.any():
        raise ValueError(
            "No real arm bins found — every edge_id is -1 (junction). "
            "Check that the environment's place_bin_centers_nodes_df_ "
            "contains at least one real edge."
        )
    if n_arms is None:
        n_arms = int(edge_id[is_real].max()) + 1
    else:
        n_arms = int(n_arms)

    out_of_range = edge_id[is_real] >= n_arms
    if out_of_range.any():
        invalid = np.unique(edge_id[is_real][out_of_range]).tolist()
        raise ValueError(
            f"real edge_id values {invalid} fall outside the canonical range "
            f"0..{n_arms - 1} for n_arms={n_arms}"
        )

    real_mass = conditional_posterior[:, is_real]
    if real_mass.shape[0]:
        # Reductions avoid allocating decoder-sized boolean/copy arrays solely
        # for validation. ``min``/``max`` propagate any NaN and expose either
        # signed infinity, while the minimum also certifies nonnegativity.
        real_min = float(real_mass.min())
        real_max = float(real_mass.max())
        if np.isnan(real_min) or np.isnan(real_max):
            raise ValueError(
                "conditional_posterior contains non-finite values on real-arm "
                "bins; NaN is permitted only at junction placeholders "
                "(edge_id == -1)"
            )
        if np.isinf(real_min) or np.isinf(real_max):
            raise ValueError("conditional_posterior contains infinite values")
        if real_min < 0:
            raise ValueError("conditional_posterior contains negative probabilities")

    junction_mass = conditional_posterior[:, ~is_real]
    if np.isinf(junction_mass).any():
        raise ValueError("conditional_posterior contains infinite values")
    if (junction_mass < 0).any():
        raise ValueError("conditional_posterior contains negative probabilities")

    # The input is a conditional probability distribution, not an arbitrary
    # score matrix. Decoder files are float32-backed and can accumulate several
    # parts in 1e-6 of row-sum error across ~250 bins, hence the explicit 1e-5
    # absolute tolerance. Structural junction NaNs carry zero probability mass.
    input_row_sum = np.nansum(conditional_posterior, axis=1)
    proper_rows = np.isclose(
        input_row_sum, 1.0, rtol=0.0, atol=PROBABILITY_ROW_SUM_ATOL
    )
    if not proper_rows.all():
        deviation = np.abs(input_row_sum[~proper_rows] - 1.0)
        raise ValueError(
            "conditional_posterior rows must sum to 1 within absolute "
            f"tolerance {PROBABILITY_ROW_SUM_ATOL:g}; "
            f"{int((~proper_rows).sum())}/{len(proper_rows)} row(s) fail "
            f"(maximum absolute deviation {float(deviation.max()):.6g})"
        )

    n_timebins = conditional_posterior.shape[0]
    arm_posterior = np.zeros((n_timebins, n_arms), dtype=float)
    real_edges = edge_id[is_real]

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
    if zero_rows.any():
        # A timebin whose entire posterior sits on junction bins has no arm
        # information at all. The former behavior filled the row with a uniform
        # 1/n_arms, which asserts *equal evidence for every arm* where the truth
        # is *no evidence* -- and it does not stay contained: arm_map_from_posterior
        # argmaxes that row and returns arm 0 (ties break to the lowest index), so
        # the bin acquires a confident, systematically biased arm assignment; and
        # the row contributes 1/n_arms of a real bin's mass to every arm in the
        # content sums, diluting real content and crediting arms the replay never
        # visited. Junction bins are interpolation placeholders, not positions, so
        # a posterior concentrated entirely on them is a decode-integrity signal.
        # This mirrors the guard above for an environment with no real bins.
        raise ValueError(
            f"{int(zero_rows.sum())} of {n_timebins} timebin(s) have their "
            "entire conditional posterior on junction bins (edge_id == -1), "
            "so they carry no arm information. Junction bins are "
            "interpolation placeholders, not positions -- check the decode "
            "and the environment rather than assigning these bins an arm."
        )
    return arm_posterior / row_sum


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
    threshold: float = DEFAULT_NON_LOCAL_THRESHOLD,
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
    this is identically equal to ``nonlocal_weight_sum``. Under the
    unnormalized "junction mass is lost" convention one would instead get
    ``posterior_mass_total_e ≤ nonlocal_weight_sum_e``;
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
    threshold: float = DEFAULT_NON_LOCAL_THRESHOLD,
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


def arm_fraction_from_event_total(
    arm_content: np.ndarray,
    event_total: np.ndarray,
    *,
    denominator_floor: float | None = None,
) -> np.ndarray:
    """Normalize candidate-arm content by its event-matched total.

    This is the shared denominator rule for MAP counts and marginalized
    posterior mass. ``event_total`` must already be broadcast to candidate-arm
    rows and describe the same included bins or weights as ``arm_content``.
    Zero-total or unavailable rows are undefined (NaN), never fabricated zero.

    ``denominator_floor`` clamps the divisor from below on the valid rows,
    guarding against division by a vanishingly small (but positive) total. The
    posterior-mass caller passes ``1e-10`` to preserve the pre-refactor
    behavior; integer-count callers leave it unset (a positive count is never
    smaller than the floor anyway).
    """
    content = np.asarray(arm_content, dtype=float)
    total = np.asarray(event_total, dtype=float)
    if content.shape != total.shape:
        raise ValueError(
            "arm_content and event_total must have identical shapes; "
            f"got {content.shape} and {total.shape}"
        )
    finite_content = np.isfinite(content)
    finite_total = np.isfinite(total)
    if (content[finite_content] < 0).any():
        raise ValueError("arm content must be nonnegative where finite")
    if (total[finite_total] < 0).any():
        raise ValueError("event totals must be nonnegative where finite")
    output = np.full(content.shape, np.nan, dtype=float)
    valid = finite_content & finite_total & (total > 0)
    denominator = total[valid]
    if denominator_floor is not None:
        denominator = np.maximum(denominator, denominator_floor)
    output[valid] = content[valid] / denominator
    return output
