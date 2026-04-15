"""Per-event decoder-confidence summaries from the arm-marginal posterior.

Three event-level variables summarising the within-event decoder
posterior over arms, each computed as a ``w_t``-weighted mean across
the event's timebins:

- ``posterior_peak_mean``   — mean height of the MAP arm-posterior peak
- ``posterior_entropy_mean`` — mean Shannon entropy of the arm posterior
- ``posterior_hpd_width``    — mean HPD credible-set width (arms needed
  to cover a fixed fraction of the posterior), at coverage 50/80/95

All three consume the **arm-marginal** posterior produced by
``summaries.posterior_to_arm_marginal`` (junction bins already excluded
and renormalized), and a per-timebin ``w_t`` vector.

Every function expects the caller to pre-filter to in-event timebins
and returns ``(sorted_unique_event_ids, per_event_array)``, matching the
convention used in ``summaries.py``.
"""

from __future__ import annotations

import numpy as np

from .summaries import _group_by_event, nonlocal_weight_sum


def _weighted_mean_by_event(
    values: np.ndarray,
    nonlocal_prob: np.ndarray,
    event_id: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """w_t-weighted mean of a per-timebin scalar, grouped by event.

    Returns ``(event_ids, mean_per_event)``. Events with zero total w
    return NaN.
    """
    values = np.asarray(values, dtype=float)
    nonlocal_prob = np.asarray(nonlocal_prob, dtype=float)
    unique, inv, n_events = _group_by_event(event_id)
    numerator = np.bincount(
        inv, weights=values * nonlocal_prob, minlength=n_events
    )
    denominator = np.bincount(inv, weights=nonlocal_prob, minlength=n_events)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(denominator > 0, numerator / denominator, np.nan)
    return unique, mean


def posterior_peak_mean(
    arm_posterior: np.ndarray,
    nonlocal_prob: np.ndarray,
    event_id: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """w_t-weighted mean of ``max_a P(arm=a|t)`` per event.

    Peak height close to 1 means the decoder is confident about the
    arm at most timebins in the event; close to ``1/n_arms`` means the
    arm posterior is diffuse.
    """
    arm_posterior = np.asarray(arm_posterior, dtype=float)
    peak = arm_posterior.max(axis=1)  # (T,)
    return _weighted_mean_by_event(peak, nonlocal_prob, event_id)


def posterior_entropy_mean(
    arm_posterior: np.ndarray,
    nonlocal_prob: np.ndarray,
    event_id: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """w_t-weighted mean Shannon entropy (nats) of the arm posterior.

    Zero entropy for a delta posterior, ``log(n_arms)`` for a uniform
    posterior.
    """
    arm_posterior = np.asarray(arm_posterior, dtype=float)
    # Clip to avoid log(0); 0 * log(0) = 0 convention preserved.
    safe = np.where(arm_posterior > 0, arm_posterior, 1.0)
    entropy = -np.sum(arm_posterior * np.log(safe), axis=1)  # (T,)
    return _weighted_mean_by_event(entropy, nonlocal_prob, event_id)


def posterior_hpd_width(
    arm_posterior: np.ndarray,
    nonlocal_prob: np.ndarray,
    event_id: np.ndarray,
    *,
    coverage: float,
) -> tuple[np.ndarray, np.ndarray]:
    """w_t-weighted mean HPD credible-set width per event, at ``coverage``.

    For each timebin: sort arm probabilities in descending order, take
    the smallest number of arms whose cumulative mass is at least
    ``coverage``, and return that count. The per-event value is the
    ``w_t``-weighted mean of these counts.

    A width of 1 means the decoder places ``coverage`` of its mass on a
    single arm (confident); a width near ``n_arms`` means the posterior
    is nearly uniform.

    Parameters
    ----------
    coverage : float
        Target credible-set coverage in (0, 1] — typically 0.5, 0.8, or
        0.95.
    """
    if not 0 < coverage <= 1:
        raise ValueError(f"coverage must be in (0, 1]; got {coverage}")
    arm_posterior = np.asarray(arm_posterior, dtype=float)

    # Sort each timebin's arm probabilities in descending order and
    # cumsum. The smallest k with cumsum[k-1] >= coverage is the HPD
    # width; np.argmax returns the first True index, so width = argmax+1.
    sorted_desc = -np.sort(-arm_posterior, axis=1)
    cumsum = np.cumsum(sorted_desc, axis=1)
    meets = cumsum >= coverage - 1e-12  # guard against fp underflow

    # argmax on a bool row returns the first True; if no True (can only
    # happen with a zero-row / numerical corner), fall back to n_arms.
    any_true = meets.any(axis=1)
    width = np.where(any_true, meets.argmax(axis=1) + 1, arm_posterior.shape[1])
    width = width.astype(float)

    return _weighted_mean_by_event(width, nonlocal_prob, event_id)


# ── Convenience ──


def posterior_hpd_widths(
    arm_posterior: np.ndarray,
    nonlocal_prob: np.ndarray,
    event_id: np.ndarray,
    *,
    coverages: tuple[float, ...] = (0.5, 0.8, 0.95),
) -> tuple[np.ndarray, dict[float, np.ndarray]]:
    """Compute HPD widths at several coverage levels in one pass.

    Sorts and cumsums the arm posterior once per timebin and thresholds
    at each coverage, then does a single weighted-mean aggregation per
    coverage. Equivalent to calling :func:`posterior_hpd_width` once per
    coverage but avoids the repeated sort and group-index computation.

    Returns ``(event_ids, {coverage: widths})``.
    """
    if len(coverages) == 0:
        raise ValueError("coverages must be non-empty")
    for c in coverages:
        if not 0 < c <= 1:
            raise ValueError(f"coverage must be in (0, 1]; got {c}")

    arm_posterior = np.asarray(arm_posterior, dtype=float)
    nonlocal_prob = np.asarray(nonlocal_prob, dtype=float)

    # One sort, one cumsum, one group lookup.
    sorted_desc = -np.sort(-arm_posterior, axis=1)
    cumsum = np.cumsum(sorted_desc, axis=1)
    unique, inv, n_events = _group_by_event(event_id)
    denom = np.bincount(inv, weights=nonlocal_prob, minlength=n_events)
    n_arms = arm_posterior.shape[1]

    widths_by_coverage: dict[float, np.ndarray] = {}
    for c in coverages:
        meets = cumsum >= c - 1e-12
        any_true = meets.any(axis=1)
        per_bin_width = np.where(any_true, meets.argmax(axis=1) + 1, n_arms).astype(
            float
        )
        numer = np.bincount(inv, weights=per_bin_width * nonlocal_prob, minlength=n_events)
        with np.errstate(divide="ignore", invalid="ignore"):
            widths_by_coverage[c] = np.where(denom > 0, numer / denom, np.nan)

    return unique, widths_by_coverage


# Re-export from summaries for callers that want both.
__all__ = [
    "posterior_peak_mean",
    "posterior_entropy_mean",
    "posterior_hpd_width",
    "posterior_hpd_widths",
    "nonlocal_weight_sum",
]
