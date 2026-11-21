"""Per-event decoder-confidence summaries from the arm-marginal posterior.

Three event-level variables summarizing the within-event decoder
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

from .summaries import _group_by_event


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
    numerator = np.bincount(inv, weights=values * nonlocal_prob, minlength=n_events)
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


def _arm_hpd_count(arm_posterior: np.ndarray, coverage: float) -> np.ndarray:
    """Per-timebin count of arms in the HPD credible set at ``coverage``.

    Uses ``non_local_detector.model_checking.highest_posterior_density.get_highest_posterior_threshold``
    to find the density threshold that captures at least ``coverage`` of
    the mass per timebin, then counts arms whose probability is at or
    above that threshold — i.e., the tie-inclusive HPD set size. This
    matches how :func:`non_local_detector` computes the position-level
    HPD (the ``hpd`` column in ``data_over_time.pkl`` represents the cm
    area of that same set via
    :func:`get_HPD_spatial_coverage`), so arm-level and position-level
    HPD carry consistent semantics across the project.

    The returned count differs from a "smallest-k reaching coverage"
    definition only when ties push additional arms to the same density
    — a rare case on a 9-arm marginal of a realistic posterior.

    Parameters
    ----------
    arm_posterior : np.ndarray, shape (T, n_arms)
        Rows sum to 1 (arm-marginal posterior).
    coverage : float
        Target coverage in (0, 1].

    Returns
    -------
    width : np.ndarray, shape (T,), float
    """
    # Imported lazily — keeps ``uncertainty`` importable without the
    # detector package on callers that only use peak / entropy.
    import xarray as xr
    from non_local_detector.model_checking.highest_posterior_density import (
        get_highest_posterior_threshold,
    )

    # get_highest_posterior_threshold takes an xr.DataArray and ignores
    # the non-time coord (it reshapes to (n_time, -1) internally) — the
    # wrap is just to satisfy the type. A nominal "position" dim is
    # required because the helper uses that name internally.
    arm_da = xr.DataArray(
        arm_posterior,
        dims=("time", "position"),
        coords={"position": np.arange(arm_posterior.shape[1])},
    )
    threshold = get_highest_posterior_threshold(arm_da, coverage=coverage)
    # Sanity: ``get_highest_posterior_threshold`` is contracted to
    # return one threshold per timebin. Explicit check so a future
    # package change would surface as a clear assertion here rather
    # than as a subtle broadcast error further downstream.
    assert threshold.shape == (arm_posterior.shape[0],), (
        f"unexpected threshold shape {threshold.shape} "
        f"for arm_posterior of shape {arm_posterior.shape}"
    )
    return (arm_posterior >= threshold[:, None]).sum(axis=1).astype(float)


def posterior_hpd_width(
    arm_posterior: np.ndarray,
    nonlocal_prob: np.ndarray,
    event_id: np.ndarray,
    *,
    coverage: float,
) -> tuple[np.ndarray, np.ndarray]:
    """w_t-weighted mean HPD credible-set width per event, at ``coverage``.

    Per-timebin HPD size via :func:`_arm_hpd_count` (uses
    :mod:`non_local_detector`'s threshold routine — see helper
    docstring for the exact definition), then aggregated as a
    ``w_t``-weighted mean per event.

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
    width = _arm_hpd_count(arm_posterior, coverage)
    return _weighted_mean_by_event(width, nonlocal_prob, event_id)


# ── Convenience ──


def posterior_hpd_widths(
    arm_posterior: np.ndarray,
    nonlocal_prob: np.ndarray,
    event_id: np.ndarray,
    *,
    coverages: tuple[float, ...] = (0.5, 0.8, 0.95),
) -> tuple[np.ndarray, dict[float, np.ndarray]]:
    """Compute HPD widths at several coverage levels.

    Equivalent to calling :func:`posterior_hpd_width` once per coverage
    but shares the per-event group-index computation across coverages.

    Returns ``(event_ids, {coverage: widths})``.
    """
    if len(coverages) == 0:
        raise ValueError("coverages must be non-empty")
    for c in coverages:
        if not 0 < c <= 1:
            raise ValueError(f"coverage must be in (0, 1]; got {c}")

    arm_posterior = np.asarray(arm_posterior, dtype=float)
    nonlocal_prob = np.asarray(nonlocal_prob, dtype=float)

    unique, inv, n_events = _group_by_event(event_id)
    denom = np.bincount(inv, weights=nonlocal_prob, minlength=n_events)

    widths_by_coverage: dict[float, np.ndarray] = {}
    for c in coverages:
        width = _arm_hpd_count(arm_posterior, c)
        numer = np.bincount(inv, weights=width * nonlocal_prob, minlength=n_events)
        with np.errstate(divide="ignore", invalid="ignore"):
            widths_by_coverage[c] = np.where(denom > 0, numer / denom, np.nan)

    return unique, widths_by_coverage


__all__ = [
    "posterior_peak_mean",
    "posterior_entropy_mean",
    "posterior_hpd_width",
    "posterior_hpd_widths",
]
