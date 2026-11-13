"""Spike evidence of non-local events: does the prior-excluded spike likelihood favour the decoded location over the animal's?

The decoder's posterior is likelihood times prior, so a confident posterior can
come from the prior while the spikes say little. Each event (and each matched
quiet window, the null) is scored on the decode's own non-local
log-likelihood ``logL_t(x)`` over its bins:

- ``dll_animal`` = mean_t [logL_t(MAP_t) - logL_t(animal_t)]; real content > 0.
- ``dll_per_spike`` = sum_t [logL_t(MAP_t) - logL_t(animal_t)] / n_spikes.
- ``dist_to_gpi_peak``: track distance from the event's mean MAP to the
  ground-process-intensity peak.

Every bin lookup is the nearest interior place bin (track-graph bins). The
windows (:func:`build_session_events`) are the immobile non-local events of
the decode frame in three groups, and the matched quiet windows outside
excluded time.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from continuum_swr_replay.analysis.constants import REMOTE_CM
from continuum_swr_replay.analysis.events.analysed_time import immobile_analysed_bins
from continuum_swr_replay.analysis.events.nonlocal_detection import (
    event_strict_overlap_mask,
)
from continuum_swr_replay.analysis.hippocampus.data_aggregation import (
    validate_dot_event_contract,
)
from continuum_swr_replay.analysis.hippocampus.state_evidence import (
    nearest_interior_bins,
)
from continuum_swr_replay.decoding.results import nonlocal_probability
from continuum_swr_replay.intervals import contiguous_true_runs

NON_EVENT_THRESHOLD = 0.2
"""Non-local probability below this marks a bin clearly local (a quiet window)."""

CLOSE_CM = 40.0
"""Mean |ahead-behind| below this makes a non-SWR event "close"; above
:data:`~continuum_swr_replay.analysis.constants.REMOTE_CM` it is "remote"."""

MIN_EVENT_BINS = 3
"""An event or a quiet window needs at least this many bins."""

MAX_BASELINE_BINS = 40
"""A quiet window has at most this many bins."""


def build_session_events(dot: pd.DataFrame) -> pd.DataFrame:
    """Classify non-local events and matched non-event baseline windows.

    The events are the frame's ``non_local_event`` labels of at least
    :data:`MIN_EVENT_BINS` bins, by SWR overlap (the strict event-level rule,
    read at the event's first bin) and mean ``|ahead_behind_distance|``:
    ``SWR-remote`` and ``nonSWR-remote`` above ``REMOTE_CM``, ``nonSWR-close``
    below :data:`CLOSE_CM`; the 40-100 cm band and SWR-close events are left
    out. The ``baseline`` windows are the runs of bins with non-local
    probability below :data:`NON_EVENT_THRESHOLD`, immobile and outside
    excluded time
    (:func:`~continuum_swr_replay.analysis.events.analysed_time.immobile_analysed_bins`),
    of :data:`MIN_EVENT_BINS` to :data:`MAX_BASELINE_BINS` bins.

    Parameters
    ----------
    dot : pd.DataFrame
        Per-timebin ``data_over_time`` (decode time axis), read through the SWR
        source, with the state-prob, ``SWR_event``, ``head_speed``,
        ``in_excluded_time`` and ``ahead_behind_distance`` columns.

    Returns
    -------
    pd.DataFrame
        One row per window: ``s, e`` (bin indices, inclusive), ``mean_abs_ab``,
        ``is_swr``, ``n_bins``, ``group``.

    Raises
    ------
    ValueError
        If an event's bins are not one contiguous run.
    """
    # Business-logic boundary: direct callers get the same fail-closed input
    # contract as the file-loading entry point.
    validate_dot_event_contract(dot)
    nlp = nonlocal_probability(dot).to_numpy()
    ab = dot["ahead_behind_distance"].to_numpy()
    # The read holds the labels to the speed rule and the probability contract.
    nonlocal_labels = dot["non_local_event"].to_numpy()
    is_swr_event = event_strict_overlap_mask(
        dot, "non_local_event", "SWR_event"
    ).to_numpy()

    rows = []
    for event_label in pd.unique(nonlocal_labels[nonlocal_labels > 0]):
        event_bins = np.flatnonzero(nonlocal_labels == event_label)
        n_bins = len(event_bins)
        if n_bins < MIN_EVENT_BINS:
            continue
        if n_bins > 1 and not (np.diff(event_bins) == 1).all():
            raise ValueError(f"non_local_event {event_label} is not one contiguous run")
        s, e = int(event_bins[0]), int(event_bins[-1])
        mab = float(np.nanmean(np.abs(ab[event_bins])))
        is_swr = bool(is_swr_event[event_bins[0]])
        if is_swr and mab > REMOTE_CM:
            group = "SWR-remote"
        elif not is_swr and mab > REMOTE_CM:
            group = "nonSWR-remote"
        elif not is_swr and mab < CLOSE_CM:
            group = "nonSWR-close"
        else:
            continue  # 40-100 cm middle band, or SWR-close: excluded for clarity
        rows.append(
            {
                "s": s,
                "e": e,
                "mean_abs_ab": mab,
                "is_swr": is_swr,
                "n_bins": n_bins,
                "group": group,
            }
        )

    # the null is analysed time too: no baseline window touches excluded time
    # (the artifact mask widened by 0.3 s; the low-non-local label is decoded)
    quiet = (nlp < NON_EVENT_THRESHOLD) & immobile_analysed_bins(dot)
    for s, stop in contiguous_true_runs(quiet):
        e = stop - 1
        n_bins = e - s + 1
        if n_bins < MIN_EVENT_BINS or n_bins > MAX_BASELINE_BINS:
            continue
        rows.append(
            {
                "s": s,
                "e": e,
                "mean_abs_ab": 0.0,
                "is_swr": False,
                "n_bins": n_bins,
                "group": "baseline",
            }
        )

    return pd.DataFrame(rows)


def stored_likelihood_columns(results: xr.Dataset, env) -> np.ndarray:
    """Columns of the stored ``log_likelihood`` holding the non-local likelihood.

    Parameters
    ----------
    results : xr.Dataset
        The decode's ``results.nc``, with ``state`` and ``position`` per state bin.
    env : Environment
        Fitted; its place bins are the Non-Local Continuous state's bins.

    Returns
    -------
    ndarray of int, shape (n_interior_bins,)
        The Non-Local Continuous columns at the interior bins, in
        ``is_track_interior_`` order (the order ``nearest_interior_bins`` indexes).

    Raises
    ------
    ValueError
        The state's bins are not the environment's place bins.
    """
    cols = np.flatnonzero(results["state"].values == "Non-Local Continuous")
    centers = np.asarray(env.place_bin_centers_).ravel()
    if len(cols) != len(centers) or not np.allclose(
        results["position"].values[cols], centers
    ):
        raise ValueError(
            "the stored Non-Local Continuous bins are not the environment's place bins"
        )
    return cols[np.asarray(env.is_track_interior_).ravel()]


def stored_event_likelihood(
    log_likelihood: xr.DataArray, cols: np.ndarray, s: int, e: int
) -> np.ndarray:
    """The stored non-local log-likelihood of bins ``s..e``.

    Returns
    -------
    ndarray, shape (e - s + 1, n_interior_bins)
    """
    rows = log_likelihood.isel(time=slice(s, e + 1)).values
    return np.asarray(rows[:, cols], dtype=float)


def window_spike_counts(
    spike_times: Sequence[np.ndarray], start: float, stop: float
) -> tuple[int, int]:
    """Spikes in ``[start, stop)`` over all tetrodes, and the tetrodes with any."""
    counts = [int(np.count_nonzero((x >= start) & (x < stop))) for x in spike_times]
    return sum(counts), sum(c > 0 for c in counts)


def event_metrics(
    s: int,
    e: int,
    L: np.ndarray,
    n_spikes: int,
    mape: np.ndarray,
    lin: np.ndarray,
    env,
    pos: np.ndarray,
    gpi_peak_pos: float,
) -> dict | None:
    """Compute the per-event reality metrics (track-distance bins).

    Parameters
    ----------
    s, e : int
        The window's first and last bin (inclusive).
    L : ndarray, shape (e - s + 1, n_interior_bins)
        Its non-local log-likelihood.
    n_spikes : int
        Its spikes.
    mape, lin : ndarray, shape (n_time,)
        The decoded MAP and the animal's linear position, cm.
    env : Environment
        Fitted.
    pos : ndarray, shape (n_interior_bins,)
        The interior place bins' positions, cm.
    gpi_peak_pos : float
        The ground-process intensity's peak position, cm.

    Returns
    -------
    dict or None
        ``dll_animal``, ``dll_per_spike``, ``n_spikes``, ``map_mean`` and
        ``dist_to_gpi_peak``, over the bins whose MAP and animal position are
        finite. None for a window with fewer than two such bins or with no
        spikes: the evidence is per spike, so a window without spikes has none
        to score. This is deliberate for quiet (baseline) windows too, so the
        quiet-window reference is the windows with at least one spike
        (2026-11-12: 5,280 of 30,447 quiet windows have none; no event window
        does).
    """
    mt, at = mape[s : e + 1], lin[s : e + 1]
    valid = np.isfinite(mt) & np.isfinite(at)
    if valid.sum() < 2 or n_spikes == 0:
        return None
    Lv = L[valid]
    i_map = nearest_interior_bins(env, mt[valid])
    i_animal = nearest_interior_bins(env, at[valid])
    n = len(i_map)
    delta = Lv[np.arange(n), i_map] - Lv[np.arange(n), i_animal]
    map_mean = float(np.nanmean(mt))
    dist_gpi = float(
        np.asarray(env.get_distances_to_interior_bins(np.array([[map_mean]]))).ravel()[
            np.argmin(np.abs(pos - gpi_peak_pos))
        ]
    )
    return {
        "dll_animal": float(np.nanmean(delta)),
        "dll_per_spike": float(np.nansum(delta) / n_spikes),
        "n_spikes": int(n_spikes),
        "map_mean": map_mean,
        "dist_to_gpi_peak": dist_gpi,
    }
