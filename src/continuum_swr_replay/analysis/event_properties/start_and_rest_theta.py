"""Each non-local event's start distance, and the theta of duration-matched quiet rest.

From one epoch's decode frame (read through the SWR source) and environment:

- **start distance**: the track-graph distance from the animal's position to the
  event's first decoded position (the MAP at the event's first bin with a finite
  MAP), both on real place bins, never a junction placeholder
  (``posterior_analysis.track_distance_cm``);
- **duration-matched quiet rest**: windows of quiet rest (below 4 cm/s, in no
  non-local, SWR or MUA event, outside excluded time:
  ``brain_state_reference.regime_masks``) with as many bins as the event, drawn
  uniformly over every such window of the epoch
  (``brain_state_reference.matched_window_means``). Each window's theta is the
  mean of ``theta_power_zscore`` over its bins, as an event's ``theta_z`` is, and
  it is theta-like when it reaches the epoch's median running theta
  (``brain_state_reference``). An event's ``rest_theta_like_share`` is the share
  of its windows that are theta-like.

``scripts/detection/measure_event_start_and_rest_theta.py`` reads each epoch's
frame, calls :func:`start_and_rest_theta` and writes one table per epoch; the
events summary reads those tables.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.decode_quality.posterior_analysis import (
    track_distance_cm,
)
from continuum_swr_replay.analysis.event_properties.brain_state_reference import (
    brain_state_reference,
    matched_window_means,
    regime_masks,
)


def start_and_rest_theta(
    dot: pd.DataFrame,
    environment: Any,
    events: pd.DataFrame,
    n_windows: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """One epoch's events: start distance and duration-matched rest theta.

    Parameters
    ----------
    dot : pd.DataFrame, shape (n_time, n_columns)
        The epoch's decode frame read through the SWR source, with
        ``non_local_event`` (0 is no event), ``map_estimate`` and
        ``linear_position`` (cm), ``theta_power_zscore``, and what
        ``brain_state_reference.regime_masks`` and ``brain_state_reference``
        read.
    environment : Environment
        The decode's environment, with ``place_bin_centers_nodes_df_`` and
        ``distance_between_nodes_``.
    events : pd.DataFrame, shape (n_events, n_columns)
        The epoch's events, with ``event_id`` and ``non_local_event``, the
        frame label of each.
    n_windows : int
        Rest windows drawn per event.
    rng : np.random.Generator
        Draws the rest windows.

    Returns
    -------
    measures : pd.DataFrame, shape (n_events, 10)
        In the order of ``events``: ``event_id``; ``n_bins``, the bins of its
        label, and ``bins_contiguous``, whether they are one run (a label can
        be split); ``start_distance_cm`` (NaN when no bin has a finite MAP);
        ``theta_mean``, the mean ``theta_power_zscore`` over its bins
        (NaN-skipping); the epoch's ``theta_running_median`` and
        ``theta_rest_median``; ``rest_windows``, the windows drawn with a
        finite mean; ``rest_theta_like_share``, the share of those at or above
        the running median (NaN with none); and ``rest_theta_mean``, their mean
        theta.

    Raises
    ------
    ValueError
        If an event's label has no bin in the frame.
    """
    labels = dot["non_local_event"].to_numpy()
    order = np.argsort(labels, kind="stable")
    bounds = np.searchsorted(labels[order], events.non_local_event.to_numpy())
    ends = np.searchsorted(
        labels[order], events.non_local_event.to_numpy(), side="right"
    )
    n_rows = ends - bounds
    if (n_rows == 0).any():
        raise ValueError("an event's label has no bin in the frame")
    map_estimate = dot["map_estimate"].to_numpy(dtype=np.float64)
    animal = dot["linear_position"].to_numpy(dtype=np.float64)
    theta = dot["theta_power_zscore"].to_numpy(dtype=np.float64)

    first_decoded = np.full(len(events), -1, dtype=np.int64)
    theta_mean = np.full(len(events), np.nan)
    contiguous = np.zeros(len(events), dtype=bool)
    for k, (a, b) in enumerate(zip(bounds, ends, strict=True)):
        rows = np.sort(order[a:b])
        contiguous[k] = rows[-1] - rows[0] + 1 == rows.size
        finite = rows[np.isfinite(map_estimate[rows])]
        if finite.size:
            first_decoded[k] = finite[0]
        theta_mean[k] = np.nanmean(theta[rows])
    has_map = first_decoded >= 0
    start = np.full(len(events), np.nan)
    start[has_map] = track_distance_cm(
        animal[first_decoded[has_map]],
        map_estimate[first_decoded[has_map]],
        environment.place_bin_centers_nodes_df_,
        environment.distance_between_nodes_,
    )

    rest = regime_masks(dot)["rest"]
    reference = brain_state_reference(dot)
    windows = matched_window_means(theta, rest, n_rows, n_windows, rng)
    valid = np.isfinite(windows)
    above = windows >= reference["theta_running_median"]
    with np.errstate(invalid="ignore"):
        share = np.where(
            valid.any(axis=1), (above & valid).sum(axis=1) / valid.sum(axis=1), np.nan
        )
    return pd.DataFrame(
        {
            "event_id": events.event_id.to_numpy(),
            "n_bins": n_rows,
            "bins_contiguous": contiguous,
            "start_distance_cm": start,
            "theta_mean": theta_mean,
            "theta_running_median": reference["theta_running_median"],
            "theta_rest_median": reference["theta_rest_median"],
            "rest_windows": valid.sum(axis=1),
            "rest_theta_like_share": share,
            "rest_theta_mean": np.nanmean(np.where(valid, windows, np.nan), axis=1)
            if valid.any()
            else np.nan,
        }
    )
