"""Where non-local content sits: each content time bin's track distance to the animal and to the nearest reward well.

The per-bin distances the content-occupancy table is made of
(``event_statistics.quadrant_fractions`` sorts them into places). Both are
track-graph distances:

- to the animal: the decode frame's ``ahead_behind_distance``;
- to the nearest reward well: from the interior place bin nearest the MAP
  estimate, over the environment's track graph, to the bin nearest each well.

A well's position is not the track graph's node: it is the animal's median
linear position while immobile on the well's leaf arm, outside excluded time
(:func:`well_positions`). A session in which fewer than :data:`MIN_WELLS` wells
can be placed is left out, as its distances to the nearest well would be to an
incomplete set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from continuum_swr_replay.analysis.event_definition.analysed_time import (
    immobile_analysed_bins,
)
from continuum_swr_replay.analysis.event_definition.nonlocal_detection import (
    event_strict_overlap_mask,
)
from continuum_swr_replay.data_loaders.constants import N_LEAF_ARMS

MIN_WELLS = 4
"""A session needs this many wells placed (:func:`well_positions`)."""

MIN_IMMOBILE_BINS = 50
"""A leaf arm needs more immobile, analysed bins than this to place its well."""


def well_positions(dot: pd.DataFrame) -> list[float]:
    """Reward-well linear positions: the animal's median immobile position per leaf arm.

    Over the immobile bins (head speed below 4 cm/s) outside excluded time
    (``in_excluded_time``) with a finite linear position. A leaf arm with
    :data:`MIN_IMMOBILE_BINS` such bins or fewer places no well.

    Parameters
    ----------
    dot : pd.DataFrame, shape (n_time, n_columns)
        A decode frame read through the SWR source, with ``head_speed``,
        ``track_segment_id``, ``linear_position`` and ``in_excluded_time``.

    Returns
    -------
    list of float
        One linear position (cm) per leaf arm that places a well, in arm order.
    """
    immobile = immobile_analysed_bins(dot)
    segment = dot["track_segment_id"].to_numpy()
    linear = dot["linear_position"].to_numpy()
    wells = []
    for arm in range(N_LEAF_ARMS):
        m = immobile & (segment == arm) & np.isfinite(linear)
        if m.sum() > MIN_IMMOBILE_BINS:
            wells.append(float(np.median(linear[m])))
    return wells


def content_distances(
    dot: pd.DataFrame, environment
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]] | None:
    """Track distances of one session's non-local content time bins.

    Parameters
    ----------
    dot : pd.DataFrame, shape (n_time, n_columns)
        The session's decode frame read through the SWR source, with the event
        contract validated, and the columns of :func:`well_positions` and
        ``non_local_event``, ``SWR_event``, ``map_estimate`` and
        ``ahead_behind_distance``.
    environment : Environment
        The decode's fitted environment: ``is_track_interior_``,
        ``place_bin_centers_`` and ``get_distances_to_interior_bins``.

    Returns
    -------
    (d_animal, d_well, is_swr) or None
        Over the bins of a non-local event whose MAP estimate and both
        distances are finite, each of shape (n_bins,): the track distance (cm)
        from the content to the animal (``|ahead_behind_distance|``) and to the
        nearest well, and whether the bin's event overlaps an SWR (the strict
        event-level rule, so an event's bins are all in one group). None when
        fewer than :data:`MIN_WELLS` wells are placed.
    """
    interior = np.asarray(environment.is_track_interior_).reshape(-1)
    pos = np.asarray(environment.place_bin_centers_).reshape(-1)[interior]

    well_lin = well_positions(dot)
    if len(well_lin) < MIN_WELLS:
        return None

    # Per-interior-bin track distance to the nearest reward well.
    dist_matrix = np.asarray(
        environment.get_distances_to_interior_bins(pos[:, None])
    )  # (n_int, n_int)
    well_bins = [int(np.argmin(np.abs(pos - wl))) for wl in well_lin]
    nearest_well_dist = dist_matrix[:, well_bins].min(1)  # (n_interior,)

    event = dot["non_local_event"].to_numpy() > 0
    # Assign every bin of a Non-Local event to one group using the canonical
    # strict event-level overlap rule. A raw per-bin SWR mask would split a
    # partially overlapping event between both groups.
    is_swr = event_strict_overlap_mask(dot, "non_local_event", "SWR_event").to_numpy()
    map_estimate = dot["map_estimate"].to_numpy()
    d_animal = np.abs(dot["ahead_behind_distance"].to_numpy())
    content_bin = np.argmin(np.abs(pos[None, :] - map_estimate[:, None]), axis=1)
    d_well = nearest_well_dist[content_bin]

    # Guard against non-finite d_well (e.g. Inf from a disconnected graph component).
    valid = (
        event & np.isfinite(map_estimate) & np.isfinite(d_animal) & np.isfinite(d_well)
    )
    return d_animal[valid], d_well[valid], is_swr[valid]
