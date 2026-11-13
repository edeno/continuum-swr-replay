"""Analysis of decoded posterior distributions.

This module provides functions for analyzing the posterior probability distributions
output by the continuum non-local detector. It computes summary statistics, extracts
maximum a posteriori (MAP) estimates, calculates trajectory speeds, and identifies
highest posterior density (HPD) regions.

These analyses are essential for characterizing hippocampal replay events: determining
what position is being represented, how fast the representation moves through space,
and how confident/localized the decoding is.

Main Functions
--------------
get_conditional_non_local_posterior
    Extract and normalize posterior probability conditioned on being in the non-local
    state (combining continuous and fragmented non-local states). This isolates the
    spatial representation during replay.
load_conditional_posterior
    The same, read from a decode's ``results.nc`` (``decoding.results.open_results``).
normalized_replay_position_by_swr
    The decoded position of non-local event bins as a fraction of the track, split
    by SWR overlap.
get_map_speed
    Compute the speed of the decoded position over time using the maximum a posteriori
    (MAP) estimate. Useful for distinguishing stationary replay from trajectory sweeps.
get_posterior_stats
    Calculate comprehensive summary statistics from posterior distributions including
    mean position, variance, highest posterior density coverage, and ahead/behind
    distance relative to actual position.

Analysis Capabilities
---------------------
- Posterior conditioning: Separate non-local (replay) from local representations
- Trajectory extraction: Identify decoded position sequences and their dynamics
- Speed computation: Quantify how fast decoded position changes (cm/s)
- HPD analysis: Determine spatial extent of confident decoding regions
- Ahead/behind distance: Compute signed distance between decoded and actual position
  along the track graph to identify forward vs. backward replay

Notes
-----
- All analyses use xarray Datasets for labeled multi-dimensional data
- MAP estimates are smoothed with Gaussian filter before differentiation
- HPD regions are computed at specified coverage levels (e.g., 95%)
- Ahead/behind distance uses shortest path on track graph (can be negative for
  backward replay)
- Position coordinates can be 1D (linear track) or 2D (open field)

See Also
--------
non_local_detector.analysis : Core decoding analysis functions
non_local_detector.model_checking.highest_posterior_density : HPD computation
continuum_swr_replay.analysis.hippocampus.data_aggregation : Time-series aggregation

Examples
--------
>>> # Get non-local posterior
>>> nl_posterior = get_conditional_non_local_posterior(results)
>>> # Compute MAP speed
>>> speed = get_map_speed(nl_posterior, track_graph, bin_centers, sampling_frequency=500)
>>> # Get comprehensive statistics
>>> stats = get_posterior_stats(results, position_info, detector, env)

"""

from pathlib import Path
from typing import Literal

import networkx as nx
import numpy as np
import pandas as pd
import xarray as xr
from non_local_detector.analysis import (
    get_ahead_behind_distance,
    get_trajectory_data,
    maximum_a_posteriori_estimate,
)
from non_local_detector.environment import Environment
from non_local_detector.model_checking.highest_posterior_density import (
    get_highest_posterior_threshold,
    get_HPD_spatial_coverage,
)
from ripple_detection.core import gaussian_smooth

from continuum_swr_replay.analysis.events.nonlocal_detection import (
    nonlocal_event_bins_by_swr,
)

from ...data_types import SessionData
from ...decoding.results import open_results
from ...parameters import PLACE_BIN_SIZE_1D, SAMPLING_FREQUENCY


def get_conditional_non_local_posterior(results: xr.Dataset) -> xr.DataArray:
    """Compute posterior conditioned on non-local state.

    Combines continuous and fragmented non-local states and normalizes.

    Parameters
    ----------
    results : xr.Dataset
        Decoding results containing acausal_posterior

    Returns
    -------
    conditional_posterior : xr.DataArray
        Normalized posterior conditioned on non-local state.
        Time points with negligible non-local probability are forward-filled
        from the nearest valid time point.

    """
    acausal_posterior = results.acausal_posterior.sel(
        state="Non-Local Continuous"
    ) + results.acausal_posterior.sel(state="Non-Local Fragmented")

    non_local_prob = acausal_posterior.sum("position")

    # Compute conditional posterior, avoiding division by zero
    conditional_posterior = xr.where(
        non_local_prob > 1e-10,
        acausal_posterior / non_local_prob,
        np.nan,
    )

    # Forward fill any all-NaN time points (e.g., first time point before spikes)
    # then backward fill any remaining (e.g., if first points are NaN)
    conditional_posterior = conditional_posterior.ffill("time").bfill("time")

    # Type ignore: xarray operations return Any but result is DataArray
    return conditional_posterior  # type: ignore[no-any-return]


def load_conditional_posterior(results_path: str | Path) -> xr.DataArray:
    """Read a decode's ``results.nc`` and return the conditional non-local posterior.

    Parameters
    ----------
    results_path : str | Path
        Path to ``{epoch_identifier}_results.nc`` saved by the decoder.

    Returns
    -------
    xr.DataArray
        Shape ``(time, position)`` — a proper probability distribution
        over place bins per timebin, conditioned on the non-local state
        (:func:`get_conditional_non_local_posterior`). Place bins that
        include junction (``edge_id == -1``) placeholders are still
        present; downstream code excludes them via
        ``summaries.posterior_to_arm_marginal``.

    Notes
    -----
    The file is read by ``decoding.results.open_results``, which rebuilds
    the ``state_bins`` multi-index the netCDF round-trip drops, so
    ``posterior.sel(state="Non-Local Continuous")`` resolves.

    The returned DataArray is lazy-loaded if the file is chunked; call
    ``.load()`` on the caller side if eager materialization is desired.
    """
    return get_conditional_non_local_posterior(open_results(results_path))


def _map_node_ids(
    posterior_array: np.ndarray,
    place_bin_center_ind_to_node: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """MAP track-graph node id per timebin, plus a validity mask.

    All-NaN rows get a placeholder node (bin index 0) and are flagged
    ``False`` in the returned mask; callers must mask those positions.

    Parameters
    ----------
    posterior_array : np.ndarray, shape (n_time, n_positions)
    place_bin_center_ind_to_node : np.ndarray, shape (n_positions,)

    Returns
    -------
    node_ids : np.ndarray, shape (n_time,)
    has_valid_data : np.ndarray of bool, shape (n_time,)
    """
    has_valid_data = ~np.all(np.isnan(posterior_array), axis=1)
    map_position_ind = np.zeros(len(posterior_array), dtype=int)
    if np.any(has_valid_data):
        map_position_ind[has_valid_data] = np.nanargmax(
            posterior_array[has_valid_data], axis=1
        )
    node_ids = place_bin_center_ind_to_node[map_position_ind]
    # Junction placeholder bins carry node_id == -1 (edge_id == -1 interpolation
    # artifacts) and are not in the track graph. If the MAP ever lands on one,
    # mark the bin invalid and swap in a real placeholder node so the downstream
    # graph-distance lookup neither uses it nor raises (junction-bin rule).
    junction = node_ids == -1
    if junction.any():
        has_valid_data = has_valid_data & ~junction
        real_nodes = place_bin_center_ind_to_node[place_bin_center_ind_to_node != -1]
        node_ids = np.where(junction, real_nodes[0], node_ids)
    return node_ids, has_valid_data


def _forward_step_distances(
    node_ids: np.ndarray,
    track_graph_with_bin_centers_edges: nx.Graph,
) -> np.ndarray:
    """Graph shortest-path distance (cm) between successive MAP nodes.

    Returns an array of shape ``(n_time - 1,)`` where element ``i`` is the
    distance ``node[i] -> node[i + 1]``.
    """
    shortest_path_lengths = dict(
        nx.shortest_path_length(track_graph_with_bin_centers_edges, weight="distance")
    )
    try:
        return np.array(
            [
                shortest_path_lengths[node1][node2]
                for node1, node2 in zip(node_ids[:-1], node_ids[1:], strict=True)
            ]
        )
    except KeyError as e:
        raise nx.NetworkXNoPath(
            f"No path between successive MAP positions (node {e}). "
            "The track graph may be disconnected."
        ) from e


def get_map_step_distance(
    posterior: xr.DataArray | np.ndarray,
    track_graph_with_bin_centers_edges: nx.Graph,
    place_bin_center_ind_to_node: np.ndarray,
) -> np.ndarray:
    """Per-timebin track-graph distance traveled by the MAP decoded position.

    Unsmoothed graph shortest-path distance (cm) between the MAP position at
    consecutive timebins, aligned as a backward difference so
    ``step_distance[t]`` is the distance arriving at bin ``t`` and
    ``step_distance[0]`` is ``0``. This differs from :func:`get_map_speed`,
    which divides by ``dt`` and Gaussian-smooths; here the raw per-bin
    displacement is preserved so downstream summaries can sum it (path length)
    or take robust order statistics (jumpiness).

    A step is ``NaN`` whenever either of its two endpoint bins had an all-NaN
    posterior; bin 0 is ``NaN`` only if its own posterior was invalid.

    Note the backward alignment is *opposite* to :func:`get_map_speed`, which
    is forward-aligned (``speed[t]`` spans ``node[t] -> node[t + 1]``). The two
    are therefore offset by one bin: ``step_distance[t] != map_speed[t] * dt``
    row-for-row.

    Parameters
    ----------
    posterior : xr.DataArray | np.ndarray, shape (n_time, n_positions)
        Posterior probability distribution over position.
    track_graph_with_bin_centers_edges : nx.Graph
        Track graph with bin centers as nodes and ``distance`` edge weights.
    place_bin_center_ind_to_node : np.ndarray, shape (n_positions,)
        Mapping from place bin center index to node ID in the track graph.

    Returns
    -------
    np.ndarray, shape (n_time,)
        Track-graph distance (cm) arriving at each timebin; index 0 is 0.0.
    """
    posterior_array = np.asarray(posterior)
    node_ids, has_valid_data = _map_node_ids(
        posterior_array, place_bin_center_ind_to_node
    )
    return step_distance_from_nodes(
        node_ids, has_valid_data, track_graph_with_bin_centers_edges
    )


def step_distance_from_nodes(
    node_ids: np.ndarray,
    has_valid_data: np.ndarray,
    track_graph_with_bin_centers_edges: nx.Graph,
) -> np.ndarray:
    """Backward-difference per-bin graph step distance from a MAP node sequence.

    Shared core of :func:`get_map_step_distance`; separated so callers that
    already hold a MAP node sequence (e.g. derived from a stored
    ``map_estimate`` rather than the full posterior) get bit-identical masking
    and alignment.

    Parameters
    ----------
    node_ids : np.ndarray, shape (n_time,)
        MAP track-graph node id per timebin. Invalid bins may hold any
        placeholder node; they are masked via ``has_valid_data``.
    has_valid_data : np.ndarray of bool, shape (n_time,)
        ``False`` where the bin had no valid MAP estimate.
    track_graph_with_bin_centers_edges : nx.Graph

    Returns
    -------
    np.ndarray, shape (n_time,)
        Track-graph distance (cm) arriving at each timebin; index 0 is 0.0.
    """
    n_time = len(node_ids)
    step_distance = np.zeros(n_time)
    if n_time < 2 or not np.any(has_valid_data):
        # No movement defined for a single bin, and an all-invalid sequence has
        # no real nodes — skip the (potentially disconnected) graph lookups.
        step_distance[~has_valid_data] = np.nan
        return step_distance

    step_distance[1:] = _forward_step_distances(
        node_ids, track_graph_with_bin_centers_edges
    )
    # A step is invalid if either endpoint bin had no valid MAP estimate.
    invalid_step = ~(has_valid_data[:-1] & has_valid_data[1:])
    step_distance[1:][invalid_step] = np.nan
    if not has_valid_data[0]:
        step_distance[0] = np.nan
    return step_distance


def map_estimate_to_map_nodes(
    map_estimate: np.ndarray,
    place_bin_centers_nodes_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """MAP track-graph node id per timebin from a stored ``map_estimate`` (cm).

    Recovers the decoder's argmax bin node from the *already stored* MAP linear
    position by nearest-neighbour match against the place-bin centers (the bin
    centers equal the decoder's ``map_estimate`` values exactly). This lets
    callers compute :func:`step_distance_from_nodes` from a saved
    ``map_estimate`` column without reloading the full posterior, and matches
    the posterior-argmax derivation in ``_map_node_ids`` whenever bin-center
    positions are distinct.

    Parameters
    ----------
    map_estimate : np.ndarray, shape (n_time,)
        MAP linear position (cm) per timebin; ``NaN`` where undecoded.
    place_bin_centers_nodes_df : pd.DataFrame
        ``env.place_bin_centers_nodes_df_`` with ``linear_position`` and
        ``node_id`` columns, ordered by place-bin index.

    Returns
    -------
    node_ids : np.ndarray, shape (n_time,)
        MAP node id per timebin (a real placeholder node where invalid). Never
        a junction placeholder node (``node_id == -1``).
    has_valid_data : np.ndarray of bool, shape (n_time,)
        ``False`` where ``map_estimate`` was ``NaN``.
    """
    # Exclude junction placeholder bins (node_id == -1, edge_id == -1
    # interpolation artifacts) from the candidate set so the nearest-neighbour
    # match never returns a node that is absent from the track graph
    # (junction-bin rule). The MAP itself never lands on these bins.
    real = np.asarray(place_bin_centers_nodes_df["node_id"]) != -1
    bin_lin = np.asarray(place_bin_centers_nodes_df["linear_position"], dtype=float)[
        real
    ]
    bin_node = np.asarray(place_bin_centers_nodes_df["node_id"])[real]
    order = np.argsort(bin_lin, kind="stable")
    sorted_lin = bin_lin[order]

    # reshape(-1) (not squeeze) so a single-timebin input stays 1-D rather than
    # collapsing to a 0-D scalar.
    me = np.asarray(map_estimate, dtype=float).reshape(-1)
    has_valid = np.isfinite(me)

    node_ids = np.full(me.shape, bin_node[order[0]])
    if np.any(has_valid):
        me_valid = me[has_valid]
        pos = np.clip(np.searchsorted(sorted_lin, me_valid), 1, len(sorted_lin) - 1)
        choose_left = (me_valid - sorted_lin[pos - 1]) <= (sorted_lin[pos] - me_valid)
        nearest_sorted = np.where(choose_left, pos - 1, pos)
        node_ids[has_valid] = bin_node[order[nearest_sorted]]
    return node_ids, has_valid


def track_distance_cm(
    from_linear_position: np.ndarray,
    to_linear_position: np.ndarray,
    place_bin_centers_nodes_df: pd.DataFrame,
    distance_between_nodes: dict,
) -> np.ndarray:
    """Track-graph distance between two linear positions, row by row (cm).

    Each position goes to its nearest real place bin (never a junction
    placeholder, :func:`map_estimate_to_map_nodes`), and the distance is the
    environment's shortest path between the two bins' nodes
    (``env.distance_between_nodes_``), as ``spatial_extent_cm`` and
    ``net_displacement_cm`` use. With the animal's position and the first
    decoded (MAP) position of an event, it is the event's start distance.

    Parameters
    ----------
    from_linear_position, to_linear_position : np.ndarray, shape (n,)
        Linear positions in cm; NaN where unknown.
    place_bin_centers_nodes_df : pd.DataFrame
        ``env.place_bin_centers_nodes_df_``.
    distance_between_nodes : dict
        ``env.distance_between_nodes_``: shortest-path cm, keyed by node then node.

    Returns
    -------
    distance : np.ndarray, shape (n,)
        NaN where either position is NaN.
    """
    a, a_ok = map_estimate_to_map_nodes(
        from_linear_position, place_bin_centers_nodes_df
    )
    b, b_ok = map_estimate_to_map_nodes(to_linear_position, place_bin_centers_nodes_df)
    out = np.full(a.shape, np.nan)
    for k in np.flatnonzero(a_ok & b_ok):
        out[k] = float(distance_between_nodes[a[k]][b[k]])
    return out


def normalized_replay_position_by_swr(
    data_over_time: pd.DataFrame,
) -> tuple[dict[str, pd.Series], int]:
    """The decoded position of non-local event bins as a fraction of the track.

    Each bin's ``map_estimate`` is divided by its session's
    ``max_linear_position``, so sessions with different track lengths share
    one [0, 1] axis. The bins are split by whether their event overlaps an SWR
    (:func:`~continuum_swr_replay.analysis.events.nonlocal_detection.nonlocal_event_bins_by_swr`).

    Parameters
    ----------
    data_over_time : pd.DataFrame
        A frame read through the SWR source, with ``map_estimate``,
        ``max_linear_position``, ``non_local_event``, ``SWR_event`` and the
        session columns.

    Returns
    -------
    positions : dict
        ``swr`` and ``non_swr``: the normalized position of each bin of the
        SWR-overlapping events and of the other events.
    n_bins : int
        The histogram bins that make one bin about one 1-D place bin
        (``PLACE_BIN_SIZE_1D``, cm) of the median session's track: the median
        over the frame's distinct ``max_linear_position`` values divided by
        the place-bin size, rounded down.
    """
    unique_positions = data_over_time["max_linear_position"].unique()
    median_track_length_cm = np.nanmedian(unique_positions[~np.isnan(unique_positions)])
    n_bins = int(median_track_length_cm / PLACE_BIN_SIZE_1D)

    swr_mask, non_swr_mask = nonlocal_event_bins_by_swr(data_over_time)
    positions = {
        word: data_over_time.loc[mask, "map_estimate"]
        / data_over_time.loc[mask, "max_linear_position"]
        for word, mask in (("swr", swr_mask), ("non_swr", non_swr_mask))
    }
    return positions, n_bins


def get_map_speed(
    posterior: xr.DataArray | np.ndarray,
    track_graph_with_bin_centers_edges: nx.Graph,
    place_bin_center_ind_to_node: np.ndarray,
    sampling_frequency: float = SAMPLING_FREQUENCY,
    smooth_sigma: float = 0.0025,
) -> np.ndarray:
    """Compute speed of the most likely decoded position.

    Uses forward differences of graph shortest-path distance between
    successive MAP positions. This correctly measures speed even when
    the decoded trajectory reverses direction, unlike central differences
    which underestimate speed at reversal points.

    Parameters
    ----------
    posterior : xr.DataArray | np.ndarray, shape (n_time, n_positions)
        Posterior probability distribution over position.
    track_graph_with_bin_centers_edges : nx.Graph
        Track graph with bin centers as nodes and edges representing connectivity.
    place_bin_center_ind_to_node : np.ndarray, shape (n_positions,)
        Mapping from place bin center index to node ID in the track graph.
    sampling_frequency : float, optional
        Samples per second, by default 500.0.
    smooth_sigma : float, optional
        Gaussian smoothing parameter (seconds), by default 0.0025.

    Returns
    -------
    np.ndarray, shape (n_time,)
        Speed (cm/s) of the most likely decoded position over time.

    Raises
    ------
    ValueError
        If sampling_frequency is not positive

    """
    if sampling_frequency <= 0:
        raise ValueError(
            f"sampling_frequency must be positive, got {sampling_frequency}"
        )

    dt = 1 / sampling_frequency
    posterior_array = np.asarray(posterior)

    # Handle all-NaN rows gracefully (np.nanargmax returns 0 with warning)
    node_ids, has_valid_data = _map_node_ids(
        posterior_array, place_bin_center_ind_to_node
    )
    if not np.any(has_valid_data):
        # All timepoints have NaN posteriors
        return np.full(len(posterior_array), np.nan)

    n_time = len(node_ids)
    if n_time == 1:
        return np.asarray([np.nan])

    # Forward differences: speed[t] = dist(node[t], node[t+1]) / dt
    # This correctly captures speed at reversal points, unlike central
    # differences which compute dist(node[t-1], node[t+1]) and
    # underestimate when the trajectory changes direction.
    step_distances = _forward_step_distances(
        node_ids, track_graph_with_bin_centers_edges
    )
    speed = step_distances / dt
    # Last timepoint uses backward difference (same as its predecessor)
    speed = np.append(speed, speed[-1])

    # Mask any speed whose forward-difference touched a placeholder
    # row. Each step ``i`` reads ``node[i] → node[i+1]``, so it must
    # be invalidated whenever EITHER endpoint had no valid posterior.
    # Without this, the rows immediately before and after an
    # all-NaN frame would carry the placeholder's contribution: e.g.
    # valid_node=1, NaN, valid_node=1 returned [10., NaN, 10.] even
    # though the two valid MAP positions are identical and the
    # motion was an artefact of node 0 being used as the placeholder.
    invalid_step = ~(has_valid_data[:-1] & has_valid_data[1:])
    speed[:-1][invalid_step] = np.nan
    # The last bin reuses the previous step (backward difference),
    # so it is invalid if either of its endpoints was invalid.
    if n_time >= 2 and not (has_valid_data[-1] and has_valid_data[-2]):
        speed[-1] = np.nan
    speed[~has_valid_data] = np.nan

    # Snapshot the full set of "speed must stay NaN" positions BEFORE
    # interpolation so we can re-apply it after the smoother runs.
    # This is the union of "step touched a placeholder row" and
    # "this row's posterior was itself invalid". Re-masking only the
    # latter (``~has_valid_data``) leaves the interpolated speeds
    # for invalidated steps next to a NaN frame in place — e.g.
    # MAP nodes ``[0, 1, NaN, 1, 1]`` ended up with a finite
    # 6.667 cm/s at index 1 even though that step's destination
    # was the all-NaN placeholder.
    invalid_speed_mask = np.isnan(speed)

    # Interpolate over NaN gaps so the smoother has continuous data
    # to convolve with. The interpolated values are temporary —
    # re-masked below.
    valid_mask = ~np.isnan(speed)
    if valid_mask.any() and not valid_mask.all():
        indices = np.arange(n_time)
        speed_filled = np.interp(indices, indices[valid_mask], speed[valid_mask])
    else:
        speed_filled = speed.copy()

    # Apply Gaussian smoothing
    smoothed_speed: np.ndarray = gaussian_smooth(
        speed_filled, smooth_sigma, sampling_frequency
    )

    # Re-apply the full pre-interpolation invalid mask so step-level
    # invalidations next to a NaN frame do not survive the
    # interpolate-smooth round trip.
    smoothed_speed[invalid_speed_mask] = np.nan

    return smoothed_speed


def get_posterior_stats(
    data: SessionData,
    env: Environment,
    results: xr.Dataset | None = None,
    sampling_frequency: int = SAMPLING_FREQUENCY,
    orientation_variable: Literal[
        "head_orientation", "body_orientation"
    ] = "head_orientation",
    posterior: xr.DataArray | None = None,
) -> tuple[
    xr.DataArray,
    np.ndarray,
    xr.DataArray,
    np.ndarray,
    float,
    float,
    float,
    xr.DataArray,
]:
    """Compute comprehensive statistics from decoded posterior.

    Parameters
    ----------
    data : SessionData
        Data dictionary containing position info, track graph, etc.
    env : Environment
        Environment configuration from detector
    results : xr.Dataset | None, optional
        Decoding results, by default None (required if posterior is None)
    sampling_frequency : int, optional
        Samples per second, by default 500
    orientation_variable : str, optional
        Name of orientation variable in position_info, by default "head_orientation"
    posterior : xr.DataArray | None, optional
        Pre-computed posterior, by default None (will be computed from results)

    Returns
    -------
    map_estimate : xr.DataArray
        Maximum a posteriori position estimate
    map_speed : np.ndarray
        Speed of MAP position
    ahead_behind_distance : xr.DataArray
        Signed distance of decoded position relative to actual position
    hpd : np.ndarray
        Highest posterior density spatial coverage
    max_coverage : float
        Maximum possible spatial coverage (total track length)
    max_distance : float
        Maximum distance for normalization
    max_linear_position : float
        Maximum linear position observed
    posterior : xr.DataArray
        Posterior distribution (returned for reuse)

    """
    smooth_sigma = 0.0025

    if posterior is None:
        if results is None:
            raise ValueError("Either results or posterior must be provided")
        posterior = get_conditional_non_local_posterior(results)

    trajectory_data = get_trajectory_data(
        posterior=posterior,
        track_graph=data["track_graph"],
        decoder=env,
        actual_projected_position=data["position_info"][
            ["projected_x_position", "projected_y_position"]
        ],
        track_segment_id=data["position_info"].track_segment_id,
        actual_orientation=data["position_info"][orientation_variable],
    )
    max_distance = float(
        np.max(np.asarray(data["position_info"].linear_position))
        - np.sum(env.edge_spacing)
    )
    max_linear_position = np.asarray(data["position_info"].linear_position).max()
    ahead_behind_distance = get_ahead_behind_distance(
        data["track_graph"], *trajectory_data
    )
    map_estimate = maximum_a_posteriori_estimate(posterior)
    map_speed = get_map_speed(
        posterior.values,
        env.track_graph_with_bin_centers_edges_,
        np.asarray(env.place_bin_centers_nodes_df_.node_id),
        sampling_frequency,
        smooth_sigma,
    )
    hpd = get_HPD_spatial_coverage(
        posterior, get_highest_posterior_threshold(posterior, coverage=0.95)
    )
    max_coverage = np.sum(
        np.diff(env.place_bin_edges_.squeeze())[env.is_track_interior_]
    )

    return (
        map_estimate,
        map_speed,
        ahead_behind_distance,
        hpd,
        max_coverage,
        max_distance,
        max_linear_position,
        posterior,
    )
