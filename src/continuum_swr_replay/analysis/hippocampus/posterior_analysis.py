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

from typing import Literal

import networkx as nx
import numpy as np
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

from ...types import SessionData


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
    acausal_posterior = (
        results.acausal_posterior.sel(state="Non-Local Continuous")
        + results.acausal_posterior.sel(state="Non-Local Fragmented")
    )

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


def get_map_speed(
    posterior: xr.DataArray | np.ndarray,
    track_graph_with_bin_centers_edges: nx.Graph,
    place_bin_center_ind_to_node: np.ndarray,
    sampling_frequency: float = 500.0,
    smooth_sigma: float = 0.0025,
) -> np.ndarray:
    """Compute speed of the most likely decoded position.

    Parameters
    ----------
    posterior : xr.DataArray | np.ndarray
        Posterior probability distribution over position
    track_graph_with_bin_centers_edges : nx.Graph
        Track graph with bin centers as nodes and edges representing connectivity
    place_bin_center_ind_to_node : np.ndarray
        Mapping from place bin center index to node ID
    sampling_frequency : float, optional
        Samples per second, by default 500.0
    smooth_sigma : float, optional
        Gaussian smoothing parameter (seconds), by default 0.0025

    Returns
    -------
    map_speed : np.ndarray
        Speed (cm/s) of the most likely decoded position over time

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
    has_valid_data = ~np.all(np.isnan(posterior_array), axis=1)
    if not np.any(has_valid_data):
        # All timepoints have NaN posteriors
        return np.full(len(posterior_array), np.nan)

    # Compute argmax, using 0 as placeholder for invalid rows
    map_position_ind = np.zeros(len(posterior_array), dtype=int)
    map_position_ind[has_valid_data] = np.nanargmax(
        posterior_array[has_valid_data], axis=1
    )

    node_ids = place_bin_center_ind_to_node[map_position_ind]
    n_time = len(node_ids)

    if n_time == 1:
        return np.asarray([np.nan])
    elif n_time == 2:
        speed = (
            np.array(
                [
                    nx.shortest_path_length(
                        track_graph_with_bin_centers_edges,
                        source=node_ids[0],
                        target=node_ids[1],
                        weight="distance",
                    ),
                    nx.shortest_path_length(
                        track_graph_with_bin_centers_edges,
                        source=node_ids[-2],
                        target=node_ids[-1],
                        weight="distance",
                    ),
                ]
            )
            / dt
        )
    else:
        shortest_path_lengths = dict(
            nx.shortest_path_length(
                track_graph_with_bin_centers_edges, weight="distance"
            )
        )
        speed = np.array(
            [
                shortest_path_lengths[node1][node2]
                for node1, node2 in zip(node_ids[:-2], node_ids[2:], strict=True)
            ]
        ) / (2.0 * dt)
        speed = np.insert(
            speed,
            0,
            nx.shortest_path_length(
                track_graph_with_bin_centers_edges,
                source=node_ids[0],
                target=node_ids[1],
                weight="distance",
            )
            / dt,
        )
        speed = np.append(
            speed,
            nx.shortest_path_length(
                track_graph_with_bin_centers_edges,
                source=node_ids[-2],
                target=node_ids[-1],
                weight="distance",
            )
            / dt,
        )
    # Apply Gaussian smoothing
    smoothed_speed: np.ndarray = gaussian_smooth(np.abs(speed), smooth_sigma, sampling_frequency)

    # Set speed to NaN for timepoints with invalid posteriors
    smoothed_speed[~has_valid_data] = np.nan

    return smoothed_speed


def get_posterior_stats(
    data: SessionData,
    env: Environment,
    results: xr.Dataset | None = None,
    sampling_frequency: int = 500,
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
