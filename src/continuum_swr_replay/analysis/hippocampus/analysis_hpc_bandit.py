from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
import xarray as xr
from non_local_detector import NonLocalClusterlessDetector
from non_local_detector.analysis import (
    get_ahead_behind_distance,
    get_map_speed,
    get_trajectory_data,
    maximum_a_posteriori_estimate,
)
from non_local_detector.model_checking.highest_posterior_density import (
    get_highest_posterior_threshold,
    get_HPD_spatial_coverage,
)
from ripple_detection.core import gaussian_smooth, segment_boolean_series
from scipy.stats import zscore
from continuum_swr_replay.data_loaders import get_labels
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR


def get_conditional_non_local_posterior(results):
    acausal_posterior = results.acausal_posterior.sel(state="Non-Local Continuous")
    acausal_posterior += results.acausal_posterior.sel(state="Non-Local Fragmented")

    return acausal_posterior / acausal_posterior.sum("position")


def load_results(
    nwb_file_name, epoch_name, decoding_brain_area="HPC", load_detector=True
):
    epoch_identifier = f"{nwb_file_name.replace('_.nwb', '')}_{epoch_name}"

    results = xr.open_dataset(
        PROCESSED_DATA_DIR / f"{epoch_identifier}_{decoding_brain_area}_results.nc"
    )
    state_bin_coord_names = list(results["state_bins"].coords)
    results = results.set_index(state_bins=state_bin_coord_names)
    if load_detector:
        detector = NonLocalClusterlessDetector.load_model(
            PROCESSED_DATA_DIR
            / f"{epoch_identifier}_{decoding_brain_area}_detector.pkl"
        )
        env = detector.environments[0]
    else:
        detector = None
        env = None

    return results, detector, env


def get_confident_times(
    prob: xr.DataArray,
    speed: Optional[np.ndarray] = None,
    time: Optional[np.ndarray] = None,
    prob_threshold: float = 0.8,
    minimum_duration: float = 0.015,
) -> pd.DataFrame:
    """_summary_

    Parameters
    ----------
    prob : xr.DataArray
        Probability of state
    speed : np.ndarray | None, optional
        Speed of animal, by default None
    time : np.ndarray | None, optional
        Time of each sample, by default None
    prob_threshold : float, optional
        Threshold for confident state, by default 0.8
    minimum_duration : float, optional
        Minimum duration of confident state, by default 0.015

    Returns
    -------
    times : pd.DataFrame
    """
    confident_times = np.asarray(
        segment_boolean_series(
            (prob > prob_threshold).to_pandas(),
            minimum_duration=minimum_duration,
        )
    )
    if len(confident_times) == 0:
        confident_times = np.full((0, 2), np.nan)

    max_speed = np.array(
        [
            np.nanmax(speed[slice(*np.digitize(time_range, time[1:-1]))])
            for time_range in confident_times
        ]
    )

    n_times = len(confident_times)
    index = np.arange(n_times) + 1 if n_times > 0 else []
    index = pd.Index(index, name="event_number")
    return pd.DataFrame(
        {
            "start_time": confident_times[:, 0],
            "end_time": confident_times[:, 1],
            "duration": confident_times[:, 1] - confident_times[:, 0],
            "max_speed": max_speed,
        },
        index=index,
    )


def get_map_speed(
    posterior: xr.DataArray,
    track_graph_with_bin_centers_edges: nx.Graph,
    place_bin_center_ind_to_node: np.ndarray,
    sampling_frequency: float = 500.0,
    smooth_sigma: float = 0.0025,
) -> np.ndarray:
    """Get the speed of the most likely decoded position.

    Parameters
    ----------
    posterior : xr.DataArray
    track_graph_with_bin_centers_edges : nx.Graph
        Track graph with bin centers as nodes and edges
    place_bin_center_ind_to_node : np.ndarray
        Mapping of place bin center index to node ID
    sampling_frequency : float, optional
        Samples per second, by default 500.0
    smooth_sigma : float, optional
        Gaussian smoothing parameter, by default 0.0025

    Returns
    -------
    map_speed : np.ndarray
        Speed of the most likely decoded position.
    """
    dt = 1 / sampling_frequency
    posterior = np.asarray(posterior)
    map_position_ind = np.nanargmax(posterior, axis=1)
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
                for node1, node2 in zip(node_ids[:-2], node_ids[2:])
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
        speed = np.insert(
            speed,
            -1,
            nx.shortest_path_length(
                track_graph_with_bin_centers_edges,
                source=node_ids[-2],
                target=node_ids[-1],
                weight="distance",
            )
            / dt,
        )
    return gaussian_smooth(np.abs(speed), smooth_sigma, sampling_frequency)


def get_overlap_times(
    non_local_times: pd.DataFrame, ripple_times: pd.DataFrame, results: xr.Dataset
) -> tuple[pd.DataFrame, pd.DataFrame]:
    non_local_labels = get_labels(non_local_times, results.time)
    ripple_labels = get_labels(ripple_times, results.time)

    overlap_non_local = non_local_labels.loc[
        ripple_labels.event_number > 0
    ].event_number.unique()
    overlap_non_local = overlap_non_local[overlap_non_local > 0]

    non_local_ind = np.arange(len(non_local_times))

    overlap_non_local_times = non_local_times.iloc[
        non_local_ind[np.isin(non_local_ind, overlap_non_local - 1)]
    ]

    no_overlap_non_local_times = non_local_times.iloc[
        non_local_ind[~np.isin(non_local_ind, overlap_non_local - 1)]
    ]

    return overlap_non_local_times, no_overlap_non_local_times


def get_posterior_stats(
    data,
    env,
    results=None,
    sampling_frequency=500,
    orientation_variable="head_orientation",
    posterior=None,
):
    SMOOTH_SIGMA = 0.0025

    if posterior is None:
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
        SMOOTH_SIGMA,
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


def get_data_over_time(
    data,
    results,
    env,
    nwb_file_name,
    epoch_name,
    brain_area="HPC",
    sampling_frequency=500,
    non_local_threshold=0.8,
):

    non_local_prob = results.acausal_state_probabilities.sel(
        states="Non-Local Continuous"
    ) + results.acausal_state_probabilities.sel(states="Non-Local Fragmented")

    posterior = get_conditional_non_local_posterior(results)

    non_local_times = get_confident_times(
        non_local_prob,
        data["position_info"].head_speed.to_numpy(),
        data["position_info"].index.to_numpy(),
        non_local_threshold,
    )
    non_local_times = non_local_times[non_local_times.max_speed < 4.0]
    non_local_labels = get_labels(non_local_times, results.time)
    SWR_labels = get_labels(data["ripple_times"], results.time)
    MUA_labels = get_labels(data["hse_times"][brain_area], results.time)

    non_local_SWR_times, non_local_no_SWR_times = get_overlap_times(
        non_local_times, data["ripple_times"], results
    )
    non_local_MUA_times, non_local_no_MUA_times = get_overlap_times(
        non_local_times, data["hse_times"][brain_area], results
    )

    SWR_MUA_times, SWR_no_MUA_times = get_overlap_times(
        data["ripple_times"], data["hse_times"][brain_area], results
    )
    (
        map_estimate,
        map_speed,
        ahead_behind_distance,
        hpd,
        max_coverage,
        max_distance,
        max_linear_position,
        posterior,
    ) = get_posterior_stats(
        data, env, sampling_frequency=sampling_frequency, posterior=posterior
    )
    animal = nwb_file_name.replace("_.nwb", "")[:-8]
    date = int(nwb_file_name.replace("_.nwb", "")[-8:])

    data_over_time = pd.DataFrame(
        {
            "non_local_event": non_local_labels.squeeze(),
            "SWR_event": SWR_labels.squeeze(),
            "MUA_event": MUA_labels.squeeze(),
            "Local_prob": results.sel(
                states="Local"
            ).acausal_state_probabilities.squeeze(),
            "No-Spike_prob": results.sel(
                states="No-Spike"
            ).acausal_state_probabilities.squeeze(),
            "Non-Local_Continuous_prob": results.sel(
                states="Non-Local Continuous"
            ).acausal_state_probabilities.squeeze(),
            "Non-Local_Fragmented_prob": results.sel(
                states="Non-Local Fragmented"
            ).acausal_state_probabilities.squeeze(),
            "map_estimate": map_estimate.squeeze(),
            "map_speed": map_speed.squeeze(),
            "ahead_behind_distance": ahead_behind_distance.squeeze(),
            "hpd": hpd.squeeze(),
            "max_coverage": max_coverage,
            "max_distance": max_distance,
            "max_linear_position": max_linear_position,
            "theta_phase": data["theta_phase"]
            .reindex(results.time, method="nearest")
            .squeeze(),
            "theta_power_zscore": zscore(data["theta_power"].squeeze())
            .reindex(results.time, method="nearest")
            .squeeze(),
            "SWR_zscore": data["zscored_ripple_consensus_trace"]
            .reindex(results.time, method="nearest")
            .squeeze(),
            "MUA_zscore": data["multiunit_rate_zscore"][brain_area].squeeze(),
            "n_non_local": len(non_local_times),
            "n_SWR": len(data["ripple_times"]),
            "n_MUA": len(data["hse_times"][brain_area]),
            "n_non_local_SWR": len(non_local_SWR_times),
            "n_non_local_no_SWR": len(non_local_no_SWR_times),
            "n_non_local_MUA": len(non_local_MUA_times),
            "n_non_local_no_MUA": len(non_local_no_MUA_times),
            "n_SWR_MUA": len(SWR_MUA_times),
            "n_SWR_no_MUA": len(SWR_no_MUA_times),
            "animal": animal,
            "day": date,
            "epoch": epoch_name,
            "nwb_file_name": nwb_file_name,
        }
    )

    return pd.concat((data_over_time, data["position_info"]), axis=1), posterior
