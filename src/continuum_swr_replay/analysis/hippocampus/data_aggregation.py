"""Aggregation of analysis results over time."""

import pandas as pd
import xarray as xr
from non_local_detector.environment import Environment
from scipy.stats import zscore

from continuum_swr_replay.data_loaders import get_labels

from .event_detection import get_confident_times, get_overlap_times
from .posterior_analysis import get_conditional_non_local_posterior, get_posterior_stats
from .types import SessionData

# Maximum speed threshold for filtering non-local events
# Events with speeds above this are likely not genuine replay events
MAX_SPEED_THRESHOLD_CM_PER_SEC = 4.0


def get_data_over_time(
    data: SessionData,
    results: xr.Dataset,
    env: Environment,
    nwb_file_name: str,
    epoch_name: str,
    brain_area: str = "HPC",
    sampling_frequency: int = 500,
    non_local_threshold: float = 0.8,
) -> tuple[pd.DataFrame, xr.DataArray]:
    """Aggregate all analysis data into time-indexed DataFrame.

    Combines decoding results, event detection, posterior statistics,
    and behavioral data into a single DataFrame aligned by time.

    Parameters
    ----------
    data : SessionData
        Data dictionary from load_data() containing position, spikes, etc.
    results : xr.Dataset
        Decoding results
    env : Environment
        Environment configuration from detector
    nwb_file_name : str
        Name of the NWB file (e.g., "chimi20200212_.nwb")
    epoch_name : str
        Name of the epoch (e.g., "02_r1")
    brain_area : str, optional
        Brain area for analysis, by default "HPC"
    sampling_frequency : int, optional
        Samples per second, by default 500
    non_local_threshold : float, optional
        Probability threshold for non-local classification, by default 0.8

    Returns
    -------
    data_over_time : pd.DataFrame
        Time-indexed DataFrame containing:
        - Event labels (non_local_event, SWR_event, MUA_event)
        - State probabilities (Local_prob, No-Spike_prob, etc.)
        - Posterior statistics (map_estimate, map_speed, ahead_behind_distance, hpd)
        - Neural signals (theta_phase, theta_power_zscore, SWR_zscore, MUA_zscore)
        - Event counts (n_non_local, n_SWR, n_MUA, etc.)
        - Session metadata (animal, day, epoch, nwb_file_name)
        - Position and behavioral variables (from position_info)
    posterior : xr.DataArray
        Conditional non-local posterior distribution
    """
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
    non_local_times = non_local_times[
        non_local_times.max_speed < MAX_SPEED_THRESHOLD_CM_PER_SEC
    ]
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
