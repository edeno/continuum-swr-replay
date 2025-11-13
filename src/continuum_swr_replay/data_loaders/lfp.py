"""LFP, ripple, theta, and multiunit data loading."""

import warnings

import numpy as np
import pandas as pd
from ripple_detection import (
    get_multiunit_population_firing_rate,
    multiunit_HSE_detector,
)
from scipy.stats import zscore
from spyglass.lfp.analysis.v1.lfp_band import LFPBandSelection, LFPBandV1
from spyglass.lfp.lfp_merge import LFPOutput
from spyglass.ripple.v1 import RippleTimesV1

# Ignore warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pynwb")
warnings.filterwarnings("ignore", category=UserWarning, module="hdmf")
warnings.filterwarnings("ignore", category=UserWarning, module="datajoint")


def get_theta(nwb_file_name: str, epoch_name: str) -> dict:
    """Load theta-band LFP data.

    Parameters
    ----------
    nwb_file_name : str
        NWB file name
    epoch_name : str
        Epoch name

    Returns
    -------
    dict
        Dictionary containing:
        - theta_filtered_lfp: pd.DataFrame with filtered LFP
        - theta_phase: pd.Series with theta phase
        - theta_power: pd.Series with theta power
    """
    theta_lfp_key = {
        "nwb_file_name": nwb_file_name,
        "filter_name": "theta_5_11",
        "filter_sampling_rate": 1000,
        "target_interval_list_name": epoch_name + " noPrePostTrialTimes",
    }
    theta_filtered_lfp = (LFPBandV1 & theta_lfp_key).fetch1_dataframe()
    # Pick the first reference electrode
    electrode_list = [
        (LFPBandV1 & theta_lfp_key).fetch_nwb()[0]["lfp_band"].electrodes.data[0]
    ]
    theta_phase = (LFPBandV1 & theta_lfp_key).compute_signal_phase(
        electrode_list=electrode_list
    )
    theta_power = (LFPBandV1 & theta_lfp_key).compute_signal_power(
        electrode_list=electrode_list
    )
    return {
        "theta_filtered_lfp": theta_filtered_lfp,
        "theta_phase": theta_phase,
        "theta_power": theta_power,
    }


def get_ripple(nwb_file_name: str, epoch_name: str, ripple_detector_name: str) -> dict:
    """Load ripple detection data.

    Parameters
    ----------
    nwb_file_name : str
        NWB file name
    epoch_name : str
        Epoch name
    ripple_detector_name : str
        Ripple detector name (currently only "Kay" is supported)

    Returns
    -------
    dict
        Dictionary containing:
        - ripple_times: pd.DataFrame with ripple start/end times
        - ripple_filtered_lfps: pd.DataFrame with ripple-band filtered LFP
        - ripple_lfps: pd.DataFrame with raw ripple LFP
        - ripple_consensus_trace: pd.Series with ripple consensus trace
        - zscored_ripple_consensus_trace: pd.Series with z-scored consensus trace
    """
    # Ripples
    if ripple_detector_name == "Kay":
        ripple_params = "default"
    else:
        raise ValueError("Invalid ripple detector name")

    ripple_key = {
        "nwb_file_name": nwb_file_name,
        "target_interval_list_name": epoch_name + " noPrePostTrialTimes",
        "group_name": "CA1",
        "ripple_params": ripple_params,
        "filter_name": "ripple_150_250",
        "filter_sampling_rate": 1000,
    }
    lfp_band_restriction = {
        "nwb_file_name": nwb_file_name,
        "filter_name": ripple_key["filter_name"],
        "target_interval_list_name": ripple_key["target_interval_list_name"],
        "lfp_band_sampling_rate": ripple_key["filter_sampling_rate"],
    }
    lfp_band_key = (LFPBandV1 & lfp_band_restriction).fetch1("KEY")
    ripple_times = (RippleTimesV1 & ripple_key).fetch1_dataframe()
    ripple_filtered_lfp = (LFPBandV1 & lfp_band_key).fetch1_dataframe()
    lfp_merge_id = (LFPBandSelection & ripple_key).fetch1("lfp_merge_id")
    lfp_df = (LFPOutput & {"merge_id": lfp_merge_id}).fetch1_dataframe()

    ripple_consensus_trace = RippleTimesV1.get_Kay_ripple_consensus_trace(
        ripple_filtered_lfp, sampling_frequency=ripple_key["filter_sampling_rate"]
    )
    zscored_ripple_consensus_trace = zscore(ripple_consensus_trace, nan_policy="omit")

    return {
        "ripple_times": ripple_times,
        "ripple_filtered_lfps": ripple_filtered_lfp,
        "ripple_lfps": lfp_df,
        "ripple_consensus_trace": ripple_consensus_trace,
        "zscored_ripple_consensus_trace": zscored_ripple_consensus_trace,
    }


def get_multiunit_HSE(
    spike_times: dict, speed: np.ndarray, bin_time: np.ndarray
) -> tuple:
    """Compute multiunit high synchrony events (HSE).

    Parameters
    ----------
    spike_times : dict
        Dictionary mapping brain area to list of spike time arrays
    speed : np.ndarray
        Speed of the animal
    bin_time : np.ndarray
        Time bins for spike counting

    Returns
    -------
    tuple
        - hse_times: pd.DataFrame with HSE start/end times
        - multiunit_firing_rate: pd.DataFrame with population firing rate
        - multiunit_rate_zscore: pd.DataFrame with z-scored firing rate
    """
    sampling_frequency = 1 / np.median(np.diff(bin_time))
    start_time, end_time = bin_time[[0, -1]]
    multiunit_spikes = np.stack(
        [
            np.bincount(
                np.digitize(
                    spike_times[
                        np.logical_and(
                            spike_times >= start_time, spike_times <= end_time
                        )
                    ],
                    bin_time[1:-1],
                ),
                minlength=bin_time.shape[0],
            )
            for spike_times in spike_times
        ],
        axis=1,
    )

    multiunit_firing_rate = pd.DataFrame(
        get_multiunit_population_firing_rate(multiunit_spikes, sampling_frequency),
        index=bin_time,
        columns=["firing_rate"],
    )
    mean = np.nanmean(multiunit_firing_rate)
    std = np.nanstd(multiunit_firing_rate)

    # Avoid division by zero if firing rate is constant
    if std == 0 or np.isnan(std):
        multiunit_rate_zscore = pd.DataFrame(
            np.zeros_like(multiunit_firing_rate),
            index=bin_time,
            columns=["firing_rate"]
        )
    else:
        multiunit_rate_zscore = (multiunit_firing_rate - mean) / std

    hse_times = multiunit_HSE_detector(
        bin_time,
        multiunit_spikes,
        speed,
        sampling_frequency,
        speed_threshold=4.0,
        minimum_duration=0.015,
        zscore_threshold=2.0,
        close_event_threshold=0.000,
        use_speed_threshold_for_zscore=False,
    )

    return hse_times, multiunit_firing_rate, multiunit_rate_zscore


def get_multiunit(spike_times: dict, position_info: pd.DataFrame) -> dict:
    """Get multiunit HSE data for all brain areas.

    Parameters
    ----------
    spike_times : dict
        Dictionary mapping brain area to list of spike time arrays
    position_info : pd.DataFrame
        Position info with head_speed column

    Returns
    -------
    dict
        Dictionary containing:
        - hse_times: dict mapping brain area to HSE times
        - multiunit_firing_rate: dict mapping brain area to firing rates
        - multiunit_rate_zscore: dict mapping brain area to z-scored rates
    """
    # Get multiunit HSE
    hse_times = {}
    multiunit_firing_rate = {}
    multiunit_rate_zscore = {}

    for brain_area in spike_times:
        (
            hse_times[brain_area],
            multiunit_firing_rate[brain_area],
            multiunit_rate_zscore[brain_area],
        ) = get_multiunit_HSE(
            spike_times[brain_area],
            position_info.head_speed.to_numpy(),
            position_info.index.to_numpy(),
        )

    return {
        "hse_times": hse_times,
        "multiunit_firing_rate": multiunit_firing_rate,
        "multiunit_rate_zscore": multiunit_rate_zscore,
    }


def filter_ripple_lfp_time(ripple_data: dict, position_time: np.ndarray) -> None:
    """Filter ripple LFP data to match position time bounds (modifies in place).

    Parameters
    ----------
    ripple_data : dict
        Dictionary with ripple data
    position_time : np.ndarray
        Position time array (defines time bounds)
    """
    position_time_slice = slice(position_time[0], position_time[-1])
    ripple_data["ripple_filtered_lfps"] = ripple_data["ripple_filtered_lfps"].loc[
        position_time_slice
    ]
    ripple_data["ripple_lfps"] = ripple_data["ripple_lfps"].loc[position_time_slice]
    ripple_data["ripple_consensus_trace"] = ripple_data["ripple_consensus_trace"].loc[
        position_time_slice
    ]
    ripple_data["zscored_ripple_consensus_trace"] = ripple_data[
        "zscored_ripple_consensus_trace"
    ].loc[position_time_slice]
