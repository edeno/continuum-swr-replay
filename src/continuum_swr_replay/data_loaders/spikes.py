"""Spike data loading and processing.

This module handles loading neural spiking data from different brain regions
and processing it for decoding analysis. Supports both clusterless (waveform-based)
and sorted spike data.

Brain Regions
-------------
- **HPC (Hippocampus CA1)**: Clusterless thresholding with waveform features
- **mPFC (Medial Prefrontal Cortex)**: MountainSort4 sorted units
- **OFC (Orbitofrontal Cortex)**: MountainSort4 sorted units

Functions
---------
detect_coincident_spikes
    Identify and remove artifact spikes that fire simultaneously across tetrodes.
get_electrode_group_info
    Extract electrode group metadata from NWB file.
get_hpc_marks
    Load clusterless marks (spike times + waveform features) for hippocampus.
get_pfc_spike_times
    Load sorted unit spike times for prefrontal cortex areas.
get_spike_data
    Main function to load spike data from all available brain regions.
filter_spike_times
    Restrict spike times to match position data time bounds.

Notes
-----
Clusterless decoding for HPC uses spike waveform features rather than
cluster assignments. This captures multiunit activity without the need
for spike sorting.

Coincident spike detection removes electrical artifacts where >33% of
tetrodes fire within 40 microseconds - timing unlikely for genuine neural
activity.

See Also
--------
load_data : Main data loading orchestrator
position : Position data that defines time bounds for spike filtering
"""

from __future__ import annotations

import itertools
import warnings
from typing import TypedDict

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.ndimage import label
from spyglass.common import Nwbfile
from spyglass.decoding.v0.clusterless import UnitMarks
from spyglass.spikesorting.v0 import CuratedSpikeSorting, SortGroup
from spyglass.utils.nwb_helper_fn import get_nwb_file

# Ignore warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pynwb")
warnings.filterwarnings("ignore", category=UserWarning, module="hdmf")
warnings.filterwarnings("ignore", category=UserWarning, module="datajoint")


class SpikeDataDict(TypedDict):
    """Type definition for spike data dictionary returned by get_spike_data."""

    spike_times: dict[str, list[NDArray[np.float64]]]
    spike_waveform_features: dict[str, list[NDArray[np.float64]]]


def detect_coincident_spikes(
    spike_times: list[NDArray[np.float64]],
    spike_closeness_threshold: float = 0.00004,
    max_coincident_fraction: float = 0.33,
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.int_]]]:
    """Detect and remove coincident spikes across tetrodes.

    Coincident spikes (within 40μs across >33% of tetrodes) are likely artifacts.

    Parameters
    ----------
    spike_times : list
        List of arrays, each containing spike times for one tetrode
    spike_closeness_threshold : float, optional
        Time threshold for considering spikes coincident (seconds)
    max_coincident_fraction : float, optional
        Fraction of tetrodes that must fire together to be considered artifact

    Returns
    -------
    tuple
        - filtered_spike_times: list of arrays with coincident spikes removed
        - filtered_time_bin_ind: list of arrays with indices of kept spikes
    """
    if not spike_times or all(len(st) == 0 for st in spike_times):
        return [], []

    concat_spike_times = np.concatenate(spike_times)
    sort_group_id = np.concatenate(
        [
            np.ones(len(spike_time), dtype=int) * i
            for i, spike_time in enumerate(spike_times)
        ]
    )
    time_bin_ind = np.concatenate(
        [np.arange(len(spike_time), dtype=int) for spike_time in spike_times]
    )

    sort_ind = np.argsort(concat_spike_times)
    sorted_spike_times = concat_spike_times[sort_ind]
    sort_group_id = sort_group_id[sort_ind]
    time_bin_ind = time_bin_ind[sort_ind]

    is_close = np.diff(sorted_spike_times) < spike_closeness_threshold
    is_close = np.concatenate([[False], is_close])
    labels, _ = label(is_close)

    df = pd.DataFrame(
        {
            "labels": labels,
            "sort_group_id": sort_group_id,
            "time_bin_ind": time_bin_ind,
            "spike_times": sorted_spike_times,
        },
    )
    # Calculate coincident fraction to identify artifacts
    n_sort_groups = len(spike_times)
    coincident_fraction = (
        df.loc[df.labels > 0]
        .groupby("labels")
        .sort_group_id.nunique()
        / n_sort_groups
    )
    artifact_labels = coincident_fraction[
        coincident_fraction > max_coincident_fraction
    ]

    df = df.loc[~df.labels.isin(artifact_labels.index)]

    filtered_spike_times = (
        df.groupby("sort_group_id").spike_times.apply(np.array).tolist()
    )
    filtered_time_bin_ind = (
        df.groupby("sort_group_id").time_bin_ind.apply(np.array).tolist()
    )

    return filtered_spike_times, filtered_time_bin_ind


def get_electrode_group_info(nwb_file_name: str) -> pd.DataFrame:
    """Get electrode group information from NWB file.

    Parameters
    ----------
    nwb_file_name : str
        NWB file name

    Returns
    -------
    pd.DataFrame
        Electrode group information with targeted location labels
    """
    nwb_file_abspath = Nwbfile.get_abs_path(nwb_file_name)
    nwb_file = get_nwb_file(nwb_file_abspath)
    electrode_group_df = []
    for electrode_group in nwb_file.electrode_groups.values():
        group_info = {
            "electrode_group_name": electrode_group.name,
            "description": electrode_group.description,
        }
        group_info.update(electrode_group.fields)
        electrode_group_df.append(group_info)

    electrode_group_df = (
        pd.DataFrame(electrode_group_df)
        .drop(columns=["device"])
        .set_index("electrode_group_name")
    )

    is_CA1 = electrode_group_df.targeted_location.str.contains("CA1") & (
        electrode_group_df.location != "CorpusCallosum"
    )
    electrode_group_df.loc[is_CA1, "targeted_location"] = "CA1"

    is_mPFC = electrode_group_df.targeted_location.str.contains("mPFC") & (
        electrode_group_df.location != "CorpusCallosum"
    )
    electrode_group_df.loc[is_mPFC, "targeted_location"] = "mPFC"

    is_OFC = electrode_group_df.targeted_location.str.contains("OFC") & (
        electrode_group_df.location != "CorpusCallosum"
    )
    electrode_group_df.loc[is_OFC, "targeted_location"] = "OFC"

    return electrode_group_df


def get_hpc_marks(
    nwb_file_name: str,
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
    """Get hippocampus clusterless marks (spike times and waveform features).

    Parameters
    ----------
    nwb_file_name : str
        NWB file name

    Returns
    -------
    tuple
        - spike_times: list of arrays with spike times per tetrode
        - spike_waveform_features: list of arrays with waveform features
    """
    try:
        restriction = {
            "nwb_file_name": nwb_file_name,
            "curation_id": 0,
            "sort_interval_name": "runs_noPrePostTrialTimes raw data valid times",
            "preproc_params_name": "franklab_tetrode_hippocampus",
            "team_name": "ac_em_xs",
            "sorter": "clusterless_thresholder",
            "sorter_params_name": "clusterless_fixed",
            "mark_param_name": "default",
        }
        marks = (UnitMarks & restriction).fetch_dataframe()
        marks = [(mark.index.to_numpy(), mark.to_numpy()) for mark in marks]
        spike_times_tuple, spike_waveform_features_tuple = zip(*marks, strict=False)
        spike_times = list(spike_times_tuple)
        spike_waveform_features = list(spike_waveform_features_tuple)
    except ValueError:
        restriction = {
            "nwb_file_name": nwb_file_name,
            "curation_id": 0,
            "sort_interval_name": "runs_noPrePostTrialTimes raw data valid times",
            "preproc_params_name": "franklab_tetrode_hippocampus",
            "team_name": "ac_em_xs",
            "sorter": "clusterless_thresholder",
            "sorter_params_name": "default_clusterless",
            "mark_param_name": "default",
        }
        marks = (UnitMarks & restriction).fetch_dataframe()
        marks = [(mark.index.to_numpy(), mark.to_numpy()) for mark in marks]
        spike_times_tuple, spike_waveform_features_tuple = zip(*marks, strict=False)
        spike_times = list(spike_times_tuple)
        spike_waveform_features = list(spike_waveform_features_tuple)

    spike_times, filtered_time_bin_ind = detect_coincident_spikes(spike_times)
    spike_waveform_features = [
        features[ind]
        for ind, features in zip(filtered_time_bin_ind, spike_waveform_features, strict=False)
    ]

    return spike_times, spike_waveform_features


def get_pfc_spike_times(nwb_file_name: str, brain_area: str) -> list[NDArray[np.float64]]:
    """Get prefrontal cortex sorted spike times.

    Parameters
    ----------
    nwb_file_name : str
        NWB file name
    brain_area : str
        Brain area name ("mPFC" or "OFC")

    Returns
    -------
    list
        List of arrays with spike times per unit
    """
    restriction = {
        "nwb_file_name": nwb_file_name,
        "preproc_params_name": "default",
        "sort_interval_name": "sleeps_runs_noPrePostTrialTimes raw data valid times",
        "sorter_params_name": "franklab_probe_ctx_30KHz_115rad_new_mountainsort2",
        "team_name": "ac_em_xs",
        "sorter": "mountainsort4",
    }

    max_curation_id = (CuratedSpikeSorting & restriction).fetch("curation_id").max()
    restriction.update({"curation_id": max_curation_id})

    curated_spikes_info = pd.DataFrame(
        (CuratedSpikeSorting & restriction) * SortGroup.SortGroupElectrode
    )
    electrode_group_df = get_electrode_group_info(nwb_file_name)
    curated_spikes_info = pd.merge(
        curated_spikes_info, electrode_group_df, on="electrode_group_name"
    )
    spikesorting_keys = pd.merge(
        pd.DataFrame(CuratedSpikeSorting() & restriction),
        curated_spikes_info.groupby("sort_group_id").targeted_location.first(),
        on="sort_group_id",
    )
    spikesorting_keys = spikesorting_keys.loc[
        spikesorting_keys.targeted_location == brain_area
    ].to_dict(orient="records")

    nwb_pfc = (CuratedSpikeSorting() & spikesorting_keys).fetch_nwb()

    return list(
        itertools.chain.from_iterable(
            [
                file["units"]["spike_times"].to_list()
                for file in nwb_pfc
                if "units" in file
            ]
        )
    )


def get_spike_data(nwb_file_name: str) -> SpikeDataDict:
    """Load spike data for all available brain areas.

    Parameters
    ----------
    nwb_file_name : str
        NWB file name

    Returns
    -------
    dict
        Dictionary containing:
        - spike_times: dict mapping brain area to list of spike time arrays
        - spike_waveform_features: dict mapping brain area to list of feature arrays
    """
    spike_times = {}
    spike_waveform_features = {}

    try:
        spike_times["HPC"], spike_waveform_features["HPC"] = get_hpc_marks(
            nwb_file_name
        )
    except ValueError:
        pass

    for brain_area in ["mPFC", "OFC"]:
        try:
            spike_times[brain_area] = get_pfc_spike_times(nwb_file_name, brain_area)
            if len(spike_times[brain_area]) < 1:
                del spike_times[brain_area]
        except ValueError:
            pass

    return {
        "spike_times": spike_times,
        "spike_waveform_features": spike_waveform_features,
    }


def filter_spike_times(
    spike_times: dict[str, list[NDArray[np.float64]]],
    spike_waveform_features: dict[str, list[NDArray[np.float64]]],
    position_time: NDArray[np.float64],
) -> tuple[dict[str, list[NDArray[np.float64]]], dict[str, list[NDArray[np.float64]]]]:
    """Filter spike times to match position data time bounds.

    Parameters
    ----------
    spike_times : dict
        Dictionary mapping brain area to list of spike time arrays
    spike_waveform_features : dict
        Dictionary mapping brain area to list of feature arrays
    position_time : np.ndarray
        Position time array (defines time bounds)

    Returns
    -------
    filtered_spike_times : dict
        Dictionary mapping brain area to list of filtered spike time arrays
    filtered_spike_waveform_features : dict
        Dictionary mapping brain area to list of filtered feature arrays
    """
    filtered_spike_times = {}
    filtered_spike_waveform_features = {}

    for brain_area, brain_area_spike_times in spike_times.items():
        brain_area_filtered_times = []
        brain_area_filtered_features = []
        for sort_ind, sort_group_spike_times in enumerate(brain_area_spike_times):

            is_in_bounds = np.logical_and(
                sort_group_spike_times >= position_time[0],
                sort_group_spike_times <= position_time[-1],
            )
            brain_area_filtered_times.append(sort_group_spike_times[is_in_bounds])

            try:
                brain_area_filtered_features.append(
                    spike_waveform_features[brain_area][sort_ind][is_in_bounds]
                )
            except KeyError:
                pass

        filtered_spike_times[brain_area] = brain_area_filtered_times

        if brain_area_filtered_features:
            filtered_spike_waveform_features[brain_area] = brain_area_filtered_features

    return filtered_spike_times, filtered_spike_waveform_features
