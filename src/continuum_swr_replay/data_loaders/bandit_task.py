"""Main data loading orchestrator for the spatial bandit task.

This module provides the primary `load_data()` function that orchestrates
loading all data types for a given session and epoch: position tracking,
neural spikes, LFP signals, task events, and computed task variables.

Functions
---------
load_data
    Load all data for a session and epoch (main entry point).

Notes
-----
This is the recommended way to load data for analysis. It coordinates
all submodules (position, spikes, lfp, events, trials, task_variables)
and ensures temporal alignment across all data types.

All data is restricted to the same time bounds (position data time range),
and all time-indexed data uses seconds as the time unit.

See Also
--------
get_epoch_info : Get list of all available sessions and epochs
get_training_timepoints : Select time points for decoder training
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from spyglass.common import PositionIntervalMap

if TYPE_CHECKING:
    from ..types import SessionData

from .events import load_dios
from .lfp import filter_ripple_lfp_time, get_multiunit, get_ripple, get_theta
from .position import get_position_info
from .spikes import filter_spike_times, get_electrode_group_info, get_spike_data
from .task_variables import make_task_variables
from .trials import make_trials_df_from_well_visits, make_well_visits_df


def load_data(
    nwb_file_name: str,
    epoch_name: str,
    ripple_detector_name: str = "Kay",
    include: frozenset[str] | None = None,
) -> SessionData:
    """Load data for a given session and epoch with selective loading.

    This is the main entry point for loading data. It orchestrates loading
    of position, spikes, LFP, task events, and computes derived task variables.

    Selective loading allows you to load only the data needed for specific
    analyses, improving performance and reducing memory usage.

    Parameters
    ----------
    nwb_file_name : str
        NWB file name (e.g., "chimi20200212_.nwb")
    epoch_name : str
        Epoch name (e.g., "02_r1")
    ripple_detector_name : str, optional
        Ripple detector name (currently only "Kay" is supported)
    include : frozenset[str] | None, optional
        Set of SessionDataKeys to load. If None (default), loads all data.
        Use DataLoadingPreset for common configurations or create custom sets.
        Position data (position_info, track_graph) is always loaded as it
        defines time bounds for other data types

    Returns
    -------
    SessionData
        TypedDict containing all loaded and computed data. Includes:
        - Position data: position_info, track_graph, linearization info
        - Neural data: spike_times, spike_waveform_features
        - LFP data: ripple and theta band signals and metrics
        - Multiunit activity: hse_times, multiunit_firing_rate, multiunit_rate_zscore
        - Task events: beam_breaks, pump_events, light_events
        - Behavioral data: well_visits, trials, task_variables
        - Metadata: electrode_group_info

        See `continuum_swr_replay.analysis.hippocampus.types.SessionData` for
        complete field documentation.

    See Also
    --------
    get_epoch_info : Get list of all available sessions
    get_training_timepoints : Select time points for decoder training
    get_position_info : Position loading (called internally)
    get_spike_data : Spike loading (called internally)

    Examples
    --------
    Load all data (backward compatible):

    >>> data = load_data("chimi20200212_.nwb", "02_r1")
    >>> data.keys()
    dict_keys(['position_info', 'track_graph', 'spike_times', ...])

    Load minimal data for basic decoding:

    >>> from continuum_swr_replay.types import DataLoadingPreset
    >>> data = load_data("chimi20200212_.nwb", "02_r1",
    ...                  include=DataLoadingPreset.MINIMAL)
    >>> data.keys()
    dict_keys(['position_info', 'track_graph', 'linear_edge_order',
               'linear_edge_spacing', 'spike_times', 'spike_waveform_features'])

    Load custom subset of data:

    >>> from continuum_swr_replay.types import SessionDataKeys as SDK
    >>> data = load_data(
    ...     "chimi20200212_.nwb", "02_r1",
    ...     include={SDK.POSITION_INFO, SDK.SPIKE_TIMES, SDK.RIPPLE_TIMES, SDK.TRIALS}
    ... )

    Extend a preset with additional keys:

    >>> custom = DataLoadingPreset.NEURAL | {SDK.TRIALS, SDK.TASK_VARIABLES}
    >>> data = load_data("chimi20200212_.nwb", "02_r1", include=custom)

    Access position data:

    >>> pos = data['position_info']
    >>> pos[['head_position_x', 'head_position_y', 'linear_position']].head()
       head_position_x  head_position_y  linear_position
    0     125.3              85.7            42.5
    1     126.1              86.2            43.8
    ...

    Check which brain areas have spike data:

    >>> list(data['spike_times'].keys())
    ['HPC', 'mPFC']  # OFC not recorded in this session

    Access hippocampal spike data:

    >>> hpc_spikes = data['spike_times']['HPC']
    >>> print(f"Number of tetrodes: {len(hpc_spikes)}")
    Number of tetrodes: 8
    >>> print(f"Total spikes: {sum(len(st) for st in hpc_spikes)}")
    Total spikes: 45231

    Get trial information:

    >>> trials = data['trials']
    >>> trials[['start_time', 'end_time', 'is_reward', 'from_well', 'to_well']].head(3)
       start_time  end_time  is_reward  from_well  to_well
    0   1234.5     1240.2    True       0          3
    1   1241.0     1247.8    False      3          1
    2   1249.3     1256.1    True       1          4

    Access task variables:

    >>> task_vars = data['task_variables']
    >>> task_vars[['trial_number', 'path_progress', 'dist_to_goal']].head()
       trial_number  path_progress  dist_to_goal
    0     1           0.00           85.3
    1     1           0.12           72.1
    2     1           0.25           58.7
    ...

    Get ripple times:

    >>> ripples = data['ripple_times']
    >>> print(f"Number of ripples: {len(ripples)}")
    Number of ripples: 342
    >>> ripples[['start_time', 'end_time', 'duration']].head(3)
       start_time  end_time  duration
    0   1235.123   1235.248   0.125
    1   1237.891   1238.012   0.121
    2   1242.456   1242.598   0.142
    """
    from ..types import SessionDataKeys as SDK

    # Helper to check if we should load this key
    def should_load(key: str) -> bool:
        return include is None or key in include

    # Always load position data (required for time bounds)
    position_interval_name = (
        PositionIntervalMap
        & {"nwb_file_name": nwb_file_name, "interval_list_name": epoch_name}
    ).fetch1("position_interval_name")
    position_data = get_position_info(nwb_file_name, epoch_name, position_interval_name)

    # Initialize result with position data (always included)
    result: dict = {
        "position_info": position_data["position_info"],
        "track_graph": position_data["track_graph"],
        "linear_edge_order": position_data["linear_edge_order"],
        "linear_edge_spacing": position_data["linear_edge_spacing"],
    }

    # Conditionally load spike data
    if should_load(SDK.SPIKE_TIMES) or should_load(SDK.SPIKE_WAVEFORM_FEATURES):
        spike_data = get_spike_data(nwb_file_name)
        spike_data["spike_times"], spike_data["spike_waveform_features"] = filter_spike_times(
            spike_data["spike_times"],
            spike_data["spike_waveform_features"],
            position_data["position_info"].index,
        )
        if should_load(SDK.SPIKE_TIMES):
            result["spike_times"] = spike_data["spike_times"]
        if should_load(SDK.SPIKE_WAVEFORM_FEATURES):
            result["spike_waveform_features"] = spike_data["spike_waveform_features"]

    # Conditionally load ripple data
    if any(should_load(key) for key in [
        SDK.RIPPLE_TIMES, SDK.RIPPLE_FILTERED_LFPS, SDK.RIPPLE_LFPS,
        SDK.RIPPLE_CONSENSUS_TRACE, SDK.ZSCORED_RIPPLE_CONSENSUS_TRACE
    ]):
        ripple_data = get_ripple(nwb_file_name, epoch_name, ripple_detector_name)
        ripple_data = filter_ripple_lfp_time(ripple_data, position_data["position_info"].index)
        for key, value in ripple_data.items():
            if should_load(key):
                result[key] = value

    # Conditionally load theta data
    if any(should_load(key) for key in [SDK.THETA_FILTERED_LFP, SDK.THETA_PHASE, SDK.THETA_POWER]):
        theta_data = get_theta(nwb_file_name, epoch_name)
        for key, value in theta_data.items():
            if should_load(key):
                result[key] = value

    # Conditionally load multiunit data
    if any(should_load(key) for key in [SDK.HSE_TIMES, SDK.MULTIUNIT_FIRING_RATE, SDK.MULTIUNIT_RATE_ZSCORE]):
        # Only load if we have spike data
        if "spike_times" in result:
            multiunit_data = get_multiunit(result["spike_times"], position_data["position_info"])
            for key, value in multiunit_data.items():
                if should_load(key):
                    result[key] = value

    # Conditionally load DIO events and behavioral data
    if any(should_load(key) for key in [
        SDK.BEAM_BREAKS, SDK.PUMP_EVENTS, SDK.LIGHT_EVENTS,
        SDK.WELL_VISITS, SDK.TRIALS, SDK.TASK_VARIABLES
    ]):
        dio_events = load_dios(
            nwb_file_name, position_data["position_info"].index.to_numpy()[[0, -1]]
        )

        # Add DIO events to result if requested
        for key, value in dio_events.items():
            if should_load(key):
                result[key] = value

        # Conditionally load well visits and trials
        if should_load(SDK.WELL_VISITS) or should_load(SDK.TRIALS) or should_load(SDK.TASK_VARIABLES):
            well_visits = make_well_visits_df(
                dio_events["beam_breaks"],
                dio_events["pump_events"],
                position_data["position_info"],
                track_graph=position_data["track_graph"],
            )

            if should_load(SDK.WELL_VISITS):
                result["well_visits"] = well_visits

            if should_load(SDK.TRIALS) or should_load(SDK.TASK_VARIABLES):
                trials_df = make_trials_df_from_well_visits(well_visits)

                if should_load(SDK.TRIALS):
                    result["trials"] = trials_df

                if should_load(SDK.TASK_VARIABLES):
                    task_variables_df = make_task_variables(
                        position_info=position_data["position_info"],
                        trials_df=trials_df,
                        track_graph=position_data["track_graph"],
                    )
                    result["task_variables"] = task_variables_df

    # Conditionally load electrode group info
    if should_load(SDK.ELECTRODE_GROUP_INFO):
        electrode_group_info = get_electrode_group_info(nwb_file_name)
        result["electrode_group_info"] = electrode_group_info

    return result  # type: ignore[return-value]
