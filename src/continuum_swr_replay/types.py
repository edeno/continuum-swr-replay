"""Central type definitions for the continuum-swr-replay project.

This module provides shared TypedDicts and type aliases used across
data loading, analysis, and visualization modules.
"""

from typing import TypedDict

import networkx as nx
import numpy as np
import pandas as pd


class SessionData(TypedDict, total=False):
    """Data structure returned by load_data() from data_loaders.

    This TypedDict defines the expected structure of session data loaded
    from NWB files through the Spyglass pipeline.

    Note: Uses total=False because not all sessions have recordings from all
    brain areas (e.g., some sessions lack mPFC or OFC data).

    Attributes
    ----------
    position_info : pd.DataFrame
        Time-indexed DataFrame with position tracking data including:
        - head_position_x, head_position_y: Raw 2D position
        - projected_x_position, projected_y_position: Position on track graph
        - linear_position: 1D linearized position along track
        - track_segment_id: Current edge of track graph
        - head_speed: Instantaneous speed
        - head_orientation: Directional heading
    track_graph : nx.Graph
        NetworkX graph representing track topology.
        Nodes are wells (0-5), edges are track segments.
    linear_edge_order : list
        List of edges for linearization
    linear_edge_spacing : list
        List of edge spacings for linearization
    spike_times : dict[str, list[np.ndarray]]
        Spike times organized by brain area (e.g., "HPC", "mPFC", "OFC").
        Each area contains a list of arrays, one per sort group/unit.
    spike_waveform_features : dict[str, list[np.ndarray]]
        Waveform features for clusterless decoding (primarily HPC).
        Shape: (n_spikes, n_features) per sort group.
    ripple_times : pd.DataFrame
        Detected sharp-wave ripple events with start_time, end_time columns.
    ripple_filtered_lfps : pd.DataFrame
        Ripple-band filtered LFP signals (time-indexed).
    ripple_lfps : pd.DataFrame
        Raw ripple LFP signals (time-indexed).
    ripple_consensus_trace : pd.Series
        Ripple consensus trace across electrodes (time-indexed).
    zscored_ripple_consensus_trace : pd.Series
        Z-scored ripple detection signal (time-indexed).
    theta_filtered_lfp : pd.DataFrame
        Theta-band filtered LFP signals (time-indexed).
    theta_phase : pd.Series
        Instantaneous theta phase in radians (time-indexed).
    theta_power : pd.Series
        Theta band power (time-indexed).
    hse_times : dict[str, pd.DataFrame]
        High-synchrony events (multi-unit activity bursts) per brain area.
    multiunit_firing_rate : dict[str, pd.DataFrame]
        Multi-unit firing rate per brain area (time-indexed).
    multiunit_rate_zscore : dict[str, pd.Series]
        Z-scored multi-unit firing rates per brain area (time-indexed).
    beam_breaks : pd.DataFrame
        Poke events (nose poke in/out at wells) with start_time, end_time.
    pump_events : pd.DataFrame
        Reward pump events with timestamps.
    light_events : pd.DataFrame
        Light cue events with timestamps.
    electrode_group_info : pd.DataFrame
        Metadata about electrode groups and their locations.
    well_visits : pd.DataFrame
        Well visit intervals with poke in/out times and well IDs.
    trials : pd.DataFrame
        Trial-level data with start_time, end_time, start_node, end_node, etc.
    task_variables : pd.DataFrame
        Time-indexed task variables including:
        - path_progress: 0-1 normalized progress through trial
        - dist_to_goal: Distance remaining to goal well
        - time_to_goal: Time since trial start
        - turn_direction: Sequence of turns in trial
    """

    # Position and tracking
    position_info: pd.DataFrame
    track_graph: nx.Graph
    linear_edge_order: list
    linear_edge_spacing: list

    # Neural data - spike times and waveforms
    spike_times: dict[str, list[np.ndarray]]
    spike_waveform_features: dict[str, list[np.ndarray]]

    # LFP data - ripple band
    ripple_times: pd.DataFrame
    ripple_filtered_lfps: pd.DataFrame
    ripple_lfps: pd.DataFrame
    ripple_consensus_trace: pd.Series
    zscored_ripple_consensus_trace: pd.Series

    # LFP data - theta band
    theta_filtered_lfp: pd.DataFrame
    theta_phase: pd.Series
    theta_power: pd.Series

    # Multi-unit activity
    hse_times: dict[str, pd.DataFrame]
    multiunit_firing_rate: dict[str, pd.DataFrame]
    multiunit_rate_zscore: dict[str, pd.Series]

    # Task events from DIO
    beam_breaks: pd.DataFrame
    pump_events: pd.DataFrame
    light_events: pd.DataFrame

    # Behavioral data
    well_visits: pd.DataFrame
    trials: pd.DataFrame
    task_variables: pd.DataFrame

    # Metadata
    electrode_group_info: pd.DataFrame


class ClusterlessParams(TypedDict):
    """Parameters for clusterless decoder (used for HPC).

    Attributes
    ----------
    position_std : float
        Standard deviation for position likelihood in cm. Controls spatial
        uncertainty in the decoding model.
    waveform_std : float
        Standard deviation for waveform feature likelihood in µV. Controls
        how tightly spikes must match waveform features to contribute to
        the posterior probability.
    block_size : int
        Block size for GPU memory management during decoding. Controls how
        many time bins are processed simultaneously. Should be a power of 2
        for optimal GPU performance.
    """

    position_std: float
    waveform_std: float
    block_size: int


class SortedSpikesParams(TypedDict):
    """Parameters for sorted spikes decoder (used for mPFC/OFC).

    Attributes
    ----------
    position_std : float
        Standard deviation for position likelihood in cm. Controls spatial
        uncertainty in the decoding model.
    block_size : int
        Block size for GPU memory management during decoding. Controls how
        many time bins are processed simultaneously. Should be a power of 2
        for optimal GPU performance.
    """

    position_std: float
    block_size: int
