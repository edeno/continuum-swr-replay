"""Shared pytest fixtures for continuum-swr-replay tests.

This module provides reusable test fixtures for creating mock data structures,
including track graphs, position data, spike times, and full session data.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

# ============================================================================
# Track Graph Fixtures
# ============================================================================


@pytest.fixture
def simple_linear_track() -> nx.Graph:
    """Create a simple 3-node linear track for testing.

    Topology: 0 --10cm-- 1 --10cm-- 2

    Returns
    -------
    track_graph : nx.Graph
        Graph with 3 nodes and 2 edges, each 10cm long
    """
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2)])
    nx.set_edge_attributes(
        graph,
        {(0, 1): 10.0, (1, 2): 10.0},
        "distance"
    )
    return graph


@pytest.fixture
def simplified_star_track() -> nx.Graph:
    """Create a 7-node star graph for tests that don't need realistic topology.

    Six wells (nodes 0-5) connect directly to a single center node (6)
    via 6 edges. NOT the real bandit topology — the production task is
    a 10-node, 9-edge spider with patch junctions between wells and
    center (see CLAUDE.md). Tests that depend on the real topology
    (multi-segment paths, trunk-arm semantics, patch junctions) should
    construct their own graph; this fixture is a minimal substrate for
    tests where only the well-to-center degree-1 property matters.

    Returns
    -------
    track_graph : nx.Graph
        7-node star: 6 wells + 1 center, 6 edges.
    """
    graph = nx.Graph()

    # Center node connects to all wells
    edges = [
        (6, 0), (6, 1),  # Patch 1
        (6, 2), (6, 3),  # Patch 2
        (6, 4), (6, 5),  # Patch 3
    ]
    graph.add_edges_from(edges)

    # All edges are 20cm
    distances = dict.fromkeys(edges, 20.0)
    nx.set_edge_attributes(graph, distances, "distance")

    return graph


@pytest.fixture
def track_with_shortcut() -> nx.Graph:
    """Create a track with a shortcut for testing shortest path logic.

    Topology:
        0 --10cm-- 1 --10cm-- 2
        |________________________|
                5cm (shortcut)

    Returns
    -------
    track_graph : nx.Graph
        Graph with shortcut path
    """
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (0, 2)])
    nx.set_edge_attributes(
        graph,
        {(0, 1): 10.0, (1, 2): 10.0, (0, 2): 5.0},
        "distance"
    )
    return graph


# ============================================================================
# Position Data Fixtures
# ============================================================================


@pytest.fixture
def mock_position_data() -> pd.DataFrame:
    """Create mock position data for a short session.

    Simulates 10 seconds at 500 Hz with animal moving along linear track.

    Returns
    -------
    position_info : pd.DataFrame
        Time-indexed position data with all standard columns
    """
    time = np.arange(0.0, 10.0, 0.002)  # 500 Hz, 10 seconds
    n_points = len(time)

    # Simulate linear movement
    linear_position = np.linspace(0, 100, n_points)  # 0 to 100 cm
    speed = np.full(n_points, 10.0)  # Constant 10 cm/s

    position_df = pd.DataFrame({
        'head_position_x': linear_position * np.cos(0.1),  # Slight curve
        'head_position_y': linear_position * np.sin(0.1),
        'head_speed': speed,
        'head_orientation': np.full(n_points, 0.1),
        'linear_position': linear_position,
        'projected_x_position': linear_position * np.cos(0.1),
        'projected_y_position': linear_position * np.sin(0.1),
        'track_segment_id': np.zeros(n_points, dtype=int),
        'patch_id': np.ones(n_points, dtype=int),
    }, index=time)

    return position_df


@pytest.fixture
def stationary_position_data() -> pd.DataFrame:
    """Create position data for stationary animal (for replay testing).

    Returns
    -------
    position_info : pd.DataFrame
        Position data with zero velocity
    """
    time = np.arange(0.0, 2.0, 0.002)  # 2 seconds
    n_points = len(time)

    position_df = pd.DataFrame({
        'head_position_x': np.full(n_points, 50.0),
        'head_position_y': np.full(n_points, 25.0),
        'head_speed': np.zeros(n_points),  # Stationary
        'head_orientation': np.zeros(n_points),
        'linear_position': np.full(n_points, 50.0),
        'projected_x_position': np.full(n_points, 50.0),
        'projected_y_position': np.full(n_points, 25.0),
        'track_segment_id': np.zeros(n_points, dtype=int),
        'patch_id': np.ones(n_points, dtype=int),
    }, index=time)

    return position_df


# ============================================================================
# Spike Data Fixtures
# ============================================================================


@pytest.fixture
def mock_spike_times() -> dict[str, list[np.ndarray]]:
    """Create mock spike times for multiple brain areas.

    Returns
    -------
    spike_times : dict
        Dictionary mapping brain areas to lists of spike time arrays
    """
    # Use isolated RNG for test isolation
    rng = np.random.RandomState(42)

    spike_times = {
        'HPC': [
            # 4 tetrodes
            np.sort(rng.uniform(0, 10, size=100)),
            np.sort(rng.uniform(0, 10, size=120)),
            np.sort(rng.uniform(0, 10, size=90)),
            np.sort(rng.uniform(0, 10, size=110)),
        ],
        'mPFC': [
            # 3 units
            np.sort(rng.uniform(0, 10, size=80)),
            np.sort(rng.uniform(0, 10, size=60)),
            np.sort(rng.uniform(0, 10, size=70)),
        ],
    }

    return spike_times


@pytest.fixture
def mock_spike_waveform_features() -> dict[str, list[np.ndarray]]:
    """Create mock waveform features for HPC clusterless decoding.

    Returns
    -------
    spike_waveform_features : dict
        Dictionary mapping 'HPC' to list of feature arrays
    """
    # Use isolated RNG for test isolation
    rng = np.random.RandomState(42)

    features = {
        'HPC': [
            # 4 tetrodes, 4 features per spike
            rng.randn(100, 4),
            rng.randn(120, 4),
            rng.randn(90, 4),
            rng.randn(110, 4),
        ]
    }

    return features


@pytest.fixture
def coincident_spike_times() -> list[np.ndarray]:
    """Create spike times with known coincident spikes for testing removal.

    Creates 4 tetrodes where some spikes occur within 40μs window.

    Returns
    -------
    spike_times : list[np.ndarray]
        List of 4 arrays with coincident spikes at t=1.0, 3.0
    """
    spike_times = [
        np.array([1.0, 2.0, 3.0, 4.0, 5.0]),          # Tetrode 0
        np.array([1.00002, 2.5, 3.00001, 4.5]),       # Tetrode 1 (coincident at 1.0, 3.0)
        np.array([1.00003, 2.7, 3.00002, 4.7]),       # Tetrode 2 (coincident at 1.0, 3.0)
        np.array([2.1, 3.5, 4.1, 5.1]),               # Tetrode 3 (no coincident)
    ]

    return spike_times


# ============================================================================
# LFP Data Fixtures
# ============================================================================


@pytest.fixture
def mock_ripple_times() -> pd.DataFrame:
    """Create mock ripple event times.

    Returns
    -------
    ripple_times : pd.DataFrame
        DataFrame with start_time, end_time, duration columns
    """
    ripple_data = {
        'start_time': [1.0, 3.5, 7.2],
        'end_time': [1.08, 3.65, 7.35],
        'duration': [0.08, 0.15, 0.15],
    }

    return pd.DataFrame(ripple_data)


@pytest.fixture
def mock_ripple_lfp() -> pd.DataFrame:
    """Create mock ripple-band filtered LFP.

    Returns
    -------
    ripple_lfp : pd.DataFrame
        Time-indexed LFP data from multiple electrodes
    """
    time = np.arange(0.0, 10.0, 0.0005)  # 2 kHz sampling
    n_points = len(time)

    # Simulate ripple oscillations (150-250 Hz)
    rng = np.random.default_rng(42)
    lfp_data = {}
    for electrode in range(4):
        # Base noise + ripple events
        signal = rng.standard_normal(n_points) * 0.1

        # Add ripple at t=1.0 and t=3.5
        ripple_freq = 200  # Hz
        ripple_1 = np.sin(2 * np.pi * ripple_freq * time) * 2.0
        ripple_1[(time < 1.0) | (time > 1.08)] = 0

        ripple_2 = np.sin(2 * np.pi * ripple_freq * time) * 2.5
        ripple_2[(time < 3.5) | (time > 3.65)] = 0

        signal += ripple_1 + ripple_2
        lfp_data[f'electrode_{electrode}'] = signal

    return pd.DataFrame(lfp_data, index=time)


# ============================================================================
# Task Event Fixtures
# ============================================================================


@pytest.fixture
def mock_trials() -> pd.DataFrame:
    """Create mock trial data.

    Returns
    -------
    trials : pd.DataFrame
        Trial metadata with standard columns
    """
    trials_data = {
        'start_time': [0.0, 3.0, 6.0],
        'end_time': [2.5, 5.5, 8.5],
        'goal_arrival_time': [2.0, 5.0, 8.0],
        'from_well': [0, 1, 2],
        'to_well': [1, 2, 0],
        'from_patch': [1, 1, 2],
        'to_patch': [1, 2, 1],
        'is_reward': [True, False, True],
        'is_patch_change': [False, True, True],
        'duration': [4.5, 4.5, 4.5],
    }

    return pd.DataFrame(trials_data)


# ============================================================================
# Composite Fixtures
# ============================================================================


@pytest.fixture
def mock_session_data(
    mock_position_data,
    simple_linear_track,
    mock_spike_times,
    mock_spike_waveform_features,
    mock_ripple_times,
    mock_trials,
) -> dict:
    """Create a complete mock session data dictionary.

    Combines all individual fixtures into the structure returned by load_data().

    Returns
    -------
    data : dict
        Complete session data matching load_data() output structure
    """
    data = {
        # Position
        'position_info': mock_position_data,
        'track_graph': simple_linear_track,
        'linear_edge_order': [(0, 1), (1, 2)],
        'linear_edge_spacing': [10.0, 10.0],

        # Spikes
        'spike_times': mock_spike_times,
        'spike_waveform_features': mock_spike_waveform_features,

        # LFP
        'ripple_times': mock_ripple_times,

        # Task
        'trials': mock_trials,
    }

    return data


# ============================================================================
# Parametrize Helpers
# ============================================================================


@pytest.fixture(params=[0, 1, 2, 3, 4])
def n_tetrodes(request):
    """Parametrize tests over different numbers of tetrodes."""
    return request.param


@pytest.fixture(params=[True, False])
def include_features(request):
    """Parametrize tests with/without waveform features."""
    return request.param
