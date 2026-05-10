"""Shared pytest fixtures for continuum-swr-replay tests.

Only fixtures that have downstream callers live here. Single-use
fixtures stay in the test module that uses them so renames / deletions
don't require cross-file coordination.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

# ============================================================================
# Track Graph Fixtures
# ============================================================================


@pytest.fixture
def simple_linear_track() -> nx.Graph:
    """3-node linear track: 0 --10cm-- 1 --10cm-- 2.

    The most-used graph fixture in the suite (~100 callers).
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
    """7-node star graph: six wells (nodes 0-5) connect directly to a single
    center node (6) via six 20cm edges.

    NOT the real bandit topology — the production task is a 10-node, 9-edge
    spider with patch junctions between wells and center (see CLAUDE.md).
    Tests that depend on the real topology (multi-segment paths, trunk-arm
    semantics, patch junctions) should construct their own graph; this
    fixture is a minimal substrate for tests where only the well-to-center
    degree-1 property matters.
    """
    graph = nx.Graph()
    edges = [
        (6, 0), (6, 1),  # Patch 1
        (6, 2), (6, 3),  # Patch 2
        (6, 4), (6, 5),  # Patch 3
    ]
    graph.add_edges_from(edges)
    distances = dict.fromkeys(edges, 20.0)
    nx.set_edge_attributes(graph, distances, "distance")
    return graph


# ============================================================================
# Spike Data Fixtures
# ============================================================================


@pytest.fixture
def mock_spike_times() -> dict[str, list[np.ndarray]]:
    """Mock spike times for HPC (4 tetrodes) + mPFC (3 units) over 10 s."""
    rng = np.random.RandomState(42)
    return {
        'HPC': [
            np.sort(rng.uniform(0, 10, size=100)),
            np.sort(rng.uniform(0, 10, size=120)),
            np.sort(rng.uniform(0, 10, size=90)),
            np.sort(rng.uniform(0, 10, size=110)),
        ],
        'mPFC': [
            np.sort(rng.uniform(0, 10, size=80)),
            np.sort(rng.uniform(0, 10, size=60)),
            np.sort(rng.uniform(0, 10, size=70)),
        ],
    }
