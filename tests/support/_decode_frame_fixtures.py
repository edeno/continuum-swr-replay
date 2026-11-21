"""The per-bin decode-frame columns the trajectory rollups read, for fixtures.

Every saved decode frame carries them (``event_tables.trajectory``,
``_REQUIRED_DOT_COLUMNS`` and ``_SIGNAL_COLUMNS``), so the event-table
builders require them. A hand-built frame that is not about the rollups gets
them as constants with :func:`with_trajectory_inputs`.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

#: Each rollup input and its constant value in a fixture frame.
TRAJECTORY_INPUTS: dict[str, float] = {
    "map_speed": 5.0,
    "step_distance": 1.0,
    "ahead_behind_distance": 3.0,
    "hpd": 12.0,
    "Local_prob": 0.1,
    "No-Spike_prob": 0.0,
    "run_speed_baseline": 50.0,
    "SWR_zscore": 1.5,
    "MUA_zscore": 2.0,
    "theta_power_zscore": 0.5,
}


def with_trajectory_inputs(dot: pd.DataFrame) -> pd.DataFrame:
    """``dot`` with every rollup input it lacks set to its constant.

    Parameters
    ----------
    dot : pd.DataFrame
        A hand-built decode frame.

    Returns
    -------
    pd.DataFrame
        The same frame, changed in place, with the columns it lacked.
    """
    for column, value in TRAJECTORY_INPUTS.items():
        if column not in dot.columns:
            dot[column] = value
    return dot


class TrackEnvStub:
    """A decoder ``Environment`` stand-in: two place bins per arm, junction bins.

    ``place_bin_centers_nodes_df_`` holds two bins per real arm (0-8, at linear
    positions 0-17) and ``n_junction_bins`` junction bins (``edge_id`` -1, at
    100, 101, ...), so a builder's junction-bin handling has something to
    exclude. The bin centers are a chain graph with its all-pairs distances, as
    ``Environment`` builds them, for the trajectory rollups.

    Parameters
    ----------
    n_junction_bins : int, optional
        Junction bins after the arm bins.
    """

    def __init__(self, n_junction_bins: int = 3) -> None:
        linear = np.concatenate(
            [np.arange(18, dtype=float), 100.0 + np.arange(n_junction_bins)]
        )
        edge = np.concatenate(
            [np.repeat(np.arange(9), 2), np.full(n_junction_bins, -1)]
        ).astype(int)
        node = np.arange(len(linear))
        self.place_bin_centers_nodes_df_ = pd.DataFrame(
            {"node_id": node, "linear_position": linear, "edge_id": edge}
        )
        graph = nx.Graph()
        for i in range(len(node) - 1):
            graph.add_edge(
                int(node[i]), int(node[i + 1]), distance=abs(linear[i + 1] - linear[i])
            )
        self.track_graph_with_bin_centers_edges_ = graph
        self.distance_between_nodes_ = dict(
            nx.all_pairs_dijkstra_path_length(graph, weight="distance")
        )
