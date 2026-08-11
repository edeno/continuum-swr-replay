"""Graph-spatial helpers for per-arm predictors and QC metrics.

- :func:`dist_to_arm_from_well` — shortest-path graph distance from a
  well node to each arm (edge).
- :func:`arm_dist_rank` — within-event rank of distances, 1 = closest.
- :func:`arm_adjacency` — which arms share a graph node (for
  ``path_continuity``).
- :func:`path_continuity` — per-event fraction of consecutive
  decoded-arm pairs that are graph-adjacent (a QC metric).
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.stats import rankdata as _scipy_rankdata

from continuum_swr_replay.data_loaders.constants import ALL_ARMS


def dist_to_arm_from_well(
    current_well: float | int,
    track_graph: nx.Graph,
) -> dict[int, float]:
    """Shortest-path graph distance from a well node to each arm.

    For each arm (edge), returns the minimum shortest-path distance from
    ``current_well`` to either endpoint of that arm. Distance is 0 for
    arms adjacent to ``current_well``.

    Uses edge weights under the ``"distance"`` attribute when any edge
    carries one, otherwise falls back to hop count. Matches the legacy
    ``predictors.compute_dist_to_arm`` contract exactly.

    Parameters
    ----------
    current_well : int or float
        Well node ID (0-5). ``NaN`` returns all-``NaN``.
    track_graph : networkx.Graph
        Track graph with ``edge_id`` attributes on edges and optional
        ``"distance"`` edge weights.

    Returns
    -------
    dict[int, float]
        ``{arm_id: distance}`` for every arm in ``ALL_ARMS``. ``NaN`` if
        the well is ``NaN`` or not in the graph, or for any arm whose
        endpoints are unreachable.
    """
    if current_well is None or (
        isinstance(current_well, float) and np.isnan(current_well)
    ):
        return dict.fromkeys(ALL_ARMS, np.nan)

    current_well = int(current_well)

    has_weights = any("distance" in d for _, _, d in track_graph.edges(data=True))
    try:
        if has_weights:
            lengths = dict(
                nx.single_source_dijkstra_path_length(
                    track_graph, current_well, weight="distance"
                )
            )
        else:
            lengths = dict(
                nx.single_source_shortest_path_length(track_graph, current_well)
            )
    except nx.NodeNotFound:
        return dict.fromkeys(ALL_ARMS, np.nan)

    arm_to_endpoints: dict[int, list[int]] = {}
    for u, v, data in track_graph.edges(data=True):
        arm_id = data.get("edge_id")
        if arm_id is not None:
            arm_to_endpoints.setdefault(arm_id, []).extend([u, v])

    result: dict[int, float] = {}
    for arm in ALL_ARMS:
        endpoints = arm_to_endpoints.get(arm, [])
        if not endpoints:
            result[arm] = np.nan
            continue
        dists = [lengths.get(n, np.nan) for n in endpoints]
        valid = [d for d in dists if not np.isnan(d)]
        result[arm] = float(min(valid)) if valid else np.nan

    return result


def arm_dist_rank(distances: np.ndarray) -> np.ndarray:
    """Rank arm distances within a single event; 1 = closest.

    Ties get the minimum rank (same rank for equal distances) and the
    ranks do not consume extra integer slots (``"min"`` method). ``NaN``
    distances rank last but are themselves returned as ``NaN``.

    Parameters
    ----------
    distances : np.ndarray, shape (n_arms,)
        One distance per arm, possibly with ``NaN``.

    Returns
    -------
    np.ndarray, shape (n_arms,)
        Integer ranks cast to float so ``NaN`` positions can be preserved.
    """
    distances = np.asarray(distances, dtype=float)
    ranks = np.full(distances.shape, np.nan, dtype=float)
    finite = np.isfinite(distances)
    if not finite.any():
        return ranks

    ranks[finite] = _scipy_rankdata(distances[finite], method="min")
    return ranks


def arm_adjacency(track_graph: nx.Graph) -> dict[int, set[int]]:
    """For each arm, the set of other arms that share a graph node.

    Two arms (edges) are "adjacent" when they share at least one
    endpoint node — e.g., on the bandit graph arms 0 and 1 both
    touch patch junction 6, so they are adjacent.

    Parameters
    ----------
    track_graph : networkx.Graph
        Track graph with ``edge_id`` on edges.

    Returns
    -------
    dict[int, set[int]]
        ``{arm_id: {neighbor_arm_ids}}`` for every arm whose edge
        carries an ``edge_id`` attribute.
    """
    node_to_arms: dict[int, set[int]] = {}
    for u, v, data in track_graph.edges(data=True):
        eid = data.get("edge_id")
        if eid is None:
            continue
        eid = int(eid)
        node_to_arms.setdefault(u, set()).add(eid)
        node_to_arms.setdefault(v, set()).add(eid)

    adj: dict[int, set[int]] = {}
    for arms in node_to_arms.values():
        for a in arms:
            adj.setdefault(a, set()).update(arms - {a})
    return adj


def path_continuity(
    decoded_arms: np.ndarray,
    adjacency: dict[int, set[int]],
) -> float:
    """Fraction of consecutive decoded-arm pairs that are graph-adjacent.

    ``path_continuity_e = (1/(T_e - 1)) Σ 1[arm_{t+1} ∈
    neighbors(arm_t)]``. Self-transitions (same arm on consecutive
    timebins) count as adjacent. Returns ``NaN`` for sequences shorter
    than 2 valid consecutive pairs.

    Parameters
    ----------
    decoded_arms : np.ndarray, shape (T,)
        Per-timebin MAP arm (integer arm IDs, may contain NaN).
    adjacency : dict[int, set[int]]
        Output of :func:`arm_adjacency`.

    Returns
    -------
    float
        In [0, 1]. ``NaN`` if fewer than 2 valid consecutive pairs.
    """
    arms = np.asarray(decoded_arms, dtype=float)
    if arms.size < 2:
        return np.nan
    a_prev = arms[:-1]
    a_next = arms[1:]
    both_valid = np.isfinite(a_prev) & np.isfinite(a_next)
    if not both_valid.any():
        return np.nan
    n_pairs = int(both_valid.sum())
    n_adjacent = 0
    for prev_arm, next_arm in zip(
        a_prev[both_valid].astype(int), a_next[both_valid].astype(int), strict=True
    ):
        if prev_arm == next_arm or next_arm in adjacency.get(prev_arm, set()):
            n_adjacent += 1
    return n_adjacent / n_pairs
