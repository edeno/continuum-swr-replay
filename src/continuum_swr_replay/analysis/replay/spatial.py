"""Graph-spatial helpers for per-arm predictors.

Two helpers used by the canonical event-arm table:

- :func:`dist_to_arm_from_well` — shortest-path graph distance from a
  well node to each arm (edge). Matches the pre-refactor
  ``predictors.compute_dist_to_arm`` semantics so replacing it in the
  pipeline does not change coefficient values.
- :func:`arm_dist_rank` — within-event rank of distances, 1 = closest.

Kept deliberately small. Other "spatial" concepts mentioned in the
analysis plan (``path_continuity``, sequence-level ``forward_score``)
live elsewhere — ``path_continuity`` is a QC-layer helper that needs
shuffles, and the sequence scores operate on the event-sequence table.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from continuum_swr_replay.data_loaders.constants import TRACK_SEGMENT_TO_PATCH

ALL_ARMS = sorted(TRACK_SEGMENT_TO_PATCH.keys())


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
    if current_well is None or (isinstance(current_well, float) and np.isnan(current_well)):
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

    finite_vals = distances[finite]
    # "min" ranking: equal values share the smallest rank of the tie.
    order = np.argsort(finite_vals, kind="stable")
    sorted_vals = finite_vals[order]
    finite_ranks = np.empty(finite_vals.shape, dtype=float)
    # Emit rank 1 for the min; equal values share that rank.
    rank = 1
    for i in range(len(sorted_vals)):
        if i > 0 and sorted_vals[i] != sorted_vals[i - 1]:
            rank = i + 1
        finite_ranks[order[i]] = rank
    ranks[finite] = finite_ranks
    return ranks
