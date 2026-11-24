"""How fast a non-local event's content moves along the track: two measures.

Both are the slope of a distance against time, fitted with the slope and the
intercept held at zero or more (:func:`nonnegative_line`): a speed is not
negative, and a distance at no elapsed time is not either.

- **MAP line-fit speed** (:func:`map_line_fit_speed`): the track-graph distance
  from the event's first decoded (MAP) position to its MAP position at each
  later time bin, against time. A position that stays put gives 0, and one
  that travels gives its speed; one that leaves and returns gives less than its
  speed along the way.
- **Transport-slope speed** (:func:`transport_slope_speed`): the distance is the
  earth mover's (Wasserstein-1) distance on the track graph between the
  Continuous-state position distributions of two time bins a lag apart
  (:func:`transport_distance`), averaged over the event's pairs of bins at that
  lag, against the lag. The distributions of an event that does not move still
  differ from bin to bin by an amount that does not grow with the lag (about
  1.5 cm in these recordings); the intercept takes that, so the slope is the
  part that grows with time. The lags start at 10 ms because below that the
  bin-to-bin difference is still rising. The earth mover's distance between two
  time bins' distributions is the least the position could have moved on
  average between them, so the speed is a lower bound on the expected one;
  and where the bin-to-bin difference is as large as the travel over the
  fitted lags, the two do not add and the slope is below the speed.

The track graph is a tree, so the earth mover's distance has a closed form:
over the graph's edges, the edge's length times the mass that has to cross it
(:func:`transport_edges`).

On simulated replays of known speed (the memoryless comparison's recording,
``detection.memoryless_sequence.sweep.build_recording``) both return about
0.95 of the speed a replay was drawn with and about 0 for one held still
(``tests/integration/test_event_speed_simulated.py``).

``scripts/event_properties/measure_event_speed.py`` reads each epoch's decode,
calls :func:`event_speeds` and writes one table per epoch.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from continuum_swr_replay.analysis.decode_quality.posterior_analysis import (
    map_estimate_to_map_nodes,
)

TRANSPORT_LAG_BINS = (5, 6, 7, 8, 9, 10)
"""The lags the transport slope is fitted over, in time bins: 10 to 20 ms on
the 2 ms decode grid."""

MIN_LAGS = 3
"""The fewest lags a transport slope is fitted to."""

CONTINUOUS_PROBABILITY = 0.5
"""A time bin's distribution is read when its Continuous-state probability is
above this."""


def nonnegative_line(
    x: NDArray[np.float64], y: NDArray[np.float64]
) -> tuple[float, float]:
    """Least-squares line through (x, y) with slope >= 0 and intercept >= 0.

    Parameters
    ----------
    x : np.ndarray, shape (n,)
        At least two distinct values, none negative.
    y : np.ndarray, shape (n,)
        None negative.

    Returns
    -------
    slope, intercept : float
        The free fit when both are at least 0. With a negative free slope, the
        slope is 0 and the intercept the mean of ``y``. With a negative free
        intercept, the line goes through the origin.

    Raises
    ------
    ValueError
        If ``x`` has fewer than two distinct values, or a value of ``x`` or
        ``y`` is negative or not finite.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or np.unique(x).size < 2:
        raise ValueError("a line needs two distinct x, one y for each")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("x and y must be finite")
    if (x < 0).any() or (y < 0).any():
        raise ValueError("x and y must not be negative")
    x_mean, y_mean = x.mean(), y.mean()
    slope = float(((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean) ** 2).sum())
    intercept = float(y_mean - slope * x_mean)
    if slope < 0:
        return 0.0, float(y_mean)
    if intercept < 0:
        return float((x * y).sum() / (x * x).sum()), 0.0
    return slope, intercept


def map_line_fit_speed(
    distance_from_start_cm: NDArray[np.float64], elapsed_s: NDArray[np.float64]
) -> float:
    """The speed of a decoded position: its distance from where it started, against time.

    Parameters
    ----------
    distance_from_start_cm : np.ndarray, shape (n_bins,)
        Track-graph distance from the event's first decoded position to its
        decoded position at each time bin; NaN where there is none.
    elapsed_s : np.ndarray, shape (n_bins,)
        Seconds since the event's first time bin.

    Returns
    -------
    float
        cm/s, at least 0 (:func:`nonnegative_line`); NaN with fewer than three
        decoded bins.
    """
    distance = np.asarray(distance_from_start_cm, dtype=np.float64)
    elapsed = np.asarray(elapsed_s, dtype=np.float64)
    decoded = np.isfinite(distance)
    if decoded.sum() < 3:
        return np.nan
    return nonnegative_line(elapsed[decoded], distance[decoded])[0]


def transport_edges(
    track_graph: nx.Graph, bin_nodes: NDArray[np.int64]
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """The edges of a tree-shaped track that mass can cross, for :func:`transport_distance`.

    Parameters
    ----------
    track_graph : nx.Graph
        A tree whose edges carry ``distance`` (cm): the environment's
        ``track_graph_with_bin_centers_edges_``.
    bin_nodes : np.ndarray of int, shape (n_bins,)
        The graph node of each place bin, in the order of the distributions'
        columns.

    Returns
    -------
    length_cm : np.ndarray, shape (n_edges,)
        The edges of positive length that have a place bin on each side.
    beyond : np.ndarray of bool, shape (n_edges, n_bins)
        Whether each place bin is on the far side of each edge from the
        graph's first node.

    Raises
    ------
    ValueError
        If the graph is not a tree, or a place bin's node is not in it.
    """
    if not nx.is_tree(track_graph):
        raise ValueError("the track graph is not a tree")
    bin_nodes = np.asarray(bin_nodes)
    column = {int(node): k for k, node in enumerate(bin_nodes)}
    if len(column) != bin_nodes.size or not set(column) <= set(track_graph.nodes):
        raise ValueError("each place bin needs its own node of the track graph")
    root = next(iter(track_graph.nodes))
    below: dict[Any, list[int]] = {}
    lengths, beyond = [], []
    for node in nx.dfs_postorder_nodes(track_graph, root):
        below[node] = [column[int(node)]] if int(node) in column else []
    parent = dict(nx.bfs_predecessors(track_graph, root))
    for node in nx.dfs_postorder_nodes(track_graph, root):
        if node == root:
            continue
        length = float(track_graph.edges[parent[node], node]["distance"])
        if length > 0 and 0 < len(below[node]) < bin_nodes.size:
            side = np.zeros(bin_nodes.size, dtype=bool)
            side[below[node]] = True
            lengths.append(length)
            beyond.append(side)
        below[parent[node]] = below[parent[node]] + below[node]
    return np.asarray(lengths), np.asarray(beyond).reshape(len(lengths), bin_nodes.size)


def transport_distance(
    first: NDArray[np.float64],
    second: NDArray[np.float64],
    length_cm: NDArray[np.float64],
    beyond: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Earth mover's distance on the track between pairs of position distributions.

    Parameters
    ----------
    first, second : np.ndarray, shape (n_pairs, n_bins)
        Probability over the place bins, each row summing to 1.
    length_cm, beyond : np.ndarray
        :func:`transport_edges`.

    Returns
    -------
    np.ndarray, shape (n_pairs,)
        cm: over the edges, the edge's length times the mass that must cross it
        to turn ``first`` into ``second``.
    """
    crossing = (np.asarray(first) - np.asarray(second)) @ beyond.T.astype(np.float64)
    return np.abs(crossing) @ length_cm


def transport_slope_speed(
    distribution: NDArray[np.float64],
    row: NDArray[np.int64],
    usable: NDArray[np.bool_],
    length_cm: NDArray[np.float64],
    beyond: NDArray[np.bool_],
    bin_s: float,
    lag_bins: tuple[int, ...] = TRANSPORT_LAG_BINS,
) -> tuple[float, float, int]:
    """The speed at which an event's position distribution moves along the track.

    Parameters
    ----------
    distribution : np.ndarray, shape (n_event_bins, n_bins)
        The position distribution at each of the event's time bins, each row
        summing to 1.
    row : np.ndarray of int, shape (n_event_bins,)
        Each time bin's place on the decode grid, increasing; two bins are a
        lag apart when their rows differ by it.
    usable : np.ndarray of bool, shape (n_event_bins,)
        Which time bins' distributions are read.
    length_cm, beyond : np.ndarray
        :func:`transport_edges`.
    bin_s : float
        The decode grid's step, seconds.
    lag_bins : tuple of int, optional
        The lags, in time bins.

    Returns
    -------
    speed_cm_s : float
        The slope of the mean transport distance against the lag
        (:func:`nonnegative_line`); NaN when fewer than :data:`MIN_LAGS` lags
        have a pair of usable bins.
    still_cm : float
        The line's intercept: the distance between two bins' distributions that
        does not grow with the lag.
    n_lags : int
        The lags with a pair.
    """
    row = np.asarray(row)
    place = {int(r): k for k, r in enumerate(row)}
    lags, distances = [], []
    for lag in lag_bins:
        later = np.array([place.get(int(r) + lag, -1) for r in row])
        paired = (later >= 0) & usable & usable[np.clip(later, 0, None)]
        if paired.any():
            lags.append(lag * bin_s)
            distances.append(
                transport_distance(
                    distribution[paired], distribution[later[paired]], length_cm, beyond
                ).mean()
            )
    if len(lags) < MIN_LAGS:
        return np.nan, np.nan, len(lags)
    speed, still = nonnegative_line(np.asarray(lags), np.asarray(distances))
    return speed, still, len(lags)


def event_speeds(
    event_id: NDArray[np.int64],
    row: NDArray[np.int64],
    map_estimate: NDArray[np.float64],
    continuous: NDArray[np.float64],
    environment: Any,
    bin_s: float,
) -> pd.DataFrame:
    """One epoch's events: the MAP line-fit speed and the transport-slope speed of each.

    Parameters
    ----------
    event_id : np.ndarray, shape (n_event_bins,)
        The event of each time bin; an event's bins are together and in time
        order.
    row : np.ndarray of int, shape (n_event_bins,)
        Each time bin's place on the decode grid.
    map_estimate : np.ndarray, shape (n_event_bins,)
        The decode's MAP linear position at each time bin, cm; NaN where
        undecoded.
    continuous : np.ndarray, shape (n_event_bins, n_real_bins)
        The posterior of the Non-Local Continuous state at each time bin, over
        the environment's real place bins (``node_id`` not -1) in their order
        in ``place_bin_centers_nodes_df_``; not renormalized, so a row's sum is
        the state's probability.
    environment : Environment
        The decode's, with ``place_bin_centers_nodes_df_``,
        ``distance_between_nodes_`` and ``track_graph_with_bin_centers_edges_``.
    bin_s : float
        The decode grid's step, seconds.

    Returns
    -------
    speeds : pd.DataFrame, shape (n_events, 7)
        ``event_id``; ``n_bins``; ``map_line_fit_speed_cm_s``
        (:func:`map_line_fit_speed`); ``transport_slope_speed_cm_s``,
        ``transport_still_cm`` and ``transport_n_lags``
        (:func:`transport_slope_speed`, on the time bins whose Continuous-state
        probability is above :data:`CONTINUOUS_PROBABILITY`, each one's
        Continuous posterior renormalized); and ``continuous_bin_share``, the
        share of the event's bins that are such bins.

    Raises
    ------
    ValueError
        If ``continuous`` does not have one column per real place bin.
    """
    nodes_df = environment.place_bin_centers_nodes_df_
    bin_nodes = nodes_df["node_id"].to_numpy()
    bin_nodes = bin_nodes[bin_nodes != -1]
    continuous = np.asarray(continuous, dtype=np.float64)
    if continuous.shape[1] != bin_nodes.size:
        raise ValueError(
            f"{continuous.shape[1]} posterior columns for {bin_nodes.size} real place bins"
        )
    length_cm, beyond = transport_edges(
        environment.track_graph_with_bin_centers_edges_, bin_nodes
    )
    between = environment.distance_between_nodes_
    node, decoded = map_estimate_to_map_nodes(np.asarray(map_estimate), nodes_df)
    probability = continuous.sum(axis=1)
    usable = probability > CONTINUOUS_PROBABILITY
    conditional = continuous / np.where(probability > 0, probability, 1.0)[:, None]

    event_id = np.asarray(event_id)
    row = np.asarray(row)
    starts = np.flatnonzero(np.r_[True, event_id[1:] != event_id[:-1]])
    ends = np.r_[starts[1:], event_id.size]
    records = []
    for a, b in zip(starts, ends, strict=True):
        bins = np.arange(a, b)
        with_map = bins[decoded[a:b]]
        distance = np.full(b - a, np.nan)
        if with_map.size:
            origin = between[node[with_map[0]]]
            distance[with_map - a] = [float(origin[node[k]]) for k in with_map]
        speed, still, n_lags = transport_slope_speed(
            conditional[a:b], row[a:b], usable[a:b], length_cm, beyond, bin_s
        )
        records.append(
            {
                "event_id": event_id[a],
                "n_bins": b - a,
                "map_line_fit_speed_cm_s": map_line_fit_speed(
                    distance, (row[a:b] - row[a]) * bin_s
                ),
                "transport_slope_speed_cm_s": speed,
                "transport_still_cm": still,
                "transport_n_lags": n_lags,
                "continuous_bin_share": float(usable[a:b].mean()),
            }
        )
    return pd.DataFrame.from_records(records)
