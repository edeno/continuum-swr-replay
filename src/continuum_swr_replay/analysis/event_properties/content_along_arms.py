"""Where along the track event content sits: each time bin's arm and place on it.

An event's content at a time bin is its MAP position. On the track graph every
arm runs from the first to the second node of its edge in the linearization
order, which all epochs share: the three trunk arms from the center junction out
to a patch junction, the six leaf arms from their patch junction out to the well
at their end. A well is a node of the graph, an arm's end point, not the arm. A
bin's place on its arm is the fraction of the arm behind it, so 1 on a leaf arm
is its end at the well, and arms of slightly different length in different
animals line up when pooled.

:func:`arm_positions` places bins on arms and :func:`distance_to_nearest_well`
measures how far along the track each one is from the nearest well;
:func:`content_density_along_arms` gives each kind's content time per cm along
every arm, the animals' mean, :func:`log2_density_ratio` the no-SWR to SWR ratio
of it, and :func:`distance_from_animal_shares` each kind's content time by
distance from the animal, at a well and elsewhere.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from continuum_swr_replay.data_loaders.constants import LEAF_ARMS

CENTER_NODE = 9
"""The track graph's center junction."""


def check_arm_orientation(
    edge_order: list[tuple[int, int]], track_graph: nx.Graph
) -> None:
    """Raise unless every arm runs outward: trunks from the center, leaf arms to
    the well at their end.

    The wells are the track graph's end points, its degree-1 nodes.

    Parameters
    ----------
    edge_order : list of (int, int)
        The linearization's edge order (``Environment.edge_order``), each edge as
        (first node, second node).
    track_graph : nx.Graph
        The track graph (``Environment.track_graph``).

    Raises
    ------
    ValueError
        If a trunk arm does not start at :data:`CENTER_NODE`, or a leaf arm does
        not end at a well node.
    """
    wells = {node for node in track_graph.nodes if track_graph.degree(node) == 1}
    for first, second in edge_order:
        trunk = first == CENTER_NODE or second == CENTER_NODE
        if trunk and first != CENTER_NODE:
            raise ValueError(
                f"trunk arm {(first, second)} does not start at the center"
            )
        if not trunk and second not in wells:
            raise ValueError(f"leaf arm {(first, second)} does not end at a well")


def arm_positions(position_cm: np.ndarray, bin_nodes: pd.DataFrame) -> pd.DataFrame:
    """Each position's arm and the fraction of that arm behind it.

    A position goes to its nearest real place bin (``edge_id`` >= 0; junction
    placeholder bins, ``edge_id`` -1, are left out). The arm spans its real bin
    centers, from the lowest linear position to the highest.

    Parameters
    ----------
    position_cm : np.ndarray, shape (n_bins,)
        Linear positions, cm (a decode's ``map_estimate``).
    bin_nodes : pd.DataFrame, shape (n_place_bins, n_columns)
        The place-bin centers (``Environment.place_bin_centers_nodes_df_``), with
        ``edge_id`` and ``linear_position``.

    Returns
    -------
    positions : pd.DataFrame, shape (n_bins, 3)
        ``arm`` (the edge id), ``fraction`` (0 at the arm's first node, 1 at its
        second, clipped to [0, 1]) and ``arm_length_cm`` (the span of the arm's
        bin centers).

    Raises
    ------
    ValueError
        If a position is not finite, or an arm has fewer than two real bins.
    """
    position_cm = np.asarray(position_cm, dtype=float)
    if not np.isfinite(position_cm).all():
        raise ValueError("a position is not finite")
    real = bin_nodes[bin_nodes["edge_id"] >= 0]
    spans = real.groupby("edge_id")["linear_position"].agg(["min", "max", "size"])
    if (spans["size"] < 2).any():
        raise ValueError("an arm has fewer than two real place bins")
    centers = real["linear_position"].to_numpy()
    order = np.argsort(centers)
    centers = centers[order]
    edge = real["edge_id"].to_numpy()[order]
    right = np.clip(np.searchsorted(centers, position_cm), 1, len(centers) - 1)
    nearer_left = position_cm - centers[right - 1] <= centers[right] - position_cm
    arm = edge[np.where(nearer_left, right - 1, right)]
    low = spans.loc[arm, "min"].to_numpy()
    length = spans.loc[arm, "max"].to_numpy() - low
    return pd.DataFrame(
        {
            "arm": arm,
            "fraction": np.clip((position_cm - low) / length, 0.0, 1.0),
            "arm_length_cm": length,
        }
    )


def distance_to_nearest_well(
    arm: np.ndarray,
    fraction: np.ndarray,
    track_graph: nx.Graph,
    edge_order: list[tuple[int, int]],
) -> np.ndarray:
    """Each position's track distance to the nearest well, cm.

    The wells are the graph's end points (degree-1 nodes). A position at
    ``fraction`` of an arm (from the arm's first node in ``edge_order``) is
    ``fraction`` times the arm's graph length from its first node; its distance
    to the nearest well is the shorter way out through either end of the arm,
    along the graph's ``distance`` weights.

    Parameters
    ----------
    arm : np.ndarray of int, shape (n_bins,)
        Each position's arm (edge id), as :func:`arm_positions` gives it.
    fraction : np.ndarray, shape (n_bins,)
        The fraction of the arm behind each position.
    track_graph : nx.Graph
        The track graph, each edge with ``edge_id`` and ``distance`` (cm).
    edge_order : list of (int, int)
        The linearization's edge order, each arm as (first node, second node).

    Returns
    -------
    np.ndarray, shape (n_bins,)
    """
    wells = [node for node in track_graph.nodes if track_graph.degree(node) == 1]
    to_well = nx.multi_source_dijkstra_path_length(
        track_graph, wells, weight="distance"
    )
    n_arms = 1 + max(track_graph.edges[edge]["edge_id"] for edge in edge_order)
    first_cm, second_cm, length = (np.full(n_arms, np.nan) for _ in range(3))
    for first, second in edge_order:
        edge = track_graph.edges[first, second]
        first_cm[edge["edge_id"]] = to_well[first]
        second_cm[edge["edge_id"]] = to_well[second]
        length[edge["edge_id"]] = edge["distance"]
    arm = np.asarray(arm, dtype=int)
    along = np.asarray(fraction, dtype=float) * length[arm]
    return np.minimum(along + first_cm[arm], length[arm] - along + second_cm[arm])


def content_density_along_arms(bins: pd.DataFrame, n_segments: int) -> pd.DataFrame:
    """Each kind's content time per cm along every arm, the animals' mean.

    Each arm is cut into ``n_segments`` equal segments. For an animal and a kind,
    a segment's density is the share of the kind's time bins in it over the
    segment's length (the arm's median length over all of ``bins``, over
    ``n_segments``), so an animal's densities integrate to 1 over the track.
    The animals' densities are averaged, each animal weighing the same.

    Parameters
    ----------
    bins : pd.DataFrame, shape (n_bins, n_columns)
        One row per event time bin, with ``animal``, ``is_swr``, ``arm``,
        ``fraction`` and ``arm_length_cm`` (:func:`arm_positions`).
    n_segments : int
        Segments per arm.

    Returns
    -------
    density : pd.DataFrame, shape (2 * n_arms * n_segments, 1)
        ``density_per_cm``, indexed by ``is_swr``, ``arm`` and ``segment``
        (0 at the arm's first node). Every arm and segment of the arms in
        ``bins`` is present, 0 where no bin falls.
    """
    segment = np.minimum((bins["fraction"] * n_segments).astype(int), n_segments - 1)
    bins = bins.assign(segment=segment)
    segment_cm = bins.groupby("arm")["arm_length_cm"].median() / n_segments
    cells = pd.MultiIndex.from_product(
        [segment_cm.index, range(n_segments)], names=["arm", "segment"]
    )
    per_cm_of_cell = np.repeat(segment_cm.to_numpy(), n_segments)
    frames = []
    for (is_swr, animal), group in bins.groupby(["is_swr", "animal"]):
        share = group.groupby(["arm", "segment"]).size().reindex(
            cells, fill_value=0
        ) / len(group)
        frames.append(
            pd.DataFrame(
                {
                    "is_swr": is_swr,
                    "animal": animal,
                    "density_per_cm": share.to_numpy() / per_cm_of_cell,
                },
                index=cells,
            )
        )
    by_animal = pd.concat(frames).reset_index()
    return by_animal.groupby(["is_swr", "arm", "segment"])[["density_per_cm"]].mean()


def leaf_and_trunk_profiles(density: pd.DataFrame) -> pd.DataFrame:
    """The mean density along the leaf arms and along the trunk arms.

    Parameters
    ----------
    density : pd.DataFrame
        :func:`content_density_along_arms`'s output.

    Returns
    -------
    profiles : pd.DataFrame
        ``density_per_cm``, indexed by ``is_swr``, ``arm_type`` (``"trunk"``,
        ``"leaf"``) and ``segment``: the mean over the arms of the type.
    """
    frame = density.reset_index()
    frame["arm_type"] = np.where(frame["arm"].isin(LEAF_ARMS), "leaf", "trunk")
    return frame.groupby(["is_swr", "arm_type", "segment"])[["density_per_cm"]].mean()


def log2_density_ratio(density: pd.DataFrame) -> pd.Series:
    """log2 of the no-SWR to SWR content density in each segment.

    Parameters
    ----------
    density : pd.DataFrame
        :func:`content_density_along_arms`'s output.

    Returns
    -------
    ratio : pd.Series
        Indexed by ``arm`` and ``segment``. NaN where either kind has no
        content (the ratio is undefined there).
    """
    non_swr = density.loc[False, "density_per_cm"]
    swr = density.loc[True, "density_per_cm"]
    both = (non_swr > 0) & (swr > 0)
    return np.log2(non_swr.where(both) / swr.where(both))


def distance_from_animal_shares(
    bins: pd.DataFrame, edges_cm: np.ndarray, at_well_cm: float
) -> pd.DataFrame:
    """Each kind's content time by distance from the animal, at a well and
    elsewhere, the animals' mean.

    For an animal and a kind, each distance bin's share is its count of the
    kind's time bins over all of them, split by whether the content is within
    ``at_well_cm`` of a well (track distance); an animal's shares sum to 1. The
    animals' shares are averaged, each animal weighing the same.

    Parameters
    ----------
    bins : pd.DataFrame, shape (n_bins, n_columns)
        One row per event time bin, with ``animal``, ``is_swr``,
        ``distance_to_animal_cm`` and ``distance_to_well_cm``.
    edges_cm : np.ndarray, shape (n_edges,)
        The distance bins' edges, cm, covering every distance.
    at_well_cm : float
        Content nearer than this to a well is at a well.

    Returns
    -------
    shares : pd.DataFrame, shape (2 * (n_edges - 1), 2)
        ``at_well`` and ``elsewhere``, indexed by ``is_swr`` and the bin's left
        edge, ``distance_cm``.

    Raises
    ------
    ValueError
        If a distance is missing or outside ``edges_cm``.
    """
    distance = bins["distance_to_animal_cm"].to_numpy()
    if not ((distance >= edges_cm[0]) & (distance < edges_cm[-1])).all():
        raise ValueError("a distance from the animal is missing or outside the bins")
    frame = bins.assign(at_well=bins["distance_to_well_cm"] < at_well_cm)
    frames = []
    for (is_swr, _), group in frame.groupby(["is_swr", "animal"]):
        here = group["distance_to_animal_cm"].to_numpy()
        well = group["at_well"].to_numpy()
        frames.append(
            pd.DataFrame(
                {
                    "is_swr": is_swr,
                    "distance_cm": edges_cm[:-1],
                    "at_well": np.histogram(here[well], edges_cm)[0] / len(group),
                    "elsewhere": np.histogram(here[~well], edges_cm)[0] / len(group),
                }
            )
        )
    return (
        pd.concat(frames)
        .groupby(["is_swr", "distance_cm"])[["at_well", "elsewhere"]]
        .mean()
    )
