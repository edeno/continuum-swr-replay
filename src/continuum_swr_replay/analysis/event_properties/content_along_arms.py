"""Where along the track event content sits: each time bin's arm and place on it.

An event's content at a time bin is its MAP position. On the track graph every
arm runs from the first to the second node of its edge in the linearization
order, which all epochs share: the three trunk arms from the center junction out
to a patch junction, the six leaf arms from their patch junction out to the well
at their end. A well is a node of the graph, an arm's end point, not the arm. A
bin's place on its arm is the fraction of the arm behind it, so 1 on a leaf arm
is its end at the well, and arms of slightly different length in different
animals line up when pooled.

Every statistic here weighs animals the same: an epoch's value is over its own
time bins, an animal's the mean over its epochs, and the result the mean over
animals (event -> epoch -> animal, as the summaries' other animal values).

:func:`arm_positions` places bins on arms and :func:`distance_to_nearest_well`
measures how far along the track each one is from the nearest well;
:func:`content_density_along_arms` gives each kind's content time per cm along
every arm, the animals' mean, :func:`log2_density_ratio` the no-SWR to SWR ratio
of it, :func:`distance_from_animal_shares` each kind's content time by
distance from the animal, at a well and elsewhere, and
:func:`content_place_shares` and :func:`place_share_statistics` where content
sits against the animal and the wells.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.event_properties.event_statistics import (
    OCCUPANCY_PLACES,
    quadrant_fractions,
)
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
    positions : pd.DataFrame, shape (n_bins, 5)
        ``arm`` (the edge id), ``fraction`` (0 at the arm's first node, 1 at its
        second, clipped to [0, 1]), ``arm_length_cm`` (the span of the arm's
        bin centers), ``place_bin`` (the nearest real bin's place along the arm,
        0 at the arm's first node) and ``n_place_bins`` (the arm's real bins).

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
    nearest = np.where(nearer_left, right - 1, right)
    arm = edge[nearest]
    # Each real bin's place along its own arm, by linear position.
    place_in_arm = pd.Series(edge).groupby(edge).cumcount().to_numpy()
    low = spans.loc[arm, "min"].to_numpy()
    length = spans.loc[arm, "max"].to_numpy() - low
    return pd.DataFrame(
        {
            "arm": arm,
            "fraction": np.clip((position_cm - low) / length, 0.0, 1.0),
            "arm_length_cm": length,
            "place_bin": place_in_arm[nearest],
            "n_place_bins": spans.loc[arm, "size"].to_numpy(),
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


def arm_lengths_cm(bins: pd.DataFrame) -> pd.Series:
    """The length each arm's place bins cover.

    The span of an arm's bin centers (``arm_length_cm``) and one bin more; the
    median over all of ``bins``.

    Parameters
    ----------
    bins : pd.DataFrame, shape (n_bins, n_columns)
        One row per event time bin, with ``arm``, ``arm_length_cm`` and
        ``n_place_bins`` (:func:`arm_positions`).

    Returns
    -------
    lengths : pd.Series
        Centimeters, indexed by ``arm``: the arms ``bins`` holds content on.
    """
    n_place_bins = bins["n_place_bins"].to_numpy()
    arm_cm = bins["arm_length_cm"] * n_place_bins / (n_place_bins - 1)
    return arm_cm.groupby(bins["arm"]).median()


def even_content(bins: pd.DataFrame, at_well_cm: float) -> dict[str, float]:
    """What content spread evenly over the track would give.

    The reference for :func:`content_density_along_arms` and for the share of
    content at a well: content as likely on any centimeter of the track as on
    any other. The track is the arms ``bins`` holds content on.

    Parameters
    ----------
    bins : pd.DataFrame, shape (n_bins, n_columns)
        As :func:`arm_lengths_cm`, with ``animal``.
    at_well_cm : float
        Content this close to a well is at it.

    Returns
    -------
    statistics : dict
        ``density_per_cm``: 1 over the track's length (the sum of
        :func:`arm_lengths_cm` over all of ``bins``, the lengths
        :func:`content_density_along_arms` divides by), the density of every
        segment; ``at_well_fraction``: the share of the track within
        ``at_well_cm`` of a well, which is ``at_well_cm`` of each leaf arm (a
        leaf arm ends at one well), on each animal's own track and averaged
        over animals, as the shares of :func:`content_place_shares` are.

    Raises
    ------
    ValueError
        If ``at_well_cm`` is longer than a leaf arm (a well's reach would then
        take in other arms), or an animal has no content on an arm the others
        have (its track's length is then not known).
    """
    lengths = arm_lengths_cm(bins)
    at_well = []
    for animal, group in bins.groupby("animal"):
        own = arm_lengths_cm(group)
        if not own.index.equals(lengths.index):
            raise ValueError(f"{animal} has no content on some arm of the track")
        leaf = own[own.index.isin(LEAF_ARMS)]
        if (leaf < at_well_cm).any():
            raise ValueError(f"at_well_cm ({at_well_cm}) is longer than a leaf arm")
        at_well.append(len(leaf) * at_well_cm / own.sum())
    return {
        "density_per_cm": 1.0 / float(lengths.sum()),
        "at_well_fraction": float(np.mean(at_well)),
    }


def content_density_along_arms(bins: pd.DataFrame, n_segments: int) -> pd.DataFrame:
    """Each kind's content time per cm along every arm, the animals' mean.

    Each arm is cut into ``n_segments`` equal segments. A decoded position is a
    place bin, so a time bin's content is spread evenly over its place bin
    (``place_bin`` of the arm's ``n_place_bins``) and each segment takes the part
    of the place bin that lies in it. Counting bin centers per segment instead
    would give a segment more content the more centers fall in it: the first and
    the last segment of an arm each hold an end-point center. For an epoch and a
    kind, a segment's density is the share of the kind's content time in it over
    the segment's length, so an epoch's densities integrate to 1 over the track;
    an animal's are the mean over its epochs, and the result the mean over
    animals.

    An arm's length is the length its place bins cover (:func:`arm_lengths_cm`).

    Parameters
    ----------
    bins : pd.DataFrame, shape (n_bins, n_columns)
        One row per event time bin, with ``animal``, ``epoch_id``, ``is_swr``,
        ``arm``, ``arm_length_cm``, ``place_bin`` and ``n_place_bins``
        (:func:`arm_positions`).
    n_segments : int
        Segments per arm.

    Returns
    -------
    density : pd.DataFrame, shape (2 * n_arms * n_segments, 1)
        ``density_per_cm``, indexed by ``is_swr``, ``arm`` and ``segment``
        (0 at the arm's first node). Every arm and segment of the arms in
        ``bins`` is present, 0 where no content falls.
    """
    segment_cm = arm_lengths_cm(bins) / n_segments
    # Time bins per place bin, then the part of each place bin in each segment.
    counts = (
        bins.groupby([*_EPOCH_KEYS, "arm", "place_bin", "n_place_bins"])
        .size()
        .rename("n_time_bins")
        .reset_index()
    )
    low = (counts["place_bin"] / counts["n_place_bins"]).to_numpy()[:, None]
    high = ((counts["place_bin"] + 1) / counts["n_place_bins"]).to_numpy()[:, None]
    edges = np.arange(n_segments + 1) / n_segments
    covered = np.clip(
        np.minimum(high, edges[1:]) - np.maximum(low, edges[:-1]), 0.0, None
    )  # (n_place_bins_with_content, n_segments), the fraction of the arm
    content = counts["n_time_bins"].to_numpy()[:, None] * covered / (high - low)
    per_arm = (
        pd.DataFrame(content, columns=pd.RangeIndex(n_segments, name="segment"))
        .join(counts[[*_EPOCH_KEYS, "arm"]])
        .groupby([*_EPOCH_KEYS, "arm"])
        .sum()
    )
    per_cell = per_arm.unstack("arm", fill_value=0.0).reindex(
        columns=pd.MultiIndex.from_product(
            [range(n_segments), segment_cm.index], names=["segment", "arm"]
        ),
        fill_value=0.0,
    )
    share = per_cell.div(per_cell.sum(axis=1), axis=0)
    per_epoch = share.melt(ignore_index=False, value_name="share").reset_index()
    per_epoch["density_per_cm"] = per_epoch["share"] / per_epoch["arm"].map(segment_cm)
    return _animals_mean(per_epoch.drop(columns="share"), ["arm", "segment"])


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


def log2_density_against_even(
    density: pd.DataFrame, even_density_per_cm: float
) -> pd.Series:
    """log2 of each kind's content density in each segment over the density of
    content spread evenly over the track.

    Parameters
    ----------
    density : pd.DataFrame
        :func:`content_density_along_arms`'s output.
    even_density_per_cm : float
        :func:`even_content`'s ``density_per_cm``.

    Returns
    -------
    against_even : pd.Series
        Indexed by ``is_swr``, ``arm`` and ``segment``: 0 where a kind's content
        is as dense as even content, 1 where twice as dense, -1 where half;
        ``-inf`` where a segment holds none of the kind's content.
    """
    with np.errstate(divide="ignore"):
        against_even: pd.Series = np.log2(
            density["density_per_cm"] / even_density_per_cm
        )
    return against_even


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

    For an epoch and a kind, each distance bin's share is its count of the
    kind's time bins over all of them, split by whether the content is within
    ``at_well_cm`` of a well (track distance); an epoch's shares sum to 1. An
    animal's are the mean over its epochs, and the result the mean over animals.

    Parameters
    ----------
    bins : pd.DataFrame, shape (n_bins, n_columns)
        One row per event time bin, with ``animal``, ``epoch_id``, ``is_swr``,
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
    for (is_swr, animal, epoch), group in frame.groupby(_EPOCH_KEYS):
        here = group["distance_to_animal_cm"].to_numpy()
        well = group["at_well"].to_numpy()
        frames.append(
            pd.DataFrame(
                {
                    "is_swr": is_swr,
                    "animal": animal,
                    "epoch_id": epoch,
                    "distance_cm": edges_cm[:-1],
                    "at_well": np.histogram(here[well], edges_cm)[0] / len(group),
                    "elsewhere": np.histogram(here[~well], edges_cm)[0] / len(group),
                }
            )
        )
    return _animals_mean(pd.concat(frames), ["distance_cm"])


def content_place_shares(bins: pd.DataFrame) -> pd.DataFrame:
    """Each animal's share of each kind's content time at each place.

    The places are :data:`~.event_statistics.OCCUPANCY_PLACES`: at a remote
    well, at the animal's own well, on a remote path, near the animal off the
    wells (:func:`~.event_statistics.quadrant_fractions`, by track distance to
    the animal and to the nearest well), and at either well (``at_well``). An
    epoch's share is over its own time bins of the kind, an animal's the mean
    over its epochs.

    Parameters
    ----------
    bins : pd.DataFrame, shape (n_bins, n_columns)
        One row per event time bin, with ``animal``, ``epoch_id``, ``is_swr``,
        ``distance_to_animal_cm`` and ``distance_to_well_cm``.

    Returns
    -------
    shares : pd.DataFrame, shape (n_kinds * n_animals, 5)
        One column per place, indexed by ``is_swr`` and ``animal``.
    """
    rows = []
    for (is_swr, animal, epoch), group in bins.groupby(_EPOCH_KEYS):
        fractions = quadrant_fractions(
            group["distance_to_animal_cm"].to_numpy(),
            group["distance_to_well_cm"].to_numpy(),
        )
        fractions["at_well"] = fractions["remote_well"] + fractions["own_well"]
        rows.append(
            {"is_swr": is_swr, "animal": animal, "epoch_id": epoch, **fractions}
        )
    per_epoch = pd.DataFrame(rows)
    return per_epoch.groupby(["is_swr", "animal"])[list(OCCUPANCY_PLACES)].mean()


def place_share_statistics(shares: pd.DataFrame) -> dict:
    """The animals' mean share at each place per kind, and the animals whose
    no-SWR content is at a well more often than their SWR content.

    Parameters
    ----------
    shares : pd.DataFrame
        :func:`content_place_shares`'s output.

    Returns
    -------
    statistics : dict
        ``means``: a DataFrame, one row per ``is_swr``, one column per place;
        ``n_animals_non_swr_more_at_well``: an int (an animal without both kinds
        is not counted).
    """
    at_well = shares["at_well"].unstack("is_swr").dropna()
    return {
        "means": shares.groupby("is_swr").mean(),
        "n_animals_non_swr_more_at_well": int((at_well[False] > at_well[True]).sum()),
    }


_EPOCH_KEYS = ["is_swr", "animal", "epoch_id"]


def _animals_mean(per_epoch: pd.DataFrame, cells: list[str]) -> pd.DataFrame:
    """Per kind and cell: the mean over epochs within each animal, then over
    animals, of every value column of ``per_epoch``."""
    values = [c for c in per_epoch.columns if c not in [*_EPOCH_KEYS, *cells]]
    by_animal = per_epoch.groupby(["is_swr", "animal", *cells])[values].mean()
    return by_animal.groupby(["is_swr", *cells]).mean()
