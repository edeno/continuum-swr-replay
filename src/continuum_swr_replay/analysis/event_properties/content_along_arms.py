"""Where along the track event content sits: each time bin's arm and place on it.

An event's content at a time bin is its MAP position, or, read as a whole, its
position distribution: the posterior of the two non-local states over the
place bins, renormalized (:func:`event_place_mass`). A table of content is then
one row per time bin (the MAP) or one row per place with a ``weight``, the
posterior mass the place holds summed over time bins (:data:`WEIGHT`); every
statistic here takes either. The MAP of a time bin whose posterior is a smooth
bump at a well is the bump's one highest place bin, so MAP content piles on a
single place bin beside each well and each junction; the mass does not.
On the track graph every
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
    AT_WELL_CM,
    FAR_CM,
    OCCUPANCY_PLACES,
    quadrant_fractions,
)
from continuum_swr_replay.analysis.shared_statistics import (
    animal_balanced_bootstrap,
    bootstrap_interval,
)
from continuum_swr_replay.data_loaders.constants import (
    LEAF_ARMS,
    TRACK_SEGMENT_TO_PATCH,
)

CENTER_NODE = 9
"""The track graph's center junction."""

WEIGHT = "weight"
"""The column of a content table that holds each row's content time (posterior
mass summed over time bins); a table without it has one time bin per row."""


def _weights(bins: pd.DataFrame) -> pd.Series:
    """Each row's content time: :data:`WEIGHT`, or 1 where the table has none."""
    if WEIGHT in bins:
        return bins[WEIGHT].astype(float)
    return pd.Series(1.0, index=bins.index)


def event_place_mass(
    event_id: np.ndarray,
    mass: np.ndarray,
    distance_to_animal_cm: np.ndarray,
    far_cm: float = FAR_CM,
) -> pd.DataFrame:
    """Each event's content time at each place, split by whether the animal is near.

    Parameters
    ----------
    event_id : np.ndarray, shape (n_time_bins,)
        The event of each time bin; an event's bins are together.
    mass : np.ndarray, shape (n_time_bins, n_places)
        Each time bin's position distribution over the places; a row sums to 1.
    distance_to_animal_cm : np.ndarray, shape (n_time_bins, n_places)
        The track distance from each place to the animal at each time bin.
    far_cm : float, optional
        A place is far from the animal beyond this.

    Returns
    -------
    table : pd.DataFrame
        One row per event, place and side (far or near) that holds mass:
        ``event_id``, ``place`` (the column of ``mass``), ``far``,
        :data:`WEIGHT` (the mass summed over the event's time bins on that
        side, so an event's weights sum to its number of time bins) and
        ``distance_to_animal_cm`` (the mass-weighted mean distance on that
        side).
    """
    event_id = np.asarray(event_id)
    mass = np.asarray(mass, dtype=np.float64)
    distance = np.asarray(distance_to_animal_cm, dtype=np.float64)
    starts = np.flatnonzero(np.r_[True, event_id[1:] != event_id[:-1]])
    frames = []
    for far in (True, False):
        side = mass * ((distance > far_cm) == far)
        weight = np.add.reduceat(side, starts, axis=0)
        moment = np.add.reduceat(side * distance, starts, axis=0)
        event, place = np.nonzero(weight)
        frames.append(
            pd.DataFrame(
                {
                    "event_id": event_id[starts][event],
                    "place": place,
                    "far": far,
                    WEIGHT: weight[event, place],
                    "distance_to_animal_cm": moment[event, place]
                    / weight[event, place],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


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
        over animals, as the shares of :func:`content_place_shares` are;
        ``leaf_arm_fraction`` and ``trunk_arm_fraction``: ``at_well_cm`` as a
        part of the mean leaf arm and of the mean trunk arm, the width on a
        profile drawn along the arms of a zone that far from an arm's end (a
        well, a junction, the center).

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
    is_leaf = lengths.index.isin(LEAF_ARMS)
    return {
        "density_per_cm": 1.0 / float(lengths.sum()),
        "at_well_fraction": float(np.mean(at_well)),
        "leaf_arm_fraction": at_well_cm / float(lengths[is_leaf].mean()),
        "trunk_arm_fraction": at_well_cm / float(lengths[~is_leaf].mean()),
    }


def content_density_along_arms(
    bins: pd.DataFrame, n_segments: int, by_animal: bool = False
) -> pd.DataFrame:
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
        (:func:`arm_positions`); with :data:`WEIGHT`, one row per place and its
        content time.
    n_segments : int
        Segments per arm.
    by_animal : bool, optional
        Return each animal's densities (the mean over its epochs), not the
        animals' mean.

    Returns
    -------
    density : pd.DataFrame, shape (2 * n_arms * n_segments, 1)
        ``density_per_cm``, indexed by ``is_swr``, ``arm`` and ``segment``
        (0 at the arm's first node), and by ``animal`` after ``is_swr`` with
        ``by_animal``. Every arm and segment of the arms in ``bins`` is
        present, 0 where no content falls.
    """
    segment_cm = arm_lengths_cm(bins) / n_segments
    # Time bins per place bin, then the part of each place bin in each segment.
    counts = (
        bins.assign(n_time_bins=_weights(bins))
        .groupby([*_EPOCH_KEYS, "arm", "place_bin", "n_place_bins"])["n_time_bins"]
        .sum()
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
    per_epoch = per_epoch.drop(columns="share")
    if by_animal:
        return per_epoch.groupby(["is_swr", "animal", "arm", "segment"])[
            ["density_per_cm"]
        ].mean()
    return _animals_mean(per_epoch, ["arm", "segment"])


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
        ``"leaf"``) and ``segment``: the mean over the arms of the type; and by
        ``animal`` after ``is_swr`` when ``density`` is each animal's.
    """
    frame = density.reset_index()
    frame["arm_type"] = np.where(frame["arm"].isin(LEAF_ARMS), "leaf", "trunk")
    kinds = ["is_swr", "animal"] if "animal" in frame else ["is_swr"]
    return frame.groupby([*kinds, "arm_type", "segment"])[["density_per_cm"]].mean()


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
    frame[WEIGHT] = _weights(bins)
    for (is_swr, animal, epoch), group in frame.groupby(_EPOCH_KEYS):
        here = group["distance_to_animal_cm"].to_numpy()
        well = group["at_well"].to_numpy()
        time = group[WEIGHT].to_numpy()

        def share(mask, here=here, time=time):
            counts, _ = np.histogram(here[mask], edges_cm, weights=time[mask])
            return counts / time.sum()

        frames.append(
            pd.DataFrame(
                {
                    "is_swr": is_swr,
                    "animal": animal,
                    "epoch_id": epoch,
                    "distance_cm": edges_cm[:-1],
                    "at_well": share(well),
                    "elsewhere": share(~well),
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
    per_epoch = epoch_place_shares(bins)
    return per_epoch.groupby(["is_swr", "animal"])[list(OCCUPANCY_PLACES)].mean()


def epoch_place_shares(bins: pd.DataFrame) -> pd.DataFrame:
    """Each epoch's share of each kind's content time at each place.

    Parameters
    ----------
    bins : pd.DataFrame
        As :func:`content_place_shares` takes them.

    Returns
    -------
    shares : pd.DataFrame
        One row per kind and epoch: ``is_swr``, ``animal``, ``epoch_id`` and
        one column per place of :data:`~.event_statistics.OCCUPANCY_PLACES`.
    """
    rows = []
    for (is_swr, animal, epoch), group in bins.groupby(_EPOCH_KEYS):
        fractions = quadrant_fractions(
            group["distance_to_animal_cm"].to_numpy(),
            group["distance_to_well_cm"].to_numpy(),
            _weights(group).to_numpy(),
        )
        fractions["at_well"] = fractions["remote_well"] + fractions["own_well"]
        rows.append(
            {"is_swr": is_swr, "animal": animal, "epoch_id": epoch, **fractions}
        )
    return pd.DataFrame(rows)


INTERVAL_PLACES = ("at_well", "remote_well", "own_well")
"""The places :func:`place_share_intervals` gives an interval for."""


def place_share_intervals(
    bins: pd.DataFrame, n_draws: int = 1000, seed: int = 0
) -> pd.DataFrame:
    """The animals' mean share of each kind's content time at a well, at a
    remote well and at the animal's own, and of the no-SWR minus SWR
    difference, each with its 95 % interval.

    The estimate is :func:`content_place_shares`' mean over animals (epoch,
    then animal); the difference is of each epoch's two shares, over the epochs
    with events of both kinds. The interval is the project's
    (``shared_statistics.bootstrap_interval``) from a nested bootstrap of the
    animals, then their epochs (``shared_statistics.animal_balanced_bootstrap``,
    each epoch its own day); an epoch's events are not resampled.

    Parameters
    ----------
    bins : pd.DataFrame
        As :func:`content_place_shares` takes them.
    n_draws, seed : int, optional
        The bootstrap's replicates and seed.

    Returns
    -------
    intervals : pd.DataFrame
        ``estimate``, ``ci_low``, ``ci_high`` and ``n_epochs``, indexed by
        ``group`` (``"non_swr"``, ``"swr"``, ``"difference"``) and ``place``
        (:data:`INTERVAL_PLACES`). A share's interval is kept inside 0 to 1.
    """
    per_epoch = epoch_place_shares(bins)
    by_kind = per_epoch.set_index(["animal", "epoch_id", "is_swr"])[
        list(INTERVAL_PLACES)
    ].unstack("is_swr")
    rows = []
    for place in INTERVAL_PLACES:
        both = by_kind[place].dropna()
        groups = {
            "non_swr": by_kind[place][False].dropna(),
            "swr": by_kind[place][True].dropna(),
            "difference": both[False] - both[True],
        }
        for group, values in groups.items():
            animal = values.index.get_level_values("animal")
            epoch = values.index.get_level_values("epoch_id")
            drawn = animal_balanced_bootstrap(
                values.to_numpy(), animal, epoch, n_draws=n_draws, seed=seed
            )
            estimate = float(drawn.estimate)
            interval = bootstrap_interval(
                estimate,
                drawn.draws,
                drawn.n_animals,
                None if group == "difference" else (0.0, 1.0),
            )
            rows.append(
                {
                    "group": group,
                    "place": place,
                    "estimate": estimate,
                    "ci_low": interval.low,
                    "ci_high": interval.high,
                    "n_epochs": len(values),
                }
            )
    return pd.DataFrame(rows).set_index(["group", "place"])


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


EXTENT_BIN_EDGES_CM = (0.0, 10.0, 20.0, 50.0, 100.0, np.inf)
"""The spatial-extent classes of :func:`remote_well_share_by_extent` (cm); a
class holds its low edge and not its high one."""


def event_place_shares(rows: pd.DataFrame) -> pd.DataFrame:
    """Each event's share of its content time at a well, and at a remote one.

    Parameters
    ----------
    rows : pd.DataFrame
        The content-mass rows of the events (:func:`event_place_mass` with each
        place's ``distance_to_well_cm``): ``event_id``, ``far``, :data:`WEIGHT`
        and ``distance_to_well_cm``.

    Returns
    -------
    shares : pd.DataFrame, shape (n_events, 3)
        :data:`INTERVAL_PLACES`, indexed by ``event_id``: the share of the event's
        content time within ``AT_WELL_CM`` of a well; of that, the part far
        from the animal (``remote_well``) and the part near it (``own_well``).
    """
    weight = rows[WEIGHT].astype(float)
    at_well = rows["distance_to_well_cm"] < AT_WELL_CM
    far = rows["far"].astype(bool)
    parts = pd.DataFrame(
        {
            "event_id": rows["event_id"],
            "total": weight,
            "at_well": weight.where(at_well, 0.0),
            "remote_well": weight.where(at_well & far, 0.0),
            "own_well": weight.where(at_well & ~far, 0.0),
        }
    )
    summed = parts.groupby("event_id").sum()
    return summed[list(INTERVAL_PLACES)].div(summed["total"], axis=0)


WELL_CONTENT_SHARE = 0.1
"""An event represents a well when at least this share of its content time is
within ``AT_WELL_CM`` of it."""

WELL_COUNT_CLASSES = ("none", "one", "two or more")
"""The classes of :func:`well_count_shares`: how many wells an event represents."""


def event_well_counts(
    rows: pd.DataFrame, share: float = WELL_CONTENT_SHARE
) -> pd.Series:
    """How many wells each event represents.

    A leaf arm ends at one well, so the content within ``AT_WELL_CM`` of a well
    on an arm is at that arm's well.

    Parameters
    ----------
    rows : pd.DataFrame
        The content-mass rows of the events: ``event_id``, ``arm``,
        :data:`WEIGHT` and ``distance_to_well_cm``.
    share : float, optional
        The share of an event's content time a well must hold to count.

    Returns
    -------
    n_wells : pd.Series of int
        Indexed by ``event_id``: the wells holding at least ``share`` of the
        event's content time (0 for an event with none).
    """
    at_well = rows["distance_to_well_cm"] < AT_WELL_CM
    return _places_holding(rows[at_well], rows[at_well]["arm"], rows, share).rename(
        "n_wells"
    )


def _places_holding(
    inside: pd.DataFrame, place: pd.Series, rows: pd.DataFrame, share: float
) -> pd.Series:
    """Per event of ``rows``, how many places hold at least ``share`` of the
    event's content time: ``inside`` are the rows that lie in a place (a subset
    of ``rows``), ``place`` each one's place. Indexed by ``event_id``; 0 for an
    event with no such place."""
    total = rows[WEIGHT].astype(float).groupby(rows["event_id"]).sum()
    by_place = inside[WEIGHT].astype(float).groupby([inside["event_id"], place]).sum()
    held = (
        by_place / total.reindex(by_place.index.get_level_values("event_id")).to_numpy()
    )
    counted = (held >= share).groupby(level="event_id").sum()
    return counted.reindex(total.index, fill_value=0).astype(int)


def well_count_shares(events: pd.DataFrame, column: str = "n_wells") -> pd.DataFrame:
    """Each animal's share of its events of each kind that represent no well,
    one well, and two or more.

    An epoch's share is over its events of the kind; an animal's is the mean
    over its epochs with such an event.

    Parameters
    ----------
    events : pd.DataFrame, shape (n_events, n_columns)
        One row per event, with ``animal``, ``epoch_id``, ``is_swr`` and
        ``column`` (:func:`event_well_counts`).

    Returns
    -------
    shares : pd.DataFrame
        One column per class of :data:`WELL_COUNT_CLASSES`, indexed by
        ``is_swr`` and ``animal``; a row sums to 1.
    """
    count = events[column].to_numpy()
    frame = events[["is_swr", "animal", "epoch_id"]].assign(
        **{
            "none": count == 0,
            "one": count == 1,
            "two or more": count >= 2,
        }
    )
    return _event_shares(frame, list(WELL_COUNT_CLASSES))


PATCH_COUNT_CLASSES = ("one", "two", "three")
"""The classes of :func:`patch_count_shares`: how many patches an event spans."""


def event_patch_counts(
    rows: pd.DataFrame, share: float = WELL_CONTENT_SHARE
) -> pd.Series:
    """How many patches each event's content spans.

    A patch is a trunk arm and the two leaf arms at its end
    (``TRACK_SEGMENT_TO_PATCH``).

    Parameters
    ----------
    rows : pd.DataFrame
        The content-mass rows of the events: ``event_id``, ``arm`` and
        :data:`WEIGHT`.
    share : float, optional
        The share of an event's content time a patch must hold to count.

    Returns
    -------
    n_patches : pd.Series of int
        Indexed by ``event_id``: the patches holding at least ``share`` of the
        event's content time (1 to 3 for a share of a third or less).
    """
    patch = rows["arm"].map(TRACK_SEGMENT_TO_PATCH)
    return _places_holding(rows, patch, rows, share).rename("n_patches")


def patch_count_shares(events: pd.DataFrame, column: str = "n_patches") -> pd.DataFrame:
    """Each animal's share of its events of each kind that span one patch, two
    and three, as :func:`well_count_shares` takes the wells.

    Parameters
    ----------
    events : pd.DataFrame, shape (n_events, n_columns)
        One row per event, with ``animal``, ``epoch_id``, ``is_swr`` and
        ``column`` (:func:`event_patch_counts`).

    Returns
    -------
    shares : pd.DataFrame
        One column per class of :data:`PATCH_COUNT_CLASSES`, indexed by
        ``is_swr`` and ``animal``.
    """
    count = events[column].to_numpy()
    frame = events[["is_swr", "animal", "epoch_id"]].assign(
        one=count == 1, two=count == 2, three=count >= 3
    )
    return _event_shares(frame, list(PATCH_COUNT_CLASSES))


PATCH_CONTENT_CLASSES = (
    "own_patch_wells",
    "own_patch_elsewhere",
    "other_patch_wells",
    "other_patch_elsewhere",
)
"""Where content is against the animal's patch (:func:`patch_content_shares`)."""


def patch_content_shares(rows: pd.DataFrame) -> pd.DataFrame:
    """Each animal's share of each kind's content time in its own patch and in
    the other two, at the wells and elsewhere.

    The animal is at a well, so its patch is that well's
    (``current_patch``); the other two patches lie beyond the center junction.
    An epoch's share is over its content time of the kind, an animal's the mean
    over its epochs. Content spread evenly over the track would put a third in
    the animal's patch.

    Parameters
    ----------
    rows : pd.DataFrame
        Content-mass rows with ``is_swr``, ``animal``, ``epoch_id``, ``arm``,
        ``distance_to_well_cm``, :data:`WEIGHT` and ``current_patch`` (the
        patch of the event's animal).

    Returns
    -------
    shares : pd.DataFrame
        One column per class of :data:`PATCH_CONTENT_CLASSES`, indexed by
        ``is_swr`` and ``animal``; a row sums to 1.
    """
    own = rows["arm"].map(TRACK_SEGMENT_TO_PATCH) == rows["current_patch"]
    at_well = rows["distance_to_well_cm"] < AT_WELL_CM
    weight = rows[WEIGHT].astype(float)
    frame = rows[_EPOCH_KEYS].assign(
        own_patch_wells=weight.where(own & at_well, 0.0),
        own_patch_elsewhere=weight.where(own & ~at_well, 0.0),
        other_patch_wells=weight.where(~own & at_well, 0.0),
        other_patch_elsewhere=weight.where(~own & ~at_well, 0.0),
    )
    per_epoch = frame.groupby(_EPOCH_KEYS)[list(PATCH_CONTENT_CLASSES)].sum()
    per_epoch = per_epoch.div(per_epoch.sum(axis=1), axis=0)
    return per_epoch.groupby(["is_swr", "animal"]).mean()


START_END_CLASSES = (
    "starts_near_animal",
    "ends_near_animal",
    "starts_at_remote_well",
    "ends_at_remote_well",
)
"""What :func:`event_start_end` says of each event."""


def event_start_end(bins: pd.DataFrame) -> pd.DataFrame:
    """Where each event's content starts and where it ends.

    The content at an event's first and at its last time bin is the decode's
    most probable place there (the content-along-arms run): near the animal
    when within ``FAR_CM`` of it along the track, at a remote well when within
    ``AT_WELL_CM`` of a well and farther than ``FAR_CM`` from the animal.

    Parameters
    ----------
    bins : pd.DataFrame
        One row per event time bin, with ``event_id``, ``time``,
        ``distance_to_animal_cm`` and ``distance_to_well_cm``.

    Returns
    -------
    places : pd.DataFrame of bool
        :data:`START_END_CLASSES`, indexed by ``event_id``.
    """
    ordered = bins.sort_values(["event_id", "time"])
    ends = ordered.groupby("event_id")[
        ["distance_to_animal_cm", "distance_to_well_cm"]
    ].agg(["first", "last"])
    out = {}
    for word, which in (("starts", "first"), ("ends", "last")):
        far = ends[("distance_to_animal_cm", which)] > FAR_CM
        at_well = ends[("distance_to_well_cm", which)] < AT_WELL_CM
        out[f"{word}_near_animal"] = ~far
        out[f"{word}_at_remote_well"] = far & at_well
    return pd.DataFrame(out)[list(START_END_CLASSES)]


def start_end_shares(events: pd.DataFrame) -> pd.DataFrame:
    """Each animal's share of its events of each kind whose content starts, and
    ends, near the animal and at a remote well.

    Parameters
    ----------
    events : pd.DataFrame, shape (n_events, n_columns)
        One row per event, with ``animal``, ``epoch_id``, ``is_swr`` and the
        columns of :data:`START_END_CLASSES` (:func:`event_start_end`).

    Returns
    -------
    shares : pd.DataFrame
        One column per class, indexed by ``is_swr`` and ``animal``.
    """
    return _event_shares(events, list(START_END_CLASSES))


def _event_shares(events: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Each animal's share of its events of each kind for which each of
    ``columns`` holds: an epoch's share over its events of the kind, then the
    mean over the animal's epochs with such an event."""
    per_epoch = (
        events.groupby(["is_swr", "animal", "epoch_id"])[columns].mean().astype(float)
    )
    return per_epoch.groupby(["is_swr", "animal"]).mean()


def remote_well_share_by_extent(
    events: pd.DataFrame,
    edges_cm: tuple[float, ...] = EXTENT_BIN_EDGES_CM,
    column: str = "remote_well",
) -> pd.DataFrame:
    """Each animal's mean share of an event at a remote well, by the event's
    spatial extent and kind.

    Whether the events whose content stays in one place are the ones at a
    remote well. An epoch's value for a kind and an extent class is the mean of
    its events' ``remote_well`` shares; an animal's is the mean over its epochs
    that have such an event.

    Parameters
    ----------
    events : pd.DataFrame, shape (n_events, n_columns)
        One row per event, with ``animal``, ``epoch_id``, ``is_swr``,
        ``spatial_extent_cm`` and ``column``.
    edges_cm : tuple of float, optional
        The extent classes' edges, covering every extent.
    column : str, optional
        Each event's share of content time at a remote well
        (:func:`event_place_shares`).

    Returns
    -------
    shares : pd.DataFrame
        ``remote_well`` and ``n_events`` (the animal's events of the kind in
        the class), indexed by ``is_swr``, ``animal`` and ``extent_class`` (the
        class's place in ``edges_cm``, 0 the narrowest); a class an animal has
        no event in has no row.

    Raises
    ------
    ValueError
        If an extent is missing or outside ``edges_cm``.
    """
    edges = np.asarray(edges_cm, dtype=float)
    extent = events["spatial_extent_cm"].to_numpy(dtype=float)
    if not ((extent >= edges[0]) & (extent < edges[-1])).all():
        raise ValueError("a spatial extent is missing or outside the classes")
    frame = events[["is_swr", "animal", "epoch_id"]].assign(
        remote_well=events[column].to_numpy(dtype=float),
        extent_class=np.searchsorted(edges, extent, side="right") - 1,
    )
    keys = ["is_swr", "animal", "extent_class"]
    per_epoch = frame.groupby([*keys, "epoch_id"])["remote_well"].agg(["mean", "size"])
    return pd.DataFrame(
        {
            "remote_well": per_epoch["mean"].groupby(keys).mean(),
            "n_events": per_epoch["size"].groupby(keys).sum(),
        }
    )


_EPOCH_KEYS = ["is_swr", "animal", "epoch_id"]


def _animals_mean(per_epoch: pd.DataFrame, cells: list[str]) -> pd.DataFrame:
    """Per kind and cell: the mean over epochs within each animal, then over
    animals, of every value column of ``per_epoch``."""
    values = [c for c in per_epoch.columns if c not in [*_EPOCH_KEYS, *cells]]
    by_animal = per_epoch.groupby(["is_swr", "animal", *cells])[values].mean()
    return by_animal.groupby(["is_swr", *cells]).mean()
