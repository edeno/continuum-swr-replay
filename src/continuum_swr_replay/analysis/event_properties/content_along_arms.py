"""Where along the track event content sits: each time bin's arm and place on it.

An event's content at a time bin is its MAP position. On the track graph every
arm runs from the first to the second node of its edge in the linearization
order, which all epochs share: the three trunk arms from the centre junction out
to a patch junction, the six leaf arms from their patch junction to their well.
A bin's place on its arm is the fraction of the arm behind it, so 1 on a leaf
arm is the well, and arms of slightly different length in different animals line
up when pooled.

:func:`arm_positions` places bins on arms; :func:`content_density_along_arms`
gives each kind's content time per cm along every arm, the animals' mean.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.data_loaders.constants import LEAF_ARMS

CENTRE_NODE = 9
"""The track graph's centre junction."""

WELL_NODES = frozenset(range(6))
"""The track graph's well nodes."""


def check_arm_orientation(edge_order: list[tuple[int, int]]) -> None:
    """Raise unless every arm runs outward: trunks from the centre, leaf arms to
    their well.

    Parameters
    ----------
    edge_order : list of (int, int)
        The linearization's edge order (``Environment.edge_order``), each edge as
        (first node, second node).

    Raises
    ------
    ValueError
        If a trunk arm does not start at :data:`CENTRE_NODE`, or a leaf arm does
        not end at a well node.
    """
    for first, second in edge_order:
        trunk = first == CENTRE_NODE or second == CENTRE_NODE
        if trunk and first != CENTRE_NODE:
            raise ValueError(
                f"trunk arm {(first, second)} does not start at the centre"
            )
        if not trunk and second not in WELL_NODES:
            raise ValueError(f"leaf arm {(first, second)} does not end at a well")


def arm_positions(position_cm: np.ndarray, bin_nodes: pd.DataFrame) -> pd.DataFrame:
    """Each position's arm and the fraction of that arm behind it.

    A position goes to its nearest real place bin (``edge_id`` >= 0; junction
    placeholder bins, ``edge_id`` -1, are left out). The arm spans its real bin
    centres, from the lowest linear position to the highest.

    Parameters
    ----------
    position_cm : np.ndarray, shape (n_bins,)
        Linear positions, cm (a decode's ``map_estimate``).
    bin_nodes : pd.DataFrame, shape (n_place_bins, n_columns)
        The place-bin centres (``Environment.place_bin_centers_nodes_df_``), with
        ``edge_id`` and ``linear_position``.

    Returns
    -------
    positions : pd.DataFrame, shape (n_bins, 3)
        ``arm`` (the edge id), ``fraction`` (0 at the arm's first node, 1 at its
        second, clipped to [0, 1]) and ``arm_length_cm`` (the span of the arm's
        bin centres).

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
    centres = real["linear_position"].to_numpy()
    order = np.argsort(centres)
    centres = centres[order]
    edge = real["edge_id"].to_numpy()[order]
    right = np.clip(np.searchsorted(centres, position_cm), 1, len(centres) - 1)
    nearer_left = position_cm - centres[right - 1] <= centres[right] - position_cm
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
