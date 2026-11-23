"""How much the tetrodes fire, in the encoding model, at an event's decoded place.

The decoder's clusterless likelihood scores a time bin without spikes by the
summed ground intensity at each place (the expected spikes per bin over all
tetrodes): where that rate is low, an empty bin is itself evidence for the
place. This module ranks the track's real places by that rate and gives each
non-local event the rank of its decoded place, so that the events decoded at a
low-rate place can be measured and reported apart.

- A place is a real place bin of the decode's environment: a track-interior
  bin (the bins the ground intensity is given at) that is not a junction
  placeholder (``edge_id`` -1; the development guide, "Junction place bins").
- A place's rank is the number of real places with a strictly lower ground
  intensity, scaled to 0 (lowest rate) to 1 (highest); tied places take the
  lowest rank of the tie (:func:`place_ranks`).
- An event's two places are the nearest real places to its replay start and
  end positions; each has its rank (:data:`RANK_COLUMNS`,
  :func:`epoch_place_rates`). A mean of the two ranks is not used: it is
  pulled toward the middle, and an event with one end at a low-rate place
  would count by the accident of its other end.
- An event is a low-rate event when both its places have a rank below
  :data:`LOW_RATE_RANK` (an event with one position is judged on that one;
  :func:`low_rate_events`).
- The share of a group's places in each tenth of the rank counts each
  event's two places, half an event each (:func:`rank_tenth_shares`), so the
  share expected by chance in a tenth is the share of the track's places in
  it, about 10 %.

The shares are reduced by the project's one averaging rule: each epoch's share,
then each animal's mean over its epochs
(``shared_statistics.animal_session_values(..., "mean")``), then the mean over
animals (:func:`low_rate_share_statistics`, :func:`rank_tenth_shares`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from continuum_swr_replay.analysis.shared_statistics import animal_session_values
from continuum_swr_replay.columns import require_columns

LOW_RATE_RANK = 0.10
"""An event whose places' ranks are both below this is a low-rate event: its
decoded places are among the lowest-rate tenth of the track's real places."""

N_RANK_BINS = 10
"""The equal bins of the rank (tenths) of :func:`rank_tenth_shares`."""

POSITION_COLUMNS = ("replay_start_position_cm", "replay_end_position_cm")
"""The event table's decoded positions (linear cm) an event is ranked at."""

RANK_COLUMNS = ("start_place_rate_rank", "end_place_rate_rank")
"""The ranks of the places nearest :data:`POSITION_COLUMNS`, in that order."""


def real_places(
    ground_intensity: ArrayLike,
    place_centers_cm: ArrayLike,
    is_track_interior: ArrayLike,
    edge_id: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The real places' linear positions and ground intensities.

    Parameters
    ----------
    ground_intensity : array_like, shape (n_interior,)
        The encoding model's ``summed_ground_process_intensity``: the expected
        spikes per time bin, summed over tetrodes, at each track-interior bin.
    place_centers_cm : array_like, shape (n_bins,)
        Every place bin's linear position (cm), interior or not.
    is_track_interior : array_like of bool, shape (n_bins,)
        The environment's ``is_track_interior_``.
    edge_id : array_like of int, shape (n_bins,)
        Each bin's track edge; -1 marks a junction placeholder.

    Returns
    -------
    centers_cm : np.ndarray, shape (n_places,)
        The real places' linear positions (cm): interior bins with
        ``edge_id >= 0``, in bin order.
    rates : np.ndarray, shape (n_places,)
        Their ground intensities.

    Raises
    ------
    ValueError
        If the ground intensity is not one value per interior bin, or the
        arrays over bins differ in length.
    """
    ground = np.asarray(ground_intensity, dtype=float).ravel()
    centers = np.asarray(place_centers_cm, dtype=float).ravel()
    interior = np.asarray(is_track_interior, dtype=bool).ravel()
    edges = np.asarray(edge_id).ravel()
    if not centers.size == interior.size == edges.size:
        raise ValueError(
            f"{centers.size} place centers, {interior.size} interior flags and "
            f"{edges.size} edge ids"
        )
    if ground.size != interior.sum():
        raise ValueError(
            f"{ground.size} ground intensities for {interior.sum()} interior bins"
        )
    real = edges[interior] >= 0
    return centers[interior][real], ground[real]


def place_ranks(rates: ArrayLike) -> NDArray[np.float64]:
    """Each place's rank by rate, 0 (lowest) to 1 (highest).

    A place's rank is the number of places with a strictly lower rate, over
    ``n_places - 1``. Tied places (rates equal as stored, no tolerance) all take
    the lowest rank of the tie: a place where the encoding model has no rate
    (an unvisited patch, rate near 0 at many places) is a lowest-rate place,
    whatever its position on the track, so its rank does not depend on the
    order of the places.

    Parameters
    ----------
    rates : array_like, shape (n_places,)
        At least two places.

    Returns
    -------
    ranks : np.ndarray, shape (n_places,)
    """
    rates = np.asarray(rates, dtype=float)
    n_lower = np.searchsorted(np.sort(rates), rates, side="left")
    return n_lower / (rates.size - 1)


def nearest_places(positions_cm: ArrayLike, centers_cm: ArrayLike) -> NDArray[np.int64]:
    """The index of the place nearest each position (the first on a tie), or
    -1 where the position is missing.

    Parameters
    ----------
    positions_cm : array_like, shape (n_positions,)
        Linear positions (cm); NaN where missing.
    centers_cm : array_like, shape (n_places,)
        The places' linear positions (cm).

    Returns
    -------
    index : np.ndarray of int, shape (n_positions,)
    """
    positions = np.asarray(positions_cm, dtype=float)
    centers = np.asarray(centers_cm, dtype=float)
    finite = np.isfinite(positions)
    index = np.full(positions.size, -1, dtype=np.int64)
    index[finite] = np.abs(positions[finite, None] - centers[None, :]).argmin(axis=1)
    return index


def epoch_place_rates(
    events: pd.DataFrame, centers_cm: ArrayLike, rates: ArrayLike
) -> pd.DataFrame:
    """One epoch's events, each with the rate ranks of its two decoded places.

    Parameters
    ----------
    events : pd.DataFrame, shape (n_events, n_columns)
        The epoch's events, with :data:`POSITION_COLUMNS` (linear cm, NaN where
        missing).
    centers_cm, rates : array_like, shape (n_places,)
        The epoch's real places and their ground intensities
        (:func:`real_places`).

    Returns
    -------
    pd.DataFrame, shape (n_events, 2)
        Indexed as ``events``: :data:`RANK_COLUMNS`, each position's nearest
        place's rank (:func:`place_ranks`), NaN where the position is missing.
    """
    require_columns(events, list(POSITION_COLUMNS), "the events to rank")
    ranks = place_ranks(np.asarray(rates, dtype=float))
    out = {}
    for column, rank_column in zip(POSITION_COLUMNS, RANK_COLUMNS, strict=True):
        index = nearest_places(events[column].to_numpy(dtype=float), centers_cm)
        out[rank_column] = np.where(index >= 0, ranks[index], np.nan)
    return pd.DataFrame(out, index=events.index)


_TABLE_COLUMNS = ["animal", "epoch_id", "group", *RANK_COLUMNS]


def low_rate_events(table: pd.DataFrame, cut: float = LOW_RATE_RANK) -> pd.Series:
    """Whether each event is a low-rate event: every one of its places present
    has a rank below ``cut`` (both, or the one when the other is missing).

    Parameters
    ----------
    table : pd.DataFrame, shape (n_events, n_columns)
        One row per event, with :data:`RANK_COLUMNS`.
    cut : float, optional

    Returns
    -------
    pd.Series of object, shape (n_events,)
        True or False, or NaN for an event with no place (no rank).
    """
    require_columns(table, list(RANK_COLUMNS), "the place-rate table")
    ranks = table[list(RANK_COLUMNS)]
    low = ((ranks < cut) | ranks.isna()).all(axis=1)
    return low.astype(object).where(ranks.notna().any(axis=1))


def low_rate_share_statistics(
    table: pd.DataFrame, cut: float = LOW_RATE_RANK
) -> pd.DataFrame:
    """For each group, the share of its events that are low-rate events
    (:func:`low_rate_events`).

    An epoch's share is the fraction of its ranked events (with at least one
    place) that are low-rate events; an animal's value is the mean over its
    epochs; the estimate is the mean over animals. An event without a place
    is left out.

    Parameters
    ----------
    table : pd.DataFrame, shape (n_events, n_columns)
        One row per event: ``animal``, ``epoch_id``, ``group`` (``"SWR"`` or
        ``"no SWR"``) and :data:`RANK_COLUMNS`. The caller selects the events
        (``cohort_selection.select_events_in``).
    cut : float, optional
        The rank below which a place is a low-rate place.

    Returns
    -------
    pd.DataFrame
        Indexed by group (sorted): ``share_animal_mean``, ``share_animal_min``,
        ``share_animal_max``, ``animals`` (the animals' shares, in sorted
        animal order), ``animal_names``, ``n_animals``, ``n_epochs``,
        ``n_events`` (ranked) and ``n_low_rate_events``.
    """
    require_columns(table, _TABLE_COLUMNS, "the place-rate table")
    low_all = low_rate_events(table, cut)
    ranked = table[low_all.notna()]
    rows = {}
    for group, mine in ranked.groupby("group", sort=True):
        low = low_all[mine.index].astype(float)
        per_animal = animal_session_values(
            low, mine["animal"], mine["epoch_id"], "mean"
        )
        rows[group] = {
            "share_animal_mean": float(per_animal.mean()),
            "share_animal_min": float(per_animal.min()),
            "share_animal_max": float(per_animal.max()),
            "animals": per_animal.to_numpy(),
            "animal_names": per_animal.index.to_numpy(),
            "n_animals": len(per_animal),
            "n_epochs": int(mine["epoch_id"].nunique()),
            "n_events": len(mine),
            "n_low_rate_events": int(low.sum()),
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("group")


def rank_tenth_shares(table: pd.DataFrame, n_bins: int = N_RANK_BINS) -> pd.DataFrame:
    """For each group, the share of its events' places in each equal bin of
    the rank.

    The bins are ``[k / n_bins, (k + 1) / n_bins)``, the last closed at 1.
    Each event counts its two places half each (its one place whole when the
    other is missing; an event without a place is left out), so an event's
    weight in a bin is the share of its places there, and the share expected
    by chance in a bin is the share of the track's places in it (about one in
    ``n_bins``). Each epoch's shares (the mean over its events of those
    weights), each animal's mean over its epochs and the mean over animals,
    as :func:`low_rate_share_statistics`; each animal's shares, and their
    mean, sum to 1 over the bins.

    Parameters
    ----------
    table : pd.DataFrame, shape (n_events, n_columns)
        As :func:`low_rate_share_statistics` takes it.
    n_bins : int, optional

    Returns
    -------
    pd.DataFrame, shape (n_groups * n_bins, n_animals + 1)
        Indexed by (``group``, ``rank_bin``: 0 the lowest rates), one column
        per animal (sorted) and ``animal_mean``. An animal with no event of a
        group has NaN in that group's rows and is left out of its mean.
    """
    require_columns(table, _TABLE_COLUMNS, "the place-rate table")
    ranks = table[list(RANK_COLUMNS)].to_numpy(dtype=float)
    present = np.isfinite(ranks)
    keep = present.any(axis=1)
    ranked, ranks, present = table[keep], ranks[keep], present[keep]
    rank_bin = np.digitize(np.nan_to_num(ranks), np.arange(1, n_bins) / n_bins)
    n_present = present.sum(axis=1)
    parts = []
    for group, mine in ranked.groupby("group", sort=True):
        rows = (ranked["group"] == group).to_numpy()
        for k in range(n_bins):
            weight = ((rank_bin[rows] == k) & present[rows]).sum(axis=1) / n_present[
                rows
            ]
            per_animal = animal_session_values(
                pd.Series(weight, index=mine.index),
                mine["animal"],
                mine["epoch_id"],
                "mean",
            )
            parts.append(per_animal.rename((group, k)))
    shares = pd.DataFrame(parts).sort_index(axis=1)
    shares.index = pd.MultiIndex.from_tuples(shares.index, names=["group", "rank_bin"])
    shares.columns.name = None
    shares["animal_mean"] = shares.mean(axis=1)
    return shares
