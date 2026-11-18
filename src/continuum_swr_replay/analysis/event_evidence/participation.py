"""Tetrode participation and spike rate of non-local events, against Local rest.

Are the non-local events population events? For each canonical event, and for
one control window of the same length drawn from Local rest in the same epoch:

- **participation**: the share of the epoch's tetrodes with at least one spike;
- **spike rate**: spikes per second, summed over tetrodes.

Local rest is a bin the decoder calls Local (above :data:`LOCAL_THRESHOLD`), with
the animal immobile and outside excluded time (``analysed_time``), and no SWR or
non-local event within :data:`MARGIN_S`. A control window is every bin of Local
rest, drawn as :func:`~.state_evidence.windows.matched_windows` draws a bout's
(a seeded order of every valid start per length); an event longer than any
Local-rest stretch of its epoch has no control.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from continuum_swr_replay.analysis.event_evidence.state_evidence.windows import (
    Seed,
    _valid_starts,
    keyed_rng,
)
from continuum_swr_replay.analysis.shared_statistics import animal_session_values

LOCAL_THRESHOLD = 0.8
"""A Local-rest bin's Local probability is above this (the event rule's 0.8)."""

MARGIN_S = 0.1
"""Local rest keeps this far (s) from every SWR and non-local event."""

ON_GRID_ATOL_S = 1e-6
"""How far (s) an event's first or last time may be from its decode bin."""

LOCAL_REST = "local_rest"
"""The regime label of a Local-rest bin, for ``matched_windows``."""

MEASURES = ("participation", "spike_rate_hz")
"""The per-window measures :func:`participation_statistics` summarizes."""


def event_bins(
    time: NDArray, starts_s: NDArray, ends_s: NDArray
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Each event's first and last decode bin.

    Parameters
    ----------
    time : ndarray, shape (n_time,)
        The decode's time axis, s.
    starts_s, ends_s : ndarray, shape (n_events,)
        The events' first and last labelled bins' times (the event table's
        ``event_start_s`` and ``event_end_s``).

    Returns
    -------
    s, e : ndarray of int, shape (n_events,)

    Raises
    ------
    ValueError
        If an event does not start and end on a decode bin.
    """
    time = np.asarray(time, dtype=float)
    s = np.minimum(np.searchsorted(time, starts_s), time.size - 1)
    e = np.minimum(np.searchsorted(time, ends_s), time.size - 1)
    on_grid = (np.abs(time[s] - starts_s) <= ON_GRID_ATOL_S) & (
        np.abs(time[e] - ends_s) <= ON_GRID_ATOL_S
    )
    if not on_grid.all():
        raise ValueError(
            f"{int((~on_grid).sum())} events do not start and end on the decode's "
            "time bins"
        )
    return s.astype(np.int64), e.astype(np.int64)


def local_rest_mask(
    local_probability: NDArray,
    immobile_analysed: NDArray,
    in_event: NDArray,
    *,
    margin_bins: int,
) -> NDArray[np.bool_]:
    """The bins of Local rest.

    Parameters
    ----------
    local_probability : ndarray, shape (n_time,)
        The decoder's Local probability.
    immobile_analysed : ndarray of bool, shape (n_time,)
        Immobile and outside excluded time (``immobile_analysed_bins``).
    in_event : ndarray of bool, shape (n_time,)
        In an SWR or a non-local event.
    margin_bins : int
        A bin this close to an event bin is not Local rest.

    Returns
    -------
    ndarray of bool, shape (n_time,)
    """
    in_event = np.asarray(in_event, dtype=bool)
    near = (
        np.convolve(
            in_event.astype(np.int64),
            np.ones(2 * margin_bins + 1, dtype=np.int64),
            "same",
        )
        > 0
    )
    return (
        (np.asarray(local_probability) > LOCAL_THRESHOLD)
        & np.asarray(immobile_analysed, dtype=bool)
        & ~near
    )


def control_windows(
    s: NDArray, e: NDArray, local_rest: NDArray, *, seed: Seed
) -> pd.DataFrame:
    """One Local-rest window of each event's length, or none.

    For each length, the windows of that length lying wholly in Local rest are
    put in one seeded random order, and the events of that length take them in
    that order, by start (``matched_windows``' rule). Events left without one
    get -1.

    Returns
    -------
    pd.DataFrame
        Aligned with the events: ``s`` and ``e``, the window's first and last
        bin, -1 for an event with no control.
    """
    s, e = np.asarray(s, dtype=np.int64), np.asarray(e, dtype=np.int64)
    lengths = e - s + 1
    mask = np.asarray(local_rest, dtype=bool)
    starts = np.full(s.size, -1, dtype=np.int64)
    for length in np.unique(lengths):
        rows = np.flatnonzero(lengths == length)
        rows = rows[np.argsort(s[rows], kind="stable")]
        order = keyed_rng(seed, LOCAL_REST, int(length), "baseline").permutation(
            _valid_starts(mask, int(length))
        )
        n = min(rows.size, order.size)
        starts[rows[:n]] = order[:n]
    return pd.DataFrame(
        {"s": starts, "e": np.where(starts >= 0, starts + lengths - 1, -1)}
    )


def window_stops(time: NDArray, last: NDArray) -> NDArray:
    """Each window's end: the decode time after its last bin.

    A window holds the spikes of ``[time[first], time[last + 1])``, so its last
    bin keeps the spikes up to the next decoder timestamp (the spike-evidence
    run's convention); a window ending on the epoch's last bin ends one median
    bin later.
    """
    time = np.asarray(time, dtype=float)
    last = np.asarray(last, dtype=np.int64)
    step = float(np.median(np.diff(time)))
    padded = np.r_[time, time[-1] + step]
    return padded[last + 1]


def window_spike_counts(
    spike_times: Sequence[NDArray], starts_s: NDArray, stops_s: NDArray
) -> NDArray[np.int64]:
    """Each tetrode's spikes in each half-open window [start, stop).

    Parameters
    ----------
    spike_times : sequence of ndarray
        One sorted array of spike times (s) per tetrode.
    starts_s, stops_s : ndarray, shape (n_windows,)

    Returns
    -------
    counts : ndarray of int, shape (n_windows, n_tetrodes)
    """
    counts = np.empty((len(starts_s), len(spike_times)), dtype=np.int64)
    for k, times in enumerate(spike_times):
        times = np.asarray(times)
        counts[:, k] = np.searchsorted(times, stops_s, side="left") - np.searchsorted(
            times, starts_s, side="left"
        )
    return counts


def participation_and_rate(counts: NDArray, duration_s: NDArray) -> pd.DataFrame:
    """Participation and spike rate of each window from its tetrode counts.

    Parameters
    ----------
    counts : ndarray of int, shape (n_windows, n_tetrodes)
    duration_s : ndarray, shape (n_windows,)

    Returns
    -------
    pd.DataFrame, shape (n_windows, 5)
        ``n_spikes``, ``n_active_tetrodes``, ``n_tetrodes``, ``participation``
        (active over recorded tetrodes) and ``spike_rate_hz`` (spikes per second,
        summed over tetrodes).
    """
    counts = np.asarray(counts)
    n_spikes = counts.sum(1)
    active = (counts > 0).sum(1)
    return pd.DataFrame(
        {
            "n_spikes": n_spikes,
            "n_active_tetrodes": active,
            "n_tetrodes": counts.shape[1],
            "participation": active / counts.shape[1],
            "spike_rate_hz": n_spikes / np.asarray(duration_s, dtype=float),
        }
    )


def participation_statistics(table: pd.DataFrame) -> pd.DataFrame:
    """Each animal's participation and spike rate, by event group and window kind.

    An animal's value is the median over its epochs of each epoch's median
    (``animal_session_values``); the summary is the mean over animals.

    Parameters
    ----------
    table : pd.DataFrame
        One row per window: ``animal``, ``epoch_id``, ``group`` (``"SWR"`` or
        ``"no SWR"``), ``kind`` (``"event"`` or ``"control"``) and
        :data:`MEASURES`.

    Returns
    -------
    statistics : pd.DataFrame
        Indexed by (group, kind): for each measure, ``{measure}_animals`` (the
        animals' values, in sorted animal order) and
        ``{measure}_animal_mean``; ``animal_names`` and ``n_animals``.
    """
    rows = {}
    for (group, kind), mine in table.groupby(["group", "kind"]):
        row = {}
        for measure in MEASURES:
            per_animal = animal_session_values(
                mine[measure], mine["animal"], mine["epoch_id"], "median"
            )
            row[f"{measure}_animals"] = per_animal.to_numpy()
            row[f"{measure}_animal_mean"] = float(per_animal.mean())
        row["animal_names"] = per_animal.index.to_numpy()
        row["n_animals"] = len(per_animal)
        rows[(group, kind)] = row
    statistics = pd.DataFrame.from_dict(rows, orient="index")
    statistics.index.names = ["group", "kind"]
    return statistics


def participation_differences(statistics: pd.DataFrame) -> pd.DataFrame:
    """For each event group, how many animals' events exceed their Local rest.

    Parameters
    ----------
    statistics : pd.DataFrame
        :func:`participation_statistics`.

    Returns
    -------
    pd.DataFrame
        Indexed by group: for each measure, ``{measure}_n_animals_above`` (the
        animals whose events' value is above their controls'); and ``n_animals``
        (the animals with both).
    """
    rows = {}
    for group in statistics.index.get_level_values("group").unique():
        event, control = (
            statistics.loc[(group, "event")],
            statistics.loc[(group, "control")],
        )
        row = {}
        for measure in MEASURES:
            above = pd.Series(
                event[f"{measure}_animals"], index=event["animal_names"]
            ) - pd.Series(control[f"{measure}_animals"], index=control["animal_names"])
            above = above.dropna()
            row[f"{measure}_n_animals_above"] = int((above > 0).sum())
        row["n_animals"] = len(above)
        rows[group] = row
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("group")
