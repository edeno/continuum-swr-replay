"""Tetrode participation and spike rate of non-local events, against Local rest.

Are the non-local events population events? For each canonical event, and for
one control window of the same length drawn from Local rest inside a well visit
of the same epoch:

- **participation**: the share of the epoch's tetrodes with at least one spike;
- **spike rate**: spikes per second, summed over tetrodes.

Local rest is a bin the decoder calls Local (above :data:`LOCAL_THRESHOLD`), with
the animal immobile and outside excluded time (``analyzed_time``), and no SWR or
non-local event within :data:`MARGIN_S`.

The control rule (:data:`CONTROL_RULE`): a control window lies wholly inside one
well visit, every bin of it Local rest and inside the same visit
(``event_definition.well_visits``: from the first poke-in at a well to the final
poke-out, ``arrival <= t < departure``). Windows are drawn as
:func:`~.state_evidence.windows.matched_windows` draws a bout's (a seeded order
of every valid start per length). Every event gets one by the same rule,
wherever the event itself starts; an event longer than any stretch of Local
rest inside one visit of its epoch has no control.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from continuum_swr_replay.analysis.event_evidence.state_evidence.windows import (
    Seed,
    _valid_starts,
    keyed_rng,
)
from continuum_swr_replay.analysis.shared_statistics import animal_session_values
from continuum_swr_replay.intervals import sample_step

LOCAL_THRESHOLD = 0.8
"""A Local-rest bin's Local probability is above this (the event rule's 0.8)."""

MARGIN_S = 0.1
"""Local rest keeps this far (s) from every SWR and non-local event."""

LOCAL_REST = "local_rest"
"""The regime label of a Local-rest bin, for ``matched_windows``."""

CONTROL_RULE = (
    "a control window has the event's number of decode bins and lies wholly inside "
    "one well visit (first poke-in to final poke-out): every bin of it is Local "
    f"rest (Local probability above {LOCAL_THRESHOLD}, immobile, outside excluded "
    f"time, at least {MARGIN_S} s from every SWR and non-local event) and in the "
    "same visit; drawn for every event of the epoch, a seeded order of every valid "
    "start per length"
)
"""The control windows' rule, as the run's settings record it."""

MEASURES = ("participation", "spike_rate_hz")
"""The per-window measures :func:`participation_statistics` summarizes."""


def local_rest_mask(
    local_probability: NDArray,
    immobile_analyzed: NDArray,
    in_event: NDArray,
    *,
    margin_bins: int,
) -> NDArray[np.bool_]:
    """The bins of Local rest.

    Parameters
    ----------
    local_probability : ndarray, shape (n_time,)
        The decoder's Local probability.
    immobile_analyzed : ndarray of bool, shape (n_time,)
        Immobile and outside excluded time (``immobile_analyzed_bins``).
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
        & np.asarray(immobile_analyzed, dtype=bool)
        & ~near
    )


def control_starts(
    local_rest: NDArray, visit_index: NDArray, length: int
) -> NDArray[np.int64]:
    """Every first bin of a ``length``-bin window wholly in Local rest and one visit.

    Parameters
    ----------
    local_rest : ndarray of bool, shape (n_time,)
        :func:`local_rest_mask`.
    visit_index : ndarray of int, shape (n_time,)
        Each bin's well visit, -1 outside every visit
        (``well_visits.well_visit_index`` on the decode times).
    length : int
        The window's number of bins.

    Returns
    -------
    ndarray of int
        The valid first bins, ascending.
    """
    visit_index = np.asarray(visit_index, dtype=np.int64)
    candidate = np.asarray(local_rest, dtype=bool) & (visit_index >= 0)
    starts = _valid_starts(candidate, length)
    return starts[visit_index[starts] == visit_index[starts + length - 1]]


def _assign_windows(
    s: NDArray, e: NDArray, starts_of_length: Callable[[int], NDArray], seed: Seed
) -> pd.DataFrame:
    """Each event's window from the valid starts of its length, or -1.

    For each length, the valid starts are put in one seeded random order, and
    the events of that length take them in that order, by start
    (``matched_windows``' rule).
    """
    s, e = np.asarray(s, dtype=np.int64), np.asarray(e, dtype=np.int64)
    lengths = e - s + 1
    starts = np.full(s.size, -1, dtype=np.int64)
    for length in np.unique(lengths):
        rows = np.flatnonzero(lengths == length)
        rows = rows[np.argsort(s[rows], kind="stable")]
        order = keyed_rng(seed, LOCAL_REST, int(length), "baseline").permutation(
            starts_of_length(int(length))
        )
        n = min(rows.size, order.size)
        starts[rows[:n]] = order[:n]
    return pd.DataFrame(
        {"s": starts, "e": np.where(starts >= 0, starts + lengths - 1, -1)}
    )


def control_windows(
    s: NDArray, e: NDArray, local_rest: NDArray, visit_index: NDArray, *, seed: Seed
) -> pd.DataFrame:
    """One window of each event's length inside one well visit, or none.

    The windows of each length lying wholly in Local rest and in one well visit
    (:func:`control_starts`) are put in one seeded random order, and the events
    of that length take them in that order, by start. Events left without one
    get -1.

    Parameters
    ----------
    s, e : ndarray of int, shape (n_events,)
        Each event's first and last decode bin.
    local_rest : ndarray of bool, shape (n_time,)
        :func:`local_rest_mask`.
    visit_index : ndarray of int, shape (n_time,)
        Each bin's well visit, -1 outside every visit.
    seed : Seed
        The draw's seed (``keyed_rng``).

    Returns
    -------
    pd.DataFrame
        Aligned with the events: ``s`` and ``e``, the window's first and last
        bin, -1 for an event with no control.
    """
    return _assign_windows(
        s, e, lambda length: control_starts(local_rest, visit_index, length), seed
    )


def windows_in_local_rest_anywhere(
    s: NDArray, e: NDArray, local_rest: NDArray, *, seed: Seed
) -> pd.DataFrame:
    """The windows Local rest alone would give, anywhere in the epoch.

    Drawn as :func:`control_windows` draws, without the well visits: the run
    counts the events this leaves without a window beside those the control
    rule leaves without one, so that what the visit restriction costs is
    recorded. These windows are not controls.

    Returns
    -------
    pd.DataFrame
        As :func:`control_windows`.
    """
    mask = np.asarray(local_rest, dtype=bool)
    return _assign_windows(s, e, lambda length: _valid_starts(mask, length), seed)


def window_stops(time: NDArray, last: NDArray) -> NDArray:
    """Each window's end: the decode time after its last bin.

    A window holds the spikes of ``[time[first], time[last + 1])``, so its last
    bin keeps the spikes up to the next decoder timestamp (the spike-evidence
    run's convention); a window ending on the epoch's last bin ends one median
    bin later.
    """
    time = np.asarray(time, dtype=float)
    last = np.asarray(last, dtype=np.int64)
    step = sample_step(time)
    padded = np.r_[time, time[-1] + step]
    return padded[last + 1]


def tetrode_spike_counts(
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
