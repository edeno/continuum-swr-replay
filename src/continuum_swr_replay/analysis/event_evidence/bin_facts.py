"""Per-bin facts of a decode: the spikes of each bin, whether it is inside a
well visit, and whether SWR detection was observed there.

The restriction scores (:mod:`.nonlocal_restriction`) are per decode bin; to read
them per spike, only inside a well visit, or only where an SWR could have been
seen, each bin needs its spike count over every tetrode, its well-visit flag
and its SWR-observed flag. :func:`bin_facts` makes that table, one row per bin
of the decode's time grid:

- ``n_spikes``: the spikes of every tetrode the decoder puts in the bin. A
  spike at ``st`` with ``time[0] <= st <= time[-1]`` goes to bin
  ``np.digitize(st, time[1:-1])``, as ``state_evidence.terms.spikes_in_bins``
  bins them (the last bin takes the spikes from ``time[-2]`` to ``time[-1]``
  inclusive, so the final sample's bin is never reached); spikes outside
  ``[time[0], time[-1]]`` are not counted.
- ``in_well_visit``: the bin's time is inside a well visit
  (``event_definition.well_visits.in_well_visit``: first poke-in at a well <= t
  < final poke-out there, one port experience).
- ``swr_observed``: SWR detection was observed at the bin's time (the decode
  frame's ``swr_observed``, read through the SWR source:
  ``analyzed_time.swr_observed_mask``, the closed stretches of
  ``swr_source.epoch_source(epoch_id).observed``). Where it was not (a gap in
  the stored ripple band), a bin outside every SWR of the source is not known
  to be outside an SWR: ``restriction_windows.outside_swr_bins`` leaves it out.

Nothing here reads an event table: the facts are the same for every event
cohort.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from continuum_swr_replay.analysis.event_definition.well_visits import in_well_visit
from continuum_swr_replay.intervals import decoder_time_bin

BIN_FACTS_COLUMNS = ("time", "n_spikes", "in_well_visit", "swr_observed")
"""The columns of a per-bin facts table, in order."""


def require_swr_observed(facts: pd.DataFrame, what: str = "the per-bin facts") -> None:
    """Refuse per-bin facts (or sums made from them) without ``swr_observed``:
    they were made before the flag was measured.

    Raises
    ------
    ValueError
        If ``facts`` has no ``swr_observed`` column.
    """
    if "swr_observed" not in facts.columns:
        raise ValueError(
            f"{what} have no swr_observed column: they were made before SWR "
            "detection's coverage was recorded per bin. Rerun "
            "scripts/event_evidence/measure_bin_facts.py and the runs made from "
            "it (docs/reproduce.md, Parts 1 and 2)"
        )


def decoder_bin_spike_counts(
    spike_times: Sequence[NDArray], time: NDArray
) -> NDArray[np.int64]:
    """The spikes of every tetrode in each decode bin, binned as the decoder
    bins them.

    Parameters
    ----------
    spike_times : sequence of np.ndarray, shape (n_tetrodes,) of (n_spikes_e,)
        Each tetrode's spike times (s), on the decode's clock, in any order.
    time : np.ndarray, shape (n_time,)
        The decode's time grid (s), ascending, at least two samples.

    Returns
    -------
    np.ndarray of int64, shape (n_time,)
        Spikes per bin over every tetrode: a spike at ``st`` with
        ``time[0] <= st <= time[-1]`` counts in bin
        ``np.digitize(st, time[1:-1])``; others are not counted. The last bin
        is always 0.

    Raises
    ------
    ValueError
        If ``time`` has fewer than two samples or is not ascending.
    """
    time = np.asarray(time, dtype=float)
    if time.ndim != 1 or time.size < 2 or (np.diff(time) <= 0).any():
        raise ValueError("time must be one-dimensional, ascending, of two samples")
    counts = np.zeros(time.size, dtype=np.int64)
    for st in spike_times:
        st = np.asarray(st, dtype=float)
        inside = st[(st >= time[0]) & (st <= time[-1])]
        counts += np.bincount(decoder_time_bin(inside, time), minlength=time.size)
    return counts


def bin_facts(
    time: NDArray,
    spike_times: Sequence[NDArray],
    visits: tuple[NDArray, NDArray],
    swr_observed: NDArray,
) -> pd.DataFrame:
    """One epoch's per-bin facts: each decode bin's spikes, well-visit flag and
    SWR-observed flag.

    Parameters
    ----------
    time : np.ndarray, shape (n_time,)
        The decode's time grid (s).
    spike_times : sequence of np.ndarray
        Each tetrode's spike times (s), as :func:`decoder_bin_spike_counts`.
    visits : tuple of np.ndarray, each shape (n_visits,)
        The epoch's well visits, ``(arrival_times, departure_times)`` (s,
        ``arrival_times`` ascending), as ``prepare_data.well_visit_times``
        gives them.
    swr_observed : np.ndarray of bool, shape (n_time,)
        Whether SWR detection was observed at each bin (the decode frame's
        ``swr_observed``, read through the SWR source).

    Returns
    -------
    pd.DataFrame
        One row per bin, in time order: :data:`BIN_FACTS_COLUMNS`, ``time``
        (float64 s), ``n_spikes`` (int32), ``in_well_visit`` and
        ``swr_observed`` (bool).

    Raises
    ------
    ValueError
        If ``swr_observed`` is not one value per bin.
    """
    time = np.asarray(time, dtype=float)
    swr_observed = np.asarray(swr_observed, dtype=bool)
    if swr_observed.shape != time.shape:
        raise ValueError("swr_observed is not one value per bin of the time grid")
    counts = decoder_bin_spike_counts(spike_times, time)
    return pd.DataFrame(
        {
            "time": time,
            "n_spikes": counts.astype(np.int32),
            "in_well_visit": in_well_visit(time, *visits),
            "swr_observed": swr_observed,
        }
    )
