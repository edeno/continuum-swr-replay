"""Per-bin facts of a decode: the spikes of each bin and whether it is inside a
well visit.

The restriction scores (:mod:`.nonlocal_restriction`) are per decode bin; to read
them per spike, or only inside a well visit, each bin needs its spike count over
every tetrode and its well-visit flag. :func:`bin_facts` makes that table, one
row per bin of the decode's time grid:

- ``n_spikes``: the spikes of every tetrode the decoder puts in the bin. A
  spike at ``st`` with ``time[0] <= st <= time[-1]`` goes to bin
  ``np.digitize(st, time[1:-1])``, as ``state_evidence.terms.spikes_in_bins``
  bins them (the last bin takes the spikes from ``time[-2]`` to ``time[-1]``
  inclusive, so the final sample's bin is never reached); spikes outside
  ``[time[0], time[-1]]`` are not counted.
- ``in_well_visit``: the bin's time is inside a well visit
  (``event_definition.well_visits.in_well_visit``: first poke-in at a well <= t
  < final poke-out there, one port experience).

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

BIN_FACTS_COLUMNS = ("time", "n_spikes", "in_well_visit")
"""The columns of a per-bin facts table, in order."""


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
    time: NDArray, spike_times: Sequence[NDArray], visits: tuple[NDArray, NDArray]
) -> pd.DataFrame:
    """One epoch's per-bin facts: each decode bin's spikes and well-visit flag.

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

    Returns
    -------
    pd.DataFrame
        One row per bin, in time order: :data:`BIN_FACTS_COLUMNS`, ``time``
        (float64 s), ``n_spikes`` (int32) and ``in_well_visit`` (bool).
    """
    time = np.asarray(time, dtype=float)
    counts = decoder_bin_spike_counts(spike_times, time)
    return pd.DataFrame(
        {
            "time": time,
            "n_spikes": counts.astype(np.int32),
            "in_well_visit": in_well_visit(time, *visits),
        }
    )
