"""Ripple power and MUA of confident Local time against confident non-local time.

Over an epoch's immobile, analyzed time bins inside the well visits (the time
of ``nonlocal_by_signal``), the bins are counted by the level of a signal
(``nonlocal_by_signal.SIGNALS``: ripple power, MUA; the same bins) and by the
decoder's state class (:func:`state_classes`: Local, non-local or No-Spike
probability above 0.8, or none of them). From the counts: how each class's time
is spread over the signal (:func:`state_distribution`), and the share of the
non-local and of the Local time at or above the signal's detection level
(:func:`detection_level_statistics`). Every statistic is each epoch's, then
each animal's mean over its epochs, then the mean over animals with the
smallest and largest animal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from continuum_swr_replay.analysis.event_definition.analyzed_time import (
    immobile_analyzed_bins,
)
from continuum_swr_replay.analysis.event_definition.nonlocal_event_rule import (
    nonlocal_probability,
)
from continuum_swr_replay.analysis.event_properties.nonlocal_by_signal import SIGNALS
from continuum_swr_replay.parameters import DEFAULT_NON_LOCAL_THRESHOLD

STATE_THRESHOLD = float(DEFAULT_NON_LOCAL_THRESHOLD)
"""A time bin is in a state class when that class's probability is above this."""
STATE_CLASSES = ("local", "non_local", "no_spike", "other")
"""The state classes, in order: Local, non-local (Continuous plus Fragmented)
or No-Spike probability above :data:`STATE_THRESHOLD`, or none of them."""
EPOCH_COLUMNS = ("animal", "session_date", "epoch")
"""The columns that name an epoch in the stored counts."""


def state_classes(dot: pd.DataFrame) -> NDArray[np.str_]:
    """Each time bin's state class (:data:`STATE_CLASSES`); the probabilities
    add to 1, so at most one is above :data:`STATE_THRESHOLD`."""
    nonlocal_ = nonlocal_probability(dot).to_numpy()
    return np.select(
        [
            dot["Local_prob"].to_numpy() > STATE_THRESHOLD,
            nonlocal_ > STATE_THRESHOLD,
            dot["No-Spike_prob"].to_numpy() > STATE_THRESHOLD,
        ],
        list(STATE_CLASSES[:3]),
        STATE_CLASSES[3],
    )


def state_signal_counts(dot: pd.DataFrame, in_visit: NDArray[np.bool_]) -> pd.DataFrame:
    """One epoch's immobile, analyzed time bins inside the well visits, by
    signal, signal bin and state class.

    The bins and the signal bins are ``nonlocal_by_signal.rest_signal_counts``':
    immobile outside excluded time, inside a well visit; a signal bin holds its
    low edge and not its high one; a time bin without a finite signal level or
    non-local probability is in none of that signal's rows.

    Parameters
    ----------
    dot : pd.DataFrame
        One epoch's decode frame read through the SWR source, with
        ``head_speed``, ``in_excluded_time``, the signals of
        ``nonlocal_by_signal.SIGNALS`` and the four state probabilities.
    in_visit : NDArray of bool, shape (n_time,)
        Whether each bin is inside a well visit.

    Returns
    -------
    counts : pd.DataFrame
        One row per signal, signal bin and state class: ``signal``,
        ``signal_low``, ``signal_high``, ``state`` and ``n_bins``.

    Raises
    ------
    ValueError
        If ``in_visit`` is not one bool per bin.
    """
    in_visit = np.asarray(in_visit)
    if in_visit.dtype != bool or in_visit.shape != (len(dot),):
        raise ValueError(
            f"in_visit must be one bool per bin, shape ({len(dot)},); got "
            f"{in_visit.dtype} of shape {in_visit.shape}"
        )
    rest = immobile_analyzed_bins(dot) & in_visit
    probability = nonlocal_probability(dot).to_numpy()[rest]
    state = state_classes(dot)[rest]
    n_states = len(STATE_CLASSES)
    which_state = pd.Categorical(state, categories=STATE_CLASSES).codes
    tables = []
    for name, (column, edges) in SIGNALS.items():
        signal = dot[column].to_numpy(dtype=float)[rest]
        keep = (
            np.isfinite(signal)
            & np.isfinite(probability)
            & (signal >= edges[0])
            & (signal <= edges[-1])
        )
        n_signal = edges.size - 1
        which_signal = np.clip(
            np.searchsorted(edges, signal[keep], side="right") - 1, 0, n_signal - 1
        )
        n_bins = np.bincount(
            which_signal * n_states + which_state[keep],
            minlength=n_signal * n_states,
        )
        tables.append(
            pd.DataFrame(
                {
                    "signal": name,
                    "signal_low": np.repeat(edges[:-1], n_states),
                    "signal_high": np.repeat(edges[1:], n_states),
                    "state": np.tile(STATE_CLASSES, n_signal),
                    "n_bins": n_bins.astype(int),
                }
            )
        )
    return pd.concat(tables, ignore_index=True)


def _epoch_table(counts: pd.DataFrame, signal: str) -> pd.DataFrame:
    """One signal's counts as a table per epoch and signal bin, a column per
    state class (``local`` and ``non_local`` at least)."""
    mine = counts[counts["signal"] == signal]
    if mine.empty:
        raise ValueError(f"the counts hold no {signal!r} rows")
    return (
        mine.pivot_table(
            index=[*EPOCH_COLUMNS, "signal_low", "signal_high"],
            columns="state",
            values="n_bins",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(columns=list(STATE_CLASSES), fill_value=0)
        .sort_index()
    )


def state_distribution(
    counts: pd.DataFrame, signal: str, states: tuple[str, ...] = ("local", "non_local")
) -> pd.DataFrame:
    """Each animal's distribution of each state class's time over the signal.

    Parameters
    ----------
    counts : pd.DataFrame
        The stored rows of every epoch: :data:`EPOCH_COLUMNS` and
        :func:`state_signal_counts`' columns.
    signal : str
        A key of ``nonlocal_by_signal.SIGNALS``.
    states : tuple of str, optional
        The state classes.

    Returns
    -------
    pd.DataFrame
        One row per state, animal and signal bin: ``state``, ``animal``,
        ``signal_low``, ``signal_high`` and ``percent``, the mean over the
        animal's epochs of each epoch's percent of its time in that class in
        the bin (an epoch with no time in the class is left out).
    """
    table = _epoch_table(counts, signal)
    epochs = list(EPOCH_COLUMNS)
    rows = []
    for state in states:
        n = table[state]
        total = n.groupby(level=epochs).transform("sum")
        share = (n / total.where(total > 0)).dropna()
        per_animal = share.groupby(level=["animal", "signal_low", "signal_high"]).mean()
        rows.append(
            per_animal.rename("percent")
            .mul(100)
            .reset_index()
            .assign(state=state)[
                ["state", "animal", "signal_low", "signal_high", "percent"]
            ]
        )
    return pd.concat(rows, ignore_index=True)


LEVEL_MEASURES = ("nonlocal_share_at_level", "local_share_at_level")
"""The measures of :func:`detection_level_statistics`, in order."""


def detection_level_statistics(
    counts: pd.DataFrame, signal: str, level: float
) -> pd.DataFrame:
    """The share of the non-local and of the Local time at or above a level.

    For each epoch with Local and non-local time, each class's share of its
    time whose signal bin's low edge is at or above ``level``; each animal's
    value is the mean over its epochs; the statistics are the mean over
    animals with the smallest and largest animal.

    Parameters
    ----------
    counts : pd.DataFrame
        As :func:`state_distribution` takes them.
    signal : str
        A key of ``nonlocal_by_signal.SIGNALS``.
    level : float
        The detection level (z), an edge of the signal's bins.

    Returns
    -------
    pd.DataFrame
        Indexed by :data:`LEVEL_MEASURES`: ``animal_mean``, ``animal_min``,
        ``animal_max``, ``n_animals`` and ``n_epochs``.

    Raises
    ------
    ValueError
        If ``level`` is not an edge of the signal's bins.
    """
    table = _epoch_table(counts, signal)
    low = table.index.get_level_values("signal_low")
    if level not in set(low):
        raise ValueError(f"{level} is not an edge of the {signal} bins")
    epochs = list(EPOCH_COLUMNS)
    above = table[["non_local", "local"]].mul(np.asarray(low >= level), axis=0)
    totals = table[["non_local", "local"]].groupby(level=epochs).sum()
    shares = above.groupby(level=epochs).sum() / totals.where(totals > 0)
    per_epoch = shares.dropna().set_axis(list(LEVEL_MEASURES), axis=1)
    per_animal = per_epoch.groupby(level="animal").mean()
    return pd.DataFrame(
        {
            "animal_mean": per_animal.mean(),
            "animal_min": per_animal.min(),
            "animal_max": per_animal.max(),
            "n_animals": len(per_animal),
            "n_epochs": len(per_epoch),
        }
    ).rename_axis("measure")
