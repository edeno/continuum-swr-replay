"""Where the non-local time sits in ripple power and in MUA, per time bin.

Over an epoch's immobile time bins inside the well visits, the bins are counted
by the level of a signal (the z-scored Kay consensus ripple power,
``SWR_zscore``, and the z-scored multiunit rate, ``MUA_zscore``) and by their
non-local probability (Continuous plus Fragmented). The probability bins have
the non-local events' threshold (``DEFAULT_NON_LOCAL_THRESHOLD``, 0.8) as an
edge, so a bin decoded non-local (probability above 0.8) is a sum over the same
table as every other count: Figure 2i's share of bins decoded non-local, the
share of non-local time below the SWR threshold, and their MUA counterparts on
Figure 2's supplement cannot disagree.

:func:`rest_signal_counts` makes one epoch's table; :func:`nonlocal_counts`
sums it, for one signal, into the bins and the non-local bins of each signal
bin, the table ``event_statistics.animal_signal_counts`` reads.
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
from continuum_swr_replay.parameters import (
    DEFAULT_NON_LOCAL_THRESHOLD,
    HSE_ZSCORE_THRESHOLD,
)

RIPPLE_POWER_EDGES = np.concatenate([[-np.inf], np.arange(-2.0, 10.01, 0.25), [np.inf]])
"""The ripple-power bins, z: 0.25 z wide from -2 to 10 with an open bin at each
end, so every time bin is counted. The SWR threshold (2 z) is an edge."""

MUA_EDGES = RIPPLE_POWER_EDGES
"""The MUA bins, z, chosen as the ripple-power bins are: 0.25 z wide from -2 to
10 with an open bin at each end. :data:`MUA_LEVEL_Z` is an edge."""

MUA_LEVEL_Z = float(HSE_ZSCORE_THRESHOLD)
"""The MUA level the reported values split at (2 z): the threshold of the MUA
event detector (``parameters.HSE_ZSCORE_THRESHOLD``), the MUA counterpart of
the SWR threshold. That detector z-scores its own smoothed rate, so this is the
same number on a closely related scale, not its exact criterion."""

SIGNALS = {
    "ripple_power": ("SWR_zscore", RIPPLE_POWER_EDGES),
    "mua": ("MUA_zscore", MUA_EDGES),
}
"""Each signal counted: its decode-frame column and its bin edges."""

PROBABILITY_EDGES = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
"""The non-local probability bins, each holding the probabilities above its low
edge and up to its high edge (the first also holds 0, the last anything above
0.8). The non-local threshold, 0.8, is an edge, so the last bin is the bins
decoded non-local."""

assert DEFAULT_NON_LOCAL_THRESHOLD in PROBABILITY_EDGES
assert MUA_LEVEL_Z in MUA_EDGES


def count_by_signal_and_probability(
    signal: NDArray, probability: NDArray, edges: NDArray
) -> pd.DataFrame:
    """Time bins counted by signal bin and non-local probability bin.

    Parameters
    ----------
    signal : ndarray, shape (n_bins,)
        Each time bin's signal level.
    probability : ndarray, shape (n_bins,)
        Each time bin's non-local probability.
    edges : ndarray, shape (n_edges,)
        Ascending signal bin edges, each bin holding its low edge and not its
        high one; a time bin outside them is left out, as is one whose signal or
        probability is not finite.

    Returns
    -------
    counts : pd.DataFrame, shape ((n_edges - 1) * 5, 5)
        One row per signal bin and probability bin (:data:`PROBABILITY_EDGES`),
        signal bins in order and probability bins within each: ``signal_low``,
        ``signal_high``, ``probability_low``, ``probability_high`` and
        ``n_bins``.
    """
    signal = np.asarray(signal, dtype=float)
    probability = np.asarray(probability, dtype=float)
    edges = np.asarray(edges, dtype=float)
    keep = (
        np.isfinite(signal)
        & np.isfinite(probability)
        & (signal >= edges[0])
        & (signal <= edges[-1])
    )
    n_signal = edges.size - 1
    n_probability = PROBABILITY_EDGES.size - 1
    which_signal = np.clip(
        np.searchsorted(edges, signal[keep], side="right") - 1, 0, n_signal - 1
    )
    # above a low edge, up to and including the high edge
    which_probability = np.searchsorted(
        PROBABILITY_EDGES[1:-1], probability[keep], side="left"
    )
    n_bins = np.bincount(
        which_signal * n_probability + which_probability,
        minlength=n_signal * n_probability,
    )
    return pd.DataFrame(
        {
            "signal_low": np.repeat(edges[:-1], n_probability),
            "signal_high": np.repeat(edges[1:], n_probability),
            "probability_low": np.tile(PROBABILITY_EDGES[:-1], n_signal),
            "probability_high": np.tile(PROBABILITY_EDGES[1:], n_signal),
            "n_bins": n_bins,
        }
    )


def rest_signal_counts(dot: pd.DataFrame, in_visit: NDArray[np.bool_]) -> pd.DataFrame:
    """One epoch's immobile time bins inside the well visits, by signal level and
    non-local probability, for each signal of :data:`SIGNALS`.

    The bins are the immobile ones outside excluded time
    (``analyzed_time.immobile_analyzed_bins``) that lie inside a well visit
    (from the first poke-in at a well to the final poke-out), the time the
    analyzed events start in.

    Parameters
    ----------
    dot : pd.DataFrame
        One epoch's decode frame read through the SWR source
        (``analyzed_time.with_swr_source``), with ``head_speed``,
        ``in_excluded_time``, ``SWR_zscore``, ``MUA_zscore`` and the two
        non-local state probabilities.
    in_visit : NDArray of bool, shape (n_time,)
        Whether each bin is inside a well visit
        (``event_definition.well_visits.in_well_visit`` on the frame's times).

    Returns
    -------
    counts : pd.DataFrame
        :func:`count_by_signal_and_probability`'s rows of each signal, after a
        first column ``signal`` (the key of :data:`SIGNALS`). A bin without a
        finite level or probability is in none of that signal's rows.

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
    return pd.concat(
        [
            count_by_signal_and_probability(
                dot[column].to_numpy()[rest], probability, edges
            ).assign(signal=name)
            for name, (column, edges) in SIGNALS.items()
        ],
        ignore_index=True,
    )[
        [
            "signal",
            "signal_low",
            "signal_high",
            "probability_low",
            "probability_high",
            "n_bins",
        ]
    ]


def nonlocal_counts(counts: pd.DataFrame, signal: str) -> pd.DataFrame:
    """One signal's time bins, and those decoded non-local, by signal bin.

    Parameters
    ----------
    counts : pd.DataFrame
        Rows of :func:`rest_signal_counts` (one or many epochs), with any
        columns that name an epoch (``epoch_id``, ``animal``, ...) before
        ``signal``.
    signal : str
        A key of :data:`SIGNALS`.

    Returns
    -------
    pd.DataFrame
        One row per epoch and signal bin: the epoch's columns, ``signal_low``,
        ``signal_high``, ``n_bins`` (over every probability bin) and
        ``n_nonlocal`` (over the bins above ``DEFAULT_NON_LOCAL_THRESHOLD``).

    Raises
    ------
    ValueError
        If ``signal`` has no rows.
    """
    mine = counts[counts["signal"] == signal]
    if mine.empty:
        raise ValueError(f"the counts hold no {signal!r} rows")
    keys = [
        column
        for column in mine.columns
        if column not in ("signal", "probability_low", "probability_high", "n_bins")
    ]
    nonlocal_ = mine["probability_low"] >= DEFAULT_NON_LOCAL_THRESHOLD
    return (
        mine.assign(n_nonlocal=np.where(nonlocal_, mine["n_bins"], 0))
        .groupby(keys, sort=False, as_index=False)[["n_bins", "n_nonlocal"]]
        .sum()
    )
