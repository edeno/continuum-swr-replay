"""How the immobile time inside the well visits is spread over non-local probability.

The decoder's non-local probability (Continuous plus Fragmented,
``nonlocal_event_rule.nonlocal_probability``) of the immobile, analyzed time
bins inside a well visit, counted per epoch in equal bins from 0 to 1
(:func:`probability_counts`, written by
``scripts/decoding/measure_nonlocal_probability.py``). Almost all the time sits
near 0 or near 1, so the event rule's cut (0.8) is not a fine choice: the
statistics are each animal's distribution (:func:`animal_probability_distribution`)
and the share of the time below 0.2, between 0.2 and 0.8, and above 0.8
(:func:`probability_share_statistics`), each the mean over an animal's epochs
of each epoch's share, then over animals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from continuum_swr_replay.analysis.event_definition.analyzed_time import rest_bins
from continuum_swr_replay.analysis.event_definition.nonlocal_event_rule import (
    nonlocal_probability,
)
from continuum_swr_replay.analysis.shared_statistics import animal_range
from continuum_swr_replay.parameters import DEFAULT_NON_LOCAL_THRESHOLD
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS

PROBABILITY_BIN_EDGES = np.arange(51) / 50
"""The probability bins: 50 equal bins from 0 to 1, each holding the
probabilities above its low edge and up to its high edge (the first also holds
0). 0.2 and the event rule's cut, 0.8, are edges."""

LOW_CUT = 0.2
"""Below it (or at it) the time counts as clearly not non-local."""
HIGH_CUT = float(DEFAULT_NON_LOCAL_THRESHOLD)
"""The event rule's cut (0.8): above it the time counts as non-local."""

SHARES = {
    "low": "at most 0.2",
    "middle": "above 0.2 and at most 0.8",
    "high": "above 0.8",
}
"""The three shares of the time, by their word in a key: what each holds."""

EPOCH_COLUMNS = SESSION_COLUMNS
"""The columns that name an epoch in :func:`probability_counts`' stored rows."""


def probability_counts(dot: pd.DataFrame, in_visit: NDArray[np.bool_]) -> pd.DataFrame:
    """One epoch's immobile, analyzed time bins inside a well visit, counted by
    non-local probability in the bins of :data:`PROBABILITY_BIN_EDGES`.

    Parameters
    ----------
    dot : pd.DataFrame
        One epoch's decode frame read through the SWR source
        (``analyzed_time.with_swr_source``), with ``head_speed``,
        ``in_excluded_time`` and the two non-local state probabilities.
    in_visit : NDArray of bool, shape (n_time,)
        Whether each bin is inside a well visit
        (``event_definition.well_visits.in_well_visit`` on the frame's times).

    Returns
    -------
    counts : pd.DataFrame, shape (50, 3)
        ``bin_low``, ``bin_high`` and ``n_bins``, every bin (a bin with no time
        has 0). A time bin without a finite probability is in none.

    Raises
    ------
    ValueError
        If ``in_visit`` is not one bool per bin.
    """
    probability = nonlocal_probability(dot).to_numpy()[rest_bins(dot, in_visit)]
    probability = probability[np.isfinite(probability)]
    # right-closed bins, the first closed on both sides
    index = np.clip(
        np.searchsorted(PROBABILITY_BIN_EDGES, probability, side="left") - 1,
        0,
        len(PROBABILITY_BIN_EDGES) - 2,
    )
    n = np.bincount(index, minlength=len(PROBABILITY_BIN_EDGES) - 1)
    return pd.DataFrame(
        {
            "bin_low": PROBABILITY_BIN_EDGES[:-1],
            "bin_high": PROBABILITY_BIN_EDGES[1:],
            "n_bins": n.astype(int),
        }
    )


def _epoch_shares(counts: pd.DataFrame) -> pd.DataFrame:
    """Each epoch's share of its time in each probability bin (an epoch with
    no time is left out)."""
    keys = list(EPOCH_COLUMNS)
    total = counts.groupby(keys)["n_bins"].transform("sum")
    return counts.loc[total > 0].assign(share=counts["n_bins"] / total)


def animal_probability_distribution(counts: pd.DataFrame) -> pd.DataFrame:
    """Each animal's distribution of its time over non-local probability.

    Parameters
    ----------
    counts : pd.DataFrame
        The stored rows of every epoch: :data:`EPOCH_COLUMNS`, ``bin_low``,
        ``bin_high``, ``n_bins``.

    Returns
    -------
    pd.DataFrame
        One row per animal and bin: ``animal``, ``bin_low``, ``bin_high`` and
        ``percent``, the mean over the animal's epochs of each epoch's percent
        of its time in the bin.
    """
    shares = _epoch_shares(counts)
    return (
        shares.groupby(["animal", "bin_low", "bin_high"], as_index=False)["share"]
        .mean()
        .assign(percent=lambda table: 100 * table["share"])
        .drop(columns="share")
    )


def probability_share_statistics(counts: pd.DataFrame) -> pd.DataFrame:
    """The share of the time at most 0.2, above 0.2 and at most 0.8, and above
    0.8 (:data:`SHARES`).

    Each epoch's share of its time, each animal's mean over its epochs, then
    the mean over animals with the smallest and largest animal.

    Parameters
    ----------
    counts : pd.DataFrame
        As :func:`animal_probability_distribution` takes them.

    Returns
    -------
    pd.DataFrame
        Indexed by the words of :data:`SHARES`: ``animal_mean``,
        ``animal_min``, ``animal_max``, ``n_animals`` and ``n_epochs``.
    """
    shares = _epoch_shares(counts)
    part = np.where(
        shares["bin_high"] <= LOW_CUT + 1e-12,
        "low",
        np.where(shares["bin_low"] >= HIGH_CUT - 1e-12, "high", "middle"),
    )
    per_epoch = (
        shares.assign(part=part)
        .groupby([*EPOCH_COLUMNS, "part"])["share"]
        .sum()
        .unstack("part", fill_value=0.0)
        .reindex(columns=list(SHARES), fill_value=0.0)
    )
    return animal_range(per_epoch).rename_axis("part")
