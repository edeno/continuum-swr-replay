"""Small tables and helpers for the non-local restriction tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from continuum_swr_replay.analysis.event_evidence.nonlocal_restriction import (
    CONTRASTS,
    FAMILY_SUFFIX,
    SCORE_COLUMNS,
    ratio_or_nan,
)


def restriction_contrasts(
    sums: pd.DataFrame, categories: Sequence[str], bin_s: float
) -> dict[str, float]:
    """The model contrasts over the bins of ``categories``, both score families.

    Parameters
    ----------
    sums : pd.DataFrame
        ``nonlocal_restriction.restriction_sums``' output.
    categories : sequence of str
        The categories to combine (a category with no bins adds nothing).
    bin_s : float
        The bin width, s.

    Returns
    -------
    dict
        ``n_bins``, ``seconds``; for each family (no suffix for the filter score,
        ``_two_sided`` for the two-sided one) and contrast of ``CONTRASTS``:
        the contrast (nats) and the contrast per second (``..._per_s``, NaN
        without bins).
    """
    combined = sums.reindex(list(categories)).fillna(0.0).sum()
    seconds = float(combined["n_bins"]) * bin_s
    out: dict[str, float] = {"n_bins": int(combined["n_bins"]), "seconds": seconds}
    for suffix in FAMILY_SUFFIX.values():
        for name, (a, b) in CONTRASTS.items():
            difference = float(combined[f"{a}{suffix}"] - combined[f"{b}{suffix}"])
            out[f"{name}{suffix}"] = difference
            out[f"{name}{suffix}_per_s"] = ratio_or_nan(difference, seconds)
    return out


def restriction_bins(
    time: NDArray,
    scores: Mapping[str, NDArray],
    in_swr: NDArray | None,
    category: Sequence[str],
    **extra: NDArray,
) -> pd.DataFrame:
    """A restriction run's per-bin table, one row per bin of ``time``.

    Parameters
    ----------
    time : ndarray, shape (n_time,)
        s.
    scores : mapping
        Each of ``SCORE_COLUMNS`` and its per-bin score; stored as float32, as
        the run stores them.
    in_swr : ndarray of bool, shape (n_time,), or None
        None leaves the column out.
    category : sequence of str, length n_time
        Each bin's category; stored as a ``pd.Categorical``.
    **extra : ndarray
        More columns, after ``category``.

    Returns
    -------
    pd.DataFrame
        ``time``, the six score columns in ``SCORE_COLUMNS``' order, ``in_swr``,
        ``category`` and ``extra``.
    """
    table = pd.DataFrame(
        {
            "time": time,
            **{
                column: np.asarray(scores[column], dtype=np.float32)
                for column in SCORE_COLUMNS
            },
        }
    )
    if in_swr is not None:
        table["in_swr"] = np.asarray(in_swr, dtype=bool)
    table["category"] = pd.Categorical(category)
    for column, values in extra.items():
        table[column] = values
    return table


def bin_facts_table(
    time: NDArray, n_spikes: NDArray, in_well_visit: NDArray
) -> pd.DataFrame:
    """A per-bin facts table (``bin_facts.bin_facts``): ``time``, ``n_spikes``
    (int32) and ``in_well_visit`` (bool)."""
    return pd.DataFrame(
        {
            "time": time,
            "n_spikes": np.asarray(n_spikes, dtype=np.int32),
            "in_well_visit": np.asarray(in_well_visit, dtype=bool),
        }
    )


def time_sums(rows: Sequence[tuple]) -> pd.DataFrame:
    """Time sums (``restriction_totals.restriction_time_sums``' rows of a
    cohort) from ``(epoch_id, category, in_well_visit, in_event_window,
    n_bins, n_spikes, swr, full)`` rows: ``animal`` is the epoch id's first
    letter, ``none`` 0, and the two-sided scores twice the filter scores."""
    table = pd.DataFrame(
        rows,
        columns=[
            "epoch_id", "category", "in_well_visit", "in_event_window", "n_bins",
            "n_spikes", "swr", "full",
        ],
    )  # fmt: skip
    return table.assign(
        animal=table["epoch_id"].str[0],
        none=0.0,
        none_two_sided=0.0,
        swr_two_sided=2 * table["swr"],
        full_two_sided=2 * table["full"],
    )
