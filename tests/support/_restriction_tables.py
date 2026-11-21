"""Small tables and helpers for the non-local restriction tests."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from continuum_swr_replay.analysis.event_evidence.nonlocal_restriction import (
    CONTRASTS,
    FAMILY_SUFFIX,
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
