"""Uncertainty of the SWR against non-SWR property contrasts: a nested bootstrap.

Part 2 compares event properties between SWR-overlapping and non-SWR events by
the median of each class over every event. Events are nested in sessions
(epochs) and sessions in animals, so the uncertainty comes from a bootstrap
that resamples at each level in turn
(:func:`~continuum_swr_replay.analysis.shared_statistics.nested_bootstrap_draw`):
the animals with replacement, then each drawn animal's sessions with
replacement, then each drawn session's events with replacement. A replicate is
a vector of counts, how many times each event was drawn, so every property and
both classes are read from the same resample.

Non-SWR events are shorter than SWR events, so a raw contrast mixes any class
difference with a duration difference. :func:`matched_difference`
compares the classes within strata of event duration (quantiles of the pooled
durations, in whole decode samples) and averages the within-stratum
differences of medians, weighted by each stratum's events: the contrast the
classes would show if both had the pooled duration distribution. A bootstrap
replicate that draws no event of a class in a stratum leaves that stratum out,
so replicates can average over different strata; the count of such replicates,
and of replicates with no estimate (NaN), is kept beside each interval. Stratification is used rather than a regression
on log duration because it keeps the medians the property contrasts report and
makes no assumption about the shape of the duration effect; it removes any
duration effect to within one stratum's width (checked on simulated data in
``tests/analysis/hippocampus/test_property_contrast.py``).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from continuum_swr_replay.analysis.shared_statistics import (
    nested_bootstrap_draw,
    nested_codes,
    weighted_median,
)
from continuum_swr_replay.parameters import SAMPLING_FREQUENCY

#: An event's duration is a whole number of decode samples of this length (s).
DECODE_SAMPLE_S = 1.0 / SAMPLING_FREQUENCY
#: The largest share of bootstrap replicates allowed to have no estimate.
MAX_NAN_REPLICATE_SHARE = 0.01


def duration_strata(
    duration: NDArray, n_strata: int, *, sample_s: float = DECODE_SAMPLE_S
) -> NDArray[np.int64]:
    """Each event's stratum: quantile bins of the pooled durations, ``0..k-1``.

    Durations are first rounded to whole samples of ``sample_s`` seconds, so
    events with the same number of samples share a stratum even when their
    float durations differ in the last digits; tied durations make fewer than
    ``n_strata`` strata possible.
    """
    samples = np.round(np.asarray(duration, dtype=float) / sample_s)
    codes = pd.qcut(samples, n_strata, labels=False, duplicates="drop")
    return np.asarray(codes, dtype=np.int64)


class _SortedGroups:
    """Event indices of each (class, stratum) cell, sorted by one property."""

    def __init__(self, values: NDArray, is_swr: NDArray, strata: NDArray) -> None:
        values = np.asarray(values, dtype=np.float64)
        keep = np.isfinite(values)
        self.values = values
        self.cells: dict[tuple[bool, int], NDArray[np.int64]] = {}
        for label in (False, True):
            for stratum in np.unique(strata[keep]):
                idx = np.flatnonzero(keep & (is_swr == label) & (strata == stratum))
                self.cells[(label, int(stratum))] = idx[
                    np.argsort(values[idx], kind="stable")
                ]
        self.classes = {
            label: np.flatnonzero(keep & (is_swr == label))[
                np.argsort(values[keep & (is_swr == label)], kind="stable")
            ]
            for label in (False, True)
        }

    def median(self, label: bool, weights: NDArray) -> float:
        idx = self.classes[label]
        return weighted_median(self.values[idx], weights[idx])

    def matched(self, weights: NDArray, eligible: list[int]) -> tuple[float, int]:
        """The weighted matched difference, and how many eligible strata were
        left out because ``weights`` draw no event of a class in them."""
        diffs, sizes = [], []
        skipped = 0
        for stratum in eligible:
            swr = self.cells[(True, stratum)]
            non = self.cells[(False, stratum)]
            w_swr, w_non = weights[swr], weights[non]
            if w_swr.sum() == 0 or w_non.sum() == 0:
                skipped += 1
                continue
            diffs.append(
                weighted_median(self.values[swr], w_swr)
                - weighted_median(self.values[non], w_non)
            )
            sizes.append(w_swr.sum() + w_non.sum())
        if not sizes:
            return float("nan"), skipped
        return float(np.average(diffs, weights=sizes)), skipped


def matched_difference(
    values: NDArray,
    is_swr: NDArray,
    strata: NDArray,
    *,
    weights: NDArray | None = None,
    min_per_class: int = 20,
) -> float:
    """SWR minus non-SWR median of ``values`` within duration strata, averaged.

    Each stratum's difference of medians is weighted by its number of events
    (both classes; by ``weights`` when given). A stratum with fewer than
    ``min_per_class`` events of either class is left out.

    Parameters
    ----------
    values : NDArray, shape (n_events,)
        NaN events are left out.
    is_swr : NDArray of bool, shape (n_events,)
    strata : NDArray of int, shape (n_events,)
        :func:`duration_strata`.
    weights : NDArray of int, shape (n_events,), optional
        Bootstrap counts; all ones by default.
    """
    is_swr = np.asarray(is_swr, dtype=bool)
    strata = np.asarray(strata, dtype=np.int64)
    groups = _SortedGroups(values, is_swr, strata)
    eligible = _eligible_strata(groups, min_per_class)
    w = np.ones(is_swr.size, dtype=np.int64) if weights is None else np.asarray(weights)
    return groups.matched(w, eligible)[0]


def _eligible_strata(groups: _SortedGroups, min_per_class: int) -> list[int]:
    strata = sorted({s for _, s in groups.cells})
    return [
        s
        for s in strata
        if groups.cells.get((True, s), np.empty(0)).size >= min_per_class
        and groups.cells.get((False, s), np.empty(0)).size >= min_per_class
    ]


def bootstrap_contrasts(
    properties: Mapping[str, NDArray],
    is_swr: NDArray,
    animal: NDArray,
    session: NDArray,
    duration: NDArray,
    *,
    n_replicates: int,
    seed: int,
    n_strata: int = 10,
    min_per_class: int = 20,
) -> pd.DataFrame:
    """Class medians, their difference and the duration-matched difference of
    each property, on the data (replicate -1) and on ``n_replicates`` nested
    bootstrap replicates.

    Parameters
    ----------
    properties : Mapping[str, NDArray]
        Each property's values, shape (n_events,), NaN where undefined.
    is_swr : NDArray of bool, shape (n_events,)
    animal, session : NDArray, shape (n_events,)
        Labels; sessions nest in animals.
    duration : NDArray, shape (n_events,)
        Seconds; the strata are :func:`duration_strata` of it.
    n_replicates, seed : int
    n_strata, min_per_class : int
        As :func:`duration_strata` and :func:`matched_difference`; the strata
        and the strata kept are fixed on the data.

    Returns
    -------
    pd.DataFrame
        One row per replicate and property: ``replicate``, ``property``,
        ``median_non_swr``, ``median_swr``, ``difference`` (SWR minus non-SWR),
        ``matched_difference`` and ``matched_strata_skipped`` (eligible strata
        the replicate drew no event of a class in, left out of its average).
    """
    is_swr = np.asarray(is_swr, dtype=bool)
    animal_codes, session_codes = nested_codes(animal, session)
    strata = duration_strata(duration, n_strata)
    groups = {name: _SortedGroups(v, is_swr, strata) for name, v in properties.items()}
    eligible = {name: _eligible_strata(g, min_per_class) for name, g in groups.items()}
    rng = np.random.default_rng(seed)
    rows = []
    for replicate in range(-1, n_replicates):
        weights = (
            np.ones(is_swr.size, dtype=np.int64)
            if replicate < 0
            else nested_bootstrap_draw(animal_codes, session_codes, rng).rows
        )
        for name, g in groups.items():
            non, swr = g.median(False, weights), g.median(True, weights)
            matched, skipped = g.matched(weights, eligible[name])
            rows.append(
                {
                    "replicate": replicate,
                    "property": name,
                    "median_non_swr": non,
                    "median_swr": swr,
                    "difference": swr - non,
                    "matched_difference": matched,
                    "matched_strata_skipped": skipped,
                }
            )
    return pd.DataFrame(rows)


def percentile_intervals(
    replicates: pd.DataFrame,
    level: float = 0.95,
    *,
    max_nan_share: float = MAX_NAN_REPLICATE_SHARE,
) -> pd.DataFrame:
    """Each property's estimate (replicate -1) and percentile interval of each
    column of :func:`bootstrap_contrasts`, with the replicates behind it counted.

    A replicate with no estimate (NaN) is left out of the percentiles, so the
    count of such replicates is kept, and more than ``max_nan_share`` of them
    raises.

    Returns
    -------
    pd.DataFrame
        Indexed by property: for each column ``c``, ``c``, ``c_low``,
        ``c_high`` and ``c_nan_replicates``; then ``n_replicates`` and
        ``matched_replicates_skipping_strata`` (replicates whose matched
        difference left out an eligible stratum).

    Raises
    ------
    ValueError
        If a property's share of NaN replicates in any column exceeds
        ``max_nan_share``.
    """
    tail = (1.0 - level) / 2.0
    columns = ["median_non_swr", "median_swr", "difference", "matched_difference"]
    point = replicates[replicates.replicate < 0].set_index("property")[columns]
    boot_rows = replicates[replicates.replicate >= 0]
    boot = boot_rows.groupby("property")[columns]
    low = boot.quantile(tail).add_suffix("_low")
    high = boot.quantile(1.0 - tail).add_suffix("_high")
    nan = boot_rows[columns].isna().groupby(boot_rows["property"]).sum()
    n_replicates = boot.size()
    share = nan.div(n_replicates, axis=0)
    if (share > max_nan_share).any(axis=None):
        bad = share[(share > max_nan_share).any(axis=1)]
        raise ValueError(
            f"more than {max_nan_share:.0%} of the replicates have no estimate:\n"
            f"{bad.to_string()}"
        )
    out = point.join(low).join(high).join(nan.add_suffix("_nan_replicates"))
    out = out[
        [f"{c}{s}" for c in columns for s in ("", "_low", "_high", "_nan_replicates")]
    ]
    out["n_replicates"] = n_replicates
    out["matched_replicates_skipping_strata"] = (
        boot_rows["matched_strata_skipped"].gt(0).groupby(boot_rows["property"]).sum()
    )
    return out
