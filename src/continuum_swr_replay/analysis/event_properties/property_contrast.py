"""Uncertainty of the SWR against non-SWR property contrasts: a nested bootstrap.

Part 2 compares event properties between SWR-overlapping and non-SWR events by
the median of each class, event -> epoch -> animal: the mean over animals of
each animal's median over its epochs of the epoch's median
(``event_statistics.class_medians``). Events are nested in sessions (epochs)
and sessions in animals, so the uncertainty comes from a bootstrap that
resamples at each level in turn: the animals with replacement, then each drawn
animal's sessions with replacement, then each drawn session's events with
replacement. For the class medians and their difference every drawn copy is
kept apart (:func:`~continuum_swr_replay.analysis.shared_statistics.nested_bootstrap_copies`):
an animal drawn twice is two animals, each with its own sessions, and each
copy of a session takes its own median, so a replicate is the statistic the
data give; every property and both classes are read from the same resample.
The duration-matched difference is pooled by design (below), so its replicate
is a vector of counts, how many times each event was drawn
(:func:`~continuum_swr_replay.analysis.shared_statistics.nested_bootstrap_draw`),
from a second stream of the same seed. The 95 % interval
(:func:`bootstrap_intervals`) is the estimate plus or minus a small-sample t
correction over the animals times the replicates' standard deviation
(:func:`~continuum_swr_replay.analysis.shared_statistics.bootstrap_interval`).

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
``tests/analysis/event_properties/test_property_contrast.py``).
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from continuum_swr_replay.analysis.shared_statistics import (
    bootstrap_interval,
    nested_bootstrap_copies,
    nested_bootstrap_draw,
    nested_codes,
    session_copy_values,
    weighted_median,
    weighted_medians,
)
from continuum_swr_replay.parameters import DECODE_SAMPLE_S

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

    Pooled by design, unlike the project's event -> epoch -> animal rule: the
    strata are cut on the pooled durations of all events (:func:`duration_strata`),
    and the medians are over every event of a stratum, so it has no
    epoch-then-animal form.

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


def animal_first_medians(
    properties: Mapping[str, NDArray],
    is_swr: NDArray,
    animal_codes: NDArray,
    session_codes: NDArray,
    *,
    n_replicates: int,
    rng: np.random.Generator,
) -> tuple[NDArray, NDArray]:
    """Each property's class medians, event -> session -> animal, on the data
    and on nested bootstrap replicates.

    A class median is the mean over animals of each animal's median over its
    sessions of the session's median of its events of the class with a value;
    a session with no such event is not one of the animal's sessions for that
    class and property. A replicate draws the animals, each drawn animal's
    sessions and each drawn session's events, every copy on its own
    (``nested_bootstrap_copies``, ``session_copy_values``), and takes the same
    statistic; one resample of a session's events gives every property and
    both classes.

    Parameters
    ----------
    properties : Mapping[str, NDArray]
        Each property's values, shape (n_events,), NaN where undefined.
    is_swr : NDArray of bool, shape (n_events,)
    animal_codes, session_codes : NDArray of int, shape (n_events,)
        :func:`~continuum_swr_replay.analysis.shared_statistics.nested_codes`.
    n_replicates : int
    rng : numpy.random.Generator

    Returns
    -------
    estimate : NDArray, shape (n_properties, 2)
        Non-SWR, then SWR, in the order of ``properties``.
    replicates : NDArray, shape (n_replicates, n_properties, 2)
    """
    values = [np.asarray(v, dtype=np.float64) for v in properties.values()]
    n_sessions = int(session_codes.max()) + 1
    session_animal = np.zeros(n_sessions, dtype=np.int64)
    session_animal[session_codes] = animal_codes
    rows = [np.flatnonzero(session_codes == s) for s in range(n_sessions)]
    # per session, property and class: the session's events (as places in the
    # session) with a value, sorted by it, and their values
    cells = []
    for mine in rows:
        of_session = []
        for v in values:
            for label in (False, True):
                place = np.flatnonzero(np.isfinite(v[mine]) & (is_swr[mine] == label))
                place = place[np.argsort(v[mine][place], kind="stable")]
                of_session.append((place, v[mine][place]))
        cells.append(of_session)

    def animal_first(session_medians: NDArray, of_animal: NDArray) -> NDArray:
        """(..., n_members, n_cells) session medians -> (..., n_cells): the
        median over each animal's members, then the mean over animals."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # a cell without events
            per_animal = np.stack(
                [
                    np.nanmedian(session_medians[of_animal == a], axis=0)
                    for a in np.unique(of_animal)
                ]
            )
            return np.nanmean(per_animal, axis=0)

    data = np.array([[np.median(v) if v.size else np.nan for _, v in c] for c in cells])
    estimate = animal_first(data, session_animal)

    blocks: list[NDArray] = []
    n_made = 0

    def resample(session: int, n_copies: int) -> NDArray:
        nonlocal n_made
        n_events = rows[session].size
        drawn = rng.multinomial(
            n_events, np.full(n_events, 1.0 / n_events), size=n_copies
        )
        block = np.full((n_copies, len(cells[session])), np.nan)
        for k, (place, sorted_values) in enumerate(cells[session]):
            if place.size:
                block[:, k] = weighted_medians(
                    np.broadcast_to(sorted_values, (n_copies, place.size)),
                    drawn[:, place],
                )
        blocks.append(block)
        made = np.arange(n_made, n_made + n_copies, dtype=np.float64)
        n_made += n_copies
        return made

    copies = nested_bootstrap_copies(session_animal, rng, n_replicates)
    # each drawn copy's place in ``blocks``; NaN beyond a slot's copies
    which = session_copy_values(copies, resample)
    table = np.concatenate(blocks)
    medians = np.full((*which.shape, table.shape[1]), np.nan)
    filled = ~np.isnan(which)
    medians[filled] = table[which[filled].astype(np.int64)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        replicates = np.nanmean(np.nanmedian(medians, axis=2), axis=1)
    shape = (len(values), 2)
    return estimate.reshape(shape), replicates.reshape(n_replicates, *shape)


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
        ``median_non_swr``, ``median_swr`` (:func:`animal_first_medians`),
        ``difference`` (SWR minus non-SWR), ``matched_difference`` (pooled:
        :func:`matched_difference`) and ``matched_strata_skipped`` (eligible
        strata the replicate drew no event of a class in, left out of its
        average).
    """
    is_swr = np.asarray(is_swr, dtype=bool)
    animal_codes, session_codes = nested_codes(animal, session)
    strata = duration_strata(duration, n_strata)
    groups = {name: _SortedGroups(v, is_swr, strata) for name, v in properties.items()}
    eligible = {name: _eligible_strata(g, min_per_class) for name, g in groups.items()}
    rng = np.random.default_rng(seed)
    estimate, drawn = animal_first_medians(
        properties,
        is_swr,
        animal_codes,
        session_codes,
        n_replicates=n_replicates,
        rng=np.random.default_rng([seed, 1]),
    )
    medians = np.concatenate([drawn, estimate[None]])  # replicate -1 is last
    rows = []
    for replicate in range(-1, n_replicates):
        weights = (
            np.ones(is_swr.size, dtype=np.int64)
            if replicate < 0
            else nested_bootstrap_draw(animal_codes, session_codes, rng).rows
        )
        for p, (name, g) in enumerate(groups.items()):
            non, swr = medians[replicate, p]
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


def bootstrap_intervals(
    replicates: pd.DataFrame,
    n_animals: int,
    *,
    max_nan_share: float = MAX_NAN_REPLICATE_SHARE,
) -> pd.DataFrame:
    """Each property's estimate (replicate -1) and 95 % interval of each column
    of :func:`bootstrap_contrasts`, with the replicates behind it counted.

    The interval is
    :func:`~continuum_swr_replay.analysis.shared_statistics.bootstrap_interval`
    of the estimate and the replicates: symmetric about the estimate, a
    small-sample t correction over the ``n_animals`` animals the replicates
    resample. A replicate with no estimate (NaN) is left out of it, so the
    count of such replicates is kept, and more than ``max_nan_share`` of them
    raises.

    Parameters
    ----------
    replicates : pd.DataFrame
        :func:`bootstrap_contrasts`.
    n_animals : int
        The animals the replicates resample.
    max_nan_share : float, optional
        The largest share of replicates allowed to have no estimate.

    Returns
    -------
    pd.DataFrame
        Indexed by property: for each column ``c``, ``c``, ``c_low``,
        ``c_high`` and ``c_nan_replicates``; then ``n_replicates``,
        ``matched_replicates_skipping_strata`` (replicates whose matched
        difference left out an eligible stratum) and ``n_animals``.

    Raises
    ------
    ValueError
        If a property's share of NaN replicates in any column exceeds
        ``max_nan_share``.
    """
    columns = ["median_non_swr", "median_swr", "difference", "matched_difference"]
    point = replicates[replicates.replicate < 0].set_index("property")[columns]
    boot_rows = replicates[replicates.replicate >= 0]
    nan = boot_rows[columns].isna().groupby(boot_rows["property"]).sum()
    n_replicates = boot_rows.groupby("property").size()
    share = nan.div(n_replicates, axis=0)
    if (share > max_nan_share).any(axis=None):
        bad = share[(share > max_nan_share).any(axis=1)]
        raise ValueError(
            f"more than {max_nan_share:.0%} of the replicates have no estimate:\n"
            f"{bad.to_string()}"
        )
    out = point.copy()
    for name, draws in boot_rows.groupby("property"):
        for c in columns:
            interval = bootstrap_interval(point.loc[name, c], draws[c], n_animals)
            out.loc[name, f"{c}_low"] = interval.low
            out.loc[name, f"{c}_high"] = interval.high
    out = out.join(nan.add_suffix("_nan_replicates"))
    out = out[
        [f"{c}{s}" for c in columns for s in ("", "_low", "_high", "_nan_replicates")]
    ]
    out["n_replicates"] = n_replicates
    out["matched_replicates_skipping_strata"] = (
        boot_rows["matched_strata_skipped"].gt(0).groupby(boot_rows["property"]).sum()
    )
    out["n_animals"] = n_animals
    return out
