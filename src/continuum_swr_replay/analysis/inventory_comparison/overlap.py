"""Every overlap between inventories, and the counts built on it.

Two events overlap when they share time of positive duration within one
epoch; touching at an instant is not agreement, and nothing is matched across
epochs. The pair table keeps every such pair, so one event can appear in many;
the counts built on it take each event once.

Membership needs coverage and a match does not. An event is ``matched`` to
another detector when a valid event of that detector overlaps it, wherever
that counterpart lies; ``unmatched`` only when it lies wholly inside the time
both detectors were exposed and nothing overlaps it; ``unknown`` otherwise.
Each ordered pair of detectors has its own denominator: the source detector's
events lying wholly inside the pair's shared exposure. Where the two share no
exposure the summary is unavailable, never zero.

Run :func:`remove_bridging_events` first: a cached event that spans its own
detector's unavailable data cannot be trusted as a match, and removing it
must not read as the other detector missing something. The counting
functions refuse an inventory in which that has not been done, and a pair
table that was not built from the inventory they are given. An event the
analysed-time rule drops leaves its own detector's coverage the same way,
through :func:`set_aside_events`.

The detectors of a comparison are the roster: by default every detector with
a coverage entry anywhere, or the ``detectors`` a caller names. A detector on
the roster with no coverage in an epoch is unavailable there; a name outside
the roster is refused, so a misspelling cannot read as "unknown" everywhere.
"""

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from itertools import combinations, permutations
from types import MappingProxyType

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from continuum_swr_replay.analysis.inventory_comparison.intervals import (
    contains,
    intersect,
    normalize,
    subtract,
    total_duration,
    union,
)
from continuum_swr_replay.session_keys import SESSION_COLUMNS_DAY

__all__ = [
    "DIRECTIONAL_DTYPES",
    "MATCHED",
    "MATCH_DTYPES",
    "PAIR_DTYPES",
    "SET_ASIDE_DTYPES",
    "TIME_COVERAGE_DTYPES",
    "UNKNOWN",
    "UNMATCHED",
    "Coverage",
    "directional_agreement",
    "event_matches",
    "flag_pairs",
    "overlap_pairs",
    "remove_bridging_events",
    "roster",
    "set_aside_events",
    "time_coverage",
    "validate_detectors",
    "validate_pairs",
]

MATCHED, UNMATCHED, UNKNOWN = "matched", "unmatched", "unknown"
"""The three memberships. ``unknown`` is reported apart and never counted."""

PAIR_DTYPES: Mapping[str, object] = MappingProxyType(
    {
        "animal": object,
        "day": np.int64,
        "epoch": object,
        "detector_a": object,
        "event_key_a": object,
        "detector_b": object,
        "event_key_b": object,
        "intersection_s": np.float64,
        "iou": np.float64,
        "fraction_a_covered": np.float64,
        "fraction_b_covered": np.float64,
        "onset_difference_s": np.float64,
        "offset_difference_s": np.float64,
    }
)
"""One row per overlapping pair, ``detector_a`` before ``detector_b`` in name
order. Differences are the second event's time minus the first's."""

MATCH_DTYPES: Mapping[str, object] = MappingProxyType(
    {
        "animal": object,
        "day": np.int64,
        "epoch": object,
        "detector": object,
        "event_key": object,
        "other_detector": object,
        "within_shared_exposure": np.bool_,
        "matched": np.bool_,
        "n_counterparts": np.int64,
        "membership": object,
    }
)

DIRECTIONAL_DTYPES: Mapping[str, object] = MappingProxyType(
    {
        "animal": object,
        "day": np.int64,
        "epoch": object,
        "source_detector": object,
        "target_detector": object,
        "shared_exposure_s": np.float64,
        "available": np.bool_,
        "n_source_events": np.int64,
        "n_eligible": np.int64,
        "n_excluded": np.int64,
        "n_matched": np.int64,
        "proportion_matched": np.float64,
    }
)

SET_ASIDE_DTYPES: Mapping[str, object] = MappingProxyType(
    {
        "animal": object,
        "day": np.int64,
        "epoch": object,
        "detector": object,
        "event_key": object,
        "interval_start_s": np.float64,
        "interval_end_s": np.float64,
        "reason": object,
    }
)
"""One row per event left out of its cohort: the inventory key, the event's
span and why it was left out (a drop reason, or that it bridges its own
detector's unavailable data)."""

TIME_COVERAGE_DTYPES: Mapping[str, object] = MappingProxyType(
    {
        "animal": object,
        "day": np.int64,
        "epoch": object,
        "detector_a": object,
        "detector_b": object,
        "shared_exposure_s": np.float64,
        "available": np.bool_,
        "union_s": np.float64,
        "intersection_s": np.float64,
        "union_fraction": np.float64,
        "intersection_fraction": np.float64,
    }
)


@dataclass(frozen=True, eq=False)
class Coverage:
    """What one detector observed in one epoch, and the still part of it.

    Attributes
    ----------
    observed : ndarray, shape (n, 2)
        Normalized and read-only once built.
    exposure : ndarray, shape (m, 2)
        Observed time with finite speed below threshold; lies within
        ``observed``.

    """

    observed: NDArray
    exposure: NDArray

    def __post_init__(self) -> None:
        observed = normalize(self.observed, strict=True)
        exposure = normalize(self.exposure, strict=True)
        if total_duration(subtract(exposure, observed)) > 0:
            raise ValueError("Exposure must lie within observed time.")
        observed.flags.writeable = False
        exposure.flags.writeable = False
        object.__setattr__(self, "observed", observed)
        object.__setattr__(self, "exposure", exposure)


CoverageMap = Mapping[tuple, Coverage]
"""Keyed ``(animal, day, epoch, detector)``. A detector that ran and observed
nothing, or did not run, is entered with empty coverage rather than left out,
so it stays on the roster and reads as unavailable."""


def roster(coverage: CoverageMap) -> list[str]:
    """Every detector of the comparison, in name order."""
    return sorted({str(key[3]) for key in coverage})


def validate_detectors(
    names: Collection[str], coverage: CoverageMap, role: str
) -> list[str]:
    """Refuse a detector name the comparison does not know.

    A misspelt name would otherwise read as a detector that observed nothing
    anywhere, and every count against it as ``unknown``.
    """
    names = list(names)
    if not names:
        raise ValueError(f"{role} must name at least one detector.")
    unknown = sorted(set(names) - set(roster(coverage)))
    if unknown:
        raise ValueError(
            f"{role} names {unknown}, which no coverage entry knows; the roster is "
            f"{roster(coverage)}."
        )
    return names


def _frame(rows: list[dict], dtypes: Mapping[str, object]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame({k: pd.Series(dtype=v) for k, v in dtypes.items()})
    return pd.DataFrame(rows)[list(dtypes)].astype(dict(dtypes))


def _sessions(
    inventory: pd.DataFrame,
    coverage: CoverageMap,
    detectors: Collection[str] | None = None,
) -> dict[tuple, list[str]]:
    """Every session, each with every detector of the comparison.

    A detector with no coverage entry for a session observed none of it, so
    its summaries there are unavailable rather than missing. A detector with
    events but no coverage is refused: whether it looked is exactly what the
    counts need to know. An event outside its own detector's observed time is
    refused too: :func:`remove_bridging_events` has not been run.
    """
    sessions = {tuple(key[:3]) for key in coverage}
    names = (
        roster(coverage)
        if detectors is None
        else validate_detectors(detectors, coverage, "detectors")
    )
    for key, group in inventory.groupby(list(SESSION_COLUMNS_DAY), sort=False):
        for detector, events in group.groupby("detector", sort=False):
            own = coverage.get((*key, detector))
            if own is None:
                raise ValueError(
                    f"{detector} has events in {key} but no coverage there."
                )
            if detector not in names:
                raise ValueError(
                    f"{detector} has events but is not among the detectors compared."
                )
            if not contains(
                own.observed,
                events.event_start_s.to_numpy(),
                events.event_end_s.to_numpy(),
            ).all():
                raise ValueError(
                    f"An event of {detector} in {key} lies outside its own observed "
                    "time; run remove_bridging_events first."
                )
    return dict.fromkeys(sorted(sessions), names)


def validate_pairs(pairs: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    """Refuse a pair table that was not built from this inventory.

    Every key must be an event of the inventory, each pair's detector columns
    must be the keys' own detectors in name order, both events must share the
    pair's epoch, and the overlap must be positive. A stale pair table, built
    before bridging events were removed, would otherwise mark their neighbours
    matched.
    """
    if pairs.empty:
        return pairs
    by_key = inventory.set_index("event_key")
    missing = sorted(
        (set(pairs.event_key_a) | set(pairs.event_key_b)) - set(by_key.index)
    )
    if missing:
        raise ValueError(
            f"The pair table names events not in the inventory, {missing[:3]}; it "
            "was built from another inventory, or before bridging events were "
            "removed."
        )
    a, b = by_key.loc[pairs.event_key_a], by_key.loc[pairs.event_key_b]
    if not (
        (a.detector.to_numpy() == pairs.detector_a.to_numpy()).all()
        and (b.detector.to_numpy() == pairs.detector_b.to_numpy()).all()
    ):
        raise ValueError("A pair's detector columns are not its events' detectors.")
    if (pairs.detector_a.to_numpy() >= pairs.detector_b.to_numpy()).any():
        raise ValueError("Pairs must name two detectors, detector_a first by name.")
    for column in SESSION_COLUMNS_DAY:
        if not (
            (a[column].to_numpy() == pairs[column].to_numpy()).all()
            and (b[column].to_numpy() == pairs[column].to_numpy()).all()
        ):
            raise ValueError("A pair's two events do not share its epoch.")
    if (pairs.intersection_s.to_numpy() <= 0).any():
        raise ValueError("A pair has no positive overlap.")
    return pairs


def _exposure(coverage: CoverageMap, session: tuple, detector: str) -> NDArray:
    """A detector's exposure in a session; none where it has no entry."""
    entry = coverage.get((*session, detector))
    return np.empty((0, 2)) if entry is None else entry.exposure


def _events(inventory: pd.DataFrame, session: tuple, detector: str) -> pd.DataFrame:
    mask = (inventory.detector == detector) & np.logical_and.reduce(
        [
            inventory[column] == value
            for column, value in zip(SESSION_COLUMNS_DAY, session, strict=True)
        ]
    )
    return inventory[mask]


def _event_time(inventory: pd.DataFrame, session: tuple, detector: str) -> NDArray:
    """The time one detector's events cover in one epoch, overlaps joined."""
    events = _events(inventory, session, detector)
    return union(events[["event_start_s", "event_end_s"]].to_numpy())


def remove_bridging_events(
    inventory: pd.DataFrame, coverage: CoverageMap
) -> tuple[pd.DataFrame, dict[tuple, Coverage], pd.DataFrame]:
    """Set aside events that span their own detector's unavailable data.

    Returns
    -------
    kept : DataFrame
        The inventory without them, in its original order.
    coverage : dict
        Each detector's coverage with those events' spans removed, so the
        time they occupied reads as unobserved rather than as a non-detection.
    bridging : DataFrame
        The events set aside, to be reported with the counts.

    """
    adjusted = dict(coverage)
    bridging = np.zeros(len(inventory), dtype=bool)
    for (session, detector), group in inventory.groupby(
        [inventory[list(SESSION_COLUMNS_DAY)].apply(tuple, axis=1), "detector"],
        sort=False,
    ):
        own = coverage.get((*session, detector))
        if own is None:
            raise ValueError(
                f"{detector} has events in {session} but no coverage there."
            )
        crosses = ~contains(
            own.observed, group.event_start_s.to_numpy(), group.event_end_s.to_numpy()
        )
        if crosses.any():
            spans = group[["event_start_s", "event_end_s"]].to_numpy()[crosses]
            adjusted[(*session, detector)] = Coverage(
                observed=subtract(own.observed, spans),
                exposure=subtract(own.exposure, spans),
            )
            bridging[inventory.index.get_indexer(group.index[crosses])] = True
    return (
        inventory[~bridging].reset_index(drop=True),
        adjusted,
        inventory[bridging].reset_index(drop=True),
    )


def set_aside_events(
    coverage: CoverageMap, events: pd.DataFrame
) -> dict[tuple, Coverage]:
    """Remove the spans of events left out of their cohorts from their own coverage.

    An event the analysed-time rule drops is not compared, so, as for a
    bridging event, the time it occupied reads as time its own detector did
    not observe: another detector's event there is ``unknown`` against it,
    never ``unmatched``. Every other detector's coverage is unchanged.

    Parameters
    ----------
    coverage : mapping
        ``(animal, day, epoch, detector) -> Coverage``.
    events : DataFrame
        Columns as ``SET_ASIDE_DTYPES``.

    Returns
    -------
    coverage : dict
        ``coverage`` with each event's span removed from its own detector's
        ``observed`` and ``exposure``.

    """
    adjusted = dict(coverage)
    for key, group in events.groupby([*SESSION_COLUMNS_DAY, "detector"], sort=False):
        own = coverage.get(key)
        if own is None:
            raise ValueError(
                f"{key[3]} has events set aside in {key[:3]} but no coverage there."
            )
        spans = group[["interval_start_s", "interval_end_s"]].to_numpy(dtype=np.float64)
        adjusted[key] = Coverage(
            observed=subtract(own.observed, spans),
            exposure=subtract(own.exposure, spans),
        )
    return adjusted


def overlap_pairs(inventory: pd.DataFrame) -> pd.DataFrame:
    """Every strict positive-duration overlap between two detectors' events.

    Within each epoch only. Returns columns and dtypes as ``PAIR_DTYPES``.
    """
    rows = []
    for session, group in inventory.groupby(list(SESSION_COLUMNS_DAY), sort=False):
        for detector_a, detector_b in combinations(sorted(group.detector.unique()), 2):
            a = group[group.detector == detector_a]
            b = group[group.detector == detector_b]
            a_start, a_end = a.event_start_s.to_numpy(), a.event_end_s.to_numpy()
            b_start, b_end = b.event_start_s.to_numpy(), b.event_end_s.to_numpy()
            inter = np.minimum(a_end[:, None], b_end[None, :]) - np.maximum(
                a_start[:, None], b_start[None, :]
            )
            for i, j in zip(*np.nonzero(inter > 0), strict=True):
                shared = inter[i, j]
                span_a, span_b = a_end[i] - a_start[i], b_end[j] - b_start[j]
                rows.append(
                    dict(zip(SESSION_COLUMNS_DAY, session, strict=True))
                    | {
                        "detector_a": detector_a,
                        "event_key_a": a.event_key.iloc[i],
                        "detector_b": detector_b,
                        "event_key_b": b.event_key.iloc[j],
                        "intersection_s": shared,
                        "iou": shared / (span_a + span_b - shared),
                        "fraction_a_covered": shared / span_a,
                        "fraction_b_covered": shared / span_b,
                        "onset_difference_s": b_start[j] - a_start[i],
                        "offset_difference_s": b_end[j] - a_end[i],
                    }
                )
    return _frame(rows, PAIR_DTYPES)


def _counterparts(pairs: pd.DataFrame) -> dict[tuple, int]:
    """How many events of each other detector overlap each event."""
    counts: dict[tuple, int] = {}
    for key_a, detector_b, key_b, detector_a in zip(
        pairs.event_key_a,
        pairs.detector_b,
        pairs.event_key_b,
        pairs.detector_a,
        strict=True,
    ):
        counts[(key_a, detector_b)] = counts.get((key_a, detector_b), 0) + 1
        counts[(key_b, detector_a)] = counts.get((key_b, detector_a), 0) + 1
    return counts


def event_matches(
    inventory: pd.DataFrame,
    pairs: pd.DataFrame,
    coverage: CoverageMap,
    *,
    detectors: Collection[str] | None = None,
) -> pd.DataFrame:
    """Each event against each other detector of the comparison.

    A detector that did not look at the epoch leaves every event ``unknown``
    against it. Returns columns and dtypes as ``MATCH_DTYPES``; membership is
    ``matched``, ``unmatched`` or ``unknown`` as the module docstring defines.

    Parameters
    ----------
    inventory, pairs, coverage
        The inventory after :func:`remove_bridging_events`, the pairs built
        from that same inventory, and the coverage it returned.
    detectors : collection of str, optional
        The roster; every detector with a coverage entry when omitted.

    """
    validate_pairs(pairs, inventory)
    counterparts = _counterparts(pairs)
    rows = []
    for session, names in _sessions(inventory, coverage, detectors).items():
        for detector, other in permutations(names, 2):
            source = _events(inventory, session, detector)
            shared = intersect(
                _exposure(coverage, session, detector),
                _exposure(coverage, session, other),
            )
            within = contains(
                shared, source.event_start_s.to_numpy(), source.event_end_s.to_numpy()
            )
            for key, inside in zip(source.event_key, within, strict=True):
                n = counterparts.get((key, other), 0)
                rows.append(
                    dict(zip(SESSION_COLUMNS_DAY, session, strict=True))
                    | {
                        "detector": detector,
                        "event_key": key,
                        "other_detector": other,
                        "within_shared_exposure": bool(inside),
                        "matched": n > 0,
                        "n_counterparts": n,
                        "membership": MATCHED
                        if n > 0
                        else (UNMATCHED if inside else UNKNOWN),
                    }
                )
    return _frame(rows, MATCH_DTYPES)


def flag_pairs(pairs: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """The pair table, with whether each event lies within the shared exposure.

    Every overlap stays in the pair table however its events sit; the flags
    say which of the two could enter its direction's denominator.

    Parameters
    ----------
    pairs : DataFrame
        From ``overlap_pairs``.
    matches : DataFrame
        ``event_matches`` over the same inventory.

    Returns
    -------
    DataFrame
        ``pairs`` with ``a_within_shared_exposure`` and
        ``b_within_shared_exposure`` appended.

    """
    within = matches.set_index(["event_key", "other_detector"]).within_shared_exposure
    return pairs.assign(
        a_within_shared_exposure=within.loc[
            list(zip(pairs.event_key_a, pairs.detector_b, strict=True))
        ].to_numpy(dtype=bool),
        b_within_shared_exposure=within.loc[
            list(zip(pairs.event_key_b, pairs.detector_a, strict=True))
        ].to_numpy(dtype=bool),
    )


def directional_agreement(
    inventory: pd.DataFrame,
    pairs: pd.DataFrame,
    coverage: CoverageMap,
    *,
    detectors: Collection[str] | None = None,
) -> pd.DataFrame:
    """For each ordered pair of detectors, the share of the source's events matched.

    The denominator is the source's events lying wholly inside the pair's
    shared exposure, each counted once however many events it overlaps; the
    rest are counted as excluded. Returns columns and dtypes as
    ``DIRECTIONAL_DTYPES``. ``proportion_matched`` is NaN where nothing was
    eligible; pool epochs by summing ``n_matched`` over ``n_eligible``, never
    by averaging the proportions, which would skip those rows.
    """
    matches = event_matches(inventory, pairs, coverage, detectors=detectors)
    rows = []
    for session, names in _sessions(inventory, coverage, detectors).items():
        for source, target in permutations(names, 2):
            shared = total_duration(
                intersect(
                    _exposure(coverage, session, source),
                    _exposure(coverage, session, target),
                )
            )
            these = matches[
                (matches.detector == source)
                & (matches.other_detector == target)
                & np.logical_and.reduce(
                    [
                        matches[c] == v
                        for c, v in zip(SESSION_COLUMNS_DAY, session, strict=True)
                    ]
                )
            ]
            eligible = these[these.within_shared_exposure]
            n_eligible = len(eligible)
            n_matched = int(eligible.matched.sum())
            available = shared > 0
            rows.append(
                dict(zip(SESSION_COLUMNS_DAY, session, strict=True))
                | {
                    "source_detector": source,
                    "target_detector": target,
                    "shared_exposure_s": shared,
                    "available": available,
                    "n_source_events": len(these),
                    "n_eligible": n_eligible,
                    "n_excluded": len(these) - n_eligible,
                    "n_matched": n_matched,
                    "proportion_matched": n_matched / n_eligible
                    if n_eligible
                    else np.nan,
                }
            )
    return _frame(rows, DIRECTIONAL_DTYPES)


def time_coverage(
    inventory: pd.DataFrame,
    coverage: CoverageMap,
    *,
    detectors: Collection[str] | None = None,
) -> pd.DataFrame:
    """For each pair of detectors, the time their events cover, within shared exposure.

    Less sensitive to segmentation than the counts: one detector splitting an
    event in three changes its count, not the time it covers. Event time
    outside the shared exposure is clipped here and in ``no_detector_time``,
    the two time summaries, and never for a count. Returns columns and dtypes
    as ``TIME_COVERAGE_DTYPES``.
    """
    rows = []
    for session, names in _sessions(inventory, coverage, detectors).items():
        for detector_a, detector_b in combinations(names, 2):
            shared = intersect(
                _exposure(coverage, session, detector_a),
                _exposure(coverage, session, detector_b),
            )
            shared_s = total_duration(shared)
            a = _event_time(inventory, session, detector_a)
            b = _event_time(inventory, session, detector_b)
            union_s = total_duration(intersect(union(a, b), shared))
            inter_s = total_duration(intersect(intersect(a, b), shared))
            available = shared_s > 0
            rows.append(
                dict(zip(SESSION_COLUMNS_DAY, session, strict=True))
                | {
                    "detector_a": detector_a,
                    "detector_b": detector_b,
                    "shared_exposure_s": shared_s,
                    "available": available,
                    "union_s": union_s,
                    "intersection_s": inter_s,
                    "union_fraction": union_s / shared_s if available else np.nan,
                    "intersection_fraction": inter_s / shared_s
                    if available
                    else np.nan,
                }
            )
    return _frame(rows, TIME_COVERAGE_DTYPES)
