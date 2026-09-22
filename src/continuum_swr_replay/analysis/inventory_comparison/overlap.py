"""Every overlap between inventories, and the counts built on it.

Spec 03, "Matching and counting". Two events overlap when they share time of
positive duration within one epoch; touching at an instant is not agreement,
and nothing is matched across epochs. The pair table keeps every such pair,
so one event can appear in many; the counts built on it take each event once.

Membership needs coverage and a match does not. An event is ``matched`` to
another detector when a valid event of that detector overlaps it, wherever
that counterpart lies; ``unmatched`` only when it lies wholly inside the time
both detectors were exposed and nothing overlaps it; ``unknown`` otherwise.
Each ordered pair of detectors has its own denominator: the source detector's
events lying wholly inside the pair's shared exposure. Where the two share no
exposure the summary is unavailable, never zero.

Run :func:`remove_bridging_events` first: a cached event that spans its own
detector's unavailable data cannot be trusted as a match, and removing it
must not read as the other detector missing something.
"""

from collections.abc import Mapping
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
    "MATCH_DTYPES",
    "PAIR_DTYPES",
    "TIME_COVERAGE_DTYPES",
    "Coverage",
    "directional_agreement",
    "event_matches",
    "overlap_pairs",
    "remove_bridging_events",
    "time_coverage",
]

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


@dataclass(frozen=True)
class Coverage:
    """What one detector observed in one epoch, and the still part of it.

    Attributes
    ----------
    observed : ndarray, shape (n, 2)
    exposure : ndarray, shape (m, 2)
        Observed time with finite speed below threshold.

    """

    observed: NDArray
    exposure: NDArray

    def __post_init__(self) -> None:
        object.__setattr__(self, "observed", normalize(self.observed))
        object.__setattr__(self, "exposure", normalize(self.exposure))
        if total_duration(subtract(self.exposure, self.observed)) > 0:
            raise ValueError("Exposure must lie within observed time.")


CoverageMap = Mapping[tuple, Coverage]
"""Keyed ``(animal, day, epoch, detector)``."""


def _frame(rows: list[dict], dtypes: Mapping[str, object]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame({k: pd.Series(dtype=v) for k, v in dtypes.items()})
    return pd.DataFrame(rows)[list(dtypes)].astype(dict(dtypes))


def _sessions(inventory: pd.DataFrame, coverage: CoverageMap) -> dict[tuple, list[str]]:
    """Every session, each with every detector of the comparison.

    A detector with no coverage entry for a session observed none of it, so
    its summaries there are unavailable rather than missing. A detector with
    events but no coverage is refused: whether it looked is exactly what the
    counts need to know.
    """
    sessions = {tuple(key[:3]) for key in coverage}
    detectors = sorted({key[3] for key in coverage})
    for key, group in inventory.groupby(list(SESSION_COLUMNS_DAY), sort=False):
        for detector in group.detector.unique():
            if (*key, detector) not in coverage:
                raise ValueError(
                    f"{detector} has events in {key} but no coverage there."
                )
    return dict.fromkeys(sorted(sessions), detectors)


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
) -> tuple[pd.DataFrame, dict[tuple, Coverage]]:
    """Set aside events that span their own detector's unavailable data.

    Returns
    -------
    kept : DataFrame
        The inventory without them, in its original order.
    coverage : dict
        Each detector's coverage with those events' spans removed, so the
        time they occupied reads as unobserved rather than as a non-detection.

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
    return inventory[~bridging].reset_index(drop=True), adjusted


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
    inventory: pd.DataFrame, pairs: pd.DataFrame, coverage: CoverageMap
) -> pd.DataFrame:
    """Each event against each other detector that looked at its epoch.

    Returns columns and dtypes as ``MATCH_DTYPES``; membership is
    ``matched``, ``unmatched`` or ``unknown`` as the module docstring defines.
    """
    counterparts = _counterparts(pairs)
    rows = []
    for session, detectors in _sessions(inventory, coverage).items():
        for detector, other in permutations(detectors, 2):
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
                        "membership": "matched"
                        if n > 0
                        else ("unmatched" if inside else "unknown"),
                    }
                )
    return _frame(rows, MATCH_DTYPES)


def directional_agreement(
    inventory: pd.DataFrame, pairs: pd.DataFrame, coverage: CoverageMap
) -> pd.DataFrame:
    """For each ordered pair of detectors, the share of the source's events matched.

    The denominator is the source's events lying wholly inside the pair's
    shared exposure, each counted once however many events it overlaps; the
    rest are counted as excluded. Returns columns and dtypes as
    ``DIRECTIONAL_DTYPES``.
    """
    matches = event_matches(inventory, pairs, coverage)
    rows = []
    for session, detectors in _sessions(inventory, coverage).items():
        for source, target in permutations(detectors, 2):
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


def time_coverage(inventory: pd.DataFrame, coverage: CoverageMap) -> pd.DataFrame:
    """For each pair of detectors, the time their events cover, within shared exposure.

    Less sensitive to segmentation than the counts: one detector splitting an
    event in three changes its count, not the time it covers. Event time
    outside the shared exposure is clipped here, and only here. Returns
    columns and dtypes as ``TIME_COVERAGE_DTYPES``.
    """
    rows = []
    for session, detectors in _sessions(inventory, coverage).items():
        for detector_a, detector_b in combinations(detectors, 2):
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
