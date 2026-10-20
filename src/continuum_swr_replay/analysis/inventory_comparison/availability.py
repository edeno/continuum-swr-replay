"""What each detector observed, apart from what it detected.

A detector that looked and found nothing observed that time; a detector whose
inputs were missing did not, and an empty event table alone cannot tell the
two apart. Coverage is built from the samples a detector needed, split at
every non-finite sample and every gap in the timestamps, never interpolated
across either; a detector that reads a window of samples observes a moment
only when the whole window is there.

Exposure is observed time during which the animal was known to be still:
finite speed below the canonical threshold, each still run reaching the sample
on either side of it, since an event is speed-eligible when the samples inside
it are still. Comparisons between detectors use the intersection of their
exposures.

Excluded time (``swr_source.excluded_intervals``: the artifact mask widened by
0.3 s) is not analysed, so the reader of the observation tables
(:func:`coverage_from_observations`) removes it from every detector's observed
time, and so from its exposure. The tables themselves stay a record of what
each detector read, as the SWR runner's ``swr_observed.parquet`` is.

:func:`exclusion_by_reason` accounts for the rest of each epoch: for a
comparison, every second outside its shared exposure under the first reason
that explains it.
"""

from collections.abc import Collection, Mapping
from functools import reduce
from types import MappingProxyType

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from continuum_swr_replay.analysis.constants import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
)
from continuum_swr_replay.analysis.hippocampus import swr_source
from continuum_swr_replay.analysis.inventory_comparison.overlap import (
    Coverage,
    set_aside_events,
    validate_detectors,
)
from continuum_swr_replay.analysis.inventory_comparison.schema import SessionIdentity
from continuum_swr_replay.intervals import (
    intersect,
    normalize,
    subtract,
    total_duration,
)
from continuum_swr_replay.session_keys import SESSION_COLUMNS_DAY

__all__ = [
    "EXCLUSION_DTYPES",
    "OBSERVATION_DTYPES",
    "coverage_from_observations",
    "exclusion_by_reason",
    "exposure",
    "missing_run_table",
    "not_still_times",
    "observation_table",
    "sample_runs",
    "still_intervals",
    "stored_coverage",
    "window_supported",
]

OBSERVATION_DTYPES: Mapping[str, object] = MappingProxyType(
    {
        "animal": object,
        "day": np.int64,
        "epoch": object,
        "brain_area": object,
        "detector": object,
        "run_id": object,
        "start_s": np.float64,
        "end_s": np.float64,
        "available": np.bool_,
        "reason": object,
    }
)

EXCLUSION_DTYPES: Mapping[str, object] = MappingProxyType(
    {
        "animal": object,
        "day": np.int64,
        "epoch": object,
        "comparison": object,
        "reason": object,
        "seconds": np.float64,
    }
)

GAP_FACTOR = 1.5
"""A spacing more than this many median sample intervals is a gap."""


def _joined(time: NDArray, valid: NDArray, max_gap_s: float | None) -> NDArray:
    """Whether each sample continues the run of the one before it.

    Returns
    -------
    joined : ndarray of bool, shape (n_samples - 1,)
        False at an invalid sample, a non-finite timestamp, or a spacing more
        than ``max_gap_s`` (1.5 median sample intervals by default).
    """
    if time.shape != valid.shape or time.ndim != 1:
        raise ValueError("time and valid must be 1-D and the same length.")
    finite = time[np.isfinite(time)]
    if np.any(np.diff(finite) <= 0):
        raise ValueError("time must be strictly increasing.")
    if finite.size < 2:
        # nothing can continue a run of one sample
        return np.zeros(max(time.size - 1, 0), dtype=bool)
    if max_gap_s is None:
        max_gap_s = GAP_FACTOR * float(np.median(np.diff(finite)))
    with np.errstate(invalid="ignore"):
        close = np.diff(time) <= max_gap_s
    # a non-finite timestamp fails the spacing test on both sides
    joined: NDArray = valid[1:] & valid[:-1] & close
    return joined


def sample_runs(
    time: NDArray, valid: NDArray, max_gap_s: float | None = None
) -> NDArray:
    """Stretches of consecutive valid samples, from first to last sample.

    A run ends at an invalid sample, a non-finite timestamp, and any gap in
    the timestamps longer than ``max_gap_s`` (1.5 median sample intervals by
    default). A single isolated sample covers no time and is dropped.

    Parameters
    ----------
    time : ndarray, shape (n_samples,)
        Strictly increasing where finite.
    valid : ndarray of bool, shape (n_samples,)
    max_gap_s : float, optional

    Returns
    -------
    runs : ndarray, shape (n_runs, 2)

    """
    time = np.asarray(time, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if time.size < 2:
        return np.empty((0, 2))
    joined = _joined(time, valid, max_gap_s)
    valid = valid & np.isfinite(time)
    # a new run begins at a valid sample after an invalid one or after a gap
    begins = valid & np.r_[True, ~joined]
    ends = valid & np.r_[~joined, True]
    return normalize(np.column_stack([time[begins], time[ends]]))


def window_supported(
    time: NDArray,
    valid: NDArray,
    window_n_samples: int,
    max_gap_s: float | None = None,
) -> NDArray:
    """Samples at which a window stamped there has its whole support.

    A window over samples ``[k, k + window_n_samples)`` is stamped at sample
    ``k + window_n_samples // 2``, as the memoryless detector stamps it, so
    a sample is covered when the window around it is valid without a break.
    Counted in samples, not seconds, so the coverage edges are the first and
    last stamps on any clock.

    Parameters
    ----------
    time : ndarray, shape (n_samples,)
    valid : ndarray of bool, shape (n_samples,)
    window_n_samples : int
    max_gap_s : float, optional
        As for ``sample_runs``.

    Returns
    -------
    supported : ndarray of bool, shape (n_samples,)
        Pass to ``sample_runs`` with ``time`` for the observed intervals.

    """
    time = np.asarray(time, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    supported = np.zeros(time.size, dtype=bool)
    if time.size < max(window_n_samples, 2):
        return supported
    # samples share a run id exactly when no break lies between them
    run = np.r_[0, np.cumsum(~_joined(time, valid, max_gap_s))]
    first = np.arange(time.size - window_n_samples + 1)
    last = first + window_n_samples - 1
    stamp = first + window_n_samples // 2
    supported[stamp] = valid[stamp] & (run[first] == run[last])
    return supported


def _still_samples(
    time: NDArray, speed_cm_per_s: NDArray, threshold_cm_per_s: float
) -> tuple[NDArray, NDArray, NDArray]:
    """The trace as arrays, with which samples are known and which still.

    Known is a finite timestamp and a finite speed; still is known and below
    the threshold.
    """
    time = np.asarray(time, dtype=np.float64)
    speed = np.asarray(speed_cm_per_s, dtype=np.float64)
    known = np.isfinite(time) & np.isfinite(speed)
    return time, known, known & (speed < threshold_cm_per_s)


def not_still_times(
    time: NDArray,
    speed_cm_per_s: NDArray,
    threshold_cm_per_s: float = DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
) -> NDArray:
    """Times of the samples that were not still: moving, or of unknown speed.

    An event holding one of them is not speed-eligible.

    Returns
    -------
    times : ndarray, shape (n_not_still,)
        The finite timestamps among them.

    """
    time, _, still = _still_samples(time, speed_cm_per_s, threshold_cm_per_s)
    times: NDArray = time[np.isfinite(time) & ~still]
    return times


def still_intervals(
    time: NDArray,
    speed_cm_per_s: NDArray,
    threshold_cm_per_s: float = DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    max_gap_s: float | None = None,
) -> NDArray:
    """Stretches of finite speed below the threshold; an unknown speed is not still.

    An event is speed-eligible when every sample inside it is still, so an
    event may end after a run's last still sample, or start before its first,
    without touching a moving sample. Each run of still samples therefore
    reaches the sample on either side of it, moving or not, unless a gap in
    the timestamps (as for ``sample_runs``), a non-finite timestamp or a
    non-finite speed lies between them: exposure grows by at most one sample
    interval at each end of a run. A still sample alone between two moving
    ones spans the time from one to the other.

    Parameters
    ----------
    time : ndarray, shape (n_samples,)
        Strictly increasing where finite.
    speed_cm_per_s : ndarray, shape (n_samples,)
    threshold_cm_per_s : float, optional
    max_gap_s : float, optional
        As for ``sample_runs``.

    Returns
    -------
    still : ndarray, shape (n_runs, 2)

    """
    time, known, still = _still_samples(time, speed_cm_per_s, threshold_cm_per_s)
    if time.size < 2:
        return np.empty((0, 2))
    # whether each pair of neighbouring samples is known and unbroken
    joined = _joined(time, known, max_gap_s)
    within_run = joined & still[1:] & still[:-1]
    first = np.flatnonzero(still & np.r_[True, ~within_run])
    last = np.flatnonzero(still & np.r_[~within_run, True])
    first = first - np.r_[False, joined][first]
    last = last + np.r_[joined, False][last]
    return normalize(np.column_stack([time[first], time[last]]))


def exposure(observed: NDArray, still: NDArray) -> NDArray:
    """Observed time during which the animal was still."""
    return intersect(observed, still)


def _epoch_bounds(epoch_bounds: tuple[float, float]) -> tuple[float, float]:
    start, end = float(epoch_bounds[0]), float(epoch_bounds[1])
    if not (np.isfinite(start) and np.isfinite(end) and start < end):
        raise ValueError(
            f"epoch_bounds must be finite with start < end, got {epoch_bounds}."
        )
    return start, end


def _table(
    session: SessionIdentity, detector: str, run_id: str, rows: list[tuple]
) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["start_s", "end_s", "available", "reason"])
    for name, value in (
        ("animal", session.animal),
        ("day", session.day),
        ("epoch", session.epoch),
        ("brain_area", session.brain_area),
        ("detector", detector),
        ("run_id", run_id),
    ):
        frame[name] = value
    return frame[list(OBSERVATION_DTYPES)].astype(dict(OBSERVATION_DTYPES))


def observation_table(
    session: SessionIdentity,
    *,
    detector: str,
    run_id: str,
    epoch_bounds: tuple[float, float],
    observed: NDArray,
    unavailable_reason: str,
) -> pd.DataFrame:
    """The epoch, tiled into observed and unavailable stretches.

    Parameters
    ----------
    session : SessionIdentity
    detector, run_id : str
    epoch_bounds : (float, float)
        First and last time of the epoch.
    observed : ndarray, shape (n, 2)
        What the detector observed; must lie within the epoch.
    unavailable_reason : str
        Why the rest was not observed.

    Returns
    -------
    table : DataFrame
        Columns and dtypes as ``OBSERVATION_DTYPES``, in time order, the
        stretches meeting end to end.

    """
    start, end = _epoch_bounds(epoch_bounds)
    observed = normalize(observed, strict=True)
    if observed.size and (observed[0, 0] < start or observed[-1, 1] > end):
        raise ValueError("Observed time must lie within the epoch.")
    unavailable = subtract(np.array([[start, end]]), observed)
    rows = [(a, b, True, "") for a, b in observed]
    rows += [(a, b, False, unavailable_reason) for a, b in unavailable]
    rows.sort(key=lambda row: row[0])
    return _table(session, detector, run_id, rows)


def missing_run_table(
    session: SessionIdentity,
    *,
    detector: str,
    run_id: str,
    epoch_bounds: tuple[float, float],
    reason: str,
) -> pd.DataFrame:
    """A detector that produced nothing for the epoch observed none of it."""
    start, end = _epoch_bounds(epoch_bounds)
    return _table(session, detector, run_id, [(start, end, False, reason)])


def coverage_from_observations(
    observations: pd.DataFrame, still: Mapping[tuple, NDArray]
) -> dict[tuple, Coverage]:
    """The coverage map the counts read, from observation tables.

    One origin for both representations of what a detector observed: the
    ``available`` stretches less the epoch's excluded time
    (``swr_source.excluded_intervals``) become ``observed``, and their
    intersection with the epoch's still intervals becomes ``exposure``. A
    detector's candidates in excluded time are dropped from its cohort, so its
    rate's denominator loses that time too, for every detector alike; with no
    artifact run configured nothing is removed. Every (epoch, detector)
    in the tables gets an entry, so a detector whose run was missing, entered
    through ``missing_run_table``, stays on the roster with empty coverage
    and reads as unavailable rather than vanishing.

    Parameters
    ----------
    observations : DataFrame
        ``observation_table`` and ``missing_run_table`` rows, concatenated.
    still : mapping
        ``(animal, day, epoch) -> ndarray (n, 2)``, the epoch's still
        intervals from ``still_intervals``; an epoch absent here is taken as
        never still.

    Returns
    -------
    coverage : dict
        ``(animal, day, epoch, detector) -> Coverage``.

    Raises
    ------
    SwrSourceError
        If an artifact run is configured and cannot serve an epoch.

    """
    excluded: dict[tuple, NDArray] = {}
    coverage = {}
    for (session, detector), group in observations.groupby(
        [observations[list(SESSION_COLUMNS_DAY)].apply(tuple, axis=1), "detector"],
        sort=False,
    ):
        available = group[group.available]
        observed = (
            available[["start_s", "end_s"]].to_numpy(dtype=np.float64)
            if len(available)
            else np.empty((0, 2))
        )
        if session not in excluded:
            excluded[session] = (
                swr_source.excluded_intervals_of(group)[["start_time", "end_time"]]
                .to_numpy(dtype=np.float64)
                .reshape(-1, 2)
            )
        observed = subtract(observed, excluded[session])
        coverage[(*session, detector)] = Coverage(
            observed=observed,
            exposure=exposure(observed, still.get(tuple(session), np.empty((0, 2)))),
        )
    return coverage


def stored_coverage(
    observations: pd.DataFrame,
    set_aside: pd.DataFrame,
    still: Mapping[tuple, NDArray],
) -> dict[tuple, Coverage]:
    """The coverage a comparison read, rebuilt from the tables it stored.

    The comparison builds its coverage from the observation tables and the
    still intervals, removes the span of each event the analysed-time rule
    dropped from its own detector's coverage (``set_aside_events``), then
    removes the span of each event bridging its own detector's unobserved time
    (``remove_bridging_events``). The stored set-aside table holds both kinds,
    the bridging ones with their event spans, and subtracting every span at
    once leaves the same interval sets as subtracting them in two steps.

    Parameters
    ----------
    observations : DataFrame
        The stored ``observations`` table: ``observation_table`` and
        ``missing_run_table`` rows, concatenated.
    set_aside : DataFrame
        The stored ``set_aside`` table, columns as
        ``overlap.SET_ASIDE_DTYPES``: dropped and bridging events alike.
    still : mapping
        ``(animal, day, epoch) -> ndarray (n, 2)``, as for
        ``coverage_from_observations``: ``still_intervals`` on the speed trace
        the comparison read, with its threshold and largest step.

    Returns
    -------
    coverage : dict
        ``(animal, day, epoch, detector) -> Coverage``, each set-aside span
        unobserved by its own detector only.

    Raises
    ------
    ValueError
        If a set-aside event's detector has no observation rows in its epoch.
    SwrSourceError
        If an artifact run is configured and cannot serve an epoch.

    """
    return set_aside_events(coverage_from_observations(observations, still), set_aside)


def _intervals(frame: pd.DataFrame, start: str, end: str) -> NDArray:
    intervals: NDArray = frame[[start, end]].to_numpy(dtype=np.float64)
    return intervals.reshape(-1, 2)


def exclusion_by_reason(
    observations: pd.DataFrame,
    coverage: Mapping[tuple, Coverage],
    still: Mapping[tuple, NDArray],
    set_aside: pd.DataFrame,
    *,
    comparisons: Mapping[str, Collection[str]],
    speed_known: Mapping[tuple, NDArray],
) -> pd.DataFrame:
    """Each epoch's seconds outside each comparison's shared exposure, by reason.

    The shared exposure is the intersection of the comparison's detectors'
    exposures. Every other second of the epoch goes to the first reason that
    explains it, in this order: ``excluded_time``; each detector's own
    unavailable reason from its observation table, ``"<detector>: <reason>"``;
    the span of each event it set aside, ``"<detector>: <reason>"``; then
    speed not still, ``moving`` where the speed was known and
    ``speed_unknown`` where it was not. Within each step the detectors are
    taken in name order, so a stretch two of them miss is counted once, under
    the first.

    Parameters
    ----------
    observations : DataFrame
        ``observation_table`` and ``missing_run_table`` rows, concatenated;
        each epoch's span is from its first to its last row.
    coverage : mapping
        ``(animal, day, epoch, detector) -> Coverage``, as the counts read
        it: from ``coverage_from_observations`` on these tables and ``still``,
        then ``set_aside_events`` with ``set_aside``.
    still : mapping
        ``(animal, day, epoch) -> ndarray (n, 2)``, as for
        ``coverage_from_observations``.
    set_aside : DataFrame
        Columns as ``overlap.SET_ASIDE_DTYPES``.
    comparisons : mapping
        Name -> the detectors compared.
    speed_known : mapping
        ``(animal, day, epoch) -> ndarray (n, 2)``, where the speed was
        known: ``sample_runs(time, np.isfinite(speed))`` on the trace
        ``still`` was read from. An epoch absent here is taken as never
        known, as one absent from ``still`` is never still.

    Returns
    -------
    rows : DataFrame
        Columns and dtypes as ``EXCLUSION_DTYPES``. Per epoch and comparison,
        a ``shared_exposure`` row and one row per reason with time, in the
        order above; the rows sum to the epoch span.

    Raises
    ------
    ValueError
        If a comparison names a detector no coverage entry knows, or if the
        coverage was not built from these inputs: a reason's time lies in the
        shared exposure, or time outside it has no reason.
    SwrSourceError
        If an artifact run is configured and cannot serve an epoch.

    """
    names = {
        name: sorted(validate_detectors(detectors, coverage, f"Comparison {name!r}"))
        for name, detectors in comparisons.items()
    }
    empty = np.empty((0, 2))
    rows = []
    for session, epoch_rows in observations.groupby(list(SESSION_COLUMNS_DAY)):
        span = np.array([[epoch_rows.start_s.min(), epoch_rows.end_s.max()]])
        excluded = _intervals(
            swr_source.excluded_intervals_of(epoch_rows), "start_time", "end_time"
        )
        known = speed_known.get(session, empty)
        not_still = [
            ("moving", subtract(known, still.get(session, empty))),
            ("speed_unknown", subtract(span, known)),
        ]
        dropped = set_aside[
            np.logical_and.reduce(
                [
                    set_aside[column] == value
                    for column, value in zip(SESSION_COLUMNS_DAY, session, strict=True)
                ]
            )
        ]
        unavailable = epoch_rows[~epoch_rows.available]
        for name, detectors in names.items():
            reasons = [("excluded_time", excluded)]
            for table, start, end in (
                (unavailable, "start_s", "end_s"),
                (dropped, "interval_start_s", "interval_end_s"),
            ):
                for detector in detectors:
                    own = table[table.detector == detector]
                    reasons += [
                        (f"{detector}: {reason}", _intervals(group, start, end))
                        for reason, group in own.groupby("reason", sort=False)
                    ]
            reasons += not_still
            shared = reduce(
                intersect,
                (
                    coverage[(*session, d)].exposure
                    if (*session, d) in coverage
                    else empty
                    for d in detectors
                ),
            )
            seconds = {"shared_exposure": total_duration(shared)}
            remainder = subtract(span, shared)
            for reason, time in reasons:
                if total_duration(intersect(shared, time)) > 0:
                    raise ValueError(
                        f"{name} in {session}: time under {reason!r} lies in the "
                        "shared exposure; the coverage was not built from these "
                        "inputs (was set_aside_events run?)."
                    )
                taken = total_duration(intersect(remainder, time))
                if taken > 0:
                    seconds[reason] = seconds.get(reason, 0.0) + taken
                remainder = subtract(remainder, time)
            if total_duration(remainder) > 0:
                raise ValueError(
                    f"{name} in {session}: {total_duration(remainder)} s outside the "
                    "shared exposure has no reason; the coverage was not built "
                    "from these inputs."
                )
            rows += [
                dict(zip(SESSION_COLUMNS_DAY, session, strict=True))
                | {"comparison": name, "reason": reason, "seconds": value}
                for reason, value in seconds.items()
            ]
    if not rows:
        return pd.DataFrame(
            {k: pd.Series(dtype=v) for k, v in EXCLUSION_DTYPES.items()}
        )
    return pd.DataFrame(rows)[list(EXCLUSION_DTYPES)].astype(dict(EXCLUSION_DTYPES))
