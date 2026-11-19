"""One analyzed-time rule for every event cohort, exposure, bin and window.

**Excluded time** is the one artifact mask (``artifact_masks``) widened by
:data:`ARTIFACT_REACH_S` on each side (:func:`excluded_time`). An event of any
kind (non-local, SWR, MUA, ContFrag label, Widloski) enters a cohort only when

- the whole event is immobile: every speed sample in its closed interval is
  finite and strictly below the threshold (``immobility``); and
- it lies wholly outside excluded time (closed intervals: touching counts). An
  event in the mask itself is ``artifact``; one within the reach of it is
  ``near_artifact``. Either way it is unobserved: not an SWR, not a non-SWR
  event, in no cohort.

A kept event that overlaps an SWR or MUA event dropped for excluded time is
dropped too (``overlaps_dropped_event``): its overlap class is unknown. One pass
is made; an event dropped by that pass drops no other.

Exposure seconds, time bins with their history windows, and peri-event windows
follow the same rule: they are analyzed only outside excluded time. Inputs (the
decoders' missing bins and encoding exclusion, the SWR detectors'
normalization, the ripple-power covariate) use the mask itself, not this.

A dropped event keeps its reason (:class:`DropReason`). Design note:
``docs/design/2026-10-09-artifact-detection.md``.
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from continuum_swr_replay.analysis.event_definition.immobility import (
    intervals_from_labels,
)
from continuum_swr_replay.intervals import (
    overlaps_intervals,
    pad_intervals,
)
from continuum_swr_replay.parameters import DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC

__all__ = [
    "ARTIFACT_REACH_S",
    "EXCLUDED_TIME_REASONS",
    "DropReason",
    "artifact_drop_reasons",
    "drop_labels",
    "event_drop_reasons",
    "excluded_time",
    "label_drop_reasons",
    "overlaps_dropped_events",
]

ARTIFACT_REACH_S = 0.3
"""Seconds by which the artifact mask is widened into excluded time.

Why 0.3 s: study C3 masked gaps in the decodes, and a gap changed analyzed
NonLocal events within at most 0.21 s and flipped ContFrag SWR labels within at
most 0.19 s; the reach did not grow with the gap's length. The lead chose
(2026-10-10) to apply it to every cohort, exposure, bin and window, not only to
decoded events, so all rates share one exposure and the NonLocal and ContFrag
cohorts agree."""


class DropReason(StrEnum):
    """Why an event leaves its cohort: the first that holds, in this order.
    ``overlaps_dropped_event`` is given only to an event every other rule
    keeps."""

    ARTIFACT = "artifact"
    """The event shares time with the artifact mask (closed: touching counts)."""
    NEAR_ARTIFACT = "near_artifact"
    """The event lies within :data:`ARTIFACT_REACH_S` of the mask, not in it."""
    NO_SPEED_SAMPLE = "no_speed_sample"
    """No sample of the speed's time axis lies in the event."""
    MOVING = "moving"
    """A finite speed sample in the event is at or above the threshold."""
    SPEED_UNKNOWN = "speed_unknown"
    """A speed sample in the event is not finite (and none is fast)."""
    OVERLAPS_DROPPED_EVENT = "overlaps_dropped_event"
    """The event overlaps an SWR or MUA event dropped for excluded time, so its
    overlap class is unknown."""


EXCLUDED_TIME_REASONS = frozenset(
    {DropReason.ARTIFACT.value, DropReason.NEAR_ARTIFACT.value}
)
"""The reasons that put an event in excluded time: the drops another event's
overlap is checked against."""


def excluded_time(artifacts: pd.DataFrame) -> pd.DataFrame:
    """Excluded time: the artifact mask widened by :data:`ARTIFACT_REACH_S`.

    Parameters
    ----------
    artifacts : pd.DataFrame
        One epoch's mask: ``start_time`` and ``end_time`` (seconds), closed.

    Returns
    -------
    pd.DataFrame
        ``start_time`` and ``end_time`` (seconds), closed, merged, in time order.
    """
    return pad_intervals(artifacts, ARTIFACT_REACH_S)


def artifact_drop_reasons(
    starts: NDArray[np.float64],
    ends: NDArray[np.float64],
    artifacts: pd.DataFrame,
    dropped: pd.DataFrame | None = None,
) -> NDArray[np.object_]:
    """Each event's reason to leave its cohort for excluded time, or None.

    Parameters
    ----------
    starts, ends : ndarray, shape (n_events,)
        Each event's closed interval, seconds.
    artifacts : pd.DataFrame
        The epoch's mask (not excluded time): ``start_time`` and ``end_time``,
        closed.
    dropped : pd.DataFrame, optional
        The SWR and MUA events dropped for excluded time
        (``analyzed_time.dropped_for_excluded_time``), for a cohort not read
        through the source (events re-detected from a posterior, Widloski
        events): an event every other rule keeps that overlaps one is
        ``overlaps_dropped_event``.

    Returns
    -------
    ndarray of object, shape (n_events,)
        ``"artifact"`` for an event sharing time with the mask,
        ``"near_artifact"`` for one in excluded time but not in the mask,
        ``"overlaps_dropped_event"`` as above, None for the rest.
    """
    starts = np.asarray(starts, dtype=np.float64)
    ends = np.asarray(ends, dtype=np.float64)
    reasons = np.full(starts.shape, None, dtype=object)
    reasons[overlaps_intervals(starts, ends, excluded_time(artifacts))] = (
        DropReason.NEAR_ARTIFACT.value
    )
    reasons[overlaps_intervals(starts, ends, artifacts)] = DropReason.ARTIFACT.value
    if dropped is not None:
        kept = pd.isna(reasons)
        reasons[kept & overlaps_dropped_events(starts, ends, dropped)] = (
            DropReason.OVERLAPS_DROPPED_EVENT.value
        )
    return reasons


def overlaps_dropped_events(
    starts: NDArray[np.float64],
    ends: NDArray[np.float64],
    dropped: pd.DataFrame,
) -> NDArray[np.bool_]:
    """True for each event that overlaps a dropped event.

    Overlap is the open test ``dropped_end > start`` and
    ``dropped_start < end`` against the dropped intervals of positive
    duration, so an event only touching a dropped one at an instant keeps a
    known (``non_swr``) class. It is not quite strict (``is_swr``'s) overlap:
    a zero-length dropped interval is ignored, but the event's own duration is
    not checked, so a zero-length event strictly inside a dropped one
    overlaps it. The conventions table in :mod:`continuum_swr_replay.intervals`
    sets this one beside the others.

    Parameters
    ----------
    starts, ends : ndarray, shape (n_events,)
        Seconds.
    dropped : pd.DataFrame
        ``start_time`` and ``end_time`` of the SWR and MUA events dropped for
        excluded time.

    Returns
    -------
    ndarray of bool, shape (n_events,)
    """
    starts = np.asarray(starts, dtype=np.float64)
    ends = np.asarray(ends, dtype=np.float64)
    dropped_starts = dropped["start_time"].to_numpy(dtype=np.float64)
    dropped_ends = dropped["end_time"].to_numpy(dtype=np.float64)
    positive = dropped_ends > dropped_starts
    dropped_starts, dropped_ends = dropped_starts[positive], dropped_ends[positive]
    if starts.size == 0 or dropped_starts.size == 0:
        return np.zeros(starts.shape, dtype=bool)
    order = np.argsort(dropped_starts, kind="stable")
    dropped_starts = dropped_starts[order]
    reach = np.maximum.accumulate(dropped_ends[order])
    # the dropped intervals starting strictly before each event's end; one of
    # them overlaps the event iff the furthest end among them passes its start
    last = np.searchsorted(dropped_starts, ends, side="left") - 1
    overlaps: NDArray[np.bool_] = (last >= 0) & (reach[np.maximum(last, 0)] > starts)
    return overlaps


def event_drop_reasons(
    events: pd.DataFrame,
    *,
    speed: NDArray[np.float64],
    time: NDArray[np.float64],
    artifacts: pd.DataFrame,
    speed_threshold: float = DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
) -> NDArray[np.object_]:
    """The rule on events given as intervals: each one's reason, or None.

    The speed part is the canonical whole-event immobility rule (as
    ``immobility.filter_event_intervals_by_speed``): the samples in the closed
    interval, every one finite and strictly below `speed_threshold`, counted
    here for every event at once. An event that fails it is named by why.
    Excluded time's reasons come first. The pass for ``overlaps_dropped_event``
    needs the other cohorts and is not made here.

    Parameters
    ----------
    events : pd.DataFrame
        ``start_time`` and ``end_time`` (seconds), one row per event, each
        ending at or after it starts.
    speed : ndarray, shape (n_time,)
        Head speed (cm/s) on `time`.
    time : ndarray, shape (n_time,)
        Seconds, strictly increasing.
    artifacts : pd.DataFrame
        The epoch's mask, closed intervals.
    speed_threshold : float
        cm/s, strict.

    Returns
    -------
    ndarray of object, shape (n_events,)
        A :class:`DropReason` value per dropped event, None per kept one.

    Raises
    ------
    ValueError
        As ``filter_event_intervals_by_speed``: `speed_threshold` not finite
        and positive; `time` and `speed` not one-dimensional or of unequal
        length; `time` not finite or not strictly increasing; an event bound
        not finite, or an event ending before it starts.
    """
    threshold = float(speed_threshold)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError(
            f"speed_threshold must be finite and positive, got {speed_threshold!r}"
        )
    time = np.asarray(time, dtype=np.float64)
    values = np.asarray(speed, dtype=np.float64)
    if time.ndim != 1 or values.ndim != 1:
        raise ValueError("time and speed must be one-dimensional")
    if len(time) != len(values):
        raise ValueError(
            f"time and speed must have equal length, got {len(time)} and {len(values)}"
        )
    if not np.isfinite(time).all():
        raise ValueError("time must contain only finite values")
    if not (np.diff(time) > 0).all():
        raise ValueError("time must be strictly increasing")
    starts = events["start_time"].to_numpy(dtype=np.float64)
    ends = events["end_time"].to_numpy(dtype=np.float64)
    if not (np.isfinite(starts).all() and np.isfinite(ends).all()):
        raise ValueError("event start_time and end_time must be finite")
    if (ends < starts).any():
        raise ValueError("event end_time must be greater than or equal to start_time")
    first = np.searchsorted(time, starts, side="left")
    stop = np.searchsorted(time, ends, side="right")
    finite = np.isfinite(values)
    fast = finite & (values >= threshold)

    def _in_event(flags: NDArray[np.bool_]) -> NDArray[np.int64]:
        cumulative = np.concatenate([[0], np.cumsum(flags, dtype=np.int64)])
        counts: NDArray[np.int64] = cumulative[stop] - cumulative[first]
        return counts

    reasons = np.full(len(events), None, dtype=object)
    reasons[_in_event(~finite) > 0] = DropReason.SPEED_UNKNOWN.value
    reasons[_in_event(fast) > 0] = DropReason.MOVING.value
    reasons[stop <= first] = DropReason.NO_SPEED_SAMPLE.value
    excluded = artifact_drop_reasons(starts, ends, artifacts)
    in_excluded = ~pd.isna(excluded)
    reasons[in_excluded] = excluded[in_excluded]
    return reasons


def label_drop_reasons(
    frame: pd.DataFrame,
    event_col: str,
    artifacts: pd.DataFrame,
) -> pd.DataFrame:
    """Every labeled event of one epoch's frame, with its excluded-time reason.

    The labels are already speed-qualified where the frame was made (the
    decode's ``get_data_over_time``), so only excluded time's reasons apply
    here; each event is read at its sampled bounds, its first and last
    labeled bin.

    Parameters
    ----------
    frame : pd.DataFrame
        One epoch, time-indexed, carrying
        `event_col` (0 outside events).
    event_col : str
        Such as ``"non_local_event"`` or ``"MUA_event"``.
    artifacts : pd.DataFrame
        The epoch's mask, closed intervals.

    Returns
    -------
    pd.DataFrame
        One row per positive label, in label order: `event_col`,
        ``start_time``, ``end_time`` (seconds) and ``drop_reason`` (a
        :class:`DropReason` value, or None for a kept event).
    """
    intervals = intervals_from_labels(frame, event_col)
    drops = pd.DataFrame(
        {
            event_col: intervals[event_col].to_numpy(),
            "start_time": intervals["start_time"].to_numpy(dtype=np.float64),
            "end_time": intervals["end_time"].to_numpy(dtype=np.float64),
        }
    )
    drops["drop_reason"] = artifact_drop_reasons(
        drops["start_time"].to_numpy(), drops["end_time"].to_numpy(), artifacts
    )
    return drops


def drop_labels(
    frame: pd.DataFrame, event_col: str, drops: pd.DataFrame
) -> pd.DataFrame:
    """`frame` with each dropped event's label zeroed whole.

    The other events keep their numbers, so an event's ``event_id`` (hashed
    from its label) does not change when a neighbor is dropped.

    Parameters
    ----------
    frame : pd.DataFrame
        One epoch, carrying `event_col`.
    event_col : str
    drops : pd.DataFrame
        One row per labeled event: `event_col` and ``drop_reason`` (None for
        a kept event), as :func:`label_drop_reasons` returns it.

    Returns
    -------
    pd.DataFrame
        A shallow copy (the untouched columns are shared); `frame` itself is
        not changed. When nothing is dropped it is the copy unchanged.
    """
    out = frame.copy(deep=False)
    dropped = drops.loc[drops["drop_reason"].notna(), event_col].to_numpy()
    if dropped.size:
        labels = out[event_col].to_numpy(copy=True)
        labels[np.isin(labels, dropped)] = 0
        out[event_col] = labels
    return out
