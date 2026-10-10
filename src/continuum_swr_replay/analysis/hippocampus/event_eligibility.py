"""One eligibility rule for every event cohort: immobile, and clear of the mask.

An event of any kind (non-local, SWR, MUA, ContFrag label, Widloski) enters a
cohort only when

- the whole event is immobile: every speed sample in its closed interval is
  finite and strictly below the threshold (``event_immobility``); and
- it shares no time with the one artifact mask (``artifacts.masks``; closed
  intervals, so touching counts). Artifact time is unobserved, so an event
  there is neither an SWR nor a non-SWR event: it leaves every cohort.

A *decoded* event (a non-local event, a ContFrag label, a Widloski event) must
also lie farther than :data:`DECODER_REACH_S` from the mask, because the
decoder's posterior near a masked gap still feels it. SWR and MUA events come
from the LFP and spike detectors, which have no such reach.

A dropped event keeps its reason (:class:`DropReason`), the first that holds
in the enum's order. Design note: ``docs/design/2026-10-09-artifact-detection.md``.
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from continuum_swr_replay.analysis.constants import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
)
from continuum_swr_replay.analysis.hippocampus.artifacts.masks import (
    overlaps_intervals,
    pad_intervals,
)
from continuum_swr_replay.analysis.hippocampus.event_immobility import (
    filter_event_intervals_by_speed,
    intervals_from_labels,
)

__all__ = [
    "DECODER_REACH_S",
    "DropReason",
    "artifact_drop_reasons",
    "drop_labels",
    "event_drop_reasons",
    "label_drop_reasons",
]

DECODER_REACH_S = 0.3
"""Seconds within which a masked gap still moves a decoded event.

Study C3 masked gaps in the decodes: analysed NonLocal events changed within at
most 0.21 s of a gap and ContFrag SWR labels flipped within at most 0.19 s,
and the reach did not grow with the gap's length (the plan's Phase C rulings,
2026-10-09). A decoded event closer than this to the mask is dropped."""


class DropReason(StrEnum):
    """Why an event leaves its cohort: the first that holds, in this order."""

    ARTIFACT = "artifact"
    """The event shares time with the artifact mask (closed: touching counts)."""
    DECODER_REACH = "decoder_reach"
    """A decoded event within :data:`DECODER_REACH_S` of the mask."""
    NO_SPEED_SAMPLE = "no_speed_sample"
    """No sample of the speed's time axis lies in the event."""
    MOVING = "moving"
    """A finite speed sample in the event is at or above the threshold."""
    SPEED_UNKNOWN = "speed_unknown"
    """A speed sample in the event is not finite (and none is fast)."""


def artifact_drop_reasons(
    starts: NDArray[np.float64],
    ends: NDArray[np.float64],
    artifacts: pd.DataFrame,
    *,
    reach_s: float = 0.0,
) -> NDArray[np.object_]:
    """Each event's reason to leave its cohort because of the mask, or None.

    Parameters
    ----------
    starts, ends : ndarray, shape (n_events,)
        Each event's closed interval, seconds.
    artifacts : pd.DataFrame
        The epoch's mask: ``start_time`` and ``end_time`` (seconds), closed.
    reach_s : float
        :data:`DECODER_REACH_S` for a decoded event, 0 for an SWR or MUA event.

    Returns
    -------
    ndarray of object, shape (n_events,)
        ``"artifact"`` for an event sharing time with the mask,
        ``"decoder_reach"`` for one within `reach_s` of it (and not in it),
        None for the rest.
    """
    starts = np.asarray(starts, dtype=np.float64)
    ends = np.asarray(ends, dtype=np.float64)
    reasons = np.full(starts.shape, None, dtype=object)
    if reach_s > 0:
        reasons[overlaps_intervals(starts, ends, pad_intervals(artifacts, reach_s))] = (
            DropReason.DECODER_REACH.value
        )
    reasons[overlaps_intervals(starts, ends, artifacts)] = DropReason.ARTIFACT.value
    return reasons


def event_drop_reasons(
    events: pd.DataFrame,
    *,
    speed: NDArray[np.float64],
    time: NDArray[np.float64],
    artifacts: pd.DataFrame,
    reach_s: float = 0.0,
    speed_threshold: float = DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
) -> NDArray[np.object_]:
    """The whole rule on events given as intervals: each one's reason, or None.

    The canonical whole-event immobility rule decides the speed part
    (``filter_event_intervals_by_speed``: samples in the closed interval, every
    one finite and strictly below `speed_threshold`); the counts here only name
    why it failed. The mask's reasons come first.

    Parameters
    ----------
    events : pd.DataFrame
        ``start_time`` and ``end_time`` (seconds), one row per event.
    speed : ndarray, shape (n_time,)
        Head speed (cm/s) on `time`.
    time : ndarray, shape (n_time,)
        Seconds, strictly increasing.
    artifacts : pd.DataFrame
        The epoch's mask, closed intervals.
    reach_s : float
        As :func:`artifact_drop_reasons`.
    speed_threshold : float
        cm/s, strict.

    Returns
    -------
    ndarray of object, shape (n_events,)
        A :class:`DropReason` value per dropped event, None per kept one.

    Raises
    ------
    RuntimeError
        If an event fails the immobility rule for no reason named here.
    """
    time = np.asarray(time, dtype=np.float64)
    values = np.asarray(speed, dtype=np.float64)
    starts = events["start_time"].to_numpy(dtype=np.float64)
    ends = events["end_time"].to_numpy(dtype=np.float64)
    numbered = pd.DataFrame(
        {"start_time": starts, "end_time": ends}, index=np.arange(len(events))
    )
    immobile = numbered.index.isin(
        filter_event_intervals_by_speed(
            numbered, speed=values, time=time, speed_threshold=speed_threshold
        ).index
    )
    first = np.searchsorted(time, starts, side="left")
    stop = np.searchsorted(time, ends, side="right")
    finite = np.isfinite(values)
    fast = finite & (values >= speed_threshold)

    def _in_event(flags: NDArray[np.bool_]) -> NDArray[np.int64]:
        cumulative = np.concatenate([[0], np.cumsum(flags, dtype=np.int64)])
        counts: NDArray[np.int64] = cumulative[stop] - cumulative[first]
        return counts

    reasons = np.full(len(events), None, dtype=object)
    reasons[~immobile & (_in_event(~finite) > 0)] = DropReason.SPEED_UNKNOWN.value
    reasons[~immobile & (_in_event(fast) > 0)] = DropReason.MOVING.value
    reasons[~immobile & (stop <= first)] = DropReason.NO_SPEED_SAMPLE.value
    if (pd.isna(reasons) & ~immobile).any():
        raise RuntimeError(
            "an event failed the immobility rule for no reason this module names"
        )
    masked = artifact_drop_reasons(starts, ends, artifacts, reach_s=reach_s)
    has_mask_reason = ~pd.isna(masked)
    reasons[has_mask_reason] = masked[has_mask_reason]
    return reasons


def label_drop_reasons(
    frame: pd.DataFrame,
    event_col: str,
    artifacts: pd.DataFrame,
    *,
    reach_s: float,
) -> pd.DataFrame:
    """Every labelled event of one epoch's frame, with its reason to leave.

    The labels are already speed-qualified where the frame was made (the
    decode's ``get_data_over_time``), so only the mask's reasons apply; each
    event is read at its sampled bounds, its first and last labelled bin.

    Parameters
    ----------
    frame : pd.DataFrame
        One epoch, time-indexed or with a ``time`` column, carrying
        `event_col` (0 outside events).
    event_col : str
        Such as ``"non_local_event"`` or ``"MUA_event"``.
    artifacts : pd.DataFrame
        The epoch's mask, closed intervals.
    reach_s : float
        As :func:`artifact_drop_reasons`.

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
        drops["start_time"].to_numpy(),
        drops["end_time"].to_numpy(),
        artifacts,
        reach_s=reach_s,
    )
    return drops


def drop_labels(
    frame: pd.DataFrame, event_col: str, drops: pd.DataFrame
) -> pd.DataFrame:
    """`frame` with each dropped event's label zeroed whole.

    The other events keep their numbers, so an event's ``event_id`` (hashed
    from its label) does not change when a neighbour is dropped.

    Parameters
    ----------
    frame : pd.DataFrame
        One epoch, carrying `event_col`.
    event_col : str
    drops : pd.DataFrame
        From :func:`label_drop_reasons` on the same frame.

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
