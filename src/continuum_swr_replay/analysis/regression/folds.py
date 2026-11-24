"""Folds of whole trials, balanced by destination well, long stays cut into pieces.

The project's one fold rule, for the regression models and the held-out scores
of the event evidence alike. Inside an epoch each row (a time bin, or an event
placed by its start time) belongs to a trial segment: from a
trial's departure from a well to the next departure, so the run to a well and
the whole stay there are one segment, its class the destination well
(:func:`task_segment_starts`). A segment longer than :data:`MAX_SEGMENT_S` is
cut into equal pieces (:func:`capped_groups`), so that a long stay is spread
over the folds. The cap is a number of rows: 30 s at the median spacing of the
rows, so a piece is 30 s of evenly spaced bins and may span more or less time
on events, which are not evenly spaced. The pieces of one segment can fall in
different folds, so one long stay can span two folds. The rows are one per
event or per bin, their times strictly increasing; a caller with several rows
per event (one per candidate arm) makes the folds on the events and gives each
event's rows its fold. The folds are scikit-learn's
``StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=False)`` with one sample per
row, its class the segment's class and its group the row's piece
(:func:`trial_folds`): every row of a piece is in one fold, each class's rows
are spread over the folds as evenly as whole pieces allow, and the folds do
not depend on a seed.

Folds compare designs and check over-fitting. They do not choose penalties
and are not the test that an effect exists.

``analysis/event_evidence/tetrode_holdout.py``'s ``epoch_folds`` calls it: it
gives each epoch's decode bins their folds (:func:`trial_folds_and_groups`) and
records the counts of segments and groups, for the held-out tetrode score and
the non-local restriction run. ``scripts/checks/check_stored_folds_reproduce.py``
checks the stored runs against this rule: on 2026-11-21, against the
restriction run of commit 3953304c and the held-out tetrode run of commit
be732dee, the folds and their records agreed in 242 of 242 epochs.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from continuum_swr_replay.intervals import sample_step

from .padding import group_sizes

__all__ = [
    "AFTER_LAST_TRIAL",
    "BEFORE_FIRST_TRIAL",
    "MAX_SEGMENT_S",
    "MIN_TRIAL_BIN_SHARE",
    "N_FOLDS",
    "capped_groups",
    "task_segment_starts",
    "trial_bin_share",
    "trial_folds",
    "trial_folds_and_groups",
    "trial_segments",
]

N_FOLDS = 5
"""Folds per epoch."""

MAX_SEGMENT_S = 30.0
"""The longest piece of a segment kept whole, s: a longer segment is cut into
equal pieces, each its own group, so a long stay is spread over the folds."""

BEFORE_FIRST_TRIAL = -1
"""The class of the segment before an epoch's first trial."""

AFTER_LAST_TRIAL = -2
"""The class of the segment after an epoch's last trial."""

MIN_TRIAL_BIN_SHARE = 0.8
"""The smallest share of an epoch's rows that its trials must cover
(:func:`task_segment_starts`). Over the 242 analyzed epochs the trials cover
93 % to 99.97 % of the decode's bins (median 98.7 %; the time before the first
trial is at most 65 s and after the last at most 120 s), so a smaller share
means the trials are not on the rows' clock, where every row would fall outside
them and the cap would cut the epoch into pieces of one class."""


def trial_segments(time: NDArray, segment_starts: NDArray) -> NDArray[np.int64]:
    """Each row's segment: segment i runs from ``segment_starts[i]`` up to the
    next start. Rows before the first start belong to segment 0, rows after the
    last start to the last segment.

    Parameters
    ----------
    time : ndarray, shape (n_rows,)
        Each row's time, s (a bin's time, or an event's start).
    segment_starts : ndarray, shape (n_segments,)
        Each segment's first time, s, strictly increasing.

    Returns
    -------
    ndarray of int, shape (n_rows,)
        0 .. n_segments - 1.

    Raises
    ------
    ValueError
        When there is no start, or the starts are not finite and strictly
        increasing.
    """
    starts = np.asarray(segment_starts, dtype=float)
    if starts.size == 0:
        raise ValueError("no segment starts")
    if not np.isfinite(starts).all() or (np.diff(starts) <= 0).any():
        raise ValueError("segment starts must be finite and strictly increasing")
    segment = np.searchsorted(starts, np.asarray(time, dtype=float), side="right") - 1
    return np.maximum(segment, 0).astype(np.int64)


def trial_bin_share(time: NDArray, departures: NDArray, last_trial_end: float) -> float:
    """The share of the rows from the first departure to the last trial's end.

    Parameters
    ----------
    time : ndarray, shape (n_rows,)
        Each row's time, s.
    departures : ndarray, shape (n_trials,)
        s, increasing.
    last_trial_end : float
        s.

    Returns
    -------
    float
    """
    time = np.asarray(time, dtype=float)
    first = float(np.asarray(departures, dtype=float)[0])
    return float(np.mean((time >= first) & (time < last_trial_end)))


def task_segment_starts(
    time: NDArray,
    departures: NDArray,
    destinations: NDArray,
    last_trial_end: float,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """The segments of an epoch of the bandit task and their classes.

    One segment per trial, from its departure from a well (the final poke-out
    at its origin) to the next departure, so the run to a well and the whole
    stay there are one segment, its class the destination well. The time
    before the first departure is a segment of class :data:`BEFORE_FIRST_TRIAL`
    and the time from the last trial's end (its final poke-out at its
    destination) a segment of class :data:`AFTER_LAST_TRIAL`, when the epoch
    has rows there.

    Parameters
    ----------
    time : ndarray, shape (n_rows,)
        Each row's time, s, increasing.
    departures : ndarray, shape (n_trials,)
        Each trial's departure, s, strictly increasing.
    destinations : ndarray of int, shape (n_trials,)
        Each trial's destination well.
    last_trial_end : float
        The last trial's end, s, after its departure.

    Returns
    -------
    starts : ndarray, shape (n_segments,)
        Each segment's first time, s (:func:`trial_segments`).
    classes : ndarray of int, shape (n_segments,)

    Raises
    ------
    ValueError
        When there is no trial, a destination is missing, the departures and
        destinations differ in length, the last trial ends before it departs, a
        departure lies outside the rows' times, or the trials cover less than
        :data:`MIN_TRIAL_BIN_SHARE` of the rows (:func:`trial_bin_share`): the
        trials are then not on the rows' clock.
    """
    time = np.asarray(time, dtype=float)
    departures = np.asarray(departures, dtype=float)
    destinations = np.asarray(destinations)
    if departures.size == 0:
        raise ValueError("no trials")
    if destinations.shape != departures.shape:
        raise ValueError(
            f"{destinations.size} destinations for {departures.size} departures"
        )
    if destinations.dtype.kind == "f" and not np.isfinite(destinations).all():
        raise ValueError("every trial needs a destination well")
    if not last_trial_end > departures[-1]:
        raise ValueError("the last trial ends before its departure")
    if departures[0] < time[0] or departures[-1] > time[-1]:
        raise ValueError(
            f"departures from {departures[0]} to {departures[-1]} s lie outside "
            f"the time axis ({time[0]} to {time[-1]} s): are the trials on the "
            "rows' clock?"
        )
    share = trial_bin_share(time, departures, last_trial_end)
    if share < MIN_TRIAL_BIN_SHARE:
        raise ValueError(
            f"the trials cover {share:.1%} of the bins, less than "
            f"{MIN_TRIAL_BIN_SHARE:.0%}: are the trials on the rows' clock?"
        )
    starts = [departures]
    classes = [destinations.astype(np.int64)]
    if departures[0] > time[0]:
        starts.insert(0, time[:1])
        classes.insert(0, np.array([BEFORE_FIRST_TRIAL]))
    if last_trial_end <= time[-1]:
        starts.append(np.array([last_trial_end]))
        classes.append(np.array([AFTER_LAST_TRIAL]))
    return np.concatenate(starts), np.concatenate(classes)


def _check_one_row_per_time(time: NDArray) -> None:
    """Refuse row times that are not strictly increasing."""
    if (np.diff(np.asarray(time, dtype=float)) <= 0).any():
        raise ValueError(
            "row times must be strictly increasing: give one row per event or per "
            "bin, and map each event's fold to its arm rows"
        )


def capped_groups(
    time: NDArray, segment: NDArray, max_segment_s: float | None = MAX_SEGMENT_S
) -> NDArray[np.int64]:
    """Each row's group: its segment, cut into equal pieces where the segment
    is longer than ``max_segment_s``.

    The cap is a number of rows, ``m``: ``max_segment_s`` over the median
    spacing of ``time``, rounded down. A segment of ``n`` rows longer than
    ``m`` is cut into ``ceil(n / m)`` pieces of consecutive rows whose lengths
    differ by at most one row, each at most ``m`` rows; a segment of at most
    ``m`` rows is one group. A piece of evenly spaced bins spans at most
    ``max_segment_s``; a piece of events (not evenly spaced) can span more or
    less time.

    Parameters
    ----------
    time : ndarray, shape (n_rows,)
        Each row's time, s, strictly increasing: one row per event or per bin.
    segment : ndarray of int, shape (n_rows,)
        Each row's segment (:func:`trial_segments`), nondecreasing.
    max_segment_s : float or None
        The cap, s; None leaves every segment whole.

    Returns
    -------
    ndarray of int, shape (n_rows,)
        0 .. n_groups - 1, nondecreasing.

    Raises
    ------
    ValueError
        When the times are not strictly increasing, or the segments are not
        runs of consecutive rows.
    """
    _check_one_row_per_time(time)
    segment = np.asarray(segment, dtype=np.int64)
    if (np.diff(segment) < 0).any():
        raise ValueError("segments must be runs of consecutive bins")
    first = np.r_[0, np.flatnonzero(np.diff(segment)) + 1]
    lengths = group_sizes(first, segment.size)
    if max_segment_s is None:
        pieces = np.ones(lengths.size, dtype=np.int64)
    else:
        bin_s = sample_step(np.asarray(time, dtype=float))
        cap_bins = max(int(np.floor(max_segment_s / bin_s + 1e-6)), 1)
        pieces = -(-lengths // cap_bins)  # ceil
    run = np.repeat(np.arange(lengths.size), lengths)
    rank = np.arange(segment.size) - first[run]
    piece = rank * pieces[run] // lengths[run]
    offset = np.r_[0, np.cumsum(pieces)[:-1]]
    return (offset[run] + piece).astype(np.int64)


def trial_folds(
    time: NDArray,
    segment_starts: NDArray,
    segment_classes: NDArray,
    n_folds: int = N_FOLDS,
    max_segment_s: float | None = MAX_SEGMENT_S,
) -> NDArray[np.int64]:
    """Each row's fold: whole groups of rows, the folds balanced by class.

    scikit-learn's ``StratifiedGroupKFold(n_splits=n_folds, shuffle=False)``
    with one sample per row, its class the class of the row's segment
    (:func:`trial_segments`) and its group the row's piece of that segment
    (:func:`capped_groups`: a segment longer than ``max_segment_s`` is cut into
    equal pieces, so a long stay is spread over the folds). Every row of a
    group is in one fold, and each class's rows are spread over the folds as
    evenly as whole groups allow. Deterministic: without shuffling the splitter
    places the groups in a fixed order (the most rows first).

    With the segments of :func:`task_segment_starts` every well with at least
    ``n_folds`` groups is sampled in every fold, where folds of contiguous time
    would leave a patch the animal visits in one stretch of the epoch to a
    single fold. A well with fewer groups is not.

    Parameters
    ----------
    time : ndarray, shape (n_rows,)
        Each row's time, s, strictly increasing: one row per bin, or per event
        at its start. The rows of one event (one per candidate arm) take the
        event's fold, which the caller maps to them.
    segment_starts : ndarray, shape (n_segments,)
        Each segment's first time, s, strictly increasing.
    segment_classes : ndarray of int, shape (n_segments,)
        Each segment's class.
    n_folds : int
        The number of folds.
    max_segment_s : float or None
        The longest group, s, as a number of rows at the median row spacing
        (:func:`capped_groups`); None leaves every segment whole.

    Returns
    -------
    ndarray of int, shape (n_rows,)
        0 .. n_folds - 1; every fold has rows.

    Raises
    ------
    ValueError
        When the times are not strictly increasing, when the classes are not
        one per start or not all finite, when fewer groups than ``n_folds``
        hold a row, or when a fold is left empty.
    """
    return trial_folds_and_groups(
        time, segment_starts, segment_classes, n_folds, max_segment_s
    )[0]


def trial_folds_and_groups(
    time: NDArray,
    segment_starts: NDArray,
    segment_classes: NDArray,
    n_folds: int = N_FOLDS,
    max_segment_s: float | None = MAX_SEGMENT_S,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """:func:`trial_folds`' folds, with each row's segment and group.

    The one computation of the folds: the segments (:func:`trial_segments`)
    and groups (:func:`capped_groups`) returned are those the folds were made
    from, so a caller that records their counts does not compute them again.

    Parameters
    ----------
    time, segment_starts, segment_classes, n_folds, max_segment_s
        As :func:`trial_folds`.

    Returns
    -------
    folds, segment, group : ndarray of int, each shape (n_rows,)

    Raises
    ------
    ValueError
        As :func:`trial_folds`.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    _check_one_row_per_time(time)
    classes = np.asarray(segment_classes)
    if classes.shape != np.shape(segment_starts):
        raise ValueError(
            f"{classes.size} segment classes for {np.size(segment_starts)} starts"
        )
    if classes.dtype.kind == "f" and not np.isfinite(classes).all():
        raise ValueError("every segment needs a class (a destination well)")
    segment = trial_segments(time, segment_starts)
    group = capped_groups(time, segment, max_segment_s)
    n_groups = int(group[-1]) + 1
    if n_groups < n_folds:
        raise ValueError(
            f"{n_groups} segments (trials) hold bins, fewer than the {n_folds} folds"
        )
    folds = np.full(segment.size, -1, dtype=np.int64)
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=False)
    for fold, (_, scored) in enumerate(
        splitter.split(np.zeros(segment.size), classes[segment], group)
    ):
        folds[scored] = fold
    empty = np.setdiff1d(np.arange(n_folds), folds)
    if empty.size:
        raise ValueError(f"folds {empty.tolist()} hold no segment")
    return folds, segment, group
