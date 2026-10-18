"""Matched baseline windows for the common spike-evidence score.

The baselines are a reference population for the evidence score, not known
negatives. They come from the **pool**: the time every detector of the
comparison was exposed (observed, outside excluded time, finite speed below
threshold), less every event of every inventory, compared, moving and set
aside alike. Nothing about the decode's posterior enters the pool.

A candidate window for an event is a run of exactly the event's number of
grid rows, every row strictly inside one pool interval. Strictly, so that a
window never shares a row with an event whose bound is a pool bound; a pool
interval never spans a timestamp gap, since the still time it comes from does
not. A window is matched at one of three levels:

- ``location``: the animal's bin at the window's middle row lies within
  :data:`LOCATION_TOLERANCE_CM` of track-graph distance from its bin at the
  event's middle row, when at least :data:`MIN_MATCHED` such windows exist;
- ``duration_only``: any window of the epoch with the event's length, when
  the location level has too few and at least :data:`MIN_MATCHED` exist;
- ``failed``: neither; the event has no baseline.

The middle row of ``n`` rows from row ``k`` is ``k + n // 2``. Up to
:data:`N_BASELINES` windows are drawn without replacement from the level's
candidates by ``keyed_rng(seed, event_key)``, so an event's draws depend only
on the seed, its key, its length and location, and the pool.
"""

from collections.abc import Collection, Mapping
from functools import reduce
from types import MappingProxyType

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from continuum_swr_replay.analysis.hippocampus.state_evidence.windows import (
    Seed,
    keyed_rng,
)
from continuum_swr_replay.analysis.inventory_comparison.evidence import (
    interval_evidence,
)
from continuum_swr_replay.analysis.inventory_comparison.features import (
    RowTraces,
    _grid_rows,
    _over_rows,
)
from continuum_swr_replay.analysis.inventory_comparison.intervals import (
    intersect,
    normalize,
    subtract,
)
from continuum_swr_replay.analysis.inventory_comparison.overlap import (
    Coverage,
    roster,
    validate_detectors,
)
from continuum_swr_replay.session_keys import SESSION_COLUMNS_DAY

__all__ = [
    "BASELINE_DTYPES",
    "BASELINE_SEED",
    "DURATION_ONLY",
    "FAILED",
    "LOCATION",
    "LOCATION_TOLERANCE_CM",
    "MIN_MATCHED",
    "N_BASELINES",
    "WINDOW_DTYPES",
    "baseline_draws",
    "baseline_pool",
    "matched_baselines",
    "window_room",
]

N_BASELINES = 100
"""Baseline windows drawn per event, at most."""
LOCATION_TOLERANCE_CM = 10.0
"""Largest track-graph distance between the animal's bins at the event's and
the window's middle rows for a location match."""
MIN_MATCHED = 20
"""Fewest candidate windows a level needs to be used."""
BASELINE_SEED = 20261018
"""The base seed of the baseline draws."""
LOCATION, DURATION_ONLY, FAILED = "location", "duration_only", "failed"
"""The matching levels, best first."""

BASELINE_DTYPES: Mapping[str, object] = MappingProxyType(
    {
        "baseline_level": object,
        "baseline_n": np.int64,
        "baseline_n_scored": np.int64,
        "baseline_median_dll": np.float64,
        "baseline_q95_dll": np.float64,
        "evidence_rank": np.float64,
    }
)
"""Per event: its level, the windows drawn and those with evidence, the
median and 95th percentile of their scores, and the share of them scoring
below the event. The statistics are NaN with no scored window; the rank is
NaN too when the event has no evidence."""

WINDOW_DTYPES: Mapping[str, object] = MappingProxyType(
    {
        "matched_event_key": object,
        "event_start_s": np.float64,
        "event_end_s": np.float64,
    }
)
"""The first draw of each event that has one: its event and its bounds, the
times of its first and last rows."""


def baseline_pool(
    coverage: Mapping[tuple, Coverage],
    events: pd.DataFrame,
    *,
    detectors: Collection[str] | None = None,
) -> NDArray:
    """One epoch's shared exposure less every event of every inventory.

    Parameters
    ----------
    coverage : mapping
        ``(animal, day, epoch, detector) -> Coverage`` of one epoch, as
        ``stored_coverage`` rebuilds it. A detector of the comparison with no
        entry was not exposed.
    events : pd.DataFrame
        ``animal``, ``day``, ``epoch``, ``event_start_s`` and ``event_end_s``
        of every event of every inventory in the epoch: the stored
        ``inventory`` (compared and moving events) and the stored
        ``set_aside`` spans (``interval_start_s``, ``interval_end_s``
        renamed).
    detectors : collection of str, optional
        The comparison's detectors; by default every detector in
        ``coverage``, the ``all`` comparison.

    Returns
    -------
    pool : NDArray, shape (n_intervals, 2)
        Normalized.

    Raises
    ------
    ValueError
        ``coverage`` holds more than one epoch, ``events`` hold another
        epoch's, or ``detectors`` names one no coverage entry knows.
    """
    sessions = {tuple(key[:3]) for key in coverage}
    if len(sessions) != 1:
        raise ValueError(
            f"The pool is one epoch's; the coverage holds {len(sessions)}."
        )
    (session,) = sessions
    same = np.logical_and.reduce(
        [
            events[column].to_numpy() == value
            for column, value in zip(SESSION_COLUMNS_DAY, session, strict=True)
        ]
    )
    if not np.all(same):
        raise ValueError(f"An event belongs to another epoch than {session}.")
    names = (
        roster(coverage)
        if detectors is None
        else validate_detectors(detectors, coverage, "detectors")
    )
    empty = np.empty((0, 2))
    shared = reduce(
        intersect,
        (
            coverage[(*session, name)].exposure
            if (*session, name) in coverage
            else empty
            for name in names
        ),
    )
    spans = events[["event_start_s", "event_end_s"]].to_numpy(dtype=np.float64)
    return subtract(shared, spans)


def window_room(time: NDArray, pool: NDArray) -> NDArray[np.int64]:
    """How many rows from each row on lie strictly inside one pool interval.

    A window of ``n`` rows from row ``k`` is a candidate exactly when
    ``room[k] >= n``. The grid's last sample is never inside.

    Parameters
    ----------
    time : NDArray, shape (n_time,)
        The grid's sample times in seconds, increasing.
    pool : NDArray, shape (n_intervals, 2)

    Returns
    -------
    room : NDArray[np.int64], shape (n_time,)
        0 at a row outside the pool.
    """
    time = np.asarray(time, dtype=np.float64)
    pool = normalize(pool)
    room = np.zeros(time.shape, dtype=np.int64)
    if pool.size == 0 or time.size == 0:
        return room
    index = np.searchsorted(pool[:, 0], time, side="right") - 1
    safe = np.clip(index, 0, None)
    inside = (index >= 0) & (time > pool[safe, 0]) & (time < pool[safe, 1])
    inside[-1] = False
    # a run of inside rows ends at an outside row or a new pool interval
    joined = inside[:-1] & inside[1:] & (index[:-1] == index[1:])
    begins = inside & np.r_[True, ~joined]
    stops = np.flatnonzero(inside & np.r_[~joined, True]) + 1
    run = np.cumsum(begins) - 1
    rows = np.flatnonzero(inside)
    room[rows] = stops[run[rows]] - rows
    return room


def baseline_draws(
    pool: NDArray,
    events: pd.DataFrame,
    traces: RowTraces,
    animal_bin: NDArray,
    bin_distance_cm: NDArray,
    *,
    seed: Seed,
) -> tuple[NDArray[np.object_], NDArray[np.int64], NDArray[np.int64]]:
    """Each event's matching level and the first rows of its drawn windows.

    Parameters
    ----------
    pool : NDArray, shape (n_intervals, 2)
        From :func:`baseline_pool`.
    events : pd.DataFrame
        The compared events: ``event_key``, ``event_start_s``,
        ``event_end_s``.
    traces : RowTraces
        The epoch's grid.
    animal_bin : NDArray of int, shape (n_time,)
        The animal's real place bin at each row (``evidence.animal_bins``),
        -1 where unknown.
    bin_distance_cm : NDArray, shape (n_bins, n_bins)
        Track-graph distance between real bins
        (``interior_bin_distances_cm``).
    seed : int or sequence
        Names the streams with each ``event_key``; the runner passes
        :data:`BASELINE_SEED`.

    Returns
    -------
    level : NDArray of str, shape (n_events,)
    starts : NDArray[np.int64], shape (n_events, N_BASELINES)
        Each drawn window's first row, in draw order; -1 past the last draw.
    n_rows : NDArray[np.int64], shape (n_events,)
        Each event's rows, and so each of its windows'.

    Raises
    ------
    ValueError
        ``animal_bin`` is not on the grid, or an event reaches the grid's last
        sample.
    """
    time = traces.time
    animal_bin = np.asarray(animal_bin, dtype=np.int64)
    if animal_bin.shape != time.shape:
        raise ValueError(
            f"animal_bin has shape {animal_bin.shape}; the grid {time.shape}."
        )
    first, n_rows = _grid_rows(time, events)
    room = window_room(time, pool)
    has_rows = n_rows > 0
    event_bin = np.full(n_rows.shape, -1, dtype=np.int64)
    event_bin[has_rows] = animal_bin[(first + n_rows // 2)[has_rows]]
    keys = events["event_key"].to_numpy()
    level = np.full(n_rows.shape, FAILED, dtype=object)
    starts = np.full((n_rows.size, N_BASELINES), -1, dtype=np.int64)
    for length in np.unique(n_rows[has_rows]):
        candidates = np.flatnonzero(room >= length)
        window_bin = animal_bin[candidates + length // 2]
        known = window_bin >= 0
        of_length = np.flatnonzero(n_rows == length)
        for at in np.unique(event_bin[of_length]):
            near = candidates[:0]
            if at >= 0:
                distance = bin_distance_cm[at, np.where(known, window_bin, 0)]
                near = candidates[known & (distance <= LOCATION_TOLERANCE_CM)]
            if near.size >= MIN_MATCHED:
                name, choices = LOCATION, near
            elif candidates.size >= MIN_MATCHED:
                name, choices = DURATION_ONLY, candidates
            else:
                continue
            n_draw = min(N_BASELINES, choices.size)
            for i in of_length[event_bin[of_length] == at]:
                stream = keyed_rng(seed, str(keys[i]))
                starts[i, :n_draw] = stream.choice(choices, n_draw, replace=False)
                level[i] = name
    return level, starts, n_rows


def matched_baselines(
    pool: NDArray,
    events: pd.DataFrame,
    traces: RowTraces,
    evidence: NDArray,
    animal_bin: NDArray,
    bin_distance_cm: NDArray,
    *,
    seed: Seed,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Each compared event's baseline scores, and its first baseline window.

    A window's score is the evidence over its rows by the rule of
    ``interval_evidence``: the mean over the rows with evidence, NaN when
    none has. Windows with no score enter no statistic.

    Parameters
    ----------
    pool, events, traces, animal_bin, bin_distance_cm, seed
        As :func:`baseline_draws`.
    evidence : NDArray, shape (n_time,)
        ``epoch_row_evidence``, computed on every row of the events and of
        the pool.

    Returns
    -------
    summary : pd.DataFrame
        On ``events``' index, columns and dtypes as :data:`BASELINE_DTYPES`.
    windows : pd.DataFrame
        One row per event that has a baseline, in event order, columns and
        dtypes as :data:`WINDOW_DTYPES`.

    Raises
    ------
    ValueError
        ``evidence`` or ``animal_bin`` is not on the grid, or an event reaches
        the grid's last sample.
    """
    evidence = np.asarray(evidence, dtype=np.float64)
    if evidence.shape != traces.time.shape:
        raise ValueError(
            f"evidence has shape {evidence.shape}; the grid {traces.time.shape}."
        )
    level, starts, n_rows = baseline_draws(
        pool, events, traces, animal_bin, bin_distance_cm, seed=seed
    )
    drawn = starts >= 0
    event_of, _ = np.nonzero(drawn)
    valid = np.isfinite(evidence)
    # by first row, so each reduceat step from one window to the next is short
    order = np.argsort(starts[drawn], kind="stable")
    window_first, window_n = starts[drawn][order], n_rows[event_of][order]
    n_valid = _over_rows(np.add, valid.astype(np.int64), window_first, window_n)
    total = _over_rows(np.add, np.where(valid, evidence, 0.0), window_first, window_n)
    by_first_row = np.full(order.shape, np.nan)
    with_evidence = n_valid > 0
    by_first_row[with_evidence] = total[with_evidence] / n_valid[with_evidence]
    window_score = np.empty_like(by_first_row)
    window_score[order] = by_first_row
    score = np.full(starts.shape, np.nan)
    score[drawn] = window_score

    n_scored = np.isfinite(score).sum(axis=1)
    scored = n_scored > 0
    median = np.full(n_rows.shape, np.nan)
    q95 = np.full(n_rows.shape, np.nan)
    median[scored] = np.nanmedian(score[scored], axis=1)
    q95[scored] = np.nanquantile(score[scored], 0.95, axis=1)
    own = interval_evidence(traces, evidence, events)["evidence_dll"].to_numpy()
    ranked = scored & np.isfinite(own)
    rank = np.full(n_rows.shape, np.nan)
    rank[ranked] = (score[ranked] < own[ranked, None]).sum(axis=1) / n_scored[ranked]
    summary = pd.DataFrame(
        {
            "baseline_level": level,
            "baseline_n": drawn.sum(axis=1),
            "baseline_n_scored": n_scored,
            "baseline_median_dll": median,
            "baseline_q95_dll": q95,
            "evidence_rank": rank,
        },
        index=events.index,
    ).astype(dict(BASELINE_DTYPES))

    has_window = drawn[:, 0]
    first_row = starts[has_window, 0]
    windows = pd.DataFrame(
        {
            "matched_event_key": events["event_key"].to_numpy()[has_window],
            "event_start_s": traces.time[first_row],
            "event_end_s": traces.time[first_row + n_rows[has_window] - 1],
        }
    ).astype(dict(WINDOW_DTYPES))
    return summary, windows
