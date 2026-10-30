"""Matched baseline windows for the common spike-evidence score, and the
memoryless rejection stages.

The event-features runner does not use the baseline windows (``baseline_pool``,
``window_room``, ``baseline_draws``, ``matched_baselines``): the lead dropped
the baseline comparison from that stage (2026-10-18), because windows matched
on duration and location but not on spike count measured engagement more than
evidence. They are kept, with their tests, for a later design matched on spike
count. The rejection stages are used.

The baselines are a reference population for the evidence score, not known
negatives. They come from the **pool**: the time every detector of the
comparison was exposed (observed, outside excluded time, finite speed below
threshold), less every event of every inventory, compared, moving and set
aside alike. Nothing about the decode's posterior enters the pool.

A candidate window for an event is a run of exactly the event's number of
grid rows, every row strictly inside one pool interval. Strictly, so that a
window never shares a row with an event whose bound is a pool bound (every
event whose bound is a grid sample leaves that sample as a pool edge); the
cost is one row at each pool edge, which no window uses. A pool interval
never spans a timestamp gap, since the still time it comes from does not;
a drawn window that spans one is refused. A window is matched at one of three
levels:

- ``location``: the animal's bin at the window's middle row lies within
  :data:`LOCATION_TOLERANCE_CM` of track-graph distance from its bin at the
  event's middle row, when at least :data:`MIN_MATCHED` such windows exist;
- ``duration_only``: any window of the epoch with the event's length, when
  the location level has too few and at least :data:`MIN_MATCHED` exist;
- ``failed``: neither; the event has no baseline.

The middle row of ``n`` rows from row ``k`` is ``k + (n - 1) // 2``, the
earlier of two, as for the event's own location features
(``features.middle_row``). Up to
:data:`N_BASELINES` windows are drawn without replacement from the level's
candidates by ``keyed_rng(seed, animal, day, epoch, start, end)``: the epoch
and the interval's bounds as stored, each written exactly as
``float.hex``. The stream names the interval, not the detector, so identical
intervals of different inventories (the memoryless twins, an SWR event equal
to another detector's) draw identical windows and get identical baselines, as
they get identical features. An event's draws depend only on the seed, its
epoch and bounds, and the pool, never on the other events. Stream names are
hashed (CRC-32), so two intervals could share a stream; that would only
correlate their draws, never change a level or a candidate set.

The rejection stages (:func:`rejection_stages`) say, for each state-space
event, how far the memoryless detector got on that activity before it did not
select it, against each memoryless cohort: ``memoryless`` (the published
cohort, after the shuffle) and ``memoryless_criteria`` (duration and
dispersion alone). Its windows are those whose midpoint lies in the event,
both bounds included, as for the ``ms_`` features; its candidates are those
sharing time of positive duration with it, as the comparison's overlaps are.
Every gate's failures are kept as their own count or flag, and one exclusive
``stage`` gives the furthest gate reached, in the detector's order. The
memoryless detector is a documented adaptation whose clusterless null was
not separately validated; the lead dropped the Phase 2 pilot checks
(2026-10-28).
"""

from collections.abc import Collection, Mapping
from functools import reduce
from types import MappingProxyType

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.detect import (
    CandidateSequences,
    # the detector's own threshold comparison, so a tie is decided as it decides
    _is_above,
)
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
    interval_rows,
    middle_row,
)
from continuum_swr_replay.analysis.inventory_comparison.overlap import (
    MATCHED,
    UNKNOWN,
    UNMATCHED,
    Coverage,
    roster,
    validate_detectors,
)
from continuum_swr_replay.analysis.inventory_comparison.populations import ALL
from continuum_swr_replay.analysis.inventory_comparison.schema import (
    MEMORYLESS,
    MEMORYLESS_COHORTS,
    _event_keys,
)
from continuum_swr_replay.intervals import (
    intersect,
    normalize,
    subtract,
)
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS_DAY

__all__ = [
    "BASELINE_DTYPES",
    "BASELINE_SEED",
    "DURATION_ONLY",
    "FAILED",
    "LOCATION",
    "LOCATION_TOLERANCE_CM",
    "MIN_MATCHED",
    "N_BASELINES",
    "REJECTION_COHORTS",
    "REJECTION_DTYPES",
    "REJECTION_STAGES",
    "WINDOW_DTYPES",
    "baseline_draws",
    "baseline_pool",
    "matched_baselines",
    "rejection_stages",
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

REJECTION_COHORTS: Mapping[str, str] = MappingProxyType(
    {MEMORYLESS: "published", "memoryless_criteria": "replay_criteria"}
)
"""Each memoryless inventory of the comparison, by its detector name, and the
cohort of ``schema.MEMORYLESS_COHORTS`` it was built from."""
REJECTION_STAGES = (
    "no_window",
    "no_kept_window",
    "no_run",
    "no_candidate",
    "duration",
    "dispersion",
    "speed",
    "shuffle_failed",
    "shuffle_untestable",
    "in_cohort",
)
"""The exclusive stages, in the detector's order: no window's midpoint lies in
the event; no window passed every per-window gate; no kept window was in a run
of two or more; no candidate shares time with it; then the furthest any of
its candidates reached: declined on duration, on dispersion, on whole-event
speed, (published cohort only) on the shuffle, failed or with an observed
statistic that is not finite, and last selected by the cohort."""
_WINDOW_COUNTS = (
    "n_windows",
    "n_undecoded",
    "n_posterior_unusable",
    "n_speed_ineligible",
    "n_spread_failed",
    "n_jump_failed",
    "n_kept",
    "n_kept_in_runs",
)
_CANDIDATE_FLAGS = (
    "any_duration_failed",
    "any_dispersion_failed",
    "any_replay",
    "any_replay_not_speed_eligible",
    "any_shuffle_failed",
    "any_shuffle_untestable",
    "any_in_cohort",
)
REJECTION_DTYPES: Mapping[str, object] = MappingProxyType(
    {
        "animal": object,
        "day": np.int64,
        "epoch": object,
        "detector": object,
        "event_key": object,
        "cohort": object,
        "membership": object,
        "stage": object,
        "unknown_reason": object,
        **dict.fromkeys(_WINDOW_COUNTS, pd.Int64Dtype()),
        "n_candidates": pd.Int64Dtype(),
        **dict.fromkeys(_CANDIDATE_FLAGS, pd.BooleanDtype()),
    }
)
"""One row per event and cohort. ``stage`` is one of :data:`REJECTION_STAGES`,
or ``unknown`` where the membership is, with the population's
``unknown_reason`` (empty otherwise); there every count and flag is missing,
since an unknown comparison is not a rejection. Over the event's windows,
undecoded ones apart (``n_undecoded``) and the rest over the decoded ones:
each gate's failures (``n_posterior_unusable``, ``n_speed_ineligible`` at the
window's midpoint, ``n_spread_failed``, ``n_jump_failed``; a window with no
center fails spread and jump too), the windows passing all of them
(``n_kept``) and those in a run of two or more (``n_kept_in_runs``). Over its
candidates: ``n_candidates`` and whether any was declined on duration or on
dispersion, met both (``any_replay``) and was not still over the whole event,
met both and failed the shuffle (unshuffled included), met both, was shuffled
and has an observed statistic that is not finite, or is one of the cohort's
compared events."""


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
        ``stored_coverage`` rebuilds it, every detector of the comparison
        with an entry; a detector that did not run has one with empty
        coverage, so it leaves no pool.
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
        epoch's, ``detectors`` names one no coverage entry knows, or an
        event has a missing or inverted bound.
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
    # one epoch, so every name of the roster has an entry there
    shared = reduce(intersect, (coverage[(*session, name)].exposure for name in names))
    spans = events[["event_start_s", "event_end_s"]].to_numpy(dtype=np.float64)
    return subtract(shared, normalize(spans, strict=True))


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
        The compared events of one epoch: ``animal``, ``day``, ``epoch``,
        ``event_start_s`` and ``event_end_s``.
    traces : RowTraces
        The epoch's grid.
    animal_bin : NDArray of int, shape (n_time,)
        The animal's real place bin at each row (``evidence.animal_bins``),
        -1 where unknown.
    bin_distance_cm : NDArray, shape (n_bins, n_bins)
        Track-graph distance between real bins
        (``interior_bin_distances_cm``).
    seed : int or sequence
        Names the streams with each event's epoch and bounds; the runner
        passes :data:`BASELINE_SEED`.

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
        ``animal_bin`` is not on the grid, or a drawn window spans a
        timestamp gap (a step over 1.5 times the grid's median step).

    Notes
    -----
    An event's rows are ``features``' (the grid's last sample is never one),
    so an event ending on it is matched on one row fewer, and one holding only
    it has no row and fails.
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
    event_bin[has_rows] = animal_bin[middle_row(first, n_rows)[has_rows]]
    names = list(
        zip(
            *(events[column].tolist() for column in SESSION_COLUMNS_DAY),
            (float(t).hex() for t in events["event_start_s"]),
            (float(t).hex() for t in events["event_end_s"]),
            strict=True,
        )
    )
    level = np.full(n_rows.shape, FAILED, dtype=object)
    starts = np.full((n_rows.size, N_BASELINES), -1, dtype=np.int64)
    for length in np.unique(n_rows[has_rows]):
        candidates = np.flatnonzero(room >= length)
        window_bin = animal_bin[middle_row(candidates, length)]
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
                stream = keyed_rng(seed, *names[i])
                starts[i, :n_draw] = stream.choice(choices, n_draw, replace=False)
                level[i] = name
    _refuse_gaps(time, starts, n_rows)
    return level, starts, n_rows


def _refuse_gaps(time: NDArray, starts: NDArray, n_rows: NDArray) -> None:
    """Raise when a drawn window holds a step over 1.5 median grid steps."""
    if time.size < 2:
        return
    steps = np.diff(time)
    gap = steps > 1.5 * np.median(steps)
    # gaps_before[k]: gaps among the steps between rows 0 and k
    gaps_before = np.r_[0, np.cumsum(gap)]
    drawn = starts >= 0
    first = starts[drawn]
    last = first + np.broadcast_to(n_rows[:, None], starts.shape)[drawn] - 1
    if (gaps_before[last] > gaps_before[first]).any():
        raise ValueError(
            "A baseline window spans a timestamp gap; the pool was not built "
            "from still time broken at gaps."
        )


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
    pool, traces, animal_bin, bin_distance_cm, seed
        As :func:`baseline_draws`.
    events : pd.DataFrame
        As :func:`baseline_draws`, with ``event_key`` too, which names each
        event's window.
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
        ``evidence`` or ``animal_bin`` is not on the grid, or a drawn window
        spans a timestamp gap.
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


def _memberships(
    events: pd.DataFrame, populations: pd.DataFrame, cohort: str
) -> tuple[NDArray[np.object_], NDArray[np.object_]]:
    """Each event's membership against ``cohort`` and its unknown reason."""
    # the labels as population_rows writes them, "<membership>_by:<detector>"
    labels = {f"{name}_by:{cohort}": name for name in (MATCHED, UNMATCHED, UNKNOWN)}
    own = populations[
        populations["event_key"].isin(events["event_key"])
        & populations["population"].isin(labels)
    ].set_index("event_key")
    if not own.index.is_unique or len(own) != len(events):
        raise ValueError(
            f"Every event needs exactly one membership against {cohort}; "
            f"{len(own)} rows for {len(events)} events."
        )
    own = own.loc[events["event_key"]]
    return (
        own["population"].map(labels).to_numpy(dtype=object),
        own["unknown_reason"].to_numpy(dtype=object),
    )


def _window_counts(
    candidates: CandidateSequences,
    undecoded: NDArray,
    starts: NDArray,
    ends: NDArray,
) -> dict[str, NDArray[np.int64]]:
    """Each count of :data:`_WINDOW_COUNTS` over the windows of each event."""
    gates = candidates.decisions.gates
    n_windows = candidates.midpoint_s.size
    undecoded = np.asarray(undecoded, dtype=bool)
    if undecoded.shape != (n_windows,):
        raise ValueError(
            f"undecoded has shape {undecoded.shape}; there are {n_windows} windows."
        )
    if (undecoded & gates.posterior_usable).any():
        raise ValueError(
            "An undecoded window has a usable posterior; the mask is not the one "
            "the run was given."
        )
    decoded = ~undecoded
    edges = np.zeros(n_windows + 1, dtype=np.int64)
    runs = candidates.pre_merge_runs
    np.add.at(edges, runs[:, 0], 1)
    np.add.at(edges, runs[:, 1] + 1, -1)
    in_run = np.cumsum(edges[:-1]) > 0
    masks = {
        "n_windows": np.ones(n_windows, dtype=bool),
        "n_undecoded": undecoded,
        "n_posterior_unusable": decoded & ~gates.posterior_usable,
        "n_speed_ineligible": decoded & ~gates.speed_eligible,
        "n_spread_failed": decoded & ~gates.spread_below_threshold,
        "n_jump_failed": decoded & ~gates.jump_below_threshold,
        "n_kept": decoded & gates.kept,
        "n_kept_in_runs": decoded & gates.kept & in_run,
    }
    first, n_in = interval_rows(candidates.midpoint_s, starts, ends)
    counts = {}
    for name, mask in masks.items():
        before = np.concatenate(([0], np.cumsum(mask, dtype=np.int64)))
        counts[name] = before[first + n_in] - before[first]
    return counts


def _candidate_gates(
    candidates: CandidateSequences,
    table: pd.DataFrame,
    cohort_keys: NDArray[np.bool_],
    shuffled: bool,
) -> tuple[dict[str, NDArray[np.bool_]], NDArray[np.bool_], NDArray[np.int64]]:
    """Each candidate's flags, whether it passed every gate outside the cohort,
    and the index in :data:`REJECTION_STAGES` of the furthest gate it reached
    (-1 for one outside the cohort that passed them all)."""
    config = candidates.config
    runs = candidates.runs
    long_enough = _is_above(
        runs[:, 1] - runs[:, 0], config.minimum_duration_s / config.stride_s
    )
    spread_out = _is_above(candidates.dispersion_cm, config.dispersion_threshold_cm)
    replay = candidates.is_replay
    still = table["whole_event_speed_eligible"].to_numpy(dtype=bool)
    passed = table["shuffle_pass"].to_numpy(dtype=bool)
    untestable = (
        replay
        & (table["n_shuffles"].to_numpy() > 0)
        & ~table["shuffle_testable"].to_numpy(dtype=bool)
    )
    shuffle_gate = passed if shuffled else np.ones_like(passed)
    through = replay & still & shuffle_gate
    if (cohort_keys & ~through).any():
        raise ValueError(
            "A compared event of the cohort fails one of its gates in the "
            "candidate table; the table is not the one the comparison read."
        )
    stage = REJECTION_STAGES.index
    rank = np.select(
        [
            ~long_enough,
            ~spread_out,
            ~still,
            ~shuffle_gate & untestable,
            ~shuffle_gate,
            cohort_keys,
        ],
        [
            stage("duration"),
            stage("dispersion"),
            stage("speed"),
            stage("shuffle_untestable"),
            stage("shuffle_failed"),
            stage("in_cohort"),
        ],
        default=-1,
    )
    flags = {
        "any_duration_failed": ~long_enough,
        "any_dispersion_failed": ~spread_out,
        "any_replay": replay,
        "any_replay_not_speed_eligible": replay & ~still,
        "any_shuffle_failed": replay & ~passed,
        "any_shuffle_untestable": untestable,
        "any_in_cohort": cohort_keys,
    }
    return flags, through & ~cohort_keys, rank.astype(np.int64)


def rejection_stages(
    events: pd.DataFrame,
    membership: pd.DataFrame,
    candidates: CandidateSequences | None,
    candidate_table: pd.DataFrame | None,
    undecoded: NDArray | None,
    *,
    cohort: str,
) -> pd.DataFrame:
    """How far the memoryless detector got on each event, against one cohort.

    A cohort's events are the comparison's own: the ``all`` population rows
    of the cohort's detector, which are the candidates the adapter selected
    (``schema.MEMORYLESS_COHORTS``, no ``artifact_drop_reason`` after its
    second pass) that are speed-eligible and not bridging. An event is
    ``matched`` against the cohort when one of those shares time of positive
    duration with it, which is the rule of ``any_in_cohort``; so ``stage`` is
    ``in_cohort`` exactly when the membership is ``matched``, and inputs that
    break that are refused.

    Parameters
    ----------
    events : pd.DataFrame
        The state-space compared events of one epoch: ``animal``, ``day``,
        ``epoch``, ``detector``, ``event_key``, ``event_start_s`` and
        ``event_end_s``.
    membership : pd.DataFrame
        The epoch's population rows (``populations.population_rows`` of the
        pair membership), holding each event's ``<membership>_by:<cohort>``
        row and the cohort's ``all`` rows.
    candidates : CandidateSequences or None
        The epoch's stored memoryless run; None when there is none, and then
        every membership must be ``unknown``.
    candidate_table : pd.DataFrame or None
        The run's ``memoryless_events`` table, row ``i`` candidate ``i``.
    undecoded : NDArray of bool, shape (n_windows,), or None
        The windows the run left undecoded (``windows_holding`` on the mask).
    cohort : str
        A key of :data:`REJECTION_COHORTS`.

    Returns
    -------
    stages : pd.DataFrame
        On ``events``' index, columns and dtypes as :data:`REJECTION_DTYPES`.

    Raises
    ------
    ValueError
        The cohort is unknown; an event has no single membership against it;
        only some of the run's inputs are given; there is no run and an event
        is not ``unknown``; the table is not the run's (other windows or
        verdicts, or a cohort event it lacks or whose gates it fails), or
        ``undecoded`` is not the run's; a known event reached no stage; or
        an event ``matched`` by the cohort overlaps none of its events, one
        ``unmatched`` overlaps one, or one ``unmatched`` overlaps a candidate
        that passed every gate but is not in the cohort (the comparison set
        it aside, so the event cannot be unmatched).
    """
    if cohort not in REJECTION_COHORTS:
        raise ValueError(
            f"cohort must be one of {list(REJECTION_COHORTS)}, got {cohort!r}."
        )
    member, unknown_reason = _memberships(events, membership, cohort)
    known = member != UNKNOWN
    n_events = len(events)
    columns: dict[str, object] = {
        **{c: events[c].to_numpy() for c in ("animal", "day", "epoch", "detector")},
        "event_key": events["event_key"].to_numpy(),
        "cohort": cohort,
        "membership": member,
        "stage": np.full(n_events, UNKNOWN, dtype=object),
        "unknown_reason": unknown_reason,
    }
    reasons: dict[str, NDArray] = {}
    absent = [given is None for given in (candidates, candidate_table, undecoded)]
    if any(absent) and not all(absent):
        raise ValueError(
            "Pass the run's candidates, table and undecoded windows together, or "
            "none of them."
        )
    if candidates is None or candidate_table is None or undecoded is None:
        if known.any():
            raise ValueError(
                f"Without a memoryless run every membership against {cohort} must "
                f"be unknown; {int(known.sum())} are not."
            )
    else:
        starts = events["event_start_s"].to_numpy(dtype=np.float64)
        ends = events["event_end_s"].to_numpy(dtype=np.float64)
        reasons |= _window_counts(candidates, undecoded, starts, ends)

        table = candidate_table
        if not (
            len(table) == len(candidates.runs)
            and np.array_equal(table["first_window"], candidates.runs[:, 0])
            and np.array_equal(table["last_window"], candidates.runs[:, 1])
            and np.array_equal(table["is_replay"], candidates.is_replay)
        ):
            raise ValueError(
                "The candidate table does not describe the run: its windows or "
                "verdicts differ."
            )
        candidate_start = table["event_start_s"].to_numpy(dtype=np.float64)
        candidate_end = table["event_end_s"].to_numpy(dtype=np.float64)
        if (candidate_start[1:] < candidate_end[:-1]).any():
            raise ValueError("The candidates are not disjoint and in time order.")
        keys = np.asarray(
            _event_keys(
                cohort,
                table["animal"],
                table["day"],
                table["epoch"],
                table["ms_event_id"],
            ),
            dtype=object,
        )
        selected = membership.loc[
            (membership["detector"] == cohort) & (membership["population"] == ALL),
            "event_key",
        ]
        if not selected.isin(keys).all():
            raise ValueError(
                f"A compared event of {cohort} is not a candidate of the table."
            )
        flags, set_aside, rank = _candidate_gates(
            candidates,
            table,
            np.isin(keys, selected.to_numpy()),
            shuffled=MEMORYLESS_COHORTS[REJECTION_COHORTS[cohort]] == "shuffle_pass",
        )

        # disjoint candidates in order: those ending after the event's start
        # and starting before its end share positive time with it; an event
        # of no duration shares none with any
        low = np.searchsorted(candidate_end, starts, side="right")
        high = np.maximum(np.searchsorted(candidate_start, ends, side="left"), low)
        high = np.where(ends > starts, high, low)
        reasons["n_candidates"] = high - low

        def any_of(flag: NDArray[np.bool_]) -> NDArray[np.bool_]:
            before = np.concatenate(([0], np.cumsum(flag, dtype=np.int64)))
            return np.asarray(before[high] > before[low])

        reasons |= {name: any_of(flag) for name, flag in flags.items()}
        matched = member == MATCHED
        unmatched = member == UNMATCHED
        overlapping = reasons["any_in_cohort"]
        if (matched & ~overlapping).any() or (unmatched & overlapping).any():
            raise ValueError(
                f"An event matched by {cohort} overlaps none of its events, or one "
                "unmatched overlaps one: the memberships are not over this run."
            )
        if (unmatched & any_of(set_aside)).any():
            raise ValueError(
                f"An event unmatched by {cohort} overlaps a candidate that passed "
                "every gate but is not among the cohort's compared events."
            )

        n_candidates = reasons["n_candidates"]
        furthest = np.full(n_events, -1, dtype=np.int64)
        with_candidates = n_candidates > 0
        if with_candidates.any():
            edges = np.column_stack([low, high])[with_candidates].ravel()
            padded = np.append(rank, -1)
            furthest[with_candidates] = np.maximum.reduceat(padded, edges)[::2]
        stage = REJECTION_STAGES.index
        before_candidates = np.select(
            [
                reasons["n_kept_in_runs"] > 0,
                reasons["n_kept"] > 0,
                reasons["n_windows"] > 0,
            ],
            [stage("no_candidate"), stage("no_run"), stage("no_kept_window")],
            default=stage("no_window"),
        )
        reached = np.where(with_candidates, furthest, before_candidates)
        if (known & (reached < 0)).any():
            raise ValueError(
                f"{int((known & (reached < 0)).sum())} events against {cohort} "
                "reached no rejection stage: every candidate they overlap passed "
                "every gate outside the cohort."
            )
        columns["stage"] = np.where(
            known, np.asarray(REJECTION_STAGES, dtype=object)[reached], UNKNOWN
        )
    for name, dtype in REJECTION_DTYPES.items():
        if name in columns:
            continue
        values = pd.Series(reasons.get(name, np.zeros(n_events)), dtype=dtype)
        columns[name] = values.where(pd.Series(known)).astype(dtype).to_numpy()
    result = pd.DataFrame(columns, index=events.index)
    return result[list(REJECTION_DTYPES)].astype(dict(REJECTION_DTYPES))
