"""One row per candidate sequence the memoryless detector builds.

Every candidate is kept, accepted or not, because a detector's rejections are
as much a result as its detections. Four flags then select the cohorts:
``is_replay`` (the duration and dispersion criteria), ``whole_event_speed_eligible``
(this project's canonical immobility rule over every native speed sample
between the endpoints, merged gaps included), and ``shuffle_pass`` (the
encoding-pair shuffle). The shuffle is run only on candidates that pass the
first two, so ``shuffle_pass`` alone selects the published replay definition.
A candidate that was not shuffled has ``n_shuffles == 0``, which keeps it
apart from one that was shuffled and could not be tested.

Column names follow the tables this one is compared against: timing as the
shipped event table spells it (``event_start_s`` ...), speed as the canonical
classified table does (``animal_max_speed_cm_s`` ...), and the SWR columns
exactly as ``build_classified_nonlocal_event_table`` produces them, by the same
geometry classifier and the same matched-SWR rule. An event whose SWR labels
were not fully observed is ``unknown``, never ``non_swr``.

The key is ``ms_event_id``, deliberately not ``event_id``: that key is hashed
from our own cohort, and a table sharing it would join to the wrong events
without any error.

The design is in ``docs/design/replay_detection_comparison/02-memoryless-sequence.md``,
sections "Data flow, per epoch" and "Output schema".
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from continuum_swr_replay.analysis.hippocampus.event_detection import (
    classify_swr_overlap_geometry,
)
from continuum_swr_replay.analysis.hippocampus.event_immobility import (
    event_speeds_are_immobile,
    filter_event_labels_by_speed,
    intervals_from_labels,
)

# the canonical matched-SWR rule itself, not a copy that could drift from it
from continuum_swr_replay.analysis.hippocampus.event_table import _match_swr
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.detect import (
    CandidateSequences,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.shuffle import (
    ShuffleResult,
)
from continuum_swr_replay.session_keys import SESSION_COLUMNS_DAY

__all__ = [
    "MEMORYLESS_EVENT_TABLE_DTYPES",
    "SHUFFLE_DRAW_TABLE_DTYPES",
    "UNKNOWN",
    "SequenceIdentity",
    "build_memoryless_event_table",
    "build_shuffle_draw_table",
    "canonical_swr_intervals",
    "observed_label_intervals",
    "shuffle_eligible",
]

UNKNOWN = "unknown"
"""SWR label of an event whose SWR labels were not fully observed."""

_TIME_TOLERANCE_S = 1e-9
"""Slack when a window midpoint is compared with a native sample time.

A midpoint is a sample time plus 0.040 s, which need not land on the sample
20 rows later to the last bit.
"""

MEMORYLESS_EVENT_TABLE_DTYPES: dict[str, object] = {
    # identity
    "animal": object,
    "day": np.int64,
    "epoch": object,
    "brain_area": object,
    "run_id": object,
    "swr_definition": object,
    "sequence_label": np.int64,
    "ms_event_id": np.int64,
    # timing, seconds: first and last retained window midpoint
    "event_start_s": np.float64,
    "event_end_s": np.float64,
    "event_duration_s": np.float64,
    "event_midpoint_s": np.float64,
    # the decoding support, half a window wider at each end
    "support_start_s": np.float64,
    "support_end_s": np.float64,
    # detector statistics
    "n_windows": np.int64,
    "n_pieces": np.int64,
    "mean_posterior_spread_cm": np.float64,
    "max_jump_cm": np.float64,
    "mean_jump_cm": np.float64,
    "max_bridge_cm": np.float64,
    "dispersion_cm": np.float64,
    "path_distance_cm": np.float64,
    "is_replay": np.bool_,
    # speed
    "animal_max_speed_cm_s": np.float64,
    "animal_mean_speed_cm_s": np.float64,
    "whole_event_speed_eligible": np.bool_,
    # SWR
    "is_swr": pd.BooleanDtype(),
    "overlap_class": object,
    "overlap_subclass": object,
    "n_swrs_overlapping": pd.Int64Dtype(),
    "matched_swr_event": pd.Int64Dtype(),
    "matched_swr_start": np.float64,
    "matched_swr_end": np.float64,
    # shuffle
    "shuffle_observed_spread_cm": np.float64,
    "shuffle_observed_jump_cm": np.float64,
    "p_spread": np.float64,
    "p_jump": np.float64,
    "shuffle_testable": np.bool_,
    "shuffle_pass": np.bool_,
    "n_shuffles": np.int64,
}
"""Every column, in order, with the dtype it always has, even when empty."""

SHUFFLE_DRAW_TABLE_DTYPES: dict[str, object] = {
    "ms_event_id": np.int64,
    "draw": np.int64,
    "null_spread_cm": np.float64,
    "null_jump_cm": np.float64,
}


@dataclass(frozen=True)
class SequenceIdentity:
    """Which epoch a table describes, and which run produced it.

    Attributes
    ----------
    animal, epoch, brain_area : str
    day : int
    run_id : str
        Joins the table to the run manifest that records its configuration,
        inputs and seeds.

    """

    animal: str
    day: int
    epoch: str
    brain_area: str
    run_id: str

    def __post_init__(self) -> None:
        for name in ("animal", "epoch", "brain_area", "run_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string, got {value!r}.")
        if isinstance(self.day, bool) or not isinstance(self.day, int | np.integer):
            raise TypeError(f"day must be an integer, got {self.day!r}.")


def _empty(dtypes: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(
        {name: pd.Series(dtype=dtype) for name, dtype in dtypes.items()}
    )


def _validated_speed(
    speed_time_s: NDArray, speed_cm_per_s: NDArray
) -> tuple[NDArray, NDArray]:
    time = np.asarray(speed_time_s, dtype=np.float64)
    speed = np.asarray(speed_cm_per_s, dtype=np.float64)
    if time.ndim != 1 or time.shape != speed.shape:
        raise ValueError(
            f"speed_time_s {time.shape} and speed_cm_per_s {speed.shape} must be "
            f"1-D and the same length."
        )
    return time, speed


def _speed_samples(
    time: NDArray, speed: NDArray, start_s: float, end_s: float
) -> NDArray:
    """Every native speed sample from the event's start to its end, inclusive."""
    within = (time >= start_s - _TIME_TOLERANCE_S) & (time <= end_s + _TIME_TOLERANCE_S)
    return speed[within]


def _event_bounds(candidates: CandidateSequences) -> tuple[NDArray, NDArray]:
    runs = candidates.runs
    return candidates.midpoint_s[runs[:, 0]], candidates.midpoint_s[runs[:, 1]]


def shuffle_eligible(
    candidates: CandidateSequences,
    speed_time_s: NDArray,
    speed_cm_per_s: NDArray,
) -> NDArray:
    """Candidates the shuffle is run on.

    Those that meet the replay criteria and are still over their whole
    interval, as the reference tests only its duration- and
    dispersion-accepted events.

    Parameters
    ----------
    candidates : CandidateSequences
    speed_time_s, speed_cm_per_s : ndarray, shape (n_samples,)
        The native speed trace, as used for the canonical immobility rule.

    Returns
    -------
    indices : ndarray of int
        Into ``candidates.runs``, ascending.

    """
    time, speed = _validated_speed(speed_time_s, speed_cm_per_s)
    starts, ends = _event_bounds(candidates)
    threshold = candidates.config.speed_threshold_cm_per_s
    still = np.array(
        [
            event_speeds_are_immobile(
                _speed_samples(time, speed, start, end), speed_threshold=threshold
            )
            for start, end in zip(starts, ends, strict=True)
        ],
        dtype=bool,
    )
    return np.flatnonzero(candidates.is_replay & still)


def _path_statistics(
    candidates: CandidateSequences, sequence: int, bin_distance_cm: NDArray
) -> dict[str, float]:
    """Jumps within pieces, bridges between them, and the path they trace.

    Read from the retained centers through the distance matrix, so a jump out
    of a piece's last window is never counted as a jump inside the sequence.
    """
    center = candidates.decisions.center_index
    pieces = [candidates.pre_merge_runs[piece] for piece in candidates.pieces[sequence]]
    jumps = np.concatenate(
        [
            bin_distance_cm[center[first:last], center[first + 1 : last + 1]]
            for first, last in pieces
        ]
    )
    bridges = np.array(
        [
            bin_distance_cm[center[previous[1]], center[following[0]]]
            for previous, following in zip(pieces[:-1], pieces[1:], strict=True)
        ],
        dtype=np.float64,
    )
    retained = np.concatenate([np.arange(first, last + 1) for first, last in pieces])
    return {
        "n_pieces": len(pieces),
        "mean_posterior_spread_cm": float(
            np.mean(candidates.decisions.spread_cm[retained])
        ),
        "max_jump_cm": float(jumps.max()),
        "mean_jump_cm": float(jumps.mean()),
        "max_bridge_cm": float(bridges.max()) if bridges.size else float("nan"),
        "path_distance_cm": float(jumps.sum() + bridges.sum()),
    }


def _swr_columns(
    starts: NDArray,
    ends: NDArray,
    swr_intervals: pd.DataFrame | None,
    swr_observed_intervals: NDArray | None,
) -> dict[str, list]:
    n = starts.size
    unknown = {
        "is_swr": [pd.NA] * n,
        "overlap_class": [UNKNOWN] * n,
        "overlap_subclass": [UNKNOWN] * n,
        "n_swrs_overlapping": [pd.NA] * n,
        "matched_swr_event": [pd.NA] * n,
        "matched_swr_start": [np.nan] * n,
        "matched_swr_end": [np.nan] * n,
    }
    if swr_intervals is None:
        return unknown
    if swr_observed_intervals is None:
        raise ValueError(
            "SWR labels were given without the intervals over which they were "
            "observed; assuming full coverage would record missing LFP as non-SWR."
        )
    observed = np.asarray(swr_observed_intervals, dtype=np.float64).reshape(-1, 2)
    covered = (
        (observed[None, :, 0] <= starts[:, None] + _TIME_TOLERANCE_S)
        & (ends[:, None] - _TIME_TOLERANCE_S <= observed[None, :, 1])
    ).any(axis=1)

    columns = unknown
    if not covered.any():
        return columns
    events = pd.DataFrame({"start_time": starts[covered], "end_time": ends[covered]})
    classified = classify_swr_overlap_geometry(events, swr_intervals)
    labels = swr_intervals["SWR_event"].to_numpy()
    swr_starts = swr_intervals["start_time"].to_numpy(dtype=np.float64)
    swr_ends = swr_intervals["end_time"].to_numpy(dtype=np.float64)
    for row, index in enumerate(np.flatnonzero(covered)):
        overlap_class = classified["overlap_class"].iloc[row]
        label, matched_start, matched_end = _match_swr(
            overlap_class, starts[index], ends[index], labels, swr_starts, swr_ends
        )
        columns["is_swr"][index] = overlap_class != "non_swr"
        columns["overlap_class"][index] = overlap_class
        columns["overlap_subclass"][index] = classified["overlap_subclass"].iloc[row]
        columns["n_swrs_overlapping"][index] = int(
            classified["n_swrs_overlapping"].iloc[row]
        )
        columns["matched_swr_event"][index] = pd.NA if label is None else label
        columns["matched_swr_start"][index] = matched_start
        columns["matched_swr_end"][index] = matched_end
    return columns


def _shuffle_columns(
    n: int, eligible: NDArray, shuffle_results: Mapping[int, ShuffleResult] | None
) -> dict[str, NDArray]:
    columns = {
        "shuffle_observed_spread_cm": np.full(n, np.nan),
        "shuffle_observed_jump_cm": np.full(n, np.nan),
        "p_spread": np.full(n, np.nan),
        "p_jump": np.full(n, np.nan),
        "shuffle_testable": np.zeros(n, dtype=bool),
        "shuffle_pass": np.zeros(n, dtype=bool),
        "n_shuffles": np.zeros(n, dtype=np.int64),
    }
    if shuffle_results is None:
        return columns
    if set(shuffle_results) != set(eligible.tolist()):
        raise ValueError(
            f"Shuffle results are for candidates {sorted(shuffle_results)}, but the "
            f"eligible candidates are {eligible.tolist()}; every eligible candidate, "
            f"and no other, must be shuffled."
        )
    for index, result in shuffle_results.items():
        columns["shuffle_observed_spread_cm"][index] = result.observed_spread_cm
        columns["shuffle_observed_jump_cm"][index] = result.observed_jump_cm
        columns["p_spread"][index] = result.p_spread
        columns["p_jump"][index] = result.p_jump
        columns["shuffle_testable"][index] = result.testable
        columns["shuffle_pass"][index] = result.passes
        columns["n_shuffles"][index] = result.n_shuffles
    return columns


def _ms_event_id(identity: SequenceIdentity, runs: NDArray) -> NDArray:
    """63-bit stable hash of the epoch, the run and the sequence's windows."""
    keys = pd.DataFrame(
        {
            "animal": identity.animal,
            "day": identity.day,
            "epoch": identity.epoch,
            "brain_area": identity.brain_area,
            "run_id": identity.run_id,
            "first_window": runs[:, 0],
            "last_window": runs[:, 1],
        }
    )
    hashed = pd.util.hash_pandas_object(keys, index=False) % (2**63)
    return np.asarray(hashed, dtype=np.int64)


def _time_indexed(data_over_time: pd.DataFrame) -> pd.DataFrame:
    """Read time from a ``time`` column when there is one, as the canonical table does."""
    if "time" in data_over_time.columns:
        return data_over_time.set_index("time")
    return data_over_time


def canonical_swr_intervals(
    data_over_time: pd.DataFrame, *, speed_threshold: float
) -> pd.DataFrame:
    """The SWR intervals the canonical event table classifies against.

    The same two steps ``build_classified_nonlocal_event_table`` takes, in the
    same order: every ``SWR_event`` label during which the animal moves is
    zeroed whole, then each surviving label's bounds are its first and last
    sample. The two cohorts are then labeled against one interval table, which
    calling the same classifier on different intervals would not give.

    Parameters
    ----------
    data_over_time : DataFrame
        One epoch, time-indexed or with a ``time`` column, carrying
        ``SWR_event``, ``head_speed`` and the session columns.
    speed_threshold : float
        cm/s; the canonical immobility threshold.

    Returns
    -------
    swr_intervals : DataFrame
        Columns ``SWR_event``, ``start_time``, ``end_time``.

    """
    frame = _time_indexed(data_over_time)
    if frame.groupby(list(SESSION_COLUMNS_DAY)).ngroups != 1:
        raise ValueError("canonical_swr_intervals reads one epoch at a time.")
    qualified = filter_event_labels_by_speed(
        frame, ["SWR_event"], speed_threshold=speed_threshold
    )
    return intervals_from_labels(qualified, "SWR_event")


def observed_label_intervals(data_over_time: pd.DataFrame, label_col: str) -> NDArray:
    """Closed intervals over which ``label_col`` holds a value.

    A sample whose label is NaN was not observed. This is only as complete as
    the upstream coding: a label written as 0 where the LFP was missing reads
    as observed and not an SWR, which is the hazard spec 03's
    observation-availability rule exists for.

    Returns
    -------
    observed : ndarray, shape (n_intervals, 2)
        First and last sample time of each run of observed samples.

    """
    frame = _time_indexed(data_over_time)
    time = frame.index.to_numpy(dtype=np.float64)
    observed = np.isfinite(frame[label_col].to_numpy(dtype=np.float64))
    edges = np.diff(observed.astype(np.int8), prepend=0, append=0)
    first = np.flatnonzero(edges == 1)
    last = np.flatnonzero(edges == -1) - 1
    return np.column_stack([time[first], time[last]]).reshape(-1, 2)


def build_memoryless_event_table(
    candidates: CandidateSequences,
    identity: SequenceIdentity,
    *,
    speed_time_s: NDArray,
    speed_cm_per_s: NDArray,
    bin_distance_cm: NDArray,
    swr_intervals: pd.DataFrame | None,
    swr_observed_intervals: NDArray | None,
    swr_definition: str,
    shuffle_results: Mapping[int, ShuffleResult] | None,
) -> pd.DataFrame:
    """One row per candidate sequence, in time order.

    Parameters
    ----------
    candidates : CandidateSequences
        From ``detect_candidate_sequences``.
    identity : SequenceIdentity
    speed_time_s, speed_cm_per_s : ndarray, shape (n_samples,)
        The native speed trace. Every sample between a sequence's endpoints
        decides its eligibility, including samples inside merged gaps.
    bin_distance_cm : ndarray, shape (n_bins, n_bins)
        The track-graph distances the detector was run with.
    swr_intervals : DataFrame or None
        Columns ``SWR_event``, ``start_time``, ``end_time``: the speed-qualified,
        sampled SWR intervals, built as ``build_classified_nonlocal_event_table``
        builds them (``filter_event_labels_by_speed`` then
        ``intervals_from_labels``). None when SWR labels are unavailable.
    swr_observed_intervals : ndarray, shape (n_intervals, 2), or None
        Closed intervals over which SWR labels were observed. Required with
        ``swr_intervals``; an event not wholly inside one is ``unknown``.
    swr_definition : str
        Which SWR detector the labels come from, e.g. ``"Kay"``.
    shuffle_results : mapping of int to ShuffleResult, or None
        Keyed by candidate index. Must cover exactly the candidates
        :func:`shuffle_eligible` returns. None when the shuffle was not run.

    Returns
    -------
    table : DataFrame
        Columns and dtypes as ``MEMORYLESS_EVENT_TABLE_DTYPES``, whether or
        not it has rows. Row ``i`` is candidate ``i``.

    Raises
    ------
    ValueError
        The speed arrays disagree, SWR labels come without their observed
        intervals, or the shuffle results do not match the eligible set.

    """
    time, speed = _validated_speed(speed_time_s, speed_cm_per_s)
    if not swr_definition:
        raise ValueError("swr_definition must name the SWR detector.")
    n = len(candidates.runs)
    eligible = shuffle_eligible(candidates, time, speed)
    shuffle = _shuffle_columns(n, eligible, shuffle_results)
    if n == 0:
        return _empty(MEMORYLESS_EVENT_TABLE_DTYPES)

    starts, ends = _event_bounds(candidates)
    half_window = candidates.config.window_duration_s / 2
    threshold = candidates.config.speed_threshold_cm_per_s
    speeds = [
        _speed_samples(time, speed, s, e) for s, e in zip(starts, ends, strict=True)
    ]
    paths = [_path_statistics(candidates, i, bin_distance_cm) for i in range(n)]

    columns = {
        "animal": identity.animal,
        "day": identity.day,
        "epoch": identity.epoch,
        "brain_area": identity.brain_area,
        "run_id": identity.run_id,
        "swr_definition": swr_definition,
        "sequence_label": np.arange(1, n + 1),
        "ms_event_id": _ms_event_id(identity, candidates.runs),
        "event_start_s": starts,
        "event_end_s": ends,
        "event_duration_s": ends - starts,
        "event_midpoint_s": (starts + ends) / 2,
        "support_start_s": starts - half_window,
        "support_end_s": ends + half_window,
        "n_windows": candidates.runs[:, 1] - candidates.runs[:, 0] + 1,
        **{key: [path[key] for path in paths] for key in paths[0]},
        "dispersion_cm": candidates.dispersion_cm,
        "is_replay": candidates.is_replay,
        "animal_max_speed_cm_s": [s.max() if s.size else np.nan for s in speeds],
        "animal_mean_speed_cm_s": [s.mean() if s.size else np.nan for s in speeds],
        "whole_event_speed_eligible": [
            event_speeds_are_immobile(s, speed_threshold=threshold) for s in speeds
        ],
        **_swr_columns(starts, ends, swr_intervals, swr_observed_intervals),
        **shuffle,
    }
    table = pd.DataFrame(columns)[list(MEMORYLESS_EVENT_TABLE_DTYPES)].astype(
        MEMORYLESS_EVENT_TABLE_DTYPES
    )
    if not table.ms_event_id.is_unique:
        raise RuntimeError("ms_event_id collided within one epoch.")
    return table


def build_shuffle_draw_table(
    table: pd.DataFrame, shuffle_results: Mapping[int, ShuffleResult]
) -> pd.DataFrame:
    """Every shuffled draw's two statistics, for auditing the null distributions.

    Parameters
    ----------
    table : DataFrame
        From :func:`build_memoryless_event_table`; row ``i`` is candidate ``i``.
    shuffle_results : mapping of int to ShuffleResult
        Keyed by candidate index, as given to the table.

    Returns
    -------
    draws : DataFrame
        One row per draw, keyed on ``ms_event_id``, columns and dtypes as
        ``SHUFFLE_DRAW_TABLE_DTYPES``.

    """
    if not shuffle_results:
        return _empty(SHUFFLE_DRAW_TABLE_DTYPES)
    parts = [
        pd.DataFrame(
            {
                "ms_event_id": table.ms_event_id.iloc[index],
                "draw": np.arange(1, result.n_shuffles + 1),
                "null_spread_cm": result.null_spread_cm,
                "null_jump_cm": result.null_jump_cm,
            }
        )
        for index, result in sorted(shuffle_results.items())
    ]
    return pd.concat(parts, ignore_index=True).astype(SHUFFLE_DRAW_TABLE_DTYPES)
