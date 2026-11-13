"""One row per candidate sequence the memoryless detector builds.

Every candidate is kept, accepted or not, because a detector's rejections are
as much a result as its detections. Three flags then select the cohorts:
``is_replay`` (the duration and dispersion criteria), ``whole_event_speed_eligible``
(this project's canonical immobility rule over every native speed sample
between the endpoints, merged gaps included), and ``shuffle_pass`` (the
encoding-pair shuffle). The shuffle is run on every candidate that meets the
replay criteria, as the reference tests every such candidate, so
``shuffle_pass`` alone selects the published replay definition, and
``shuffle_pass & whole_event_speed_eligible`` selects this project's cohort,
whose immobility rule is stricter than the reference's per-window gate. A
candidate that was not shuffled has ``n_shuffles == 0``. A fourth column,
``artifact_drop_reason``, applies the project's one analysed-time rule
(``event_definition.eligibility``): a candidate sharing time with the artifact mask
(``"artifact"``), within ``ARTIFACT_REACH_S`` of it (``"near_artifact"``), or
overlapping an SWR or MUA event dropped for either (``"overlaps_dropped_event"``)
is in no cohort; it is None for every other candidate.

Three columns average three different window sets, each right for its
purpose: ``n_windows`` counts every window of the span, merged gaps included;
``mean_posterior_spread_cm``, ``mean_jump_cm`` and ``max_jump_cm`` read the
retained windows only, with jumps taken within pieces; and
``shuffle_observed_spread_cm`` / ``shuffle_observed_jump_cm`` are the
reference's test statistics over every window of the span, zero jumps set
aside. The SWR columns classify against the speed-qualified SWR intervals the
canonical table uses, so on a row that is not ``whole_event_speed_eligible``
``is_swr == False`` means no *speed-qualified* SWR overlapped, not that no
SWR was detected there.

Column names follow the tables this one is compared against: timing as the
shipped event table spells it (``event_start_s`` ...), speed as the canonical
classified table does (``animal_max_speed_cm_s`` ...), and the SWR columns
exactly as ``build_classified_nonlocal_event_table`` produces them, by the same
geometry classifier and the same matched-SWR rule. An event whose SWR labels
were not fully observed is ``unknown``, never ``non_swr``.

The key is ``ms_event_id``, deliberately not ``event_id``: that key is hashed
from our own cohort, and a table sharing it would join to the wrong events
without any error. It is hashed from the decode and the sequence's window
bounds, not from ``run_id``, so two runs of one decode (a sensitivity run
against its baseline, say) can be joined on it; ``run_id`` is a column.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from continuum_swr_replay.analysis.detection.memoryless_sequence.detect import (
    CandidateSequences,
    _is_above,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.shuffle import (
    ShuffleResult,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.window_statistics import (
    dispersion_cm,
)
from continuum_swr_replay.analysis.event_definition.classified_table import match_swr
from continuum_swr_replay.analysis.event_definition.eligibility import (
    artifact_drop_reasons,
)
from continuum_swr_replay.analysis.event_definition.immobility import (
    event_speeds_are_immobile,
    filter_event_labels_by_speed,
    intervals_from_labels,
    require_canonical_speed_threshold,
)
from continuum_swr_replay.analysis.event_definition.nonlocal_event_rule import (
    classify_swr_overlap_geometry,
)
from continuum_swr_replay.data_loaders.constants import BRAIN_AREAS
from continuum_swr_replay.decoding.identifiers import parse_epoch_identifier

# the canonical matched-SWR rule itself, not a copy that could drift from it
from continuum_swr_replay.intervals import contiguous_true_runs
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS_DAY

__all__ = [
    "MEMORYLESS_EVENT_TABLE_DTYPES",
    "SHUFFLE_DRAW_TABLE_DTYPES",
    "UNKNOWN",
    "SequenceIdentity",
    "build_memoryless_event_table",
    "build_shuffle_draw_table",
    "canonical_swr_intervals",
    "observed_intervals",
    "outside_window_call_statistics",
    "shuffle_eligible",
]

UNKNOWN = "unknown"
"""SWR label of an event whose SWR labels were not fully observed."""


MEMORYLESS_EVENT_TABLE_DTYPES: Mapping[str, object] = MappingProxyType(
    {
        # identity
        "animal": object,
        "day": np.int64,
        "epoch": object,
        "brain_area": object,
        "epoch_id": object,
        "run_id": object,
        "swr_definition": object,
        "sequence_label": np.int64,
        "ms_event_id": np.int64,
        # the sequence's first and last window index, which with the decode key
        # the event id; n_windows is their difference plus one
        "first_window": np.int64,
        "last_window": np.int64,
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
        # the same dispersion from the pieces' windows alone, and how many kept
        # windows of the span lie outside every piece; a verdict that rests on
        # those windows has a dispersion well above its pieces-only one
        "pieces_dispersion_cm": np.float64,
        "n_kept_windows_outside_pieces": np.int64,
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
        # the analysed-time rule: "artifact", "near_artifact",
        # "overlaps_dropped_event" or None (kept)
        "artifact_drop_reason": object,
    }
)
"""Every column, in order, with the dtype it always has, even when empty."""

SHUFFLE_DRAW_TABLE_DTYPES: Mapping[str, object] = MappingProxyType(
    {
        "ms_event_id": np.int64,
        "draw": np.int64,
        "null_spread_cm": np.float64,
        "null_jump_cm": np.float64,
    }
)


@dataclass(frozen=True)
class SequenceIdentity:
    """Which epoch a table describes, and which run produced it.

    Attributes
    ----------
    animal, epoch, brain_area : str
    day : int
    epoch_id : str
        The decode's full identifier, e.g. ``chimi20200212_02_r1_HPC_all``.
        One epoch can be decoded more than one way, and its sequences must
        not share keys across those decodes.
    run_id : str
        Joins the table to the run manifest that records its configuration,
        inputs and seeds.

    """

    animal: str
    day: int
    epoch: str
    brain_area: str
    epoch_id: str
    run_id: str

    def __post_init__(self) -> None:
        for name in ("animal", "epoch", "brain_area", "epoch_id", "run_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string, got {value!r}.")
        if isinstance(self.day, bool) or not isinstance(self.day, int | np.integer):
            raise TypeError(f"day must be an integer, got {self.day!r}.")
        if self.day <= 0:
            raise ValueError(f"day must be positive, got {self.day}.")
        if self.brain_area not in BRAIN_AREAS:
            raise ValueError(
                f"brain_area must be one of {BRAIN_AREAS}, got {self.brain_area!r}."
            )
        # the identifier names the same epoch as the parts do
        try:
            key = parse_epoch_identifier(self.epoch_id)
            named = (key.animal, key.date, key.epoch_name, key.brain_area)
        except ValueError:
            named = None
        if named != (self.animal, self.day, self.epoch, self.brain_area):
            raise ValueError(
                f"epoch_id {self.epoch_id!r} does not name animal {self.animal!r}, "
                f"epoch {self.epoch!r} and brain area {self.brain_area!r}."
            )


def _empty(dtypes: Mapping[str, object]) -> pd.DataFrame:
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
    if np.any(np.diff(time) <= 0):
        raise ValueError("speed_time_s must be strictly increasing.")
    return time, speed


def _speed_samples(
    time: NDArray, speed: NDArray, start_s: float, end_s: float
) -> NDArray:
    """Every native speed sample from the event's start to its end, inclusive.

    Event bounds are window midpoints, samples of the detector's grid. On the
    decode's own grid they are speed samples and are included; on the 5 ms
    grid every second bound falls halfway between two speed samples, and the
    samples inside the event are those between the bounds.
    """
    first = np.searchsorted(time, start_s, side="left")
    last = np.searchsorted(time, end_s, side="right")
    return speed[first:last]


def _event_bounds(candidates: CandidateSequences) -> tuple[NDArray, NDArray]:
    runs = candidates.runs
    return candidates.midpoint_s[runs[:, 0]], candidates.midpoint_s[runs[:, 1]]


def shuffle_eligible(candidates: CandidateSequences) -> NDArray:
    """Candidates the shuffle is run on: those that meet both replay criteria.

    The reference tests every duration- and dispersion-accepted event, and so
    does the port, whether or not the animal was still over the whole event.
    This project's whole-event immobility rule is a column,
    ``whole_event_speed_eligible``, to filter on afterwards; running the
    shuffle on the eligible subset only would bake that cohort choice into
    what was computed rather than into how it is read.

    Parameters
    ----------
    candidates : CandidateSequences

    Returns
    -------
    indices : ndarray of int
        Into ``candidates.runs``, ascending.

    """
    return np.flatnonzero(candidates.is_replay)


def _path_statistics(
    candidates: CandidateSequences, sequence: int, bin_distance_cm: NDArray
) -> dict[str, float]:
    """Jumps within pieces, bridges between them, and the path they trace.

    Read from the retained centers through the distance matrix, so a jump out
    of a piece's last window is never counted as a jump inside the sequence.
    Also the pieces-only dispersion and the count of kept windows outside the
    pieces: ``dispersion_cm`` follows the reference over every kept window of
    the merged span, a single window dropped before the merge included.
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
    start, stop = candidates.runs[sequence]
    kept = np.flatnonzero(candidates.decisions.gates.kept[start : stop + 1]) + start
    return {
        "n_pieces": len(pieces),
        "mean_posterior_spread_cm": float(
            np.mean(candidates.decisions.spread_cm[retained])
        ),
        # a sequence merged from single windows alone has no consecutive jump
        "max_jump_cm": float(jumps.max()) if jumps.size else float("nan"),
        "mean_jump_cm": float(jumps.mean()) if jumps.size else float("nan"),
        "max_bridge_cm": float(bridges.max()) if bridges.size else float("nan"),
        "pieces_dispersion_cm": dispersion_cm(center[retained], bin_distance_cm),
        "n_kept_windows_outside_pieces": int(np.setdiff1d(kept, retained).size),
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
        (observed[None, :, 0] <= starts[:, None])
        & (ends[:, None] <= observed[None, :, 1])
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
        label, matched_start, matched_end = match_swr(
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
            f"candidates meeting the replay criteria are {eligible.tolist()}; every "
            f"one of those, and no other, must be shuffled."
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
    """63-bit stable hash of the decode and the sequence's window bounds.

    Not of ``run_id``: the same sequence found by two runs of one decode gets
    one key, so a sensitivity run joins to its baseline on it. Two decodes of
    one epoch differ in ``epoch_id`` and never share a key. Uniqueness is
    asserted within an epoch; across the corpus it holds to about one in
    ``2**63 / n**2``.
    """
    keys = pd.DataFrame(
        {
            "animal": identity.animal,
            "day": np.int64(identity.day),
            "epoch": identity.epoch,
            "brain_area": identity.brain_area,
            "epoch_id": identity.epoch_id,
            "first_window": np.asarray(runs[:, 0], dtype=np.int64),
            "last_window": np.asarray(runs[:, 1], dtype=np.int64),
        }
    )
    hashed = pd.util.hash_pandas_object(keys, index=False) % (2**63)
    return np.asarray(hashed, dtype=np.int64)


def canonical_swr_intervals(
    data_over_time: pd.DataFrame, *, speed_threshold: float
) -> pd.DataFrame:
    """The SWR intervals the canonical event table classifies against.

    At the canonical threshold, the same two steps the canonical table's SWRs
    take, in the same order: every ``SWR_event`` label during which the animal
    moves is zeroed whole (the read does this at the canonical threshold),
    then each surviving label's bounds are its first and last sample. A
    stricter threshold zeroes, in addition, the SWRs the read kept that it
    rejects; a looser one is refused, because it could not re-admit the SWRs
    the read dropped. The two cohorts are then labeled against one interval
    table, which calling the same classifier on different intervals would not
    give.

    Parameters
    ----------
    data_over_time : DataFrame
        One epoch, time-indexed, carrying
        ``SWR_event``, ``head_speed`` and the session columns: a frame read
        through the SWR source (``analysed_time.read_dot_with_swr_source``), whose
        ``SWR_event`` is the direct Kay inventory's, as the canonical table's is.
    speed_threshold : float
        cm/s; the canonical immobility threshold, or a stricter (lower) one.

    Returns
    -------
    swr_intervals : DataFrame
        Columns ``SWR_event``, ``start_time``, ``end_time``.

    Raises
    ------
    ValueError
        If `speed_threshold` is looser (higher) than the canonical
        ``DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC``, or the frame holds more
        than one epoch.
    """
    require_canonical_speed_threshold(
        speed_threshold, "canonical_swr_intervals", allow_stricter=True
    )
    if data_over_time.groupby(list(SESSION_COLUMNS_DAY)).ngroups != 1:
        raise ValueError("canonical_swr_intervals reads one epoch at a time.")
    # A no-op at the canonical threshold, where the read already gated the
    # SWRs; kept so the SWR cohort follows the detector's configured speed
    # threshold (MemorylessConfig.speed_threshold_cm_per_s) when it is stricter.
    qualified = filter_event_labels_by_speed(
        data_over_time, ["SWR_event"], speed_threshold=speed_threshold
    )
    return intervals_from_labels(qualified, "SWR_event")


def observed_intervals(data_over_time: pd.DataFrame, column: str) -> NDArray:
    """Closed intervals over which ``column`` holds a finite value.

    Not for SWR labels: ``SWR_event`` is an integer and never NaN, so a
    stretch of missing LFP would read as observed and not an SWR. Where SWR
    detection was observed is the SWR source's
    (``analysed_time.swr_observed_stretches``), which the runner passes.

    Returns
    -------
    observed : ndarray, shape (n_intervals, 2)
        First and last sample time of each run of observed samples.

    """
    time = data_over_time.index.to_numpy(dtype=np.float64)
    observed = np.isfinite(data_over_time[column].to_numpy(dtype=np.float64))
    first, last = (contiguous_true_runs(observed) - [0, 1]).T
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
    artifact_intervals: pd.DataFrame | None = None,
    dropped_events: pd.DataFrame | None = None,
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
        builds them (the read's speed-gated labels, then
        ``intervals_from_labels``; :func:`canonical_swr_intervals`). None when SWR labels are unavailable.
    swr_observed_intervals : ndarray, shape (n_intervals, 2), or None
        Closed intervals over which SWR labels were observed: the SWR source's
        stretches (``analysed_time.swr_observed_stretches``). Required with
        ``swr_intervals``; an event not wholly inside one is ``unknown``.
    swr_definition : str
        Which SWR source the labels come from: ``swr_source.swr_source_name()``,
        such as ``"kay@<run id>+artifacts@none"``.
    shuffle_results : mapping of int to ShuffleResult, or None
        Keyed by candidate index. Must cover exactly the candidates that meet
        both replay criteria, as :func:`shuffle_eligible` returns them. None
        when the shuffle was not run.
    artifact_intervals : DataFrame or None
        The epoch's artifact mask (``swr_source.artifact_intervals``):
        ``start_time`` and ``end_time``, closed. Each candidate's
        ``artifact_drop_reason`` is read from its ``[event_start_s,
        event_end_s]`` against excluded time (the mask widened by
        ``ARTIFACT_REACH_S``). None masks nothing.
    dropped_events : DataFrame or None
        The SWR and MUA events dropped for excluded time
        (``analysed_time.dropped_for_excluded_time``): a candidate otherwise kept
        that overlaps one is ``"overlaps_dropped_event"``. None checks none.

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
    eligible = shuffle_eligible(candidates)
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
        "epoch_id": identity.epoch_id,
        "run_id": identity.run_id,
        "swr_definition": swr_definition,
        "sequence_label": np.arange(1, n + 1),
        "ms_event_id": _ms_event_id(identity, candidates.runs),
        "first_window": candidates.runs[:, 0],
        "last_window": candidates.runs[:, 1],
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
        "artifact_drop_reason": (
            np.full(n, None, dtype=object)
            if artifact_intervals is None
            else artifact_drop_reasons(starts, ends, artifact_intervals, dropped_events)
        ),
    }
    table = pd.DataFrame(columns)[list(MEMORYLESS_EVENT_TABLE_DTYPES)].astype(
        dict(MEMORYLESS_EVENT_TABLE_DTYPES)
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
    return pd.concat(parts, ignore_index=True).astype(dict(SHUFFLE_DRAW_TABLE_DTYPES))


def outside_window_call_statistics(
    table: pd.DataFrame, dispersion_threshold_cm: float
) -> dict[str, int]:
    """Replay calls whose verdict rests on kept windows outside every piece.

    A replay call is a candidate that meets the duration and dispersion
    criteria (``is_replay``). It is decided by outside windows when it has at
    least one kept window outside every piece and its pieces' own dispersion
    (``pieces_dispersion_cm``) is not above the threshold: its dispersion
    clears the threshold only with those windows. "Above" is the detector's
    own comparison, the one ``is_replay`` was decided with: a dispersion
    within its relative tolerance of the threshold is equal to it, not above,
    and neither is a missing (NaN) one, so a call with an outside window and
    no pieces' dispersion counts as decided by it.

    Every candidate of the table is read, whatever its
    ``whole_event_speed_eligible`` and ``artifact_drop_reason``.

    Parameters
    ----------
    table : DataFrame
        From :func:`build_memoryless_event_table`.
    dispersion_threshold_cm : float
        The detector's dispersion threshold, in cm.

    Returns
    -------
    statistics : dict
        Counts of candidates.

        - ``n_replay_calls``: the replay calls.
        - ``n_decided_by_outside_window``: of those, the calls decided by
          outside windows.
        - ``n_decided_passing_shuffle``: of the calls decided by outside
          windows, those that passed the shuffle (``shuffle_pass``).

    """
    replay = table[table["is_replay"]]
    decided = replay[
        (replay["n_kept_windows_outside_pieces"] > 0)
        & ~_is_above(replay["pieces_dispersion_cm"], dispersion_threshold_cm)
    ]
    return {
        "n_replay_calls": len(replay),
        "n_decided_by_outside_window": len(decided),
        "n_decided_passing_shuffle": int(decided["shuffle_pass"].sum()),
    }
