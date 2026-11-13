"""Canonical whole-event immobility rules for neural event cohorts.

An event is eligible only when it contains at least one sampled speed and
every sampled speed is finite and strictly below the configured threshold.
The strict comparison matches the repository's immobility-bout definition.

*Atomic* throughout this module means an event is accepted or rejected as a
whole — never clipped bin by bin.

Caveats worth knowing before citing a filtered count:

- Ineligibility has three causes that are not distinguished in the returned
  mask: the animal was moving, the speed sample was non-finite, or the
  interval covers no sample at all (a position-tracking gap, or an interval
  past the position tail). All three yield ``False``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.constants import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    DEFAULT_NON_LOCAL_THRESHOLD,
)
from continuum_swr_replay.data_loaders.utils import get_labels
from continuum_swr_replay.tables.columns import require_columns
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS_DAY

logger = logging.getLogger(__name__)

DEFAULT_EVENT_GROUP_COLS = SESSION_COLUMNS_DAY


class EventDefinitionError(ValueError):
    """Raised when saved event labels violate the canonical definition.

    This dedicated subtype distinguishes stale or internally inconsistent
    event artifacts from ordinary per-session data gaps. Batch table builders
    must propagate it instead of silently skipping the affected session.
    """


def _validate_speed_threshold(speed_threshold: float) -> float:
    threshold = float(speed_threshold)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError(
            f"speed_threshold must be finite and positive, got {speed_threshold!r}"
        )
    return threshold


def require_time_index(frame: pd.DataFrame, what: str) -> None:
    """Refuse a frame whose index is not one epoch's time axis.

    A decode frame's time is its index, in seconds, and nothing else: every
    reader takes an event's bounds from its first and last labelled row. A
    frame that lost that index (``reset_index(drop=True)``, a concat with
    ``ignore_index=True``) would have its row numbers read as seconds, and a
    ``time`` column would be a second time axis, so both are refused.

    Parameters
    ----------
    frame : pd.DataFrame
        The frame to check.
    what : str
        The caller's name, for the message.

    Raises
    ------
    ValueError
        If `frame` has a ``time`` column, an index that is not float (a
        ``RangeIndex`` or other integer index among them), or an index that
        is not strictly increasing.
    """
    index = frame.index
    if "time" in frame.columns:
        raise ValueError(
            f"{what}: the frame has a 'time' column, a second time axis; set the "
            "time index instead (frame.set_index('time'))"
        )
    if isinstance(index, pd.RangeIndex) or not pd.api.types.is_float_dtype(index):
        raise ValueError(
            f"{what}: the frame's index ({type(index).__name__}, dtype "
            f"{index.dtype}) is not a float time axis in seconds; set the time "
            "index (the decode frame's own, as get_data_over_time saves it)"
        )
    if not (index.is_monotonic_increasing and index.is_unique):
        raise ValueError(
            f"{what}: the frame's time index is not strictly increasing; pass one "
            "epoch at a time, sorted by time, with no repeated timestamp"
        )


def require_canonical_speed_threshold(
    speed_threshold: float, where: str, *, allow_stricter: bool = False
) -> None:
    """Refuse a speed threshold the read's SWR and MUA cohorts do not follow.

    The read (``analysed_time.with_swr_source``) gates the SWR and MUA labels at
    the canonical :data:`DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC`, once for
    every consumer. A builder that gated its own events at another value would
    classify them against SWRs and MUA events gated at the canonical one, a
    mixed cohort, so it refuses the value instead.

    Parameters
    ----------
    speed_threshold : float
        The caller's threshold, cm/s.
    where : str
        The caller's name, for the message.
    allow_stricter : bool
        Accept a threshold below the canonical one too: for a filter that
        re-gates the read's SWRs itself, which can drop SWRs the read kept but
        cannot re-admit ones it dropped.

    Raises
    ------
    ValueError
        If `speed_threshold` is not the canonical value (or, with
        `allow_stricter`, is above it).
    """
    canonical = DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC
    threshold = float(speed_threshold)
    if allow_stricter and threshold <= canonical:
        return
    if threshold == canonical:
        return
    accepted = (
        f"{canonical:g} cm/s or a stricter (lower) value"
        if allow_stricter
        else f"the canonical {canonical:g} cm/s"
    )
    raise ValueError(
        f"{where}: speed_threshold={speed_threshold!r} cm/s is not supported; "
        f"pass {accepted}. The read (analysed_time.with_swr_source) gates the "
        f"SWR and MUA labels at {canonical:g} cm/s, so another value would "
        "mix speed cohorts."
    )


def event_speeds_are_immobile(
    speed: Sequence[float] | np.ndarray,
    *,
    speed_threshold: float = DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
) -> bool:
    """Return whether all samples in one event satisfy the canonical rule."""
    threshold = _validate_speed_threshold(speed_threshold)
    values = np.asarray(speed, dtype=float)
    if values.ndim != 1:
        raise ValueError("speed must be one-dimensional")
    return bool(
        values.size > 0 and np.isfinite(values).all() and (values < threshold).all()
    )


def whole_event_immobility_mask(
    data: pd.DataFrame,
    event_col: str,
    *,
    speed_col: str = "head_speed",
    speed_threshold: float = DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    group_cols: Sequence[str] = DEFAULT_EVENT_GROUP_COLS,
) -> pd.Series:
    """Return an atomic per-row mask for eligible labeled events.

    Positive event labels are grouped within each session. Every row of a
    qualifying event is ``True`` and every row of an ineligible event is
    ``False``; label-zero rows are always ``False``.
    """
    threshold = _validate_speed_threshold(speed_threshold)
    key_cols = [*group_cols, event_col]
    require_columns(data, [*key_cols, speed_col], "data")

    labels = data[event_col].to_numpy()
    speed = data[speed_col].to_numpy(dtype=float)
    active = labels > 0
    mask = np.zeros(len(data), dtype=bool)
    if not active.any():
        return pd.Series(mask, index=data.index, name=f"{event_col}_is_immobile")

    active_groups = data.loc[active, key_cols].reset_index(drop=True)
    active_groups["_sample_is_immobile"] = np.isfinite(speed[active]) & (
        speed[active] < threshold
    )
    event_is_immobile = (
        active_groups.groupby(key_cols, sort=False, dropna=False)["_sample_is_immobile"]
        .transform("all")
        .to_numpy(dtype=bool)
    )
    mask[active] = event_is_immobile
    return pd.Series(mask, index=data.index, name=f"{event_col}_is_immobile")


def filter_event_labels_by_speed(
    data: pd.DataFrame,
    event_cols: Sequence[str],
    *,
    speed_col: str = "head_speed",
    speed_threshold: float = DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    group_cols: Sequence[str] = DEFAULT_EVENT_GROUP_COLS,
) -> pd.DataFrame:
    """Return a copy with every ineligible event label set atomically to zero.

    A frame read through ``analysed_time.with_swr_source`` already holds only
    eligible events at the canonical threshold, so this is for a stricter
    threshold than the read's, or for a frame that was not read that way.
    """
    # Share untouched column blocks: pooled data-over-time frames can be many
    # gigabytes. Each event-label column is replaced with its own copied array,
    # so the caller's frame remains immutable without duplicating every signal.
    filtered = data.copy(deep=False)
    for event_col in event_cols:
        eligible = whole_event_immobility_mask(
            filtered,
            event_col,
            speed_col=speed_col,
            speed_threshold=speed_threshold,
            group_cols=group_cols,
        ).to_numpy()
        labels = filtered[event_col].to_numpy(copy=True)
        active = labels > 0
        labels[active & ~eligible] = 0
        filtered[event_col] = labels
    return filtered


def validate_whole_event_immobility_contract(
    data: pd.DataFrame,
    event_col: str,
    *,
    speed_threshold: float = DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
) -> None:
    """Fail when a saved event label holds a sample the immobility rule rejects.

    Saved labels are created from whole-event-immobile intervals, so an
    ineligible label is a stale or malformed artifact. It is refused, not
    zeroed: zeroing it would silently change the cohort.

    Parameters
    ----------
    data : pd.DataFrame
        Carrying ``event_col``, ``head_speed`` and the session columns
        (:data:`DEFAULT_EVENT_GROUP_COLS`).
    event_col : str
        The label column to check.
    speed_threshold : float
        cm/s; the rule is strictly ``<``.

    Raises
    ------
    EventDefinitionError
        If any labeled event has a sampled speed that is not finite and below
        `speed_threshold`.
    """
    labels = data[event_col].to_numpy()
    active = labels > 0
    if not active.any():
        return
    immobile = whole_event_immobility_mask(
        data, event_col, speed_threshold=speed_threshold
    ).to_numpy()
    invalid = active & ~immobile
    if invalid.any():
        invalid_ids = sorted(pd.unique(labels[invalid]).tolist())
        raise EventDefinitionError(
            f"{event_col} labels violate the whole-event immobility "
            f"contract (all speeds finite and < {speed_threshold:g} cm/s); "
            f"invalid event labels: {invalid_ids[:10]}. Regenerate the "
            "data-over-time file with the unified event definition."
        )


def validate_nonlocal_event_probability_contract(
    data: pd.DataFrame,
    *,
    event_col: str = "non_local_event",
    probability_cols: Sequence[str] = (
        "Non-Local_Continuous_prob",
        "Non-Local_Fragmented_prob",
    ),
    probability_threshold: float = DEFAULT_NON_LOCAL_THRESHOLD,
) -> None:
    """Fail when a saved Non-Local label contains an invalid probability bin.

    Non-Local labels are created from complete runs whose summed posterior is
    strictly above the configured threshold. A stale or malformed artifact
    must not be clipped per bin or admitted by only some downstream consumers;
    every active label is therefore validated atomically, once, where the
    frame is read (``analysed_time.with_swr_source``). Unlabeled bins are
    intentionally ignored.
    """
    threshold = float(probability_threshold)
    if not np.isfinite(threshold):
        raise ValueError(
            f"probability_threshold must be finite, got {probability_threshold!r}"
        )
    require_columns(data, [event_col], "data")
    labels = data[event_col].to_numpy()
    active = labels > 0
    if not active.any():
        return

    require_columns(data, probability_cols, "data")

    probability = np.zeros(len(data), dtype=float)
    for column in probability_cols:
        probability += data[column].to_numpy(dtype=float)
    invalid = active & ~(np.isfinite(probability) & (probability > threshold))
    if invalid.any():
        invalid_ids = sorted(pd.unique(labels[invalid]).tolist())
        raise EventDefinitionError(
            "non_local_event labels violate the probability-threshold "
            f"contract (all probabilities finite and > {threshold:g}); "
            f"invalid event labels: {invalid_ids[:10]}. Regenerate the "
            "data-over-time file at the requested threshold."
        )


def filter_event_intervals_by_speed(
    event_times: pd.DataFrame,
    *,
    speed: Sequence[float] | np.ndarray,
    time: Sequence[float] | np.ndarray,
    speed_threshold: float = DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
) -> pd.DataFrame:
    """Filter interval rows using the canonical whole-event speed rule.

    Samples in the closed interval ``[start_time, end_time]`` are evaluated,
    matching :func:`continuum_swr_replay.data_loaders.utils.get_labels`.
    The input row order, index, and all columns are preserved.

    How many intervals this removed is logged at INFO, so a caller reporting
    ``n_SWR`` / ``n_MUA`` leaves a record of the drop rather than only the
    surviving count.
    """
    threshold = _validate_speed_threshold(speed_threshold)
    require_columns(event_times, ["start_time", "end_time"], "event_times")

    time_values = np.asarray(time, dtype=float)
    speed_values = np.asarray(speed, dtype=float)
    if time_values.ndim != 1 or speed_values.ndim != 1:
        raise ValueError("time and speed must be one-dimensional")
    if len(time_values) != len(speed_values):
        raise ValueError(
            "time and speed must have equal length, "
            f"got {len(time_values)} and {len(speed_values)}"
        )
    if not np.isfinite(time_values).all():
        raise ValueError("time must contain only finite values")
    if len(time_values) > 1 and not (np.diff(time_values) > 0).all():
        raise ValueError("time must be strictly increasing")

    starts = event_times["start_time"].to_numpy(dtype=float)
    ends = event_times["end_time"].to_numpy(dtype=float)
    if not (np.isfinite(starts).all() and np.isfinite(ends).all()):
        raise ValueError("event start_time and end_time must be finite")
    if (ends < starts).any():
        raise ValueError("event end_time must be greater than or equal to start_time")

    start_indices = np.searchsorted(time_values, starts, side="left")
    end_indices = np.searchsorted(time_values, ends, side="right")
    keep = np.fromiter(
        (
            event_speeds_are_immobile(
                speed_values[start:end], speed_threshold=threshold
            )
            for start, end in zip(start_indices, end_indices, strict=True)
        ),
        dtype=bool,
        count=len(event_times),
    )
    n_removed = int((~keep).sum())
    if n_removed:
        logger.info(
            "whole-event speed rule (< %g cm/s) removed %d of %d intervals",
            threshold,
            n_removed,
            len(keep),
        )
    return event_times.loc[keep].copy()


def intervals_from_labels(
    session_df: pd.DataFrame,
    label_col: str,
) -> pd.DataFrame:
    """Return sampled interval bounds for each positive event label.

    Time is the frame's index, the decode frame's one time axis
    (:func:`require_time_index`: float, strictly increasing, no ``time``
    column). Bounds therefore describe the exact samples used by downstream
    tables, rather than the detector's continuous threshold-crossing bounds.

    Raises
    ------
    ValueError
        If `label_col` is missing, or the frame's index is not a time axis
        (:func:`require_time_index`).
    """
    require_columns(session_df, [label_col], "session_df")
    require_time_index(session_df, "intervals_from_labels")

    is_active = session_df[label_col].to_numpy() > 0
    active = session_df.loc[is_active, [label_col]]
    if active.empty:
        return pd.DataFrame(columns=[label_col, "start_time", "end_time"])

    return (
        active.assign(time=active.index.to_numpy())
        .groupby(label_col, sort=True)["time"]
        .agg(start_time="min", end_time="max")
        .reset_index()
    )


def build_speed_eligible_event_cohort(
    event_times: pd.DataFrame,
    *,
    speed: Sequence[float] | np.ndarray,
    time: Sequence[float] | np.ndarray | pd.Index,
    event_col: str,
    speed_threshold: float = DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    preserve_source_ids: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Filter, label, and quantize one event source on a single time axis.

    The returned interval table is reconstructed from the labels, so event
    counts and overlap geometry use the same sampled bounds as model tables.
    Using one ``time`` argument for both speed qualification and label writing
    also makes it impossible for small decoder/position time drift to leave an
    unchecked speed sample inside a labeled event.

    Returns
    -------
    canonical_intervals : pd.DataFrame
        Columns ``event_col``, ``start_time``, ``end_time``, and ``duration``.
    labels : pd.Series
        Integer event labels aligned to ``time`` and named ``event_col``.

    Notes
    -----
    ``preserve_source_ids=True`` retains the input detector-row ordinal, so
    excluding an SWR or MUA leaves a gap without changing later detector IDs.
    Set it to ``False`` for Non-Local probability intervals, whose historical
    IDs were assigned only after the confidence/speed cohort was selected.
    """
    time_index = (
        time.copy()
        if isinstance(time, pd.Index)
        else pd.Index(np.asarray(time), name="time")
    )
    numbered_times = event_times.copy()
    numbered_times["_source_event_label"] = np.arange(
        1, len(numbered_times) + 1, dtype=int
    )
    eligible_times = filter_event_intervals_by_speed(
        numbered_times,
        speed=speed,
        time=time_index.to_numpy(),
        speed_threshold=speed_threshold,
    )

    # ``get_labels`` writes intervals in row order, so a later detector row
    # overwrites an earlier row on any shared sample. Zeroing a rejected later
    # label after that overwrite would then silently truncate an eligible
    # earlier event. Same-source events must therefore be mutually exclusive on
    # the sampled axis; fail before constructing any labels when that invariant
    # is violated (including intervals that share an inclusive endpoint bin).
    time_values = time_index.to_numpy(dtype=float)
    starts = np.searchsorted(
        time_values,
        numbered_times["start_time"].to_numpy(dtype=float),
        side="left",
    )
    ends = np.searchsorted(
        time_values,
        numbered_times["end_time"].to_numpy(dtype=float),
        side="right",
    )
    has_samples = starts < ends
    coverage_delta = np.zeros(len(time_values) + 1, dtype=int)
    np.add.at(coverage_delta, starts[has_samples], 1)
    np.add.at(coverage_delta, ends[has_samples], -1)
    coverage = np.cumsum(coverage_delta[:-1])
    collisions = np.flatnonzero(coverage > 1)
    if collisions.size:
        first_collision = int(collisions[0])
        source_ids = (
            np.flatnonzero(
                has_samples & (starts <= first_collision) & (ends > first_collision)
            )
            + 1
        )
        raise EventDefinitionError(
            f"{event_col} same-source event intervals overlap on the sampled "
            f"time axis at t={time_values[first_collision]:.9g}; source event "
            f"IDs {source_ids.tolist()} claim the same bin. Regenerate or "
            "repair the detector intervals before cohort construction."
        )

    if preserve_source_ids:
        eligible_labels = eligible_times["_source_event_label"].to_numpy(dtype=int)
        # Label the raw detector rows first, then atomically zero rejected
        # labels. This preserves source IDs (including gaps) when an earlier
        # detector event is excluded.
        labels = get_labels(event_times, time_index)["event_number"].rename(event_col)
        labels = labels.where(labels.isin(eligible_labels), 0).astype(int)
    else:
        # Non-Local IDs have historically numbered the accepted confidence
        # intervals, not all raw probability excursions. Label the eligible
        # rows directly so an unchanged physical cohort retains its established
        # IDs and event hashes.
        labels = get_labels(
            eligible_times.drop(columns="_source_event_label"),
            time_index,
        )["event_number"].rename(event_col)
        labels = labels.astype(int)
    canonical_intervals = intervals_from_labels(labels.to_frame(), event_col)
    canonical_intervals["duration"] = (
        canonical_intervals["end_time"] - canonical_intervals["start_time"]
    )
    return canonical_intervals, labels


__all__ = [
    "DEFAULT_EVENT_GROUP_COLS",
    "EventDefinitionError",
    "build_speed_eligible_event_cohort",
    "event_speeds_are_immobile",
    "filter_event_labels_by_speed",
    "filter_event_intervals_by_speed",
    "intervals_from_labels",
    "require_canonical_speed_threshold",
    "require_time_index",
    "validate_nonlocal_event_probability_contract",
    "validate_whole_event_immobility_contract",
    "whole_event_immobility_mask",
]
