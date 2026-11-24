"""The analyzed-time read point: a decode frame held to the one analyzed-time rule.

The SWR source itself (which inventory, which artifact run, which epochs it
serves, the source's name) is ``swr_source``. This module reads it and
applies the project's one analyzed-time rule (``eligibility``) to every cohort
of an analysis frame.

What this module fixes
----------------------
- **Excluded time** (:func:`excluded_intervals`): the epoch's artifact mask
  (``swr_source.artifact_intervals``) widened by
  ``eligibility.ARTIFACT_REACH_S`` on each side, the intervals every analysis
  consumer reads; :func:`excluded_intervals_of` and
  :func:`excluded_intervals_by_session` give them for a frame or a table.
- **The source's stamp**: every table built from these labels is stamped with
  the source's name (:func:`stamp_swr_source`), and a reader that combines a
  table's SWR columns with this source's live labels refuses a table stamped
  with another name (:func:`require_swr_source`).
- **Every cohort of an analysis frame** (:func:`with_swr_source`): every SWR,
  MUA and non-local event in excluded time is dropped whole, and so is a kept
  event that overlaps an SWR or MUA event dropped there; the other events keep
  their numbers, and NonLocal and ContFrag frames give the same SWR and MUA
  cohorts. :func:`artifact_event_drops` gives every event of a frame with its
  reason.
- **The per-bin labels** (:func:`swr_event_labels`): exactly the convention
  ``SWR_event`` has had since the unified immobility change. That is
  ``build_speed_eligible_event_cohort(..., preserve_source_ids=True)``, the call
  ``get_data_over_time`` makes: a bin is labeled with an event's number when
  its time lies in the event's closed interval ``[start_time, end_time]``; the
  number is the inventory's own ``event_number``, so a dropped event leaves a
  gap; and a dropped event's bins are 0.
- **Where detection was observed** (:func:`swr_observed_stretches`,
  :func:`swr_observed_mask`): the gap-free stretches of the stored ripple band
  that Kay read (``scope == "stored_band"`` in ``swr_observed.parquet``), less
  excluded time. Excluded time is not analyzed: an event there is neither an
  SWR nor a non-SWR event.

Analyses read a decode's ``data_over_time`` through :func:`with_swr_source`
or :func:`read_dot_with_swr_source`. Its ``SWR_event`` column then holds this
source's labels, ``swr_observed`` the coverage mask and ``in_excluded_time``
the rows in excluded time (a per-bin summary keeps the rows
:func:`analyzed_bins` returns); every event cohort of the frame follows the one
analyzed-time rule, and the frame's two epoch normalizers
(``theta_power_zscore``, ``run_speed_baseline``) leave the artifact mask out,
as the other inputs do. An HPC decode made by the
decoders since 2026-10-09 (commit ``b66f9d28``, the code of the October 2026
full re-decode; ``docs/design/2026-10-09-swr-source.md``) already holds the
source's SWR labels: its
``organize_results`` reads the frame through :func:`with_swr_source` (keeping
every event it decoded, ``drop_masked_events=False``), so the decode's
provenance is this source, and its encoding leaves out every event of this
inventory and the artifact mask (``decode_masks.hpc_decode_masks``). In an older
decode, or a PFC decode, ``SWR_event`` (from ``RippleTimesV1``, written by
``get_data_over_time``) stays in the pickle as the decode's provenance and never
reaches an analysis, and neither do the decode's stored-ripple counts
(:data:`DECODE_SWR_COUNT_COLUMNS`) or its other event counts
(:data:`DECODE_EVENT_COUNT_COLUMNS`).
"""

from __future__ import annotations

import hashlib
import io
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# Called through the module, not bound by name, so a patch on
# ``swr_source.artifact_intervals`` / ``swr_source_files`` /
# ``swr_source_name`` / ``epoch_source`` reaches every caller here too.
from continuum_swr_replay.analysis.event_definition import swr_source
from continuum_swr_replay.analysis.event_definition.eligibility import (
    EXCLUDED_TIME_REASONS,
    DropReason,
    drop_labels,
    excluded_time,
    label_drop_reasons,
    overlaps_dropped_events,
)
from continuum_swr_replay.analysis.event_definition.epoch_normalizers import (
    run_speed_baseline,
)
from continuum_swr_replay.analysis.event_definition.immobility import (
    build_speed_eligible_event_cohort,
    intervals_from_labels,
    require_time_index,
    validate_nonlocal_event_probability_contract,
    validate_whole_event_immobility_contract,
)
from continuum_swr_replay.analysis.event_definition.swr_source import (
    SWR_LABEL_COLUMN,
    SwrSourceError,
    swr_epoch_id_of_session,
)
from continuum_swr_replay.decoding.identifiers import parse_nwb_filename
from continuum_swr_replay.intervals import as_frame, interval_mask
from continuum_swr_replay.parameters import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    DEFAULT_NON_LOCAL_THRESHOLD,
)
from continuum_swr_replay.tables.columns import require_columns
from continuum_swr_replay.tables.io import (
    SWR_SOURCE_ATTR,
    publish_canonical,
    read_table,
)
from continuum_swr_replay.tables.session_keys import (
    SESSION_COLUMNS,
    SESSION_COLUMNS_DAY,
    one_session,
)

logger = logging.getLogger(__name__)

__all__ = [
    "read_swr_table",
    "DECODE_EVENT_COUNT_COLUMNS",
    "DECODE_SWR_COUNT_COLUMNS",
    "IN_EXCLUDED_TIME_COLUMN",
    "SWR_OBSERVED_COLUMN",
    "analyzed_bins",
    "artifact_event_drops",
    "dropped_for_excluded_time",
    "dropped_on_grid",
    "excluded_intervals",
    "publish_swr_table",
    "excluded_intervals_by_session",
    "excluded_intervals_of",
    "immobile_analyzed_bins",
    "read_artifact_event_drops",
    "read_dot_with_swr_source",
    "read_verified_dot_with_swr_source",
    "require_swr_source",
    "sampled_bounds",
    "stamp_swr_source",
    "swr_event_labels",
    "swr_intervals",
    "swr_observed_mask",
    "swr_observed_stretches",
    "with_swr_source",
]


SWR_OBSERVED_COLUMN = "swr_observed"
"""The per-bin coverage column :func:`with_swr_source` adds: True where SWR
detection was observed."""


IN_EXCLUDED_TIME_COLUMN = "in_excluded_time"
"""The per-bin column :func:`with_swr_source` adds: True where the row's time
lies in the epoch's excluded time (:func:`excluded_intervals`, closed), all
False with no artifact run. A reader that summarizes bins leaves these rows out.
It is not ``swr_observed``, which is also False at gaps in the stored band."""


DECODE_SWR_COUNT_COLUMNS = (
    "n_SWR",
    "n_non_local_SWR",
    "n_non_local_no_SWR",
    "n_SWR_MUA",
    "n_SWR_no_MUA",
)
"""The decode's stored-ripple counts in ``data_over_time``; dropped from every
analysis frame, as the decode's own ``SWR_event`` labels are replaced."""


DECODE_EVENT_COUNT_COLUMNS = (
    "n_non_local",
    "n_MUA",
    "n_non_local_MUA",
    "n_non_local_no_MUA",
)
"""The decode's other event counts in ``data_over_time`` (one number per epoch).
They count every event the decode labeled, the ones the analyzed-time rule
drops too, so they are dropped from every analysis frame as well. No reader in
``src/`` or ``scripts/`` used them (checked 2026-10-11)."""


_THETA_COLUMN = "theta_power_zscore"


_RUN_SPEED_BASELINE_COLUMN = "run_speed_baseline"


LABEL_COLUMNS = ("non_local_event", "MUA_event")
"""A decode frame's own event labels besides ``SWR_event``; each is held to the
one rule where the frame is read."""


OVERLAP_SOURCES = (SWR_LABEL_COLUMN, "MUA_event")
"""The cohorts whose events dropped for excluded time drop the kept events that
overlap them (``overlaps_dropped_event``)."""


DROP_COLUMNS = [
    "event_col",
    "label",
    "start_time",
    "end_time",
    "sampled_start_time",
    "sampled_end_time",
    "speed_eligible",
    "drop_reason",
]
"""The columns of :func:`artifact_event_drops`."""


def stamp_swr_source(frame: pd.DataFrame) -> pd.DataFrame:
    """Stamp `frame` (in place, and return it) as built from this source.

    Only a producer that built the frame from decode frames read through this
    source stamps it; :func:`publish_swr_table` and the table readers refuse a
    table of ``tables.io.SWR_SOURCE_TABLES`` without the current stamp.
    """
    frame.attrs[SWR_SOURCE_ATTR] = swr_source.swr_source_name()
    return frame


def require_swr_source(frame: pd.DataFrame, *, what: str) -> None:
    """Refuse a frame or table that was not built from the current source.

    Parameters
    ----------
    frame : pd.DataFrame
        A table as read (``read_table`` restores its ``attrs``).
    what : str
        What it is, for the message (a path, say).

    Raises
    ------
    SwrSourceError
        If its stamp is missing or names another source.
    """
    stamp = frame.attrs.get(SWR_SOURCE_ATTR)
    if stamp != swr_source.swr_source_name():
        raise SwrSourceError(
            f"{what} was built from SWR source {stamp!r}, not the configured "
            f"{swr_source.swr_source_name()!r}; regenerate it (and its atomic "
            "set) from the current source"
        )


def read_swr_table(
    path: str | Path,
    columns: list[str] | None = None,
    filters: list[tuple[str, str, Any]] | None = None,
) -> pd.DataFrame:
    """Read a canonical table and refuse one built from another SWR source.

    The one way to read a table of ``tables.io.SWR_SOURCE_TABLES``:
    ``tables.io.read_table`` (which restores the stamp) followed by
    :func:`require_swr_source`. ``pd.read_parquet`` drops the stamp, so a table
    read with it cannot be checked.

    Parameters
    ----------
    path : str or Path
        The table's path (``paths.table_path``).
    columns : list of str, optional
        Read only these columns.
    filters : list of (column, operator, value), optional
        Read only the rows that pass.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    SwrSourceError
        If the table's stamp is missing or names another source.
    """
    frame = read_table(path, columns=columns, filters=filters)
    require_swr_source(frame, what=str(path))
    return frame


def publish_swr_table(
    frame: pd.DataFrame,
    path: str | Path,
    *,
    table: str,
    basename: str | None = None,
) -> Path:
    """Publish a canonical table built from this source's SWR labels.

    :func:`require_swr_source` first (the frame must carry this source's
    current stamp), then ``tables.io.publish_canonical`` stamps the covariate
    schema and writes it atomically.

    Parameters
    ----------
    frame : pd.DataFrame
    path : str or Path
        The directory with `basename`, or the file without.
    table : str
        The covariate-schema key, one of ``tables.io.SWR_SOURCE_TABLES``.
    basename : str, optional

    Raises
    ------
    SwrSourceError
        If the frame's stamp is missing or names another source; nothing is
        written.
    """
    require_swr_source(frame, what=f"the {table} table to publish")
    return publish_canonical(frame, path, table=table, basename=basename)


def excluded_intervals(epoch_id: str) -> pd.DataFrame:
    """The epoch's excluded time: :func:`artifact_intervals` widened by
    ``eligibility.ARTIFACT_REACH_S`` on each side, merged.

    The one function every analysis consumer reads: an event, an exposure
    second, a time bin with its history window and a peri-event window are
    analyzed only wholly outside these intervals. Inputs (the decoders' missing
    bins, the SWR detectors, the ripple-power covariate) read the mask itself.

    Parameters
    ----------
    epoch_id : str

    Returns
    -------
    pd.DataFrame
        ``start_time`` and ``end_time`` (seconds), closed, disjoint and in time
        order; no rows while ``swr_source.ARTIFACT_INTERVALS_RUN_ID`` is unset.

    Raises
    ------
    SwrSourceError
        As :func:`artifact_intervals`.
    """
    return excluded_time(swr_source.artifact_intervals(epoch_id))


def swr_observed_stretches(epoch_id: str) -> NDArray[np.float64]:
    """Where SWR detection was observed and is analyzed: the stored band's
    gap-free stretches, less the epoch's :func:`excluded_intervals`.

    Excluded time is not analyzed. Next to an excluded interval a stretch ends
    (or starts) at the float just outside it, so a time lies in a stretch
    exactly when it was observed and lies in no excluded interval.

    Returns
    -------
    ndarray, shape (n_stretches, 2)
        First and last time (seconds) of each stretch, closed.
    """
    return swr_source.epoch_source(epoch_id).observed


def swr_observed_mask(time: NDArray | pd.Index, epoch_id: str) -> NDArray[np.bool_]:
    """True at each time that lies in one of the observed stretches (closed):
    detection was observed there, and it is in no excluded interval.

    Parameters
    ----------
    time : array-like, shape (n_time,)
        Seconds, such as a decode grid.
    epoch_id : str
    """
    return interval_mask(time, as_frame(swr_observed_stretches(epoch_id)))


def _speed_eligible_labels(
    events: pd.DataFrame,
    time_index: pd.Index,
    speed_values: NDArray,
    speed_threshold: float,
) -> NDArray[np.int64]:
    """Every speed-eligible event's label on the grid, the ones in excluded time
    too (the builders' convention, ``build_speed_eligible_event_cohort``)."""
    _, labels = build_speed_eligible_event_cohort(
        events[["start_time", "end_time"]],
        speed=speed_values,
        time=time_index,
        event_col=SWR_LABEL_COLUMN,
        speed_threshold=speed_threshold,
    )
    eligible: NDArray[np.int64] = labels.to_numpy(dtype=np.int64)
    return eligible


def sampled_bounds(
    events: pd.DataFrame,
    time: NDArray | pd.Index,
    speed: NDArray,
    speed_threshold: float = DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Each event's labels on a decode grid and its sampled bounds there.

    The events are labeled by their row ordinals (1..n) on the grid, as
    ``build_speed_eligible_event_cohort`` labels them (the ones in excluded
    time too); an event's sampled bounds are its first and last labeled
    sample. The rule's second pass (``overlaps_dropped_event``) reads every
    event at these bounds.

    Parameters
    ----------
    events : pd.DataFrame
        ``start_time`` and ``end_time`` (seconds), one row per event.
    time : array-like, shape (n_time,)
        The grid (seconds), strictly increasing.
    speed : ndarray, shape (n_time,)
        Head speed on `time` (cm/s).
    speed_threshold : float
        cm/s, strict.

    Returns
    -------
    labels : ndarray of int, shape (n_time,)
        The row ordinal of the event each sample lies in, 0 outside the events
        the speed rule keeps.
    bounds : ndarray, shape (n_events, 2)
        First and last labeled sample (seconds) of each event; NaN for an
        event the speed rule rejects.

    Raises
    ------
    EventDefinitionError
        If two events claim one sample.
    """
    time_index = pd.Index(np.asarray(time, dtype=float), name="time")
    labels = _speed_eligible_labels(
        events, time_index, np.asarray(speed, dtype=float), speed_threshold
    )
    sampled = intervals_from_labels(
        pd.DataFrame({"label": labels}, index=time_index), "label"
    )
    bounds = np.full((len(events), 2), np.nan)
    bounds[sampled["label"].to_numpy(dtype=int) - 1] = sampled[
        ["start_time", "end_time"]
    ].to_numpy(dtype=np.float64)
    return labels, bounds


def _keep_labels(labels: NDArray[np.int64], kept: NDArray) -> NDArray[np.int64]:
    kept_labels: NDArray[np.int64] = np.where(np.isin(labels, kept), labels, 0)
    return kept_labels


def swr_event_labels(
    time: NDArray | pd.Index,
    epoch_id: str,
    *,
    speed: NDArray,
    speed_threshold: float = DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
) -> NDArray[np.int64]:
    """The per-bin ``SWR_event`` labels on a decode time grid, by the SWR
    cohort's own rule (``swr_source.swr_events``).

    The convention ``get_data_over_time`` has written since the unified
    immobility change, and the table builders read:
    ``build_speed_eligible_event_cohort(events, speed=speed, time=time,
    event_col="SWR_event")``, here on the inventory's events, with the bins of
    an event dropped for excluded time also set to 0. A bin is labeled with
    the event's inventory number when its time lies in ``[start_time,
    end_time]``. A frame read through :func:`with_swr_source` also drops the
    SWRs that overlap an MUA event dropped for excluded time.

    Parameters
    ----------
    time : array-like, shape (n_time,)
        The grid (seconds), strictly increasing.
    epoch_id : str
    speed : array-like, shape (n_time,)
        Head speed on `time` (cm/s).
    speed_threshold : float
        As ``swr_source.swr_events``.

    Returns
    -------
    ndarray of int, shape (n_time,)
        0 outside the kept events.

    Raises
    ------
    SwrSourceError
        If the epoch cannot be served.
    EventDefinitionError
        If two events claim one bin (the builders' same-source check).
    """
    time_index = (
        time.copy()
        if isinstance(time, pd.Index)
        else pd.Index(np.asarray(time, dtype=float), name="time")
    )
    speed_values = np.asarray(speed, dtype=float)
    events = swr_source.classify(
        swr_source.epoch_source(epoch_id),
        pd.Series(speed_values, index=time_index),
        speed_threshold,
    )
    labels = _speed_eligible_labels(events, time_index, speed_values, speed_threshold)
    kept = events.loc[events["drop_reason"].isna(), SWR_LABEL_COLUMN].to_numpy()
    return _keep_labels(labels, kept)


def _frame_epoch(
    dot: pd.DataFrame,
    what: str,
    non_local_threshold: float = DEFAULT_NON_LOCAL_THRESHOLD,
) -> tuple[str, NDArray]:
    """The one epoch a frame holds, as the inventory's id, and its time axis.

    The time axis is the index, read nowhere else: every reader downstream
    (``intervals_from_labels``, the event tables) takes time from the index,
    and an event's first and last row as its bounds, so the index must be a
    float time axis that strictly increases
    (``immobility.require_time_index``). A ``time`` column would be a
    second axis and is refused.

    The decode's own labels (:data:`LABEL_COLUMNS`), which the read passes on,
    are checked here too, once for every consumer: each event is immobile as a
    whole (``immobility``, the canonical threshold), and each non-local
    event's bins have a summed Non-Local posterior above
    `non_local_threshold`. The decoder made them so, so a frame that breaks
    either is stale and is refused, never clipped. ``SWR_event`` is the
    source's own and its speed rule is a drop (``moving``), in :func:`_drops`.
    """
    require_columns(
        dot, [*SESSION_COLUMNS_DAY, "head_speed"], f"{what}: data_over_time"
    )
    animal, day, epoch = one_session(dot, what)
    require_time_index(dot, what)
    if "non_local_event" in dot.columns:
        validate_nonlocal_event_probability_contract(
            dot, probability_threshold=non_local_threshold
        )
    for column in LABEL_COLUMNS:
        if column in dot.columns:
            validate_whole_event_immobility_contract(dot, column)
    time = dot.index.to_numpy()
    return swr_epoch_id_of_session(animal, day, epoch), np.asarray(time, float)


def _drops(
    source: swr_source.EpochSource, dot: pd.DataFrame, time: NDArray
) -> tuple[pd.DataFrame, NDArray[np.int64]]:
    """Every event of the frame with its reason, and the speed-eligible SWR
    labels on its grid.

    The SWR rows are the inventory's (its bounds); the label columns' rows are
    the frame's (sampled bounds). The overlap pass reads every event at its
    sampled bounds on the frame's grid (``sampled_start_time``,
    ``sampled_end_time``).
    """
    speed_values = dot["head_speed"].to_numpy(dtype=float)
    swrs = swr_source.classify(
        source,
        pd.Series(speed_values, index=pd.Index(time, name="time")),
        DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    )
    # the inventory numbers its events 1..n in row order, so a row's ordinal
    # is its SWR_event label
    swr_labels, swr_bounds = sampled_bounds(swrs, time, speed_values)
    parts = [
        pd.DataFrame(
            {
                "event_col": SWR_LABEL_COLUMN,
                "label": swrs[SWR_LABEL_COLUMN].to_numpy(),
                "start_time": swrs["start_time"].to_numpy(dtype=float),
                "end_time": swrs["end_time"].to_numpy(dtype=float),
                "sampled_start_time": swr_bounds[:, 0],
                "sampled_end_time": swr_bounds[:, 1],
                "speed_eligible": swrs["speed_eligible"].to_numpy(dtype=bool),
                "drop_reason": swrs["drop_reason"].to_numpy(),
            }
        )
    ]
    for column in LABEL_COLUMNS:
        if column in dot.columns:
            labeled = label_drop_reasons(dot, column, source.artifacts)
            parts.append(
                labeled.rename(columns={column: "label"}).assign(
                    event_col=column,
                    sampled_start_time=labeled["start_time"],
                    sampled_end_time=labeled["end_time"],
                    speed_eligible=True,
                )[DROP_COLUMNS]
            )
    drops = pd.concat(parts, ignore_index=True)[DROP_COLUMNS]

    # One pass: a kept event that overlaps an SWR or MUA event dropped for
    # excluded time (and eligible but for it) has an unknown overlap class.
    dropped = dropped_on_grid(drops)
    kept = drops["drop_reason"].isna().to_numpy() & np.isfinite(
        drops["sampled_start_time"].to_numpy(dtype=float)
    )
    if len(dropped) and kept.any():
        overlapping = overlaps_dropped_events(
            drops.loc[kept, "sampled_start_time"].to_numpy(dtype=float),
            drops.loc[kept, "sampled_end_time"].to_numpy(dtype=float),
            dropped,
        )
        rows = np.flatnonzero(kept)[overlapping]
        drops.loc[rows, "drop_reason"] = DropReason.OVERLAPS_DROPPED_EVENT.value
    return drops, swr_labels


def artifact_event_drops(
    dot: pd.DataFrame, *, non_local_threshold: float = DEFAULT_NON_LOCAL_THRESHOLD
) -> pd.DataFrame:
    """Every event of one epoch's decode frame, with its reason to leave.

    The reasons :func:`with_swr_source` acts on, the one rule
    (``eligibility``) on every cohort:

    - the source's SWRs (``swr_source.swr_events``: excluded time, then speed);
    - each event of the frame's ``non_local_event`` and ``MUA_event`` labels,
      read at its sampled bounds, with excluded time's reasons (the labels were
      speed-qualified when the decode was made);
    - then one pass: a kept event of any of them that overlaps an SWR or MUA
      event dropped for excluded time (one the speed rule keeps) is
      ``overlaps_dropped_event``.

    NonLocal and ContFrag frames of one epoch therefore give the same SWR and
    MUA cohorts.

    Parameters
    ----------
    dot : pd.DataFrame
        One epoch's decode frame as the decoder saved it, indexed by
        increasing time (no ``time`` column), carrying ``animal``, ``day``,
        ``epoch`` and ``head_speed``.
    non_local_threshold : float
        The threshold the frame was decoded at; its non-local labels are
        checked against it, as :func:`with_swr_source` checks them.

    Returns
    -------
    pd.DataFrame
        Columns :data:`DROP_COLUMNS`: the label column, the event's label, its
        bounds (seconds; an SWR's are the inventory's), its sampled bounds on
        the frame's grid (``sampled_start_time``, ``sampled_end_time``: an
        SWR's first and last labeled bin, :func:`sampled_bounds`, NaN for one
        the speed rule rejects; a labeled event's are its bounds), whether the
        speed rule alone keeps it, and its reason (None for a kept event); SWR
        events first, then each label column, in label order.

    Raises
    ------
    ValueError
        If the frame holds more than one epoch, lacks a needed column, has a
        ``time`` column, or its index is not a float time axis that strictly
        increases (``immobility.require_time_index``).
    EventDefinitionError
        If a ``non_local_event`` or ``MUA_event`` label breaks the whole-event
        immobility rule, or a non-local label has a bin not above
        `non_local_threshold` (:func:`with_swr_source`).
    SwrSourceError
        If the epoch cannot be served.
    """
    epoch_id, time = _frame_epoch(dot, "artifact_event_drops", non_local_threshold)
    drops, _ = _drops(swr_source.epoch_source(epoch_id), dot, time)
    return drops


def read_artifact_event_drops(
    path: str | Path, *, non_local_threshold: float = DEFAULT_NON_LOCAL_THRESHOLD
) -> pd.DataFrame:
    """:func:`artifact_event_drops` of a decode's ``data_over_time`` pickle, as
    the decoder saved it: for a consumer that needs the dropped events
    themselves (an exposure count, a control window, a re-detected cohort).
    `non_local_threshold` is the frame's, as for :func:`read_dot_with_swr_source`."""
    frame: Any = pd.read_pickle(path)
    return artifact_event_drops(frame, non_local_threshold=non_local_threshold)


def dropped_for_excluded_time(drops: pd.DataFrame) -> pd.DataFrame:
    """The SWR and MUA events dropped for excluded time that the speed rule
    keeps: the events a kept event's overlap is checked against.

    Parameters
    ----------
    drops : pd.DataFrame
        From :func:`artifact_event_drops`.

    Returns
    -------
    pd.DataFrame
        Its rows for those events, ``start_time`` and ``end_time`` among them.
    """
    selected = (
        drops["event_col"].isin(OVERLAP_SOURCES)
        & drops["drop_reason"].isin(EXCLUDED_TIME_REASONS)
        & drops["speed_eligible"]
    )
    return drops.loc[selected]


def dropped_on_grid(drops: pd.DataFrame) -> pd.DataFrame:
    """:func:`dropped_for_excluded_time` at its events' sampled bounds: the
    intervals the rule's second pass (``overlaps_dropped_event``) checks a kept
    event against.

    The speed rule keeps each of these events, so each has sampled bounds on
    the grid.

    Parameters
    ----------
    drops : pd.DataFrame
        From :func:`artifact_event_drops`.

    Returns
    -------
    pd.DataFrame
        Its rows for those events, ``start_time`` and ``end_time`` set to
        ``sampled_start_time`` and ``sampled_end_time``.
    """
    dropped = dropped_for_excluded_time(drops)
    return dropped.assign(
        start_time=dropped["sampled_start_time"],
        end_time=dropped["sampled_end_time"],
    )


def _normalize_outside_mask(frame: pd.DataFrame, in_mask: NDArray[np.bool_]) -> None:
    """Take the frame's two epoch normalizers again without the artifact mask.

    Both are inputs, so they leave out the mask itself (not excluded time), as
    ``SWR_zscore`` and ``MUA_zscore`` already do. ``frame`` is changed in place:
    each column is replaced by a new array, so a shallow copy's source keeps
    its own.

    - ``theta_power_zscore``: the decode z-scored theta power over every bin.
      It is re-z-scored over the rows outside the mask (NaN ignored, SD with
      ``ddof=0`` as ``scipy.stats.zscore``) and set to NaN inside it. A z-score
      is affine in raw power, so this equals z-scoring the raw power over the
      rows outside the mask, and a frame already renormalized is unchanged (to
      float rounding).
    - ``run_speed_baseline``: ``epoch_normalizers.run_speed_baseline`` of the
      head speed outside the mask, the percentile the decoders took over every
      bin.

    A frame without a column (a rule frame, say) is left without it.
    """
    if _THETA_COLUMN in frame.columns:
        theta = frame[_THETA_COLUMN].to_numpy(dtype=float)
        outside = theta[~in_mask]
        renormalized = (theta - np.nanmean(outside)) / np.nanstd(outside)
        renormalized[in_mask] = np.nan
        frame[_THETA_COLUMN] = renormalized
    if _RUN_SPEED_BASELINE_COLUMN in frame.columns:
        speed = frame["head_speed"].to_numpy(dtype=float)
        frame[_RUN_SPEED_BASELINE_COLUMN] = run_speed_baseline(speed[~in_mask])


def with_swr_source(
    dot: pd.DataFrame,
    *,
    drop_masked_events: bool = True,
    non_local_threshold: float = DEFAULT_NON_LOCAL_THRESHOLD,
) -> pd.DataFrame:
    """One epoch's ``data_over_time`` with its SWR labels from this source,
    and every event cohort held to the one analyzed-time rule.

    Parameters
    ----------
    dot : pd.DataFrame
        One epoch's decode frame, indexed by increasing time (no ``time``
        column), carrying ``animal``, ``day``, ``epoch`` and ``head_speed``.
    drop_masked_events : bool
        True (every analysis read): every event the rule drops
        (:func:`artifact_event_drops`) has its label zeroed whole. False only
        for a decoder saving its own frame, which keeps every event it decoded
        (its SWR labels still leave out the SWRs in the mask, as the inventory
        does).
    non_local_threshold : float
        The posterior threshold the frame's ``non_local_event`` labels were
        made at (the ``_80_`` in a decode's file name is 0.8). Every labeled
        bin's summed Non-Local posterior must be above it.

    Returns
    -------
    pd.DataFrame
        A new frame (`dot` is not changed) with the same rows: ``SWR_event``
        holds the kept SWRs' labels on the frame's time axis and head speed
        (the inventory's numbers), ``swr_observed`` is
        :func:`swr_observed_mask` (excluded time unobserved), and ``attrs``
        carries the source's stamp (:func:`stamp_swr_source`). The epoch's
        source files are checked against the manifest once. In both modes
        also:

        - ``in_excluded_time`` (:data:`IN_EXCLUDED_TIME_COLUMN`) is True at
          each row whose time lies in :func:`excluded_intervals` (closed), all
          False with no artifact run;
        - the decode's event counts are gone: the stored-ripple counts
          (:data:`DECODE_SWR_COUNT_COLUMNS`) and the others
          (:data:`DECODE_EVENT_COUNT_COLUMNS`), which count events the rule
          drops;
        - when the epoch has artifacts, the two epoch normalizers leave the
          artifact mask out, as inputs do: ``theta_power_zscore`` is
          re-z-scored over the rows outside the mask and NaN inside it, and
          ``run_speed_baseline`` is the same percentile of head speed over the
          rows outside the mask. Applying this to a frame already read (or
          saved by a decoder through this function) changes nothing (theta to
          float rounding).

        With no artifact run configured, every other label and column is the
        decode's.

        Every event label it hands out holds whole events the speed rule
        admits (every sampled speed finite and below 4 cm/s), and every
        non-local label's bins are above `non_local_threshold`, so a consumer
        neither re-filters nor re-validates them.

    Raises
    ------
    ValueError
        If the frame holds more than one epoch, lacks a needed column, has a
        ``time`` column, or its index is not a float time axis that strictly
        increases (``immobility.require_time_index``).
    EventDefinitionError
        If a ``non_local_event`` or ``MUA_event`` label breaks the whole-event
        immobility rule, or a ``non_local_event`` label has a bin whose summed
        Non-Local posterior is not finite and above `non_local_threshold`: a
        stale frame, to regenerate.
    SwrSourceError
        If the epoch cannot be served.
    """
    epoch_id, time = _frame_epoch(dot, "with_swr_source", non_local_threshold)
    source = swr_source.epoch_source(epoch_id)
    masking = drop_masked_events and len(source.artifacts) > 0
    drops, swr_labels = _drops(source, dot, time)
    if masking:
        kept_swrs = drops.loc[
            (drops["event_col"] == SWR_LABEL_COLUMN) & drops["drop_reason"].isna(),
            "label",
        ].to_numpy()
    else:
        # the decoder's own frame: only the inventory's drop (the mask itself)
        kept_swrs = drops.loc[
            (drops["event_col"] == SWR_LABEL_COLUMN)
            & (drops["drop_reason"] != DropReason.ARTIFACT.value),
            "label",
        ].to_numpy()
    # A shallow copy shares the untouched columns of a frame that can be half a
    # gigabyte; deleting and assigning columns on it leaves `dot` as it was.
    out = dot.copy(deep=False)
    out.attrs = dict(dot.attrs)
    for column in (*DECODE_SWR_COUNT_COLUMNS, *DECODE_EVENT_COUNT_COLUMNS):
        if column in out.columns:
            del out[column]
    out[SWR_LABEL_COLUMN] = _keep_labels(swr_labels, kept_swrs)
    out[SWR_OBSERVED_COLUMN] = interval_mask(time, as_frame(source.observed))
    out[IN_EXCLUDED_TIME_COLUMN] = interval_mask(time, excluded_time(source.artifacts))
    if len(source.artifacts) > 0:
        _normalize_outside_mask(out, interval_mask(time, source.artifacts))
    if masking:
        for column in LABEL_COLUMNS:
            if column in out.columns:
                rows = drops.loc[drops["event_col"] == column]
                out = drop_labels(out, column, rows.rename(columns={"label": column}))
        counts = (
            drops.dropna(subset=["drop_reason"])
            .groupby(["event_col", "drop_reason"])
            .size()
            .to_dict()
        )
        if counts:
            logger.info("%s: dropped %s", epoch_id, counts)
    return stamp_swr_source(out)


def _session_epoch_ids(frame: pd.DataFrame, what: str) -> list[tuple]:
    date = "session_date" if "session_date" in frame.columns else "day"
    columns = ["animal", date, "epoch"]
    require_columns(frame, columns, what)
    return list(frame[columns].drop_duplicates().itertuples(index=False))


def excluded_intervals_of(frame: pd.DataFrame) -> pd.DataFrame:
    """:func:`excluded_intervals` of the one epoch a frame or table holds.

    Parameters
    ----------
    frame : pd.DataFrame
        Carrying ``animal``, ``epoch`` and the date as ``day`` or
        ``session_date``; one epoch.

    Returns
    -------
    pd.DataFrame
        As :func:`excluded_intervals`.

    Raises
    ------
    ValueError
        If the frame holds more than one epoch, or lacks the session columns.
    SwrSourceError
        As :func:`artifact_intervals`.
    """
    sessions = _session_epoch_ids(frame, "excluded_intervals_of")
    if len(sessions) != 1:
        raise ValueError(
            f"excluded_intervals_of reads one epoch at a time, not {len(sessions)}"
        )
    animal, day, epoch = sessions[0]
    return excluded_intervals(swr_epoch_id_of_session(animal, day, epoch))


def excluded_intervals_by_session(sessions: pd.DataFrame) -> pd.DataFrame:
    """:func:`excluded_intervals` of every session a table holds, in one frame.

    Parameters
    ----------
    sessions : pd.DataFrame
        Carrying ``animal``, ``session_date`` and ``epoch`` (rows may repeat).

    Returns
    -------
    pd.DataFrame
        Columns ``animal``, ``session_date``, ``epoch``, ``start_time`` and
        ``end_time`` (seconds, closed): each session's excluded time, one row
        per interval, none for a session without artifacts.

    Raises
    ------
    SwrSourceError
        As :func:`artifact_intervals`.
    """
    keys = sessions[list(SESSION_COLUMNS)].drop_duplicates()
    parts = [
        excluded_intervals(swr_epoch_id_of_session(animal, day, epoch)).assign(
            animal=animal, session_date=day, epoch=epoch
        )
        for animal, day, epoch in keys.itertuples(index=False)
    ]
    columns = [*SESSION_COLUMNS, "start_time", "end_time"]
    if not parts:
        return pd.DataFrame({column: [] for column in columns}).astype(
            {"start_time": float, "end_time": float}
        )
    return pd.concat(parts, ignore_index=True)[columns]


def read_dot_with_swr_source(
    path: str | Path, *, non_local_threshold: float = DEFAULT_NON_LOCAL_THRESHOLD
) -> pd.DataFrame:
    """Read a decode's ``data_over_time`` pickle for analysis.

    :func:`with_swr_source` on the pickle: every analysis reader of a decode
    frame goes through here, so none sees the decode's stored-ripple labels.
    `non_local_threshold` is the one the file was decoded at (its name's
    ``_80_``), as :func:`with_swr_source`.
    """
    frame: Any = pd.read_pickle(path)
    return with_swr_source(frame, non_local_threshold=non_local_threshold)


def read_verified_dot(path: str | Path, expected_sha256: str) -> pd.DataFrame:
    """A decode's frame as the decoder saved it (not read through the SWR
    source), if its bytes are the run's.

    The file is read once; the same bytes are hashed and unpickled, so the frame
    is the one the hash vouches for.

    Parameters
    ----------
    path : str or Path
        A decode's ``data_over_time`` pickle.
    expected_sha256 : str
        The hex sha256 the run that used the decode recorded for it.

    Raises
    ------
    ValueError
        The file's sha256 is not `expected_sha256`.
    """
    raw = Path(path).read_bytes()
    sha256 = hashlib.sha256(raw).hexdigest()
    if sha256 != expected_sha256:
        raise ValueError(
            f"{Path(path).name}: decode sha256 {sha256} is not the run's "
            f"{expected_sha256}"
        )
    frame: Any = pickle.load(io.BytesIO(raw))
    return frame


def read_verified_dot_with_swr_source(
    path: str | Path,
    expected_sha256: str,
) -> pd.DataFrame:
    """:func:`read_dot_with_swr_source` of a decode whose bytes are the run's
    (:func:`read_verified_dot`).

    Raises
    ------
    ValueError
        The file's sha256 is not `expected_sha256`.
    """
    return with_swr_source(read_verified_dot(path, expected_sha256))


def analyzed_bins(frame: pd.DataFrame) -> NDArray[np.bool_]:
    """The rows of a read frame that lie outside excluded time.

    The one check of :data:`IN_EXCLUDED_TIME_COLUMN` that every per-bin summary
    uses (the quality metrics, the figure panels, the content scan). It sits
    beside the column's writer, :func:`with_swr_source`, so the library and the
    scripts import it from the module that defines the column.

    Parameters
    ----------
    frame : pd.DataFrame
        A frame read through this source (:func:`read_dot_with_swr_source`,
        or ``load_data_over_time``), which carries the column.

    Returns
    -------
    np.ndarray of bool, shape (n_rows,)
        True where the row is analyzed: ``~in_excluded_time``.

    Raises
    ------
    ValueError
        If the frame has no ``in_excluded_time`` column.
    TypeError
        If the column is not boolean. There is no coercion: cast to bool, a NaN
        (left by a concat with a frame read another way) would read as excluded
        and a None as analyzed.
    """
    require_columns(
        frame,
        [IN_EXCLUDED_TIME_COLUMN],
        "the frame (read it through analyzed_time.read_dot_with_swr_source or "
        "load_data_over_time, which mark excluded time)",
    )
    column = frame[IN_EXCLUDED_TIME_COLUMN]
    if column.dtype != bool:
        raise TypeError(
            f"{IN_EXCLUDED_TIME_COLUMN!r} must be boolean, not {column.dtype}: "
            "read every frame through analyzed_time.read_dot_with_swr_source"
        )
    return ~column.to_numpy()


def immobile_analyzed_bins(
    frame: pd.DataFrame,
    speed_threshold: float = DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
) -> NDArray[np.bool_]:
    """The rows of a read frame that are immobile and outside excluded time.

    Immobile is ``head_speed`` strictly below the threshold, the project's
    immobility rule for a bin: a bin at the threshold, or without a speed, is
    not immobile.

    Parameters
    ----------
    frame : pd.DataFrame
        A frame read through this source, with ``head_speed`` and
        :data:`IN_EXCLUDED_TIME_COLUMN`.
    speed_threshold : float, optional
        Speed (cm/s) below which a bin is immobile.

    Returns
    -------
    np.ndarray of bool, shape (n_rows,)
        True where the row is immobile and analyzed.

    Raises
    ------
    ValueError, TypeError
        As :func:`analyzed_bins`.
    """
    return (frame["head_speed"].to_numpy() < speed_threshold) & analyzed_bins(frame)


def rest_bins(frame: pd.DataFrame, in_visit: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """The rows of a read frame that are immobile and analyzed
    (:func:`immobile_analyzed_bins`) and inside a well visit (``in_visit``,
    ``event_definition.well_visits.in_well_visit`` on the frame's times): the
    time the counts by signal and by probability are over.

    Raises
    ------
    ValueError
        If ``in_visit`` is not one bool per row.
    """
    in_visit = np.asarray(in_visit)
    if in_visit.dtype != bool or in_visit.shape != (len(frame),):
        raise ValueError(
            f"in_visit must be one bool per bin, shape ({len(frame)},); got "
            f"{in_visit.dtype} of shape {in_visit.shape}"
        )
    return immobile_analyzed_bins(frame) & in_visit


def swr_intervals(
    dot: pd.DataFrame,
    *,
    nwb_file_name: str | None = None,
    epoch_name: str | None = None,
) -> pd.DataFrame:
    """One epoch's SWR intervals from a frame read through this source.

    The sampled bounds of the frame's ``SWR_event`` labels
    (:func:`~continuum_swr_replay.analysis.event_definition.immobility.intervals_from_labels`).
    The labels are already speed-gated and artifact-masked, so these are the
    intervals the event table classifies against; for plots and example
    selection, which must not re-gate on another position frame.

    Parameters
    ----------
    dot : pd.DataFrame
        A frame from :func:`with_swr_source`, :func:`read_dot_with_swr_source`
        or a pooled loader built on them.
    nwb_file_name, epoch_name : str, optional
        The epoch, when `dot` pools several.

    Returns
    -------
    pd.DataFrame
        Columns ``SWR_event``, ``start_time``, ``end_time``.

    Raises
    ------
    ValueError
        If `dot` was not read through the source (no ``swr_observed``), or the
        selection does not hold exactly one epoch.
    """
    require_columns(
        dot,
        [SWR_OBSERVED_COLUMN],
        "swr_intervals's frame (read it through read_dot_with_swr_source)",
    )
    frame = dot
    if nwb_file_name is not None or epoch_name is not None:
        if nwb_file_name is None or epoch_name is None:
            raise ValueError("give both nwb_file_name and epoch_name, or neither")
        animal, day = parse_nwb_filename(nwb_file_name)
        is_epoch = (
            dot["animal"].eq(animal) & dot["day"].eq(day) & dot["epoch"].eq(epoch_name)
        ).to_numpy()
        frame = dot.loc[is_epoch]
    n_sessions = len(frame[list(SESSION_COLUMNS_DAY)].drop_duplicates())
    if n_sessions != 1:
        raise ValueError(f"swr_intervals needs exactly one epoch, not {n_sessions}")
    return intervals_from_labels(frame, SWR_LABEL_COLUMN)
