"""The paper's one SWR source: the direct ripple_detection 2.0 Kay inventory.

The project lead decided on 2026-10-05 that every SWR label of the paper
(``is_swr``, ``SWR_event``, ``is_swr_stop`` and every SWR-aligned analysis)
comes from the Kay inventory that the SWR runner (``run_swr_detection.py``,
refresh stage R3) writes, not from Spyglass's stored ``RippleTimesV1``. The
2.0 rule is the published Kay et al. (2016) definition: at least 15 ms
continuously above z = 2. Design note: ``docs/design/2026-10-09-swr-source.md``.

What this module fixes
----------------------
- **Which inventory**: :data:`SWR_INVENTORY_RUN_ID`, the run folder, and
  :data:`SWR_DETECTOR`, the detector, under :data:`SWR_INVENTORY_ROOT`. They are
  the only place either is written; rerunning R3 (with Task 16's artifact mask,
  say) is a change of :data:`SWR_INVENTORY_RUN_ID` and nothing else.
- **Which epochs it serves**: one whose manifest status is ``ok``, or
  ``partial`` with the Kay inventory written. Anything else raises
  :class:`SwrSourceError`. The event and observed files are checked against
  the sha256 the manifest recorded for them.
- **Which events count** (:func:`swr_events`): the project's whole-event
  immobility rule (every speed sample in the closed interval finite and
  strictly below 4 cm/s, ``event_immobility``), the rule of the non-local and
  MUA cohorts, on the head speed of the decode grid. An event that overlaps an
  artifact interval is dropped too (the hook for Task 16; none today). Every
  dropped event keeps its row and a :class:`DropReason`.
- **The per-bin labels** (:func:`swr_event_labels`): exactly the convention
  ``SWR_event`` has had since the unified immobility change. That is
  ``build_speed_eligible_event_cohort(..., preserve_source_ids=True)``, the call
  ``get_data_over_time`` makes: a bin is labelled with an event's number when
  its time lies in the event's closed interval ``[start_time, end_time]``; the
  number is the inventory's own ``event_number``, so a dropped event leaves a
  gap; and a dropped event's bins are 0.
- **Where detection was observed** (:func:`swr_observed_stretches`,
  :func:`swr_observed_mask`): the gap-free stretches of the stored ripple band
  that Kay read (``scope == "stored_band"`` in ``swr_observed.parquet``).

Analyses read a decode's ``data_over_time`` through :func:`with_swr_source`
or :func:`read_dot_with_swr_source`. Its ``SWR_event`` column then holds this
source's labels and ``swr_observed`` the coverage mask. The decode's own
``SWR_event`` (from ``RippleTimesV1``, written by ``get_data_over_time``) stays
in the pickle as the decode's provenance and never reaches an analysis, and
neither do the decode's stored-ripple counts (:data:`DECODE_SWR_COUNT_COLUMNS`).
The decoders' ``no_ripple`` encoding-training mask still reads the stored
ripples; that is a separate decision.
"""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from continuum_swr_replay.analysis.constants import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
)
from continuum_swr_replay.analysis.hippocampus.event_immobility import (
    build_speed_eligible_event_cohort,
    filter_event_intervals_by_speed,
)
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR
from continuum_swr_replay.session_keys import SESSION_COLUMNS_DAY
from continuum_swr_replay.table_io import sha256_file

__all__ = [
    "DECODE_SWR_COUNT_COLUMNS",
    "DropReason",
    "SWR_DETECTOR",
    "SWR_INVENTORY_ROOT",
    "SWR_INVENTORY_RUN_ID",
    "SWR_LABEL_COLUMN",
    "SWR_OBSERVED_COLUMN",
    "SwrSourceError",
    "read_dot_with_swr_source",
    "read_swr_inventory",
    "swr_epoch_id",
    "swr_event_labels",
    "swr_events",
    "swr_inventory_dir",
    "swr_observed_mask",
    "swr_observed_stretches",
    "swr_source_files",
    "swr_source_name",
    "with_swr_source",
]

SWR_INVENTORY_RUN_ID = "50a98592acee"
"""The SWR runner's run folder every analysis reads its SWRs from.

R3's full run of 2026-10-05 at commit ``50a98592acee``: 216 epochs ``ok``, the 3
known no-input epochs skipped, and j1620210710 ``02_r1`` and ``04_r2``
``failed`` (Spyglass's own Kay inputs could not be fetched), so this source
refuses those two until R3 reruns them. R3 will be rerun with Task 16's
artifact mask under a new run id; changing this constant is the whole switch.
"""

SWR_DETECTOR = "kay"
"""The detector whose inventory is the paper's SWRs: ripple_detection 2.0's
``Kay_ripple_detector``, at least 15 ms continuously above z = 2 (Kay et al.
2016), with the runner's pinned ``DETECTOR_SETTINGS["kay"]``."""

SWR_INVENTORY_ROOT = PROCESSED_DATA_DIR / "replay_detection_comparison" / "swr"
"""Where the SWR runner writes its run folders, one per run id."""

SWR_LABEL_COLUMN = "SWR_event"
"""The per-bin SWR label column of an analysis frame (and of the decode's own
``data_over_time``, where it holds the stored ripples' labels)."""

SWR_OBSERVED_COLUMN = "swr_observed"
"""The per-bin coverage column :func:`with_swr_source` adds: True where SWR
detection was observed."""

OBSERVED_SCOPE = "stored_band"
"""The ``swr_observed`` scope of the band Kay reads."""

USABLE_STATUSES = ("ok", "partial")
"""Manifest statuses this source serves, given a Kay inventory."""

DECODE_SWR_COUNT_COLUMNS = (
    "n_SWR",
    "n_non_local_SWR",
    "n_non_local_no_SWR",
    "n_SWR_MUA",
    "n_SWR_no_MUA",
)
"""The decode's stored-ripple counts in ``data_over_time``; dropped from every
analysis frame, as the decode's own ``SWR_event`` labels are replaced."""


class DropReason(StrEnum):
    """Why :func:`swr_events` drops an inventory event. The first that holds,
    in this order."""

    ARTIFACT = "artifact"
    """The event overlaps an artifact interval (closed intervals: touching
    counts)."""
    NO_SPEED_SAMPLE = "no_speed_sample"
    """No sample of the speed's time axis lies in the event."""
    MOVING = "moving"
    """A finite speed sample in the event is at or above the threshold."""
    SPEED_UNKNOWN = "speed_unknown"
    """A speed sample in the event is not finite (and none is fast)."""


class SwrSourceError(ValueError):
    """The SWR source cannot serve an epoch: it is not in the inventory, its
    run did not produce a Kay inventory, or the files do not match the
    manifest."""


def swr_inventory_dir() -> Path:
    """The run folder of :data:`SWR_INVENTORY_RUN_ID` under
    :data:`SWR_INVENTORY_ROOT`, read when called."""
    return SWR_INVENTORY_ROOT / SWR_INVENTORY_RUN_ID


def swr_source_name() -> str:
    """The source as one string, for provenance records: ``kay@<run id>``."""
    return f"{SWR_DETECTOR}@{SWR_INVENTORY_RUN_ID}"


def swr_epoch_id(nwb_file_name: str, epoch_name: str) -> str:
    """The inventory's epoch id, such as ``chimi20200225_02_r1_HPC``.

    SWRs are detected on CA1's LFP, so the id is the HPC one whatever area an
    analysis decodes.
    """
    return f"{nwb_file_name.removesuffix('_.nwb')}_{epoch_name}_HPC"


def _resolve(inventory_dir: Path | None) -> Path:
    return swr_inventory_dir() if inventory_dir is None else Path(inventory_dir)


def swr_source_files(
    epoch_id: str, *, inventory_dir: Path | None = None
) -> dict[str, dict[str, str]]:
    """The epoch's Kay events and observed stretches, checked against the manifest.

    Parameters
    ----------
    epoch_id : str
        Such as ``"chimi20200225_02_r1_HPC"``.
    inventory_dir : Path, optional
        A run folder; :func:`swr_inventory_dir` by default.

    Returns
    -------
    dict
        ``{"events": {"path", "sha256"}, "observed": {"path", "sha256"}}``.

    Raises
    ------
    SwrSourceError
        If the run folder's manifest names another run, does not hold the
        epoch, holds it with a status other than ``ok`` or ``partial``, or
        without a Kay inventory; or if a file is missing or its sha256 is not
        the one the manifest recorded.
    """
    directory = _resolve(inventory_dir)
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise SwrSourceError(f"no SWR inventory manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("run_id") != directory.name:
        raise SwrSourceError(
            f"{manifest_path} is run {manifest.get('run_id')!r}, not the folder's "
            f"{directory.name!r}"
        )
    record = manifest.get("epochs", {}).get(epoch_id)
    if record is None:
        raise SwrSourceError(
            f"{epoch_id} is not in the SWR inventory {directory}; run the SWR "
            "runner for it"
        )
    status = record.get("status")
    if status not in USABLE_STATUSES:
        why = record.get("error") or record.get("reason") or "no reason recorded"
        raise SwrSourceError(
            f"{epoch_id} is {status!r} in the SWR inventory {directory.name} "
            f"({str(why)[:200]}); it has no SWRs until the SWR runner reruns it"
        )
    if record.get("detectors", {}).get(SWR_DETECTOR) != "ok":
        raise SwrSourceError(
            f"{epoch_id} is {status!r} in the SWR inventory {directory.name} "
            f"without a {SWR_DETECTOR} inventory: "
            f"{record.get('detectors', {}).get(SWR_DETECTOR)!r}"
        )
    files = {}
    for name, key in (("events", SWR_DETECTOR), ("observed", "swr_observed")):
        suffix = f"_{SWR_DETECTOR}_events" if name == "events" else "_swr_observed"
        path = directory / f"{epoch_id}{suffix}.parquet"
        if not path.exists():
            raise SwrSourceError(f"{epoch_id}: {path.name} is missing from {directory}")
        sha = sha256_file(path)
        expected = record.get("outputs", {}).get(key, {}).get("sha256")
        if sha != expected:
            raise SwrSourceError(
                f"{epoch_id}: {path.name} has sha256 {sha[:12]}, not the "
                f"{str(expected)[:12]} the manifest recorded"
            )
        files[name] = {"path": str(path), "sha256": sha}
    return files


def read_swr_inventory(
    epoch_id: str, *, inventory_dir: Path | None = None
) -> pd.DataFrame:
    """The epoch's Kay events as the SWR runner wrote them.

    Returns
    -------
    pd.DataFrame
        One row per event in time order, with a fresh integer index: column
        ``SWR_event``, the inventory's ``event_number`` (1, 2, ...), then the
        inventory's own columns (``start_time``, ``end_time``, ``duration``,
        ``peak_time``, z-score and speed summaries, ...). Times in seconds.

    Raises
    ------
    SwrSourceError
        As :func:`swr_source_files`; or if the events are not numbered 1..n in
        time order, a bound is not finite, or an event ends before it starts.
    """
    files = swr_source_files(epoch_id, inventory_dir=inventory_dir)
    events = pd.read_parquet(files["events"]["path"])
    numbers = events.index.to_numpy()
    if events.index.name != "event_number" or not np.array_equal(
        numbers, np.arange(1, len(events) + 1)
    ):
        raise SwrSourceError(
            f"{epoch_id}: Kay events are not indexed by event_number 1..n"
        )
    starts = events["start_time"].to_numpy(dtype=float)
    ends = events["end_time"].to_numpy(dtype=float)
    if not (np.isfinite(starts).all() and np.isfinite(ends).all()):
        raise SwrSourceError(f"{epoch_id}: a Kay event bound is not finite")
    if (ends < starts).any() or (np.diff(starts) <= 0).any():
        raise SwrSourceError(
            f"{epoch_id}: Kay events are not ordered intervals in time order"
        )
    events = events.reset_index().rename(columns={"event_number": SWR_LABEL_COLUMN})
    events.attrs = {}
    return events


def swr_observed_stretches(
    epoch_id: str, *, inventory_dir: Path | None = None
) -> NDArray[np.float64]:
    """Where SWR detection was observed: the stored band's gap-free stretches.

    Returns
    -------
    ndarray, shape (n_stretches, 2)
        First and last sample time (seconds) of each stretch, closed.
    """
    files = swr_source_files(epoch_id, inventory_dir=inventory_dir)
    observed = pd.read_parquet(files["observed"]["path"])
    band = observed.loc[observed["scope"] == OBSERVED_SCOPE, ["start_s", "end_s"]]
    stretches: NDArray[np.float64] = band.to_numpy(dtype=np.float64).reshape(-1, 2)
    return stretches


def swr_observed_mask(
    time: NDArray | pd.Index, epoch_id: str, *, inventory_dir: Path | None = None
) -> NDArray[np.bool_]:
    """True at each time that lies in one of the observed stretches (closed).

    Parameters
    ----------
    time : array-like, shape (n_time,)
        Seconds, such as a decode grid.
    epoch_id : str
    inventory_dir : Path, optional
    """
    stretches = swr_observed_stretches(epoch_id, inventory_dir=inventory_dir)
    values = np.asarray(time, dtype=float)
    observed = np.zeros(values.shape, dtype=bool)
    for start, end in stretches:
        observed |= (values >= start) & (values <= end)
    return observed


def _artifact_overlap(
    starts: NDArray, ends: NDArray, artifact_intervals: pd.DataFrame
) -> NDArray[np.bool_]:
    """Events whose closed interval shares any time with an artifact interval."""
    missing = [
        column
        for column in ("start_time", "end_time")
        if column not in artifact_intervals.columns
    ]
    if missing:
        raise KeyError(f"artifact_intervals is missing columns {missing}")
    art_starts = artifact_intervals["start_time"].to_numpy(dtype=float)
    art_ends = artifact_intervals["end_time"].to_numpy(dtype=float)
    if not (np.isfinite(art_starts).all() and np.isfinite(art_ends).all()):
        raise ValueError("artifact interval bounds must be finite")
    if (art_ends < art_starts).any():
        raise ValueError("an artifact interval ends before it starts")
    overlaps: NDArray[np.bool_] = (
        (starts[:, None] <= art_ends[None, :]) & (art_starts[None, :] <= ends[:, None])
    ).any(axis=1)
    return overlaps


def swr_events(
    epoch_id: str,
    *,
    speed: pd.Series,
    artifact_intervals: pd.DataFrame | None = None,
    inventory_dir: Path | None = None,
    speed_threshold: float = DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
) -> pd.DataFrame:
    """Every inventory event of the epoch, the dropped ones with their reason.

    Parameters
    ----------
    epoch_id : str
        Such as ``"chimi20200225_02_r1_HPC"``.
    speed : pd.Series
        Head speed (cm/s) indexed by time (seconds): the decode grid's own
        ``head_speed``, the samples the whole-event rule is applied to, as for
        the non-local and MUA cohorts.
    artifact_intervals : pd.DataFrame, optional
        Columns ``start_time`` and ``end_time`` (seconds). An event sharing any
        time with one is dropped as ``artifact``. None (today) drops none; Task
        16 supplies them.
    inventory_dir : Path, optional
        A run folder; :func:`swr_inventory_dir` by default.
    speed_threshold : float
        cm/s; the canonical immobility threshold (strict ``<``).

    Returns
    -------
    pd.DataFrame
        :func:`read_swr_inventory`'s rows and columns plus ``drop_reason``: a
        :class:`DropReason` value, or None for a kept event. The kept events
        are the paper's SWRs on this grid.

    Raises
    ------
    SwrSourceError
        If the epoch cannot be served (:func:`swr_source_files`).
    """
    events = read_swr_inventory(epoch_id, inventory_dir=inventory_dir)
    time = speed.index.to_numpy(dtype=float)
    values = speed.to_numpy(dtype=float)
    starts = events["start_time"].to_numpy(dtype=float)
    ends = events["end_time"].to_numpy(dtype=float)

    # The canonical rule decides; the counts below only say why it failed.
    immobile = events.index.isin(
        filter_event_intervals_by_speed(
            events, speed=values, time=time, speed_threshold=speed_threshold
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

    reason = np.full(len(events), None, dtype=object)
    reason[~immobile & (_in_event(~finite) > 0)] = DropReason.SPEED_UNKNOWN.value
    reason[~immobile & (_in_event(fast) > 0)] = DropReason.MOVING.value
    reason[~immobile & (stop <= first)] = DropReason.NO_SPEED_SAMPLE.value
    if artifact_intervals is not None and len(events):
        reason[_artifact_overlap(starts, ends, artifact_intervals)] = (
            DropReason.ARTIFACT.value
        )
    if (pd.isna(reason) & ~immobile).any():
        raise RuntimeError(
            f"{epoch_id}: an event failed the immobility rule for no reason this "
            "module names"
        )
    events["drop_reason"] = reason
    return events


def swr_event_labels(
    time: NDArray | pd.Index,
    epoch_id: str,
    *,
    speed: NDArray,
    artifact_intervals: pd.DataFrame | None = None,
    inventory_dir: Path | None = None,
    speed_threshold: float = DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
) -> NDArray[np.int64]:
    """The per-bin ``SWR_event`` labels on a decode time grid.

    The convention ``get_data_over_time`` has written since the unified
    immobility change, and the table builders read:
    ``build_speed_eligible_event_cohort(events, speed=speed, time=time,
    event_col="SWR_event")``, here on the inventory's events, with an
    artifact-dropped event's bins also set to 0. A bin is labelled with the
    event's inventory number when its time lies in ``[start_time, end_time]``.

    Parameters
    ----------
    time : array-like, shape (n_time,)
        The grid (seconds), strictly increasing.
    epoch_id : str
    speed : array-like, shape (n_time,)
        Head speed on `time` (cm/s).
    artifact_intervals, inventory_dir, speed_threshold
        As :func:`swr_events`.

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
    events = swr_events(
        epoch_id,
        speed=pd.Series(speed_values, index=time_index),
        artifact_intervals=artifact_intervals,
        inventory_dir=inventory_dir,
        speed_threshold=speed_threshold,
    )
    _, labels = build_speed_eligible_event_cohort(
        events[["start_time", "end_time"]],
        speed=speed_values,
        time=time_index,
        event_col=SWR_LABEL_COLUMN,
        speed_threshold=speed_threshold,
    )
    kept = events.loc[events["drop_reason"].isna(), SWR_LABEL_COLUMN].to_numpy()
    kept_labels: NDArray[np.int64] = labels.where(labels.isin(kept), 0).to_numpy(
        dtype=np.int64
    )
    return kept_labels


def with_swr_source(
    dot: pd.DataFrame,
    *,
    artifact_intervals: pd.DataFrame | None = None,
    inventory_dir: Path | None = None,
) -> pd.DataFrame:
    """One epoch's ``data_over_time`` with its SWR labels from this source.

    Parameters
    ----------
    dot : pd.DataFrame
        One epoch's decode frame, time-indexed (or with a ``time`` column),
        carrying ``animal``, ``day``, ``epoch`` and ``head_speed``.
    artifact_intervals, inventory_dir
        As :func:`swr_events`.

    Returns
    -------
    pd.DataFrame
        A new frame (`dot` is not changed) with the same rows: ``SWR_event``
        holds :func:`swr_event_labels` on the frame's time axis and head speed,
        ``swr_observed`` is :func:`swr_observed_mask`, and the decode's
        stored-ripple counts (:data:`DECODE_SWR_COUNT_COLUMNS`) are gone.

    Raises
    ------
    ValueError
        If the frame holds more than one epoch, or lacks a needed column.
    SwrSourceError
        If the epoch cannot be served.
    """
    missing = [c for c in (*SESSION_COLUMNS_DAY, "head_speed") if c not in dot.columns]
    if missing:
        raise ValueError(f"with_swr_source: data_over_time lacks columns {missing}")
    sessions = dot[list(SESSION_COLUMNS_DAY)].drop_duplicates()
    if len(sessions) != 1:
        raise ValueError(
            f"with_swr_source reads one epoch at a time, not {len(sessions)}"
        )
    animal, day, epoch = sessions.iloc[0]
    epoch_id = swr_epoch_id(f"{animal}{day}_.nwb", str(epoch))
    time = dot["time"].to_numpy() if "time" in dot.columns else dot.index.to_numpy()
    labels = swr_event_labels(
        pd.Index(np.asarray(time, dtype=float), name="time"),
        epoch_id,
        speed=dot["head_speed"].to_numpy(dtype=float),
        artifact_intervals=artifact_intervals,
        inventory_dir=inventory_dir,
    )
    observed = swr_observed_mask(time, epoch_id, inventory_dir=inventory_dir)
    # A shallow copy shares the untouched columns of a frame that can be half a
    # gigabyte; deleting and assigning columns on it leaves `dot` as it was.
    out = dot.copy(deep=False)
    for column in DECODE_SWR_COUNT_COLUMNS:
        if column in out.columns:
            del out[column]
    out[SWR_LABEL_COLUMN] = labels
    out[SWR_OBSERVED_COLUMN] = observed
    return out


def read_dot_with_swr_source(
    path: str | Path, *, inventory_dir: Path | None = None
) -> pd.DataFrame:
    """Read a decode's ``data_over_time`` pickle for analysis.

    :func:`with_swr_source` on the pickle: every analysis reader of a decode
    frame goes through here, so none sees the decode's stored-ripple labels.
    """
    frame: Any = pd.read_pickle(path)
    return with_swr_source(frame, inventory_dir=inventory_dir)
