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
  the only place either is written; rerunning R3 is a change of
  :data:`SWR_INVENTORY_RUN_ID` and nothing else.
- **Which artifacts**: :data:`ARTIFACT_INTERVALS_RUN_ID` under
  :data:`ARTIFACT_INTERVALS_ROOT`, the artifact detector's run
  (``scripts/decoding/run_artifact_detection.py``). It is None today: no
  artifact is masked. Every function here resolves an epoch's artifact
  intervals itself (:func:`artifact_intervals`), so no caller passes them and
  every caller gets the same ones: the one artifact mask, the detector's cores
  padded by ``artifacts.masks.PADDING_S``. The SWR runner, MUA detection, the
  decoders and the ripple-power covariate (:func:`ripple_power_inputs`) use the
  mask itself; every analysis consumer uses *excluded time*, the mask widened
  by ``event_eligibility.ARTIFACT_REACH_S`` (:func:`excluded_intervals`).
- **The source's name** (:func:`swr_source_name`): the detector, the inventory
  run and the artifact run, such as ``kay@50a98592acee+artifacts@none``. Every
  table built from these labels is stamped with it (:func:`stamp_swr_source`),
  and a reader that combines a table's SWR columns with this source's live
  labels refuses a table stamped with another name
  (:func:`require_swr_source`).
- **Which epochs it serves**: one whose manifest status is ``ok``, or
  ``partial`` with the Kay inventory written. Anything else raises
  :class:`SwrSourceError`. The event and observed files are checked against
  the sha256 the manifest recorded for them.
- **Which events count** (:func:`swr_events`): the project's one
  analysed-time rule (``event_eligibility``): the whole-event immobility rule
  (every speed sample in the closed interval finite and strictly below 4 cm/s,
  ``event_immobility``), the rule of the non-local and MUA cohorts, on the
  head speed of the decode grid, and no time shared with excluded time. Every
  dropped event keeps its row and a :class:`DropReason`.
- **Every cohort of an analysis frame** (:func:`with_swr_source`): every SWR,
  MUA and non-local event in excluded time is dropped whole, and so is a kept
  event that overlaps an SWR or MUA event dropped there; the other events keep
  their numbers, and NonLocal and ContFrag frames give the same SWR and MUA
  cohorts. :func:`artifact_event_drops` gives every event of a frame with its
  reason.
- **The per-bin labels** (:func:`swr_event_labels`): exactly the convention
  ``SWR_event`` has had since the unified immobility change. That is
  ``build_speed_eligible_event_cohort(..., preserve_source_ids=True)``, the call
  ``get_data_over_time`` makes: a bin is labelled with an event's number when
  its time lies in the event's closed interval ``[start_time, end_time]``; the
  number is the inventory's own ``event_number``, so a dropped event leaves a
  gap; and a dropped event's bins are 0.
- **Where detection was observed** (:func:`swr_observed_stretches`,
  :func:`swr_observed_mask`): the gap-free stretches of the stored ripple band
  that Kay read (``scope == "stored_band"`` in ``swr_observed.parquet``), less
  excluded time. Excluded time is not analysed: an event there is neither an
  SWR nor a non-SWR event.

Analyses read a decode's ``data_over_time`` through :func:`with_swr_source`
or :func:`read_dot_with_swr_source`. Its ``SWR_event`` column then holds this
source's labels and ``swr_observed`` the coverage mask, and every event cohort
of the frame follows the one analysed-time rule. An HPC decode made by the
decoders since Task 17a already holds the source's SWR labels: its
``organize_results`` reads the frame through :func:`with_swr_source` (keeping
every event it decoded, ``drop_masked_events=False``), so the decode's
provenance is this source, and its encoding leaves out every event of this
inventory and the artifact mask (``decoding_common.hpc_decode_masks``). In an older
decode, or a PFC decode, ``SWR_event`` (from ``RippleTimesV1``, written by
``get_data_over_time``) stays in the pickle as the decode's provenance and never
reaches an analysis, and neither do the decode's stored-ripple counts
(:data:`DECODE_SWR_COUNT_COLUMNS`).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from continuum_swr_replay.analysis.constants import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
)
from continuum_swr_replay.analysis.hippocampus.artifacts.io import (
    ArtifactRunError,
    read_epoch_cores,
)
from continuum_swr_replay.analysis.hippocampus.artifacts.masks import (
    artifact_intervals as padded_cores,
)
from continuum_swr_replay.analysis.hippocampus.artifacts.masks import (
    subtract_intervals,
)
from continuum_swr_replay.analysis.hippocampus.event_eligibility import (
    EXCLUDED_TIME_REASONS,
    DropReason,
    drop_labels,
    event_drop_reasons,
    excluded_time,
    label_drop_reasons,
    overlaps_dropped_events,
)
from continuum_swr_replay.analysis.hippocampus.event_immobility import (
    build_speed_eligible_event_cohort,
    intervals_from_labels,
)
from continuum_swr_replay.data_loaders.utils import parse_nwb_filename
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR
from continuum_swr_replay.session_keys import SESSION_COLUMNS, SESSION_COLUMNS_DAY
from continuum_swr_replay.table_io import sha256_file

logger = logging.getLogger(__name__)

__all__ = [
    "ARTIFACT_INTERVALS_ROOT",
    "ARTIFACT_INTERVALS_RUN_ID",
    "DECODE_SWR_COUNT_COLUMNS",
    "DropReason",
    "SWR_DETECTOR",
    "SWR_INVENTORY_ROOT",
    "SWR_INVENTORY_RUN_ID",
    "SWR_LABEL_COLUMN",
    "SWR_OBSERVED_COLUMN",
    "SWR_SOURCE_ATTR",
    "SWR_SOURCE_TABLES",
    "SwrSourceError",
    "artifact_event_drops",
    "artifact_intervals",
    "artifact_intervals_dir",
    "artifact_run_of",
    "dropped_for_excluded_time",
    "excluded_intervals",
    "excluded_intervals_by_session",
    "excluded_intervals_of",
    "read_artifact_event_drops",
    "read_dot_with_swr_source",
    "read_swr_inventory",
    "require_swr_source",
    "ripple_power_inputs",
    "stamp_swr_source",
    "swr_epoch_id",
    "swr_event_labels",
    "swr_events",
    "swr_intervals",
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

ARTIFACT_INTERVALS_RUN_ID: str | None = None
"""The artifact detector's run whose intervals drop overlapping SWRs and leave
the coverage.

None today: no artifact is masked, and the source's name says so
(``artifacts@none``). Runbook R9 sets it to its run id. The folder,
:data:`ARTIFACT_INTERVALS_ROOT` ``/ <run id>``, holds ``manifest.json`` and one
``{epoch_id}_artifact_intervals.parquet`` per epoch: the epoch's unpadded
cores (``artifacts.io.CORE_COLUMNS``), no rows when it has none. A missing
file is an error, not an epoch without artifacts.
"""

ARTIFACT_INTERVALS_ROOT = PROCESSED_DATA_DIR / "artifact_intervals"
"""Where artifact-interval run folders go, one per run id."""

SWR_LABEL_COLUMN = "SWR_event"
"""The per-bin SWR label column of an analysis frame (and of the decode's own
``data_over_time``, where it holds the stored ripples' labels)."""

SWR_OBSERVED_COLUMN = "swr_observed"
"""The per-bin coverage column :func:`with_swr_source` adds: True where SWR
detection was observed."""

SWR_SOURCE_ATTR = "swr_source"
"""The ``attrs`` key (kept in a parquet table's metadata) that names the SWR
source a frame or table was built from: :func:`swr_source_name`."""

SWR_SOURCE_TABLES = frozenset(
    {
        "event",
        "event_arm",
        "event_patch",
        "event_sequence",
        "stop",
        "event_onset",
        "stop_timebin",
    }
)
"""The canonical tables (covariate-schema keys) built from decode frames read
through this source; ``publish_canonical`` refuses one without the current
stamp. qvals, built from behavior alone, is the one canonical table that is
not among them. event_sequence holds no SWR column but is published with its
atomic set, so it carries the set's stamp."""

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

_ARTIFACT_COLUMNS = ["start_time", "end_time"]

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
    "speed_eligible",
    "drop_reason",
]
"""The columns of :func:`artifact_event_drops`."""


class SwrSourceError(ValueError):
    """The SWR source cannot serve an epoch (it is not in the inventory, its
    run did not produce a Kay inventory, a file does not match the manifest,
    or its artifact intervals are missing), or a table was built from another
    SWR source."""


def swr_inventory_dir() -> Path:
    """The run folder of :data:`SWR_INVENTORY_RUN_ID` under
    :data:`SWR_INVENTORY_ROOT`, read when called."""
    return SWR_INVENTORY_ROOT / SWR_INVENTORY_RUN_ID


def artifact_intervals_dir() -> Path | None:
    """The artifact run folder, or None while :data:`ARTIFACT_INTERVALS_RUN_ID`
    is unset. Read when called."""
    if ARTIFACT_INTERVALS_RUN_ID is None:
        return None
    return ARTIFACT_INTERVALS_ROOT / ARTIFACT_INTERVALS_RUN_ID


def swr_source_name() -> str:
    """The source as one string, the stamp of every table built from it.

    ``<detector>@<inventory run id>+artifacts@<artifact run id or none>``, such
    as ``kay@50a98592acee+artifacts@none``. Read when called.
    """
    artifacts = ARTIFACT_INTERVALS_RUN_ID or "none"
    return f"{SWR_DETECTOR}@{SWR_INVENTORY_RUN_ID}+artifacts@{artifacts}"


def artifact_run_of(source_name: str) -> str | None:
    """The artifact run a source name (:func:`swr_source_name`) was built under.

    ``None`` for ``artifacts@none``, and for a name with no ``+artifacts@``
    part (written before the name recorded the artifact run): neither was
    built under a run.
    """
    _, separator, run = str(source_name).partition("+artifacts@")
    if not separator or run == "none":
        return None
    return run


def stamp_swr_source(frame: pd.DataFrame) -> pd.DataFrame:
    """Stamp `frame` (in place, and return it) as built from this source.

    Only a producer that built the frame from decode frames read through this
    source stamps it; ``publish_canonical`` refuses a table of
    :data:`SWR_SOURCE_TABLES` without the current stamp.
    """
    frame.attrs[SWR_SOURCE_ATTR] = swr_source_name()
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
    if stamp != swr_source_name():
        raise SwrSourceError(
            f"{what} was built from SWR source {stamp!r}, not the configured "
            f"{swr_source_name()!r}; regenerate it (and its atomic set) from the "
            "current source"
        )


def swr_epoch_id(nwb_file_name: str, epoch_name: str) -> str:
    """The inventory's epoch id, such as ``chimi20200225_02_r1_HPC``.

    SWRs are detected on CA1's LFP, so the id is the HPC one whatever area an
    analysis decodes.
    """
    return f"{nwb_file_name.removesuffix('_.nwb')}_{epoch_name}_HPC"


def swr_source_files(epoch_id: str) -> dict[str, dict[str, str]]:
    """The epoch's Kay events and observed stretches, checked against the manifest.

    Parameters
    ----------
    epoch_id : str
        Such as ``"chimi20200225_02_r1_HPC"``.

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
    directory = swr_inventory_dir()
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


def ripple_power_inputs(epoch_id: str) -> dict[str, Any]:
    """What ``load_data`` needs to make the ripple-power covariate this source's.

    ``SWR_zscore`` in a decode's ``data_over_time`` (``ripple_power_z`` in the
    analyses) is then the consensus the Kay detector thresholded for this
    inventory: over the tetrodes rule R-T kept, z-scored with the artifact mask
    left out and no value inside it (``data_loaders.lfp.with_ripple_power``).
    An inventory made before rule R-T read every band tetrode.

    Parameters
    ----------
    epoch_id : str

    Returns
    -------
    dict
        ``band_electrode_ids`` (the band the runner read, in column order),
        ``kept_electrode_ids`` (the tetrodes it detected on), the band's
        nominal ``sampling_frequency`` (Hz), the ``position_span_s`` it cut the
        band to and normalized over, and ``artifact_intervals``
        (:func:`artifact_intervals`; the covariate is an input, so it reads the
        mask itself, not excluded time).

    Raises
    ------
    SwrSourceError
        As :func:`swr_source_files` and :func:`artifact_intervals`.
    """
    swr_source_files(epoch_id)  # the epoch is served, and its files unchanged
    manifest = json.loads((swr_inventory_dir() / "manifest.json").read_text())
    record = manifest["epochs"][epoch_id]
    kept = [int(electrode) for electrode in record["electrode_ids"]]
    selection = record.get("tetrode_selection")
    band = (
        kept
        if selection is None
        else [int(electrode) for electrode in selection["electrode_ids"]]
    )
    return {
        "band_electrode_ids": band,
        "kept_electrode_ids": kept,
        "sampling_frequency": float(record["sampling"]["nominal"]),
        "position_span_s": tuple(float(t) for t in record["position_span_s"]),
        "artifact_intervals": artifact_intervals(epoch_id),
    }


def _read_events(epoch_id: str, path: str) -> pd.DataFrame:
    events = pd.read_parquet(path)
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


def _read_observed(path: str) -> NDArray[np.float64]:
    observed = pd.read_parquet(path)
    band = observed.loc[observed["scope"] == OBSERVED_SCOPE, ["start_s", "end_s"]]
    stretches: NDArray[np.float64] = band.to_numpy(dtype=np.float64).reshape(-1, 2)
    return stretches


def artifact_intervals(epoch_id: str) -> pd.DataFrame:
    """The epoch's artifact mask, from the one configured run.

    The run's cores for the epoch, padded by ``artifacts.masks.PADDING_S`` (S
    by 1 s, a step with an LFP signature by 0.3 s and a spike-only line by
    0.1 s, on each side), merged and clipped to the epoch's span. Every
    function here uses it, and so do the SWR runner, MUA detection and the HPC
    decoders.

    Parameters
    ----------
    epoch_id : str

    Returns
    -------
    pd.DataFrame
        Columns ``start_time`` and ``end_time`` (seconds), closed, disjoint and
        in time order; no rows while :data:`ARTIFACT_INTERVALS_RUN_ID` is unset.

    Raises
    ------
    SwrSourceError
        If an artifact run is configured but its manifest does not serve the
        epoch (not run, failed, another run's folder), its file is missing or
        changed since the run, or the cores are malformed.
    """
    directory = artifact_intervals_dir()
    if directory is None:
        return pd.DataFrame(
            {column: np.empty(0, dtype=float) for column in _ARTIFACT_COLUMNS}
        )
    try:
        epoch = read_epoch_cores(directory, epoch_id)
    except ArtifactRunError as error:
        raise SwrSourceError(str(error)) from error
    return padded_cores(epoch.cores, epoch.span)


def excluded_intervals(epoch_id: str) -> pd.DataFrame:
    """The epoch's excluded time: :func:`artifact_intervals` widened by
    ``event_eligibility.ARTIFACT_REACH_S`` on each side, merged.

    The one function every analysis consumer reads: an event, an exposure
    second, a time bin with its history window and a peri-event window are
    analysed only wholly outside these intervals. Inputs (the decoders' missing
    bins, the SWR detectors, the ripple-power covariate) read the mask itself.

    Parameters
    ----------
    epoch_id : str

    Returns
    -------
    pd.DataFrame
        ``start_time`` and ``end_time`` (seconds), closed, disjoint and in time
        order; no rows while :data:`ARTIFACT_INTERVALS_RUN_ID` is unset.

    Raises
    ------
    SwrSourceError
        As :func:`artifact_intervals`.
    """
    return excluded_time(artifact_intervals(epoch_id))


@dataclass(frozen=True)
class _EpochSource:
    """What the source holds for one epoch, read with one manifest check."""

    events: pd.DataFrame
    observed: NDArray[np.float64]
    """The stored band's stretches less excluded time."""
    artifacts: pd.DataFrame


def _epoch_source(epoch_id: str) -> _EpochSource:
    files = swr_source_files(epoch_id)
    artifacts = artifact_intervals(epoch_id)
    return _EpochSource(
        events=_read_events(epoch_id, files["events"]["path"]),
        observed=subtract_intervals(
            _read_observed(files["observed"]["path"]), excluded_time(artifacts)
        ),
        artifacts=artifacts,
    )


def read_swr_inventory(epoch_id: str) -> pd.DataFrame:
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
    files = swr_source_files(epoch_id)
    return _read_events(epoch_id, files["events"]["path"])


def swr_observed_stretches(epoch_id: str) -> NDArray[np.float64]:
    """Where SWR detection was observed and is analysed: the stored band's
    gap-free stretches, less the epoch's :func:`excluded_intervals`.

    Excluded time is not analysed. Next to an excluded interval a stretch ends
    (or starts) at the float just outside it, so a time lies in a stretch
    exactly when it was observed and lies in no excluded interval.

    Returns
    -------
    ndarray, shape (n_stretches, 2)
        First and last time (seconds) of each stretch, closed.
    """
    files = swr_source_files(epoch_id)
    return subtract_intervals(
        _read_observed(files["observed"]["path"]), excluded_intervals(epoch_id)
    )


def _mask(
    stretches: NDArray[np.float64], time: NDArray | pd.Index
) -> NDArray[np.bool_]:
    values = np.asarray(time, dtype=float)
    observed = np.zeros(values.shape, dtype=bool)
    for start, end in stretches:
        observed |= (values >= start) & (values <= end)
    return observed


def swr_observed_mask(time: NDArray | pd.Index, epoch_id: str) -> NDArray[np.bool_]:
    """True at each time that lies in one of the observed stretches (closed):
    detection was observed there, and it is in no excluded interval.

    Parameters
    ----------
    time : array-like, shape (n_time,)
        Seconds, such as a decode grid.
    epoch_id : str
    """
    return _mask(swr_observed_stretches(epoch_id), time)


def _classify(
    epoch_id: str,
    source: _EpochSource,
    speed: pd.Series,
    speed_threshold: float,
) -> pd.DataFrame:
    """The inventory's events with ``drop_reason`` (excluded time, then speed)
    and ``speed_eligible`` (the speed rule alone)."""
    events = source.events.copy()
    no_mask = source.artifacts.iloc[0:0]
    try:
        for column, artifacts in (
            ("speed_reason", no_mask),
            ("drop_reason", source.artifacts),
        ):
            events[column] = event_drop_reasons(
                events[["start_time", "end_time"]],
                speed=speed.to_numpy(dtype=float),
                time=speed.index.to_numpy(dtype=float),
                artifacts=artifacts,
                speed_threshold=speed_threshold,
            )
    except RuntimeError as error:
        raise RuntimeError(f"{epoch_id}: {error}") from error
    events["speed_eligible"] = events.pop("speed_reason").isna()
    return events


def swr_events(
    epoch_id: str,
    *,
    speed: pd.Series,
    speed_threshold: float = DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
) -> pd.DataFrame:
    """Every inventory event of the epoch, the dropped ones with their reason.

    The SWR cohort's own rule: excluded time and the whole-event immobility
    rule. The pass that drops an SWR overlapping an MUA event dropped for
    excluded time needs a decode frame's MUA labels: :func:`artifact_event_drops`
    makes it.

    Parameters
    ----------
    epoch_id : str
        Such as ``"chimi20200225_02_r1_HPC"``.
    speed : pd.Series
        Head speed (cm/s) indexed by time (seconds): the decode grid's own
        ``head_speed``, the samples the whole-event rule is applied to, as for
        the non-local and MUA cohorts.
    speed_threshold : float
        cm/s; the canonical immobility threshold (strict ``<``).

    Returns
    -------
    pd.DataFrame
        :func:`read_swr_inventory`'s rows and columns plus ``speed_eligible``
        (the immobility rule alone) and ``drop_reason``: a :class:`DropReason`
        value, or None for a kept event. An event in the epoch's mask is
        ``artifact``, one within the reach of it ``near_artifact``.

    Raises
    ------
    SwrSourceError
        If the epoch cannot be served (:func:`swr_source_files`,
        :func:`artifact_intervals`).
    """
    return _classify(epoch_id, _epoch_source(epoch_id), speed, speed_threshold)


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
    cohort's own rule (:func:`swr_events`).

    The convention ``get_data_over_time`` has written since the unified
    immobility change, and the table builders read:
    ``build_speed_eligible_event_cohort(events, speed=speed, time=time,
    event_col="SWR_event")``, here on the inventory's events, with the bins of
    an event dropped for excluded time also set to 0. A bin is labelled with
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
    events = _classify(
        epoch_id,
        _epoch_source(epoch_id),
        pd.Series(speed_values, index=time_index),
        speed_threshold,
    )
    labels = _speed_eligible_labels(events, time_index, speed_values, speed_threshold)
    kept = events.loc[events["drop_reason"].isna(), SWR_LABEL_COLUMN].to_numpy()
    return _keep_labels(labels, kept)


def _frame_epoch(dot: pd.DataFrame, what: str) -> tuple[str, NDArray]:
    """The one epoch a frame holds, as the inventory's id, and its time axis."""
    missing = [c for c in (*SESSION_COLUMNS_DAY, "head_speed") if c not in dot.columns]
    if missing:
        raise ValueError(f"{what}: data_over_time lacks columns {missing}")
    sessions = dot[list(SESSION_COLUMNS_DAY)].drop_duplicates()
    if len(sessions) != 1:
        raise ValueError(f"{what} reads one epoch at a time, not {len(sessions)}")
    animal, day, epoch = sessions.iloc[0]
    time = dot["time"].to_numpy() if "time" in dot.columns else dot.index.to_numpy()
    return swr_epoch_id(f"{animal}{day}_.nwb", str(epoch)), np.asarray(time, float)


def _drops(
    epoch_id: str, source: _EpochSource, dot: pd.DataFrame, time: NDArray
) -> tuple[pd.DataFrame, NDArray[np.int64]]:
    """Every event of the frame with its reason, and the speed-eligible SWR
    labels on its grid.

    The SWR rows are the inventory's (its bounds); the label columns' rows are
    the frame's (sampled bounds). The overlap pass reads every event at its
    sampled bounds on the frame's grid.
    """
    time_index = pd.Index(time, name="time")
    speed_values = dot["head_speed"].to_numpy(dtype=float)
    swrs = _classify(
        epoch_id,
        source,
        pd.Series(speed_values, index=time_index),
        DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    )
    swr_labels = _speed_eligible_labels(
        swrs, time_index, speed_values, DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC
    )
    parts = [
        pd.DataFrame(
            {
                "event_col": SWR_LABEL_COLUMN,
                "label": swrs[SWR_LABEL_COLUMN].to_numpy(),
                "start_time": swrs["start_time"].to_numpy(dtype=float),
                "end_time": swrs["end_time"].to_numpy(dtype=float),
                "speed_eligible": swrs["speed_eligible"].to_numpy(dtype=bool),
                "drop_reason": swrs["drop_reason"].to_numpy(),
            }
        )
    ]
    for column in LABEL_COLUMNS:
        if column in dot.columns:
            parts.append(
                label_drop_reasons(dot, column, source.artifacts)
                .rename(columns={column: "label"})
                .assign(event_col=column, speed_eligible=True)[DROP_COLUMNS]
            )
    drops = pd.concat(parts, ignore_index=True)[DROP_COLUMNS]

    # One pass: a kept event that overlaps an SWR or MUA event dropped for
    # excluded time (and eligible but for it) has an unknown overlap class.
    sampled = drops[["event_col", "label", "start_time", "end_time"]].copy()
    swr_sampled = intervals_from_labels(
        pd.DataFrame({SWR_LABEL_COLUMN: swr_labels}, index=time_index),
        SWR_LABEL_COLUMN,
    ).set_index(SWR_LABEL_COLUMN)
    is_swr = (sampled["event_col"] == SWR_LABEL_COLUMN).to_numpy()
    on_grid = sampled["label"].where(is_swr).isin(swr_sampled.index).to_numpy()
    sampled.loc[is_swr, ["start_time", "end_time"]] = np.nan
    if on_grid.any():
        on_grid_labels = sampled.loc[on_grid, "label"]
        for bound in ("start_time", "end_time"):
            sampled.loc[on_grid, bound] = swr_sampled.loc[
                on_grid_labels, bound
            ].to_numpy(dtype=float)
    dropped = (
        drops["event_col"].isin(OVERLAP_SOURCES)
        & drops["drop_reason"].isin(EXCLUDED_TIME_REASONS)
        & drops["speed_eligible"]
    ).to_numpy() & np.isfinite(sampled["start_time"].to_numpy(dtype=float))
    kept = drops["drop_reason"].isna().to_numpy() & np.isfinite(
        sampled["start_time"].to_numpy(dtype=float)
    )
    if dropped.any() and kept.any():
        overlapping = overlaps_dropped_events(
            sampled.loc[kept, "start_time"].to_numpy(dtype=float),
            sampled.loc[kept, "end_time"].to_numpy(dtype=float),
            sampled.loc[dropped, ["start_time", "end_time"]],
        )
        rows = np.flatnonzero(kept)[overlapping]
        drops.loc[rows, "drop_reason"] = DropReason.OVERLAPS_DROPPED_EVENT.value
    return drops, swr_labels


def artifact_event_drops(dot: pd.DataFrame) -> pd.DataFrame:
    """Every event of one epoch's decode frame, with its reason to leave.

    The reasons :func:`with_swr_source` acts on, the one rule
    (``event_eligibility``) on every cohort:

    - the source's SWRs (:func:`swr_events`: excluded time, then speed);
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
        One epoch's decode frame as the decoder saved it (time-indexed or with
        a ``time`` column), carrying ``animal``, ``day``, ``epoch`` and
        ``head_speed``.

    Returns
    -------
    pd.DataFrame
        Columns :data:`DROP_COLUMNS`: the label column, the event's label, its
        bounds (seconds; an SWR's are the inventory's), whether the speed rule
        alone keeps it, and its reason (None for a kept event); SWR events
        first, then each label column, in label order.

    Raises
    ------
    ValueError
        If the frame holds more than one epoch, or lacks a needed column.
    SwrSourceError
        If the epoch cannot be served.
    """
    epoch_id, time = _frame_epoch(dot, "artifact_event_drops")
    drops, _ = _drops(epoch_id, _epoch_source(epoch_id), dot, time)
    return drops


def read_artifact_event_drops(path: str | Path) -> pd.DataFrame:
    """:func:`artifact_event_drops` of a decode's ``data_over_time`` pickle, as
    the decoder saved it: for a consumer that needs the dropped events
    themselves (an exposure count, a control window, a re-detected cohort)."""
    frame: Any = pd.read_pickle(path)
    return artifact_event_drops(frame)


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


def with_swr_source(
    dot: pd.DataFrame, *, drop_masked_events: bool = True
) -> pd.DataFrame:
    """One epoch's ``data_over_time`` with its SWR labels from this source,
    and every event cohort held to the one analysed-time rule.

    Parameters
    ----------
    dot : pd.DataFrame
        One epoch's decode frame, time-indexed (or with a ``time`` column),
        carrying ``animal``, ``day``, ``epoch`` and ``head_speed``.
    drop_masked_events : bool
        True (every analysis read): every event the rule drops
        (:func:`artifact_event_drops`) has its label zeroed whole. False only
        for a decoder saving its own frame, which keeps every event it decoded
        (its SWR labels still leave out the SWRs in the mask, as the inventory
        does).

    Returns
    -------
    pd.DataFrame
        A new frame (`dot` is not changed) with the same rows: ``SWR_event``
        holds the kept SWRs' labels on the frame's time axis and head speed
        (the inventory's numbers), ``swr_observed`` is
        :func:`swr_observed_mask` (excluded time unobserved), the decode's
        stored-ripple counts (:data:`DECODE_SWR_COUNT_COLUMNS`) are gone, and
        ``attrs`` carries the source's stamp (:func:`stamp_swr_source`). The
        epoch's source files are checked against the manifest once. With no
        artifact run configured, every other label is the decode's.

    Raises
    ------
    ValueError
        If the frame holds more than one epoch, or lacks a needed column.
    SwrSourceError
        If the epoch cannot be served.
    """
    epoch_id, time = _frame_epoch(dot, "with_swr_source")
    source = _epoch_source(epoch_id)
    masking = drop_masked_events and len(source.artifacts) > 0
    drops, swr_labels = _drops(epoch_id, source, dot, time)
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
    for column in DECODE_SWR_COUNT_COLUMNS:
        if column in out.columns:
            del out[column]
    out[SWR_LABEL_COLUMN] = _keep_labels(swr_labels, kept_swrs)
    out[SWR_OBSERVED_COLUMN] = _mask(source.observed, time)
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
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{what}: the frame lacks columns {missing}")
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
    return excluded_intervals(swr_epoch_id(f"{animal}{day}_.nwb", str(epoch)))


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
        excluded_intervals(swr_epoch_id(f"{animal}{day}_.nwb", str(epoch))).assign(
            animal=animal, session_date=day, epoch=epoch
        )
        for animal, day, epoch in keys.itertuples(index=False)
    ]
    columns = [*SESSION_COLUMNS, *_ARTIFACT_COLUMNS]
    if not parts:
        return pd.DataFrame({column: [] for column in columns}).astype(
            {"start_time": float, "end_time": float}
        )
    return pd.concat(parts, ignore_index=True)[columns]


def read_dot_with_swr_source(path: str | Path) -> pd.DataFrame:
    """Read a decode's ``data_over_time`` pickle for analysis.

    :func:`with_swr_source` on the pickle: every analysis reader of a decode
    frame goes through here, so none sees the decode's stored-ripple labels.
    """
    frame: Any = pd.read_pickle(path)
    return with_swr_source(frame)


def swr_intervals(
    dot: pd.DataFrame,
    *,
    nwb_file_name: str | None = None,
    epoch_name: str | None = None,
) -> pd.DataFrame:
    """One epoch's SWR intervals from a frame read through this source.

    The sampled bounds of the frame's ``SWR_event`` labels
    (:func:`~continuum_swr_replay.analysis.hippocampus.event_immobility.intervals_from_labels`).
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
    if SWR_OBSERVED_COLUMN not in dot.columns:
        raise ValueError(
            "swr_intervals needs a frame read through the SWR source "
            "(read_dot_with_swr_source); this one has no swr_observed column"
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
