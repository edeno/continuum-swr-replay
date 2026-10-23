"""The paper's one SWR source: the direct ripple_detection 2.0 Kay inventory.

The project lead decided on 2026-10-05 that every SWR label of the paper
(``is_swr``, ``SWR_event``, ``is_swr_stop`` and every SWR-aligned analysis)
comes from the Kay inventory that the SWR runner (``run_swr_detection.py``,
refresh stage R3) writes, not from Spyglass's stored ``RippleTimesV1``. The
2.0 rule is the published Kay et al. (2016) definition: at least 15 ms
continuously above z = 2. Design note: ``docs/design/2026-10-09-swr-source.md``.

This module reads the inventory. The read point that holds a decode frame to
the one analysed-time rule is ``analysed_time``, which imports this module,
never the reverse.

What this module fixes
----------------------
- **Which inventory**: :data:`SWR_INVENTORY_RUN_ID`, the run folder, and
  :data:`SWR_DETECTOR`, the detector, under ``paths.SWR_INVENTORY_ROOT``. They are
  the only place either is written; rerunning R3 is a change of
  :data:`SWR_INVENTORY_RUN_ID` and nothing else.
- **Which artifacts**: :data:`ARTIFACT_INTERVALS_RUN_ID` under
  ``paths.ARTIFACT_INTERVALS_DIR``, the artifact detector's run
  (``scripts/detection/run_artifact_detection.py``). It is set (runbook R9's
  run); None would mean no artifact is masked. Every function here resolves an
  epoch's artifact intervals itself (:func:`artifact_intervals`), so no caller
  passes them and every caller gets the same ones: the one artifact mask, the
  detector's cores padded by ``artifact_masks.PADDING_S``. The SWR runner, MUA
  detection, the decoders and the ripple-power covariate
  (:func:`ripple_power_inputs`) use the mask itself; every analysis consumer
  uses *excluded time*, the mask widened by ``eligibility.ARTIFACT_REACH_S``
  (``analysed_time.excluded_intervals``).
- **The source's name** (:func:`swr_source_name`): the detector, the inventory
  run and the artifact run, such as ``kay@056eee0dcb5a+artifacts@fbbff144a123``. Every
  table built from these labels is stamped with it
  (``analysed_time.stamp_swr_source``), and a reader that combines a table's
  SWR columns with this source's live labels refuses a table stamped with
  another name (``analysed_time.require_swr_source``).
- **Which epochs it serves**: one whose manifest status is ``ok``, or
  ``partial`` with the Kay inventory written. Anything else raises
  :class:`SwrSourceError`. The event and observed files are checked against
  the sha256 the manifest recorded for them.
- **Which events count** (:func:`swr_events`): the project's one
  analysed-time rule (``eligibility``): the whole-event immobility rule
  (every speed sample in the closed interval finite and strictly below 4 cm/s,
  ``immobility``), the rule of the non-local and MUA cohorts, on the
  head speed of the decode grid, and no time shared with excluded time. Every
  dropped event keeps its row and a ``DropReason``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from continuum_swr_replay.analysis.constants import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
)
from continuum_swr_replay.analysis.events.artifact_intervals import (
    ArtifactRunError,
    epoch_artifact_intervals,
)
from continuum_swr_replay.analysis.events.eligibility import (
    event_drop_reasons,
    excluded_time,
)
from continuum_swr_replay.data_loaders.utils import format_epoch_identifier
from continuum_swr_replay.intervals import subtract_intervals
from continuum_swr_replay.paths import ARTIFACT_INTERVALS_DIR, SWR_INVENTORY_ROOT
from continuum_swr_replay.runs.io import sha256_file

__all__ = [
    "ARTIFACT_INTERVALS_RUN_ID",
    "SWR_DETECTOR",
    "SWR_INVENTORY_RUN_ID",
    "SWR_LABEL_COLUMN",
    "EpochSource",
    "SwrSourceError",
    "artifact_intervals",
    "artifact_intervals_dir",
    "artifact_run_of",
    "classify",
    "epoch_source",
    "read_observed",
    "read_swr_inventory",
    "ripple_power_inputs",
    "swr_epoch_id",
    "swr_events",
    "swr_inventory_dir",
    "swr_source_files",
    "swr_source_name",
]


SWR_INVENTORY_RUN_ID = "056eee0dcb5a"
"""The SWR runner's run folder every analysis reads its SWRs from.

Runbook R10's run of 2026-10-10 at commit ``056eee0dcb5a``: all 255 epochs
``ok`` (the register's 242 analysed and its 13 ``SUSTAINED_NOISE``), with the
artifact mask of :data:`ARTIFACT_INTERVALS_RUN_ID` and rule R-T for all four
detectors (gate report
``.claude/research/refresh_artifact_detector/r10_gate/r10_gate_report.md``).
Changing this constant is the whole switch to another run.
"""


SWR_DETECTOR = "kay"
"""The detector whose inventory is the paper's SWRs: ripple_detection 2.0's
``Kay_ripple_detector``, at least 15 ms continuously above z = 2 (Kay et al.
2016), with the runner's pinned ``DETECTOR_SETTINGS["kay"]``."""


ARTIFACT_INTERVALS_RUN_ID: str | None = "fbbff144a123"
"""The artifact detector's run whose intervals drop overlapping SWRs and leave
the coverage.

Runbook R9's run (2026-10-10, commit fbbff14, all 255 epochs, gate report
``.claude/research/refresh_artifact_detector/r9_gate/rerun/r9_regate_report.md``).
None would mean no artifact is masked (``artifacts@none``). The folder,
``paths.ARTIFACT_INTERVALS_DIR / <run id>``, holds ``manifest.json`` and one
``{epoch_id}_artifact_intervals.parquet`` per epoch: the epoch's unpadded
cores (``artifact_intervals.CORE_COLUMNS``), no rows when it has none. A missing
file is an error, not an epoch without artifacts.
"""


SWR_LABEL_COLUMN = "SWR_event"
"""The per-bin SWR label column of an analysis frame (and of the decode's own
``data_over_time``, where it holds the stored ripples' labels)."""


OBSERVED_SCOPE = "stored_band"
"""The ``swr_observed`` scope of the band Kay reads."""


USABLE_STATUSES = ("ok", "partial")
"""Manifest statuses this source serves, given a Kay inventory."""


class SwrSourceError(ValueError):
    """The SWR source cannot serve an epoch (it is not in the inventory, its
    run did not produce a Kay inventory, a file does not match the manifest,
    or its artifact intervals are missing), or a table was built from another
    SWR source."""


def swr_inventory_dir() -> Path:
    """The run folder of :data:`SWR_INVENTORY_RUN_ID` under this module's
    ``SWR_INVENTORY_ROOT`` (imported from ``paths``).

    Both names are this module's globals, read when called, so a patch on
    ``swr_source.SWR_INVENTORY_ROOT`` reaches this function; a patch on
    ``paths.SWR_INVENTORY_ROOT`` does not."""
    return SWR_INVENTORY_ROOT / SWR_INVENTORY_RUN_ID


def artifact_intervals_dir() -> Path | None:
    """The artifact run folder, or None while :data:`ARTIFACT_INTERVALS_RUN_ID`
    is unset.

    The folder is :data:`ARTIFACT_INTERVALS_RUN_ID` under this module's
    ``ARTIFACT_INTERVALS_DIR`` (imported from ``paths``). Both names are this
    module's globals, read when called, so a patch on
    ``swr_source.ARTIFACT_INTERVALS_DIR`` reaches this function; a patch on
    ``paths.ARTIFACT_INTERVALS_DIR`` does not."""
    if ARTIFACT_INTERVALS_RUN_ID is None:
        return None
    return ARTIFACT_INTERVALS_DIR / ARTIFACT_INTERVALS_RUN_ID


def swr_source_name() -> str:
    """The source as one string, the stamp of every table built from it.

    ``<detector>@<inventory run id>+artifacts@<artifact run id or none>``, such
    as ``kay@056eee0dcb5a+artifacts@fbbff144a123``. Read when called.
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


def swr_epoch_id(nwb_file_name: str, epoch_name: str) -> str:
    """The inventory's epoch id, such as ``chimi20200225_02_r1_HPC``.

    SWRs are detected on CA1's LFP, so the id is the HPC one whatever area an
    analysis decodes.
    """
    return format_epoch_identifier(nwb_file_name, epoch_name, "HPC")


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


def read_observed(path: str) -> NDArray[np.float64]:
    observed = pd.read_parquet(path)
    band = observed.loc[observed["scope"] == OBSERVED_SCOPE, ["start_s", "end_s"]]
    stretches: NDArray[np.float64] = band.to_numpy(dtype=np.float64).reshape(-1, 2)
    return stretches


def artifact_intervals(epoch_id: str) -> pd.DataFrame:
    """The epoch's artifact mask, from the one configured run.

    The run's cores for the epoch, padded by ``artifact_masks.PADDING_S`` (S
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
    try:
        return epoch_artifact_intervals(artifact_intervals_dir(), epoch_id)
    except ArtifactRunError as error:
        raise SwrSourceError(str(error)) from error


@dataclass(frozen=True)
class EpochSource:
    """What the source holds for one epoch, read with one manifest check."""

    events: pd.DataFrame
    observed: NDArray[np.float64]
    """The stored band's stretches less excluded time."""
    artifacts: pd.DataFrame


def epoch_source(epoch_id: str) -> EpochSource:
    files = swr_source_files(epoch_id)
    artifacts = artifact_intervals(epoch_id)
    return EpochSource(
        events=_read_events(epoch_id, files["events"]["path"]),
        observed=subtract_intervals(
            read_observed(files["observed"]["path"]), excluded_time(artifacts)
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


def classify(
    epoch_id: str,
    source: EpochSource,
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
    excluded time needs a decode frame's MUA labels: ``analysed_time.artifact_event_drops``
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
        (the immobility rule alone) and ``drop_reason``: an ``eligibility.DropReason``
        value, or None for a kept event. An event in the epoch's mask is
        ``artifact``, one within the reach of it ``near_artifact``.

    Raises
    ------
    SwrSourceError
        If the epoch cannot be served (:func:`swr_source_files`,
        :func:`artifact_intervals`).
    """
    return classify(epoch_id, epoch_source(epoch_id), speed, speed_threshold)
