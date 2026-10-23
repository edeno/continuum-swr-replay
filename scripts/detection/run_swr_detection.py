#!/usr/bin/env python
"""Run the four SWR detectors on every epoch's CA1 LFP, from shared inputs.

Kay, Karlsson, Yu and Zugaro, from ``ripple_detection`` at its pinned commit,
each with its own settings (``swr_detectors.DETECTOR_SETTINGS``), on one
epoch's shared inputs: the stored CA1 ripple band and the referenced raw LFP
behind it (``get_ripple_band_inputs``), and the head speed and time span that
``load_data`` uses (``get_epoch_position_info``). Design: Stage 3 of
``docs/design/2026-10-03-full-pipeline-refresh-design.md``.

Per epoch:

1. **Band check.** The raw LFP is re-filtered as ``LFPBandV1.make`` filtered it
   (``band_reproduction_error``). A re-filter that does not reproduce the whole
   stored band within tolerance stops the epoch: it is recorded as failed, with
   the reason and the check's values, and nothing is written. A nonzero
   ``reference_overflow_count`` is recorded, not a stop.
2. **Span and speed.** LFP rows outside the position span are dropped, as
   ``filter_ripple_lfp_time`` drops them, and ``head_speed`` is interpolated
   linearly onto the LFP times.
3. **Artifacts.** The epoch's artifact intervals are the configured artifact
   run's cores (``swr_source.ARTIFACT_INTERVALS_RUN_ID``, read once when the
   run starts and recorded in its settings) padded by the one table every
   stage uses (``events.artifact_masks.PADDING_S``: S by 1 s, a step with an LFP
   signature by 0.3 s, a spike-only line by 0.1 s). Their time is the artifact
   mask on the LFP times; with no artifact run configured it is empty. A
   configured run that cannot serve the epoch fails it.
4. **Tetrodes (rule R-T).** ``detection_channels`` applies
   ``ripple_tetrodes.select_ripple_tetrodes`` once, before any detector: a
   CA1 band tetrode whose ripple envelope stays low while the other tetrodes
   ripple (coupling below 1.0, over immobile samples outside the artifact
   mask) is dropped. The band, the raw LFP behind it, the electrode ids, their
   references and their groups are subset to the kept tetrodes, which every
   detector reads and among which Zugaro chooses. The rule fails the epoch
   closed, with its reason (and every tetrode's coupling, once measured)
   recorded, when the band holds two channels of one tetrode, when it would
   drop more than a third of the tetrodes, or when a tetrode's coupling is
   undefined. Reference electrodes are not band channels and are never
   dropped: a kept electrode keeps its own reference. The Kay check names the
   subset among the direct Kay's differences.
5. **Zugaro's channel.** ``best_ripple_channel`` (buzcode's
   ``bz_GetBestRippleChan``) chooses one of the kept CA1 electrodes from their
   referenced raw LFP, and ``zero_phase_bandpass`` filters it. A candidate with
   no finite sample is excluded with that reason, so it costs the others no
   samples. If the channel cannot be chosen or filtered (no channel passes the
   dead-channel guard, say), Zugaro's entry is that error.
6. **Detectors.** ``run_swr_detectors`` runs the four; Kay, Karlsson and Yu read
   the stored band. Each is given its normalization sample less the artifact
   mask's time: every sample for Kay, Karlsson and Zugaro, the immobile ones
   for Yu, whose threshold is then estimated on the masked immobile sample.
   Yu's per-channel z-score before the median still pools every valid sample,
   masked or not, as the package does. The LFP is not NaN-filled. A
   detector's error is recorded and the rest still report: the epoch is
   ``ok`` when all four succeed, ``partial`` when some do, and ``failed``
   when none does.
7. **Event filter.** ``filter_detected_events`` drops every detector's events
   that overlap the artifact mask, counted per detector in the record, and
   renumbers the rest 1..n.
8. **Yu's diagnostics**, recomputed by ``yu_diagnostics`` on the same noise
   sample: the scalars in the manifest, the histogram arrays in the epoch's
   sidecar.
9. **Kay version check** against the canonical Spyglass events
   (``RippleTimesV1``), in two comparisons.

   - ``version_only`` runs Kay at the pinned commit on Spyglass's own inputs
     (``RippleTimesV1.get_ripple_lfps_and_position_info``) with Spyglass's
     stored parameters (``RippleParameters``), so only the package version
     differs. Its events are named ``kay_spyglass_inputs``, so that an event key
     names one event across both comparisons.
   - ``version_and_inputs`` takes the direct Kay inventory, run with
     ``DETECTOR_SETTINGS["kay"]``. The record flags a mismatch between those and
     Spyglass's stored parameters, which then also separates this comparison.

   The canonical events and the version-only events are kept only inside the
   position span, by ``filter_ripple_lfp_time``'s rule, as ``load_data`` keeps
   ``SWR_event``. If Spyglass's own inputs cannot be read, the version-only
   comparison is recorded as that error under ``kay_version_check``, the epoch
   is ``partial`` with that reason, and its inventories and the other
   comparison are still written.

Each epoch writes, under ``Processed-Data/replay_detection_comparison/swr/<run_id>/``:

- ``{epoch_id}_{detector}_events.parquet``: the detector's own event table, with
  all its statistics, for each detector that succeeded;
- ``{epoch_id}_swr_observed.parquet``: where the LFP was observed, one row per
  gap-free stretch, ``scope`` ``"stored_band"`` (the band Kay, Karlsson and Yu
  read) or ``"zugaro"`` (Zugaro's filtered channel, less its filter's edges),
  with ``start_s`` and ``end_s``, the stretch's first and last sample. The
  artifact mask's time is not observed in either scope;
- ``{epoch_id}_kay_version_check.parquet``: ``kay_version_check``'s rows, with a
  ``comparison`` column, ``"version_only"`` or ``"version_and_inputs"``;
- ``{epoch_id}_yu_diagnostics.json``: every field of Yu's threshold
  diagnostics, the histogram arrays included, when Yu's threshold could be
  estimated.

An epoch's other outputs, left by an earlier run, are removed, so the folder
always matches the manifest; a failed epoch keeps none.

``manifest.json`` beside them holds the run's settings (the package commit,
every detector's settings, the Zugaro filter and channel choice) and one record
per epoch: its status and, for an epoch that ran, every detector's outcome, the
provenance, the electrodes and references, Zugaro's channel with every score,
the band check, Yu's diagnostic scalars, the sampling rates, the LFP's units and
conversion, the position span, event counts, the artifact run with the masked
seconds and each detector's dropped events, rule R-T's coupling per tetrode and
dropped electrode groups (``tetrode_selection``), the Kay check's counts, input
hashes, and each output's path and sha256. ``electrode_ids`` and
``reference_ids`` are the kept tetrodes' electrodes and their references, pair
by pair; ``tetrode_selection`` keeps every band electrode's. Only this process
writes it, atomically, after each epoch finishes. A rerun under the same run id
skips the epochs already ``ok`` unless ``--overwrite`` is given, removes the
record of each epoch it redoes before starting it, and refuses settings other
than the ones the manifest was written with. A run holds an exclusive lock on its
folder's ``.lock`` throughout, so a second run under the same run id exits at
once instead of overwriting the first one's records.

The epochs are the epoch register's analysed ones, or with ``--pending-only``
its pending ones alone, for the incremental update after their populates
(``epochs_to_run``, read against ``get_epoch_info``), narrowed by ``--animal``,
``--date`` and ``--epoch``. An epoch the register does not analyse (no inputs, another task,
and so on: ``docs/epoch_register.md``) never reaches the runner, so the manifest
records none. The epochs run in parallel on a
dask ``LocalCluster`` of ``--n-workers`` processes, one epoch per worker at a
time, each logging to ``logs/{epoch_id}_swr.log`` under the working directory.
An epoch that fails is recorded with its error and the run goes on; the run
then exits non-zero.

Usage:
    python scripts/detection/run_swr_detection.py --animal chimi --date 20200225 \\
        --epoch 02_r1 --n-workers 1
    python scripts/detection/run_swr_detection.py --n-workers 8   # every epoch
"""

import argparse
import functools
import json
import math
import resource
import sys
import time as clock
import traceback
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
from dask.distributed import LocalCluster
from numpy.typing import NDArray
from ripple_detection import Kay_ripple_detector

from continuum_swr_replay.analysis.events import swr_source
from continuum_swr_replay.analysis.events.artifact_intervals import (
    epoch_artifact_intervals,
)
from continuum_swr_replay.analysis.events.artifact_masks import (
    PADDING_S,
)
from continuum_swr_replay.analysis.hippocampus.ripple_tetrodes import (
    COUPLING_CUT,
    LOO_SMOOTHING_SIGMA_S,
    LOO_THRESHOLD_Z,
    MAXIMUM_DROPPED_FRACTION,
    REFERENCE_SPEED_CM_PER_S,
    TetrodeSelection,
    TetrodeSelectionError,
    check_one_channel_per_tetrode,
    select_ripple_tetrodes,
)
from continuum_swr_replay.analysis.hippocampus.swr_detectors import (
    CHANNEL_SELECTION_BAND_HZ,
    CHANNEL_SELECTION_FILTER_ORDER,
    CHANNEL_SELECTION_RMS_WINDOW_S,
    DEAD_CHANNEL_RMS,
    DETECTOR_SETTINGS,
    SETTLING_TOLERANCE,
    ZUGARO_BAND_HZ,
    ZUGARO_FILTER_ORDER,
    ZUGARO_MAXIMUM_DURATION_S,
    NoUsableChannelError,
    band_reproduction_error,
    best_ripple_channel,
    drop_overlapping_events,
    gap_free_stretches,
    run_swr_detectors,
    yu_diagnostics,
    zero_phase_bandpass,
)
from continuum_swr_replay.analysis.inventory_comparison.kay_version import (
    kay_version_check,
)
from continuum_swr_replay.analysis.inventory_comparison.schema import SessionIdentity
from continuum_swr_replay.data_loaders.epoch_register import epochs_to_run
from continuum_swr_replay.data_loaders.lfp import (
    _ripple_keys,
    get_ripple_band_inputs,
    get_spyglass_kay_inputs,
)
from continuum_swr_replay.data_loaders.position import get_epoch_position_info
from continuum_swr_replay.data_loaders.utils import parse_nwb_filename
from continuum_swr_replay.intervals import (
    interval_mask,
    masked_seconds,
)
from continuum_swr_replay.parameters import ROOT_DIR
from continuum_swr_replay.paths import SWR_INVENTORY_ROOT
from continuum_swr_replay.runs.batch import (
    raise_open_file_limit,
    run_epoch_logged,
    run_epochs,
    run_lock,
)
from continuum_swr_replay.runs.cli import filter_epoch_info, parse_run_id
from continuum_swr_replay.runs.io import (
    array_sha256,
    json_ready,
    replace_outputs,
    write_json,
)
from continuum_swr_replay.runs.provenance import provenance_record

OUTPUT_ROOT = SWR_INVENTORY_ROOT
"""Where the run folders go: where the paper's SWR source reads them."""
PRODUCED_BY = "scripts/detection/run_swr_detection.py"
DESIGN_SPEC = "docs/design/2026-10-03-full-pipeline-refresh-design.md"

SPYGLASS_INPUTS_KAY = "kay_spyglass_inputs"
"""The Kay check's name for Kay run on Spyglass's own inputs (version only)."""


def epoch_id(nwb_file_name: str, epoch_name: str) -> str:
    """``{stem}_{epoch}_HPC``, the decoders' name for the epoch's CA1 data.

    ``epoch_id("chimi20200225_.nwb", "02_r1")`` is ``"chimi20200225_02_r1_HPC"``.
    The rule is ``swr_source.swr_epoch_id``, which the other runners call.
    """
    return swr_source.swr_epoch_id(nwb_file_name, epoch_name)


def output_paths(identifier: str, output_dir: Path) -> dict[str, Path]:
    """Every file an epoch can write, keyed by detector or table name."""
    paths = swr_source.swr_inventory_paths(identifier, output_dir, DETECTOR_SETTINGS)
    paths["kay_version_check"] = output_dir / f"{identifier}_kay_version_check.parquet"
    paths["yu_diagnostics"] = output_dir / f"{identifier}_yu_diagnostics.json"
    return paths


def canonical_kay_events(nwb_file_name: str, epoch_name: str) -> pd.DataFrame:
    """The canonical Spyglass Kay events: ``RippleTimesV1``'s, read only.

    Restricted by ``_ripple_keys``, as ``get_ripple`` reads them; one row of
    ``RippleTimesV1`` must match.
    """
    from spyglass.ripple.v1 import RippleTimesV1

    ripple_key, _ = _ripple_keys(nwb_file_name, epoch_name)
    return (RippleTimesV1 & ripple_key).fetch1_dataframe()


def spyglass_kay_inputs(
    nwb_file_name: str, epoch_name: str
) -> tuple[pd.Series, pd.DataFrame, float, dict]:
    """The inputs and parameters ``RippleTimesV1.make`` gave Kay for the
    canonical events, read only, as ``make`` reads them.

    ``get_spyglass_kay_inputs``: ``RippleTimesV1.get_ripple_lfps_and_position_info``'s
    steps on the epoch's full ``RippleTimesV1`` key, with the position read
    from the source its merge entry names, as ``load_data`` reads it (Spyglass's
    own read refuses j1620210710's two entries held in two parts), and that
    key's ``RippleParameters`` row.

    Returns
    -------
    speed : pd.Series
        Head speed in cm/s, interpolated onto the LFP times.
    ripple_lfps : pd.DataFrame, shape (n_time, n_channels)
        The stored ripple band, indexed by time in seconds.
    sampling_frequency : float
        The band's nominal rate in Hz.
    ripple_param_dict : dict
        The stored parameters: ``speed_name``, ``ripple_detection_algorithm``
        and ``ripple_detection_params``.
    """
    return get_spyglass_kay_inputs(nwb_file_name, epoch_name)


def kay_on_spyglass_inputs(
    speed: pd.Series,
    ripple_lfps: pd.DataFrame,
    sampling_frequency: float,
    ripple_param_dict: Mapping[str, Any],
) -> pd.DataFrame:
    """Kay at the pinned commit on Spyglass's own inputs, as ``make`` calls it.

    With Spyglass's stored ``ripple_detection_params``, so only the package
    version differs from the canonical run.

    Raises
    ------
    ValueError
        If the stored algorithm is not ``Kay_ripple_detector``: the canonical
        events then cannot check Kay's version.
    """
    algorithm = ripple_param_dict["ripple_detection_algorithm"]
    if algorithm != "Kay_ripple_detector":
        msg = (
            f"Spyglass's canonical events come from {algorithm!r}, not "
            "Kay_ripple_detector, so they cannot check Kay's version."
        )
        raise ValueError(msg)
    return Kay_ripple_detector(
        time=np.asarray(ripple_lfps.index),
        filtered_lfps=np.asarray(ripple_lfps),
        speed=np.asarray(speed),
        sampling_frequency=sampling_frequency,
        **ripple_param_dict["ripple_detection_params"],
    )


def within_span(events: pd.DataFrame, span_s: tuple[float, float]) -> pd.DataFrame:
    """The events wholly inside ``span_s``, by ``filter_ripple_lfp_time``'s rule."""
    keep = (events["start_time"] >= span_s[0]) & (events["end_time"] <= span_s[1])
    return events.loc[keep]


def observation_table(
    time: NDArray,
    ripple_band: NDArray,
    zugaro_lfp: NDArray,
    unobserved: NDArray[np.bool_],
) -> pd.DataFrame:
    """Where each input was observed: one row per gap-free stretch.

    Parameters
    ----------
    time : NDArray, shape (n_time,)
        Timestamps in seconds.
    ripple_band : NDArray, shape (n_time, n_channels)
        The stored band, scope ``"stored_band"``.
    zugaro_lfp : NDArray, shape (n_time, 1)
        Zugaro's filtered channel, NaN where unobserved, scope ``"zugaro"``.
    unobserved : NDArray of bool, shape (n_time,)
        Rows neither scope observed, such as the artifact mask: they
        end a stretch as a non-finite row does.

    Returns
    -------
    pd.DataFrame
        ``scope``, and ``start_s`` and ``end_s``, the times of a stretch's first
        and last sample.
    """
    rows = [
        (scope, time[start], time[stop - 1])
        for scope, values in (("stored_band", ripple_band), ("zugaro", zugaro_lfp))
        # one value per row, NaN where the row is unobserved, so the input is
        # not copied
        for start, stop in gap_free_stretches(
            time,
            np.where(np.isfinite(values).all(axis=1) & ~unobserved, 0.0, np.nan),
        )
    ]
    return pd.DataFrame(rows, columns=["scope", "start_s", "end_s"]).astype(
        {"scope": object, "start_s": np.float64, "end_s": np.float64}
    )


def measured_rate(time: NDArray, values: NDArray) -> float:
    """Samples per second within the gap-free stretches: steps over elapsed time.

    Not ``1 / median step``, which the float64 rounding of Unix timestamps
    biases: near 1.58e9 s a timestamp resolves 2**-22 s, so exact 1 ms steps
    are stored as 4194 or 4195 of those, and ``1 / median step`` reads
    1000.07 Hz for an exact 1000 Hz.

    Parameters
    ----------
    time : NDArray, shape (n_time,)
        Timestamps in seconds.
    values : NDArray, shape (n_time, n_channels)
        The samples, whose non-finite rows and timestamp gaps end a stretch.

    Returns
    -------
    float
        The rate in Hz; NaN when no stretch has two samples.
    """
    stretches = gap_free_stretches(time, values)
    n_steps = int(np.sum(stretches[:, 1] - stretches[:, 0] - 1))
    elapsed = float(np.sum(time[stretches[:, 1] - 1] - time[stretches[:, 0]]))
    return n_steps / elapsed if elapsed > 0 else math.nan


def choose_zugaro_channel(
    raw_referenced: NDArray,
    time: NDArray,
    sampling_frequency: float,
    electrode_ids: NDArray,
) -> tuple[dict, NDArray, Exception | None]:
    """Choose and filter Zugaro's channel among the band's CA1 electrodes.

    A candidate with no finite sample is excluded with that reason before
    ``best_ripple_channel`` runs on the rest, which treats a row with any NaN
    as unobserved on every channel. Any error in choosing or filtering the
    channel is Zugaro's alone: it is returned, not raised, so the other
    detectors still run.

    Parameters
    ----------
    raw_referenced : NDArray, shape (n_time, n_channels)
        The candidates' referenced raw LFP at the band's timestamps.
    time : NDArray, shape (n_time,)
        Timestamps in seconds.
    sampling_frequency : float
        Nominal rate in Hz.
    electrode_ids : NDArray, shape (n_channels,)
        Each column's electrode.

    Returns
    -------
    channel : dict
        ``electrode_id`` (None when no channel was chosen), ``scores`` (every
        candidate's, NaN where it has none) and ``excluded`` (with reasons),
        plus ``error`` when the channel could not be chosen or filtered.
    zugaro_lfp : NDArray, shape (n_time, 1)
        The chosen channel band-passed by ``zero_phase_bandpass`` at
        ``ZUGARO_BAND_HZ``; all NaN after an error.
    error : Exception or None
        The error, such as ``NoUsableChannelError``, or None.
    """
    has_finite = np.isfinite(raw_referenced).any(axis=0)
    excluded = {int(e): "no finite sample" for e in electrode_ids[~has_finite]}
    scores: dict[int, float] = dict.fromkeys(excluded, math.nan)
    channel: dict[str, Any] = {
        "electrode_id": None,
        "scores": scores,
        "excluded": excluded,
    }
    try:
        if not has_finite.any():
            raise NoUsableChannelError("No candidate channel has a finite sample.")
        candidates = (
            raw_referenced if has_finite.all() else raw_referenced[:, has_finite]
        )
        choice = best_ripple_channel(
            candidates, time, sampling_frequency, electrode_ids[has_finite]
        )
        channel = {
            "electrode_id": choice.electrode_id,
            "scores": dict(sorted((choice.scores | scores).items())),
            "excluded": dict(sorted((choice.excluded | excluded).items())),
        }
        column = int(np.flatnonzero(electrode_ids == choice.electrode_id)[0])
        zugaro_lfp = zero_phase_bandpass(
            raw_referenced[:, [column]],
            time,
            sampling_frequency,
            ZUGARO_BAND_HZ,
            ZUGARO_FILTER_ORDER,
        )
    except Exception as error:  # noqa: BLE001 - Zugaro's failure, not the epoch's
        unobserved = np.full((time.size, 1), np.nan)
        return channel | {"error": repr(error)}, unobserved, error
    return channel, zugaro_lfp, None


def detection_channels(
    ripple_band: NDArray,
    speed: NDArray,
    unobserved: NDArray[np.bool_],
    sampling_frequency: float,
    electrode_group_names: NDArray,
) -> TetrodeSelection:
    """The band's tetrodes the four detectors read, and Zugaro chooses among.

    The one place an epoch's detecting tetrodes are chosen: rule R-T
    (``ripple_tetrodes.select_ripple_tetrodes``), on a band with one channel
    per tetrode (``check_one_channel_per_tetrode``).

    Parameters
    ----------
    ripple_band : NDArray, shape (n_time, n_channels)
        The stored band within the position span, one channel per tetrode.
    speed : NDArray, shape (n_time,)
        Head speed in cm/s on the band's times.
    unobserved : NDArray of bool, shape (n_time,)
        The artifact mask on the band's times.
    sampling_frequency : float
        The band's nominal rate in Hz.
    electrode_group_names : NDArray, shape (n_channels,)
        Each column's electrode group (its tetrode).

    Returns
    -------
    TetrodeSelection
        ``keep`` is True for each kept column of ``ripple_band`` and
        ``raw_referenced``.

    Raises
    ------
    TetrodeSelectionError
        When the rule fails the epoch closed.
    """
    check_one_channel_per_tetrode(electrode_group_names)
    return select_ripple_tetrodes(ripple_band, speed, unobserved, sampling_frequency)


def filter_detected_events(
    results: Mapping[str, pd.DataFrame | Exception], artifact_mask: pd.DataFrame
) -> tuple[dict[str, pd.DataFrame | Exception], dict[str, int | None]]:
    """Every detector's events after the post-detection filter.

    The one place detected events are vetoed. Today the filter drops each
    event that overlaps the artifact intervals
    (``drop_overlapping_events``: closed, touching included) and renumbers
    the rest 1..n. A detector's error passes through.

    Parameters
    ----------
    results : Mapping
        ``run_swr_detectors``' results, by detector.
    artifact_mask : pd.DataFrame
        ``start_time`` and ``end_time`` in seconds.

    Returns
    -------
    filtered : dict
        By detector, the kept events or the error.
    n_dropped : dict
        By detector, the events dropped; None for a detector that failed.
    """
    filtered: dict[str, pd.DataFrame | Exception] = {}
    n_dropped: dict[str, int | None] = {}
    for name, events in results.items():
        if isinstance(events, pd.DataFrame):
            filtered[name], n_dropped[name] = drop_overlapping_events(
                events, artifact_mask
            )
        else:
            filtered[name], n_dropped[name] = events, None
    return filtered, n_dropped


def kay_check_counts(kay_check: pd.DataFrame) -> dict:
    """The Kay check's events by comparison, detector and status.

    Each count is taken over every event (``"all"``) and over the whole-event
    speed-eligible ones (``"speed_eligible"``). An eligible event keeps the
    status it got against every event of the other set, eligible or not.
    """
    counts: dict[str, dict[str, dict[str, dict[str, int]]]] = {}
    for (comparison, detector), rows in kay_check.groupby(
        ["comparison", "detector"], sort=False
    ):
        eligible = rows["whole_event_speed_eligible"].to_numpy(dtype=bool)
        counts.setdefault(comparison, {})[detector] = {
            "all": rows["status"].value_counts().to_dict(),
            "speed_eligible": rows.loc[eligible, "status"].value_counts().to_dict(),
        }
    return counts


def _detect(
    nwb_file_name: str,
    epoch_name: str,
    *,
    run_id: str,
    artifact_dir: Path | None,
    fetch_inputs: Callable,
    fetch_position: Callable,
    fetch_canonical_kay: Callable,
    fetch_spyglass_kay_inputs: Callable,
    fir_filter: Callable | None,
) -> tuple[dict, dict[str, pd.DataFrame | dict]]:
    """One epoch's record, and the outputs to write (none for a failed epoch):
    tables by name, and the Yu sidecar's content."""
    # before any input is read: a configured artifact run that cannot serve
    # the epoch fails it
    artifact_mask = epoch_artifact_intervals(
        artifact_dir, epoch_id(nwb_file_name, epoch_name)
    )
    inputs = fetch_inputs(nwb_file_name, epoch_name)
    band_check: dict[str, Any] = {
        "max_abs_difference": None,
        "tolerance": None,
        "band_dtype": inputs["ripple_band_dtype"],
        "reference_overflow_count": inputs["reference_overflow_count"],
    }
    try:
        difference, tolerance = band_reproduction_error(inputs, fir_filter)
    except ValueError as error:  # such as a re-filter missing stored timestamps
        reason = f"The band check could not be completed: {error}"
        return {"status": "failed", "reason": reason, "band_check": band_check}, {}
    band_check["max_abs_difference"], band_check["tolerance"] = difference, tolerance
    if not difference <= tolerance:  # NaN fails too
        reason = (
            f"The re-filtered raw LFP differs from the stored band by up to "
            f"{difference:g}, more than the tolerance {tolerance:g}, so the "
            "referenced channels are not the ones the band was made from."
        )
        return {"status": "failed", "reason": reason, "band_check": band_check}, {}

    position_info = fetch_position(nwb_file_name, epoch_name)["position_info"]
    position_time = position_info.index.to_numpy(dtype=np.float64)
    head_speed = position_info["head_speed"].to_numpy(dtype=np.float64)
    span_s = (float(position_time[0]), float(position_time[-1]))
    # The band's timestamps ascend (get_ripple_band_inputs checks them against
    # the raw ones), so the span is one slice, and the arrays stay views.
    first = int(np.searchsorted(inputs["time"], span_s[0], side="left"))
    stop = int(np.searchsorted(inputs["time"], span_s[1], side="right"))
    time = inputs["time"][first:stop]
    ripple_band = inputs["ripple_band"][first:stop]
    raw_referenced = inputs["raw_referenced"][first:stop]
    electrode_ids = inputs["electrode_ids"]
    reference_ids = inputs["reference_ids"]
    electrode_group_names = inputs["electrode_group_names"]
    speed = np.interp(time, position_time, head_speed, left=np.nan, right=np.nan)
    sampling_frequency = inputs["sampling_frequency"]
    unobserved = interval_mask(time, artifact_mask)

    try:
        selection = detection_channels(
            ripple_band, speed, unobserved, sampling_frequency, electrode_group_names
        )
    except TetrodeSelectionError as error:
        failed: dict[str, Any] = {
            "status": "failed",
            "reason": f"Rule R-T failed the epoch closed: {error}",
            "band_check": band_check,
        }
        if error.selection is not None:  # None: the rule could not start
            failed["tetrode_selection"] = error.selection.record(
                electrode_ids, electrode_group_names, reference_ids
            )
        return failed, {}
    tetrode_selection = selection.record(
        electrode_ids, electrode_group_names, reference_ids
    )
    channels = selection.keep
    if not channels.all():  # a copy, made only when a channel is left out
        ripple_band = ripple_band[:, channels]
        raw_referenced = raw_referenced[:, channels]
        electrode_ids = electrode_ids[channels]
        # each kept electrode with its own reference; no reference is dropped
        reference_ids = reference_ids[channels]
        electrode_group_names = electrode_group_names[channels]

    zugaro_channel, zugaro_lfp, channel_error = choose_zugaro_channel(
        raw_referenced, time, sampling_frequency, electrode_ids
    )
    results = run_swr_detectors(
        time, ripple_band, zugaro_lfp, speed, sampling_frequency, unobserved=unobserved
    )
    if channel_error is not None:
        # Zugaro was given an unobserved channel, which it refuses; its entry
        # is the error that left it without a filtered channel instead
        results["zugaro"] = channel_error
    zugaro_events = results["zugaro"]
    # read at once: pandas attrs do not reliably survive a copy or a concat
    zugaro_dropped = (
        int(zugaro_events.attrs["n_dropped_over_maximum_duration"])
        if isinstance(zugaro_events, pd.DataFrame)
        else None
    )
    results, n_dropped = filter_detected_events(results, artifact_mask)
    succeeded = [
        name for name, events in results.items() if isinstance(events, pd.DataFrame)
    ]
    try:
        yu = yu_diagnostics(
            time, ripple_band, speed, sampling_frequency, unobserved=unobserved
        )
    except ValueError as error:  # where Yu itself cannot estimate its threshold
        yu = {"error": repr(error)}

    record = {
        "status": (
            "ok"
            if len(succeeded) == len(results)
            else "partial"
            if succeeded
            else "failed"
        ),
        "detectors": {
            name: "ok" if name in succeeded else repr(events)
            for name, events in results.items()
        },
        "nwb_file_name": nwb_file_name,
        "epoch_name": epoch_name,
        "produced_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "provenance": provenance_record(ROOT_DIR)
        | {"numpy_version": np.__version__, "scipy_version": scipy.__version__},
        "electrode_ids": electrode_ids,
        "reference_ids": reference_ids,
        "tetrode_selection": tetrode_selection,
        "zugaro_channel": zugaro_channel,
        "band_check": band_check,
        # the scalars; the histogram arrays go to the epoch's Yu sidecar
        "yu_diagnostics": {
            name: value for name, value in yu.items() if not isinstance(value, list)
        },
        "sampling": {
            "nominal": sampling_frequency,
            "measured": measured_rate(time, ripple_band),
        },
        "lfp_filter_name": inputs["lfp_filter_name"],
        "lfp_units": inputs["lfp_units"],
        "lfp_conversion": inputs["lfp_conversion"],
        "position_span_s": span_s,
        "n_events": {
            name: len(events) if isinstance(events, pd.DataFrame) else None
            for name, events in results.items()
        },
        "zugaro_dropped_over_150ms": zugaro_dropped,
        "artifacts": {
            "run_id": None if artifact_dir is None else artifact_dir.name,
            "n_intervals": len(artifact_mask),
            # the masked time within the LFP rows read
            "masked_s": masked_seconds(artifact_mask.clip(time[0], time[-1])),
            "n_dropped": n_dropped,
        },
        # the distinct electrode groups (tetrodes) of the detecting electrodes;
        # their anatomy is not checked here, hence "unknown" below
        "n_detecting_tetrodes": int(np.unique(electrode_group_names).size),
        "meets_published_tetrode_criterion": "unknown",
        "input_sha256": {
            "time": array_sha256(time),
            "ripple_band": array_sha256(ripple_band),
            "raw_referenced": array_sha256(raw_referenced),
        },
    }
    if not succeeded:
        record["reason"] = "Every detector failed; see detectors."
        return record, {}

    animal, day = parse_nwb_filename(nwb_file_name)
    session = SessionIdentity(
        animal=animal, day=day, epoch=epoch_name, brain_area="HPC"
    )
    canonical = within_span(fetch_canonical_kay(nwb_file_name, epoch_name), span_s)
    compared: dict[str, tuple[str, pd.DataFrame]] = {}
    spyglass_parameters: dict | None = None
    try:
        spyglass_speed, spyglass_lfps, spyglass_rate, spyglass_parameters = (
            fetch_spyglass_kay_inputs(nwb_file_name, epoch_name)
        )
    except Exception as error:  # noqa: BLE001 - recorded; the detectors stand
        # Only the version-only comparison reads Spyglass's own inputs, so a
        # failure to read them (a PositionOutput row with two parents, say)
        # loses that comparison, not the epoch's inventories.
        spyglass_error: str | None = repr(error)
    else:
        spyglass_error = None
        # Kay on Spyglass's inputs is named apart from the direct Kay, so that
        # an event key names one event across both comparisons
        compared["version_only"] = (
            SPYGLASS_INPUTS_KAY,
            within_span(
                kay_on_spyglass_inputs(
                    spyglass_speed, spyglass_lfps, spyglass_rate, spyglass_parameters
                ),
                span_s,
            ),
        )
    if "kay" in succeeded:
        compared["version_and_inputs"] = ("kay", results["kay"])
    kay_check = (
        pd.concat(
            [
                kay_version_check(
                    direct,
                    canonical,
                    session=session,
                    run_id=run_id,
                    speed_time_s=position_time,
                    speed_cm_per_s=head_speed,
                    direct_detector=name,
                ).assign(comparison=comparison)
                for comparison, (name, direct) in compared.items()
            ],
            ignore_index=True,
        )
        if compared
        else None
    )
    # "unknown" when Spyglass's stored parameters could not be read
    settings_mismatch: bool | str = (
        "unknown"
        if spyglass_parameters is None
        else dict(spyglass_parameters["ripple_detection_params"])
        != dict(DETECTOR_SETTINGS["kay"])
    )
    differences = {
        "version_only": ["package version"],
        "version_and_inputs": ["package version", "inputs"]
        + (
            ["settings unknown"]
            if settings_mismatch == "unknown"
            else ["settings"]
            if settings_mismatch
            else []
        )
        # the direct Kay is normalized and filtered with the artifact mask;
        # the canonical events are not
        + ([] if artifact_dir is None else ["artifact mask"])
        # and it reads only the tetrodes R-T kept; the canonical Kay read all
        + ([] if channels.all() else ["tetrode selection (R-T)"]),
    }
    record["kay_version_check"] = {
        "spyglass_parameters": spyglass_parameters,
        "settings_mismatch": settings_mismatch,
        # what each comparison's events differ from the canonical ones in
        "differences": {name: differences[name] for name in compared},
        "counts": {} if kay_check is None else kay_check_counts(kay_check),
    }
    if spyglass_error is not None:
        record["kay_version_check"]["version_only"] = {"error": spyglass_error}
        record["status"] = "partial"
        record["reason"] = (
            "The version-only Kay check did not run: Spyglass's own Kay inputs "
            f"could not be read ({spyglass_error})."
        )

    frames = {
        name: events
        for name, events in results.items()
        if isinstance(events, pd.DataFrame)
    }
    frames["swr_observed"] = observation_table(
        time, ripple_band, zugaro_lfp, unobserved
    )
    if kay_check is not None:
        frames["kay_version_check"] = kay_check
    stamp = {
        "produced_by": PRODUCED_BY,
        "run_id": run_id,
        "epoch_id": epoch_id(nwb_file_name, epoch_name),
        "commit": record["provenance"]["commit"],
    }
    for frame in frames.values():
        frame.attrs = frame.attrs | stamp
    tables: dict[str, pd.DataFrame | dict] = dict(frames)
    if "error" not in yu:
        tables["yu_diagnostics"] = stamp | {"yu_diagnostics": yu}
    return record, tables


def run_epoch(
    nwb_file_name: str,
    epoch_name: str,
    *,
    output_dir: Path,
    fetch_inputs: Callable = get_ripple_band_inputs,
    fetch_position: Callable = get_epoch_position_info,
    fetch_canonical_kay: Callable = canonical_kay_events,
    fetch_spyglass_kay_inputs: Callable = spyglass_kay_inputs,
    fir_filter: Callable | None = None,
    run_id: str | None = None,
    artifact_dir: Path | None = None,
) -> dict:
    """Run the four detectors and the checks on one epoch, and write the outputs.

    Parameters
    ----------
    nwb_file_name, epoch_name : str
        The epoch, such as ``"chimi20200225_.nwb"`` and ``"02_r1"``.
    output_dir : Path
        The run's folder, where the outputs are written.
    fetch_inputs : Callable, optional
        ``(nwb_file_name, epoch_name) -> RippleBandInputs``, as
        ``get_ripple_band_inputs`` returns them.
    fetch_position : Callable, optional
        ``(nwb_file_name, epoch_name) -> PositionInfoDict``, whose
        ``position_info`` gives the span and ``head_speed``.
    fetch_canonical_kay : Callable, optional
        ``(nwb_file_name, epoch_name) -> pd.DataFrame`` of canonical events.
    fetch_spyglass_kay_inputs : Callable, optional
        ``(nwb_file_name, epoch_name) -> (speed, ripple_lfps,
        sampling_frequency, ripple_param_dict)``, as ``spyglass_kay_inputs``
        returns them.
    fir_filter : Callable, optional
        The band check's filter, passed to ``band_reproduction_error``; None
        is Spyglass's own ``FirFilterParameters().filter_data``.
    run_id : str, optional
        Recorded on the Kay check's events; default ``output_dir.name``.
    artifact_dir : Path, optional
        The artifact run folder whose cores, padded by ``PADDING_S``, are the
        epoch's artifact mask; None masks nothing. ``run`` passes the
        SWR source's configured run.

    Returns
    -------
    dict
        The epoch's manifest record, in strict JSON types. ``status`` is
        ``"failed"`` with
        ``reason`` and ``band_check`` when the band check fails, with
        ``reason`` and every detector's error when all four fail, or with
        ``error`` (and ``traceback``) for any other exception; else ``"ok"``
        or ``"partial"``, with every field the module docstring lists.
    """
    started = clock.perf_counter()
    identifier = epoch_id(nwb_file_name, epoch_name)
    try:
        record, tables = _detect(
            nwb_file_name,
            epoch_name,
            run_id=output_dir.name if run_id is None else run_id,
            artifact_dir=artifact_dir,
            fetch_inputs=fetch_inputs,
            fetch_position=fetch_position,
            fetch_canonical_kay=fetch_canonical_kay,
            fetch_spyglass_kay_inputs=fetch_spyglass_kay_inputs,
            fir_filter=fir_filter,
        )
        record["outputs"] = replace_outputs(
            output_paths(identifier, output_dir), tables
        )
    except Exception as error:  # noqa: BLE001 - recorded; the run goes on
        replace_outputs(output_paths(identifier, output_dir), {})
        record = {
            "status": "failed",
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }
    record["wall_clock_s"] = clock.perf_counter() - started
    # the worker process's peak resident set over its whole life so far, so
    # every epoch it ran before counts too (ru_maxrss is in KiB on Linux)
    record["worker_peak_rss_gib"] = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    )
    ready: dict = json_ready(record)
    return ready


def summary(record: dict) -> str:
    """One epoch's outcome, as the logs and the console report it."""
    status = record["status"]
    if "n_events" not in record:
        return f"{status}: {record.get('reason') or record.get('error')}"
    events = ", ".join(
        f"{name} {'error' if n is None else n}"
        for name, n in record["n_events"].items()
    )
    band = record["band_check"]
    parts = [
        f"{status}: events {events}",
        f"band check {band['max_abs_difference']} (tolerance {band['tolerance']}, "
        f"{band['reference_overflow_count']} overflows)",
        f"Zugaro channel {record['zugaro_channel']['electrode_id']}",
    ]
    checks = record.get("kay_version_check", {}).get("counts")
    if checks:  # none when neither comparison could run
        parts.append(
            "Kay check "
            + " | ".join(
                f"{comparison}: "
                + ", ".join(
                    f"{detector} {counts['all']}"
                    for detector, counts in detectors.items()
                )
                for comparison, detectors in checks.items()
            )
        )
    if "reason" in record:
        parts.append(record["reason"])
    parts.append(
        f"{record['wall_clock_s']:.0f} s, worker peak RSS "
        f"{record['worker_peak_rss_gib']:.1f} GiB"
    )
    return "; ".join(parts)


def _run_epoch_logged(
    nwb_file_name: str, epoch_name: str, run_epoch_kwargs: dict, log_directory: Path
) -> dict:
    """One dask task: ``run_epoch``, logged to ``logs/{epoch_id}_swr.log``.

    When the epoch ends, pass or fail, the NWB files Spyglass opened for it are
    closed (``release_nwb_files``): ``main``'s workers run one task at a time
    (``threads_per_worker=1``), so no other epoch in the process reads them.
    """
    identifier = epoch_id(nwb_file_name, epoch_name)
    return run_epoch_logged(
        lambda log: run_epoch(nwb_file_name, epoch_name, **run_epoch_kwargs),
        log_name=f"{identifier}_swr",
        log_path=log_directory / f"{identifier}_swr.log",
        start=f"{identifier}: run {run_epoch_kwargs}",
        describe=summary,
        log_provenance=True,
    )


def run_settings(provenance: dict, artifact_run_id: str | None) -> dict:
    """What every epoch of a run shares; a rerun under its run id must match.

    Parameters
    ----------
    provenance : dict
        ``provenance_record``'s, for the ``ripple_detection`` commit.
    artifact_run_id : str or None
        The artifact run whose mask every epoch gets; None for none.
    """
    settings: dict = json_ready(
        {
            "ripple_detection_commit": provenance["ripple_detection_commit"],
            "artifacts": {
                "run_id": artifact_run_id,
                "padding_s": PADDING_S,
            },
            "detectors": DETECTOR_SETTINGS,
            "tetrode_selection": {
                "rule": "R-T",
                "coupling_cut": COUPLING_CUT,
                "loo_threshold_z": LOO_THRESHOLD_Z,
                "loo_smoothing_sigma_s": LOO_SMOOTHING_SIGMA_S,
                "reference_speed_cm_per_s": REFERENCE_SPEED_CM_PER_S,
                "maximum_dropped_fraction": str(MAXIMUM_DROPPED_FRACTION),
            },
            "zugaro": {
                "filter": "Butterworth band-pass, second-order sections, forward "
                "and backward (sosfiltfilt) over each gap-free stretch, padded "
                "(odd) and edge-excluded by its settling length",
                "band_hz": ZUGARO_BAND_HZ,
                "order": ZUGARO_FILTER_ORDER,
                "settling_tolerance": SETTLING_TOLERANCE,
                "maximum_duration_s": ZUGARO_MAXIMUM_DURATION_S,
            },
            "channel_selection": {
                "rule": "buzcode bz_GetBestRippleChan (0969ddf): mean / median "
                "of a centered moving RMS; the highest wins, a tie goes to the "
                "lowest electrode id",
                "band_hz": CHANNEL_SELECTION_BAND_HZ,
                "order": CHANNEL_SELECTION_FILTER_ORDER,
                "rms_window_s": CHANNEL_SELECTION_RMS_WINDOW_S,
                "dead_channel_rms": DEAD_CHANNEL_RMS,
            },
            "speed": "head_speed of get_epoch_position_info, interpolated "
            "linearly onto the LFP times; LFP outside the position span dropped",
        }
    )
    return settings


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--animal", help="Only this animal's epochs, such as chimi.")
    parser.add_argument(
        "--date", type=int, help="Only epochs recorded on this date, such as 20200225."
    )
    parser.add_argument("--epoch", help="Only this epoch, such as 02_r1.")
    parser.add_argument(
        "--pending-only",
        action="store_true",
        help="Run only the epochs the epoch register holds pending a Spyglass "
        "populate, after their populates, instead of its analysed epochs.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Redo epochs already written."
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=8,
        help="Dask worker processes, each running one epoch at a time (default 8).",
    )
    parser.add_argument(
        "--run-id",
        type=parse_run_id,
        help="The run's folder under --output-root: letters, digits, '.', '_' "
        "and '-'. Defaults to the first 12 characters of git rev-parse HEAD.",
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args(argv)

    # before the cluster starts: its worker processes inherit this limit
    old_soft, new_soft = raise_open_file_limit()
    print(f"open-file soft limit {old_soft} -> {new_soft}", flush=True)
    provenance = provenance_record(ROOT_DIR)
    run_id = provenance["commit"][:12] if args.run_id is None else args.run_id
    if provenance["dirty"]:
        print(
            f"warning: uncommitted changes; run {run_id} cannot be tied to a commit",
            flush=True,
        )
    output_dir = args.output_root / run_id
    with run_lock(output_dir):
        run(args, provenance, output_dir)


def run(args: argparse.Namespace, provenance: dict, output_dir: Path) -> None:
    """``main``'s run, under the run folder's lock.

    Parameters
    ----------
    args : argparse.Namespace
        ``main``'s arguments.
    provenance : dict
        ``provenance_record``'s, for the settings.
    output_dir : Path
        The run's folder, named for its run id.
    """
    run_id = output_dir.name
    # read once, here: the workers are other processes, and every epoch of the
    # run must be masked by the same artifact run
    artifact_dir = swr_source.artifact_intervals_dir()
    settings = run_settings(
        provenance, None if artifact_dir is None else artifact_dir.name
    )
    selected = epochs_to_run(args.pending_only)
    epochs = list(dict.fromkeys(filter_epoch_info(selected, args).index))
    if not epochs:
        sys.exit("No epochs match --animal, --date and --epoch.")

    manifest_path = output_dir / "manifest.json"
    records: dict[str, dict] = {}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["settings"] != settings:
            sys.exit(
                f"{manifest_path} was written with other settings; a run must "
                "not mix them. Start a new --run-id."
            )
        records = previous["epochs"]

    def write_manifest() -> None:
        write_json(
            manifest_path,
            {
                "produced_by": PRODUCED_BY,
                "run_id": run_id,
                "design_spec": DESIGN_SPEC,
                "settings": settings,
                "epochs": records,
            },
            indent=2,
        )

    to_run = []
    for nwb_file_name, epoch_name in epochs:
        identifier = epoch_id(nwb_file_name, epoch_name)
        if records.get(identifier, {}).get("status") == "ok" and not args.overwrite:
            print(f"{identifier}: already written", flush=True)
        else:
            # A record says what the folder holds for its epoch. One about to
            # be redone goes first, so a run stopped part way cannot leave new
            # files under an old record.
            records.pop(identifier, None)
            to_run.append((nwb_file_name, epoch_name))
    write_manifest()
    print(
        f"{len(to_run)} epochs to run on {args.n_workers} worker processes, each "
        f"logging to {Path.cwd() / 'logs' / '<epoch_id>_swr.log'}",
        flush=True,
    )

    # the cluster is made only when there is an epoch to run
    cluster_factory = functools.partial(
        LocalCluster, n_workers=args.n_workers, processes=True, threads_per_worker=1
    )
    failed = []
    run_epoch_kwargs = {
        "output_dir": output_dir,
        "run_id": run_id,
        "artifact_dir": artifact_dir,
    }
    # the workers only return records: this process alone writes the manifest
    log_directory = Path.cwd() / "logs"
    tasks = {
        epoch_id(nwb_file_name, epoch_name): (
            nwb_file_name,
            epoch_name,
            run_epoch_kwargs,
            log_directory,
        )
        for nwb_file_name, epoch_name in to_run
    }
    for number, (identifier, record) in enumerate(
        run_epochs(tasks, _run_epoch_logged, cluster_factory, key_prefix="swr"),
        start=1,
    ):
        records[identifier] = record
        print(f"[{number}/{len(to_run)}] {identifier}: {summary(record)}", flush=True)
        if record["status"] == "failed":
            failed.append(identifier)
        write_manifest()
    print(f"manifest -> {manifest_path}", flush=True)
    if failed:
        sys.exit(
            f"{len(failed)} epochs failed: {', '.join(failed)}; see {manifest_path}"
        )


if __name__ == "__main__":
    main()
