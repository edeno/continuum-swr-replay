#!/usr/bin/env python
"""Run the four SWR detectors on every epoch's CA1 LFP, from shared inputs.

Kay, Karlsson, Yu and Zugaro, from ``ripple_detection`` at its pinned commit,
each with its own settings (``swr_inventories.DETECTOR_SETTINGS``), on one
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
3. **Zugaro's channel.** ``best_ripple_channel`` (buzcode's
   ``bz_GetBestRippleChan``) chooses one of the band's CA1 electrodes from their
   referenced raw LFP, and ``zero_phase_bandpass`` filters it. A candidate with
   no finite sample is excluded with that reason, so it costs the others no
   samples. If no channel is usable, Zugaro's entry is that error.
4. **Detectors.** ``run_swr_detectors`` runs the four; Kay, Karlsson and Yu read
   the stored band. A detector's error is recorded and the rest still report:
   the epoch is ``ok`` when all four succeed, ``partial`` when some do, and
   ``failed`` when none does.
5. **Yu's diagnostics**, recomputed by ``yu_diagnostics``.
6. **Kay version check** against the canonical Spyglass events
   (``RippleTimesV1``), in two comparisons. ``version_only`` runs Kay at the
   pinned commit on Spyglass's own inputs
   (``RippleTimesV1.get_ripple_lfps_and_position_info``); ``version_and_inputs``
   takes the direct Kay inventory. The canonical events and the version-only
   events are kept only inside the position span, by ``filter_ripple_lfp_time``'s
   rule, as ``load_data`` keeps ``SWR_event``.

Each epoch writes, under ``Processed-Data/replay_detection_comparison/swr/<run_id>/``:

- ``{epoch_id}_{detector}_events.parquet``: the detector's own event table, with
  all its statistics, for each detector that succeeded;
- ``{epoch_id}_swr_observed.parquet``: where the LFP was observed, one row per
  gap-free stretch, ``scope`` ``"stored_band"`` (the band Kay, Karlsson and Yu
  read) or ``"zugaro"`` (Zugaro's filtered channel, less its filter's edges),
  with ``start_s`` and ``end_s``, the stretch's first and last sample;
- ``{epoch_id}_kay_version_check.parquet``: ``kay_version_check``'s rows, with a
  ``comparison`` column, ``"version_only"`` or ``"version_and_inputs"``.

An epoch's other outputs, left by an earlier run, are removed, so the folder
always matches the manifest.

``manifest.json`` beside them holds the run's settings (the package commit,
every detector's settings, the Zugaro filter and channel choice) and one record
per epoch: its status and, for an epoch that ran, every detector's outcome, the
provenance, the electrodes and references, Zugaro's channel with every score,
the band check, Yu's diagnostics, the sampling rates, the position span, event
counts and input hashes. Only this process writes it, atomically, after each
epoch finishes. A rerun under the same run id skips the epochs already ``ok``
unless ``--overwrite`` is given, and refuses settings other than the ones the
manifest was written with.

The epochs come from ``get_epoch_info``, narrowed by ``--animal``, ``--date`` and
``--epoch``. The documented epochs with no inputs (``KNOWN_NO_INPUT_EPOCHS``) are
recorded as skipped without a database read. The epochs run in parallel on a
dask ``LocalCluster`` of ``--n-workers`` processes, one epoch per worker at a
time, each logging to ``logs/{epoch_id}_swr.log`` under the working directory.
An epoch that fails is recorded with its error and the run goes on; the run
then exits non-zero.

Usage:
    python scripts/decoding/run_swr_detection.py --animal chimi --date 20200225 \\
        --epoch 02_r1 --n-workers 1
    python scripts/decoding/run_swr_detection.py --n-workers 8   # every epoch
"""

import argparse
import functools
import hashlib
import json
import math
import resource
import sys
import time as clock
import traceback
from collections.abc import Callable, Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
import scipy
from dask.distributed import Client, LocalCluster, as_completed
from distributed.deploy import Cluster
from numpy.typing import NDArray
from ripple_detection import Kay_ripple_detector

from continuum_swr_replay.analysis.hippocampus.swr_inventories import (
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
    gap_free_stretches,
    kay_version_check,
    run_swr_detectors,
    yu_diagnostics,
    zero_phase_bandpass,
)
from continuum_swr_replay.analysis.inventory_comparison.schema import SessionIdentity
from continuum_swr_replay.data_loaders import get_epoch_info
from continuum_swr_replay.data_loaders.lfp import _ripple_keys, get_ripple_band_inputs
from continuum_swr_replay.data_loaders.position import get_epoch_position_info
from continuum_swr_replay.data_loaders.utils import parse_nwb_filename
from continuum_swr_replay.decoding_common import (
    filter_epoch_info,
    provenance_record,
    setup_logger,
    write_atomically,
)
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR, ROOT_DIR
from continuum_swr_replay.table_io import write_table

OUTPUT_ROOT = PROCESSED_DATA_DIR / "replay_detection_comparison" / "swr"
PRODUCED_BY = "scripts/decoding/run_swr_detection.py"
DESIGN_SPEC = "docs/design/2026-10-03-full-pipeline-refresh-design.md"

_NO_RIPPLE_BAND = "no LFPBandV1 ripple band"
_NO_TRACK_GRAPH = "no TrackGraph, so its position cannot be loaded"
_NO_CA1_EXTRACTION = (
    "no CA1 ripple extraction: RippleTimesV1 holds only group sb_wilbur_ripple"
)
KNOWN_NO_INPUT_EPOCHS: Mapping[tuple[str, str], str] = MappingProxyType(
    {
        ("senor20201104_.nwb", "06_r3"): _NO_RIPPLE_BAND,
        ("senor20201104_.nwb", "08_r4"): _NO_RIPPLE_BAND,
        ("senor20201105_.nwb", "09_r4"): _NO_TRACK_GRAPH,
        **dict.fromkeys(
            (
                ("wilbur20210404_.nwb", epoch_name)
                for epoch_name in ("02_r1", "04_r2", "06_r3", "08_r4", "10_r5")
            ),
            _NO_CA1_EXTRACTION,
        ),
    }
)
"""The epochs with no inputs, documented in ``data_loaders/constants.py``, each
with its reason: skipped without a database read. None of them has a decode."""


def epoch_id(nwb_file_name: str, epoch_name: str) -> str:
    """``{stem}_{epoch}_HPC``, the decoders' name for the epoch's CA1 data.

    ``epoch_id("chimi20200225_.nwb", "02_r1")`` is ``"chimi20200225_02_r1_HPC"``.
    """
    return f"{nwb_file_name.removesuffix('_.nwb')}_{epoch_name}_HPC"


def output_paths(identifier: str, output_dir: Path) -> dict[str, Path]:
    """Every file an epoch can write, keyed by detector or table name."""
    paths = {
        name: output_dir / f"{identifier}_{name}_events.parquet"
        for name in DETECTOR_SETTINGS
    }
    paths["swr_observed"] = output_dir / f"{identifier}_swr_observed.parquet"
    paths["kay_version_check"] = output_dir / f"{identifier}_kay_version_check.parquet"
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
) -> tuple[pd.Series, pd.DataFrame, float]:
    """The inputs ``RippleTimesV1.make`` gave Kay for the canonical events.

    ``RippleTimesV1.get_ripple_lfps_and_position_info`` on the epoch's full
    ``RippleTimesV1`` key, which only reads.

    Returns
    -------
    speed : pd.Series
        Head speed in cm/s, interpolated onto the LFP times.
    ripple_lfps : pd.DataFrame, shape (n_time, n_channels)
        The stored ripple band, indexed by time in seconds.
    sampling_frequency : float
        The band's nominal rate in Hz.
    """
    from spyglass.ripple.v1 import RippleTimesV1

    ripple_key, _ = _ripple_keys(nwb_file_name, epoch_name)
    key = (RippleTimesV1 & ripple_key).fetch1("KEY")
    inputs: tuple[pd.Series, pd.DataFrame, float] = (
        RippleTimesV1.get_ripple_lfps_and_position_info(key)
    )
    return inputs


def kay_on_spyglass_inputs(
    speed: pd.Series, ripple_lfps: pd.DataFrame, sampling_frequency: float
) -> pd.DataFrame:
    """Kay at the pinned commit on Spyglass's own inputs, as ``make`` calls it.

    With ``DETECTOR_SETTINGS["kay"]``, which equal Spyglass's ``"default"``
    parameter set, so only the package version differs from the canonical run.
    """
    return Kay_ripple_detector(
        time=np.asarray(ripple_lfps.index),
        filtered_lfps=np.asarray(ripple_lfps),
        speed=np.asarray(speed),
        sampling_frequency=sampling_frequency,
        **DETECTOR_SETTINGS["kay"],
    )


def within_span(events: pd.DataFrame, span_s: tuple[float, float]) -> pd.DataFrame:
    """The events wholly inside ``span_s``, by ``filter_ripple_lfp_time``'s rule."""
    keep = (events["start_time"] >= span_s[0]) & (events["end_time"] <= span_s[1])
    return events.loc[keep]


def observation_table(
    time: NDArray, ripple_band: NDArray, zugaro_lfp: NDArray
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

    Returns
    -------
    pd.DataFrame
        ``scope``, and ``start_s`` and ``end_s``, the times of a stretch's first
        and last sample.
    """
    rows = [
        (scope, time[start], time[stop - 1])
        for scope, values in (("stored_band", ripple_band), ("zugaro", zugaro_lfp))
        for start, stop in gap_free_stretches(time, values)
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


def array_sha256(values: NDArray) -> str:
    """The sha256 of an array's float64 bytes, in C order."""
    return hashlib.sha256(
        np.ascontiguousarray(values, dtype=np.float64).data
    ).hexdigest()


def json_ready(value: Any) -> Any:
    """``value`` in strict JSON types, for the manifest.

    Mappings (``MappingProxyType`` included) become dicts with string keys;
    arrays, lists and tuples become lists; numpy scalars become Python ones;
    and a non-finite float becomes None.
    """
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple | np.ndarray):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def choose_zugaro_channel(
    raw_referenced: NDArray,
    time: NDArray,
    sampling_frequency: float,
    electrode_ids: NDArray,
) -> tuple[dict, NDArray, NoUsableChannelError | None]:
    """Choose and filter Zugaro's channel among the band's CA1 electrodes.

    A candidate with no finite sample is excluded with that reason before
    ``best_ripple_channel`` runs on the rest, which treats a row with any NaN
    as unobserved on every channel.

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
        ``electrode_id`` (None with no usable channel), ``scores`` (every
        candidate's, NaN where it has none) and ``excluded`` (with reasons),
        plus ``error`` with no usable channel.
    zugaro_lfp : NDArray, shape (n_time, 1)
        The chosen channel band-passed by ``zero_phase_bandpass`` at
        ``ZUGARO_BAND_HZ``; all NaN with no usable channel.
    error : NoUsableChannelError or None
        Why there is no channel, or None.
    """
    has_finite = np.isfinite(raw_referenced).any(axis=0)
    excluded = {int(e): "no finite sample" for e in electrode_ids[~has_finite]}
    scores: dict[int, float] = dict.fromkeys(excluded, math.nan)
    try:
        if not has_finite.any():
            raise NoUsableChannelError("No candidate channel has a finite sample.")
        candidates = (
            raw_referenced if has_finite.all() else raw_referenced[:, has_finite]
        )
        choice = best_ripple_channel(
            candidates, time, sampling_frequency, electrode_ids[has_finite]
        )
    except NoUsableChannelError as error:
        channel = {
            "electrode_id": None,
            "scores": scores,
            "excluded": excluded,
            "error": repr(error),
        }
        return channel, np.full((time.size, 1), np.nan), error
    column = int(np.flatnonzero(electrode_ids == choice.electrode_id)[0])
    channel = {
        "electrode_id": choice.electrode_id,
        "scores": dict(sorted((choice.scores | scores).items())),
        "excluded": dict(sorted((choice.excluded | excluded).items())),
    }
    zugaro_lfp = zero_phase_bandpass(
        raw_referenced[:, [column]],
        time,
        sampling_frequency,
        ZUGARO_BAND_HZ,
        ZUGARO_FILTER_ORDER,
    )
    return channel, zugaro_lfp, None


def _detect(
    nwb_file_name: str,
    epoch_name: str,
    *,
    run_id: str,
    fetch_inputs: Callable,
    fetch_position: Callable,
    fetch_canonical_kay: Callable,
    fetch_spyglass_kay_inputs: Callable,
    fir_filter: Callable | None,
) -> tuple[dict, dict[str, pd.DataFrame]]:
    """One epoch's record, and the tables to write (none for a failed epoch)."""
    inputs = fetch_inputs(nwb_file_name, epoch_name)
    band_check: dict[str, Any] = {
        "max_abs_difference": None,
        "tolerance": None,
        "band_dtype": inputs["ripple_band_dtype"],
        "reference_overflow_count": inputs["reference_overflow_count"],
    }
    try:
        difference, tolerance = band_reproduction_error(inputs, fir_filter)
    except ValueError as error:  # the re-filter does not cover the whole band
        reason = f"The band check could not compare the whole stored band: {error}"
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
    speed = np.interp(time, position_time, head_speed, left=np.nan, right=np.nan)
    sampling_frequency = inputs["sampling_frequency"]
    electrode_ids = inputs["electrode_ids"]

    zugaro_channel, zugaro_lfp, channel_error = choose_zugaro_channel(
        raw_referenced, time, sampling_frequency, electrode_ids
    )
    results = run_swr_detectors(
        time, ripple_band, zugaro_lfp, speed, sampling_frequency
    )
    zugaro_events = results["zugaro"]
    # read at once: pandas attrs do not reliably survive a copy or a concat
    zugaro_dropped = (
        int(zugaro_events.attrs["n_dropped_over_maximum_duration"])
        if isinstance(zugaro_events, pd.DataFrame)
        else None
    )
    if channel_error is not None:
        # Zugaro was given an unobserved channel, which it refuses; its entry
        # says why there was no channel instead
        results["zugaro"] = channel_error
    succeeded = [
        name for name, events in results.items() if isinstance(events, pd.DataFrame)
    ]
    try:
        yu = yu_diagnostics(time, ripple_band, speed, sampling_frequency)
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
        "reference_ids": inputs["reference_ids"],
        "zugaro_channel": zugaro_channel,
        "band_check": band_check,
        "yu_diagnostics": yu,
        "sampling": {
            "nominal": sampling_frequency,
            "measured": measured_rate(time, ripple_band),
        },
        "lfp_filter_name": inputs["lfp_filter_name"],
        "lfp_units": inputs["lfp_units"],
        "position_span_s": span_s,
        "n_events": {
            name: len(events) if name in succeeded else None
            for name, events in results.items()
        },
        "zugaro_dropped_over_150ms": zugaro_dropped,
        # the band's electrodes, taking the CA1 ripple selection to hold one
        # per tetrode, as its electrode ids suggest; neither that nor the
        # tetrodes' anatomy is checked here, hence "unknown" below
        "n_detecting_tetrodes": int(electrode_ids.size),
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
    compared = {
        "version_only": within_span(
            kay_on_spyglass_inputs(
                *fetch_spyglass_kay_inputs(nwb_file_name, epoch_name)
            ),
            span_s,
        )
    }
    if "kay" in succeeded:
        compared["version_and_inputs"] = results["kay"]
    kay_check = pd.concat(
        [
            kay_version_check(
                direct,
                canonical,
                session=session,
                run_id=run_id,
                speed_time_s=position_time,
                speed_cm_per_s=head_speed,
            ).assign(comparison=comparison)
            for comparison, direct in compared.items()
        ],
        ignore_index=True,
    )
    counts: dict[str, dict[str, dict[str, int]]] = {}
    for (comparison, detector, status), n in (
        kay_check.groupby(["comparison", "detector", "status"]).size().items()
    ):
        counts.setdefault(comparison, {}).setdefault(detector, {})[status] = int(n)
    record["kay_version_check"] = counts

    tables = {name: results[name] for name in succeeded}
    tables["swr_observed"] = observation_table(time, ripple_band, zugaro_lfp)
    tables["kay_version_check"] = kay_check
    stamp = {
        "produced_by": PRODUCED_BY,
        "run_id": run_id,
        "epoch_id": epoch_id(nwb_file_name, epoch_name),
        "commit": record["provenance"]["commit"],
    }
    for table in tables.values():
        table.attrs = table.attrs | stamp
    return record, tables


def replace_outputs(
    identifier: str, output_dir: Path, tables: dict[str, pd.DataFrame]
) -> dict[str, str]:
    """Write an epoch's tables and remove its other outputs.

    The folder then matches the manifest: a detector that failed this time
    keeps no file from an earlier run.

    Returns
    -------
    dict
        Each written table's path, by name.
    """
    written = {}
    for name, path in output_paths(identifier, output_dir).items():
        if name in tables:
            written[name] = str(write_table(tables[name], path))
        else:
            path.unlink(missing_ok=True)
    return written


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
) -> dict:
    """Run the four detectors and the checks on one epoch, and write the outputs.

    Parameters
    ----------
    nwb_file_name, epoch_name : str
        The epoch, such as ``"chimi20200225_.nwb"`` and ``"02_r1"``.
    output_dir : Path
        The run's folder, where the outputs are written.
    fetch_inputs : Callable, optional
        ``(nwb_file_name, epoch_name) -> RippleBandInputs``.
    fetch_position : Callable, optional
        ``(nwb_file_name, epoch_name) -> PositionInfoDict``, whose
        ``position_info`` gives the span and ``head_speed``.
    fetch_canonical_kay : Callable, optional
        ``(nwb_file_name, epoch_name) -> pd.DataFrame`` of canonical events.
    fetch_spyglass_kay_inputs : Callable, optional
        ``(nwb_file_name, epoch_name) -> (speed, ripple_lfps,
        sampling_frequency)``, as ``spyglass_kay_inputs`` returns them.
    fir_filter : Callable, optional
        The band check's filter, passed to ``band_reproduction_error``; None
        is Spyglass's own ``FirFilterParameters().filter_data``.
    run_id : str, optional
        Recorded on the Kay check's events; default ``output_dir.name``.

    Returns
    -------
    dict
        The epoch's manifest record, in strict JSON types. ``status`` is
        ``"skipped"`` (with ``reason``) for an epoch in
        ``KNOWN_NO_INPUT_EPOCHS``, fetched nothing; ``"failed"`` with
        ``reason`` and ``band_check`` when the band check fails, with
        ``reason`` and every detector's error when all four fail, or with
        ``error`` (and ``traceback``) for any other exception; else ``"ok"``
        or ``"partial"``, with every field the module docstring lists.
    """
    reason = KNOWN_NO_INPUT_EPOCHS.get((nwb_file_name, epoch_name))
    if reason is not None:
        return {"status": "skipped", "reason": reason}
    started = clock.perf_counter()
    identifier = epoch_id(nwb_file_name, epoch_name)
    try:
        record, tables = _detect(
            nwb_file_name,
            epoch_name,
            run_id=output_dir.name if run_id is None else run_id,
            fetch_inputs=fetch_inputs,
            fetch_position=fetch_position,
            fetch_canonical_kay=fetch_canonical_kay,
            fetch_spyglass_kay_inputs=fetch_spyglass_kay_inputs,
            fir_filter=fir_filter,
        )
        record["outputs"] = replace_outputs(identifier, output_dir, tables)
    except Exception as error:  # noqa: BLE001 - recorded; the run goes on
        replace_outputs(identifier, output_dir, {})
        record = {
            "status": "failed",
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }
    record["wall_clock_s"] = clock.perf_counter() - started
    # the worker process's peak so far (ru_maxrss is in KiB on Linux)
    record["peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
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
    if "kay_version_check" in record:
        parts.append(f"Kay check {record['kay_version_check']}")
    if "reason" in record:
        parts.append(record["reason"])
    parts.append(
        f"{record['wall_clock_s']:.0f} s, worker peak RSS "
        f"{record['peak_rss_gib']:.1f} GiB"
    )
    return "; ".join(parts)


def _run_epoch_logged(
    nwb_file_name: str, epoch_name: str, run_epoch_kwargs: dict, log_directory: Path
) -> dict:
    """One dask task: ``run_epoch``, logged to ``logs/{epoch_id}_swr.log``."""
    identifier = epoch_id(nwb_file_name, epoch_name)
    log = setup_logger(
        name_logfile=f"{identifier}_swr",
        path_logfile=log_directory / f"{identifier}_swr.log",
    )
    log.info("%s: run %s", identifier, run_epoch_kwargs)
    record = run_epoch(nwb_file_name, epoch_name, **run_epoch_kwargs)
    if "provenance" in record:
        log.info("PROVENANCE %s", json.dumps(record["provenance"], sort_keys=True))
    if record["status"] == "failed":
        log.error("%s\n%s", summary(record), record.get("traceback", ""))
    else:
        log.info("%s", summary(record))
    return record


def run_epochs(
    epochs: list[tuple[str, str]],
    run_epoch_kwargs: dict,
    cluster_factory: Callable[[], Cluster],
) -> Iterator[tuple[str, dict]]:
    """Run each epoch as its own dask task; yield each record as it finishes.

    Nothing here writes the manifest: the caller does, after each record, so
    it has a single writer.

    Parameters
    ----------
    epochs : list of (str, str)
        ``(nwb_file_name, epoch_name)`` pairs, each named once.
    run_epoch_kwargs : dict
        Every keyword of ``run_epoch`` to pass.
    cluster_factory : Callable
        Makes the cluster, closed after the last record; none is made for no
        epochs. Its workers should run one task at a time: ``setup_logger``
        reconfigures logging for the whole worker process.

    Yields
    ------
    epoch_id : str
    record : dict
        ``run_epoch``'s record, in the order the epochs finish; ``status``
        ``"failed"`` with the error when the task could not return one, as
        when its worker died.
    """
    if not epochs:
        return
    log_directory = Path.cwd() / "logs"
    with cluster_factory() as cluster, Client(cluster) as client:
        futures = {}
        for nwb_file_name, epoch_name in epochs:
            identifier = epoch_id(nwb_file_name, epoch_name)
            future = client.submit(
                _run_epoch_logged,
                nwb_file_name,
                epoch_name,
                run_epoch_kwargs,
                log_directory,
                key=f"swr-{identifier}",
            )
            futures[future] = identifier
        for future in as_completed(futures):
            try:
                record = future.result()
            except Exception as error:  # noqa: BLE001 - the task could not return
                record = {
                    "status": "failed",
                    "error": repr(error),
                    "traceback": "".join(traceback.format_exception(error)),
                }
            yield futures[future], record


def run_settings(provenance: dict) -> dict:
    """What every epoch of a run shares; a rerun under its run id must match.

    Parameters
    ----------
    provenance : dict
        ``provenance_record``'s, for the ``ripple_detection`` commit.
    """
    settings: dict = json_ready(
        {
            "ripple_detection_commit": provenance["ripple_detection_commit"],
            "detectors": DETECTOR_SETTINGS,
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
        help="The run's folder under --output-root; defaults to the first 12 "
        "characters of git rev-parse HEAD.",
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args(argv)

    provenance = provenance_record(ROOT_DIR)
    run_id = args.run_id or provenance["commit"][:12]
    if provenance["dirty"]:
        print(
            f"warning: uncommitted changes; run {run_id} cannot be tied to a commit",
            flush=True,
        )
    settings = run_settings(provenance)
    epochs = list(dict.fromkeys(filter_epoch_info(get_epoch_info(), args).index))
    if not epochs:
        sys.exit("No epochs match --animal, --date and --epoch.")

    output_dir = args.output_root / run_id
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
        text = json.dumps(
            {
                "produced_by": PRODUCED_BY,
                "run_id": run_id,
                "design_spec": DESIGN_SPEC,
                "settings": settings,
                "epochs": records,
            },
            indent=2,
            allow_nan=False,
        )
        write_atomically(manifest_path, lambda path: path.write_text(text))

    to_run = []
    for nwb_file_name, epoch_name in epochs:
        identifier = epoch_id(nwb_file_name, epoch_name)
        if records.get(identifier, {}).get("status") == "ok" and not args.overwrite:
            print(f"{identifier}: already written", flush=True)
        else:
            to_run.append((nwb_file_name, epoch_name))
    output_dir.mkdir(parents=True, exist_ok=True)
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
    run_epoch_kwargs = {"output_dir": output_dir, "run_id": run_id}
    # the workers only return records: this process alone writes the manifest
    for number, (identifier, record) in enumerate(
        run_epochs(to_run, run_epoch_kwargs, cluster_factory), start=1
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
