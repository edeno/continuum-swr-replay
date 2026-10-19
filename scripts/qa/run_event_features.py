#!/usr/bin/env python
"""Event features of every compared event, one epoch per task.

Reads the comparison runner's stored tables (``run_inventory_comparison.py``,
``--inventory-run-id``) and the memoryless run its manifest names, and reads
every compared event of the nine inventories on the decode grid by one rule.
No baseline windows are drawn: the lead dropped the baseline comparison from
this stage (2026-10-18), since windows matched on duration and location but
not on spike count measured engagement more than evidence. Design:
``docs/design/replay_detection_comparison/04-event-features.md``.

Per epoch:

1. **Decode.** The epoch is refused, and writes nothing, unless the decode's
   sha256 is the one the comparison run recorded for it. The frame is read
   through the SWR source, and its grid must step by the comparison's
   ``SAMPLE_INTERVAL_S`` (the median step; ``_speed_trace``).
2. **Memberships.** The pairwise memberships are rebuilt from
   the stored tables and the frame's still time (``rebuild_matches``); the
   epoch fails unless they reproduce the stored ``directional`` and ``cells``
   exactly. The populations follow (``population_rows``).
3. **Marks.** The decode's marks (``decode_marks``, from the memoryless
   runner's cache), refused unless every mark of the encoding model is among
   them (``check_marks_match_encoding``). When the memoryless run observed
   the epoch, the epoch is refused, and writes nothing, unless the decode's
   detector, the decode and the marks have the sha256 that run recorded.
4. **Evidence.** The spike evidence is computed on every row of the compared
   events (``epoch_row_evidence``), the rows a scored interval holds; every
   scored interval's rows are checked to be among them.
5. **Features** of every compared event: rows, engagement, LFP and MUA
   (``interval_features``), evidence (``interval_evidence``), the stop holding
   the onset (``stop_context``), the state-space posterior (``ss_features``),
   the memoryless windows (``ms_features``) and the neighbour gaps.
6. **Rejection stages** of every state-space event against each memoryless
   cohort (``rejection_stages``).

An epoch the memoryless run has no record for has every ``ms_*`` feature
missing, its memberships against the memoryless inventories ``unknown`` (as
stored) and every rejection stage ``unknown``.

Each epoch writes, under
``Processed-Data/replay_detection_comparison/event_features/<run_id>/`` (the
run id is the first 12 characters of the commit), one set of
``{epoch_id}_<table>.parquet``, one per table of :data:`TABLE_DTYPES`:
``features`` (:data:`FEATURE_DTYPES`), ``pair_membership`` (the rebuilt
``event_matches``) and ``rejection``. Every table is made before any is
written, and a failed epoch keeps none. ``manifest.json`` beside them holds the
run's settings (the input runs, the stop table, the grid and evidence rules,
tracking confidence recorded as unavailable) and one record per epoch:
its status, the decode's and the marks' sha256, the counts, the outputs, its
wall clock, the share of it spent computing the evidence, and the worker's
peak RSS. Only the main process writes it, atomically, after each epoch. A
rerun skips the epochs already ``ok`` unless ``--overwrite`` is given, and
refuses settings other than the ones the manifest was written with. A run
whose SWR source (its SWR and artifact runs, and the source's name) is not the
one the comparison read is refused before any epoch.

The epochs are the epoch register's analysed ones, narrowed by ``--animal``,
``--date`` and ``--epoch``, or the one ``--epoch-id``, run on a dask
``LocalCluster`` of ``--n-workers`` processes, one epoch at a time each, each
logging to ``logs/{epoch_id}_event_features.log`` under the working directory.
The device is the environment's: ``JAX_PLATFORMS`` (``cpu`` when unset) and
``CUDA_VISIBLE_DEVICES``. The BLAS and OpenMP thread counts are 1 in every
worker (dask's nanny sets them; this process sets them too when unset); JAX's
CPU threads follow the process's CPU affinity, so pin the run with
``taskset``.

All memoryless results are pending the Phase 2 pilot checks.

Usage:
    JAX_PLATFORMS=cpu taskset -c 56-71 python scripts/qa/run_event_features.py \\
        --inventory-run-id afeb897d850f --epoch-id chimi20200225_02_r1_HPC \\
        --n-workers 1
"""

import os

for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse  # noqa: E402
import functools  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import resource  # noqa: E402
import sys  # noqa: E402
import time as clock  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402
from collections.abc import Mapping  # noqa: E402
from pathlib import Path  # noqa: E402
from types import MappingProxyType  # noqa: E402
from typing import Any  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from dask.distributed import LocalCluster  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

# Sibling-script import: run as a script, only this file's directory is on
# sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from continuum_swr_replay.analysis.constants import (  # noqa: E402
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    DEFAULT_NON_LOCAL_THRESHOLD,
)
from continuum_swr_replay.analysis.hippocampus import swr_source  # noqa: E402
from continuum_swr_replay.analysis.hippocampus.artifacts.masks import (  # noqa: E402
    interval_mask,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.detect import (  # noqa: E402
    CandidateSequences,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.likelihood import (  # noqa: E402
    check_marks_match_encoding,
    interior_bin_distances_cm,
    window_midpoints_s,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.statistics import (  # noqa: E402
    windows_holding,
)
from continuum_swr_replay.analysis.hippocampus.state_evidence.terms import (  # noqa: E402
    encoding_inputs,
)
from continuum_swr_replay.analysis.inventory_comparison.availability import (  # noqa: E402
    still_intervals,
)
from continuum_swr_replay.analysis.inventory_comparison.evidence import (  # noqa: E402
    BLOCK_ROWS,
    animal_bins,
    epoch_row_evidence,
    interval_evidence,
)
from continuum_swr_replay.analysis.inventory_comparison.features import (  # noqa: E402
    DURATION_STRATA_S,
    DURATION_STRATUM_LABELS,
    STOP_CONTEXT_DTYPES,
    PosteriorTraces,
    RowTraces,
    _grid_rows,
    duration_stratum,
    interval_features,
    ms_features,
    neighbour_gaps,
    ss_features,
    stop_context,
)
from continuum_swr_replay.analysis.inventory_comparison.overlap import (  # noqa: E402
    MATCH_DTYPES,
)
from continuum_swr_replay.analysis.inventory_comparison.populations import (  # noqa: E402
    population_rows,
    rebuild_matches,
)
from continuum_swr_replay.analysis.inventory_comparison.recount import (  # noqa: E402
    FRAGMENT_GAPS_S,
)
from continuum_swr_replay.analysis.inventory_comparison.schema import (  # noqa: E402
    INVENTORY_DTYPES,
    STATE_SPACE,
)
from continuum_swr_replay.analysis.inventory_comparison.selection import (  # noqa: E402
    REJECTION_COHORTS,
    REJECTION_DTYPES,
    rejection_stages,
)
from continuum_swr_replay.decoding_common import (  # noqa: E402
    parse_run_id,
    provenance_record,
    raise_open_file_limit,
    release_nwb_files,
    replace_outputs,
    setup_logger,
    write_atomically,
)
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR, ROOT_DIR  # noqa: E402
from continuum_swr_replay.session_keys import SESSION_COLUMNS_DAY  # noqa: E402
from continuum_swr_replay.table_io import (  # noqa: E402
    read_table,
    sha256_file,
)
from scripts.decoding import run_carey_detection as carey  # noqa: E402
from scripts.decoding.run_memoryless_detection import (  # noqa: E402
    MARKS_RULE,
    marks_sha256,
    parse_epoch_identifier,
)
from scripts.decoding.run_memoryless_detection import (  # noqa: E402
    output_paths as memoryless_outputs,
)
from scripts.decoding.run_swr_detection import (  # noqa: E402
    epoch_id,
    json_ready,
    run_epochs,
    run_lock,
)
from scripts.qa import run_inventory_comparison as comparison  # noqa: E402

# the spike-evidence script installs ignore filters for whole warning classes
# when imported; they stay inside this block, out of the run and its workers
with warnings.catch_warnings():
    from scripts.qa.analyze_event_spike_evidence import REMOTE_CM  # noqa: E402

OUTPUT_ROOT = comparison.COMPARISON_ROOT / "event_features"
INVENTORY_ROOT = comparison.OUTPUT_ROOT
"""Where the comparison runs are."""
MEMORYLESS_ROOT = comparison.MEMORYLESS_ROOT
MARKS_CACHE_DIR = carey.MARKS_CACHE_DIR
"""The memoryless runner's marks cache, keyed by the decode's detector file."""
STOP_TABLE = PROCESSED_DATA_DIR / "replay_content" / "stop_table.parquet"
PRODUCED_BY = "scripts/qa/run_event_features.py"
DESIGN_SPEC = "docs/design/replay_detection_comparison/04-event-features.md"

GRID_RULE = (
    "every interval is read on the decode grid: its rows are the samples t with "
    "event_start_s <= t <= event_end_s, and row k holds the spikes with "
    "t_k <= s < t_{k+1}; the grid's last sample holds no such span and is never "
    "a row, so an interval holding only it has no row"
)
TRACKING_CONFIDENCE = (
    "unavailable: the decode frame carries no tracking confidence, so no "
    "feature reads one"
)
EVIDENCE_LIKELIHOOD = (
    "predict_clusterless_kde_log_likelihood, is_local=False, on the decode's "
    "saved encoding model; the rows of the compared events"
)

_MS_DTYPES: Mapping[str, object] = MappingProxyType(
    {
        "ms_n_windows": "Int64",
        "ms_n_windows_decoded": "Int64",
        "ms_n_windows_with_center": "Int64",
        "ms_spread_cm_mean": np.float64,
        "ms_extent_cm": np.float64,
        "ms_distance_cm_mean": np.float64,
        "ms_frac_windows_kept": np.float64,
    }
)
"""The ``ms_features`` columns; ``ms_n_windows`` is missing, not 0, in an epoch
with no memoryless run."""

FEATURE_DTYPES: Mapping[str, object] = MappingProxyType(
    {
        **INVENTORY_DTYPES,
        "duration_stratum": object,
        "first_row": np.int64,
        "n_rows": np.int64,
        "n_spikes": "Int64",
        "n_active_tetrodes": "Int64",
        "n_tetrodes": np.int64,
        "spike_rate_hz": np.float64,
        "ripple_z_max": np.float64,
        "ripple_z_mean": np.float64,
        "mua_z_max": np.float64,
        "mua_z_mean": np.float64,
        "linear_position_cm": np.float64,
        "track_segment_id": "Int64",
        "patch_id": "Int64",
        "evidence_dll": np.float64,
        "evidence_n_valid_rows": "Int64",
        "evidence_valid_fraction": np.float64,
        "evidence_unscorable": object,
        "previous_gap_s": np.float64,
        "next_gap_s": np.float64,
        **STOP_CONTEXT_DTYPES,
        "time_in_stop_s": np.float64,
        "time_since_arrival_s": np.float64,
        "ss_local_prob_mean": np.float64,
        "ss_no_spike_prob_mean": np.float64,
        "ss_continuous_prob_mean": np.float64,
        "ss_fragmented_prob_mean": np.float64,
        "ss_nonlocal_prob_max": np.float64,
        "ss_frac_rows_nonlocal": np.float64,
        "ss_extent_cm": np.float64,
        "ss_distance_cm_mean": np.float64,
        "ss_hpd_cm_mean": np.float64,
        "ss_n_runs_nonlocal": "Int64",
        "ss_longest_run_nonlocal_s": np.float64,
        "ss_nearest_event_gap_s": np.float64,
        "ss_disagreement": object,
        **_MS_DTYPES,
    }
)
"""One row per compared event: the inventory schema's columns, the span's
stratum (a label of ``DURATION_STRATUM_LABELS``, in that order), and every
feature."""

TABLE_DTYPES: Mapping[str, Mapping[str, object]] = MappingProxyType(
    {
        "features": FEATURE_DTYPES,
        "pair_membership": MATCH_DTYPES,
        "rejection": REJECTION_DTYPES,
    }
)
"""Every table an epoch writes, with its columns and dtypes."""

INVENTORY_TABLES = (
    "observations",
    "set_aside",
    "inventory",
    "pairs",
    "directional",
    "cells",
)
"""The comparison's stored tables an epoch reads."""


class _EpochRefused(Exception):
    """The epoch's inputs are not the comparison's; the message says why."""


def output_path(identifier: str, output_dir: Path, name: str) -> Path:
    return output_dir / f"{identifier}_{name}.parquet"


def output_paths(identifier: str, output_dir: Path) -> dict[str, Path]:
    """Every table an epoch can write, by name."""
    return {name: output_path(identifier, output_dir, name) for name in TABLE_DTYPES}


def _row_mask(n_time: int, first: NDArray, n_rows: NDArray) -> NDArray[np.bool_]:
    """The rows of every interval, as one mask over the grid."""
    edges = np.zeros(n_time + 1, dtype=np.int64)
    np.add.at(edges, first, 1)
    np.add.at(edges, first + n_rows, -1)
    mask: NDArray[np.bool_] = np.cumsum(edges[:-1]) > 0
    return mask


def evidence_rows(time: NDArray, events: pd.DataFrame) -> NDArray[np.bool_]:
    """The rows whose evidence is computed: every row of the compared events.

    Parameters
    ----------
    time : NDArray, shape (n_time,)
        The decode grid.
    events : pd.DataFrame
        ``event_start_s`` and ``event_end_s`` of the compared events.

    Returns
    -------
    rows : NDArray of bool, shape (n_time,)
        Never the grid's last sample, which is no row (``features``' row
        rule).
    """
    first, n_rows = _grid_rows(time, events)
    return _row_mask(time.size, first, n_rows)


def _require_requested(rows: NDArray[np.bool_], intervals: pd.DataFrame) -> None:
    """Raise unless every row of every interval had its evidence computed.

    A row not computed reads NaN as an invalid row does, so an interval left
    out would read ``no_valid_row`` rather than fail.
    """
    starts = intervals["first_row"].to_numpy(dtype=np.int64)
    counts = intervals["n_rows"].to_numpy(dtype=np.int64)
    scored = _row_mask(rows.size, starts, counts)
    if (scored & ~rows).any():
        raise RuntimeError(
            f"{int((scored & ~rows).sum())} rows of scored intervals were not among "
            "the rows the evidence was computed on."
        )


def _memoryless_run(
    identifier: str,
    observations: pd.DataFrame,
    memoryless_dir: Path,
    time: NDArray,
) -> tuple[CandidateSequences, pd.DataFrame, NDArray[np.bool_]] | None:
    """The epoch's memoryless candidates, events table and undecoded windows;
    None when the comparison found no memoryless run for it.

    Raises
    ------
    ValueError
        The two memoryless inventories disagree on whether the run is there,
        or the stored windows are not the decode grid's.
    """
    observed = observations.groupby("detector").available.any()
    available = {bool(observed[name]) for name in REJECTION_COHORTS}
    if len(available) != 1:
        raise ValueError(
            "The memoryless inventories disagree on whether the run observed the epoch."
        )
    if not available.pop():
        return None
    paths = memoryless_outputs(identifier, memoryless_dir)
    candidates: CandidateSequences = joblib.load(paths["candidates"])
    config = candidates.config
    if not np.array_equal(candidates.midpoint_s, window_midpoints_s(time[:-1], config)):
        raise ValueError(
            f"The stored memoryless windows of {identifier} are not the decode grid's."
        )
    # the run's is_missing: a bin inside a closed interval of the mask
    is_missing = interval_mask(time, swr_source.artifact_intervals(identifier))
    undecoded = windows_holding(
        is_missing[:-1], config.window_n_samples, config.stride_n_samples
    )
    return candidates, read_table(paths["events"]), undecoded


def memoryless_input_mismatch(
    record: Mapping[str, Any], read: Mapping[str, str]
) -> str | None:
    """Why the inputs read here are not the ones the memoryless run read, or
    None.

    Parameters
    ----------
    record : mapping
        The memoryless run's manifest record of the epoch, with its
        ``inputs`` (the decode's ``detector`` and ``data_over_time``) and
        ``marks``, each with a ``sha256``.
    read : mapping
        The sha256 of what this run read, by the same names: ``detector``,
        ``data_over_time`` and ``marks``.
    """
    recorded = {
        "detector": record["inputs"]["detector"]["sha256"],
        "data_over_time": record["inputs"]["data_over_time"]["sha256"],
        "marks": record["marks"]["sha256"],
    }
    differ = [name for name in recorded if recorded[name] != read[name]]
    if not differ:
        return None
    return (
        f"The {', '.join(differ)} sha256 differ from the memoryless run's record: "
        "its windows and candidates are not of these inputs."
    )


def _missing_ms(index: pd.Index) -> pd.DataFrame:
    """Every ``ms_*`` feature missing, for an epoch with no memoryless run."""
    return pd.DataFrame(
        {
            name: pd.Series(index=index, dtype=dtype)
            for name, dtype in _MS_DTYPES.items()
        }
    )


def _counts(features: pd.DataFrame, rejection: pd.DataFrame) -> dict[str, Any]:
    """What the console and the manifest report of an epoch's tables: per
    detector the events and those with or without an evidence score; per
    cohort the rejection stages."""
    unscorable = features[features.evidence_unscorable != ""]
    return {
        "events": features.detector.value_counts().to_dict(),
        "scorable": features[features.evidence_unscorable == ""]
        .detector.value_counts()
        .to_dict(),
        "unscorable": {
            detector: group.evidence_unscorable.value_counts().to_dict()
            for detector, group in unscorable.groupby("detector")
        },
        "rejection_stages": {
            cohort: group.stage.value_counts().to_dict()
            for cohort, group in rejection.groupby("cohort")
        },
    }


def epoch_tables(
    identifier: str,
    *,
    inventory_dir: Path,
    memoryless_dir: Path,
    stop_table: Path,
    decode_sha256: str,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    """One epoch's tables by name, with :data:`TABLE_DTYPES`, and its record.

    ``decode_sha256`` is the decode's, checked against the comparison's record
    by the caller and here against the memoryless run's.

    Raises
    ------
    _EpochRefused
        The decode's detector, the decode or the marks are not the ones the
        memoryless run read (its manifest record of the epoch).
    ValueError
        The grid does not step by the comparison's ``SAMPLE_INTERVAL_S``; the
        rebuilt memberships do not reproduce the stored ``directional`` or
        ``cells``; the marks are not the encoding model's; or the memoryless
        run is not the comparison's.
    """
    saved = pd.read_pickle(comparison.decode_path(identifier))
    dot = swr_source.with_swr_source(saved)
    del saved
    # raises unless the grid's median step is SAMPLE_INTERVAL_S, the step
    # interval_features is given
    time, speed = comparison._speed_trace(dot)
    animal, day, epoch = (dot[column].iloc[0] for column in SESSION_COLUMNS_DAY)
    session = (str(animal), int(day), str(epoch))
    still = {
        session: still_intervals(
            time, speed, DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC, comparison.MAX_GAP_S
        )
    }

    stored = {
        name: read_table(comparison.output_path(identifier, inventory_dir, name))
        for name in INVENTORY_TABLES
    }
    _, matches = rebuild_matches(
        stored, still, detectors=comparison.DETECTORS, cells=comparison.CELLS
    )
    inventory = stored["inventory"]
    compared = inventory[inventory.whole_event_speed_eligible].reset_index(drop=True)
    populations = population_rows(
        compared, matches, stored["cells"], stored["directional"]
    )

    marks = carey.decode_marks(
        identifier, root=comparison.DECODE_ROOT, cache_dir=MARKS_CACHE_DIR
    )
    spike_times = marks["spike_times"]
    spike_waveform_features = marks["spike_waveform_features"]
    check_marks_match_encoding(
        marks["encoding_model"], spike_times, spike_waveform_features
    )
    marks_hash = marks_sha256(spike_times, spike_waveform_features)
    run = _memoryless_run(identifier, stored["observations"], memoryless_dir, time)
    if run is not None:
        memoryless = json.loads((memoryless_dir / "manifest.json").read_text())
        reason = memoryless_input_mismatch(
            memoryless["epochs"][identifier],
            {
                "detector": marks["record"]["detector_sha256"],
                "data_over_time": decode_sha256,
                "marks": marks_hash,
            },
        )
        if reason is not None:
            raise _EpochRefused(reason)
    encoding = encoding_inputs(marks["encoding_model"])
    environment = encoding["env"]
    traces = RowTraces.from_frame(dot, spike_times)
    posterior = PosteriorTraces.from_frame(dot)
    bin_distance_cm = interior_bin_distances_cm(environment)
    animal_bin = animal_bins(environment, traces.linear_position)
    map_bin = animal_bins(environment, dot["map_estimate"].to_numpy(dtype=np.float64))

    rows = evidence_rows(time, compared)
    started = clock.perf_counter()
    evidence = epoch_row_evidence(
        traces.time,
        rows,
        traces.linear_position,
        spike_times,
        spike_waveform_features,
        encoding["encargs"],
        remote_cm=REMOTE_CM,
        block_rows=BLOCK_ROWS,
    )
    likelihood_s = clock.perf_counter() - started
    if evidence.shape != traces.time.shape:
        raise RuntimeError("The evidence is not on the traces' grid.")

    intervals = compared[list(INVENTORY_DTYPES)]
    engagement = interval_features(
        traces, intervals, sample_interval_s=comparison.SAMPLE_INTERVAL_S
    )
    _require_requested(rows, engagement)

    stops = pd.read_parquet(
        stop_table,
        filters=[
            ("animal", "==", session[0]),
            ("session_date", "==", session[1]),
            ("epoch", "==", session[2]),
        ],
    )
    if run is None:
        ms = _missing_ms(intervals.index)
    else:
        candidates = run[0]
        at_window = animal_bin[np.searchsorted(time, candidates.midpoint_s)]
        ms = ms_features(candidates, intervals, run[2], bin_distance_cm, at_window)
    features = pd.concat(
        [
            intervals,
            engagement,
            interval_evidence(traces, evidence, intervals),
            neighbour_gaps(compared).reindex(intervals.index),
            stop_context(intervals, stops),
            ss_features(
                traces,
                posterior,
                intervals,
                bin_distance_cm,
                map_bin,
                animal_bin,
                state_space_events=compared[compared.detector == STATE_SPACE],
            ),
            ms,
        ],
        axis=1,
    )
    # stored as its label: parquet keeps no categories for an empty column
    features["duration_stratum"] = duration_stratum(features["interval_span_s"]).astype(
        object
    )

    ours = compared[compared.detector == STATE_SPACE]
    rejection = [
        rejection_stages(
            ours,
            populations,
            *((None, None, None) if run is None else run),
            cohort=cohort,
        )
        for cohort in REJECTION_COHORTS
    ]
    tables = {
        "features": comparison._typed([features], FEATURE_DTYPES),
        "pair_membership": comparison._typed([matches], MATCH_DTYPES),
        "rejection": comparison._typed(rejection, REJECTION_DTYPES),
    }
    record = {
        "memoryless_available": run is not None,
        "marks": marks["record"] | {"sha256": marks_hash},
        "n_grid_rows": int(time.size),
        "n_evidence_rows": int(rows.sum()),
        "likelihood_s": likelihood_s,
        "counts": _counts(tables["features"], tables["rejection"]),
    }
    return record, tables


def run_epoch(
    nwb_file_name: str,
    epoch_name: str,
    *,
    output_dir: Path,
    inventory_dir: Path,
    memoryless_dir: Path,
    decode_sha256: Mapping[str, str],
    stop_table: Path,
    provenance: Mapping[str, Any],
) -> dict:
    """Read one epoch's features and write its tables.

    Parameters
    ----------
    nwb_file_name, epoch_name : str
        The epoch, such as ``"chimi20200225_.nwb"`` and ``"02_r1"``.
    output_dir : Path
        The run's folder.
    inventory_dir, memoryless_dir : Path
        The comparison run's folder and the memoryless run's.
    decode_sha256 : mapping
        Epoch id -> the decode's sha256 the comparison recorded, for its
        ``ok`` epochs.
    stop_table : Path
        The stop table.
    provenance : mapping
        ``provenance_record``'s, whose ``commit`` stamps the tables.

    Returns
    -------
    dict
        The epoch's manifest record, in strict JSON types: ``status`` ``"ok"``,
        or ``"failed"`` with a ``reason`` (the decode is not the comparison's,
        the comparison has no ``ok`` record of the epoch, or the inputs are
        not the memoryless run's) or the ``error`` and its ``traceback``,
        with the decode's ``path`` and ``sha256`` once read. A failed epoch's
        outputs are removed.
    """
    started = clock.perf_counter()
    identifier = epoch_id(nwb_file_name, epoch_name)
    stamp = {
        "produced_by": PRODUCED_BY,
        "run_id": output_dir.name,
        "epoch_id": identifier,
        "commit": str(provenance["commit"]),
    }
    record: dict[str, Any] = {}
    try:
        if identifier not in decode_sha256:
            raise _EpochRefused(
                f"Comparison run {inventory_dir.name} has no ok record of the epoch."
            )
        path = comparison.decode_path(identifier)
        record["decode"] = {"path": str(path), "sha256": sha256_file(path)}
        if record["decode"]["sha256"] != decode_sha256[identifier]:
            raise _EpochRefused(
                f"The decode's sha256 {record['decode']['sha256'][:12]} is not the "
                f"{decode_sha256[identifier][:12]} comparison run "
                f"{inventory_dir.name} recorded: its tables are not of this decode."
            )
        found, tables = epoch_tables(
            identifier,
            inventory_dir=inventory_dir,
            memoryless_dir=memoryless_dir,
            stop_table=stop_table,
            decode_sha256=record["decode"]["sha256"],
        )
        record |= {"status": "ok", **found}
        record["outputs"] = replace_outputs(
            output_paths(identifier, output_dir),
            comparison.stamped_tables(tables, stamp),
        )
    except _EpochRefused as refused:
        replace_outputs(output_paths(identifier, output_dir), {})
        record = _failed(record, reason=str(refused))
    except Exception as error:  # noqa: BLE001 - recorded; the run goes on
        replace_outputs(output_paths(identifier, output_dir), {})
        record = _failed(record, error=repr(error), traceback=traceback.format_exc())
    record["provenance"] = dict(provenance)
    record["wall_clock_s"] = clock.perf_counter() - started
    if "likelihood_s" in record:
        record["likelihood_share"] = record["likelihood_s"] / record["wall_clock_s"]
    # the worker's peak resident set over its life so far (KiB on Linux)
    record["worker_peak_rss_gib"] = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    )
    ready: dict = json_ready(record)
    return ready


def _failed(record: Mapping[str, Any], **why: str) -> dict[str, Any]:
    """A failed epoch's record: the decode it read, if any, and why it failed;
    nothing of the ok record it may have nearly been."""
    kept = {name: record[name] for name in ("decode",) if name in record}
    return {**kept, "status": "failed", **why}


def summary(record: dict) -> str:
    """One epoch's outcome, as the logs and the console report it."""
    if record["status"] != "ok":
        return f"{record['status']}: {record.get('reason') or record.get('error')}"
    counts = record["counts"]
    unscorable = counts["unscorable"] or "none"
    return (
        f"ok: events {counts['events']}; unscorable {unscorable}; "
        f"{record['wall_clock_s']:.0f} s, "
        f"{100 * record['likelihood_share']:.0f} % in the likelihood, worker "
        f"peak RSS {record['worker_peak_rss_gib']:.1f} GiB"
    )


def _run_epoch_logged(
    nwb_file_name: str, epoch_name: str, run_epoch_kwargs: dict, log_directory: Path
) -> dict:
    """One dask task: ``run_epoch``, logged to
    ``logs/{epoch_id}_event_features.log``.

    When the epoch ends, pass or fail, the NWB files Spyglass opened for it
    (a marks cache miss reads Spyglass) are closed (``release_nwb_files``).
    """
    identifier = epoch_id(nwb_file_name, epoch_name)
    log = setup_logger(
        name_logfile=f"{identifier}_event_features",
        path_logfile=log_directory / f"{identifier}_event_features.log",
    )
    log.info("%s: run %s", identifier, run_epoch_kwargs["output_dir"])
    try:
        record = run_epoch(nwb_file_name, epoch_name, **run_epoch_kwargs)
    finally:
        log.info("closed %d NWB files Spyglass held open", release_nwb_files())
    log.info("PROVENANCE %s", json.dumps(record["provenance"], sort_keys=True))
    if record["status"] == "failed":
        log.error("%s\n%s", summary(record), record.get("traceback", ""))
    else:
        log.info("%s", summary(record))
        log.info(
            "likelihood %.1f s of %.1f s (%.1f %%)",
            record["likelihood_s"],
            record["wall_clock_s"],
            100 * record["likelihood_share"],
        )
    return record


def run_settings(inventory_run_id: str, inventory: Mapping[str, Any]) -> dict:
    """What every epoch of a run shares; a rerun under its run id must match.

    Parameters
    ----------
    inventory_run_id : str
    inventory : mapping
        The comparison run's manifest.
    """
    from non_local_detector.likelihoods.clusterless_kde import (
        predict_clusterless_kde_log_likelihood,
    )

    block_size = (
        inspect.signature(predict_clusterless_kde_log_likelihood)
        .parameters["block_size"]
        .default
    )
    settings: dict = json_ready(
        {
            "input_runs": {
                "inventory": inventory_run_id,
                **inventory["settings"]["input_runs"],
            },
            "swr_source": swr_source.swr_source_name(),
            # relative, so a worktree whose Processed-Data links to the same
            # folder records the same settings
            "stop_table": {
                "path": str(STOP_TABLE.relative_to(PROCESSED_DATA_DIR)),
                "sha256": sha256_file(STOP_TABLE),
            },
            "decode": comparison.decode_path("{epoch_id}").name
            + ", read through the SWR source; its sha256 the comparison's",
            "marks_rule": MARKS_RULE,
            "grid": {
                "rule": GRID_RULE,
                "sample_interval_s": comparison.SAMPLE_INTERVAL_S,
            },
            "evidence": {
                "likelihood": EVIDENCE_LIKELIHOOD,
                "remote_cm": REMOTE_CM,
                "block_rows": BLOCK_ROWS,
                # encoding_inputs passes none, so the library's default is used
                "library_block_size": block_size,
            },
            "speed": {
                "threshold_cm_per_s": DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
                "max_gap_s": comparison.MAX_GAP_S,
            },
            "non_local_threshold": DEFAULT_NON_LOCAL_THRESHOLD,
            "near_s": max(FRAGMENT_GAPS_S),
            "duration_strata": {
                "bounds_s": DURATION_STRATA_S,
                "labels": DURATION_STRATUM_LABELS,
            },
            "rejection_cohorts": REJECTION_COHORTS,
            "tracking_confidence": TRACKING_CONFIDENCE,
            "jax_platforms": os.environ.get("JAX_PLATFORMS"),
            "peak_rss": "worker_peak_rss_gib: the worker's peak so far, cumulative "
            "over the epochs it ran before",
        }
    )
    return settings


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--inventory-run-id",
        required=True,
        type=parse_run_id,
        help="The comparison run folder's name, such as afeb897d850f.",
    )
    parser.add_argument("--animal", help="Only this animal's epochs, such as chimi.")
    parser.add_argument(
        "--date", type=int, help="Only epochs recorded on this date, such as 20200225."
    )
    parser.add_argument("--epoch", help="Only this epoch, such as 02_r1.")
    parser.add_argument(
        "--epoch-id",
        help="Only this epoch, such as chimi20200225_02_r1_HPC; not with --animal, "
        "--date or --epoch.",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help="Dask worker processes, each running one epoch at a time (default 1).",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Redo epochs already written."
    )
    args = parser.parse_args(argv)
    if args.epoch_id is not None and any(
        value is not None for value in (args.animal, args.date, args.epoch)
    ):
        parser.error("--epoch-id names one epoch; drop --animal, --date and --epoch.")

    # before the cluster starts: its worker processes inherit this limit
    raise_open_file_limit()
    provenance = provenance_record(ROOT_DIR)
    run_id = provenance["commit"][:12]
    if provenance["dirty"]:
        print(
            f"warning: uncommitted changes; run {run_id} cannot be tied to a commit",
            flush=True,
        )
    output_dir = OUTPUT_ROOT / run_id
    with run_lock(output_dir):
        run(args, provenance, output_dir)


def run(args: argparse.Namespace, provenance: dict, output_dir: Path) -> None:
    """``main``'s run, under the run folder's lock."""
    inventory_dir = INVENTORY_ROOT / args.inventory_run_id
    inventory_manifest = inventory_dir / "manifest.json"
    if not inventory_manifest.exists():
        sys.exit(f"--inventory-run-id: no run manifest at {inventory_manifest}.")
    inventory = json.loads(inventory_manifest.read_text())
    input_runs = inventory["settings"]["input_runs"]
    current = {
        "swr": swr_source.SWR_INVENTORY_RUN_ID,
        "artifacts": swr_source.ARTIFACT_INTERVALS_RUN_ID,
    }
    if {name: input_runs[name] for name in current} != current:
        sys.exit(
            f"Comparison run {args.inventory_run_id} read SWR run "
            f"{input_runs['swr']} and artifact run {input_runs['artifacts']}; the "
            f"SWR source is {current['swr']} and {current['artifacts']}."
        )
    if inventory["settings"]["swr_source"] != swr_source.swr_source_name():
        sys.exit(
            f"Comparison run {args.inventory_run_id} read SWR source "
            f"{inventory['settings']['swr_source']}; the SWR source is "
            f"{swr_source.swr_source_name()}."
        )
    memoryless_dir = MEMORYLESS_ROOT / input_runs["memoryless"]
    if not (memoryless_dir / "manifest.json").exists():
        sys.exit(f"No memoryless run manifest at {memoryless_dir}.")
    settings = run_settings(args.inventory_run_id, inventory)
    epochs = comparison.select_epochs(args)
    if not epochs:
        sys.exit("No analysed epochs match --animal, --date and --epoch.")

    manifest_path = output_dir / "manifest.json"
    records: dict[str, dict] = {}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["settings"] != settings:
            sys.exit(
                f"{manifest_path} was written with other settings; a run must "
                "not mix them. Commit, and run under the new commit."
            )
        records = previous["epochs"]

    def write_manifest() -> None:
        text = json.dumps(
            {
                "produced_by": PRODUCED_BY,
                "run_id": output_dir.name,
                "design_spec": DESIGN_SPEC,
                "settings": settings,
                "epochs": records,
            },
            indent=2,
            allow_nan=False,
        )
        write_atomically(manifest_path, lambda path: path.write_text(text))

    to_run = []
    for identifier in epochs:
        if records.get(identifier, {}).get("status") == "ok" and not args.overwrite:
            print(f"{identifier}: already written", flush=True)
        else:
            # the record goes first, so a run stopped part way cannot leave new
            # files under an old record
            records.pop(identifier, None)
            key = parse_epoch_identifier(identifier)
            to_run.append((key.nwb_file_name, key.epoch_name))
    write_manifest()
    print(
        f"{len(to_run)} epochs to run on {args.n_workers} worker processes, each "
        f"logging to {Path.cwd() / 'logs' / '<epoch_id>_event_features.log'}",
        flush=True,
    )

    cluster_factory = functools.partial(
        LocalCluster, n_workers=args.n_workers, processes=True, threads_per_worker=1
    )
    run_epoch_kwargs = {
        "output_dir": output_dir,
        "inventory_dir": inventory_dir,
        "memoryless_dir": memoryless_dir,
        "decode_sha256": {
            identifier: record["decode"]["sha256"]
            for identifier, record in inventory["epochs"].items()
            if record["status"] == "ok"
        },
        "stop_table": STOP_TABLE,
        "provenance": provenance,
    }
    failed = []
    # the workers only return records: this process alone writes the manifest
    for number, (identifier, record) in enumerate(
        run_epochs(
            to_run,
            run_epoch_kwargs,
            cluster_factory,
            task=_run_epoch_logged,
            key_prefix="event-features",
        ),
        start=1,
    ):
        records[identifier] = record
        print(f"[{number}/{len(to_run)}] {identifier}: {summary(record)}", flush=True)
        if record["status"] != "ok":
            failed.append(identifier)
        write_manifest()
    print(f"manifest -> {manifest_path}", flush=True)
    if failed:
        sys.exit(
            f"{len(failed)} epochs failed: {', '.join(failed)}; see {manifest_path}"
        )


if __name__ == "__main__":
    main()
