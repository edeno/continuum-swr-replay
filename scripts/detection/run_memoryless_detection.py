#!/usr/bin/env python
"""Run the memoryless sequence detector over recorded HPC epochs.

The Widloski & Foster sliding-window detector, ported in
``continuum_swr_replay.analysis.hippocampus.memoryless_sequence``, run on each
epoch's saved clusterless decode: the encoding model from
``{epoch_id}_detector.pkl``, the decode grid, position and speed from
``{epoch_id}_80_data_over_time.pkl``, the SWR labels and where they were
observed from the paper's SWR source (``analysis/events/swr_source.py``,
the direct Kay inventory, on that grid), and the spike marks read as
``load_data`` read them for the decode (``MARKS_RULE``): ``get_hpc_marks``,
then ``filter_spike_times`` over the decode's time span.
Both procedures therefore read one encoding model. The decode's encoding
model already leaves out the artifact mask (the decoders' encoding exclusion),
but this detector computes its own likelihood over every bin, so the runner
marks the mask's bins on the decode grid missing (``is_missing``, the decoders'
rule: a bin inside a closed interval of ``swr_source.artifact_intervals``).
No window holding one is decoded, no candidate touches one, and neither the
shuffle null nor any normalization reads them (``memoryless_sequence.epoch``).
A candidate in excluded time (the mask widened by
``events.eligibility.ARTIFACT_REACH_S``), or overlapping an SWR or MUA event
dropped for it, keeps its row with an ``artifact_drop_reason`` and is in no
cohort.
``check_marks_match_encoding`` refuses them unless every mark the encoding
model was fitted on is among them. Those are the training span's, so it cannot
see a spike the reload adds, or one dropped only during SWRs, which
``no_ripple`` training leaves out; the refresh design leaves that gap to its
Stage 8 one-commit check (``docs/design/2026-10-03-full-pipeline-refresh-design.md``).

Each epoch writes, under ``Processed-Data/replay_detection_comparison/memoryless/<run_id>/``:

- ``{epoch_id}_memoryless_events.parquet``, one row per candidate sequence;
- ``{epoch_id}_memoryless_shuffle_draws.parquet``, every shuffled draw;
- ``{epoch_id}_memoryless_observed.parquet``, what was observed: the stretches
  of decoded windows (none holds a masked bin), and the stretches where SWR
  labels were observed. It records what was read, as the SWR runner's
  ``swr_observed.parquet`` does: the comparison's reader
  (``inventory_comparison.availability.coverage_from_observations``) removes
  excluded time from it, as from every detector's;
- ``{epoch_id}_memoryless_candidates.pkl``, the per-window statistics, gates
  and pre-merge runs that explain each rejection.

``manifest.json`` records the run's settings and, per epoch, the commit, the
time grid, a hash of every input, the marks' rule and hash, the encoding model's
settings, and the outcome. It is rewritten atomically after each epoch, so a
run that stops part way still says what it did, and a rerun under the same
run id refuses to continue with different settings, code, library versions,
environment (``uv.lock``'s sha256, the commit ``ripple_detection`` was
installed from, the CUDA plugin's version) or XLA flags. An epoch that fails
is recorded as ``failed`` with its error and the run goes on to the next; the run then exits
non-zero, and a rerun under the same run id retries the failed epochs. That holds for a transient failure; after a code fix the
commit, and so the default run id, has changed, and the finished epochs are
recomputed under the new one, which is what the settings gate is for.
``run.log`` beside the manifest keeps everything printed.

Without ``--cuda_visible_devices`` the epochs run one after another in this
process. With it they run in parallel, as the decoders do: a dask
``LocalCUDACluster`` with one worker per listed GPU runs one epoch per task
(``cuda_cluster``). Each worker may use its share of this process's CPUs for
BLAS and OpenMP: dask's nanny would otherwise cap each at one thread.
Every process that runs JAX, the workers or this process without a device
list, runs XLA deterministically and with Triton GEMM off:
``--xla_gpu_deterministic_ops=true`` and ``--xla_gpu_enable_triton_gemm=false``
are added to the ``XLA_FLAGS`` it inherits (``xla_flags``), so two runs at
identical settings are bit-for-bit equal and every epoch runs under one
setting (``KNOWN_LIMITS``). The run settings record those flags.
Each epoch logs to ``logs/{epoch_id}_memoryless.log`` under the working
directory, and only this process writes ``manifest.json``, as each epoch
finishes, in the order they finish. A device list is refused when ``dask_cuda``
is not importable.

The marks come from the Spyglass database unless ``--cache-dir`` already holds
this decode's from an earlier run. A cache file is named for the epoch, the
rule (``MARKS_RULE_TAG``) and the first 16 hex digits of the decode's
``_detector.pkl`` sha256, so it serves only marks read under this rule for this
decode file: one made for another decode of the epoch is never read, and
neither is the pilot's ``{epoch_id}_decode_time_marks.pkl``, made under the
April decodes' rule.

Epoch identifiers name the decode: ``chimi20200212_02_r1_HPC`` is the
canonical one (Kay's SWRs, trained on ``no_ripple``, stationary). A decode
made another way carries suffixes in the order the decode appends them:
``_contfrag``, a ripple detector (``_Karlsson``), ``_non_stationary``, and a
training type (``_all``). ``--animal`` and ``--all`` run the epoch register's
analysed epochs of that animal, or of every animal, or with ``--pending-only``
its pending ones alone (``epochs_to_run``): each one's canonical decode, and
with ``--include-variants`` also the variants of it found under ``--root``. An
epoch so selected whose decode is missing is run, and recorded as failed, so a
run never covers fewer epochs than the register without saying so; a decode
under ``--root`` of any other epoch is left out, and recorded as skipped. The
pending epochs' decodes are staged outside the canonical folder until they are
analysed (``docs/epoch_register.md``), so a ``--pending-only`` run takes
``--root`` at that staging folder. ``--epoch-id`` runs whatever it
names. ``--date`` and ``--epoch`` narrow any of the three. The date is the
last 8 characters of the NWB file's stem, so ``j1620210707`` was recorded on
20210707; the epoch is a name such as ``02_r1``.

Usage:
    python scripts/detection/run_memoryless_detection.py --run-seed 20260922 \\
        --epoch-id chimi20200212_02_r1_HPC
    python scripts/detection/run_memoryless_detection.py --run-seed 20260922 \\
        --animal chimi
    python scripts/detection/run_memoryless_detection.py --run-seed 20260922 \\
        --animal chimi --no-shuffle      # the detector alone, much faster
    python scripts/detection/run_memoryless_detection.py --run-seed 20260922 \\
        --animal chimi --date 20200212 --epoch 02_r1
    python scripts/detection/run_memoryless_detection.py --run-seed 20260922 \\
        --all --cuda_visible_devices 0,1,2,3   # an epoch at a time on each GPU
"""

import argparse
import contextlib
import functools
import json
import logging
import os
import sys
import traceback
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import dask
import joblib
import numpy as np
import pandas as pd
from distributed.deploy import Cluster
from numpy.typing import NDArray

from continuum_swr_replay.analysis.events.analysed_time import (
    dropped_for_excluded_time,
    read_artifact_event_drops,
    read_dot_with_swr_source,
    stamp_swr_source,
    swr_observed_stretches,
)
from continuum_swr_replay.analysis.events.swr_source import (
    artifact_intervals,
    swr_epoch_id,
    swr_epoch_id_of_session,
    swr_source_files,
    swr_source_name,
)
from continuum_swr_replay.analysis.hippocampus.detection_inputs import (
    MARKS_RULE,
    input_paths,
    load_marks,
    marks_cache_dir,
    marks_cache_path,
    marks_sha256,
    memoryless_output_paths,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence import (
    MemorylessSequenceConfig,
    SequenceIdentity,
    ShuffleSeed,
    load_clusterless_encoding_model,
    run_epoch,
    window_midpoints_s,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.detect import (
    contiguous_runs,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.event_table import (
    canonical_swr_intervals,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.likelihood import (
    check_marks_match_encoding,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.shuffle import (
    DEFAULT_N_SHUFFLES,
    MINIMUM_N_SHUFFLES,
    SIGNIFICANCE_LEVEL,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.window_statistics import (
    windows_holding,
)
from continuum_swr_replay.analysis.replay.fit_provenance import git_hash
from continuum_swr_replay.data_loaders.epoch_register import epochs_to_run
from continuum_swr_replay.decoding.identifiers import (
    EpochKey,
    parse_epoch_identifier,
    parse_nwb_filename,
)
from continuum_swr_replay.intervals import (
    interval_mask,
    masked_seconds,
)
from continuum_swr_replay.parameters import ROOT_DIR
from continuum_swr_replay.paths import memoryless_root, processed_data_dir
from continuum_swr_replay.runs.batch import (
    cuda_cluster_class,
    raise_open_file_limit,
    release_nwb_files,
    run_epoch_logged,
)
from continuum_swr_replay.runs.batch import run_epochs as batch_run_epochs
from continuum_swr_replay.runs.cli import parse_run_id
from continuum_swr_replay.runs.device import xla_flags
from continuum_swr_replay.runs.io import sha256_file
from continuum_swr_replay.runs.provenance import produced_by, provenance_record
from continuum_swr_replay.runs.threads import BLAS_THREAD_VARIABLES
from continuum_swr_replay.tables.io import write_atomically, write_table

DESIGN_SPEC = "docs/design/replay_detection_comparison/02-memoryless-sequence.md"
NOT_ANALYSED = "not an analysed epoch in the epoch register"
"""Why a decode under ``--root`` is left out when the register does not analyse
its epoch."""
NOT_PENDING = "not a pending epoch in the epoch register"
"""Why a ``--pending-only`` run leaves out a decode under ``--root``."""


def find_epochs(
    animal: str | None,
    root: Path,
    selected: Iterable[tuple[str, str]],
    *,
    include_variants: bool = False,
    left_out: str = NOT_ANALYSED,
) -> tuple[list[str], dict[str, str]]:
    """The selected epochs' HPC decodes, and why every other decode was left out.

    Every selected epoch's canonical decode is listed, whether or not its files
    are under `root`: ``run_one`` then records a missing one as failed. HPC
    only: this port reads a clusterless encoding model. A decode under `root`
    that is not a selected epoch's canonical one is recorded rather than
    raised, so one stray file cannot stop a run over many epochs.

    Parameters
    ----------
    animal : str or None
        Only this animal's epochs; None for every animal's.
    root : Path
        Where the decodes are.
    selected : Iterable of (str, str)
        ``(nwb_file_name, epoch_name)`` of the epochs this run covers:
        ``epochs_to_run().index``, the register's analysed epochs or its
        pending ones.
    include_variants : bool, optional
        Also list the selected epochs' decode variants found under `root`.
    left_out : str, optional
        Why a decode of another epoch is left out: ``NOT_ANALYSED``, or
        ``NOT_PENDING`` for a ``--pending-only`` run.

    Returns
    -------
    epochs : list of str
        Sorted.
    skipped : dict of str to str
        Epoch identifier to the reason it was left out.
    """
    keys = {
        (nwb_file_name, epoch_name)
        for nwb_file_name, epoch_name in selected
        if animal is None or parse_nwb_filename(nwb_file_name)[0] == animal
    }
    epochs = {
        swr_epoch_id(nwb_file_name, epoch_name) for nwb_file_name, epoch_name in keys
    }
    skipped = {}
    pattern = f"{animal}*_HPC*_detector.pkl" if animal else "*_HPC*_detector.pkl"
    for detector in sorted(root.glob(pattern)):
        epoch_id = detector.name.removesuffix("_detector.pkl")
        if epoch_id in epochs:
            continue
        try:
            key = parse_epoch_identifier(epoch_id)
        except ValueError as error:
            skipped[epoch_id] = str(error)
            continue
        missing = [
            name
            for name, path in input_paths(epoch_id, root).items()
            if not path.exists()
        ]
        if key.brain_area != "HPC":
            skipped[epoch_id] = f"brain area {key.brain_area}, not HPC"
        elif (key.nwb_file_name, key.epoch_name) not in keys:
            skipped[epoch_id] = left_out
        elif not include_variants:
            skipped[epoch_id] = "a decode variant; pass --include-variants to run it"
        elif missing:
            skipped[epoch_id] = f"missing {', '.join(missing)}"
        else:
            epochs.add(epoch_id)
    return sorted(epochs), skipped


def select_epochs(
    epoch_ids: list[str], *, date: int | None, epoch: str | None
) -> list[str]:
    """The epochs recorded on `date` and named `epoch`, in their given order.

    Parameters
    ----------
    epoch_ids : list of str
    date : int or None
        Such as ``20200225``: the identifier's ``EpochKey.date``, the last 8
        digits of the NWB file's stem, so ``j1620210707`` is 20210707. None
        keeps every date.
    epoch : str or None
        Such as ``02_r1``, the identifier's epoch name. None keeps every epoch.

    Returns
    -------
    list of str
    """
    if date is None and epoch is None:
        # nothing is parsed, so a name no decode writes still reaches run_one,
        # which records it as failed
        return list(epoch_ids)
    selected = []
    for epoch_id in epoch_ids:
        key = parse_epoch_identifier(epoch_id)
        if (date is None or key.date == date) and (
            epoch is None or key.epoch_name == epoch
        ):
            selected.append(epoch_id)
    return selected


def _one_value(frame: pd.DataFrame, column: str) -> Any:
    values = pd.unique(frame[column])
    if len(values) != 1:
        raise ValueError(
            f"data_over_time holds {len(values)} values of {column!r}, not one."
        )
    return values[0]


def observation_table(
    time: np.ndarray,
    swr_observed: np.ndarray,
    config: MemorylessSequenceConfig,
    missing_windows: np.ndarray,
) -> pd.DataFrame:
    """The stretches this epoch observed, independent of what it detected.

    ``decoded_windows`` are the stretches of window midpoints the grid
    supports, the final row, which no spike can reach, excluded, less the
    windows holding a missing bin, each from its first to its last midpoint;
    ``swr_labels`` are the stretches where SWR labels were observed.

    Parameters
    ----------
    time : ndarray, shape (n_time,)
        The decode grid.
    swr_observed : ndarray, shape (n_stretches, 2)
    config : MemorylessSequenceConfig
    missing_windows : ndarray of bool, shape (n_windows,)
        The windows holding a missing bin, which were not decoded.
    """
    midpoints = window_midpoints_s(time[:-1], config)
    rows = [
        ("decoded_windows", midpoints[first], midpoints[last])
        for first, last in contiguous_runs(~np.asarray(missing_windows, dtype=bool))
    ]
    rows += [("swr_labels", start, end) for start, end in swr_observed]
    return pd.DataFrame(rows, columns=["kind", "start_s", "end_s"]).astype(
        {"kind": object, "start_s": np.float64, "end_s": np.float64}
    )


def run_one(
    epoch_id: str,
    *,
    root: Path,
    output_dir: Path,
    cache_dir: Path,
    run_id: str,
    run_seed: int,
    shuffle: bool,
    n_shuffles: int,
    config: MemorylessSequenceConfig,
    provenance: dict,
    marks_loader: Callable[
        [EpochKey, Path, tuple[float, float], str],
        tuple[list[NDArray], list[NDArray]],
    ] = load_marks,
) -> dict:
    """Detect, test and save one epoch; return what the manifest records for it.

    Raises
    ------
    FileNotFoundError
        An input is missing. A run over the register asks for one when an
        analysed epoch's decode is missing, and records the epoch as failed;
        :func:`find_epochs` leaves out a variant with a missing input.
    ValueError
        The marks are not the ones the encoding model was fitted on, or the
        grid, the identifier or ``data_over_time`` is not what the detector
        needs.
    """
    key = parse_epoch_identifier(epoch_id)
    inputs = input_paths(epoch_id, root)
    missing = [name for name, path in inputs.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{epoch_id}: missing {', '.join(missing)}.")
    input_sha256 = {name: sha256_file(path) for name, path in inputs.items()}

    encoding_model = load_clusterless_encoding_model(inputs["detector"])
    # SWR labels from the one SWR source, whichever decode variant this is
    data_over_time = read_dot_with_swr_source(inputs["data_over_time"])
    identity = SequenceIdentity(
        animal=str(_one_value(data_over_time, "animal")),
        day=int(_one_value(data_over_time, "day")),
        epoch=str(_one_value(data_over_time, "epoch")),
        brain_area=key.brain_area,
        epoch_id=epoch_id,
        run_id=run_id,
    )
    time = data_over_time.index.to_numpy(dtype=np.float64)
    spike_times, spike_waveform_features = marks_loader(
        key, cache_dir, (float(time[0]), float(time[-1])), input_sha256["detector"]
    )
    check_marks_match_encoding(encoding_model, spike_times, spike_waveform_features)

    speed = data_over_time["head_speed"].to_numpy(dtype=np.float64)
    midpoint_s = window_midpoints_s(time[:-1], config)
    # where the source's detector read a gap-free stored band
    swr_epoch = swr_epoch_id_of_session(identity.animal, identity.day, identity.epoch)
    swr_observed = swr_observed_stretches(swr_epoch)
    swr_files = swr_source_files(swr_epoch)
    # the one artifact mask, on the decode grid, as the decoders' is_missing
    artifact_mask = artifact_intervals(swr_epoch)
    is_missing = interval_mask(time, artifact_mask)
    missing_windows = windows_holding(
        is_missing[:-1], config.window_n_samples, config.stride_n_samples
    )
    detection = run_epoch(
        time,
        data_over_time["linear_position"].to_numpy(dtype=np.float64),
        speed,
        spike_times,
        spike_waveform_features,
        encoding_model,
        identity,
        swr_intervals=canonical_swr_intervals(
            data_over_time, speed_threshold=config.speed_threshold_cm_per_s
        ),
        swr_observed_intervals=swr_observed,
        swr_definition=swr_source_name(),
        seed=ShuffleSeed(run=run_seed, session=epoch_id) if shuffle else None,
        config=config,
        n_shuffles=n_shuffles,
        is_missing=is_missing,
        artifact_intervals=artifact_mask,
        dropped_events=dropped_for_excluded_time(
            read_artifact_event_drops(inputs["data_over_time"])
        ),
    )

    outputs = memoryless_output_paths(epoch_id, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    observed = observation_table(time, swr_observed, config, missing_windows)
    for frame in (detection.table, detection.shuffle_draws, observed):
        frame.attrs = provenance | {"epoch_id": epoch_id}
    # The table's SWR columns are the source's: stamped as every canonical table is.
    stamp_swr_source(detection.table)
    write_table(detection.table, outputs["events"])
    write_table(detection.shuffle_draws, outputs["shuffle_draws"])
    write_table(observed, outputs["observed"])
    write_atomically(
        outputs["candidates"], lambda path: joblib.dump(detection.candidates, path)
    )

    table = detection.table
    gates = detection.candidates.decisions.gates
    return {
        "status": "ok",
        "git_commit": provenance["git_commit"],
        "produced_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "decode": {
            "swr_definition": key.swr_definition,
            "contfrag": key.contfrag,
            "non_stationary": key.non_stationary,
            "training_type": key.training_type,
        },
        "time_grid": {
            "start_s": float(time[0]),
            "end_s": float(time[-1]),
            "n_samples": int(time.size),
            "sampling_frequency": config.sampling_frequency,
        },
        "encoding": {
            "n_electrodes": len(encoding_model["encoding_positions"]),
            "n_encoding_spikes": [
                int(np.asarray(p).shape[0])
                for p in encoding_model["encoding_positions"]
            ],
            "position_std": np.asarray(encoding_model["position_std"]).tolist(),
            "waveform_std": np.asarray(encoding_model["waveform_std"]).tolist(),
            "block_size": int(encoding_model["block_size"]),
        },
        "artifacts": {
            "n_intervals": len(artifact_mask),
            # the masked time within the decode grid
            "masked_s": masked_seconds(artifact_mask.clip(time[0], time[-1])),
            "n_missing_bins": int(np.count_nonzero(is_missing)),
            "n_missing_windows": int(np.count_nonzero(missing_windows)),
            # candidates the mask reaches, in no cohort (events.eligibility)
            "n_dropped": {
                reason: int((table["artifact_drop_reason"] == reason).sum())
                for reason in ("artifact", "near_artifact", "overlaps_dropped_event")
            },
        },
        "n_windows": int(gates.speed_eligible.size),
        # a window whose speed was NaN is rejected as not still; how many
        # says whether a quiet epoch was quiet or untracked
        "n_windows_speed_unknown": int(
            np.count_nonzero(~np.isfinite(np.interp(midpoint_s, time, speed)))
        ),
        "n_candidates": int(len(table)),
        "n_replay": int(table.is_replay.sum()),
        "n_swr_unknown": int((table.overlap_class == "unknown").sum()),
        "n_shuffled": int((table.n_shuffles > 0).sum()),
        "n_shuffle_untestable": int(
            ((table.n_shuffles > 0) & ~table.shuffle_testable).sum()
        ),
        "n_shuffle_pass": int(table.shuffle_pass.sum()),
        "inputs": {
            name: {"path": str(path), "sha256": input_sha256[name]}
            for name, path in inputs.items()
        },
        "swr_source": {"name": swr_source_name(), **swr_files},
        "marks": {
            "rule": MARKS_RULE,
            "detector_sha256": input_sha256["detector"],
            "cache": marks_cache_path(key, cache_dir, input_sha256["detector"]).name,
            "sha256": marks_sha256(spike_times, spike_waveform_features),
        },
        "outputs": {name: str(path) for name, path in outputs.items()},
    }


KNOWN_LIMITS = {
    "bit_reproducible": {
        "value": True,
        "measured": "R12 Triton spike, 2026-10-11, on breeze, one epoch per GPU, "
        "seed 20260922, canonical R11 decodes. The flags were "
        "XLA_FLAGS=--xla_gpu_enable_triton_gemm=false exported and "
        "--xla_gpu_deterministic_ops=true appended by the runner; the runner now "
        "sets both itself. (1) chimi20200224_08_r4_HPC, which failed with "
        "deterministic ops alone (4:34 to the error), runs with Triton GEMM off: "
        "233 candidates, 11 replay, 7 passing the shuffle (the 2026-10-05 run, "
        "with neither flag, had 232, 11 and 7). Two such runs are bit-for-bit "
        "equal; only run_id differs. This is the only epoch run twice with "
        "Triton GEMM off. (2) On one passing epoch, chimi20200225_02_r1_HPC, the "
        "output with Triton GEMM off differs from the R12 smoke test's "
        "(deterministic, Triton GEMM on): the counts are the same (237 "
        "candidates, 4 replay, 3 passing the shuffle); 38 spread and 73 jump "
        "gates flip out of about 291,000 windows that have a spread (295,369 "
        "windows in all); one event's bounds move by a window; decisions "
        "spread_cm differs in all 291,301 windows with a finite spread, max "
        "relative 3.5e-3; jump_cm differs in 2,769 windows (max relative 1.0); "
        "the null spread differs by at most 1.7e-4 relative and null_jump_cm in "
        "7 values (max relative 1.8e-2); wall time is 4:56 against 4:20. (3) On "
        "the same epoch, the output with Triton GEMM off is much closer to the "
        "2026-10-10 run with neither flag than the R12 smoke test's output is. "
        "Against the 2026-10-10 run there are 0 gate flips, the spread differs by "
        "at most 4.2e-4 relative, and the event bounds are the same. The 38 "
        "spread and 73 jump flips against the R12 smoke test's output are the "
        "same windows as the 111 flips the 2026-10-10 run showed against it",
        "history": "R12 smoke test, 2026-10-10, with XLA_FLAGS="
        "--xla_gpu_deterministic_ops=true only (the Triton GEMM flag not set, "
        "Triton GEMM on). "
        "chimi20200225_02_r1_HPC, seed 20260922, decodes read from the R11 "
        "staging folder, on breeze: two runs are bit-for-bit equal (every column "
        "of the events, shuffle draws, observed and candidates outputs; only "
        "run_id differs), 4:20 wall clock each against 5:10 without the flag. A "
        "run without the flag differs from them: 111 of 295,369 windows flip a "
        "gate, spread differs by up to 0.35 % relative, one event's bounds move "
        "by a window; the counts are equal (237 candidates, 4 replay, 3 passing "
        "the shuffle). Task 17a, 2026-10-09, with neither flag: two runs were not "
        "bit-for-bit equal. chimi20200225_02_r1, two runs at identical settings, "
        "one thread on one GPU: 19,199 windows differ, max relative difference "
        "1.2e-4, one window's gate flips (no event-level comparison reported). "
        "j1620210706_02_r1, the runner before and after its thread fix: 6,326 of "
        "631,632 windows differ (max relative 1.3e-5), one window's gate flips, "
        "every event-level output equal",
        "runs_before_this_change": "R12 (run c55cf318d199-seed20260922) "
        "ran with --xla_gpu_deterministic_ops=true and Triton GEMM on, and lost "
        "3 of 242 epochs: chimi20200224_08_r4_HPC, "
        "chimi20200227_04_r2_HPC and chimi20200228_02_r1_HPC. Each failed in the "
        "shuffle with JaxRuntimeError('CANCELLED: Too small divisible part of the "
        "contracting dimension.'), at non_local_detector/likelihoods/"
        "clusterless_kde.py:89 (spike_waveform_feature_distance.T @ "
        "position_distance). All three ran without error on 2026-10-05, with "
        "neither flag. Of these three, only chimi20200224_08_r4_HPC has been run "
        "with Triton GEMM "
        "off so far (it runs); chimi20200227_04_r2_HPC and "
        "chimi20200228_02_r1_HPC have not been run yet, and the full rerun will "
        "run them. Why Triton GEMM is off (the working explanation; not "
        "confirmed): deterministic ops turn off XLA's autotuning, and the default "
        "Triton GEMM setup then fails on some shapes; the spike tested the fix, "
        "not the mechanism. Runs before this change used Triton GEMM. On "
        "chimi20200225_02_r1_HPC, the one epoch compared, the R12 smoke test's "
        "output differs from a run made under the new flags (see measured); R12's "
        "own outputs and other epochs have not been compared",
    }
}
"""What is known of a run's reproducibility, recorded in every manifest's
settings: two runs are bit-for-bit equal with ``runs.device.DETERMINISTIC_XLA_FLAG`` and
``TRITON_GEMM_OFF_XLA_FLAG``, which the runner sets (``runs.device.xla_flags``), and
were not without them. ``history`` keeps the earlier measurements, each with the
flags it used; ``runs_before_this_change`` says that R12 used Triton GEMM and lost
3 epochs to an XLA error. See the design note
(``docs/design/replay_detection_comparison/02-memoryless-sequence.md``,
"Run-to-run reproducibility")."""


def run_settings(
    run_seed: int,
    shuffle: bool,
    n_shuffles: int,
    config: MemorylessSequenceConfig,
    git_commit: str,
) -> dict:
    """What every epoch of one run must share; a resume must match it.

    ``environment`` is the decoders' provenance record (``provenance_record``)
    without ``dirty``, which the run id and ``git_commit`` already carry: it
    holds ``uv.lock``'s sha256, the commit ``ripple_detection`` was installed
    from and the CUDA plugin's version. ``swr_source`` names the SWR source
    the labels come from (``swr_source_name()``), so a new inventory or
    artifact run is a new run. ``xla_flags`` are the ``XLA_FLAGS`` every
    process that runs JAX gets (:func:`xla_flags`), so a rerun under other
    flags is refused. ``known_limits`` states what is known of a run's
    reproducibility: with those flags (deterministic ops on, Triton GEMM off) two
    runs at identical settings are bit-for-bit equal (:data:`KNOWN_LIMITS`;
    design note, "Run-to-run reproducibility").
    """
    return {
        "git_commit": git_commit,
        "swr_source": swr_source_name(),
        "libraries": {
            name: version(name) for name in ("non_local_detector", "numpy", "pandas")
        },
        "environment": {
            name: value
            for name, value in provenance_record(ROOT_DIR).items()
            if name != "dirty"
        },
        "config": asdict(config),
        "xla_flags": xla_flags(),
        "known_limits": KNOWN_LIMITS,
        "shuffle": {
            "run": shuffle,
            "definition": "encoding-pair: within each electrode, encoding "
            "positions permuted against their waveform marks",
            "n_shuffles": n_shuffles if shuffle else 0,
            "run_seed": run_seed,
            "seeded_by": "run seed, epoch_id, candidate's first window, draw, electrode",
            "tested": "every candidate meeting the replay criteria, as the reference",
            "statistics": "mean posterior spread and mean forward jump over every "
            "window; the jump of a pair no spike entered or left set aside, which "
            "is the reference's jumps(jumps==0)=nan on a continuous center; NaN "
            "skipped",
            "p_value": "(1 + count(null < observed)) / (n_shuffles + 1), as the reference",
            "passes": f"both p-values at or below {SIGNIFICANCE_LEVEL}, as the reference",
        },
    }


def write_manifest(path: Path, manifest: dict) -> None:
    """Write through a temporary file, so a stopped run never leaves half a manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomically(
        path, lambda temporary: temporary.write_text(json.dumps(manifest, indent=2))
    )


def _counts(record: dict) -> str:
    """What one epoch found, as the logs report it."""
    return (
        f"{record['n_windows']} windows ({record['n_windows_speed_unknown']} with "
        f"unknown speed), {record['n_candidates']} candidates, "
        f"{record['n_replay']} replay, {record['n_swr_unknown']} unknown SWR "
        f"coverage, {record['n_shuffled']} shuffled "
        f"({record['n_shuffle_untestable']} untestable), "
        f"{record['n_shuffle_pass']} passing the shuffle"
    )


def _failure_record(error: BaseException, git_commit: str) -> dict:
    """What the manifest records for an epoch that failed in a dask task."""
    return {
        "status": "failed",
        "git_commit": git_commit,
        "produced_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "error": repr(error),
        "traceback": "".join(traceback.format_exception(error)),
    }


def _run_one_logged(epoch_id: str, run_one_kwargs: dict, log_directory: Path) -> dict:
    """One dask task: :func:`run_one`, logged to the epoch's own file
    (``run_epoch_logged``).

    A failure is returned rather than raised, so that it is recorded and the
    other epochs run on. Either way the epoch ends by closing the NWB files
    Spyglass opened for its marks: a ``LocalCUDACluster`` worker runs one task
    at a time, so no other epoch in the process reads them.
    """
    git_commit = run_one_kwargs["provenance"]["git_commit"]

    def run(log: logging.Logger) -> dict:
        try:
            return run_one(epoch_id, **run_one_kwargs)
        except Exception as error:  # noqa: BLE001 - one epoch must not stop the rest
            return _failure_record(error, git_commit)

    return run_epoch_logged(
        run,
        log_name=f"{epoch_id}_memoryless",
        log_path=log_directory / f"{epoch_id}_memoryless.log",
        start=f"{epoch_id}: run {run_one_kwargs['run_id']}, commit {git_commit}",
        describe=_describe,
        log_provenance=False,
    )


def _describe(record: dict) -> str:
    """The epoch's log summary: its counts, or its error when it failed."""
    if record["status"] == "failed":
        return f"failed: {record['error']}"
    return _counts(record)


THREAD_VARIABLES = BLAS_THREAD_VARIABLES
"""The BLAS and OpenMP thread counts dask's nanny sets to 1 in each worker it
spawns (``distributed.nanny.pre-spawn-environ``). Set before the worker starts,
since numpy reads them when it is imported."""


def available_cpus() -> int:
    """The CPUs this process may run on (its affinity), as its workers inherit."""
    return len(os.sched_getaffinity(0))


def worker_thread_environ(n_workers: int, n_cpus: int) -> dict[str, int]:
    """Each worker's BLAS and OpenMP thread count: its share of the CPUs.

    Parameters
    ----------
    n_workers : int
        Workers sharing the CPUs, one per GPU.
    n_cpus : int

    Returns
    -------
    dict
        Each of ``THREAD_VARIABLES`` set to ``n_cpus // n_workers``, at least 1.
    """
    return dict.fromkeys(THREAD_VARIABLES, max(1, n_cpus // n_workers))


@contextlib.contextmanager
def environment_holding(values: dict[str, str]) -> Iterator[None]:
    """This process's environment holds ``values`` inside the block. Each
    variable is then restored, or removed if it was not set."""
    inherited = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in inherited.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def cuda_cluster(
    cuda_visible_devices: str, local_directory: str, worker_xla_flags: str
) -> Cluster:
    """A ``LocalCUDACluster``, one worker per listed GPU, whose workers may each
    use their share of the CPUs and run XLA deterministically.

    dask's nanny sets ``THREAD_VARIABLES`` to 1 in the environment it spawns
    each worker with (``distributed.nanny.pre-spawn-environ``), so every worker
    ran its BLAS and OpenMP on one core. A value already in this process's
    environment overrides that configuration, and a nanny leaves its values in
    its parent's environment. So while the cluster is made, and its nannies
    spawn their workers, both this process's environment and the
    configuration hold ``worker_thread_environ``'s share of ``available_cpus``,
    and ``XLA_FLAGS`` set to ``worker_xla_flags``; the environment is then
    restored. The nannies keep the values for any restart.

    Parameters
    ----------
    cuda_visible_devices : str
        The GPUs, such as ``"3,4,6,7"``: one worker each.
    local_directory : str
        The workers' scratch folder.
    worker_xla_flags : str
        Each worker's ``XLA_FLAGS``: the run settings' ``xla_flags``
        (:func:`xla_flags`), so the manifest records what the workers get.
    """
    n_workers = len(str(cuda_visible_devices).split(","))
    worker_environ = {
        **{
            name: str(n)
            for name, n in worker_thread_environ(n_workers, available_cpus()).items()
        },
        "XLA_FLAGS": worker_xla_flags,
    }
    environ = {
        **dask.config.get("distributed.nanny.pre-spawn-environ"),
        **worker_environ,
    }
    with (
        environment_holding(worker_environ),
        dask.config.set({"distributed.nanny.pre-spawn-environ": environ}),
    ):
        cluster: Cluster = cuda_cluster_class()(
            CUDA_VISIBLE_DEVICES=cuda_visible_devices,
            local_directory=local_directory,
        )
    return cluster


def run_epochs(
    epoch_ids: list[str],
    run_one_kwargs: dict,
    cluster_factory: Callable[[], Cluster],
) -> Iterator[tuple[str, dict]]:
    """Run each epoch as its own dask task; yield each record as it finishes.

    Each task runs :func:`run_one` and logs to ``logs/{epoch_id}_memoryless.log``
    under the working directory. Nothing here writes the manifest: the caller
    does, after each record, so it has a single writer.

    Parameters
    ----------
    epoch_ids : list of str
    run_one_kwargs : dict
        Every argument of :func:`run_one` but the epoch.
    cluster_factory : callable
        Makes the cluster, which is closed once the last record is yielded;
        none is made for no epochs. Its workers should run one task at a time,
        as ``LocalCUDACluster``'s do: ``setup_logger`` reconfigures logging for
        the whole worker process, which closes the log of any other task
        running in it.

    Yields
    ------
    epoch_id : str
    record : dict
        What :func:`run_one` returned, in the order the epochs finish. A
        failure is ``status`` ``"failed"`` with ``error``, its repr, and its
        traceback. It is caught in its worker, or here when the worker could
        not return it, as when the worker's process died.
    """
    if not epoch_ids:
        return
    log_directory = Path.cwd() / "logs"
    git_commit = run_one_kwargs["provenance"]["git_commit"]
    yield from batch_run_epochs(
        {epoch_id: (epoch_id, run_one_kwargs, log_directory) for epoch_id in epoch_ids},
        _run_one_logged,
        cluster_factory,
        key_prefix="memoryless",
        lost=functools.partial(_failure_record, git_commit=git_commit),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--epoch-id", action="append", help="One epoch; repeatable.")
    which.add_argument("--animal", help="Every canonical HPC decode of this animal.")
    which.add_argument("--all", action="store_true", help="Every canonical HPC decode.")
    parser.add_argument(
        "--date",
        type=int,
        help="Only epochs recorded on this date, such as 20200225: the last 8 "
        "characters of the NWB file's stem.",
    )
    parser.add_argument("--epoch", help="Only this epoch, such as 02_r1.")
    parser.add_argument(
        "--run-seed", type=int, required=True, help="Recorded in the manifest."
    )
    parser.add_argument("--n-shuffles", type=int, default=DEFAULT_N_SHUFFLES)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument(
        "--pending-only",
        action="store_true",
        help="With --all or --animal: run only the epochs the epoch register holds "
        "pending, from their staged decodes under --root, instead of its "
        "analysed epochs.",
    )
    parser.add_argument(
        "--include-variants",
        action="store_true",
        help="Also run decodes made another way (_all, _non_stationary, ...).",
    )
    parser.add_argument(
        "--run-id", type=parse_run_id, help="Defaults to the commit and the seed."
    )
    parser.add_argument("--root", type=Path, default=processed_data_dir())
    parser.add_argument("--output-root", type=Path, default=memoryless_root())
    parser.add_argument("--cache-dir", type=Path, default=marks_cache_dir())
    parser.add_argument(
        "--overwrite", action="store_true", help="Redo epochs already written."
    )
    parser.add_argument(
        "--keep-single-window-runs",
        action="store_true",
        help="The sensitivity setting: do not drop runs of one kept window "
        "before merging. Recorded in the manifest's config.",
    )
    parser.add_argument(
        "--cuda_visible_devices",
        help="Run the epochs in parallel on these GPUs, such as 0,1,2: a dask "
        "LocalCUDACluster with one worker per GPU and one epoch per task. "
        "Without it the epochs run one after another in this process.",
    )
    parser.add_argument(
        "--local_scratch_dir",
        default=".local_scratch",
        help="The dask workers' scratch folder, with --cuda_visible_devices.",
    )
    args = parser.parse_args(argv)

    if args.pending_only and args.epoch_id:
        parser.error("--pending-only selects from the register: use --all or --animal")
    shuffle = not args.no_shuffle
    if shuffle and args.n_shuffles < MINIMUM_N_SHUFFLES:
        parser.error(
            f"--n-shuffles must be at least {MINIMUM_N_SHUFFLES}, so that a "
            f"p-value at or below {SIGNIFICANCE_LEVEL} is possible."
        )
    if args.cuda_visible_devices is not None and cuda_cluster_class() is None:
        sys.exit(
            "--cuda_visible_devices needs dask_cuda, which is not importable. "
            "Install dask-cuda by hand (see docs/README.md), or leave out "
            "--cuda_visible_devices to run the epochs one after another in "
            "this process."
        )
    commit = git_hash()
    run_id = args.run_id or (
        f"{commit[:12]}{'-dirty' if commit.endswith('-dirty') else ''}"
        f"-seed{args.run_seed}{'' if shuffle else '-noshuffle'}"
        f"{'-singles' if args.keep_single_window_runs else ''}"
    )
    output_dir = args.output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    log = _logger(output_dir / "run.log")
    # before any cluster starts: its worker processes inherit this limit
    log.info("open-file soft limit %d -> %d", *raise_open_file_limit())
    if commit.endswith("-dirty"):
        log.warning(
            "uncommitted changes (%s); this run cannot be tied to a commit", commit
        )
    config = MemorylessSequenceConfig(
        drop_single_window_runs=not args.keep_single_window_runs
    )
    settings = run_settings(args.run_seed, shuffle, args.n_shuffles, config, commit)
    provenance = {
        "produced_by": produced_by(__file__),
        "git_commit": commit,
        "run_id": run_id,
    }

    skipped: dict[str, str] = {}
    if args.epoch_id:
        epoch_ids = list(args.epoch_id)
    else:
        epoch_ids, skipped = find_epochs(
            None if args.all else args.animal,
            args.root,
            epochs_to_run(args.pending_only).index,
            include_variants=args.include_variants,
            left_out=NOT_PENDING if args.pending_only else NOT_ANALYSED,
        )
    epoch_ids = select_epochs(epoch_ids, date=args.date, epoch=args.epoch)
    for epoch_id, why in sorted(skipped.items()):
        log.info("left out %s: %s", epoch_id, why)
    if skipped:
        log.info("%d epochs left out; %d to run", len(skipped), len(epoch_ids))
    if not epoch_ids:
        sys.exit(
            f"No HPC decodes to run under {args.root}"
            + ("" if args.date is None else f" on --date {args.date}")
            + ("" if args.epoch is None else f" for --epoch {args.epoch}")
            + "."
        )

    manifest_path = output_dir / "manifest.json"
    epochs: dict[str, dict] = {}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get("settings") != settings:
            sys.exit(
                f"{manifest_path} was written with other settings, code or "
                "library versions; a run must not mix them. Start a new --run-id."
            )
        epochs = previous["epochs"]
    for epoch_id, why in skipped.items():
        # an epoch already written under this run stays written; a skip only
        # says why it was not run this time
        if epochs.get(epoch_id, {}).get("status") != "ok":
            epochs[epoch_id] = {"status": "skipped", "reason": why}

    def manifest() -> dict:
        return {
            "produced_by": provenance["produced_by"],
            "run_id": run_id,
            "design_spec": DESIGN_SPEC,
            "settings": settings,
            "epochs": epochs,
        }

    run_one_kwargs = {
        "root": args.root,
        "output_dir": output_dir,
        "cache_dir": args.cache_dir,
        "run_id": run_id,
        "run_seed": args.run_seed,
        "shuffle": shuffle,
        "n_shuffles": args.n_shuffles,
        "config": config,
        "provenance": provenance,
    }
    write_manifest(manifest_path, manifest())
    failed = []
    if args.cuda_visible_devices is None:
        for number, epoch_id in enumerate(epoch_ids, start=1):
            done = epochs.get(epoch_id, {}).get("status") == "ok"
            if done and not args.overwrite:
                log.info(
                    "[%d/%d] %s: already written", number, len(epoch_ids), epoch_id
                )
                continue
            log.info("[%d/%d] %s", number, len(epoch_ids), epoch_id)
            try:
                # this process runs JAX itself: XLA reads the flags when JAX
                # starts its backend, in the first epoch
                with environment_holding({"XLA_FLAGS": settings["xla_flags"]}):
                    epochs[epoch_id] = run_one(epoch_id, **run_one_kwargs)
            except Exception as error:  # noqa: BLE001 - one epoch must not stop the rest
                # recorded, not swallowed: the manifest says which epoch failed
                # and why, the run exits non-zero, and a rerun retries it
                failed.append(epoch_id)
                epochs[epoch_id] = {
                    "status": "failed",
                    "git_commit": commit,
                    "produced_at": datetime.now(UTC).isoformat(timespec="seconds"),
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                }
                log.error("    failed: %s: %s", type(error).__name__, error)
            else:
                log.info("    %s", _counts(epochs[epoch_id]))
            finally:
                # one epoch at a time in this process: nothing else reads them
                log.info(
                    "    NWB files Spyglass held open closed: %s", release_nwb_files()
                )
            write_manifest(manifest_path, manifest())
    else:
        to_run = []
        # an epoch named twice runs once: its task is keyed by the epoch
        for epoch_id in dict.fromkeys(epoch_ids):
            if epochs.get(epoch_id, {}).get("status") == "ok" and not args.overwrite:
                log.info("%s: already written", epoch_id)
            else:
                to_run.append(epoch_id)
        log.info(
            "%d epochs to run on CUDA devices %s, one per task, each logging to %s",
            len(to_run),
            args.cuda_visible_devices,
            Path.cwd() / "logs" / "<epoch_id>_memoryless.log",
        )
        # the workers get the XLA flags the settings record, computed once
        cluster_factory = functools.partial(
            cuda_cluster,
            args.cuda_visible_devices,
            args.local_scratch_dir,
            settings["xla_flags"],
        )
        # the workers only return records: this process alone writes the manifest
        for number, (epoch_id, record) in enumerate(
            run_epochs(to_run, run_one_kwargs, cluster_factory), start=1
        ):
            epochs[epoch_id] = record
            if record["status"] == "ok":
                log.info(
                    "[%d/%d] %s: %s", number, len(to_run), epoch_id, _counts(record)
                )
            else:
                failed.append(epoch_id)
                log.error(
                    "[%d/%d] %s failed: %s",
                    number,
                    len(to_run),
                    epoch_id,
                    record["error"],
                )
            write_manifest(manifest_path, manifest())
    log.info("manifest -> %s", manifest_path)
    if failed:
        sys.exit(
            f"{len(failed)} epochs failed: {', '.join(failed)}; see {manifest_path}"
        )


def _logger(path: Path) -> logging.Logger:
    """Everything the run says, on the console and in ``run.log`` beside the manifest."""
    log = logging.getLogger("run_memoryless_detection")
    log.setLevel(logging.INFO)
    log.propagate = False
    for handler in list(log.handlers):
        log.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(path)):
        handler.setFormatter(formatter)
        log.addHandler(handler)
    return log


if __name__ == "__main__":
    main()
