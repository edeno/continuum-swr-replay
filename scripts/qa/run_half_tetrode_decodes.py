#!/usr/bin/env python
"""Held-out state labels: decode with one tetrode half, score with the other.

Task 12 (``run_state_evidence.py``) scores the all-tetrode decode's labels, so
the scoring half C helped make every label it judges. Here each label comes
from a decode that saw one half S alone, and C alone scores it. The method is in
``continuum_swr_replay.analysis.hippocampus.state_evidence.half_decode``.

Two steps, two subcommands:

``decode`` (a GPU, or the CPU for a smoke test). Per epoch, the first two of
Task 12's tetrode partitions, each in both directions: four decodes, ``p0d0``,
``p0d1``, ``p1d0`` and ``p1d1``. Each one:

1. fits the encoding model on S's tetrodes alone, at the canonical decode's own
   training bins (the inputs its fitted detector keeps), with no weights: the
   canonical encoding is the plain fit, because the library's clusterless KDE
   ignores EM's weights, and the record checks that the half encoding is the
   canonical one restricted to S;
2. keeps the canonical fit's discrete transitions (12 numbers) and initial
   conditions fixed, and runs no EM;
3. decodes S's spikes.

``--all-tetrodes`` decodes with every tetrode instead (split ``all``), to check
that the pipeline reproduces the canonical decode. Each decode writes
``decodes/{epoch}_{split}.nc`` (the four state probabilities, float32, on the
decode's time axis, with S, C and the fixed transitions in its attributes) and
``decodes/{epoch}_{split}.json``, its record: the half encoding's check, the
agreement of its states and non-local events with the canonical decode's, the
device and the run time. A decode already written is kept unless
``--overwrite``.

A GPU must be named, one card, idle at launch (at most 2 GiB in use and 0 %
utilization by nvidia-smi): ``--cuda_visible_devices N``. Without it, JAX is
held to the CPU (``JAX_PLATFORMS=cpu``, ``CUDA_VISIBLE_DEVICES`` empty).
``--max-bins`` decodes only an epoch's first bins, for a smoke test; ``score``
refuses such decodes.

``score`` (the CPU). Task 12's per-epoch evidence (``run_state_evidence``) with
eight label sets per epoch, each scored by one (S, C) run (``window_evidence``
with ``both_directions=False``):

- ``label_source`` "half": each half decode's labels, scored by its own C;
- ``label_source`` "all": the canonical decode's labels, scored by the same
  four runs: the matched comparison, which differs from the half labels only
  in where the labels come from. (Task 12's own summary averages 20 runs per
  window, so its window values are less spread.)

``split`` names the run. It writes ``evidence/per_epoch/{epoch}_windows.parquet``
and ``{epoch}_validation.json``; then, per label source, Task 12's summary
tables and figures (``evidence/half/``, ``evidence/all/``), with each epoch's
value pooling its four splits' windows; ``evidence/half_vs_all_bouts.csv``
(the primary bout statistics side by side); ``evidence/manifest.json`` and a
``README.md``.

It writes nothing outside ``--output-dir``.

Usage. One GPU per process; the five pilot epochs by default. The decodes need
JAX's CUDA plugin (the ``cuda`` extra). An environment that has it, with the same
``uv.lock``, can run this work tree's code through ``PYTHONPATH``; ``-B`` writes
no bytecode there. Pick idle cards with ``nvidia-smi --query-gpu=index,
memory.used,utilization.gpu --format=csv``; ``G1`` and ``G2`` below stand for
them. Run the check (1) first, then the half decodes (2), then score (3)::

    WT=/cumulus/edeno/continuum-swr-replay-refresh-dev
    GPU_PY="env PYTHONPATH=$WT/src /cumulus/edeno/continuum-swr-replay/.venv/bin/python -B"
    DEC=/cumulus/edeno/continuum-swr-replay/Processed-Data/_staging_redecode_20261004
    OUT=/cumulus/edeno/continuum-swr-replay/Processed-Data/_refresh_20261004/half_tetrode
    cd $WT
    # 1. S = every tetrode, one pilot epoch: the agreement with the canonical decode
    $GPU_PY scripts/qa/run_half_tetrode_decodes.py decode --all-tetrodes \\
        --epochs peanut20201202_10_r5_HPC --decode-dir $DEC --output-dir $OUT \\
        --cuda_visible_devices G1
    # 2. the 20 half decodes, on two cards
    $GPU_PY scripts/qa/run_half_tetrode_decodes.py decode --epochs \\
        peanut20201202_10_r5_HPC j1620210707_06_r3_HPC chimi20200225_02_r1_HPC \\
        --decode-dir $DEC --output-dir $OUT --cuda_visible_devices G1
    $GPU_PY scripts/qa/run_half_tetrode_decodes.py decode --epochs \\
        wilbur20210329_12_r6_HPC senor20201101_06_r3_HPC \\
        --decode-dir $DEC --output-dir $OUT --cuda_visible_devices G2
    # 3. score, on the CPU (this work tree's own environment; reads Spyglass)
    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= .venv/bin/python \\
        scripts/qa/run_half_tetrode_decodes.py score \\
        --decode-dir $DEC --output-dir $OUT --n-workers 5
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time as clock
import traceback
from collections.abc import Callable
from datetime import date
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# the sibling Task 12 runner, imported only by ``score``: importing it holds JAX
# to the CPU, and the decodes may run on a GPU
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from continuum_swr_replay.decoding_common import (  # noqa: E402
    provenance_record,
    setup_logger,
    write_atomically,
)
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR, ROOT_DIR  # noqa: E402

PRODUCED_BY = "scripts/qa/run_half_tetrode_decodes.py"
PILOT_EPOCHS = (
    "peanut20201202_10_r5_HPC",
    "j1620210707_06_r3_HPC",
    "chimi20200225_02_r1_HPC",
    "wilbur20210329_12_r6_HPC",
    "senor20201101_06_r3_HPC",
)
IDLE_MAX_MIB = 2048
IDLE_MAX_UTILIZATION = 0
ALL_TETRODES = "all"
OCCUPANCY_MAX_REL = 1e-2
"""The half encoding's occupancy against the canonical one's, as a share of its
maximum. The two are the same KDE of the same training positions, but its
float32 sums over about 600,000 positions differ between the CPU and the
canonical decode's GPU: by 5.1e-4 (peanut) and 1.4e-4 (chimi) of the maximum on
the pilot. Training spikes, marks and mean rates must match exactly, which
catches any change of training bins."""
HALF_SPLIT_NAMES = ("p0d0", "p0d1", "p1d0", "p1d1")
"""``Split.name`` of ``half_splits``'s four splits, in its order."""
LABEL_SOURCES = {
    "half": "a decode that saw S alone, scored by C",
    "all": "the canonical all-tetrode decode, scored by the same (S, C) runs",
}


# --- devices ------------------------------------------------------------------------------
def gpu_usage() -> dict[int, tuple[float, float]]:
    """Each GPU's memory in use (MiB) and utilization (%), from nvidia-smi."""
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    usage = {}
    for line in out.strip().splitlines():
        index, *values = (field.strip() for field in line.split(","))
        memory, utilization = (_number(value) for value in values)
        usage[int(index)] = (memory, utilization)
    return usage


def _number(text: str) -> float:
    """A number nvidia-smi printed; NaN for "[N/A]" and the like."""
    try:
        return float(text)
    except ValueError:
        return float("nan")


def configure_devices(
    cuda_visible_devices: str | None,
    *,
    query: Callable[[], dict[int, tuple[float, float]]] = gpu_usage,
) -> str:
    """Hold JAX to the CPU, or to the one idle GPU named. Call before JAX loads.

    Parameters
    ----------
    cuda_visible_devices : str or None
        One GPU index, as nvidia-smi numbers them; None for the CPU.
    query : callable, optional
        Each GPU's memory in use (MiB) and utilization (%), by index.

    Returns
    -------
    str
        ``"cpu"`` or ``"gpu:{index}"``.

    Raises
    ------
    SystemExit
        Not exactly one index, no such GPU, or the card is not idle: more than
        :data:`IDLE_MAX_MIB` MiB in use or more than
        :data:`IDLE_MAX_UTILIZATION` % utilization.
    """
    if cuda_visible_devices is None:
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return "cpu"
    names = [name.strip() for name in cuda_visible_devices.split(",") if name.strip()]
    if len(names) != 1 or not names[0].isdigit():
        raise SystemExit(
            "--cuda_visible_devices must name exactly one GPU by its nvidia-smi "
            f"index, not {cuda_visible_devices!r}; run one process per card"
        )
    index = int(names[0])
    usage = query()
    if index not in usage:
        raise SystemExit(f"no GPU {index}: nvidia-smi lists {sorted(usage)}")
    memory, utilization = usage[index]
    # an unreadable value (NaN) is not idle
    if not (memory <= IDLE_MAX_MIB and utilization <= IDLE_MAX_UTILIZATION):
        raise SystemExit(
            f"GPU {index} is not idle: {memory:.0f} MiB in use, {utilization:.0f} % "
            f"busy (idle: at most {IDLE_MAX_MIB} MiB and {IDLE_MAX_UTILIZATION} %)"
        )
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # nvidia-smi's numbering
    os.environ["CUDA_VISIBLE_DEVICES"] = str(index)
    os.environ["JAX_PLATFORMS"] = "cuda"
    # allocate as needed rather than most of the card up front
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    return f"gpu:{index}"


def current_device() -> str:
    """Where JAX computes: ``"cpu"`` or ``"gpu:{CUDA_VISIBLE_DEVICES}"``."""
    import jax

    if jax.default_backend() == "cpu":
        return "cpu"
    return f"gpu:{os.environ.get('CUDA_VISIBLE_DEVICES', '?')}"


# --- decode -------------------------------------------------------------------------------
def decode_paths(epoch_id: str, split: str, output_dir: Path) -> dict[str, Path]:
    """One decode's probabilities (``decode``) and record (``record``)."""
    stem = Path(output_dir) / "decodes" / f"{epoch_id}_{split}"
    return {"decode": stem.with_suffix(".nc"), "record": stem.with_suffix(".json")}


def _write_json(path: Path, content: object) -> None:
    text = json.dumps(content, indent=1, allow_nan=False, default=str)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomically(path, lambda temporary: temporary.write_text(text))


def _finite_or_none(value: Any) -> Any:
    """``value`` with every non-finite float as None, for strict JSON."""
    if isinstance(value, dict):
        return {k: _finite_or_none(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite_or_none(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def canonical_detector_stamp(epoch_id: str, decode_dir: Path) -> dict:
    """The canonical detector file's path, size and ``mtime_ns``."""
    path = (Path(decode_dir) / f"{epoch_id}_detector.pkl").resolve()
    info = path.stat()
    return {"path": str(path), "size": info.st_size, "mtime_ns": info.st_mtime_ns}


def reusable(epoch_id: str, split: str, output_dir: Path, n_bins: int | None) -> bool:
    """Whether a decode written before can stand: ok, and of the length asked for.

    A smoke test's decode of the first bins (``--max-bins``) never stands in for
    a whole epoch's, nor the reverse.
    """
    paths = decode_paths(epoch_id, split, output_dir)
    if not all(path.exists() for path in paths.values()):
        return False
    record = json.loads(paths["record"].read_text())
    if record.get("status") != "ok":
        return False
    if n_bins is None:
        return bool(record.get("complete"))
    return record.get("n_bins") == n_bins and not record.get("complete")


def decode_epoch(
    epoch_id: str,
    *,
    decode_dir: Path,
    output_dir: Path,
    all_tetrodes: bool = False,
    n_bins: int | None = None,
    overwrite: bool = False,
    log: Callable[[str], None] = print,
) -> list[dict]:
    """Decode one epoch with each split's half S (or every tetrode); write the records.

    Returns
    -------
    list of dict
        One record per split, also written beside its decode: ``status`` "ok",
        "exists" (kept from an earlier run that decoded as many bins, every bin
        when ``n_bins`` is None; not rewritten) or "failed" (with ``error`` and
        ``traceback``; no decode is written, so a rerun redoes it).
    """
    import xarray as xr

    from continuum_swr_replay.analysis.hippocampus.state_evidence import (
        ENCODING_KEY,
        HALVES_SEED,
        STATES,
        decode_inputs,
        encoding_comparison,
        half_splits,
        half_tetrode_decode,
        label_agreement,
        load_detector,
        non_local_event_labels,
    )

    names = [ALL_TETRODES] if all_tetrodes else list(HALF_SPLIT_NAMES)
    kept = (
        set()
        if overwrite
        else {name for name in names if reusable(epoch_id, name, output_dir, n_bins)}
    )
    for name in names:
        if name in kept:
            log(f"{epoch_id} {name}: written before; kept")
    if kept == set(names):
        return [
            {"epoch_id": epoch_id, "split": name, "status": "exists"} for name in names
        ]

    started = clock.perf_counter()
    stamp = canonical_detector_stamp(epoch_id, decode_dir)
    canonical = load_detector(epoch_id, decode_dir)
    inputs = decode_inputs(canonical)
    dot = pd.read_pickle(Path(decode_dir) / f"{epoch_id}_80_data_over_time.pkl")
    time = dot.index.to_numpy(dtype=float)
    if not np.array_equal(time, np.asarray(inputs["position_time"], dtype=float)):
        raise ValueError(
            "the detector's decode inputs are not on data_over_time's axis"
        )
    columns = [f"{state.replace(' ', '_')}_prob" for state in STATES]
    reference = dot[columns].to_numpy()  # float32, as the decode wrote them
    reference_labels = dot["non_local_event"].to_numpy(dtype=np.int64)
    speed = dot["head_speed"].to_numpy(dtype=float)
    del dot
    # the event labels here are made as the canonical pipeline makes them
    labels_reproduced = bool(
        np.array_equal(non_local_event_labels(reference, speed, time), reference_labels)
    )
    log(f"{epoch_id}: inputs read in {clock.perf_counter() - started:.0f} s")

    n_tetrodes = len(inputs["spike_times"])
    every = np.arange(n_tetrodes)
    jobs = (
        [(ALL_TETRODES, None, None, every, every[:0])]
        if all_tetrodes
        else [
            (s.name, s.partition, s.direction, s.selecting, s.scoring)
            for s in half_splits(n_tetrodes, seed=HALVES_SEED)
        ]
    )
    if [job[0] for job in jobs] != names:
        raise ValueError(f"the splits are {[job[0] for job in jobs]}, not {names}")
    stored = canonical.encoding_model_[ENCODING_KEY]
    provenance = provenance_record(ROOT_DIR)
    records = []
    for split, partition, direction, selecting, scoring in jobs:
        paths = decode_paths(epoch_id, split, output_dir)
        if split in kept:
            records.append({"epoch_id": epoch_id, "split": split, "status": "exists"})
            continue
        record: dict[str, Any] = {
            "epoch_id": epoch_id,
            "split": split,
            "partition": partition,
            "direction": direction,
            "selecting_tetrodes": selecting.tolist(),
            "scoring_tetrodes": scoring.tolist(),
            "decode_dir": str(decode_dir),
            "canonical_detector": stamp,
            "code": {k: provenance[k] for k in ("commit", "dirty")},
        }
        began = clock.perf_counter()
        try:
            results, half = half_tetrode_decode(canonical, selecting, n_bins=n_bins)
            probabilities = results.acausal_state_probabilities.values
            decoded = probabilities.shape[0]
            if list(results.states.values) != list(STATES) or not np.array_equal(
                results.time.values, time[:decoded]
            ):
                raise ValueError(
                    "the decode's states or time axis are not the canonical"
                )
            record["device"] = current_device()
            record["decode_s"] = clock.perf_counter() - began
            encoding = half.encoding_model_[ENCODING_KEY]
            subset = {
                key: [stored[key][i] for i in selecting]
                for key in (
                    "encoding_positions",
                    "encoding_spike_waveform_features",
                    "mean_rates",
                )
            }
            check = encoding_comparison(encoding, subset)
            occupancy = np.asarray(stored["occupancy"], dtype=float)
            check["max_abs_occupancy"] = float(
                np.max(np.abs(np.asarray(encoding["occupancy"]) - occupancy))
            )
            check["max_rel_occupancy"] = check["max_abs_occupancy"] / float(
                np.max(np.abs(occupancy))
            )
            check["same_as_canonical_subset"] = bool(
                check["same_n_encoding_spikes"]
                and check["max_abs_encoding_positions"] == 0.0
                and check["max_abs_encoding_marks"] == 0.0
                and check["max_abs_mean_rates"] == 0.0
                and check["max_rel_occupancy"] <= OCCUPANCY_MAX_REL
            )
            record["half_encoding"] = {
                "n_electrodes": len(subset["mean_rates"]),
                **check,
            }
            if not check["same_as_canonical_subset"]:
                raise ValueError(
                    "the half encoding is not the canonical encoding restricted to "
                    f"its tetrodes: {check}"
                )
            labels = non_local_event_labels(
                probabilities, speed[:decoded], time[:decoded]
            )
            record["agreement"] = label_agreement(
                reference[:decoded], reference_labels[:decoded], probabilities, labels
            )
            record["agreement"]["reference_labels_reproduced"] = labels_reproduced
            record["n_bins"] = int(decoded)
            record["complete"] = bool(decoded == time.size)
            transitions = np.asarray(canonical.discrete_state_transitions_).tolist()
            record["fixed"] = {
                "discrete_state_transitions": transitions,
                "initial_conditions": "the canonical fit's: its smoother's posterior "
                "at the first bin, over every state bin",
            }
            record["marginal_log_likelihood"] = float(
                results.attrs["marginal_log_likelihoods"]
            )
            dataset = xr.Dataset(
                {
                    "acausal_state_probabilities": (
                        ("time", "states"),
                        np.asarray(probabilities, dtype=np.float32),
                    )
                },
                coords={"time": time[:decoded], "states": list(STATES)},
                attrs={
                    "epoch_id": epoch_id,
                    "split": split,
                    "selecting_tetrodes": json.dumps(record["selecting_tetrodes"]),
                    "scoring_tetrodes": json.dumps(record["scoring_tetrodes"]),
                    "discrete_state_transitions": json.dumps(transitions),
                    "canonical_detector": json.dumps(stamp),
                    "marginal_log_likelihood": record["marginal_log_likelihood"],
                    "n_bins": int(decoded),
                    "complete": int(record["complete"]),
                    "produced_by": PRODUCED_BY,
                    "provenance": json.dumps(provenance, sort_keys=True),
                },
            )
            paths["decode"].parent.mkdir(parents=True, exist_ok=True)
            write_atomically(
                paths["decode"], partial(dataset.to_netcdf, engine="h5netcdf")
            )
            record["status"] = "ok"
            del results, half
        except Exception as error:  # noqa: BLE001 - recorded; the next split runs
            paths["decode"].unlink(missing_ok=True)
            record.update(
                status="failed", error=repr(error), traceback=traceback.format_exc()
            )
        record["wall_clock_s"] = clock.perf_counter() - began
        _write_json(paths["record"], _finite_or_none(record))
        log(
            f"{epoch_id} {split}: {record['status']} in {record['wall_clock_s']:.0f} s"
            + (f" ({record['error']})" if record["status"] == "failed" else "")
        )
        records.append(record)
    return records


# --- score --------------------------------------------------------------------------------
def evidence_paths(epoch_id: str, output_dir: Path) -> dict[str, Path]:
    """One epoch's per-window table (``windows``) and record (``validation``)."""
    from scripts.qa import run_state_evidence

    return run_state_evidence.output_paths(epoch_id, Path(output_dir) / "evidence")


def half_label_sets(epoch_id: str, *, output_dir: Path, decode_dir: Path) -> list:
    """An epoch's eight label sets: its four half decodes', and the canonical decode's.

    Each half decode's labels are scored by its own C only; the canonical
    labels are scored by the same four (S, C) runs.

    Raises
    ------
    ValueError
        A half decode is missing, incomplete (``--max-bins``), not on the first
        two partitions' tetrodes, or made from another canonical detector file
        (path, size or ``mtime_ns``) than the one in ``decode_dir``.
    """
    import xarray as xr

    from continuum_swr_replay.analysis.hippocampus.state_evidence import (
        HALVES_SEED,
        SPLIT_COLUMN,
        STATES,
        half_splits,
    )
    from scripts.qa.run_state_evidence import LabelSet

    stamp = canonical_detector_stamp(epoch_id, decode_dir)
    half, matched = [], []
    for split in HALF_SPLIT_NAMES:
        path = decode_paths(epoch_id, split, output_dir)["decode"]
        if not path.exists():
            raise ValueError(f"no half decode {path.name}; run decode first")
        with xr.open_dataset(path, engine="h5netcdf") as decoded:
            if not decoded.attrs.get("complete"):
                raise ValueError(
                    f"{path.name} decodes only the first bins (--max-bins)"
                )
            if list(decoded.states.values) != list(STATES):
                raise ValueError(f"{path.name} holds states {decoded.states.values}")
            if json.loads(decoded.attrs["canonical_detector"]) != stamp:
                raise ValueError(
                    f"{path.name} was decoded from another canonical detector file "
                    "than the one scored now; decode again"
                )
            selecting = np.asarray(json.loads(decoded.attrs["selecting_tetrodes"]))
            scoring = np.asarray(json.loads(decoded.attrs["scoring_tetrodes"]))
            probabilities = decoded.acausal_state_probabilities.values
            time = decoded.time.values
        n_tetrodes = selecting.size + scoring.size
        expected = {s.name: s for s in half_splits(n_tetrodes, seed=HALVES_SEED)}[split]
        if not (
            np.array_equal(selecting, expected.selecting)
            and np.array_equal(scoring, expected.scoring)
        ):
            raise ValueError(f"{path.name} is not split {split} of the partitions")
        halves = [(selecting, scoring)]
        half.append(
            LabelSet(
                name=f"half|{split}",
                probabilities=probabilities,
                time=time,
                halves=halves,
                both_directions=False,
                columns={"label_source": "half", SPLIT_COLUMN: split},
            )
        )
        matched.append(
            LabelSet(
                name=f"all|{split}",
                halves=halves,
                both_directions=False,
                columns={"label_source": "all", SPLIT_COLUMN: split},
            )
        )
    return half + matched


def score_epoch(
    epoch_id: str,
    *,
    decode_dir: Path,
    output_dir: Path,
    load_inputs: Callable[[str], dict] | None = None,
    log: Callable[[str], None] = print,
    code: dict | None = None,
) -> dict:
    """Task 12's ``run_epoch`` for one epoch, with its half-decode label sets."""
    from scripts.qa import run_state_evidence

    kwargs: dict[str, Any] = {
        "decode_dir": Path(decode_dir),
        "output_dir": Path(output_dir) / "evidence",
        "code": code,
        "log": log,
        "label_sets": partial(
            half_label_sets, output_dir=Path(output_dir), decode_dir=Path(decode_dir)
        ),
    }
    if load_inputs is not None:
        kwargs["load_inputs"] = load_inputs
    return run_state_evidence.run_epoch(epoch_id, **kwargs)


def summarize_label_sources(output_dir: Path, epochs: list[str]) -> None:
    """Per label source, Task 12's summary tables and figures; then side by side.

    Each epoch's value pools its four splits' windows (Task 12's epoch summary),
    and bouts pair with their matched windows within their split.
    """
    from continuum_swr_replay.analysis.hippocampus.state_evidence import (
        summarize_state_evidence,
    )
    from continuum_swr_replay.table_io import read_table
    from scripts.qa import run_state_evidence as task12

    evidence_dir = Path(output_dir) / "evidence"
    frames = [
        read_table(evidence_paths(epoch_id, output_dir)["windows"])
        for epoch_id in epochs
    ]
    if not frames:
        print("no epoch scored; no summary", flush=True)
        return
    windows = pd.concat(frames, ignore_index=True)
    bouts = {}
    for source in LABEL_SOURCES:
        subset = windows[windows.label_source == source]
        summary = summarize_state_evidence(subset)
        folder = evidence_dir / source
        folder.mkdir(parents=True, exist_ok=True)
        for name, file_name in [
            ("bouts", "bout_stats.csv"),
            ("calibration", "calibration_stats.csv"),
            ("trends", "trend_stats.csv"),
            ("rate_calibration", "rate_calibration.csv"),
            ("silence", "silence_stats.csv"),
        ]:
            task12.write_csv(summary[name], folder / file_name)
        task12.figure_bouts(
            subset, summary["bouts"], folder / "fig_bouts_vs_matched.png"
        )
        task12.figure_calibration(
            summary["calibration"], folder / "fig_calibration.png"
        )
        task12.figure_silence(
            summary["silence"], folder / "fig_coordinated_silence.png"
        )
        bouts[source] = summary["bouts"]
    keys = ["state", "regime", "metric", "primary", "epoch_summary"]
    shown = [
        "n", "n_epochs", "n_animals", "median", "animal_estimate", "animal_ci_lo",
        "animal_ci_hi", "p_animal_t", "animals_pos", "epochs_pos",
        "diff_animal_estimate", "diff_animal_ci_lo", "diff_animal_ci_hi",
        "diff_p_animal_t", "diff_animals_pos",
    ]  # fmt: skip
    side_by_side = bouts["half"][bouts["half"].primary][keys + shown].merge(
        bouts["all"][bouts["all"].primary][keys + shown],
        on=keys,
        how="outer",
        suffixes=("_half", "_all"),
        validate="one_to_one",
    )
    task12.write_csv(side_by_side, evidence_dir / "half_vs_all_bouts.csv")
    print(side_by_side.to_string(index=False), flush=True)


def readme(records: dict[str, dict], provenance: dict, decode_dir: Path) -> str:
    """The evidence folder's README: what was run and what is held out."""
    from scripts.qa import run_state_evidence as task12

    status = pd.Series({k: r["status"] for k, r in records.items()}, dtype=object)
    lines = [
        "# Held-out state labels: half-tetrode decodes scored by the other half",
        "",
        f"Produced by `{PRODUCED_BY} score` at commit {provenance['commit']}"
        f"{' (uncommitted changes)' if provenance['dirty'] else ''}, from the "
        f"canonical decodes in `{decode_dir}` and the half decodes in "
        "`../decodes/`. Method: `continuum_swr_replay.analysis.hippocampus."
        "state_evidence.half_decode`.",
        "",
        "Status: "
        + ", ".join(f"{k} {v}" for k, v in status.value_counts().sort_index().items())
        + ".",
        "",
    ]
    failed = {k: r for k, r in records.items() if r["status"] != "ok"}
    if failed:
        lines += ["| epoch | status | reason |", "|---|---|---|"]
        lines += [
            f"| {k} | {r['status']} | {r.get('reason') or r.get('error')} |"
            for k, r in sorted(failed.items())
        ]
        lines.append("")
    lines += [
        "## What the labels hold out",
        "",
        "- `label_source` half: each split's labels come from a decode that saw "
        "half S alone: its encoding was fit on S's tetrodes at the canonical "
        "training bins, and it decoded S's spikes. It kept the canonical fit's "
        "discrete transitions (a 4 x 4 matrix, 12 free numbers, 9 of them learned "
        "by EM) and initial conditions (the canonical smoother's posterior at the "
        "first bin): the only information from C's spikes in the labels. C alone "
        "scores them (one run per window).",
        "- `label_source` all: the canonical decode's labels, scored by the same "
        "four (S, C) runs, one per window: the matched comparison. Task 12's own "
        "summary averages 20 runs per window, so its window values are less "
        "spread and its medians are not directly comparable with these.",
        "- `split` p{partition}d{direction}: the first two of Task 12's tetrode "
        "partitions, each in both directions. An epoch's value pools its four "
        "splits' windows, and bouts pair with their matched windows within their "
        "split.",
        "- Encodings, by column prefix: "
        + "; ".join(f"{p}_ {text}" for p, text in task12.ENCODING_HELD_OUT.items())
        + ".",
        "- Inference as in Task 12: each epoch enters as one value and the animal "
        "is the unit of replication (`run_state_evidence.py`'s README).",
        "",
        "## Files",
        "",
        "- `per_epoch/{epoch}_windows.parquet`, `per_epoch/{epoch}_validation.json`",
        "- `half/`, `all/`: `bout_stats.csv`, `calibration_stats.csv`, "
        "`trend_stats.csv`, `rate_calibration.csv`, `silence_stats.csv` and "
        "the three figures",
        "- `half_vs_all_bouts.csv`: the primary bout statistics, side by side",
        "- `manifest.json`: provenance and every epoch's record",
        "",
    ]
    return "\n".join(lines)


# --- command line -------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, text in [
        ("decode", "Decode each epoch with half its tetrodes (GPU or CPU)."),
        ("score", "Score the half decodes' labels with the other half (CPU)."),
    ]:
        command = commands.add_parser(name, help=text)
        command.add_argument(
            "--epochs",
            nargs="+",
            default=list(PILOT_EPOCHS),
            help="HPC epoch ids (default: the five pilot epochs).",
        )
        command.add_argument(
            "--decode-dir",
            type=Path,
            default=PROCESSED_DATA_DIR,
            help="The canonical decodes (default: Processed-Data).",
        )
        command.add_argument(
            "--output-dir",
            type=Path,
            default=PROCESSED_DATA_DIR
            / f"_refresh_{date.today():%Y%m%d}"
            / "half_tetrode",
            help="The only place written to (default: "
            "Processed-Data/_refresh_<today>/half_tetrode).",
        )
    decode = commands.choices["decode"]
    decode.add_argument(
        "--cuda_visible_devices",
        help="One idle GPU, by its nvidia-smi index. Without it the decodes run "
        "on the CPU.",
    )
    decode.add_argument(
        "--all-tetrodes",
        action="store_true",
        help="Decode with every tetrode (split 'all'): the check that the "
        "pipeline reproduces the canonical decode.",
    )
    decode.add_argument(
        "--max-bins",
        type=int,
        help="Decode only each epoch's first bins (a smoke test).",
    )
    decode.add_argument(
        "--overwrite", action="store_true", help="Redo decodes already written."
    )
    score = commands.choices["score"]
    score.add_argument(
        "--n-workers",
        type=int,
        default=5,
        help="Dask CPU worker processes, one epoch each (default 5).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    decode_dir, output_dir = args.decode_dir.resolve(), args.output_dir.resolve()
    if output_dir == decode_dir:
        parser.error("--output-dir must not be the decode folder itself")
    if args.command == "decode":
        device = configure_devices(args.cuda_visible_devices)  # before JAX loads
        if current_device() != device:
            sys.exit(f"JAX computes on {current_device()}, not {device}")
        logger = setup_logger(
            name_logfile="half_tetrode_decode",
            path_logfile=output_dir / "logs" / f"decode_{os.getpid()}.log",
        )
        logger.info("decoding %s on %s", ", ".join(args.epochs), device)
        failed = []
        for epoch_id in args.epochs:
            try:
                split_records = decode_epoch(
                    epoch_id,
                    decode_dir=decode_dir,
                    output_dir=output_dir,
                    all_tetrodes=args.all_tetrodes,
                    n_bins=args.max_bins,
                    overwrite=args.overwrite,
                    log=logger.info,
                )
            except Exception as error:  # noqa: BLE001 - the next epoch runs
                logger.exception("%s: %r", epoch_id, error)
                failed.append(epoch_id)
                continue
            failed += [
                f"{epoch_id} {r['split']}"
                for r in split_records
                if r["status"] == "failed"
            ]
        if failed:
            sys.exit(f"{len(failed)} decodes failed: {', '.join(failed)}")
        return

    from scripts.qa import run_state_evidence as task12  # holds JAX to the CPU

    provenance = provenance_record(ROOT_DIR)
    code = task12.code_version(provenance)
    evidence_dir = output_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    kwargs = {
        "decode_dir": decode_dir,
        "output_dir": evidence_dir,
        "code": code,
        "label_sets": partial(
            half_label_sets, output_dir=output_dir, decode_dir=decode_dir
        ),
    }
    records: dict[str, dict] = {}
    for epoch_id, record in task12.run_epochs(args.epochs, kwargs, args.n_workers):
        records[epoch_id] = record
        print(f"{epoch_id}: {record['status']}", flush=True)
    _write_json(
        evidence_dir / "manifest.json",
        {
            "produced_by": PRODUCED_BY,
            "provenance": provenance,
            "settings": task12.run_settings(),
            "label_sources": LABEL_SOURCES,
            "decode_dir": str(decode_dir),
            "epochs": {k: records[k] for k in sorted(records)},
        },
    )
    ok = [k for k in sorted(records) if records[k]["status"] == "ok"]
    summarize_label_sources(output_dir, ok)
    (evidence_dir / "README.md").write_text(readme(records, provenance, decode_dir))
    failed = [k for k, r in records.items() if r["status"] != "ok"]
    if failed:
        sys.exit(f"{len(failed)} epochs not scored: {', '.join(sorted(failed))}")


if __name__ == "__main__":
    main()
