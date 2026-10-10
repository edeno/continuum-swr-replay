#!/usr/bin/env python
"""Held-out state labels: decode with one tetrode half, score with the other.

Task 12 (``run_state_evidence.py``) scores the all-tetrode decode's labels, so
the scoring half C helped make every label it judges. Here each label comes
from a decode that saw one half S alone, and C alone scores it. The method is in
``continuum_swr_replay.analysis.hippocampus.state_evidence.half_decode``.

Two steps, two subcommands:

``decode`` (a GPU, or the CPU for a smoke test). Per epoch, the first two of
Task 12's tetrode partitions, each in both directions: four decodes, ``p0d0``,
``p0d1``, ``p1d0`` and ``p1d1``. Each one (``decode_split``):

1. fits the encoding model on S's tetrodes alone, at the canonical decode's own
   training bins (the inputs its fitted detector keeps), with no weights: the
   canonical encoding is the plain fit, because the library's clusterless KDE
   ignores EM's weights, and the record checks that the half encoding is the
   canonical one restricted to S (``encoding_subset_check``);
2. keeps the canonical fit's discrete transitions (12 numbers) and initial
   conditions fixed, and runs no EM;
3. decodes S's spikes, with the canonical decode's missing bins (its artifact
   mask) set missing.

The canonical decode's masks are built as the decoder built them
(``decoding_common.hpc_decode_masks``, from the configured SWR source) and
checked against what it used (``canonical_masks``): its saved ``decode_masks``
must name the configured runs and the same counts, and the training bins must
be the ones its detector kept. A decode made under other masks (one from before
R11, say) is refused. Its ``data_over_time`` is read through the SWR source
(``swr_source.with_swr_source``), so its non-local events are the analysed
ones, and each half decode's events are held to the same analysed-time rule
(``analysed_labels``) before the two are compared.

``--all-tetrodes`` decodes with every tetrode instead (split ``all``), to check
that the pipeline reproduces the canonical decode. Each decode writes
``decodes/{epoch}_{split}.nc`` (the four state probabilities, float32, on the
decode's time axis, with S, C, the fixed transitions and the canonical detector
file's stamp in its attributes) and ``decodes/{epoch}_{split}.json``, its
record: the half encoding's check, the agreement of its states and non-local
events with the canonical decode's, the device and the run time. A decode
written before is kept only when it is complete (or as long as ``--max-bins``
asks), was made from the same canonical detector file (name, size and
``mtime_ns``) and by the same code commit, with neither tree dirty; otherwise
it is redone and the log says why (``reuse_reason``). ``--overwrite`` redoes
every one.

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
  four (S, C) runs. (Task 12's own summary averages 20 runs per window, so its
  window values are less spread.)

**The two sources share their scoring runs, not their bouts.** The same splits
and the same C score both, but each source's labels select their own bouts,
matched windows and calibration bins. A difference between the sources is
therefore partly which bouts are scored (for the half labels, those S alone is
confident about), not only how the same bouts score.
``half_vs_all_agreement.csv`` gives how far the selections differ.

``split`` names the run. It writes ``evidence/per_epoch/{epoch}_windows.parquet``
and ``{epoch}_validation.json``; then, per label source, Task 12's summary
tables and figures (``evidence/half/``, ``evidence/all/``), with each epoch's
value pooling its four splits' windows; ``evidence/half_vs_all_bouts.csv``
(the primary bout statistics side by side); ``evidence/half_vs_all_agreement.csv``
(per epoch, split and source: the bouts found per state and regime, the
non-local events, the events shared with the other source and their bins'
Jaccard index); ``evidence/manifest.json`` (with the same agreement) and a
``README.md`` (with it as a table).

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

The decodes' canonical detector files may move (from the staging folder into
the canonical one, say) before ``score``: the stamp is the file's name, size
and ``mtime_ns``, which a move keeps.
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
from dataclasses import dataclass
from datetime import date
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

# the sibling Task 12 runner, imported only by ``score``: importing it holds JAX
# to the CPU, and the decodes may run on a GPU. Nothing imported at this level
# loads JAX, so ``configure_devices`` can still choose its device.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from continuum_swr_replay.analysis.hippocampus.state_evidence import (  # noqa: E402
    HALVES_SEED,
    REGIMES,
    SPLIT_COLUMN,
    STATES,
    code_version,
    half_splits,
    json_ready,
    parse_epoch_identifier,
    write_json,
)
from continuum_swr_replay.analysis.hippocampus.swr_source import (  # noqa: E402
    with_swr_source,
)
from continuum_swr_replay.decoding_common import (  # noqa: E402
    HpcDecodeMasks,
    hpc_artifact_intervals,
    hpc_decode_masks,
    provenance_record,
    raise_open_file_limit,
    require_saved_decode_masks,
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
HALF_SPLIT_NAMES = tuple(split.name for split in half_splits(2, seed=HALVES_SEED))
"""``p0d0``, ``p0d1``, ``p1d0``, ``p1d1``: ``half_splits``'s names, in its order
(they depend on the number of partitions only)."""
LABEL_SOURCES = {
    "half": "a decode that saw half S alone; scored by C",
    "all": "the canonical all-tetrode decode; scored by the same (S, C) runs, "
    "which matches the scoring runs, not the bouts",
}
SELECTION_CAVEAT = (
    "The two label sources share their scoring runs (the same splits, the same "
    "scoring half C), not their bouts: each source's labels select their own "
    "bouts, matched windows and calibration bins. A difference between them is "
    "therefore partly which bouts are scored (for the half labels, those S alone "
    "is confident about), not only how the same bouts score. "
    "half_vs_all_agreement.csv gives, per epoch, split and source, the bouts found "
    "per state and regime, the non-local events, the events shared with the other "
    "source and the Jaccard index of their bins."
)
COUNT_CAVEAT = (
    "Counts are per split: for source all, n counts bout x split, since each "
    "canonical bout appears once per split, scored by that split's C; for source "
    "half, n sums the four decodes' own bouts. The calibration verdicts' window "
    "gate likewise counts window x split. The inference is unchanged: each epoch "
    "enters as one value, pooling its splits' windows."
)
_SHORT_STATES = {
    "Local": "Local",
    "No-Spike": "No-Spike",
    "Non-Local Continuous": "NL-Cont",
    "Non-Local Fragmented": "NL-Frag",
}


def bout_column(state: str, regime: str) -> str:
    """The agreement table's column of a state x regime's bouts found."""
    slug = _SHORT_STATES[state].lower().replace("-", "_")
    return f"bouts_{slug}_{regime}"


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
    """Where JAX computes: ``"cpu"`` or ``"gpu:{CUDA_VISIBLE_DEVICES}"``.

    It confirms only that JAX's backend is or is not the CPU; the index it
    reports is read back from ``CUDA_VISIBLE_DEVICES``, not from the card. What
    ties that index to the physical card nvidia-smi checked is
    ``CUDA_DEVICE_ORDER=PCI_BUS_ID``, set by :func:`configure_devices`, with
    which CUDA numbers the cards as nvidia-smi does.
    """
    import jax

    if jax.default_backend() == "cpu":
        return "cpu"
    return f"gpu:{os.environ.get('CUDA_VISIBLE_DEVICES', '?')}"


# --- decode -------------------------------------------------------------------------------
class Job(NamedTuple):
    """One decode: its split's name, partition and direction, S and C."""

    name: str
    partition: int | None
    direction: int | None
    selecting: np.ndarray
    scoring: np.ndarray


def decode_jobs(n_tetrodes: int, *, all_tetrodes: bool = False) -> list[Job]:
    """The four half splits' decodes, or the one with every tetrode."""
    if all_tetrodes:
        every = np.arange(n_tetrodes)
        return [Job(ALL_TETRODES, None, None, every, every[:0])]
    return [
        Job(s.name, s.partition, s.direction, s.selecting, s.scoring)
        for s in half_splits(n_tetrodes, seed=HALVES_SEED)
    ]


def decode_paths(epoch_id: str, split: str, output_dir: Path) -> dict[str, Path]:
    """One decode's probabilities (``decode``) and record (``record``)."""
    stem = Path(output_dir) / "decodes" / f"{epoch_id}_{split}"
    return {"decode": stem.with_suffix(".nc"), "record": stem.with_suffix(".json")}


def canonical_detector_stamp(epoch_id: str, decode_dir: Path) -> dict:
    """The canonical detector file's name, size and ``mtime_ns``.

    Not its folder: moving the file (from a staging folder into the canonical
    one, say) keeps its size and modification time, so its decodes stay valid;
    an edit changes them.
    """
    path = Path(decode_dir) / f"{epoch_id}_detector.pkl"
    info = path.stat()
    return {"name": path.name, "size": info.st_size, "mtime_ns": info.st_mtime_ns}


def reuse_reason(
    epoch_id: str,
    split: str,
    output_dir: Path,
    n_bins: int | None,
    *,
    stamp: dict,
    code: dict,
) -> str | None:
    """Why a decode written before must be redone, or None to keep it.

    It is kept only when its record is "ok" beside its decode file, it covers
    the bins asked for (every bin when ``n_bins`` is None, so a ``--max-bins``
    smoke test never stands in for a whole epoch, nor the reverse), it was made
    from the same canonical detector file (``stamp``) and by the same code
    commit, and neither tree had uncommitted changes (then the commit does not
    identify the code).

    Parameters
    ----------
    epoch_id, split : str
    output_dir : Path
    n_bins : int or None
        The bins asked for (``--max-bins``).
    stamp : dict
        :func:`canonical_detector_stamp` now.
    code : dict
        :func:`code_version` now.
    """
    paths = decode_paths(epoch_id, split, output_dir)
    if not all(path.exists() for path in paths.values()):
        return "not written before"
    record = json.loads(paths["record"].read_text())
    if record.get("status") != "ok":
        return "the earlier decode failed"
    if n_bins is None and not record.get("complete"):
        return f"the earlier decode covers only the first {record.get('n_bins')} bins"
    if n_bins is not None and (
        record.get("complete") or record.get("n_bins") != n_bins
    ):
        return f"the earlier decode covers {record.get('n_bins')} bins, not {n_bins}"
    if record.get("canonical_detector") != stamp:
        return "the canonical detector file changed (name, size or mtime_ns)"
    earlier = record.get("code") or {}
    if earlier.get("commit") != code["commit"]:
        return f"code commit {str(earlier.get('commit'))[:9]} -> {code['commit'][:9]}"
    if code["dirty"] or earlier.get("dirty", True):
        return "uncommitted changes, so the commit does not identify the code"
    return None


@dataclass
class CanonicalEpoch:
    """What every decode of one epoch reads from its canonical decode.

    Attributes
    ----------
    epoch_id : str
    decode_dir : Path
    detector : NonLocalClusterlessDetector
        The canonical fitted detector.
    time, speed : ndarray, shape (n_time,)
        The decode's time axis (s) and the head speed (cm/s).
    probabilities : ndarray, shape (n_time, 4)
        The canonical state probabilities, float32 as the decode wrote them.
    labels : ndarray of int, shape (n_time,)
        The canonical ``non_local_event`` labels, read through the SWR source:
        the events the analysed-time rule keeps.
    labels_reproduced : bool
        Whether :func:`non_local_event_labels`, held to the same rule
        (:func:`analysed_labels`), remakes them from the probabilities exactly.
    stamp : dict
        :func:`canonical_detector_stamp`, taken before the detector is read.
    n_tetrodes : int
        The tetrodes the canonical decode used.
    is_missing : ndarray of bool, shape (n_time,)
        The canonical decode's missing bins (:func:`canonical_masks`).
    rule_frame : pd.DataFrame
        The frame's session columns, head speed and ``MUA_event`` as the
        decoder saved them, time-indexed: what :func:`analysed_labels` holds a
        half decode's events to the rule with.
    """

    epoch_id: str
    decode_dir: Path
    detector: Any
    time: np.ndarray
    speed: np.ndarray
    probabilities: np.ndarray
    labels: np.ndarray
    labels_reproduced: bool
    stamp: dict
    n_tetrodes: int
    is_missing: np.ndarray
    rule_frame: pd.DataFrame


RULE_FRAME_COLUMNS = ("animal", "day", "epoch", "head_speed", "MUA_event")
"""The columns of a decode frame the analysed-time rule reads, besides the
labels it holds to it (:func:`analysed_labels`)."""


def canonical_masks(
    epoch_id: str, decode_dir: Path, time: np.ndarray, is_training: np.ndarray
) -> HpcDecodeMasks:
    """The canonical decode's masks, built as the decoder built them, and checked.

    ``hpc_decode_masks`` on the decode's time axis, from the configured SWR
    source, with the training type the decode recorded.

    Parameters
    ----------
    epoch_id : str
    decode_dir : Path
        Holds ``{epoch_id}_results.nc``, whose ``decode_masks`` the decoder
        stamped.
    time : ndarray, shape (n_time,)
        The decode's time axis (s).
    is_training : ndarray of bool, shape (n_time,)
        The training bins the canonical detector kept (``decode_inputs``).

    Returns
    -------
    HpcDecodeMasks

    Raises
    ------
    ValueError
        The decode has no ``decode_masks`` or names other runs
        (``require_saved_decode_masks``), its counts of bins, missing bins or
        bins left out of encoding differ from the masks', or the masks'
        training bins are not the ones the decode used: it was made under
        other masks (before R11, say), so a half decode would not share its
        encoding.
    """
    import xarray as xr

    nwb_file_name, epoch_name, _ = parse_epoch_identifier(epoch_id)
    path = Path(decode_dir) / f"{epoch_id}_results.nc"
    with xr.open_dataset(path, engine="h5netcdf") as results:
        attrs = dict(results.attrs)
    require_saved_decode_masks(attrs, str(path))
    saved = json.loads(attrs["decode_masks"])
    masks = hpc_decode_masks(
        nwb_file_name,
        epoch_name,
        pd.DataFrame(index=pd.Index(time, name="time")),
        hpc_artifact_intervals(nwb_file_name, epoch_name),
        saved["training_type"],
    )
    differ = {
        name: {"saved": saved.get(name), "now": masks.record[name]}
        for name in ("n_bins", "n_missing_bins", "n_encoding_removed_bins")
        if saved.get(name) != masks.record[name]
    }
    if differ:
        raise ValueError(
            f"{path.name}'s decode_masks are not the masks built now from the "
            f"configured SWR source: {differ}"
        )
    if not np.array_equal(masks.is_training, np.asarray(is_training)):
        raise ValueError(
            f"{epoch_id}: the canonical decode's training bins are not the ones "
            "built now from the configured SWR source (hpc_decode_masks); "
            "it was decoded under other masks"
        )
    return masks


def analysed_labels(rule_frame: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
    """Non-local event labels held to the one analysed-time rule.

    ``swr_source.with_swr_source`` on the decode frame's own columns with
    `labels` as its ``non_local_event``: an event the rule drops (excluded
    time, or overlapping an SWR or MUA event dropped for it) is zeroed, as
    reading the canonical decode's frame through the source zeroes its own.

    Parameters
    ----------
    rule_frame : pd.DataFrame
        :attr:`CanonicalEpoch.rule_frame`, or its first rows.
    labels : ndarray of int, shape (n_time,)
        On `rule_frame`'s rows (:func:`non_local_event_labels`).

    Returns
    -------
    ndarray of int, shape (n_time,)
    """
    frame = rule_frame.assign(non_local_event=np.asarray(labels, dtype=np.int64))
    return with_swr_source(frame)["non_local_event"].to_numpy(dtype=np.int64)


def load_canonical(epoch_id: str, decode_dir: Path) -> CanonicalEpoch:
    """The canonical decode's detector, masks, states, events and speed for one epoch.

    The frame is read through the SWR source (``with_swr_source``), and its
    masks are :func:`canonical_masks`.

    Raises
    ------
    ValueError
        The detector's decode inputs are not on ``data_over_time``'s axis, or
        :func:`canonical_masks` refuses the decode.
    SwrSourceError
        The SWR source cannot serve the epoch.
    """
    from continuum_swr_replay.analysis.hippocampus.state_evidence import (
        decode_inputs,
        load_detector,
        non_local_event_labels,
    )

    stamp = canonical_detector_stamp(epoch_id, decode_dir)
    detector = load_detector(epoch_id, decode_dir)
    inputs = decode_inputs(detector)
    saved = pd.read_pickle(Path(decode_dir) / f"{epoch_id}_80_data_over_time.pkl")
    time = saved.index.to_numpy(dtype=float)
    if not np.array_equal(time, np.asarray(inputs["position_time"], dtype=float)):
        raise ValueError(
            "the detector's decode inputs are not on data_over_time's axis"
        )
    masks = canonical_masks(epoch_id, decode_dir, time, inputs["is_training"])
    rule_frame = saved[[c for c in RULE_FRAME_COLUMNS if c in saved.columns]].copy()
    dot = with_swr_source(saved)
    del saved
    columns = [f"{state.replace(' ', '_')}_prob" for state in STATES]
    probabilities = dot[columns].to_numpy()
    labels = dot["non_local_event"].to_numpy(dtype=np.int64)
    speed = dot["head_speed"].to_numpy(dtype=float)
    del dot
    return CanonicalEpoch(
        epoch_id=epoch_id,
        decode_dir=Path(decode_dir),
        detector=detector,
        time=time,
        speed=speed,
        probabilities=probabilities,
        labels=labels,
        labels_reproduced=bool(
            np.array_equal(
                analysed_labels(
                    rule_frame, non_local_event_labels(probabilities, speed, time)
                ),
                labels,
            )
        ),
        stamp=stamp,
        n_tetrodes=len(inputs["spike_times"]),
        is_missing=masks.is_missing,
        rule_frame=rule_frame,
    )


def decode_split(
    canonical: CanonicalEpoch,
    job: Job,
    *,
    output_dir: Path,
    provenance: dict,
    n_bins: int | None = None,
) -> dict:
    """Decode one split, check it, write its decode and record; return the record.

    Parameters
    ----------
    canonical : CanonicalEpoch
    job : Job
        Its tetrodes S decode.
    output_dir : Path
        The decode goes to ``decodes/{epoch}_{split}.nc``, the record beside it.
    provenance : dict
        ``provenance_record`` of the code.
    n_bins : int, optional
        Decode only the first bins.

    Returns
    -------
    dict
        ``status`` "ok", or "failed" with ``error`` and ``traceback`` (then no
        decode file is left, so a rerun redoes it). An "ok" record holds the
        half encoding's check (``encoding_subset_check``; it fails the split
        unless the half encoding is the canonical one restricted to S), the
        agreement of the states and events with the canonical decode
        (``label_agreement``), the fixed transitions, the device and timings.
    """
    import xarray as xr

    from continuum_swr_replay.analysis.hippocampus.state_evidence import (
        ENCODING_KEY,
        encoding_subset_check,
        half_tetrode_decode,
        label_agreement,
        non_local_event_labels,
    )

    paths = decode_paths(canonical.epoch_id, job.name, output_dir)
    record: dict[str, Any] = {
        "epoch_id": canonical.epoch_id,
        "split": job.name,
        "partition": job.partition,
        "direction": job.direction,
        "selecting_tetrodes": np.asarray(job.selecting).tolist(),
        "scoring_tetrodes": np.asarray(job.scoring).tolist(),
        "decode_dir": str(canonical.decode_dir),
        "canonical_detector": canonical.stamp,
        "code": code_version(provenance),
    }
    began = clock.perf_counter()
    try:
        results, half = half_tetrode_decode(
            canonical.detector,
            job.selecting,
            is_missing=canonical.is_missing,
            n_bins=n_bins,
        )
        probabilities = results.acausal_state_probabilities.values
        decoded = probabilities.shape[0]
        time = canonical.time[:decoded]
        if list(results.states.values) != list(STATES) or not np.array_equal(
            results.time.values, time
        ):
            raise ValueError("the decode's states or time axis are not the canonical")
        record["device"] = current_device()
        record["decode_s"] = clock.perf_counter() - began
        check = encoding_subset_check(
            half.encoding_model_[ENCODING_KEY],
            canonical.detector.encoding_model_[ENCODING_KEY],
            job.selecting,
        )
        record["half_encoding"] = check
        if not check["same_as_canonical_subset"]:
            raise ValueError(
                "the half encoding is not the canonical encoding restricted to "
                f"its tetrodes: {check}"
            )
        labels = analysed_labels(
            canonical.rule_frame.iloc[:decoded],
            non_local_event_labels(probabilities, canonical.speed[:decoded], time),
        )
        record["agreement"] = label_agreement(
            canonical.probabilities[:decoded],
            canonical.labels[:decoded],
            probabilities,
            labels,
        )
        record["agreement"]["reference_labels_reproduced"] = canonical.labels_reproduced
        record["n_bins"] = int(decoded)
        record["n_missing_bins"] = int(canonical.is_missing[:decoded].sum())
        record["complete"] = bool(decoded == canonical.time.size)
        transitions = np.asarray(canonical.detector.discrete_state_transitions_)
        record["fixed"] = {
            "discrete_state_transitions": transitions.tolist(),
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
            coords={"time": time, "states": list(STATES)},
            attrs={
                "epoch_id": canonical.epoch_id,
                "split": job.name,
                "selecting_tetrodes": json.dumps(record["selecting_tetrodes"]),
                "scoring_tetrodes": json.dumps(record["scoring_tetrodes"]),
                "discrete_state_transitions": json.dumps(transitions.tolist()),
                "canonical_detector": json.dumps(canonical.stamp),
                "marginal_log_likelihood": record["marginal_log_likelihood"],
                "n_bins": int(decoded),
                "complete": int(record["complete"]),
                "produced_by": PRODUCED_BY,
                "provenance": json.dumps(provenance, sort_keys=True),
            },
        )
        paths["decode"].parent.mkdir(parents=True, exist_ok=True)
        write_atomically(paths["decode"], partial(dataset.to_netcdf, engine="h5netcdf"))
        record["status"] = "ok"
    except Exception as error:  # noqa: BLE001 - recorded; the next split runs
        paths["decode"].unlink(missing_ok=True)
        record.update(
            status="failed", error=repr(error), traceback=traceback.format_exc()
        )
    record["wall_clock_s"] = clock.perf_counter() - began
    ready: dict = json_ready(record)
    write_json(paths["record"], ready)
    return ready


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
        One record per split (:func:`decode_split`), or ``status`` "exists" for
        a decode kept from an earlier run (:func:`reuse_reason`; not rewritten).
    """
    names = [ALL_TETRODES] if all_tetrodes else list(HALF_SPLIT_NAMES)
    stamp = canonical_detector_stamp(epoch_id, decode_dir)
    provenance = provenance_record(ROOT_DIR)
    code = code_version(provenance)
    reasons = {
        name: "--overwrite"
        if overwrite
        else reuse_reason(epoch_id, name, output_dir, n_bins, stamp=stamp, code=code)
        for name in names
    }
    for name, reason in reasons.items():
        log(f"{epoch_id} {name}: " + ("written before; kept" if reason is None
                                       else f"to decode ({reason})"))  # fmt: skip
    if all(reason is None for reason in reasons.values()):
        return [
            {"epoch_id": epoch_id, "split": name, "status": "exists"} for name in names
        ]

    started = clock.perf_counter()
    canonical = load_canonical(epoch_id, decode_dir)
    log(f"{epoch_id}: inputs read in {clock.perf_counter() - started:.0f} s")
    jobs = decode_jobs(canonical.n_tetrodes, all_tetrodes=all_tetrodes)
    if [job.name for job in jobs] != names:
        raise ValueError(f"the splits are {[job.name for job in jobs]}, not {names}")
    records = []
    for job in jobs:
        if reasons[job.name] is None:
            records.append(
                {"epoch_id": epoch_id, "split": job.name, "status": "exists"}
            )
            continue
        record = decode_split(
            canonical, job, output_dir=output_dir, provenance=provenance, n_bins=n_bins
        )
        log(
            f"{epoch_id} {job.name}: {record['status']} in "
            f"{record['wall_clock_s']:.0f} s"
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
    labels are scored by the same four (S, C) runs (the same runs, not the
    same bouts: each source's labels select their own).

    Raises
    ------
    ValueError
        A half decode is missing, incomplete (``--max-bins``), not on the first
        two partitions' tetrodes, or made from another canonical detector file
        (name, size or ``mtime_ns``) than the one in ``decode_dir``.
    """
    import xarray as xr

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


def agreement_table(output_dir: Path, windows: pd.DataFrame) -> pd.DataFrame:
    """How far the two label sources' selections differ, per epoch, split and source.

    Parameters
    ----------
    output_dir : Path
        Holds the decodes' records (``decodes/{epoch}_{split}.json``), whose
        ``agreement`` compares each half decode with the canonical decode.
    windows : pd.DataFrame
        The scored per-window tables (``label_source``, ``split``, ...).

    Returns
    -------
    pd.DataFrame
        One row per epoch x split x ``label_source``:

        - ``bouts_{state}_{regime}``: the bouts its labels found in each state
          and regime (``n_found``, before the cap of 1500 scored per cell);
        - ``events``: its non-local events; ``events_shared``: those overlapping
          an event of the other source in at least one bin;
        - ``events_identical`` (the same first and last bin) and
          ``event_bin_jaccard`` (bins in both sources' events over bins in
          either's), the same for both sources.
    """
    bouts = windows[windows.kind == "bout"]
    found = (
        bouts.groupby(["epoch_id", SPLIT_COLUMN, "label_source", "state", "regime"])
        .n_found.first()
        .to_dict()
    )
    rows = []
    pairs = windows[["epoch_id", SPLIT_COLUMN]].drop_duplicates()
    for epoch_id, split in sorted(pairs.itertuples(index=False, name=None)):
        record = json.loads(
            decode_paths(epoch_id, split, output_dir)["record"].read_text()
        )
        agreement = record["agreement"]
        for source in LABEL_SOURCES:
            mine, shared = (
                ("events_other", "other_events_shared")
                if source == "half"
                else ("events_reference", "reference_events_shared")
            )
            row: dict[str, Any] = {
                "epoch_id": epoch_id,
                SPLIT_COLUMN: split,
                "label_source": source,
                "events": agreement[mine],
                "events_shared": agreement[shared],
                "events_identical": agreement["events_identical"],
                "event_bin_jaccard": agreement["event_bin_jaccard"],
            }
            for state in STATES:
                for regime in REGIMES:
                    row[bout_column(state, regime)] = int(
                        found.get((epoch_id, split, source, state, regime), 0)
                    )
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_label_sources(output_dir: Path, epochs: list[str]) -> pd.DataFrame:
    """Per label source, Task 12's summary tables and figures; then side by side.

    Each epoch's value pools its four splits' windows (Task 12's epoch summary),
    and bouts pair with their matched windows within their split. Counts are per
    split: for source "all", ``n`` counts bout x split, since each canonical bout
    appears once per split, scored by that split's C; for source "half", ``n``
    sums the four decodes' own bouts. The calibration verdicts' window gate
    likewise counts window x split. The inference is unchanged: the epoch is the
    unit.

    The two sources share their scoring runs, not their bouts (each source's
    labels select their own), so the side-by-side table compares partly
    different bouts; :func:`agreement_table` says how different, and is written
    beside it.

    Returns
    -------
    pd.DataFrame
        :func:`agreement_table` of the epochs (empty when there are none).
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
        return pd.DataFrame()
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
    agreement = agreement_table(output_dir, windows)
    task12.write_csv(agreement, evidence_dir / "half_vs_all_agreement.csv")
    print(side_by_side.to_string(index=False), flush=True)
    return agreement


def agreement_markdown(agreement: pd.DataFrame) -> list[str]:
    """:func:`agreement_table` as a Markdown table, one row per epoch x split."""
    if agreement.empty:
        return ["No epoch was scored, so there is no agreement table."]
    cells = [(state, regime) for regime in REGIMES for state in STATES]
    header = (
        "| epoch | split | events all / half | shared all / half | identical "
        "| Jaccard | "
        + " | ".join(f"{_SHORT_STATES[s]} {r} all / half" for s, r in cells)
        + " |"
    )
    lines = [header, "|" + "---|" * (6 + len(cells))]
    for (epoch_id, split), rows in agreement.groupby(["epoch_id", SPLIT_COLUMN]):
        by_source = rows.set_index("label_source")
        every, half = by_source.loc["all"], by_source.loc["half"]
        jaccard = every["event_bin_jaccard"]
        lines.append(
            f"| {epoch_id} | {split} | {every['events']} / {half['events']} | "
            f"{every['events_shared']} / {half['events_shared']} | "
            f"{every['events_identical']} | "
            + ("-" if pd.isna(jaccard) else f"{jaccard:.2f}")
            + " | "
            + " | ".join(
                f"{every[bout_column(s, r)]} / {half[bout_column(s, r)]}"
                for s, r in cells
            )
            + " |"
        )
    return lines


def score_manifest(
    records: dict[str, dict],
    provenance: dict,
    decode_dir: Path,
    agreement: pd.DataFrame,
) -> dict:
    """The score's manifest: provenance, settings, records and the agreement."""
    from scripts.qa import run_state_evidence as task12

    by_epoch = (
        {
            epoch_id: rows.drop(columns="epoch_id").to_dict(orient="records")
            for epoch_id, rows in agreement.groupby("epoch_id")
        }
        if not agreement.empty
        else {}
    )
    manifest: dict = json_ready(
        {
            "produced_by": PRODUCED_BY,
            "provenance": provenance,
            "settings": task12.run_settings(),
            "label_sources": LABEL_SOURCES,
            "selection_caveat": SELECTION_CAVEAT,
            "count_caveat": COUNT_CAVEAT,
            "decode_dir": str(decode_dir),
            "agreement": by_epoch,
            "epochs": {k: records[k] for k in sorted(records)},
        }
    )
    return manifest


def readme(
    records: dict[str, dict],
    provenance: dict,
    decode_dir: Path,
    agreement: pd.DataFrame,
) -> str:
    """The evidence folder's README: what was run, what is held out, the agreement."""
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
        "four (S, C) runs, one per window. Task 12's own summary averages 20 runs "
        "per window, so its window values are less spread and its medians are not "
        "directly comparable with these.",
        f"- **Not the same bouts.** {SELECTION_CAVEAT}",
        f"- {COUNT_CAVEAT}",
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
        "## How far the two sources' selections differ",
        "",
        "Per epoch and split: the non-local events of each source (all = the "
        "canonical decode, half = the split's half decode), those shared with the "
        "other source, those identical in both, the Jaccard index of their bins, "
        "and the bouts each source found per state and regime (before the cap of "
        "1500 scored per cell). Also in `half_vs_all_agreement.csv` and "
        "`manifest.json`.",
        "",
        *agreement_markdown(agreement),
        "",
        "## Files",
        "",
        "- `per_epoch/{epoch}_windows.parquet`, `per_epoch/{epoch}_validation.json`",
        "- `half/`, `all/`: `bout_stats.csv`, `calibration_stats.csv`, "
        "`trend_stats.csv`, `rate_calibration.csv`, `silence_stats.csv` and "
        "the three figures",
        "- `half_vs_all_bouts.csv`: the primary bout statistics, side by side",
        "- `half_vs_all_agreement.csv`: the selections' agreement, above",
        "- `manifest.json`: provenance, every epoch's record and the agreement",
        "",
    ]
    return "\n".join(lines)


# --- command line -------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """The command line: the ``decode`` and ``score`` subcommands.

    Returns
    -------
    argparse.ArgumentParser
        Both take ``--epochs`` (default the five pilot epochs), ``--decode-dir``
        and ``--output-dir``; ``decode`` also ``--cuda_visible_devices``,
        ``--all-tetrodes``, ``--max-bins`` and ``--overwrite``; ``score``
        ``--n-workers``.
    """
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
    # before score's workers start: they inherit this limit. Their epochs close
    # the NWB files Spyglass opened (Task 12's _run_epoch_logged); decode reads
    # no NWB file through Spyglass.
    old_soft, new_soft = raise_open_file_limit()
    if args.command == "decode":
        device = configure_devices(args.cuda_visible_devices)  # before JAX loads
        if current_device() != device:
            sys.exit(f"JAX computes on {current_device()}, not {device}")
        logger = setup_logger(
            name_logfile="half_tetrode_decode",
            path_logfile=output_dir / "logs" / f"decode_{os.getpid()}.log",
        )
        logger.info("open-file soft limit %d -> %d", old_soft, new_soft)
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

    print(f"open-file soft limit {old_soft} -> {new_soft}", flush=True)
    provenance = provenance_record(ROOT_DIR)
    evidence_dir = output_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    kwargs = {
        "decode_dir": decode_dir,
        "output_dir": evidence_dir,
        "code": code_version(provenance),
        "label_sets": partial(
            half_label_sets, output_dir=output_dir, decode_dir=decode_dir
        ),
    }
    records: dict[str, dict] = {}
    for epoch_id, record in task12.run_epochs(args.epochs, kwargs, args.n_workers):
        records[epoch_id] = record
        print(f"{epoch_id}: {record['status']}", flush=True)
    ok = [k for k in sorted(records) if records[k]["status"] == "ok"]
    agreement = summarize_label_sources(output_dir, ok)
    write_json(
        evidence_dir / "manifest.json",
        score_manifest(records, provenance, decode_dir, agreement),
    )
    (evidence_dir / "README.md").write_text(
        readme(records, provenance, decode_dir, agreement)
    )
    failed = [k for k, r in records.items() if r["status"] != "ok"]
    if failed:
        sys.exit(f"{len(failed)} epochs not scored: {', '.join(sorted(failed))}")


if __name__ == "__main__":
    main()
