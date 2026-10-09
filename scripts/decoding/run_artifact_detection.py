#!/usr/bin/env python
"""Detect LFP and spike artifacts in every epoch's CA1 data, on CPU.

The detector (``analysis/hippocampus/artifacts``) masks a period only on an
artifact signature, never on power alone: S (sustained ripple-band noise on a
fixed subset of CA1 channels), T-ref (a jump on the reference electrode),
T-ca1 (a persistent step on several CA1 channels) and T-spk (a synchronous line
of out-of-distribution spikes). Design note:
``docs/design/2026-10-09-artifact-detection.md``.

Per epoch, read only:

1. **LFP.** ``get_ripple_band_inputs``: the raw referenced CA1 LFP at the
   stored band's timestamps, and the reference electrodes, which are the LFP's
   columns that are not band electrodes (checked: the LFP holds exactly the
   band's electrodes and their references, in id order). A sample where the
   stored band, the CA1 LFP or a reference is not finite is not read.
2. **Span.** The LFP rows inside ``get_epoch_position_info``'s position span,
   as the SWR runner keeps them. The epoch's span is the first and last LFP
   sample kept.
3. **Marks.** ``get_hpc_marks``, after the project's coincidence filter; the
   spikes inside the span.
4. **Detection.** ``detect_artifacts`` returns the epoch's unpadded cores and
   their evidence. No decode is read: the marks' out-of-distribution reference
   is the epoch's own marks outside the LFP signatures.

Each epoch writes, under ``Processed-Data/artifact_intervals/<run_id>/``,
``{epoch_id}_artifact_intervals.parquet``: its cores (``io.CORE_COLUMNS``), no
rows when it has no artifact. A reader pads the cores into the one artifact
mask (``masks.PADDING_S``), as ``swr_source.artifact_intervals`` does. A failed
epoch keeps no file.

``manifest.json`` beside them holds the run's settings (every constant and
the padding table) and one record per epoch: its status, span, input row
counts, the constants, the code commit, the cores' counts and the mask's
seconds, input hashes, and the file's path and sha256. Only this process
writes it, atomically, after each epoch. A rerun under the same run id skips
the epochs already ``ok`` unless ``--overwrite`` is given, and refuses
settings other than the manifest's. A run holds an exclusive lock on its
folder's ``.lock``. The run id is the code commit's first 12 characters unless
``--run-id`` is given.

The epochs are the epoch register's analysed ones; with ``--include-pending``
also its pending ones, and with ``--pending-only`` its pending ones alone;
narrowed by ``--animal``, ``--date`` and ``--epoch``. They run in parallel on a
dask ``LocalCluster`` of ``--n-workers`` processes, one epoch per worker at a
time, each logging to ``logs/{epoch_id}_artifacts.log`` under the working
directory, and closing the NWB files Spyglass opened when its epoch ends. An
epoch that fails is recorded with its error and the run goes on; the run then
exits non-zero.

Usage:
    python scripts/decoding/run_artifact_detection.py --animal chimi \\
        --date 20200225 --epoch 02_r1 --n-workers 1 --output-root <scratch>
    python scripts/decoding/run_artifact_detection.py --include-pending --n-workers 8
"""

import argparse
import functools
import json
import resource
import sys
import time as clock
import traceback
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
from dask.distributed import LocalCluster
from numpy.typing import NDArray

# Sibling-script import: run as a script, only this file's directory is on
# sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from continuum_swr_replay.analysis.hippocampus.artifacts import detect  # noqa: E402
from continuum_swr_replay.analysis.hippocampus.artifacts.detect import (  # noqa: E402
    detect_artifacts,
)
from continuum_swr_replay.analysis.hippocampus.artifacts.io import (  # noqa: E402
    cores_path,
    write_cores,
    write_manifest,
)
from continuum_swr_replay.analysis.hippocampus.artifacts.masks import (  # noqa: E402
    PADDING_S,
    artifact_intervals,
    masked_seconds,
)
from continuum_swr_replay.analysis.hippocampus.swr_source import (  # noqa: E402
    ARTIFACT_INTERVALS_ROOT,
)
from continuum_swr_replay.data_loaders.epoch_register import (  # noqa: E402
    analysed_epochs,
    pending_epochs,
)
from continuum_swr_replay.data_loaders.lfp import get_ripple_band_inputs  # noqa: E402
from continuum_swr_replay.data_loaders.position import (  # noqa: E402
    get_epoch_position_info,
)
from continuum_swr_replay.data_loaders.spikes import get_hpc_marks  # noqa: E402
from continuum_swr_replay.decoding_common import (  # noqa: E402
    filter_epoch_info,
    provenance_record,
    raise_open_file_limit,
    release_nwb_files,
    setup_logger,
)
from continuum_swr_replay.parameters import ROOT_DIR  # noqa: E402
from scripts.decoding.run_swr_detection import (  # noqa: E402
    RUN_ID_PATTERN,
    array_sha256,
    epoch_id,
    json_ready,
    run_epochs,
    run_lock,
)

OUTPUT_ROOT = ARTIFACT_INTERVALS_ROOT
"""Where the run folders go: where the SWR source reads them."""
PRODUCED_BY = "scripts/decoding/run_artifact_detection.py"
DESIGN_SPEC = "docs/design/2026-10-09-artifact-detection.md"


def detector_constants() -> dict[str, Any]:
    """Every detector constant and the padding table, as JSON types.

    Read when called, so a run records the values it ran with.
    """
    constants: dict[str, Any] = json_ready(
        {
            **detect.detector_constants(),
            "padding_s": PADDING_S,
        }
    )
    return constants


def run_settings() -> dict[str, Any]:
    """What every epoch of a run shares; a rerun under its run id must match."""
    settings: dict[str, Any] = json_ready(
        {
            "constants": detector_constants(),
            "inputs": {
                "lfp": "get_ripple_band_inputs: raw referenced CA1 LFP at the "
                "stored band's timestamps; reference electrodes = the LFP's "
                "non-band columns, raw (unreferenced)",
                "span": "LFP rows inside get_epoch_position_info's position span",
                "marks": "get_hpc_marks after the coincidence filter, spikes in "
                "the span",
            },
            "ood_reference": "per tetrode, the epoch's marks outside the S, "
            "T-ref and T-ca1 cores padded into the one artifact mask "
            "(masks.PADDING_S); the spike itself left out; Euclidean distance "
            "on the raw marks",
        }
    )
    return settings


def reference_lfp(
    inputs: dict[str, Any], first: int, stop: int
) -> tuple[NDArray[np.int64], NDArray]:
    """The reference electrodes' raw LFP at the band's samples ``first:stop``.

    The LFP's columns that are not band electrodes are the reference
    electrodes, raw (no reference subtracted), when the LFP holds exactly the
    band's electrodes and their references in electrode-id order; that is
    checked.

    Parameters
    ----------
    inputs : dict
        ``get_ripple_band_inputs``'s.
    first, stop : int
        The band samples kept (the span).

    Returns
    -------
    reference_ids : ndarray of int, shape (n_references,)
    reference : ndarray, shape (stop - first, n_references)
        In the LFP's stored dtype.

    Raises
    ------
    ValueError
        If the LFP's columns are not the band's electrodes and references in
        id order, so the reference columns cannot be told apart.
    """
    electrode_ids = np.asarray(inputs["electrode_ids"], dtype=np.int64)
    reference_ids = np.asarray(inputs["reference_ids"], dtype=np.int64)
    native = inputs["raw_referenced_native"]
    all_ids = np.unique(np.r_[electrode_ids, reference_ids[reference_ids >= 0]])
    if native.shape[1] != all_ids.size or not np.array_equal(
        np.searchsorted(all_ids, electrode_ids), inputs["lfp_band_elect_index"]
    ):
        raise ValueError(
            "The LFP's columns are not the band's electrodes and their reference "
            "electrodes in id order, so the reference columns cannot be told apart."
        )
    only_references = np.setdiff1d(all_ids, electrode_ids)
    rows = np.flatnonzero(np.isin(inputs["raw_timestamps"], inputs["time"]))[first:stop]
    columns = np.searchsorted(all_ids, only_references)
    return only_references, native[np.ix_(rows, columns)]


def _detect(
    nwb_file_name: str,
    epoch_name: str,
    *,
    fetch_inputs: Callable,
    fetch_position: Callable,
    fetch_marks: Callable,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """One epoch's record and cores."""
    inputs = fetch_inputs(nwb_file_name, epoch_name)
    position_time = fetch_position(nwb_file_name, epoch_name)[
        "position_info"
    ].index.to_numpy(dtype=np.float64)
    position_span = (float(position_time[0]), float(position_time[-1]))
    # The band's timestamps ascend (get_ripple_band_inputs checks them), so the
    # span is one slice, as the SWR runner takes it.
    first = int(np.searchsorted(inputs["time"], position_span[0], side="left"))
    stop = int(np.searchsorted(inputs["time"], position_span[1], side="right"))
    if stop - first < 2:
        raise ValueError(f"only {stop - first} LFP samples inside the position span")
    time = inputs["time"][first:stop]
    ca1 = inputs["raw_referenced"][first:stop]
    reference_ids, reference = reference_lfp(inputs, first, stop)
    band_observed = np.isfinite(inputs["ripple_band"][first:stop]).all(axis=1)
    spike_times, spike_marks = fetch_marks(nwb_file_name)

    cores = detect_artifacts(
        time,
        ca1,
        reference,
        spike_times,
        spike_marks,
        float(inputs["sampling_frequency"]),
        valid=band_observed,
    )
    counts = dict(cores.attrs["counts"])  # read at once: attrs are fragile
    span = (float(time[0]), float(time[-1]))
    record = {
        "status": "ok",
        "nwb_file_name": nwb_file_name,
        "epoch_name": epoch_name,
        "produced_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "provenance": provenance_record(ROOT_DIR)
        | {"numpy_version": np.__version__, "scipy_version": scipy.__version__},
        "span_s": list(span),
        "position_span_s": list(position_span),
        "sampling_frequency": float(inputs["sampling_frequency"]),
        "electrode_ids": inputs["electrode_ids"],
        "reference_electrode_ids": reference_ids,
        "lfp_units": inputs.get("lfp_units"),
        "lfp_conversion": inputs.get("lfp_conversion"),
        **counts,
        "n_tetrodes_in_file": len(spike_times),
        "n_cores": {kind: int((cores["kind"] == kind).sum()) for kind in ("S", "T")},
        "n_cores_by_signatures": {
            str(name): int(n)
            for name, n in cores["signatures"].value_counts().sort_index().items()
        },
        "masked_s": masked_seconds(artifact_intervals(cores, span)),
        "constants": detector_constants(),
        "input_sha256": {
            "time": array_sha256(time),
            "ca1_lfp": array_sha256(ca1),
            "reference_lfp": array_sha256(reference),
        },
    }
    return record, cores


def run_epoch(
    nwb_file_name: str,
    epoch_name: str,
    *,
    output_dir: Path,
    fetch_inputs: Callable = get_ripple_band_inputs,
    fetch_position: Callable = get_epoch_position_info,
    fetch_marks: Callable = get_hpc_marks,
    run_id: str | None = None,
) -> dict:
    """Detect one epoch's artifacts and write its cores file.

    Parameters
    ----------
    nwb_file_name, epoch_name : str
        The epoch, such as ``"chimi20200225_.nwb"`` and ``"02_r1"``.
    output_dir : Path
        The run's folder.
    fetch_inputs : Callable, optional
        ``(nwb_file_name, epoch_name) -> RippleBandInputs``.
    fetch_position : Callable, optional
        ``(nwb_file_name, epoch_name) -> PositionInfoDict``; its
        ``position_info``'s time index gives the span.
    fetch_marks : Callable, optional
        ``(nwb_file_name) -> (spike_times, spike_marks)``, per tetrode.
    run_id : str, optional
        Stamped on the file; default ``output_dir.name``.

    Returns
    -------
    dict
        The epoch's manifest record, in strict JSON types: ``status`` ``"ok"``
        with every field the module docstring lists, or ``"failed"`` with
        ``error`` and ``traceback`` (the epoch's file is then removed).
    """
    started = clock.perf_counter()
    identifier = epoch_id(nwb_file_name, epoch_name)
    path = cores_path(output_dir, identifier)
    try:
        record, cores = _detect(
            nwb_file_name,
            epoch_name,
            fetch_inputs=fetch_inputs,
            fetch_position=fetch_position,
            fetch_marks=fetch_marks,
        )
        stamp = {
            "produced_by": PRODUCED_BY,
            "run_id": output_dir.name if run_id is None else run_id,
            "epoch_id": identifier,
            "commit": record["provenance"]["commit"],
            "span_s": record["span_s"],
        }
        sha256 = write_cores(cores, path, stamp=stamp)
        record["outputs"] = {"cores": {"path": str(path), "sha256": sha256}}
    except Exception as error:  # noqa: BLE001 - recorded; the run goes on
        path.unlink(missing_ok=True)
        record = {
            "status": "failed",
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }
    record["wall_clock_s"] = clock.perf_counter() - started
    # the worker's peak resident set over its life so far (KiB on Linux)
    record["worker_peak_rss_gib"] = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    )
    ready: dict = json_ready(record)
    return ready


def summary(record: dict) -> str:
    """One epoch's outcome, as the logs and the console report it."""
    if "masked_s" not in record:
        return f"{record['status']}: {record.get('error')}"
    return (
        f"ok: cores {record['n_cores_by_signatures'] or 'none'}; masked "
        f"{record['masked_s']:.2f} s; "
        f"{record['n_spikes']} spikes on {record['n_tetrodes']} tetrodes, "
        f"{record['n_ood_spikes']} OOD; {record['wall_clock_s']:.0f} s, worker "
        f"peak RSS {record['worker_peak_rss_gib']:.1f} GiB"
    )


def _run_epoch_logged(
    nwb_file_name: str, epoch_name: str, run_epoch_kwargs: dict, log_directory: Path
) -> dict:
    """One dask task: ``run_epoch``, logged to ``logs/{epoch_id}_artifacts.log``.

    When the epoch ends, pass or fail, the NWB files Spyglass opened for it are
    closed (``release_nwb_files``): the workers run one task at a time.
    """
    identifier = epoch_id(nwb_file_name, epoch_name)
    log = setup_logger(
        name_logfile=f"{identifier}_artifacts",
        path_logfile=log_directory / f"{identifier}_artifacts.log",
    )
    log.info("%s: run %s", identifier, run_epoch_kwargs)
    try:
        record = run_epoch(nwb_file_name, epoch_name, **run_epoch_kwargs)
    finally:
        log.info("closed %d NWB files Spyglass held open", release_nwb_files())
    if record["status"] == "failed":
        log.error("%s\n%s", summary(record), record.get("traceback", ""))
    else:
        log.info("%s", summary(record))
    return record


def register_epochs(include_pending: bool, pending_only: bool) -> pd.DataFrame:
    """The register's epochs: analysed, analysed and pending, or pending alone."""
    if pending_only:
        return pending_epochs()
    return analysed_epochs(include_pending=include_pending)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--animal", help="Only this animal's epochs, such as chimi.")
    parser.add_argument(
        "--date", type=int, help="Only epochs recorded on this date, such as 20200225."
    )
    parser.add_argument("--epoch", help="Only this epoch, such as 02_r1.")
    which = parser.add_mutually_exclusive_group()
    which.add_argument(
        "--include-pending",
        action="store_true",
        help="Also run the epochs the epoch register holds pending a Spyglass "
        "populate (after their populates).",
    )
    which.add_argument(
        "--pending-only",
        action="store_true",
        help="Run only the register's pending epochs.",
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
        help="The run's folder under --output-root: letters, digits, '.', '_' "
        "and '-'. Defaults to the first 12 characters of git rev-parse HEAD.",
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args(argv)
    if args.run_id is not None and (
        not RUN_ID_PATTERN.fullmatch(args.run_id) or args.run_id in {".", ".."}
    ):
        parser.error(
            f"--run-id {args.run_id!r} must name one folder: letters, digits, '.', "
            "'_' and '-', and not '.' or '..'."
        )

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
        run(args, output_dir)


def run(args: argparse.Namespace, output_dir: Path) -> None:
    """``main``'s run, under the run folder's lock."""
    run_id = output_dir.name
    settings = run_settings()
    selected = register_epochs(args.include_pending, args.pending_only)
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

    def write() -> None:
        write_manifest(
            output_dir,
            {
                "produced_by": PRODUCED_BY,
                "run_id": run_id,
                "design_spec": DESIGN_SPEC,
                "settings": settings,
                "epochs": records,
            },
        )

    to_run = []
    for nwb_file_name, epoch_name in epochs:
        identifier = epoch_id(nwb_file_name, epoch_name)
        if records.get(identifier, {}).get("status") == "ok" and not args.overwrite:
            print(f"{identifier}: already written", flush=True)
        else:
            # the record goes first, so a run stopped part way cannot leave a
            # new file under an old record
            records.pop(identifier, None)
            to_run.append((nwb_file_name, epoch_name))
    write()
    print(
        f"{len(to_run)} epochs to run on {args.n_workers} worker processes, each "
        f"logging to {Path.cwd() / 'logs' / '<epoch_id>_artifacts.log'}",
        flush=True,
    )

    cluster_factory = functools.partial(
        LocalCluster, n_workers=args.n_workers, processes=True, threads_per_worker=1
    )
    failed = []
    run_epoch_kwargs = {"output_dir": output_dir, "run_id": run_id}
    for number, (identifier, record) in enumerate(
        run_epochs(
            to_run,
            run_epoch_kwargs,
            cluster_factory,
            task=_run_epoch_logged,
            key_prefix="artifacts",
        ),
        start=1,
    ):
        records[identifier] = record
        print(f"[{number}/{len(to_run)}] {identifier}: {summary(record)}", flush=True)
        if record["status"] == "failed":
            failed.append(identifier)
        write()
    print(f"manifest -> {manifest_path}", flush=True)
    if failed:
        sys.exit(
            f"{len(failed)} epochs failed: {', '.join(failed)}; see {manifest_path}"
        )


if __name__ == "__main__":
    main()
