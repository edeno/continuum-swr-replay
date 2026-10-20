#!/usr/bin/env python
"""Run Carey's candidate detector on every epoch's CA1 LFP and multiunit spikes.

``Carey_candidate_detector`` from ``ripple_detection`` at its pinned commit, in
the package's default Hilbert configuration with tetrodes as units
(``swr_inventories.CAREY_SETTINGS``), run by ``run_carey_detector`` on the SWR
runner's inputs. Its four adaptations of the published detector are recorded
in the run's settings (``ADAPTATIONS``). Design:
``docs/design/replay_detection_comparison/03-inventory-comparison.md``.

Per epoch:

1. **Band.** The stored ripple band (``get_ripple_band_inputs``) is cut to the
   position span (``get_epoch_position_info``), as the SWR runner cuts it, and
   subset to the tetrodes the SWR source run's rule R-T kept
   (``swr_inventory.ripple_power_inputs``). The epoch is refused, and nothing is
   written, unless the band holds that run's electrodes and the sha256 of the
   times and of the kept band are the ones that run recorded
   (``input_sha256["time"]`` and ``["ripple_band"]``): the same inputs, byte
   for byte.
2. **Speed.** ``head_speed`` is interpolated linearly onto the LFP times, NaN
   outside the position span, as the SWR runner interpolates it.
3. **Artifacts.** The mask is the configured artifact run's cores padded by
   ``PADDING_S`` (``epoch_artifact_intervals``), read when the run starts.
4. **Marks.** The decode's HPC marks, read as the memoryless runner reads them
   (``load_marks`` over the decode grid's span, from its cache when present),
   are refused unless every mark the decode's encoding model was fitted on is
   among them (``check_marks_match_encoding``). Each tetrode's spikes are
   binned onto the LFP times (``tetrode_spike_counts``): one unit per tetrode.
   The LFP samples outside the decode span (half a step's tolerance at each
   edge), where no spike was loaded, join the artifact mask as unobserved; the
   LFP itself is not trimmed, so its hash stays the SWR run's.
5. **Detector.** ``run_carey_detector`` leaves the unobserved samples out of
   the joint score's scaling and drops the events that overlap them; they are
   not in the observed stretches either. Its error fails the
   epoch, recorded with the error.

Each epoch writes, under ``Processed-Data/replay_detection_comparison/carey/<run_id>/``:

- ``{epoch_id}_carey_events.parquet``: the detector's event table, with its
  ``attrs`` stamped as the SWR runner stamps its tables;
- ``{epoch_id}_carey_observed.parquet``: where Carey could detect, one row per
  stretch (``carey_observed_stretches``), ``scope`` ``"carey"``, with
  ``start_s`` and ``end_s``, the stretch's first and last sample.

A failed epoch keeps neither. ``manifest.json`` beside them holds the run's
settings and one record per epoch; only this process writes it, atomically,
after each epoch. A rerun under the same run id skips the epochs already
``ok`` unless ``--overwrite`` is given, and refuses settings other than the
ones the manifest was written with. The epochs are the epoch register's
analysed ones, narrowed by ``--animal``, ``--date`` and ``--epoch``, run on a
dask ``LocalCluster`` of ``--n-workers`` processes, each epoch logging to
``logs/{epoch_id}_carey.log`` under the working directory.

Usage:
    python scripts/decoding/run_carey_detection.py --animal chimi --date 20200225 \\
        --epoch 02_r1 --n-workers 1
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

from continuum_swr_replay.analysis.events import swr_inventory  # noqa: E402
from continuum_swr_replay.analysis.events.artifact_masks import (  # noqa: E402
    PADDING_S,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence import (  # noqa: E402
    load_clusterless_encoding_model,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.likelihood import (  # noqa: E402
    check_marks_match_encoding,
)
from continuum_swr_replay.analysis.hippocampus.swr_inventories import (  # noqa: E402
    CAREY_SETTINGS,
    carey_observed_stretches,
    run_carey_detector,
    tetrode_spike_counts,
)
from continuum_swr_replay.data_loaders.epoch_register import (  # noqa: E402
    epochs_to_run,
)
from continuum_swr_replay.data_loaders.lfp import get_ripple_band_inputs  # noqa: E402
from continuum_swr_replay.data_loaders.position import (  # noqa: E402
    get_epoch_position_info,
)
from continuum_swr_replay.data_loaders.utils import parse_epoch_identifier  # noqa: E402
from continuum_swr_replay.decoding_common import (  # noqa: E402
    filter_epoch_info,
    parse_run_id,
    provenance_record,
    raise_open_file_limit,
    release_nwb_files,
    replace_outputs,
    setup_logger,
    write_atomically,
)
from continuum_swr_replay.intervals import (  # noqa: E402
    interval_mask,
    masked_seconds,
)
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR, ROOT_DIR  # noqa: E402
from continuum_swr_replay.tables.io import sha256_file  # noqa: E402
from scripts.decoding.run_memoryless_detection import (  # noqa: E402
    MARKS_RULE,
    input_paths,
    load_marks,
    marks_cache_path,
    marks_sha256,
)
from scripts.decoding.run_memoryless_detection import (  # noqa: E402
    OUTPUT_ROOT as MEMORYLESS_ROOT,
)
from scripts.decoding.run_swr_detection import (  # noqa: E402
    array_sha256,
    epoch_artifact_intervals,
    epoch_id,
    json_ready,
    run_epochs,
    run_lock,
)

OUTPUT_ROOT = PROCESSED_DATA_DIR / "replay_detection_comparison" / "carey"
PRODUCED_BY = "scripts/decoding/run_carey_detection.py"
DESIGN_SPEC = "docs/design/replay_detection_comparison/03-inventory-comparison.md"

DECODE_ROOT = PROCESSED_DATA_DIR
"""Where the canonical HPC decodes are, whose marks Carey reads."""
MARKS_CACHE_DIR = MEMORYLESS_ROOT / "marks_cache"
"""The memoryless runner's marks cache, keyed by the decode's detector file."""

CONFIGURATION = (
    "the package's default Hilbert configuration of Carey_candidate_detector, "
    "with tetrodes as units"
)
ADAPTATIONS = (
    "a tetrode's multiunit train is one unit",
    "the Hilbert ripple score, not the published spectral one",
    "the stored 150-250 Hz band averaged over the R-T kept tetrodes, not one "
    "140-250 Hz channel",
    "no theta restriction",
)
"""Where this run departs from the published detector."""
GRID = (
    "the stored band's own samples within the position span; each tetrode's "
    "spikes binned onto them by the MUA rule (tetrode_spike_counts); samples "
    "outside the decode span, where no spike was loaded, unobserved"
)
"""The rate is the band's own, not asserted here: the time hash ties the
samples to the SWR run's, and each record keeps the nominal rate."""


def output_paths(identifier: str, output_dir: Path) -> dict[str, Path]:
    """Every file an epoch writes, by name."""
    return {
        "events": output_dir / f"{identifier}_carey_events.parquet",
        "observed": output_dir / f"{identifier}_carey_observed.parquet",
    }


def swr_run_inputs(identifier: str) -> dict[str, Any]:
    """What the SWR source run read for the epoch.

    ``swr_inventory.ripple_power_inputs`` (the band's and the kept electrodes, the
    rate and the position span), with the run's ``input_sha256`` record.

    Raises
    ------
    SwrSourceError
        If the run does not serve the epoch.
    """
    inputs = swr_inventory.ripple_power_inputs(identifier)
    manifest = json.loads(
        (swr_inventory.swr_inventory_dir() / "manifest.json").read_text()
    )
    with_hashes: dict[str, Any] = inputs | {
        "input_sha256": manifest["epochs"][identifier]["input_sha256"]
    }
    return with_hashes


def decode_marks(
    identifier: str,
    *,
    root: Path = DECODE_ROOT,
    cache_dir: Path = MARKS_CACHE_DIR,
    marks_loader: Callable = load_marks,
) -> dict[str, Any]:
    """The canonical decode's HPC marks and its encoding model.

    The marks are read as the memoryless runner reads them: ``load_marks``
    over the decode grid's first and last time, cached by the decode's
    ``_detector.pkl`` sha256.

    Returns
    -------
    dict
        ``spike_times`` and ``spike_waveform_features`` (one array per
        tetrode), ``encoding_model``, and ``record``: the marks rule, the
        detector's sha256, the cache file's name and the decode's span.
    """
    paths = input_paths(identifier, root)
    detector_sha256 = sha256_file(paths["detector"])
    encoding_model = load_clusterless_encoding_model(paths["detector"])
    data_over_time = pd.read_pickle(paths["data_over_time"])
    time = data_over_time.index.to_numpy(dtype=np.float64)
    span_s = (float(time[0]), float(time[-1]))
    del data_over_time
    key = parse_epoch_identifier(identifier)
    spike_times, spike_waveform_features = marks_loader(
        key, cache_dir, span_s, detector_sha256
    )
    return {
        "spike_times": spike_times,
        "spike_waveform_features": spike_waveform_features,
        "encoding_model": encoding_model,
        "record": {
            "rule": MARKS_RULE,
            "detector_sha256": detector_sha256,
            "cache": marks_cache_path(key, cache_dir, detector_sha256).name,
            "decode_span_s": list(span_s),
        },
    }


def observation_table(time: NDArray, stretches: NDArray) -> pd.DataFrame:
    """``carey_observed_stretches``' stretches, as the SWR runner lays them out.

    Parameters
    ----------
    time : NDArray, shape (n_time,)
        Sample times in seconds.
    stretches : NDArray, shape (n_stretches, 2)
        ``[start, stop)`` row indices.

    Returns
    -------
    pd.DataFrame
        ``scope`` (``"carey"``), and ``start_s`` and ``end_s``, the times of a
        stretch's first and last sample.
    """
    return pd.DataFrame(
        {
            "scope": ["carey"] * len(stretches),
            "start_s": time[stretches[:, 0]],
            "end_s": time[stretches[:, 1] - 1],
        }
    ).astype({"scope": object, "start_s": np.float64, "end_s": np.float64})


def _detect(
    nwb_file_name: str,
    epoch_name: str,
    *,
    run_id: str,
    artifact_dir: Path | None,
    fetch_inputs: Callable,
    fetch_position: Callable,
    fetch_swr_inputs: Callable,
    fetch_marks: Callable,
) -> tuple[dict, dict[str, pd.DataFrame]]:
    """One epoch's record, and its tables to write (none for a failed epoch)."""
    identifier = epoch_id(nwb_file_name, epoch_name)
    inputs = fetch_inputs(nwb_file_name, epoch_name)
    position_info = fetch_position(nwb_file_name, epoch_name)["position_info"]
    position_time = position_info.index.to_numpy(dtype=np.float64)
    head_speed = position_info["head_speed"].to_numpy(dtype=np.float64)
    span_s = (float(position_time[0]), float(position_time[-1]))
    # the band's timestamps ascend, so the span is one slice, as the SWR
    # runner cuts it
    first = int(np.searchsorted(inputs["time"], span_s[0], side="left"))
    stop = int(np.searchsorted(inputs["time"], span_s[1], side="right"))
    time = inputs["time"][first:stop]
    ripple_band = inputs["ripple_band"][first:stop]

    swr_run = fetch_swr_inputs(identifier)
    swr_run_id = swr_inventory.SWR_INVENTORY_RUN_ID
    electrode_ids = np.asarray(inputs["electrode_ids"], dtype=np.int64)
    kept_ids = np.asarray(swr_run["kept_electrode_ids"], dtype=np.int64)
    kept = np.isin(electrode_ids, kept_ids)
    if not np.array_equal(electrode_ids, swr_run["band_electrode_ids"]) or (
        kept.sum() != kept_ids.size
    ):
        reason = (
            f"The band holds electrodes {electrode_ids.tolist()}, not the band "
            f"{list(swr_run['band_electrode_ids'])} SWR run {swr_run_id} read with "
            f"its kept tetrodes {kept_ids.tolist()}."
        )
        return {"status": "failed", "reason": reason}, {}
    if not kept.all():  # a copy, made only when a tetrode is left out
        ripple_band = ripple_band[:, kept]
    input_sha256 = {
        "time": array_sha256(time),
        "ripple_band": array_sha256(ripple_band),
    }
    for name, sha in input_sha256.items():
        recorded = swr_run["input_sha256"][name]
        if sha != recorded:
            reason = (
                f"The {name} hash {sha[:12]} is not the {recorded[:12]} SWR run "
                f"{swr_run_id} recorded: these are not the inputs it chose its "
                "tetrodes on."
            )
            return {
                "status": "failed",
                "reason": reason,
                "input_sha256": input_sha256,
                "swr_input_sha256": dict(swr_run["input_sha256"]),
            }, {}

    speed = np.interp(time, position_time, head_speed, left=np.nan, right=np.nan)
    artifact_mask = epoch_artifact_intervals(artifact_dir, identifier)
    unobserved = interval_mask(time, artifact_mask)

    marks = fetch_marks(identifier)
    spike_times = marks["spike_times"]
    spike_waveform_features = marks["spike_waveform_features"]
    check_marks_match_encoding(
        marks["encoding_model"], spike_times, spike_waveform_features
    )
    sampling_frequency = float(inputs["sampling_frequency"])
    # The marks were loaded over the decode span only: outside it a zero count
    # means no spike was loaded, not that none fired. Those samples join the
    # mask; half a step at each edge keeps float rounding off an edge sample.
    decode_lo, decode_hi = marks["record"]["decode_span_s"]
    half_step = 0.5 / sampling_frequency
    outside_decode = (time < decode_lo - half_step) | (time > decode_hi + half_step)
    unobserved = unobserved | outside_decode
    spike_counts = tetrode_spike_counts(spike_times, time)
    events = run_carey_detector(
        time,
        ripple_band,
        spike_counts,
        speed,
        sampling_frequency,
        unobserved=unobserved,
    )

    record: dict[str, Any] = {
        "status": "ok",
        "nwb_file_name": nwb_file_name,
        "epoch_name": epoch_name,
        "produced_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "provenance": provenance_record(ROOT_DIR)
        | {"numpy_version": np.__version__, "scipy_version": scipy.__version__},
        "electrode_ids": electrode_ids[kept],
        "n_multiunit_tetrodes": len(spike_times),
        "sampling_frequency": sampling_frequency,
        "position_span_s": span_s,
        "decode_span_s": [decode_lo, decode_hi],
        # the LFP samples outside the decode span, as samples / nominal rate
        "seconds_outside_decode_span": int(np.count_nonzero(outside_decode))
        / sampling_frequency,
        "input_sha256": input_sha256,
        "marks": marks["record"]
        | {"sha256": marks_sha256(spike_times, spike_waveform_features)},
        "artifacts": {
            "run_id": None if artifact_dir is None else artifact_dir.name,
            "n_intervals": len(artifact_mask),
            # the masked time within the LFP rows read
            "masked_s": masked_seconds(artifact_mask.clip(time[0], time[-1])),
        },
    }
    if isinstance(events, Exception):
        record["status"] = "failed"
        record["error"] = repr(events)
        return record, {}
    record["n_events"] = len(events)
    record["n_dropped_overlapping_unobserved"] = int(
        events.attrs["n_dropped_overlapping_unobserved"]
    )

    observed = observation_table(
        time, carey_observed_stretches(time, ripple_band, spike_counts, unobserved)
    )
    stamp = {
        "produced_by": PRODUCED_BY,
        "run_id": run_id,
        "epoch_id": identifier,
        "commit": record["provenance"]["commit"],
    }
    for frame in (events, observed):
        frame.attrs = frame.attrs | stamp
    return record, {"events": events, "observed": observed}


def run_epoch(
    nwb_file_name: str,
    epoch_name: str,
    *,
    output_dir: Path,
    fetch_inputs: Callable = get_ripple_band_inputs,
    fetch_position: Callable = get_epoch_position_info,
    fetch_swr_inputs: Callable = swr_run_inputs,
    fetch_marks: Callable = decode_marks,
    run_id: str | None = None,
    artifact_dir: Path | None = None,
) -> dict:
    """Run Carey on one epoch and write its outputs.

    Parameters
    ----------
    nwb_file_name, epoch_name : str
        The epoch, such as ``"chimi20200225_.nwb"`` and ``"02_r1"``.
    output_dir : Path
        The run's folder.
    fetch_inputs : Callable, optional
        ``(nwb_file_name, epoch_name) -> RippleBandInputs``; its ``time``,
        ``ripple_band``, ``electrode_ids`` and ``sampling_frequency`` are read.
    fetch_position : Callable, optional
        ``(nwb_file_name, epoch_name) -> PositionInfoDict``, whose
        ``position_info`` gives the span and ``head_speed``.
    fetch_swr_inputs : Callable, optional
        ``(epoch_id) -> dict``, as ``swr_run_inputs`` returns it.
    fetch_marks : Callable, optional
        ``(epoch_id) -> dict``, as ``decode_marks`` returns it.
    run_id : str, optional
        Stamped on the tables; default ``output_dir.name``.
    artifact_dir : Path, optional
        The artifact run folder; None masks nothing.

    Returns
    -------
    dict
        The epoch's manifest record, in strict JSON types: ``status`` ``"ok"``,
        or ``"failed"`` with a ``reason`` (inputs other than the SWR run's),
        the detector's ``error``, or any other ``error`` with its
        ``traceback``. A failed epoch's outputs are removed.
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
            fetch_swr_inputs=fetch_swr_inputs,
            fetch_marks=fetch_marks,
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
    # the worker's peak resident set over its life so far (KiB on Linux)
    record["worker_peak_rss_gib"] = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    )
    ready: dict = json_ready(record)
    return ready


def summary(record: dict) -> str:
    """One epoch's outcome, as the logs and the console report it."""
    if "n_events" not in record:
        return f"{record['status']}: {record.get('reason') or record.get('error')}"
    return (
        f"{record['status']}: {record['n_events']} events "
        f"({record['n_dropped_overlapping_unobserved']} dropped as unobserved) "
        f"on {len(record['electrode_ids'])} band tetrodes and "
        f"{record['n_multiunit_tetrodes']} multiunit tetrodes; "
        f"{record['wall_clock_s']:.0f} s, worker peak RSS "
        f"{record['worker_peak_rss_gib']:.1f} GiB"
    )


def _run_epoch_logged(
    nwb_file_name: str, epoch_name: str, run_epoch_kwargs: dict, log_directory: Path
) -> dict:
    """One dask task: ``run_epoch``, logged to ``logs/{epoch_id}_carey.log``.

    When the epoch ends, pass or fail, the NWB files Spyglass opened for it are
    closed (``release_nwb_files``): the workers run one task at a time.
    """
    identifier = epoch_id(nwb_file_name, epoch_name)
    log = setup_logger(
        name_logfile=f"{identifier}_carey",
        path_logfile=log_directory / f"{identifier}_carey.log",
    )
    log.info("%s: run %s", identifier, run_epoch_kwargs)
    try:
        record = run_epoch(nwb_file_name, epoch_name, **run_epoch_kwargs)
    finally:
        log.info("closed %d NWB files Spyglass held open", release_nwb_files())
    if "provenance" in record:
        log.info("PROVENANCE %s", json.dumps(record["provenance"], sort_keys=True))
    if record["status"] == "failed":
        log.error("%s\n%s", summary(record), record.get("traceback", ""))
    else:
        log.info("%s", summary(record))
    return record


def run_settings(provenance: dict, artifact_run_id: str | None) -> dict:
    """What every epoch of a run shares; a rerun under its run id must match."""
    settings: dict = json_ready(
        {
            "detector": "Carey_candidate_detector",
            "configuration": CONFIGURATION,
            "carey_settings": CAREY_SETTINGS,
            "adaptations": ADAPTATIONS,
            "ripple_detection_commit": provenance["ripple_detection_commit"],
            "swr_run_id": swr_inventory.SWR_INVENTORY_RUN_ID,
            "artifacts": {"run_id": artifact_run_id, "padding_s": PADDING_S},
            "marks_rule": MARKS_RULE,
            "grid": GRID,
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
    """``main``'s run, under the run folder's lock."""
    run_id = output_dir.name
    # read once, here: every epoch of the run is masked by the same artifact run
    artifact_dir = swr_inventory.artifact_intervals_dir()
    settings = run_settings(
        provenance, None if artifact_dir is None else artifact_dir.name
    )
    epochs = list(dict.fromkeys(filter_epoch_info(epochs_to_run(), args).index))
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
            # the record goes first, so a run stopped part way cannot leave new
            # files under an old record
            records.pop(identifier, None)
            to_run.append((nwb_file_name, epoch_name))
    write_manifest()
    print(
        f"{len(to_run)} epochs to run on {args.n_workers} worker processes, each "
        f"logging to {Path.cwd() / 'logs' / '<epoch_id>_carey.log'}",
        flush=True,
    )

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
    for number, (identifier, record) in enumerate(
        run_epochs(
            to_run,
            run_epoch_kwargs,
            cluster_factory,
            task=_run_epoch_logged,
            key_prefix="carey",
        ),
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
