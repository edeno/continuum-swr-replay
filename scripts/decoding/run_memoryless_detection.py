#!/usr/bin/env python
"""Run the memoryless sequence detector over recorded HPC epochs.

The Widloski & Foster sliding-window detector, ported in
``continuum_swr_replay.analysis.hippocampus.memoryless_sequence``, run on each
epoch's saved clusterless decode: the encoding model from
``{epoch_id}_detector.pkl``, the decode grid, position, speed and SWR labels
from ``{epoch_id}_80_data_over_time.pkl``, and the spike marks from
``load_data``. Both procedures therefore read one encoding model.

Each epoch writes, under ``Processed-Data/replay_detection_comparison/memoryless/<run_id>/``:

- ``{epoch_id}_memoryless_events.parquet``, one row per candidate sequence;
- ``{epoch_id}_memoryless_shuffle_draws.parquet``, every shuffled draw;
- ``{epoch_id}_memoryless_candidates.pkl``, the per-window statistics, gates
  and pre-merge runs that explain each rejection.

``manifest.json`` records the configuration, the shuffle definition and seed,
the code and library versions, every input's hash, and every epoch's outcome,
including those skipped for a missing input. It is rewritten after each epoch,
so a run that stops part way still says what it did.

The SWR labels are whichever ripple detector the decode used, read from the
epoch identifier: ``chimi20200212_02_r1_HPC`` is Kay's, and a decode with
another detector carries its name, ``..._HPC_Karlsson``.

The design is in ``docs/design/replay_detection_comparison/02-memoryless-sequence.md``.

Usage:
    python scripts/decoding/run_memoryless_detection.py --run-seed 20260922 \\
        --epoch-id chimi20200212_02_r1_HPC
    python scripts/decoding/run_memoryless_detection.py --run-seed 20260922 \\
        --animal chimi
    python scripts/decoding/run_memoryless_detection.py --run-seed 20260922 \\
        --animal chimi --no-shuffle      # the detector alone, much faster
"""

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.constants import DEFAULT_NON_LOCAL_THRESHOLD
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence import (
    MemorylessSequenceConfig,
    SequenceIdentity,
    ShuffleSeed,
    load_clusterless_encoding_model,
    run_epoch,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.event_table import (
    canonical_swr_intervals,
    observed_intervals,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.shuffle import (
    DEFAULT_N_SHUFFLES,
    SIGNIFICANCE_LEVEL,
)
from continuum_swr_replay.analysis.replay.fit_provenance import git_hash
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR
from continuum_swr_replay.table_io import sha256_file, write_table

OUTPUT_ROOT = PROCESSED_DATA_DIR / "replay_detection_comparison" / "memoryless"
DESIGN_SPEC = "docs/design/replay_detection_comparison/02-memoryless-sequence.md"
NON_LOCAL_THRESHOLD_PERCENT = round(100 * DEFAULT_NON_LOCAL_THRESHOLD)
"""The ``data_over_time`` file is named for the decode's non-local threshold."""

DEFAULT_SWR_DEFINITION = "Kay"
"""A decode run with the default ripple detector adds no suffix to its identifier."""


@dataclass(frozen=True)
class EpochKey:
    """What an epoch identifier says about the decode it names."""

    epoch_id: str
    nwb_file_name: str
    epoch_name: str
    brain_area: str
    swr_definition: str


def parse_epoch_identifier(epoch_id: str) -> EpochKey:
    """Split ``{animal}{date}_{epoch}_{brain_area}[_{ripple_detector}]``.

    ``"chimi20200212_02_r1_HPC"`` is ``chimi20200212_.nwb``, epoch ``02_r1``,
    HPC, Kay's SWRs; ``"chimi20200212_02_r1_HPC_Karlsson"`` is the same epoch
    decoded with Karlsson's.
    """
    parts = epoch_id.split("_")
    if len(parts) not in (4, 5) or not all(parts):
        raise ValueError(
            f"{epoch_id!r} is not {{animal}}{{date}}_{{epoch}}_{{brain_area}} with "
            f"an optional _{{ripple_detector}}."
        )
    return EpochKey(
        epoch_id=epoch_id,
        nwb_file_name=f"{parts[0]}_.nwb",
        epoch_name=f"{parts[1]}_{parts[2]}",
        brain_area=parts[3],
        swr_definition=parts[4] if len(parts) == 5 else DEFAULT_SWR_DEFINITION,
    )


def input_paths(epoch_id: str, root: Path) -> dict[str, Path]:
    """The two saved decode artifacts the detector reads."""
    return {
        "detector": root / f"{epoch_id}_detector.pkl",
        "data_over_time": root
        / f"{epoch_id}_{NON_LOCAL_THRESHOLD_PERCENT}_data_over_time.pkl",
    }


def find_epochs(animal: str | None, root: Path) -> list[str]:
    """HPC epochs with both a saved detector and a ``data_over_time`` file.

    HPC only: this port reads a clusterless encoding model.
    """
    pattern = f"{animal}*_HPC*_detector.pkl" if animal else "*_HPC*_detector.pkl"
    epochs = []
    for detector in sorted(root.glob(pattern)):
        epoch_id = detector.name.removesuffix("_detector.pkl")
        if parse_epoch_identifier(epoch_id).brain_area != "HPC":
            continue
        if all(path.exists() for path in input_paths(epoch_id, root).values()):
            epochs.append(epoch_id)
    return epochs


def load_marks(key: EpochKey, cache_dir: Path) -> tuple[list, list]:
    """The epoch's clusterless spike times and waveform features, cached.

    ``load_data`` is slow and needs the Spyglass database; the marks do not
    depend on the ripple detector, so one cache serves every decode of an
    epoch.
    """
    cache = (
        cache_dir / f"{key.nwb_file_name}_{key.epoch_name}_{key.brain_area}_marks.pkl"
    )
    if cache.exists():
        cached: tuple[list, list] = joblib.load(cache)
        return cached
    from continuum_swr_replay.data_loaders import load_data

    data = load_data(key.nwb_file_name, key.epoch_name)
    marks = (
        data["spike_times"][key.brain_area],
        data["spike_waveform_features"][key.brain_area],
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(marks, cache)
    return marks


def _one_value(frame: pd.DataFrame, column: str) -> Any:
    values = pd.unique(frame[column])
    if len(values) != 1:
        raise ValueError(
            f"data_over_time holds {len(values)} values of {column!r}, not one."
        )
    return values[0]


def output_paths(epoch_id: str, output_dir: Path) -> dict[str, Path]:
    return {
        "events": output_dir / f"{epoch_id}_memoryless_events.parquet",
        "shuffle_draws": output_dir / f"{epoch_id}_memoryless_shuffle_draws.parquet",
        "candidates": output_dir / f"{epoch_id}_memoryless_candidates.pkl",
    }


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
    marks_loader: Callable[[EpochKey, Path], tuple[list, list]] = load_marks,
) -> dict:
    """Detect, test and save one epoch; return what the manifest records for it.

    A missing input is a skip, recorded with its reason. Anything else that
    goes wrong raises: a silent skip would shrink the cohort without saying so.
    """
    key = parse_epoch_identifier(epoch_id)
    inputs = input_paths(epoch_id, root)
    missing = [name for name, path in inputs.items() if not path.exists()]
    if missing:
        return {"status": "skipped", "reason": f"missing {', '.join(missing)}"}

    encoding_model = load_clusterless_encoding_model(inputs["detector"])
    data_over_time = pd.read_pickle(inputs["data_over_time"])
    if "time" in data_over_time.columns:
        data_over_time = data_over_time.set_index("time")
    identity = SequenceIdentity(
        animal=str(_one_value(data_over_time, "animal")),
        day=int(_one_value(data_over_time, "day")),
        epoch=str(_one_value(data_over_time, "epoch")),
        brain_area=key.brain_area,
        epoch_id=epoch_id,
        run_id=run_id,
    )
    spike_times, spike_waveform_features = marks_loader(key, cache_dir)

    detection = run_epoch(
        data_over_time.index.to_numpy(dtype=np.float64),
        data_over_time["linear_position"].to_numpy(dtype=np.float64),
        data_over_time["head_speed"].to_numpy(dtype=np.float64),
        spike_times,
        spike_waveform_features,
        encoding_model,
        identity,
        swr_intervals=canonical_swr_intervals(
            data_over_time, speed_threshold=config.speed_threshold_cm_per_s
        ),
        # SWR_event is an integer and never NaN; SWR_zscore is NaN where the LFP is missing
        swr_observed_intervals=observed_intervals(data_over_time, "SWR_zscore"),
        swr_definition=key.swr_definition,
        seed=ShuffleSeed(run=run_seed, session=epoch_id) if shuffle else None,
        config=config,
        n_shuffles=n_shuffles,
    )

    outputs = output_paths(epoch_id, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for frame in (detection.table, detection.shuffle_draws):
        frame.attrs = provenance | {"epoch_id": epoch_id}
    write_table(detection.table, outputs["events"])
    write_table(detection.shuffle_draws, outputs["shuffle_draws"])
    joblib.dump(detection.candidates, outputs["candidates"])

    table = detection.table
    return {
        "status": "ok",
        "swr_definition": key.swr_definition,
        "n_candidates": int(len(table)),
        "n_replay": int(table.is_replay.sum()),
        "n_shuffled": int((table.n_shuffles > 0).sum()),
        "n_shuffle_pass": int(table.shuffle_pass.sum()),
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in inputs.items()
        },
        "outputs": {name: str(path) for name, path in outputs.items()},
    }


def run_manifest(
    run_id: str,
    run_seed: int,
    shuffle: bool,
    n_shuffles: int,
    config: MemorylessSequenceConfig,
    provenance: dict,
    epochs: dict[str, dict],
) -> dict:
    return provenance | {
        "run_id": run_id,
        "design_spec": DESIGN_SPEC,
        "config": asdict(config),
        "shuffle": {
            "run": shuffle,
            "definition": "encoding-pair: within each electrode, encoding "
            "positions permuted against their waveform marks",
            "n_shuffles": n_shuffles if shuffle else 0,
            "run_seed": run_seed,
            "seeded_by": "run seed, epoch_id, candidate's first window, draw, electrode",
            "tested": "candidates meeting the replay criteria and still over "
            "their whole interval",
            "statistics": "mean posterior spread and mean forward jump over every "
            "window, zero jumps kept",
            "p_value": "(1 + count(null <= observed)) / (n_shuffles + 1)",
            "passes": f"both p-values below {SIGNIFICANCE_LEVEL}",
        },
        "libraries": {
            name: version(name) for name in ("non_local_detector", "numpy", "pandas")
        },
        "epochs": epochs,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--epoch-id", action="append", help="One epoch; repeatable.")
    which.add_argument("--animal", help="Every HPC epoch of this animal.")
    which.add_argument("--all", action="store_true", help="Every HPC epoch.")
    parser.add_argument(
        "--run-seed", type=int, required=True, help="Recorded in the manifest."
    )
    parser.add_argument("--n-shuffles", type=int, default=DEFAULT_N_SHUFFLES)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--run-id", help="Defaults to the commit and the seed.")
    parser.add_argument("--root", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=OUTPUT_ROOT / "marks_cache")
    parser.add_argument(
        "--overwrite", action="store_true", help="Redo epochs already written."
    )
    args = parser.parse_args(argv)

    commit = git_hash()
    if commit.endswith("-dirty"):
        print(
            f"warning: uncommitted changes ({commit}); this run cannot be tied "
            f"to a commit",
            file=sys.stderr,
        )
    shuffle = not args.no_shuffle
    run_id = args.run_id or (
        f"{commit[:12]}{'-dirty' if commit.endswith('-dirty') else ''}"
        f"-seed{args.run_seed}{'' if shuffle else '-noshuffle'}"
    )
    output_dir = args.output_root / run_id
    config = MemorylessSequenceConfig()
    provenance = {
        "produced_by": "scripts/decoding/run_memoryless_detection.py",
        "produced_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": commit,
        "run_id": run_id,
    }

    if args.epoch_id:
        epoch_ids = list(args.epoch_id)
    else:
        epoch_ids = find_epochs(None if args.all else args.animal, args.root)
    if not epoch_ids:
        sys.exit(f"No HPC epochs with a detector and data_over_time under {args.root}.")

    manifest_path = output_dir / "manifest.json"
    epochs: dict[str, dict] = {}
    if manifest_path.exists():
        epochs = json.loads(manifest_path.read_text())["epochs"]

    for number, epoch_id in enumerate(epoch_ids, start=1):
        done = epochs.get(epoch_id, {}).get("status") == "ok"
        if done and not args.overwrite:
            print(f"[{number}/{len(epoch_ids)}] {epoch_id}: already written")
            continue
        print(f"[{number}/{len(epoch_ids)}] {epoch_id}", flush=True)
        epochs[epoch_id] = run_one(
            epoch_id,
            root=args.root,
            output_dir=output_dir,
            cache_dir=args.cache_dir,
            run_id=run_id,
            run_seed=args.run_seed,
            shuffle=shuffle,
            n_shuffles=args.n_shuffles,
            config=config,
            provenance=provenance,
        )
        print(f"    {epochs[epoch_id]['status']}: {epochs[epoch_id]}"[:200])
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(
                run_manifest(
                    run_id,
                    args.run_seed,
                    shuffle,
                    args.n_shuffles,
                    config,
                    provenance,
                    epochs,
                ),
                indent=2,
            )
        )
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
