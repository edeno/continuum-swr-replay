"""What the per-event runners share: their arguments, the canonical event table,
and the epoch loop that writes each epoch's record and gathers the run's
manifest (``runs.epoch_records``).

Used by ``measure_event_participation.py``, ``score_event_likelihood_advantage.py``
and ``score_tetrode_holdout.py``.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from continuum_swr_replay.paths import (
    processed_data_dir,
    replay_content_dir,
    table_path,
)
from continuum_swr_replay.runs.batch import lost_epoch_record
from continuum_swr_replay.runs.epoch_records import (
    epochs_to_run,
    gather_manifest,
    write_record,
)
from continuum_swr_replay.runs.io import sha256_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def event_table_path(data_dir: Path) -> Path:
    """The canonical event table under the processed-data root ``data_dir``."""
    return table_path(
        "event_table", data_dir / replay_content_dir().relative_to(processed_data_dir())
    )


def event_table_input(path: Path) -> dict[str, str]:
    """The event table as a run's input: its digest and where it was read."""
    return {"sha256": sha256_file(path), "path": str(path)}


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    """The arguments every per-event runner takes."""
    parser.add_argument("--epochs", nargs="+", help="epoch ids; default every epoch")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--shards",
        type=int,
        nargs="+",
        help="run only the epochs (sorted) whose position modulo --n-shards is "
        "one of these: a host's share of a run split across hosts",
    )
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument(
        "--gather", action="store_true", help="only write the manifest from the records"
    )


def select_epochs(
    epochs: Sequence[str], shards: Sequence[int] | None, n_shards: int
) -> list[str]:
    """The epochs of ``shards`` (all when None), in sorted order."""
    return [
        epoch_id
        for i, epoch_id in enumerate(sorted(epochs))
        if shards is None or i % n_shards in shards
    ]


def run(
    *,
    epochs: Sequence[str],
    task: Callable[..., dict],
    task_args: tuple,
    output: Path,
    name: str,
    header: Mapping[str, Any],
    overwrite: bool,
    max_workers: int,
    pool_options: Mapping[str, Any] | None = None,
) -> None:
    """Run ``task(epoch_id, *task_args)`` for each epoch without an ``ok``
    record of this run, write each record as it arrives, then gather the
    manifest; exit with an error if an epoch failed.
    """
    output.mkdir(parents=True, exist_ok=True)
    to_run = epochs_to_run(output, name, epochs, header, overwrite)
    logging.info(f"{len(to_run)} of {len(epochs)} epochs to run")
    failed = []
    with ProcessPoolExecutor(max_workers=max_workers, **(pool_options or {})) as pool:
        futures = {
            pool.submit(task, epoch_id, *task_args): epoch_id for epoch_id in to_run
        }
        for k, future in enumerate(as_completed(futures), start=1):
            epoch_id = futures[future]
            try:
                record = future.result()
            except Exception as error:  # noqa: BLE001 - recorded per epoch
                record = lost_epoch_record(error)
            write_record(output, epoch_id, name, header, record)
            if record["status"] != "ok":
                failed.append(epoch_id)
            shown = {k: v for k, v in record.items() if k != "traceback"}
            logging.info(f"[{k}/{len(to_run)}] {epoch_id}: {shown}")
    gather_manifest(output, name, header)
    if failed:
        raise SystemExit(f"{len(failed)} epochs failed: {failed[:5]}")
