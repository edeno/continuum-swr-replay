#!/usr/bin/env python
"""Held-out spike evidence in each canonical non-local event, split by SWR.

A reference copy (2026-11-18): this leave-one-tetrode-out test was set aside for
Figure 2 in favour of the whole-event score
(``scripts/event_evidence/score_event_likelihood_advantage.py``), which answers
the descriptive question with all tetrodes; this one is kept for a claim of
independent population support. Its 2026-11-18 run is in
``Processed-Data/event_heldout_evidence``.

Are the SWR and the non-SWR non-local events real, judged by spikes that did
not choose their remote location? ``run_state_evidence.py`` asks it of the
decoder's confident state bouts; this runner asks it of the events the paper
analyses: every immobile non-local event of the canonical event table
(``replay_content/event_table.parquet``, read through the SWR source), one
window per event over every bin of it, labelled by the table's ``is_swr``
(strict positive-duration overlap, the canonical rule).

Per epoch it runs ``run_state_evidence``'s engine unchanged (``run_epoch``: the
decode's checks, the marks, the four encodings and their checks, the held-out
evidence of every window under each, the primary being the time-held-out
running-only ``runcv_``), with the windows of ``event_windows`` in place of the
state bouts:

- an event's window is its decode bins from ``event_start_s`` to
  ``event_end_s``, inclusive; it is ``kind == "bout"`` with ``state``
  ``"Non-Local | SWR"`` or ``"Non-Local | no SWR"``, and carries its
  ``event_id``, ``is_swr`` and ``overlap_class``;
- its matched window (``kind == "baseline"``, the same ``event_id``) is drawn
  as a bout's is (``matched_windows``: every bin immobile, the same length,
  a seeded order per length, seeded by ``(SEED, epoch_id)``), so the event -
  matched difference is paired. A matched window may overlap an event
  (``event_bin_share``), as a bout's may;
- an event with a bin outside the immobile regime (excluded time, no finite
  linear position, a speed of 4 cm/s or more) is not scored; each such event
  and its reason is in the epoch's record (``events``). Every bin of an event
  must have P(Non-Local) above the event rule's threshold on the decode it is
  scored on, else the epoch fails: the table is not that decode's.

It writes, under ``--output-dir`` (default
``Processed-Data/event_heldout_evidence``):

- ``per_epoch/{epoch}_windows.parquet`` and ``per_epoch/{epoch}_validation.json``,
  the engine's per-window table and record (the record also holds ``events``:
  the epoch's events in the table, scored and dropped, by label and reason);
- ``event_stats.csv``: ``bout_statistics`` over every ``ok`` epoch, per label
  (``state``) and metric, with the animal-level estimate and 95 % interval
  (``animal_*``) and the paired event - matched difference (``diff_animal_*``),
  over the metrics of the bout split (``split_state_evidence_by_swr.METRICS``);
- ``manifest.json`` (settings, provenance, every epoch's record) and
  ``README.md``; per-epoch logs in ``logs/``.

The settings hold the event table's path, size and ``mtime_ns``, so a rerun
reuses an epoch only while the table, the code commit (clean trees), the run
settings and the decode files are those it was computed with
(``run_state_evidence.reuse_reason``). ``--summary-only`` rebuilds the pooled
statistics from the manifest's ``ok`` epochs without computing any.

It runs on the CPU only, one epoch per dask worker.

Usage::

    uv run --frozen python scripts/exploratory/_run_event_heldout_evidence.py \\
        --n-workers 16
    uv run --frozen python scripts/exploratory/_run_event_heldout_evidence.py \\
        --summary-only
"""

import os

# a CPU job, as the engine's: set before any JAX import
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse  # noqa: E402
import functools  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from collections.abc import Callable, Iterator  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import pandas as pd  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

from continuum_swr_replay.analysis.event_definition.analysed_time import (  # noqa: E402
    read_swr_table,
)
from continuum_swr_replay.analysis.event_definition.swr_source import (  # noqa: E402
    swr_source_name,
)
from continuum_swr_replay.analysis.event_evidence.state_evidence import (  # noqa: E402
    EVENT_STATE,
    SWR_LABELS,
    bout_statistics,
    code_version,
    event_windows,
)
from continuum_swr_replay.decoding.identifiers import (  # noqa: E402
    format_epoch_identifier,
    nwb_file_name,
    parse_epoch_identifier,
)
from continuum_swr_replay.parameters import ROOT_DIR  # noqa: E402
from continuum_swr_replay.paths import (  # noqa: E402
    EVENT_HELDOUT_EVIDENCE_DIR,
    processed_data_dir,
    table_path,
)
from continuum_swr_replay.runs.batch import (  # noqa: E402
    raise_open_file_limit,
    run_epoch_logged,
)
from continuum_swr_replay.runs.batch import (  # noqa: E402
    run_epochs as batch_run_epochs,
)
from continuum_swr_replay.runs.io import (  # noqa: E402
    json_ready,
    ok_epochs,
    read_manifest,
    write_json,
)
from continuum_swr_replay.runs.provenance import (  # noqa: E402
    produced_by,
    provenance_record,
)
from continuum_swr_replay.tables.io import read_table  # noqa: E402

# Sibling-script import: run as a script, only this file's directory is on
# sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.event_evidence import run_state_evidence as engine  # noqa: E402
from scripts.event_evidence.split_state_evidence_by_swr import METRICS  # noqa: E402

PRODUCED_BY = produced_by(__file__)
LABEL_SET = "events"
"""The engine's name for this run's one label set (the record's ``label_sets``)."""
EVENT_COLUMNS = ["event_id", "is_swr", "overlap_class"]
"""The event table's columns each scored window carries."""
EVENT_STATES = [f"{EVENT_STATE} | {label}" for label in SWR_LABELS.values()]
STATS_FILE = "event_stats.csv"


def file_stat(path: Path) -> dict:
    """A file's resolved path, size and ``mtime_ns``."""
    info = path.stat()
    return {
        "path": str(path.resolve()),
        "size": info.st_size,
        "mtime_ns": info.st_mtime_ns,
    }


def run_settings(event_table: Path) -> dict:
    """The engine's settings with this run's windows and the event table's stamp."""
    settings = engine.run_settings()
    del settings["bouts"], settings["calibration"]
    settings["windows"] = (
        "one per canonical event of the event table, every bin of it "
        "(event_start_s to event_end_s, inclusive decode bins), labelled SWR / "
        "no SWR by is_swr; one matched window per event (matched_windows: every "
        "bin immobile, the same length, seeded); an event with a bin outside the "
        "immobile regime is not scored (event_windows)"
    )
    settings["event_table"] = file_stat(event_table)
    return settings


def epoch_events(event_table: Path, epoch_id: str) -> pd.DataFrame:
    """One epoch's events: ``EVENT_COLUMNS`` and their start and end, s."""
    key = parse_epoch_identifier(epoch_id)
    return read_swr_table(
        event_table,
        columns=[*EVENT_COLUMNS, "event_start_s", "event_end_s"],
        filters=[
            ("animal", "==", key.animal),
            ("session_date", "==", key.date),
            ("epoch", "==", key.epoch_name),
        ],
    )


def table_epochs(event_table: Path) -> list[str]:
    """The HPC epoch id of every epoch with an event in the table, sorted."""
    keys = read_swr_table(
        event_table, columns=["animal", "session_date", "epoch"]
    ).drop_duplicates()
    return sorted(
        format_epoch_identifier(nwb_file_name(a, d), e, "HPC")
        for a, d, e in keys.itertuples(index=False)
    )


def event_record(events: pd.DataFrame, dropped: pd.DataFrame) -> dict:
    """The epoch's events in the table, scored and dropped, by label and reason."""
    in_table = events.is_swr.map(SWR_LABELS).value_counts()
    dropped_label = dropped.is_swr.map(SWR_LABELS)
    by_reason = dropped.groupby([dropped_label, "reason"]).size()
    return {
        "in_table": in_table.to_dict(),
        "scored": (
            in_table
            - dropped_label.value_counts().reindex(in_table.index, fill_value=0)
        ).to_dict(),
        "dropped": {f"{label}|{reason}": n for (label, reason), n in by_reason.items()},
        "dropped_event_ids": dropped.event_id.tolist(),
    }


def run_epoch(
    epoch_id: str,
    *,
    decode_dir: Path,
    output_dir: Path,
    event_table: Path,
    settings: dict,
    code: dict | None = None,
    load_inputs: Callable[[str], dict] = engine.load_marks_and_training_mask,
    log: Callable[[str], None] = print,
) -> dict:
    """Compute and write one epoch (``engine.run_epoch`` on the event windows).

    An ``"ok"`` record also holds ``events`` (:func:`event_record`), and its
    validation file is rewritten with it.
    """
    events: dict[str, Any] = {}

    def windows(time: NDArray, probabilities: NDArray, regime: NDArray) -> pd.DataFrame:
        table = epoch_events(event_table, epoch_id)
        built, dropped = event_windows(
            time, probabilities, regime, table, seed=(engine.SEED, epoch_id)
        )
        events.update(event_record(table, dropped))
        log(f"{len(table)} events, {len(dropped)} not scored: {events['dropped']}")
        return built

    label_set = engine.LabelSet(name=LABEL_SET, windows=windows)
    record = engine.run_epoch(
        epoch_id,
        decode_dir=decode_dir,
        output_dir=output_dir,
        code=code,
        load_inputs=load_inputs,
        log=log,
        label_sets=lambda _epoch_id: [label_set],
        settings=settings,
    )
    if record["status"] == "ok":
        record["events"] = json_ready(events)
        write_json(
            engine.output_paths(epoch_id, output_dir)["validation"],
            record,
            indent=1,
            default=str,
        )
    return record


def _run_epoch_logged(epoch_id: str, kwargs: dict, log_dir: Path) -> dict:
    """One dask task: :func:`run_epoch`, logged to ``logs/{epoch_id}.log``."""

    def describe(record: dict) -> str:
        if record["status"] == "failed":
            return record["error"]
        return f"{epoch_id}: {record['status']}"

    return run_epoch_logged(
        lambda logger: run_epoch(epoch_id, log=logger.info, **kwargs),
        log_name=f"{epoch_id}_event_heldout_evidence",
        log_path=log_dir / f"{epoch_id}.log",
        start=f"{epoch_id}: start",
        describe=describe,
        log_provenance=False,
    )


def run_epochs(
    epochs: list[str], kwargs: dict, n_workers: int
) -> Iterator[tuple[str, dict]]:
    """Each epoch on its own dask CPU worker; yield records as they finish."""
    from dask.distributed import LocalCluster

    log_dir = kwargs["output_dir"] / "logs"
    cluster_factory = functools.partial(
        LocalCluster,
        n_workers=n_workers,
        processes=True,
        threads_per_worker=1,
        dashboard_address=":0",
    )
    yield from batch_run_epochs(
        {epoch_id: (epoch_id, kwargs, log_dir) for epoch_id in epochs},
        _run_epoch_logged,
        cluster_factory,
    )


def summarize(output_dir: Path, epochs: list[str]) -> pd.DataFrame | None:
    """``event_stats.csv`` over the given epochs' windows; None with no epoch."""
    if not epochs:
        print("no epoch written; no summary", flush=True)
        return None
    windows = pd.concat(
        [
            read_table(engine.output_paths(epoch_id, output_dir)["windows"])
            for epoch_id in epochs
        ],
        ignore_index=True,
    )
    stats = bout_statistics(windows, metrics=dict.fromkeys(EVENT_STATES, METRICS))
    stats = stats[stats.n > 0].reset_index(drop=True)
    engine.write_csv(stats, output_dir / STATS_FILE)
    shown = ["state", "metric", "n", "n_epochs", "n_animals", "median", "base_median",
             "animal_estimate", "animal_ci_lo", "animal_ci_hi", "diff_animal_estimate",
             "diff_animal_ci_lo", "diff_animal_ci_hi", "diff_animals_pos"]  # fmt: skip
    print(stats[[c for c in shown if c in stats]].to_string(index=False), flush=True)
    return stats


def readme(manifest: dict) -> str:
    """The output folder's README: what was run, coverage, drops and settings."""
    records = manifest["epochs"]
    status = pd.Series({k: r["status"] for k, r in records.items()}, dtype=object)
    drops: dict[str, int] = {}
    for record in records.values():
        for key, n in (record.get("events") or {}).get("dropped", {}).items():
            drops[key] = drops.get(key, 0) + n
    provenance = manifest["provenance"]
    lines = [
        "# Held-out spike evidence in the canonical non-local events",
        "",
        f"Produced by `{PRODUCED_BY}` at commit {provenance['commit']}"
        f"{' (uncommitted changes)' if provenance['dirty'] else ''}; the engine "
        "and its caveats are `scripts/event_evidence/run_state_evidence.py`'s.",
        "",
        f"Epochs: {manifest['discovery']['rule']}; {len(records)} listed. Status: "
        + ", ".join(f"{k} {v}" for k, v in status.value_counts().sort_index().items())
        + ".",
        "",
        "Events not scored (label|reason: count): "
        + (", ".join(f"{k}: {v}" for k, v in sorted(drops.items())) or "none")
        + ".",
        "",
    ]
    not_ok = {k: r for k, r in records.items() if r["status"] != "ok"}
    if not_ok:
        lines += ["| epoch | status | reason |", "|---|---|---|"]
        lines += [
            f"| {k} | {r['status']} | {r.get('reason') or r.get('error')} |"
            for k, r in sorted(not_ok.items())
        ]
        lines.append("")
    lines += [
        "## Settings",
        "",
        "```json",
        json.dumps(manifest["settings"], indent=1),
        "```",
        "",
        "## Files",
        "",
        "- `per_epoch/{epoch}_windows.parquet`, `per_epoch/{epoch}_validation.json`",
        f"- `{STATS_FILE}`: per label (state) and metric; animal_* the events, "
        "diff_animal_* the paired event - matched difference",
        "- `manifest.json`, `logs/{epoch}.log`",
        "",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """The command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--epochs",
        nargs="+",
        help="HPC epoch ids, such as wilbur20210329_02_r1_HPC (default: every "
        "epoch with an event in --event-table).",
    )
    parser.add_argument(
        "--decode-dir",
        type=Path,
        default=processed_data_dir(),
        help="Where the decodes are read from (default: Processed-Data).",
    )
    parser.add_argument(
        "--event-table",
        type=Path,
        default=table_path("event_table"),
        help="The canonical event table (default: replay_content/event_table.parquet).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=processed_data_dir() / EVENT_HELDOUT_EVIDENCE_DIR,
        help="The only place written to (default: "
        f"Processed-Data/{EVENT_HELDOUT_EVIDENCE_DIR}).",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=8,
        help="Dask CPU worker processes, one epoch each at a time (default 8).",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Recompute epochs already written."
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help=f"Rebuild {STATS_FILE} from the manifest's ok epochs; compute nothing.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    decode_dir, output_dir = args.decode_dir.resolve(), args.output_dir.resolve()
    event_table = args.event_table.resolve()
    if args.summary_only:
        summarize(output_dir, ok_epochs(read_manifest(output_dir)))
        return
    if output_dir == decode_dir:
        parser.error("--output-dir must not be the decode folder itself")
    old_soft, new_soft = raise_open_file_limit()
    print(f"open-file soft limit {old_soft} -> {new_soft}", flush=True)
    epochs = args.epochs or table_epochs(event_table)
    discovery = {
        "rule": "the epochs given with --epochs"
        if args.epochs
        else f"every epoch with an event in {event_table}",
        "n_epochs": len(epochs),
    }
    provenance = provenance_record(ROOT_DIR)
    code = code_version(provenance)
    settings = run_settings(event_table)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict] = {}
    recomputed: dict[str, str] = {}
    to_run = []
    for epoch_id in epochs:
        paths = engine.output_paths(epoch_id, output_dir)
        previous = (
            json.loads(paths["validation"].read_text())
            if paths["validation"].exists()
            else None
        )
        reason = (
            "--overwrite"
            if args.overwrite
            else engine.reuse_reason(
                previous,
                settings=settings,
                code=code,
                decode_stats=engine.decode_file_stats(epoch_id, decode_dir),
                windows_written=paths["windows"].exists(),
            )
        )
        if reason is None and previous is not None:
            records[epoch_id] = previous
            print(f"{epoch_id}: reused (same commit, settings and decode files)")
        else:
            recomputed[epoch_id] = str(reason)
            to_run.append(epoch_id)

    def manifest() -> dict:
        return {
            "produced_by": PRODUCED_BY,
            "provenance": provenance,
            "settings": settings,
            "swr_source": swr_source_name(),
            "decode_dir": str(decode_dir),
            "discovery": discovery,
            "reused": sorted(set(records) - set(recomputed)),
            "recomputed": dict(sorted(recomputed.items())),
            "epochs": {k: records[k] for k in sorted(records)},
        }

    print(
        f"{len(to_run)} of {len(epochs)} epochs to run on {args.n_workers} workers; "
        f"logs in {output_dir / 'logs'}",
        flush=True,
    )
    kwargs = {
        "decode_dir": decode_dir,
        "output_dir": output_dir,
        "event_table": event_table,
        "settings": settings,
        "code": code,
    }
    for number, (epoch_id, record) in enumerate(
        run_epochs(to_run, kwargs, args.n_workers), start=1
    ):
        records[epoch_id] = record
        detail = (
            record.get("reason")
            or record.get("error")
            or (
                f"{record['wall_clock_s']:.0f} s, peak RSS "
                f"{record['worker_peak_rss_gib']:.1f} GiB"
            )
        )
        print(f"[{number}/{len(to_run)}] {epoch_id}: {record['status']} ({detail})",
              flush=True)  # fmt: skip
        write_json(output_dir / "manifest.json", manifest(), indent=1, default=str)
    final = manifest()
    write_json(output_dir / "manifest.json", final, indent=1, default=str)
    summarize(output_dir, sorted(k for k, r in records.items() if r["status"] == "ok"))
    (output_dir / "README.md").write_text(readme(final))
    failed = [k for k, r in records.items() if r["status"] == "failed"]
    if failed:
        sys.exit(f"{len(failed)} epochs failed: {', '.join(sorted(failed))}")


if __name__ == "__main__":
    main()
