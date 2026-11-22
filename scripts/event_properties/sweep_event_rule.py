"""Count the non-local events, and those with no SWR, at other settings of the event rule.

A sensitivity check of the canonical events (question 2a.5): how the count of
immobile non-local events that start inside a well visit, and the share of
them with no SWR, change with the non-local probability threshold, the minimum
run length and the SWR threshold, one at a time
(``analysis.event_properties.rule_sweep``). The events are detected again from
each decode's stored non-local probability by the rule itself, held to the
analyzed-time rule and labeled by SWR overlap as the canonical table labels
them; the SWR threshold is swept through each event's Kay criterion on the
decode grid (the module says why and how faithful that is). The paper's events
are not changed: at the rule's own settings every epoch's counts must equal
the canonical event table's, or nothing is written.

For the ``ok`` epochs of an event features run, from the same decodes (each
decode's sha256 must be the one the run recorded), with each epoch's well
visits from the session cache. One row per epoch, setting and value, written
to ``Processed-Data/event_rule_sweep/counts.parquet`` (or ``--output-dir``)
with a ``manifest.json`` beside it.

Usage::

    uv run python scripts/event_properties/sweep_event_rule.py \\
        --event-features-run e2331f976fd1 --n-workers 16
    uv run python scripts/event_properties/sweep_event_rule.py \\
        --event-features-run e2331f976fd1 --epochs chimi20200212_02_r1_HPC \\
        --output-dir /tmp/check
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from continuum_swr_replay.analysis.detector_comparison import inventory_run
from continuum_swr_replay.analysis.event_definition.analyzed_time import read_swr_table
from continuum_swr_replay.analysis.event_definition.cohort_selection import (
    ANALYZED_SUBSET,
    select_visit_context,
)
from continuum_swr_replay.analysis.event_definition.swr_source import event_epoch_ids
from continuum_swr_replay.analysis.event_properties.rule_sweep import (
    RULE,
    SETTINGS,
    epoch_sweep_counts,
)
from continuum_swr_replay.analysis.event_tables.prepare_data import (
    session_cache_record,
    well_visit_times,
)
from continuum_swr_replay.paths import (
    EVENT_RULE_SWEEP_DIR,
    event_features_root,
    processed_data_dir,
    replay_content_dir,
    table_path,
)
from continuum_swr_replay.runs.io import (
    ok_epochs,
    read_manifest,
    sha256_file,
    write_json,
)
from continuum_swr_replay.runs.provenance import described_commit, produced_by


def _epoch(job: tuple[str, str]) -> pd.DataFrame:
    """One epoch's counts at every setting, from the decode the run recorded."""
    identifier, expected_sha256 = job
    raw_bytes = Path(inventory_run.decode_path(identifier)).read_bytes()
    if (sha256 := hashlib.sha256(raw_bytes).hexdigest()) != expected_sha256:
        raise ValueError(
            f"{identifier}: decode sha256 {sha256} is not the run's {expected_sha256}"
        )
    raw = pickle.loads(raw_bytes)
    animal, day, epoch = (raw[c].iloc[0] for c in ("animal", "day", "epoch"))
    counts = epoch_sweep_counts(raw, *well_visit_times(identifier))
    counts.insert(0, "epoch", str(epoch))
    counts.insert(0, "session_date", int(day))
    counts.insert(0, "animal", str(animal))
    counts.insert(0, "epoch_id", identifier)
    return counts


def check_rule_counts(counts: pd.DataFrame, events: pd.DataFrame) -> None:
    """Refuse counts whose events at the rule's settings are not the canonical
    table's, epoch by epoch (the events that start inside a well visit, and
    those with no SWR).

    Raises
    ------
    ValueError
        If an epoch's counts differ.
    """
    rule = counts[
        (counts["setting"] == "probability_threshold")
        & (counts["value"] == RULE["probability_threshold"])
    ].set_index("epoch_id")[["n_events", "n_no_swr"]]
    epochs = event_epoch_ids(events)
    table = pd.DataFrame(
        {
            "n_events": events.groupby(epochs.to_numpy()).size(),
            "n_no_swr": (~events["is_swr"].astype(bool))
            .groupby(epochs.to_numpy())
            .sum(),
        }
    ).reindex(rule.index, fill_value=0)
    differ = rule.index[(rule != table).any(axis=1)]
    if len(differ):
        raise ValueError(
            f"at the rule's settings, {len(differ)} epochs' counts are not the "
            f"canonical event table's: {list(differ[:5])}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--event-features-run", required=True)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument(
        "--epochs", nargs="+", help="only these epochs (a smoke test; default: all)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=f"default: Processed-Data/{EVENT_RULE_SWEEP_DIR}",
    )
    args = parser.parse_args(argv)
    manifest = read_manifest(event_features_root() / args.event_features_run)
    jobs = [
        (identifier, manifest["epochs"][identifier]["decode"]["sha256"])
        for identifier in ok_epochs(manifest)
        if args.epochs is None or identifier in args.epochs
    ]
    event_table = table_path("event_table", replay_content_dir())
    events = select_visit_context(read_swr_table(event_table))
    with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
        table = pd.concat(pool.map(_epoch, jobs), ignore_index=True)
    check_rule_counts(table, events)
    out_dir = args.output_dir or processed_data_dir() / EVENT_RULE_SWEEP_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "counts.parquet.tmp"
    table.to_parquet(tmp, index=False)
    tmp.replace(out_dir / "counts.parquet")
    write_json(
        out_dir / "manifest.json",
        {
            "produced_by": produced_by(__file__),
            "commit": described_commit(),
            "event_features_run": args.event_features_run,
            "subset": ANALYZED_SUBSET,
            "session_cache": session_cache_record(identifier for identifier, _ in jobs),
            "inputs": {
                "event_table": {
                    "path": str(event_table),
                    "sha256": sha256_file(event_table),
                }
            },
            "n_epochs": len(jobs),
            "settings": {name: list(values) for name, values in SETTINGS.items()},
            "rule": RULE,
            "checks": {"rule_counts_equal_event_table": True},
        },
        indent=1,
        final_newline=True,
    )
    print(f"wrote {len(jobs)} epochs to {out_dir / 'counts.parquet'}")


if __name__ == "__main__":
    main()
