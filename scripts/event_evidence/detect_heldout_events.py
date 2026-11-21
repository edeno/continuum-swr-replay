#!/usr/bin/env python
"""Non-local events detected from held-out data, for Figure 2's panel g.

For each epoch of the restriction run (``score_nonlocal_restriction.py``, the
fold fit; its ``ok`` records): read its per-bin table, the epoch's decode frame
as the decoder saved it, the per-bin facts (``measure_bin_facts.py``) and the
canonical event table; check that the three time axes are one; and apply the
canonical event rule (``heldout_events.labeled_frame`` and
``heldout_events.event_rows``) to the ``full`` model's held-out smoothed
non-local probabilities. Each fold's bins were scored under the decoder fit on
the other folds, so these events are chosen without the decoder having seen
their bins. A trial segment (the run to a well and the whole stay there) longer
than ``tetrode_holdout.MAX_SEGMENT_S`` (30 s) is cut into pieces of at most
30 s that can fall in different folds, so a small share of the events span two
folds: their probability trace and per-bin scores come from two fits, each bin
still scored under a fit that did not see it. Then sum the restriction scores over each event
(``restriction_windows.window_table``),
say which held-out and which canonical events share a bin
(``heldout_events.shared_bin_events``), and sum the epoch's bins by category,
well visit and held-out event window (``restriction_totals.restriction_time_sums``).
CPU only: pandas and numpy, nothing is decoded or refit.

The epoch fails (status ``failed``, with the reason) unless the same rule
applied to the canonical decode's own non-local probabilities gives the decode
frame's canonical labels as read through the SWR source exactly, unless
those labels are the event table's events of the epoch, and unless, for each of
those events, the rule's SWR group is the table's ``is_swr`` and its start
inside a well visit is what ``cohort_selection.select_visit_context`` gives.

``--restriction-dir`` reads another restriction run and ``--output-dir`` writes
elsewhere; by default the canonical restriction run and
``EVENT_NONLOCAL_RESTRICTION_HELDOUT_EVENTS_DIR``.

Writes, under ``processed_data_dir() / EVENT_NONLOCAL_RESTRICTION_HELDOUT_EVENTS_DIR``:

- ``{epoch_id}_heldout_events.parquet``: one row per held-out event,
  ``epoch_id``, ``animal``, ``heldout_event_id`` (``{epoch_id}:{label}``, unique
  over the cohort), ``label``, ``kind`` (``"event"``), ``group`` (``"SWR"`` or
  ``"no SWR"``, by strict overlap with an SWR of the source),
  ``event_start_s``, ``event_end_s``, ``starts_in_well_visit`` (its first bin is
  inside a well visit, by the per-bin facts), ``shares_bin_with_canonical``
  (a canonical event of the epoch, of any well-visit label, holds one of its
  bins), then the columns of the window table of
  ``sum_nonlocal_restriction_windows.py`` (``s``, ``e``, the bins, the six
  summed scores whole and over the bins outside SWRs, ``n_spikes``, and the
  columns from the per-bin facts, ``restriction_windows.BIN_FACT_COLUMNS``);
- ``{epoch_id}_canonical_events_found.parquet``: one row per canonical event of
  the epoch, ``epoch_id``, ``animal``, ``event_id`` (the event table's),
  ``label``, ``group`` (the table's ``is_swr``) and ``shares_bin_with_heldout``
  (a held-out event, of any well-visit start, holds one of its bins);
- ``{epoch_id}_heldout_time_sums.parquet``: the epoch's bins, spikes and six
  summed scores by ``category``, ``in_well_visit`` and ``in_event_window``
  (inside a held-out event of either group, wherever it starts;
  ``restriction_totals.restriction_time_sums``), with ``epoch_id`` and
  ``animal``;
- ``{epoch_id}_heldout_events.json``: the epoch's record (events by group and
  well visit, events found again, the time step, the digests of the per-bin
  table, the per-bin facts and the decode frame, provenance);

and ``manifest.json`` gathered from the records. The header records the digests
of the event table and of the restriction and per-bin facts runs' manifests.

Usage::

    uv run --frozen python scripts/event_evidence/detect_heldout_events.py \\
        --epochs wilbur20210329_02_r1_HPC
    uv run --frozen python scripts/event_evidence/detect_heldout_events.py \\
        --workers 8
    uv run --frozen python scripts/event_evidence/detect_heldout_events.py --gather
"""

from __future__ import annotations

import argparse
import sys
import time as clock
from pathlib import Path

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.event_definition.analyzed_time import (
    read_swr_table,
    with_swr_source,
)
from continuum_swr_replay.analysis.event_definition.cohort_selection import (
    select_visit_context,
)
from continuum_swr_replay.analysis.event_definition.nonlocal_event_rule import (
    NON_LOCAL_STATES,
)
from continuum_swr_replay.analysis.event_definition.swr_source import event_epoch_ids
from continuum_swr_replay.analysis.event_evidence import heldout_events as he
from continuum_swr_replay.analysis.event_evidence import restriction_totals as rt
from continuum_swr_replay.analysis.event_evidence import restriction_windows as rw
from continuum_swr_replay.analysis.event_evidence.nonlocal_restriction import (
    STATE_PROBABILITY_COLUMNS,
)
from continuum_swr_replay.analysis.event_evidence.state_evidence.windows import (
    SWR_LABELS,
)
from continuum_swr_replay.decoding.files import decode_paths
from continuum_swr_replay.decoding.identifiers import parse_epoch_identifier
from continuum_swr_replay.intervals import sample_step
from continuum_swr_replay.parameters import DEFAULT_NON_LOCAL_THRESHOLD, ROOT_DIR
from continuum_swr_replay.paths import (
    BIN_FACTS_NAME,
    CANONICAL_EVENTS_FOUND_NAME,
    EVENT_BIN_FACTS_DIR,
    EVENT_NONLOCAL_RESTRICTION_DIR,
    EVENT_NONLOCAL_RESTRICTION_HELDOUT_EVENTS_DIR,
    HELDOUT_EVENTS_NAME,
    HELDOUT_TIME_SUMS_NAME,
    NONLOCAL_RESTRICTION_BINS_NAME,
    processed_data_dir,
)
from continuum_swr_replay.runs.epoch_records import gather_manifest
from continuum_swr_replay.runs.io import (
    json_ready,
    ok_epochs,
    output_path,
    read_manifest,
)
from continuum_swr_replay.runs.provenance import produced_by, provenance_record
from continuum_swr_replay.tables.io import write_table

# Sibling-script import: run as a script, only this file's directory is on
# sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.event_evidence._per_event_run import (  # noqa: E402
    add_cpu_arguments,
    add_restriction_dir_argument,
    add_run_arguments,
    cpu_pool,
    event_table_path,
    file_input,
    restriction_run_dir,
    run,
    run_input,
    select_epochs,
)

PRODUCED_BY = produced_by(__file__)
HELDOUT_PROBABILITIES = tuple(STATE_PROBABILITY_COLUMNS[s] for s in NON_LOCAL_STATES)
"""The per-bin table's held-out P(Non-Local Continuous) and P(Non-Local
Fragmented) of the ``full`` model."""


def read_raw_frame(epoch_id: str, data_dir: Path) -> pd.DataFrame:
    """The epoch's decode frame as the decoder saved it (not read through the
    SWR source)."""
    return pd.read_pickle(decode_paths(epoch_id, data_dir).data_over_time)


def run_epoch(
    epoch_id: str,
    event_table: Path,
    data_dir: Path,
    output: Path,
    restriction_dir: Path | None = None,
) -> dict:
    """Detect one epoch's held-out events, write its tables and return its record.

    ``restriction_dir`` is the restriction run read, by default
    ``data_dir / EVENT_NONLOCAL_RESTRICTION_DIR``.

    Raises
    ------
    ValueError
        If the per-bin table's or the per-bin facts' time is not the decode
        frame's, the rule on the canonical decode's probabilities does not give
        its canonical labels, those labels are not the event table's events
        of the epoch, or the rule gives one of them another SWR group than the
        table's ``is_swr`` or another start inside a well visit than
        ``cohort_selection.select_visit_context``.
    """
    started = clock.time()
    restriction_dir = restriction_dir or data_dir / EVENT_NONLOCAL_RESTRICTION_DIR
    bins_path = output_path(epoch_id, restriction_dir, NONLOCAL_RESTRICTION_BINS_NAME)
    facts_path = output_path(epoch_id, data_dir / EVENT_BIN_FACTS_DIR, BIN_FACTS_NAME)
    decode_path = decode_paths(epoch_id, data_dir).data_over_time
    bins = pd.read_parquet(bins_path)
    facts = pd.read_parquet(facts_path)
    raw = read_raw_frame(epoch_id, data_dir)
    time = raw.index.to_numpy(dtype=float)
    for name, table in (("per-bin table's", bins), ("per-bin facts'", facts)):
        if not np.array_equal(table["time"].to_numpy(dtype=float), time):
            raise ValueError(f"the {name} time is not the decode frame's")

    # the canonical labels as every analysis reads them, and the rule's check
    canonical = with_swr_source(raw)["non_local_event"].to_numpy()
    again_frame = he.labeled_frame(
        raw, *(raw[column].to_numpy() for column in he.PROBABILITY_COLUMNS)
    )
    again = again_frame["non_local_event"].to_numpy()
    if not np.array_equal(again, canonical):
        raise ValueError(
            f"the rule on the canonical decode's probabilities gives other labels "
            f"than its canonical ones at {int((again != canonical).sum())} bins"
        )
    table = read_swr_table(event_table)
    table = table[event_epoch_ids(table) == epoch_id]
    labels = np.unique(canonical[canonical > 0])
    if not np.array_equal(np.sort(table["non_local_event"].to_numpy()), labels):
        raise ValueError(
            "the decode frame's canonical labels are not the event table's events "
            "of the epoch"
        )
    # the rule's SWR group and start inside a well visit of each canonical event
    in_visit = facts["in_well_visit"].to_numpy(dtype=bool)
    rule = (
        he.event_rows(again_frame, in_visit)
        .set_index("label")
        .loc[table["non_local_event"].to_numpy()]
    )
    selected = table["event_id"].isin(select_visit_context(table)["event_id"])
    for name, theirs, ours in (
        (
            "SWR group than the event table's is_swr",
            table["is_swr"].to_numpy(dtype=bool),
            rule["group"].to_numpy() == SWR_LABELS[True],
        ),
        (
            "start inside a well visit than select_visit_context",
            selected.to_numpy(),
            rule["starts_in_well_visit"].to_numpy(dtype=bool),
        ),
    ):
        if (theirs != ours).any():
            raise ValueError(
                f"the rule on the canonical decode's probabilities gives another "
                f"{name} for {int((theirs != ours).sum())} of the epoch's events"
            )

    # the held-out events
    frame = he.labeled_frame(
        raw, *(bins[column].to_numpy() for column in HELDOUT_PROBABILITIES)
    )
    heldout = frame["non_local_event"].to_numpy()
    events = he.event_rows(frame, in_visit)
    animal = parse_epoch_identifier(epoch_id).animal
    windows = events.assign(
        window_id=np.arange(len(events)),
        event_id=[f"{epoch_id}:{label}" for label in events["label"]],
        kind="event",
    )
    sums = rw.window_table(bins, facts, windows)
    found = he.shared_bin_events(heldout, canonical)
    rows = pd.concat(
        [
            pd.DataFrame(
                {
                    "epoch_id": epoch_id,
                    "animal": animal,
                    "heldout_event_id": windows["event_id"].to_numpy(),
                    "label": events["label"].to_numpy(),
                    "kind": "event",
                    "group": events["group"].to_numpy(),
                    "event_start_s": events["event_start_s"].to_numpy(),
                    "event_end_s": events["event_end_s"].to_numpy(),
                    "starts_in_well_visit": events["starts_in_well_visit"].to_numpy(),
                    "shares_bin_with_canonical": found.reindex(
                        events["label"].to_numpy()
                    ).to_numpy(dtype=bool),
                }
            ),
            sums.drop(columns=list(rw.WINDOW_KEYS)),
        ],
        axis=1,
    )
    write_table(rows, output_path(epoch_id, output, HELDOUT_EVENTS_NAME))

    # the canonical events found again
    found_canonical = he.shared_bin_events(canonical, heldout)
    found_table = pd.DataFrame(
        {
            "epoch_id": epoch_id,
            "animal": animal,
            "event_id": table["event_id"].to_numpy(),
            "label": table["non_local_event"].to_numpy(),
            "group": np.where(
                table["is_swr"].to_numpy(dtype=bool),
                SWR_LABELS[True],
                SWR_LABELS[False],
            ),
            "shares_bin_with_heldout": found_canonical.reindex(
                table["non_local_event"].to_numpy()
            ).to_numpy(dtype=bool),
        }
    )
    write_table(found_table, output_path(epoch_id, output, CANONICAL_EVENTS_FOUND_NAME))

    # the epoch's time, with the held-out events' windows as the event windows
    time_sums = rt.restriction_time_sums(bins, facts, windows)
    time_sums.insert(0, "animal", animal)
    time_sums.insert(0, "epoch_id", epoch_id)
    write_table(time_sums, output_path(epoch_id, output, HELDOUT_TIME_SUMS_NAME))

    return json_ready(
        {
            "status": "ok",
            "n_heldout_events": len(rows),
            "n_heldout_events_by_group_start": {
                f"{group}, {'in' if inside else 'outside'} a visit": int(n)
                for (group, inside), n in rows.groupby(
                    ["group", "starts_in_well_visit"]
                )
                .size()
                .items()
            },
            "n_heldout_events_sharing_a_bin": int(
                rows["shares_bin_with_canonical"].sum()
            ),
            "n_canonical_events": len(found_table),
            "n_canonical_events_sharing_a_bin": int(
                found_table["shares_bin_with_heldout"].sum()
            ),
            "n_events_without_bins_outside_swr": int(
                (rows["n_bins_outside_swr"] == 0).sum()
            ),
            "bin_s": sample_step(time),
            "epoch_inputs": {
                "bins": file_input(bins_path),
                "bin_facts": file_input(facts_path),
                "data_over_time": file_input(Path(decode_path)),
            },
            "seconds": round(clock.time() - started, 1),
        }
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_run_arguments(parser)
    add_cpu_arguments(parser, workers=8)
    add_restriction_dir_argument(parser)
    args = parser.parse_args(argv)

    data_dir = args.data_dir or processed_data_dir()
    output = args.output_dir or data_dir / EVENT_NONLOCAL_RESTRICTION_HELDOUT_EVENTS_DIR
    event_table = event_table_path(data_dir)
    restriction = restriction_run_dir(args, data_dir)
    header = {
        "produced_by": PRODUCED_BY,
        "provenance": provenance_record(ROOT_DIR),
        "settings": {
            "probabilities": list(HELDOUT_PROBABILITIES),
            "nonlocal_threshold": DEFAULT_NON_LOCAL_THRESHOLD,
        },
        "inputs": {
            "event_table": file_input(event_table),
            "nonlocal_restriction": run_input(restriction),
            "bin_facts": run_input(data_dir / EVENT_BIN_FACTS_DIR),
        },
    }
    if args.gather:
        gather_manifest(output, HELDOUT_EVENTS_NAME, header)
        return
    max_workers, pool_options = cpu_pool(args)
    run(
        epochs=select_epochs(
            args.epochs or ok_epochs(read_manifest(restriction)),
            args.shards,
            args.n_shards,
        ),
        task=run_epoch,
        task_args=(event_table, data_dir, output, restriction),
        output=output,
        name=HELDOUT_EVENTS_NAME,
        header=header,
        overwrite=args.overwrite,
        max_workers=max_workers,
        pool_options=pool_options,
    )


if __name__ == "__main__":
    main()
