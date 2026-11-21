#!/usr/bin/env python
"""Tetrode participation and spike rate of each non-local event, against Local rest.

For every canonical immobile non-local event (``replay_content/event_table.parquet``,
read through the SWR source) and one control window of the same length drawn from
Local rest inside one well visit of the same epoch
(``analysis.event_evidence.participation``: every bin of the window is Local rest
and in the same visit, the visits read from the session cache only,
``prepare_data.well_visit_times``), count each HPC tetrode's spikes and write:

- ``{epoch_id}_participation.parquet`` per epoch: one row per window (``kind`` ``"event"`` or
  ``"control"``), with ``event_id``, ``group`` (``"SWR"`` or ``"no SWR"``, the
  event's), ``n_bins``, ``duration_s``, ``n_spikes``, ``n_active_tetrodes``,
  ``n_tetrodes``, ``participation`` and ``spike_rate_hz``;
- ``{epoch_id}_participation.json`` per epoch: its record (status, the run's
  settings and inputs, the code's provenance), and ``manifest.json`` gathered
  from every epoch's record (``runs.epoch_records``); an epoch with an ``ok``
  record of the same settings and inputs is not run again.

Every event of the epoch has its ``"event"`` row, whatever its well-visit label
(the readers select the events); the control rule is the same for all of them.
An event with no window of its length inside one visit has no ``"control"`` row:
each epoch's record counts those per group (``n_without_control``) and, beside
them, the events Local rest anywhere in the epoch would leave without one
(``n_without_control_any_local_rest``). The settings record the subset
(``cohort_selection.ANALYZED_SUBSET``) and the control rule
(``participation.CONTROL_RULE``); the inputs, the event table's SHA-256 and the
session-cache entries' (``prepare_data.session_cache_record``, over every epoch
of the event table).

Spike times are read from the lab's database through ``load_data`` (read only).

Usage::

    uv run --frozen python scripts/event_evidence/measure_event_participation.py \\
        --epochs wilbur20210329_02_r1_HPC
    uv run --frozen python scripts/event_evidence/measure_event_participation.py \\
        --n-workers 8
"""

from __future__ import annotations

import argparse
import sys
import time as clock
from pathlib import Path

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.event_definition.analyzed_time import (
    immobile_analyzed_bins,
    read_dot_with_swr_source,
    read_swr_table,
)
from continuum_swr_replay.analysis.event_definition.cohort_selection import (
    ANALYZED_SUBSET,
)
from continuum_swr_replay.analysis.event_definition.swr_source import (
    event_epoch_ids,
)
from continuum_swr_replay.analysis.event_definition.well_visits import (
    well_visit_index,
)
from continuum_swr_replay.analysis.event_evidence import participation as pt
from continuum_swr_replay.analysis.event_evidence.state_evidence.windows import (
    SWR_LABELS,
    event_bins,
)
from continuum_swr_replay.analysis.event_tables.prepare_data import (
    session_cache_record,
    well_visit_times,
)
from continuum_swr_replay.decoding.files import decode_paths
from continuum_swr_replay.decoding.identifiers import parse_epoch_identifier
from continuum_swr_replay.intervals import sample_step
from continuum_swr_replay.parameters import ROOT_DIR
from continuum_swr_replay.paths import (
    EVENT_PARTICIPATION_DIR,
    PARTICIPATION_NAME,
    processed_data_dir,
)
from continuum_swr_replay.runs.epoch_records import gather_manifest
from continuum_swr_replay.runs.io import output_path
from continuum_swr_replay.runs.provenance import produced_by, provenance_record
from continuum_swr_replay.tables.io import write_table

# Sibling-script import: run as a script, only this file's directory is on
# sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.event_evidence._per_event_run import (  # noqa: E402
    add_run_arguments,
    event_table_path,
    file_input,
    run,
    select_epochs,
)

PRODUCED_BY = produced_by(__file__)
SEED = 20261118
"""The control windows' seed, with the epoch id."""


def epoch_events(events: pd.DataFrame, epoch_id: str) -> pd.DataFrame:
    """The table's events of one epoch, by start."""
    mine = events[event_epoch_ids(events) == epoch_id]
    return mine.sort_values("event_start_s").reset_index(drop=True)


def load_spike_times(epoch_id: str) -> list[np.ndarray]:
    """Each HPC tetrode's sorted spike times (s), from ``load_data`` (read only)."""
    from continuum_swr_replay.data_loaders import load_data
    from continuum_swr_replay.data_types import SessionDataKeys as K

    key = parse_epoch_identifier(epoch_id)
    data = load_data(
        key.nwb_file_name, key.epoch_name, include=frozenset({K.SPIKE_TIMES})
    )
    return [np.sort(np.asarray(t)) for t in data["spike_times"][key.brain_area]]


def epoch_table(
    epoch_id: str, events: pd.DataFrame, data_dir: Path
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """One epoch's windows: its events and their Local-rest controls inside a visit.

    Returns
    -------
    table : pd.DataFrame
        One row per window, the events first, then the controls.
    without_control : dict
        Per group (``"SWR"``, ``"no SWR"``), the events with no control
        (``n_without_control``) and those Local rest anywhere in the epoch would
        leave without a window (``n_without_control_any_local_rest``).
    """
    frame = read_dot_with_swr_source(decode_paths(epoch_id, data_dir).data_over_time)
    time = frame.index.to_numpy(dtype=float)
    bin_s = sample_step(time)  # for the margin only
    in_event = (frame["SWR_event"].to_numpy() > 0) | (
        frame["non_local_event"].to_numpy() > 0
    )
    local_rest = pt.local_rest_mask(
        frame["Local_prob"].to_numpy(),
        immobile_analyzed_bins(frame),
        in_event,
        margin_bins=int(round(pt.MARGIN_S / bin_s)),
    )
    s, e = event_bins(
        time, events["event_start_s"].to_numpy(), events["event_end_s"].to_numpy()
    )
    visit_index = well_visit_index(time, *well_visit_times(epoch_id))
    seed = (SEED, epoch_id)
    controls = pt.control_windows(s, e, local_rest, visit_index, seed=seed)
    anywhere = pt.windows_in_local_rest_anywhere(s, e, local_rest, seed=seed)
    spike_times = load_spike_times(epoch_id)
    group = np.where(
        events["is_swr"].to_numpy(dtype=bool), SWR_LABELS[True], SWR_LABELS[False]
    )
    has_control = controls["s"].to_numpy() >= 0
    without_control = {
        "n_without_control": _count_by_group(group, ~has_control),
        "n_without_control_any_local_rest": _count_by_group(
            group, anywhere["s"].to_numpy() < 0
        ),
    }
    parts = []
    for kind, keep, first, last in (
        ("event", np.ones(len(events), dtype=bool), s, e),
        ("control", has_control, controls["s"].to_numpy(), controls["e"].to_numpy()),
    ):
        first, last = first[keep], last[keep]
        start, stop = time[first], pt.window_stops(time, last)
        counts = pt.tetrode_spike_counts(spike_times, start, stop)
        measures = pt.participation_and_rate(counts, stop - start)
        parts.append(
            measures.assign(
                event_id=events["event_id"].to_numpy()[keep],
                animal=events["animal"].to_numpy()[keep],
                epoch_id=epoch_id,
                group=group[keep],
                kind=kind,
                n_bins=last - first + 1,
                start_s=start,
                duration_s=stop - start,
            )
        )
    return pd.concat(parts, ignore_index=True), without_control


def _count_by_group(group: np.ndarray, flagged: np.ndarray) -> dict[str, int]:
    """How many flagged events each group has (both groups, zero included)."""
    return {
        label: int((flagged & (group == label)).sum()) for label in SWR_LABELS.values()
    }


def run_epoch(epoch_id: str, event_table: Path, data_dir: Path, output: Path) -> dict:
    """Measure one epoch and write its table; return its status record."""
    started = clock.time()
    events = epoch_events(read_swr_table(event_table), epoch_id)
    table, without_control = epoch_table(epoch_id, events, data_dir)
    write_table(table, output_path(epoch_id, output, PARTICIPATION_NAME))
    return {
        "status": "ok",
        "n_events": len(events),
        **without_control,
        "seconds": round(clock.time() - started, 1),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_run_arguments(parser)
    parser.add_argument("--n-workers", type=int, default=1)
    args = parser.parse_args(argv)

    data_dir = args.data_dir or processed_data_dir()
    output = args.output_dir or data_dir / EVENT_PARTICIPATION_DIR
    event_table = event_table_path(data_dir)
    every_epoch = sorted(set(event_epoch_ids(read_swr_table(event_table))))
    header = {
        "produced_by": PRODUCED_BY,
        "provenance": provenance_record(ROOT_DIR),
        "settings": {
            "seed": SEED,
            "local_threshold": pt.LOCAL_THRESHOLD,
            "margin_s": pt.MARGIN_S,
            "subset": ANALYZED_SUBSET,
            "control_rule": pt.CONTROL_RULE,
        },
        "inputs": {
            "event_table": file_input(event_table),
            "session_cache": session_cache_record(every_epoch),
        },
    }
    if args.gather:
        gather_manifest(output, PARTICIPATION_NAME, header)
        return
    run(
        epochs=select_epochs(args.epochs or every_epoch, args.shards, args.n_shards),
        task=run_epoch,
        task_args=(event_table, data_dir, output),
        output=output,
        name=PARTICIPATION_NAME,
        header=header,
        overwrite=args.overwrite,
        max_workers=args.n_workers,
    )


if __name__ == "__main__":
    main()
