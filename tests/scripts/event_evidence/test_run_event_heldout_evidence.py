"""The canonical events' held-out evidence runner, on the synthetic decode.

The decode, its SWR source and the stand-in loader are
``test_run_state_evidence``'s. The event table is made here, from the decode's
own confident Non-Local runs, as the classified table times them (first and
last labelled bin).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.event_definition.analysed_time import (
    stamp_swr_source,
)
from continuum_swr_replay.intervals import contiguous_true_runs
from continuum_swr_replay.tables.io import read_table, write_table
from scripts.event_evidence import run_event_heldout_evidence as events_runner
from scripts.event_evidence import run_state_evidence as engine
from tests.scripts.event_evidence.test_run_state_evidence import (  # noqa: F401
    EPOCH,
    TIME,
    decode_copy,
    short_blocks,
    simulated,
    source,
)


def write_event_table(decode_dir, path):
    """Every run of P(Non-Local) > 0.8 of two bins or more, as an event table."""
    dot = pd.read_pickle(decode_dir / f"{EPOCH}_80_data_over_time.pkl")
    probabilities = dot[engine.PROBABILITY_COLUMNS].to_numpy()
    runs = contiguous_true_runs(probabilities[:, 2] + probabilities[:, 3] > 0.8)
    runs = runs[runs[:, 1] - runs[:, 0] >= 2]
    assert len(runs) >= 10, "the simulated decode has too few Non-Local runs"
    table = pd.DataFrame(
        {
            "event_id": np.arange(len(runs)) + 1000,
            "animal": "aa",
            "session_date": 20200101,
            "epoch": "02_r1",
            "event_start_s": TIME[runs[:, 0]],
            "event_end_s": TIME[runs[:, 1] - 1],
            "is_swr": np.arange(len(runs)) % 2 == 0,
            "overlap_class": "non_swr",
        }
    )
    write_table(stamp_swr_source(table), path)
    return table


def test_every_kept_event_is_scored_with_its_matched_window(simulated, tmp_path):  # noqa: F811
    decode_dir = decode_copy(simulated, tmp_path)
    event_table = tmp_path / "event_table.parquet"
    events = write_event_table(decode_dir, event_table)
    output_dir = tmp_path / "out"
    settings = events_runner.run_settings(event_table)

    record = events_runner.run_epoch(
        EPOCH,
        decode_dir=decode_dir,
        output_dir=output_dir,
        event_table=event_table,
        settings=settings,
        load_inputs=lambda _epoch_id: simulated[1],
        log=lambda _message: None,
    )

    assert record["status"] == "ok", record.get("traceback")
    assert record["settings"] == json.loads(json.dumps(settings))
    counts = record["events"]
    assert sum(counts["in_table"].values()) == len(events)
    n_dropped = sum(counts["dropped"].values())
    assert n_dropped == len(counts["dropped_event_ids"])
    assert sum(counts["scored"].values()) == len(events) - n_dropped
    # the simulated animal runs between its pauses: some runs are not immobile
    assert 0 < n_dropped < len(events)
    paths = engine.output_paths(EPOCH, output_dir)
    assert json.loads(paths["validation"].read_text())["events"] == counts

    table = read_table(paths["windows"])
    bouts, matched = table[table.kind == "bout"], table[table.kind == "baseline"]
    assert set(table.kind) == {"bout", "baseline"}
    assert len(bouts) == len(matched) == len(events) - n_dropped
    assert set(bouts.event_id) == set(events.event_id) - set(
        counts["dropped_event_ids"]
    )
    assert set(matched.event_id) == set(bouts.event_id)
    # every bin of the event is scored
    by_id = events.set_index("event_id")
    assert np.allclose(TIME[bouts.s], by_id.loc[bouts.event_id, "event_start_s"])
    assert np.allclose(TIME[bouts.e], by_id.loc[bouts.event_id, "event_end_s"])
    assert (bouts.state == np.where(bouts.is_swr, "Non-Local | SWR",
                                    "Non-Local | no SWR")).all()  # fmt: skip
    assert bouts.runcv_nl_anim_mark_ps.notna().any()

    stats = events_runner.summarize(output_dir, [EPOCH])
    assert set(stats.state) <= set(events_runner.EVENT_STATES)
    assert (output_dir / events_runner.STATS_FILE).exists()
    row = stats[stats.metric == "runcv_nl_anim_mark_ps"].iloc[0]
    assert row.n == (bouts.state == row.state).sum()
    assert "diff_animal_estimate" in stats


def test_events_off_the_decode_grid_fail_the_epoch(simulated, tmp_path):  # noqa: F811
    decode_dir = decode_copy(simulated, tmp_path)
    event_table = tmp_path / "event_table.parquet"
    events = write_event_table(decode_dir, event_table)
    write_table(
        stamp_swr_source(events.assign(event_start_s=events.event_start_s + 0.001)),
        event_table,
    )
    record = events_runner.run_epoch(
        EPOCH,
        decode_dir=decode_dir,
        output_dir=tmp_path / "out",
        event_table=event_table,
        settings=events_runner.run_settings(event_table),
        load_inputs=lambda _epoch_id: simulated[1],
        log=lambda _message: None,
    )
    assert record["status"] == "failed"
    assert "decode's time bins" in record["error"]
