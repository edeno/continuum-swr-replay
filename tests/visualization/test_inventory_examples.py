"""The inventory examples' run and figure: a draw whose figure cannot be made is
recorded as not written and the run goes on, and the figure colors detectors by
the paper's families."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import scripts.figures.generate_inventory_examples as examples
from continuum_swr_replay.analysis.detector_comparison import inventory_run
from continuum_swr_replay.analysis.detector_comparison.selection import draw_examples
from continuum_swr_replay.runs.io import output_path
from continuum_swr_replay.tables.io import write_table
from continuum_swr_replay.visualization import inventory_examples
from tests.support._inventory_cells import cells


def test_a_draw_that_cannot_be_made_is_recorded_and_the_run_goes_on(
    tmp_path, monkeypatch
) -> None:
    """In epoch one, one draw's recomputed windows are not the stored ones
    (``draw_example`` itself meets the recomputation's refusal); in epoch two,
    the epoch's inputs cannot be read. Their draws are recorded as not
    written, with the reason, and every other draw is written."""
    run_dir, figure_dir = tmp_path / "run01", tmp_path / "figures"
    run_dir.mkdir()
    epochs = {"sim20200101_02_r1_HPC": "a", "sim20200101_04_r2_HPC": "b"}
    cell = "exactly_one_detector"
    for identifier, animal in epochs.items():
        table = cells([(cell, "kay", animal, "unmatched", 3)]).assign(
            epoch=identifier.split("_", 1)[1].removesuffix("_HPC")
        )
        write_table(table, output_path(identifier, run_dir, "cells"))
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "settings": {
                    "input_runs": {"memoryless": "memorylessrun01"},
                    "cells": {cell: {"against": ["kay", "state_space"]}},
                },
                "epochs": dict.fromkeys(epochs, {"status": "ok"}),
            }
        )
    )
    closed = []

    def load_epoch(identifier, *, run_dir, memoryless_dir):
        if identifier == "sim20200101_04_r2_HPC":
            raise ValueError("The band holds no tetrode the SWR source kept.")
        return SimpleNamespace(
            results=SimpleNamespace(close=lambda: closed.append(1)),
            inventory=pd.DataFrame(
                {
                    "event_key": [disagreeing],
                    "event_start_s": [1.0],
                    "event_end_s": [1.1],
                }
            ),
            candidates=SimpleNamespace(midpoint_s=np.arange(0.0, 3.0, 0.02)),
        )

    def recompute_windows(inputs, first, last):
        raise ValueError(
            f"windows {first} to {last}, recomputed, are not the stored ones"
        )

    # the first draw, of epoch one, disagrees; an earlier run left its figure
    first = draw_examples(
        pd.concat(
            pd.read_parquet(output_path(identifier, run_dir, "cells"))
            for identifier in epochs
        )
    ).iloc[0]
    disagreeing = first.event_key
    stale = figure_dir / inventory_examples.figure_name(first)
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"an earlier run's figure")

    real_draw_example = inventory_examples.draw_example

    def draw_example(inputs, drawn, *, against, path, run_id):
        if drawn.event_key == disagreeing:
            return real_draw_example(
                inputs, drawn, against=against, path=path, run_id=run_id
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")
        return {**drawn.to_dict(), "against": against, "path": str(path)}

    monkeypatch.setattr(examples, "load_epoch", load_epoch)
    monkeypatch.setattr(examples, "draw_example", draw_example)
    # the real draw_example recomputes through the library's recompute_windows
    monkeypatch.setattr(inventory_examples, "recompute_windows", recompute_windows)

    records = examples.write_examples(run_dir, figure_dir)

    status = {r["event_key"]: (r["status"], r.get("reason", "")) for r in records}
    assert len(records) == 4 and closed == [1]
    assert status[disagreeing] == (
        "not_written",
        # the plotted windows: midpoints within PAD_S of 1.0-1.1 s
        "windows 25 to 80, recomputed, are not the stored ones",
    )
    failed_epoch = [r for r in records if r["animal"] == "b"]
    assert len(failed_epoch) == 2
    assert {r["status"] for r in failed_epoch} == {"not_written"}
    assert {r["reason"] for r in failed_epoch} == {
        "The band holds no tetrode the SWR source kept."
    }
    written = [r for r in records if r["status"] == "written"]
    assert len(written) == 1 and Path(written[0]["path"]).exists()
    assert not stale.exists()
    manifest = json.loads((figure_dir / "manifest.json").read_text())
    assert [r["status"] for r in manifest["figures"]] == [r["status"] for r in records]
    captions = (figure_dir / "captions.md").read_text()
    assert captions.count("(not written: ") == 3


def test_detector_colors_follow_the_paper_families():
    """Every SWR definition shares COLORS["SWR"]; MUA, state-space and the two
    memoryless inventories each take their own family color."""
    from continuum_swr_replay.visualization.palettes import COLORS

    colors = {d: inventory_examples.detector_color(d) for d in inventory_run.DETECTORS}
    assert {colors[d] for d in inventory_run.SWR_DEFINITIONS} == {COLORS["SWR"]}
    assert colors["mua"] == COLORS["MUA"]
    assert colors["state_space"] == COLORS["Non-Local"]
    assert colors["memoryless"] == colors["memoryless_criteria"]
    assert colors["memoryless"] not in COLORS.values()
