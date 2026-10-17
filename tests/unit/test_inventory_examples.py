"""The examples' sampler: which events of each cell are drawn for a figure.

For each cell, source detector and animal the sampler draws up to two
``unmatched`` events, and as counterexamples up to two ``matched`` events of
the same cell, detector and animal, by a generator seeded from the run seed
and that triple. ``unknown`` events are never drawn. A draw whose figure cannot
be made is recorded as not written, and the run goes on.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import scripts.figures.generate_inventory_examples as examples
import scripts.qa.run_inventory_comparison as runner
from continuum_swr_replay.table_io import write_table

SEED = 7


def cells(records: list[tuple[str, str, str, str, int]]) -> pd.DataFrame:
    """A ``cells`` table from (cell, detector, animal, membership, count)."""
    rows = []
    for cell, detector, animal, membership, count in records:
        for number in range(count):
            rows.append(
                {
                    "cell": cell,
                    "animal": animal,
                    "day": 20200101,
                    "epoch": "02_r1",
                    "detector": detector,
                    "event_key": f"{detector}:{animal}20200101_02_r1:{membership}{number}",
                    "n_against": 7,
                    "membership": membership,
                }
            )
    dtypes = runner.TABLE_DTYPES["cells"]
    return pd.DataFrame(rows, columns=list(dtypes)).astype(dict(dtypes))


LARGE = cells(
    [
        ("exactly_one_detector", "kay", "a", "unmatched", 30),
        ("exactly_one_detector", "kay", "a", "matched", 30),
        ("exactly_one_detector", "kay", "a", "unknown", 30),
        ("exactly_one_detector", "kay", "b", "unmatched", 30),
        ("exactly_one_detector", "kay", "b", "matched", 30),
        ("swr_matched_by_no_decoder", "kay", "a", "unmatched", 30),
        ("swr_matched_by_no_decoder", "kay", "a", "matched", 30),
        ("swr_matched_by_no_decoder", "yu", "a", "unmatched", 30),
        ("swr_matched_by_no_decoder", "yu", "a", "matched", 30),
    ]
)
UNIT = ["cell", "detector", "animal"]


def test_the_draw_is_deterministic_for_a_seed() -> None:
    first = examples.draw_examples(LARGE, seed=SEED)
    again = examples.draw_examples(LARGE, seed=SEED)
    pd.testing.assert_frame_equal(first, again)
    # the input's row order does not change what is drawn
    shuffled = LARGE.sample(frac=1.0, random_state=3).reset_index(drop=True)
    pd.testing.assert_frame_equal(examples.draw_examples(shuffled, seed=SEED), first)
    # another seed draws other events from these 30-event pools
    other = examples.draw_examples(LARGE, seed=SEED + 1)
    assert set(other.event_key) != set(first.event_key)


def test_each_unit_draws_two_unmatched_and_two_matched() -> None:
    drawn = examples.draw_examples(LARGE, seed=SEED)
    sizes = drawn.groupby([*UNIT, "membership"]).size()
    assert (sizes == 2).all()
    assert len(sizes) == 2 * len(LARGE[UNIT].drop_duplicates())
    roles = dict(zip(drawn.membership, drawn.role, strict=True))
    assert roles == {"unmatched": "example", "matched": "counterexample"}
    assert not drawn.duplicated("event_key").any()


def test_unknown_events_are_never_drawn() -> None:
    only_unknown = cells(
        [
            ("exactly_one_detector", "mua", "a", "unknown", 50),
            ("exactly_one_detector", "mua", "a", "unmatched", 1),
        ]
    )
    for seed in range(20):
        drawn = examples.draw_examples(only_unknown, seed=seed)
        assert set(drawn.membership) == {"unmatched"}
        assert len(drawn) == 1
    assert (
        examples.draw_examples(LARGE, seed=SEED)
        .membership.isin(["unmatched", "matched"])
        .all()
    )


def test_a_small_cell_gives_fewer() -> None:
    small = cells(
        [
            ("exactly_one_detector", "carey", "a", "matched", 1),
            ("exactly_one_detector", "carey", "a", "unknown", 5),
            ("sequence_matched_by_no_trigger", "memoryless", "a", "unmatched", 1),
            ("sequence_matched_by_no_trigger", "memoryless", "a", "matched", 3),
        ]
    )
    drawn = examples.draw_examples(small, seed=SEED)
    counts = drawn.groupby(["cell", "membership"]).size().to_dict()
    assert counts == {
        ("exactly_one_detector", "matched"): 1,
        ("sequence_matched_by_no_trigger", "matched"): 2,
        ("sequence_matched_by_no_trigger", "unmatched"): 1,
    }
    assert examples.draw_examples(small.iloc[:0], seed=SEED).empty


def test_one_units_draw_does_not_depend_on_the_others() -> None:
    """Each (cell, detector, animal) has its own generator."""
    alone = LARGE[(LARGE.animal == "a") & (LARGE.detector == "kay")]
    unit = (
        (LARGE.animal == "a")
        & (LARGE.detector == "kay")
        & (LARGE.cell == "exactly_one_detector")
    )
    full = examples.draw_examples(LARGE, seed=SEED)
    part = examples.draw_examples(alone, seed=SEED)
    keys = set(LARGE[unit].event_key)
    assert set(full.event_key) & keys == set(part.event_key) & keys
    assert np.isin(part.event_key, LARGE.event_key).all()


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
        write_table(table, runner.output_path(identifier, run_dir, "cells"))
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
    first = examples.draw_examples(
        pd.concat(
            pd.read_parquet(runner.output_path(identifier, run_dir, "cells"))
            for identifier in epochs
        )
    ).iloc[0]
    disagreeing = first.event_key
    stale = figure_dir / examples.figure_name(first)
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"an earlier run's figure")

    real_draw_example = examples.draw_example

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
    monkeypatch.setattr(examples, "recompute_windows", recompute_windows)

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
