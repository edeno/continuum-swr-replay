"""The restriction-window runner's wiring: what ``run_epoch`` reads, checks and
writes. The decode frame's time and the event table's read are replaced at their
boundaries; the per-bin, participation and held-out tetrode tables are made by
hand under a temporary processed-data root."""

import json

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_evidence import restriction_windows as rw
from continuum_swr_replay.paths import (
    EVENT_NONLOCAL_RESTRICTION_DIR,
    EVENT_PARTICIPATION_DIR,
    EVENT_TETRODE_HOLDOUT_DIR,
)
from continuum_swr_replay.runs.io import output_path
from scripts.event_evidence import sum_nonlocal_restriction_windows as runner

EPOCH = "wilbur20210329_02_r1_HPC"
N = 20
TIME = 50.0 + 0.002 * np.arange(N)
# events x (bins 2-4, no SWR) and y (bins 8-9, SWR, bin 9 inside a source SWR);
# y's control at bins 14-15
EVENTS = pd.DataFrame(
    {"event_id": ["x", "y"], "event_start_s": TIME[[2, 8]], "event_end_s": TIME[[4, 9]]}
)
PARTICIPATION = pd.DataFrame(
    {
        "event_id": ["x", "y", "y"],
        "kind": ["event", "event", "control"],
        "group": ["no SWR", "SWR", "SWR"],
        "start_s": TIME[[2, 8, 14]],
        "n_bins": [3, 2, 2],
    }
)


def _bins() -> pd.DataFrame:
    """``none`` 0, ``swr`` 1 and ``full`` 3 nats at every bin; the two-sided
    scores twice those."""
    one = np.ones(N, dtype=np.float32)
    return pd.DataFrame(
        {
            "time": TIME,
            "none": 0 * one,
            "swr": one,
            "full": 3 * one,
            "none_two_sided": 0 * one,
            "swr_two_sided": 2 * one,
            "full_two_sided": 6 * one,
            "in_swr": np.arange(N) == 9,
            "category": pd.Categorical(
                np.where(np.arange(N) == 9, "immobile_in_swr", "immobile_outside_swr")
            ),
        }
    )


def _holdout() -> pd.DataFrame:
    """Two tetrodes per window; 2, 0 and 5 spikes in the three windows."""
    return pd.DataFrame(
        {
            "window_id": [0, 0, 1, 1, 2, 2],
            "event_id": ["x", "x", "y", "y", "y", "y"],
            "kind": ["event", "event", "event", "event", "control", "control"],
            "group": ["no SWR", "no SWR", "SWR", "SWR", "SWR", "SWR"],
            "tetrode": [0, 1, 0, 1, 0, 1],
            "n_bins": [3, 3, 2, 2, 2, 2],
            "n_spikes": [1, 1, 0, 0, 4, 1],
        }
    )


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    for folder, name, table in (
        (EVENT_NONLOCAL_RESTRICTION_DIR, runner.BINS_NAME, _bins()),
        (EVENT_PARTICIPATION_DIR, runner.PARTICIPATION_NAME, PARTICIPATION),
        (EVENT_TETRODE_HOLDOUT_DIR, runner.HOLDOUT_NAME, _holdout()),
    ):
        (tmp_path / folder).mkdir()
        table.to_parquet(output_path(EPOCH, tmp_path / folder, name))
    monkeypatch.setattr(runner, "decode_time", lambda epoch_id, data_dir: TIME)
    monkeypatch.setattr(runner, "read_swr_table", lambda path: EVENTS)
    return tmp_path


def test_run_epoch_sums_each_window_and_adds_its_spikes(data_dir, tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    record = runner.run_epoch(EPOCH, tmp_path / "event_table.parquet", data_dir, output)
    table = pd.read_parquet(output_path(EPOCH, output, runner.OUTPUT_NAME))
    assert table["window_id"].tolist() == [0, 1, 2]
    assert table["n_bins"].tolist() == [3, 2, 2]
    assert table["n_spikes"].tolist() == [2, 0, 5]
    assert table["n_bins_in_swr"].tolist() == [0, 1, 0]
    np.testing.assert_allclose(table["full"] - table["swr"], [6, 4, 4])
    np.testing.assert_allclose(table["swr_two_sided"], [6, 4, 4])
    assert set(table["epoch_id"]) == {EPOCH} and set(table["animal"]) == {"wilbur"}
    assert record["status"] == "ok"
    assert record["n_windows_by_kind_group"] == {
        "control, SWR": 1,
        "event, SWR": 1,
        "event, no SWR": 1,
    }
    assert record["n_windows_without_spikes"] == 1
    assert record["n_no_swr_events_with_in_swr_bins"] == 0
    assert record["bin_s"] == pytest.approx(0.002)
    json.dumps(record, allow_nan=False)
    # outside SWRs: y's bin 9 is inside its SWR, the rest are outside
    assert table["n_bins_outside_swr"].tolist() == [3, 1, 2]
    np.testing.assert_allclose(
        table["full_outside_swr"] - table["swr_outside_swr"], [6, 2, 4]
    )
    np.testing.assert_allclose(table["duration_outside_swr_s"], [0.006, 0.002, 0.004])
    assert record["n_events_without_bins_outside_swr"] == 0


def test_a_time_grid_that_is_not_the_decodes_raises(data_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "decode_time", lambda epoch_id, data_dir: TIME + 1e-3)
    with pytest.raises(ValueError, match="decode frame"):
        runner.run_epoch(EPOCH, tmp_path / "e.parquet", data_dir, tmp_path)


def test_windows_the_holdout_run_did_not_score_raise(data_dir, tmp_path):
    holdout = _holdout()
    holdout = holdout[holdout["window_id"] != 2]
    holdout.to_parquet(
        output_path(EPOCH, data_dir / EVENT_TETRODE_HOLDOUT_DIR, runner.HOLDOUT_NAME)
    )
    with pytest.raises(ValueError, match="not the held-out"):
        runner.run_epoch(EPOCH, tmp_path / "e.parquet", data_dir, tmp_path)


def _main_header(
    tmp_path, monkeypatch, holdout_inputs: dict, event_table: str = "a"
) -> dict | None:
    """Run ``main`` with its inputs' digests and the held-out tetrode run's
    manifest replaced; return the header it ran the epochs under, or None when
    it ran none. Participation tables have digest ``"b"``."""
    ran = {}
    manifests = {
        EVENT_TETRODE_HOLDOUT_DIR: {
            "inputs": holdout_inputs,
            "epochs": {EPOCH: {"status": "ok"}},
        },
        EVENT_NONLOCAL_RESTRICTION_DIR: {"epochs": {EPOCH: {"status": "ok"}}},
    }
    monkeypatch.setattr(runner, "event_table_path", lambda data_dir: tmp_path / "e")
    monkeypatch.setattr(
        runner, "event_table_input", lambda path: {"sha256": event_table, "path": "e"}
    )
    monkeypatch.setattr(runner, "files_sha256", lambda paths: "b")
    monkeypatch.setattr(runner, "ok_table_paths", lambda folder, name: [])
    monkeypatch.setattr(runner, "run_input", lambda run_dir: {"sha256": run_dir.name})
    monkeypatch.setattr(runner, "provenance_record", lambda root: {})
    monkeypatch.setattr(runner, "read_manifest", lambda folder: manifests[folder.name])
    monkeypatch.setattr(runner, "run", lambda **kwargs: ran.update(kwargs))
    runner.main(["--data-dir", str(tmp_path)])
    assert ran["epochs"] == [EPOCH]
    return ran["header"]


def test_the_held_out_tetrode_runs_event_table_adds_no_note(tmp_path, monkeypatch):
    """The run on the held-out tetrode run's own event table keeps the header
    it had, so its records resume."""
    header = _main_header(
        tmp_path,
        monkeypatch,
        {"event_table": {"sha256": "a"}, "event_participation": {"sha256": "b"}},
    )
    assert set(header["settings"]) == {"contrasts"}


def test_another_event_table_runs_with_a_note_of_both_digests(
    tmp_path, monkeypatch, caplog
):
    header = _main_header(
        tmp_path,
        monkeypatch,
        {"event_table": {"sha256": "z"}, "event_participation": {"sha256": "b"}},
    )
    note = header["settings"][rw.EVENT_TABLE_NOTE]
    assert note["event_table_sha256"] == "a"
    assert note["tetrode_holdout_event_table_sha256"] == "z"
    assert "not the one the held-out tetrode run read" in note["note"]
    assert "will be pending" in caplog.text


def test_other_participation_tables_stop_the_run(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="event_participation"):
        _main_header(
            tmp_path,
            monkeypatch,
            {"event_table": {"sha256": "a"}, "event_participation": {"sha256": "c"}},
        )
    # with another event table as well: still stopped
    with pytest.raises(ValueError, match="event_participation"):
        _main_header(
            tmp_path,
            monkeypatch,
            {"event_table": {"sha256": "z"}, "event_participation": {"sha256": "c"}},
        )
