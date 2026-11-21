"""The restriction-window runner's wiring: what ``run_epoch`` reads, checks and
writes. The decode frame's time and the event table's read are replaced at their
boundaries; the per-bin, participation and held-out tetrode tables are made by
hand under a temporary processed-data root."""

import json

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_evidence import nonlocal_restriction as nr
from continuum_swr_replay.analysis.event_evidence import restriction_windows as rw
from continuum_swr_replay.paths import (
    BIN_FACTS_NAME,
    EVENT_BIN_FACTS_DIR,
    EVENT_NONLOCAL_RESTRICTION_DIR,
    EVENT_PARTICIPATION_DIR,
    EVENT_TETRODE_HOLDOUT_DIR,
    NONLOCAL_RESTRICTION_BINS_NAME,
    NONLOCAL_RESTRICTION_WINDOWS_NAME,
    PARTICIPATION_NAME,
    TETRODE_HOLDOUT_NAME,
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
    """A fold-fit bins table: ``none`` 0, ``swr`` 1 and ``full`` 3 nats at
    every bin, the two-sided scores twice those, and the ``full`` model's state
    probabilities and non-local place bin, which the window run does not read."""
    one = np.ones(N, dtype=np.float32)
    fold_fit = dict.fromkeys(nr.STATE_PROBABILITY_COLUMNS.values(), 0.25 * one)
    fold_fit[nr.NONLOCAL_PLACE_BIN_COLUMN] = np.zeros(N, dtype=np.int32)
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
            **fold_fit,
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


def _facts() -> pd.DataFrame:
    """Spikes per bin as the held-out tetrode table counts them: 2 in x (bins 3
    and 4), none in y, 5 in y's control (bins 14 and 15); one more at bin 18,
    in no window. Bins 0-3 and 9-16 are inside a well visit."""
    spikes = np.zeros(N, dtype=np.int32)
    spikes[[3, 4, 14, 15, 18]] = [1, 1, 4, 1, 1]
    inside = np.zeros(N, dtype=bool)
    inside[0:4] = True
    inside[9:17] = True
    return pd.DataFrame({"time": TIME, "n_spikes": spikes, "in_well_visit": inside})


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    restriction = tmp_path / EVENT_NONLOCAL_RESTRICTION_DIR
    restriction.mkdir()
    # a fold-fit record: it counts no spikes
    output_path(EPOCH, restriction, "nonlocal_restriction").with_suffix(
        ".json"
    ).write_text(json.dumps({"status": "ok", "fit": "folds", "n_bins": N}))
    for folder, name, table in (
        (EVENT_NONLOCAL_RESTRICTION_DIR, NONLOCAL_RESTRICTION_BINS_NAME, _bins()),
        (EVENT_BIN_FACTS_DIR, BIN_FACTS_NAME, _facts()),
        (EVENT_PARTICIPATION_DIR, PARTICIPATION_NAME, PARTICIPATION),
        (EVENT_TETRODE_HOLDOUT_DIR, TETRODE_HOLDOUT_NAME, _holdout()),
    ):
        (tmp_path / folder).mkdir(exist_ok=True)
        table.to_parquet(output_path(EPOCH, tmp_path / folder, name))
    monkeypatch.setattr(runner, "decode_time", lambda epoch_id, data_dir: TIME)
    monkeypatch.setattr(runner, "read_swr_table", lambda path, columns: EVENTS[columns])
    return tmp_path


def test_run_epoch_sums_each_window_and_adds_its_spikes(data_dir, tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    record = runner.run_epoch(EPOCH, tmp_path / "event_table.parquet", data_dir, output)
    table = pd.read_parquet(
        output_path(EPOCH, output, NONLOCAL_RESTRICTION_WINDOWS_NAME)
    )
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
    # from the per-bin facts: x's 2 spikes are outside SWRs, its bins 2-3 are
    # inside a visit (1 spike); y's bin 8 is outside its SWR, not in a visit
    assert table["n_spikes_outside_swr"].tolist() == [2, 0, 5]
    assert table["n_bins_outside_swr_in_visit"].tolist() == [2, 0, 2]
    assert table["n_spikes_outside_swr_in_visit"].tolist() == [1, 0, 5]
    np.testing.assert_allclose(
        table["full_outside_swr_in_visit"] - table["swr_outside_swr_in_visit"],
        [4, 0, 4],
    )
    assert record["n_events_without_spikes_outside_swr"] == 1
    assert record["n_spikes"] == 8
    # the time sums are the held-out event run's (detect_heldout_events.py)
    assert sorted(path.name for path in output.iterdir()) == [
        f"{EPOCH}_{NONLOCAL_RESTRICTION_WINDOWS_NAME}.parquet"
    ]


def test_per_bin_facts_whose_window_spikes_are_not_the_holdout_runs_raise(
    data_dir, tmp_path
):
    """The facts' spikes are tied to the held-out tetrode run's window by
    window; the restriction run's record counts no spikes, so a change outside
    every window is not checked."""
    facts = _facts()
    facts.loc[18, "n_spikes"] = 0  # bin 18 is in no window
    facts.to_parquet(output_path(EPOCH, data_dir / EVENT_BIN_FACTS_DIR, "bin_facts"))
    output = tmp_path / "out"
    output.mkdir()
    record = runner.run_epoch(EPOCH, tmp_path / "e.parquet", data_dir, output)
    assert record["status"] == "ok" and record["n_spikes"] == 7
    facts = _facts()
    facts.loc[[3, 18], "n_spikes"] = np.array(
        [0, 2], dtype=np.int32
    )  # x holds 1 spike, not its 2
    facts.to_parquet(output_path(EPOCH, data_dir / EVENT_BIN_FACTS_DIR, "bin_facts"))
    with pytest.raises(ValueError, match="1 windows' spikes"):
        runner.run_epoch(EPOCH, tmp_path / "e.parquet", data_dir, tmp_path)


def test_another_restriction_run_is_read(data_dir, tmp_path):
    other = tmp_path / "shared_fit"
    other.mkdir()
    bins = _bins()
    bins["full"] = np.float32(5.0)
    bins.to_parquet(output_path(EPOCH, other, NONLOCAL_RESTRICTION_BINS_NAME))
    output = tmp_path / "out"
    output.mkdir()
    runner.run_epoch(EPOCH, tmp_path / "e.parquet", data_dir, output, other)
    table = pd.read_parquet(
        output_path(EPOCH, output, NONLOCAL_RESTRICTION_WINDOWS_NAME)
    )
    np.testing.assert_allclose(table["full"] - table["swr"], [12, 8, 8])


def test_a_time_grid_that_is_not_the_decodes_raises(data_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "decode_time", lambda epoch_id, data_dir: TIME + 1e-3)
    with pytest.raises(ValueError, match="decode frame"):
        runner.run_epoch(EPOCH, tmp_path / "e.parquet", data_dir, tmp_path)


def test_windows_the_holdout_run_did_not_score_raise(data_dir, tmp_path):
    holdout = _holdout()
    holdout = holdout[holdout["window_id"] != 2]
    holdout.to_parquet(
        output_path(EPOCH, data_dir / EVENT_TETRODE_HOLDOUT_DIR, TETRODE_HOLDOUT_NAME)
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
        runner, "file_input", lambda path: {"sha256": event_table, "path": "e"}
    )
    monkeypatch.setattr(
        runner, "participation_input", lambda data_dir: {"sha256": "b", "path": "p"}
    )
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
