"""``scripts/qa/table_parity.py diff`` must fail on every kind of table change.

Its "0 difference(s)" stands for the branch's byte-identical constraint, so the
probes below -- a dtype change, a column or row reorder, a value change -- must
each be a difference, as must a missing or an extra table. ``--restrict`` must
take its sessions from the slice's register, not from what the candidate built.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.paths import BASENAME_TO_TABLE, CANONICAL_TABLE_BASENAMES
from continuum_swr_replay.tables.covariate_registry import row_keys
from continuum_swr_replay.tables.io import write_table
from scripts.qa import table_parity as tp


def _frame() -> pd.DataFrame:
    """Rows unique on every registered row key, with int, float and object data."""
    n = 4
    return pd.DataFrame(
        {
            "animal": ["chimi", "chimi", "j16", "j16"],
            "session_date": np.array([20200212, 20200212, 20210706, 20210706]),
            "epoch": ["02_r1"] * n,
            "event_id": np.arange(n, dtype=np.int64),
            "candidate_arm": np.arange(n, dtype=np.int64),
            "patch": np.arange(n, dtype=np.int64),
            "t_in_event": np.arange(n, dtype=np.int64),
            "stop_id": np.arange(10, 10 + n, dtype=np.uint64),
            "bin_in_stop": np.arange(n, dtype=np.int64),
            "event_type": ["SWR", "MUA", "SWR", "MUA"],
            "onset_time": np.arange(n, dtype=float),
            "count": np.array([1, 2, 3, 4], dtype=np.int64),
            "value": np.array([0.5, np.nan, 1.5, 2.5]),
            "label": ["a", "b", "a", None],
        }
    )


KEYS = ["event_id"]


def _probes() -> dict[str, pd.DataFrame]:
    base = _frame()
    return {
        "int64 -> float64": base.astype({"count": "float64"}),
        "int64 -> int32": base.astype({"count": "int32"}),
        "float64 -> float32": base.astype({"value": "float32"}),
        "object -> category": base.astype({"label": "category"}),
        "reversed columns": base[base.columns[::-1]],
        "reversed rows": base.iloc[::-1].reset_index(drop=True),
        "changed value": base.assign(value=[0.5, np.nan, 1.5, 2.5000001]),
    }


def test_identical_frames_have_no_problems():
    assert tp.frame_problems(_frame(), _frame(), keys=KEYS, label="t") == []


@pytest.mark.parametrize("probe", list(_probes()))
def test_every_probe_is_a_difference(probe):
    problems = tp.frame_problems(_probes()[probe], _frame(), keys=KEYS, label="t")
    assert problems, f"{probe} passed the diff"


def _write_root(root: Path) -> None:
    for folder in (tp.RC, tp.CF):
        (root / folder).mkdir(parents=True)
    for name in CANONICAL_TABLE_BASENAMES:
        write_table(_frame(), root / tp.RC / name)
    for name, _ in tp.CONTFRAG_TABLES:
        write_table(_frame(), root / tp.CF / name)
    for relative in tp.REFERENCE_FILES:
        (root / relative).write_text(json.dumps({"x": {"mean": 0.0}}))


def test_row_keys_of_the_fixture_are_unique():
    frame = _frame()
    tables = [BASENAME_TO_TABLE[n] for n in CANONICAL_TABLE_BASENAMES]
    tables += [table for _, table in tp.CONTFRAG_TABLES]
    for table in tables:
        assert not frame.duplicated(list(row_keys(table=table))).any(), table


def test_identical_roots_diff_clean(tmp_path):
    _write_root(tmp_path / "a")
    _write_root(tmp_path / "b")
    assert tp.diff(tmp_path / "a", tmp_path / "b", restrict=False) == 0


def test_missing_and_extra_tables_are_all_reported(tmp_path, capsys):
    _write_root(tmp_path / "cand")
    _write_root(tmp_path / "ref")
    (tmp_path / "cand" / tp.RC / "event_table.parquet").unlink()
    (tmp_path / "cand" / tp.CF / "stop_table_mua_HPC.parquet").unlink()
    write_table(_frame(), tmp_path / "cand" / tp.RC / "surprise_table")
    write_table(
        _frame().astype({"count": "float64"}), tmp_path / "cand" / tp.RC / "stop_table"
    )
    n_bad = tp.diff(tmp_path / "cand", tmp_path / "ref", restrict=False)
    out = capsys.readouterr().out
    assert n_bad == 4
    assert "MISSING  replay_content/event_table" in out
    assert "MISSING  contfrag_replay_content/stop_table_mua_HPC" in out
    assert "EXTRA    replay_content/surprise_table.parquet" in out
    assert "DIFFERS  replay_content/stop_table" in out
    # the tables after the first missing one are still compared
    assert "same     contfrag_replay_content/stop_table_swr_HPC" in out


def test_restrict_takes_the_sessions_from_the_slice_register(tmp_path):
    cand, ref = tmp_path / "cand", tmp_path / "ref"
    (ref / tp.RC).mkdir(parents=True)
    write_table(_frame(), ref / tp.RC / "stop_table")
    cand.mkdir()
    pd.DataFrame(
        [
            ["chimi20200212_.nwb", "02_r1", "analysed", "", ""],
            ["j1620210706_.nwb", "02_r1", "excluded", "MISSING_INPUT", "slice"],
        ],
        columns=["nwb_file_name", "epoch", "status", "reason", "evidence"],
    ).to_csv(cand / "epoch_register.csv", index=False)
    # The candidate's own stop table is never read: it lost every row.
    (cand / tp.RC).mkdir()
    write_table(_frame().iloc[:0], cand / tp.RC / "stop_table")
    keep = tp._session_restriction(cand, ref)
    assert keep(_frame())["animal"].tolist() == ["chimi", "chimi"]
    # qvals (no session columns) is cut by the reference's stops of the slice.
    qvals = _frame()[["stop_id", "value"]]
    assert keep(qvals)["stop_id"].tolist() == [10, 11]


def test_restrict_refuses_a_root_without_a_slice_register(tmp_path):
    with pytest.raises(FileNotFoundError, match="--restrict needs a slice root"):
        tp._session_restriction(tmp_path, tmp_path)


def test_regen_validates_sessions_before_creating_the_root(tmp_path):
    root = tmp_path / "slice"
    with pytest.raises(ValueError, match="not analysed epochs"):
        tp.regen(root, ["nosuchanimal20200101_02_r1_HPC"])
    assert not root.exists()
