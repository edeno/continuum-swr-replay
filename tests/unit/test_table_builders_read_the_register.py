"""The table builders take their sessions from the epoch register, not the disk.

``prepare_data.find_sessions`` serves ``prepare_replay_content_data.py``,
``build_timebin_tables.py`` and ``regenerate_stop_table.py``; the ContFrag
script has its own ``find_sessions``, which ``refresh_contfrag_behavioral_
covariates.py`` also uses. Each lists the register's analysed epochs, and
refuses to start when one of them has no decode, rather than building tables
that silently cover fewer epochs than the register.

Offline: the register's selection is replaced by a frame of the epochs a test
names, the decodes are empty files under ``tmp_path`` (the data root), and the ContFrag script
is imported with ``spyglass`` stubbed in ``sys.modules``.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay import prepare_data as pdm
from continuum_swr_replay.paths import PROCESSED_DATA_DIR_ENV

CONTFRAG_SCRIPT = "scripts.tables.prepare_contfrag_replay_content_data"


def _analysed(*keys: tuple[str, str]):
    """``analysed_epochs`` returning these (nwb_file_name, epoch) keys."""

    def analysed_epochs() -> pd.DataFrame:
        return pd.DataFrame(
            {"epoch": [int(epoch.split("_")[0]) for _, epoch in keys]},
            index=pd.MultiIndex.from_tuples(
                list(keys), names=["nwb_file_name", "interval_list_name"]
            ),
        )

    return analysed_epochs


def _decode(root: Path, identifier: str, dot_suffix: str, *, results: bool) -> None:
    """Empty stand-ins for a decode's files under `root`."""
    (root / f"{identifier}_{dot_suffix}").write_bytes(b"")
    (root / f"{identifier}_environment.pkl").write_bytes(b"")
    if results:
        (root / f"{identifier}_results.nc").write_bytes(b"")


@pytest.fixture
def contfrag_prepare(monkeypatch):
    """The ContFrag script, imported offline with ``spyglass`` stubbed."""
    common = types.ModuleType("spyglass.common")
    common.PositionIntervalMap = type("PositionIntervalMap", (), {})
    monkeypatch.setitem(sys.modules, "spyglass", types.ModuleType("spyglass"))
    monkeypatch.setitem(sys.modules, "spyglass.common", common)
    fresh = CONTFRAG_SCRIPT not in sys.modules

    yield importlib.import_module(CONTFRAG_SCRIPT)

    if fresh:
        sys.modules.pop(CONTFRAG_SCRIPT, None)
        parent, _, child = CONTFRAG_SCRIPT.rpartition(".")
        vars(sys.modules[parent]).pop(child, None)


# --------------------------------------------------------------------------
# prepare_data.find_sessions
# --------------------------------------------------------------------------


def test_replay_sessions_are_the_registers_analysed_epochs(tmp_path, monkeypatch):
    monkeypatch.setenv(PROCESSED_DATA_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(
        pdm,
        "analysed_epochs",
        _analysed(("b20200102_.nwb", "04_r2"), ("a20200101_.nwb", "02_r1")),
    )
    _decode(tmp_path, "a20200101_02_r1_HPC", "80_data_over_time.pkl", results=True)
    _decode(tmp_path, "b20200102_04_r2_HPC", "80_data_over_time.pkl", results=False)
    # a decode on disk whose epoch the register does not analyse
    _decode(tmp_path, "a20200101_06_r3_HPC", "80_data_over_time.pkl", results=True)

    sessions = pdm.find_sessions()

    assert sessions == [
        {
            "nwb_file_name": "a20200101_.nwb",
            "epoch_name": "02_r1",
            "session_id": "a20200101_02_r1_HPC",
            "dot_file": tmp_path / "a20200101_02_r1_HPC_80_data_over_time.pkl",
            "env_file": tmp_path / "a20200101_02_r1_HPC_environment.pkl",
            "results_file": tmp_path / "a20200101_02_r1_HPC_results.nc",
        },
        {
            "nwb_file_name": "b20200102_.nwb",
            "epoch_name": "04_r2",
            "session_id": "b20200102_04_r2_HPC",
            "dot_file": tmp_path / "b20200102_04_r2_HPC_80_data_over_time.pkl",
            "env_file": tmp_path / "b20200102_04_r2_HPC_environment.pkl",
            "results_file": None,
        },
    ]


def test_replay_sessions_keep_the_first_of_each_animal(tmp_path, monkeypatch):
    keys = [("a20200101_.nwb", "02_r1"), ("a20200101_.nwb", "04_r2")]
    keys.append(("b20200102_.nwb", "02_r1"))
    monkeypatch.setenv(PROCESSED_DATA_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(pdm, "analysed_epochs", _analysed(*keys))
    for nwb_file_name, epoch in keys:
        identifier = f"{nwb_file_name.removesuffix('_.nwb')}_{epoch}_HPC"
        _decode(tmp_path, identifier, "80_data_over_time.pkl", results=True)

    sessions = pdm.find_sessions(max_per_animal=1)

    assert [s["session_id"] for s in sessions] == [
        "a20200101_02_r1_HPC",
        "b20200102_02_r1_HPC",
    ]


@pytest.mark.parametrize("missing", ["80_data_over_time.pkl", "environment.pkl"])
def test_an_analysed_epoch_without_its_decode_stops_the_replay_build(
    tmp_path, monkeypatch, missing
):
    keys = [("a20200101_.nwb", "02_r1"), ("a20200101_.nwb", "04_r2")]
    monkeypatch.setenv(PROCESSED_DATA_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(pdm, "analysed_epochs", _analysed(*keys))
    for identifier in ("a20200101_02_r1_HPC", "a20200101_04_r2_HPC"):
        _decode(tmp_path, identifier, "80_data_over_time.pkl", results=True)
    (tmp_path / f"a20200101_04_r2_HPC_{missing}").unlink()

    with pytest.raises(FileNotFoundError, match="a20200101_04_r2_HPC"):
        pdm.find_sessions()


# --------------------------------------------------------------------------
# The ContFrag script's find_sessions
# --------------------------------------------------------------------------


def test_contfrag_sessions_are_the_registers_analysed_epochs(
    contfrag_prepare, tmp_path, monkeypatch
):
    monkeypatch.setenv(PROCESSED_DATA_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(
        contfrag_prepare, "analysed_epochs", _analysed(("a20200101_.nwb", "02_r1"))
    )
    _decode(
        tmp_path, "a20200101_02_r1_HPC_contfrag", "data_over_time.pkl", results=True
    )
    _decode(
        tmp_path, "a20200101_06_r3_HPC_contfrag", "data_over_time.pkl", results=True
    )

    sessions = contfrag_prepare.find_sessions(brain_area="HPC")

    assert sessions == [
        {
            "nwb_file_name": "a20200101_.nwb",
            "epoch_name": "02_r1",
            "brain_area": "HPC",
            "dot_file": tmp_path / "a20200101_02_r1_HPC_contfrag_data_over_time.pkl",
            "env_file": tmp_path / "a20200101_02_r1_HPC_contfrag_environment.pkl",
            "results_file": tmp_path / "a20200101_02_r1_HPC_contfrag_results.nc",
        }
    ]


def test_an_analysed_epoch_without_its_hpc_contfrag_decode_stops_the_build(
    contfrag_prepare, tmp_path, monkeypatch
):
    monkeypatch.setenv(PROCESSED_DATA_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(
        contfrag_prepare,
        "analysed_epochs",
        _analysed(("a20200101_.nwb", "02_r1"), ("a20200101_.nwb", "04_r2")),
    )
    _decode(
        tmp_path, "a20200101_02_r1_HPC_contfrag", "data_over_time.pkl", results=True
    )

    with pytest.raises(FileNotFoundError, match="a20200101_04_r2_HPC_contfrag"):
        contfrag_prepare.find_sessions(brain_area="HPC")


def test_a_cortical_contfrag_build_keeps_the_analysed_epochs_it_decoded(
    contfrag_prepare, tmp_path, monkeypatch
):
    """mPFC and OFC are recorded in some sessions only: an analysed epoch with no
    decode for the area is outside that area's cohort, not missing."""
    monkeypatch.setenv(PROCESSED_DATA_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(
        contfrag_prepare,
        "analysed_epochs",
        _analysed(("a20200101_.nwb", "02_r1"), ("a20200101_.nwb", "04_r2")),
    )
    _decode(
        tmp_path, "a20200101_04_r2_mPFC_contfrag", "data_over_time.pkl", results=True
    )

    sessions = contfrag_prepare.find_sessions(brain_area="mPFC")

    assert [(s["epoch_name"], s["brain_area"]) for s in sessions] == [("04_r2", "mPFC")]


# --------------------------------------------------------------------------
# compute_hpc_raw_stats.py, whose stats feed quality_summary.csv
# --------------------------------------------------------------------------


def test_hpc_raw_stats_run_over_the_registers_analysed_epochs(monkeypatch):
    import scripts.qa.compute_hpc_raw_stats as raw_stats

    monkeypatch.setattr(
        raw_stats,
        "analysed_epochs",
        lambda: _analysed(
            ("b20200102_.nwb", "04_r2"), ("a20200101_.nwb", "02_r1")
        )().assign(animal=["b", "a"], date=[20200102, 20200101]),
    )

    sessions = raw_stats._build_session_list(animal=None, limit=None)

    assert sessions.to_dict("records") == [
        {"nwb_file_name": "a20200101_.nwb", "animal": "a", "day": "20200101",
         "epoch": "02_r1"},
        {"nwb_file_name": "b20200102_.nwb", "animal": "b", "day": "20200102",
         "epoch": "04_r2"},
    ]  # fmt: skip
    assert not hasattr(raw_stats, "get_epoch_info")
