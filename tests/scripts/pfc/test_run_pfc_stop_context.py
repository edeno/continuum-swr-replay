"""The stop-context runner: which sessions it reads, and when a cached one is reused."""

from __future__ import annotations

import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts/pfc/run_pfc_stop_context.py"
_SPEC = importlib.util.spec_from_file_location("run_pfc_stop_context", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)


def _cohort(root: Path, monkeypatch, *, bundles: list[str]) -> Namespace:
    """Three analysed epochs, one left out of the PFC cohort; bundles as named."""
    from continuum_swr_replay.data_loaders import epoch_register

    epochs = pd.DataFrame(
        {"animal": "j16", "date": 20210706},
        index=pd.MultiIndex.from_tuples(
            [
                ("j1620210706_.nwb", "02_r1"),
                ("j1620210706_.nwb", "04_r2"),
                ("j1620210706_.nwb", "06_r3"),
            ],
            names=["nwb_file_name", "interval_list_name"],
        ),
    )
    monkeypatch.setattr(epoch_register, "analysed_epochs", lambda: epochs)
    (root / "pfc_cohort_mPFC.json").write_text(
        json.dumps({"left_out": {"j1620210706_.nwb 06_r3": "no sorting"}})
    )
    for epoch in bundles:
        (root / f"j1620210706_{epoch}_mPFC_modulation.pkl").write_bytes(b"")
    return Namespace(animal=None, date=None, epoch=None)


def test_selected_sessions_are_the_pfc_cohort(processed_data_root, monkeypatch):
    args = _cohort(processed_data_root, monkeypatch, bundles=["02_r1", "04_r2"])
    assert runner.selected_sessions(args) == [
        ("j1620210706_.nwb", "02_r1"),
        ("j1620210706_.nwb", "04_r2"),
    ]


def test_a_cohort_epoch_without_a_bundle_raises(processed_data_root, monkeypatch):
    args = _cohort(processed_data_root, monkeypatch, bundles=["02_r1"])
    with pytest.raises(FileNotFoundError, match="04_r2"):
        runner.selected_sessions(args)


def test_a_cached_session_is_reused_only_from_the_current_tables(tmp_path):
    current = {"canonical_event_table_sha256": "e", "stop_table_sha256": "s"}
    path = tmp_path / "x.provenance.json"
    assert not runner.cached_is_current(path, current)
    path.write_text(json.dumps({"input_provenance": dict(current)}))
    assert runner.cached_is_current(path, current)
    path.write_text(
        json.dumps({"input_provenance": {**current, "stop_table_sha256": "old"}})
    )
    assert not runner.cached_is_current(path, current)
