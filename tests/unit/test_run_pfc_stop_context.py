"""The Q8 stop-context runner: which sessions and which units each fit uses."""

from __future__ import annotations

import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.pfc.load_population import HELDOUT_P_COLUMNS

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts/pfc/run_pfc_stop_context.py"
_SPEC = importlib.util.spec_from_file_location("run_pfc_stop_context", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)

SWR, NON = "non_local_SWR", "non_local_no_SWR"


def _units() -> pd.DataFrame:
    """Six units whose signs and p-values differ between the two classes."""
    return pd.DataFrame(
        {
            f"{SWR}: peak_log_modulation": [0.5, 0.5, -0.4, -0.4, 0.3, np.nan],
            HELDOUT_P_COLUMNS[SWR]: [0.01, 0.20, 0.01, 0.30, 0.04, 0.01],
            f"{NON}: peak_log_modulation": [-0.2, 0.6, 0.6, -0.1, 0.0, 0.2],
            HELDOUT_P_COLUMNS[NON]: [0.50, 0.01, 0.02, 0.01, 0.01, 0.90],
        }
    )


@pytest.mark.parametrize(
    ("event_class", "sign", "unit_set", "expected"),
    [
        # sign from that class's own peak; NaN and 0 have no sign
        (SWR, "increase", "all", [0, 1, 4]),
        (SWR, "decrease", "all", [2, 3]),
        (NON, "increase", "all", [1, 2, 5]),
        (NON, "decrease", "all", [0, 3]),
        # called = that class's own p below 0.05
        (SWR, "increase", "called", [0, 4]),
        (SWR, "decrease", "called", [2]),
        (NON, "increase", "called", [1, 2]),
        (NON, "decrease", "called", [3]),
    ],
)
def test_units_are_chosen_by_their_own_class_peak_and_p(
    event_class, sign, unit_set, expected
):
    chosen = runner.select_units(_units(), event_class, sign, unit_set)
    np.testing.assert_array_equal(np.flatnonzero(chosen), expected)


def test_one_class_p_never_selects_another_class_units():
    units = _units()
    before = runner.select_units(units, NON, "increase", "called")
    # Making every unit significant for the SWR-coupled class changes nothing
    # for the non-SWR class, and the reverse.
    units[HELDOUT_P_COLUMNS[SWR]] = 0.0
    units[f"{SWR}: peak_log_modulation"] = 1.0
    np.testing.assert_array_equal(
        runner.select_units(units, NON, "increase", "called"), before
    )
    units = _units()
    before = runner.select_units(units, SWR, "decrease", "called")
    units[HELDOUT_P_COLUMNS[NON]] = 1.0
    np.testing.assert_array_equal(
        runner.select_units(units, SWR, "decrease", "called"), before
    )


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
