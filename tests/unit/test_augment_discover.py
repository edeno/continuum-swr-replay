"""Filename routing in the data_over_time augmentation script."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "exploratory" / "_augment_data_over_time.py"
_spec = importlib.util.spec_from_file_location("_augment_dot", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def test_discover_groups_nonlocal_and_contfrag_by_epoch():
    files = [
        Path("chimi20200212_02_r1_HPC_80_data_over_time.pkl"),
        Path("chimi20200212_02_r1_HPC_contfrag_data_over_time.pkl"),
        Path("j1620210706_04_r2_HPC_80_data_over_time.pkl"),
        Path("unrecognized_file.pkl"),
    ]
    epochs = _mod._discover(files)
    assert epochs[("chimi20200212", "02_r1", "HPC")] == {
        "nonlocal:80": files[0],
        "contfrag": files[1],
    }
    assert epochs[("j1620210706", "04_r2", "HPC")] == {"nonlocal:80": files[2]}
    # unrecognized filename is not grouped
    assert all("unrecognized" not in str(p) for d in epochs.values() for p in d.values())


def test_new_and_stale_column_constants():
    assert _mod.NEW_COLUMNS == ("step_distance", "run_speed_baseline")
    assert set(_mod._STALE_COLUMNS) == {"ripple_power", "mua_rate", "theta_power"}
