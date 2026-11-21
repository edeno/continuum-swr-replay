"""The check that the held-out runs' stored folds are the fold rule's, on
made-up runs."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.event_evidence import tetrode_holdout
from continuum_swr_replay.analysis.event_tables.prepare_data import (
    behavioral_context_cache_path,
)
from continuum_swr_replay.paths import (
    EVENT_NONLOCAL_RESTRICTION_DIR,
    EVENT_TETRODE_HOLDOUT_DIR,
    NONLOCAL_RESTRICTION_BINS_NAME,
)
from continuum_swr_replay.runs.io import output_path
from scripts.checks import check_stored_folds_reproduce as check

EPOCHS = ("aa20200101_02_r1_HPC", "bb20200101_02_r1_HPC")


def _trials(shift: float) -> pd.DataFrame:
    """Six trials of 1.5 s to wells 2 and 3 in turn, in a 10 s epoch."""
    return pd.DataFrame(
        {
            "movement_start_time": shift + np.array([0.5, 2.0, 3.5, 5.0, 6.5, 8.0]),
            "end_time": shift + np.array([2.0, 3.5, 5.0, 6.5, 8.0, 9.5]),
            "to_well": [2, 3, 2, 3, 2, 3],
        }
    )


def _write_runs(root: Path) -> dict:
    """Both runs and the session cache for :data:`EPOCHS`, all consistent."""
    restriction = root / EVENT_NONLOCAL_RESTRICTION_DIR
    holdout = root / EVENT_TETRODE_HOLDOUT_DIR
    restriction.mkdir(parents=True)
    holdout.mkdir(parents=True)
    records = {}
    for i, epoch_id in enumerate(EPOCHS):
        time = 100.0 * i + np.arange(0.0, 10.0, 0.002)
        trials = _trials(100.0 * i)
        folds, margin, counts = tetrode_holdout.epoch_folds(time, trials)
        pd.DataFrame({"time": time, "block": folds}).to_parquet(
            output_path(epoch_id, restriction, NONLOCAL_RESTRICTION_BINS_NAME)
        )
        cache = behavioral_context_cache_path({"session_id": epoch_id})
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "wb") as f:
            pickle.dump({"trials": trials}, f)
        records[epoch_id] = {**counts, "margin_bins": margin}
    for folder in (restriction, holdout):
        (folder / "manifest.json").write_text(json.dumps({"epochs": records}))
    return records


def test_stored_folds_that_match_pass(processed_data_root, capsys) -> None:
    _write_runs(processed_data_root)
    assert check.main(["--workers", "1"]) == 0
    out = capsys.readouterr().out
    assert "reproduced from the session cache's trials: 2 | folds differ: 0" in out
    assert "equal the restriction run's: 2 | differ: 0" in out
    assert check.main(["--epochs", EPOCHS[0]]) == 0
    assert "(a) epochs: 1 |" in capsys.readouterr().out


def test_a_changed_block_is_reported(processed_data_root, capsys) -> None:
    _write_runs(processed_data_root)
    path = output_path(
        EPOCHS[1],
        processed_data_root / EVENT_NONLOCAL_RESTRICTION_DIR,
        NONLOCAL_RESTRICTION_BINS_NAME,
    )
    bins = pd.read_parquet(path)
    bins.loc[10, "block"] = (bins.loc[10, "block"] + 1) % 5
    bins.to_parquet(path)
    assert check.main([]) == 1
    out = capsys.readouterr().out
    assert "reproduced from the session cache's trials: 1 | folds differ: 1" in out
    assert f"FOLDS DIFFER {EPOCHS[1]} 1 bins" in out


def test_a_different_record_in_the_holdout_manifest_is_reported(
    processed_data_root, capsys
) -> None:
    records = _write_runs(processed_data_root)
    records[EPOCHS[0]] = {**records[EPOCHS[0]], "n_groups": 99}
    manifest = processed_data_root / EVENT_TETRODE_HOLDOUT_DIR / "manifest.json"
    manifest.write_text(json.dumps({"epochs": records}))
    assert check.main([]) == 1
    out = capsys.readouterr().out
    assert "equal the restriction run's: 1 | differ: 1" in out
    assert f"RECORD DIFFERS {EPOCHS[0]} n_groups 99" in out


def test_an_epoch_that_cannot_be_read_is_an_error(processed_data_root, capsys) -> None:
    _write_runs(processed_data_root)
    behavioral_context_cache_path({"session_id": EPOCHS[0]}).unlink()
    assert check.main([]) == 1
    assert "| errors: 1" in capsys.readouterr().out
