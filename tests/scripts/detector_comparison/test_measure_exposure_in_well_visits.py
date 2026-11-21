"""``measure_exposure_in_well_visits`` run to its manifest on one synthetic epoch:
the inventory run's stored tables, a decode frame read through a stub and one well
visit. The exposure itself is tested in
``tests/analysis/detector_comparison/test_inventory_nonlocal_comparison.py``."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.detector_comparison.nonlocal_comparison import (
    exposure_in_well_visits,
)
from continuum_swr_replay.analysis.event_definition.cohort_selection import (
    ANALYZED_SUBSET,
)
from continuum_swr_replay.paths import (
    EXPOSURE_IN_VISITS_DIR,
    event_features_root,
    inventory_comparison_root,
)
from continuum_swr_replay.runs.io import files_sha256, output_path
from scripts.detector_comparison import measure_exposure_in_well_visits as script
from tests.support._session_cache_fixtures import write_session_cache_entries

EPOCH = "j1620210706_02_r1_HPC"
TIME = np.arange(0.0, 100.0 + 1e-9, 0.002)
SESSION = {"animal": "j16", "day": 20210706, "epoch": "02_r1"}


SPEED = np.where((TIME > 30.0) & (TIME < 60.0), 20.0, 1.0)


def _lay_out(offset_s: float) -> list:
    """The features run, its inventory run's tables (observed 0 to 100 s, still
    except 30 to 60 s, nothing set aside: about 70 s of exposure, stored as the
    rebuilt exposure plus ``offset_s``) and the cache entry."""
    run = event_features_root() / "run1"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "settings": {"input_runs": {"inventory": "inv1"}},
                "epochs": {EPOCH: {"status": "ok", "decode": {"sha256": "d"}}},
            }
        )
    )
    inventory = inventory_comparison_root() / "inv1"
    inventory.mkdir(parents=True)
    pd.DataFrame(
        {
            **SESSION,
            "brain_area": "HPC",
            "detector": ["state_space"],
            "run_id": "r",
            "start_s": 0.0,
            "end_s": 100.0,
            "available": True,
            "reason": "",
        }
    ).to_parquet(output_path(EPOCH, inventory, "observations"))
    set_aside = pd.DataFrame(
        {
            **{key: pd.Series([], dtype=type(value)) for key, value in SESSION.items()},
            "detector": pd.Series([], dtype=object),
            "interval_start_s": pd.Series([], dtype=float),
            "interval_end_s": pd.Series([], dtype=float),
        }
    )
    set_aside.to_parquet(output_path(EPOCH, inventory, "set_aside"))
    observations = pd.read_parquet(output_path(EPOCH, inventory, "observations"))
    rebuilt_s, _ = exposure_in_well_visits(
        observations, set_aside, TIME, SPEED, np.array([0.0]), np.array([1.0])
    )
    pd.DataFrame(
        {**SESSION, "detector": ["state_space"], "exposure_s": rebuilt_s + offset_s}
    ).to_parquet(output_path(EPOCH, inventory, "counts"))
    return write_session_cache_entries([EPOCH])


@pytest.fixture
def stubbed(processed_data_root, monkeypatch):
    frame = pd.DataFrame({"head_speed": SPEED}, index=pd.Index(TIME, name="time"))
    monkeypatch.setattr(
        script.analyzed_time,
        "read_verified_dot_with_swr_source",
        lambda path, sha256: frame,
    )
    monkeypatch.setattr(
        script,
        "well_visit_times",
        lambda identifier: (np.array([20.0]), np.array([70.0])),
    )
    monkeypatch.setattr(script, "ProcessPoolExecutor", ThreadPoolExecutor)
    return processed_data_root


def test_main_writes_the_exposure_inside_the_visits_and_the_manifest(stubbed):
    entries = _lay_out(0.0)

    script.main(["--event-features-run", "run1"])

    folder = stubbed / EXPOSURE_IN_VISITS_DIR
    table = pd.read_parquet(folder / "exposure.parquet")
    (row,) = table.itertuples()
    assert row.epoch_id == EPOCH
    assert row.exposure_s == pytest.approx(70.0, abs=0.01)
    # still 20 to 30 s and 60 to 70 s inside the visit
    assert row.exposure_in_visit_s == pytest.approx(20.0, abs=0.01)
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["subset"] == ANALYZED_SUBSET
    assert manifest["session_cache"]["sha256"] == files_sha256(entries)
    assert manifest["inventory_run"] == "inv1"
    assert manifest["n_epochs"] == 1


def test_an_exposure_that_is_not_the_inventory_run_s_is_refused(stubbed):
    _lay_out(1.0)
    with pytest.raises(ValueError, match="is not the inventory run's"):
        script.main(["--event-features-run", "run1"])
