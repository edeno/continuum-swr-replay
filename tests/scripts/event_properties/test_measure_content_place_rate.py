"""The place-rate run to its manifest on a tiny synthetic epoch: every event of
the epoch has its row, whatever its well-visit label, ranked over the real
places only, and the record counts the places, the junction bins and the
events without a rank. The event table and the saved encoding model are
stubbed; the ranks themselves are tested in
``tests/analysis/event_properties/test_content_place_rate.py``."""

from __future__ import annotations

import json
import types
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.decoding.files import decode_paths
from continuum_swr_replay.paths import (
    EVENT_CONTENT_PLACE_RATE_DIR,
    processed_data_dir,
    replay_content_dir,
    table_path,
)
from continuum_swr_replay.runs.io import sha256_file
from scripts.event_evidence import _per_event_run
from scripts.event_properties import measure_content_place_rate as script

EPOCH = "j1620210706_02_r1_HPC"
OTHER = "j1620210706_04_r2_HPC"


def _table() -> pd.DataFrame:
    """Three events of the epoch (one on the track, one without a position)
    and one of another epoch."""
    return pd.DataFrame(
        {
            "animal": "j16",
            "session_date": 20210706,
            "epoch": ["02_r1", "02_r1", "02_r1", "04_r2"],
            "event_id": pd.Series([1, 2, 3, 4], dtype="uint64"),
            "is_swr": [True, False, False, True],
            "visit_context": ["in_visit", "track_center", "in_visit", "in_visit"],
            "replay_start_position_cm": [0.0, 41.0, np.nan, 0.0],
            "replay_end_position_cm": [21.0, np.nan, np.nan, 0.0],
        }
    )


def _model() -> dict:
    """Six bins at 0, 10, ..., 50 cm: bin 5 outside the track, bin 2 an interior
    junction placeholder. Real places 0, 10, 30, 40 cm with rates 2, 1, 3, 4:
    ranks 1/3, 0, 2/3, 1; median rate 2.5."""
    bins = pd.DataFrame(
        {"linear_position": np.arange(6) * 10.0, "edge_id": [0, 0, -1, 1, 1, -1]}
    )
    environment = types.SimpleNamespace(
        place_bin_centers_nodes_df_=bins,
        is_track_interior_=np.array([True] * 5 + [False]),
    )
    return {
        "environment": environment,
        "summed_ground_process_intensity": np.array([2.0, 1.0, 0.0, 3.0, 4.0]),
    }


@pytest.fixture
def run_inputs(processed_data_root, monkeypatch):
    path = table_path("event_table", replay_content_dir())
    path.parent.mkdir(parents=True, exist_ok=True)
    _table().to_parquet(path)
    detector = decode_paths(EPOCH, processed_data_root).detector
    detector.write_bytes(b"detector")
    monkeypatch.setattr(
        script, "read_swr_table", lambda path, columns: _table()[columns]
    )
    monkeypatch.setattr(script, "load_clusterless_encoding_model", lambda p: _model())
    monkeypatch.setattr(script, "analyzed_epoch_ids", lambda: [EPOCH, OTHER])
    monkeypatch.setattr(script, "cpu_pool", lambda args: (1, {}))
    monkeypatch.setattr(_per_event_run, "ProcessPoolExecutor", ThreadPoolExecutor)
    return sha256_file(path), sha256_file(detector)


def test_every_event_of_the_epoch_is_ranked_over_the_real_places(run_inputs):
    event_table_sha256, detector_sha256 = run_inputs
    script.main(["--epochs", EPOCH])
    folder = processed_data_dir() / EVENT_CONTENT_PLACE_RATE_DIR
    table = pd.read_parquet(folder / f"{EPOCH}_content_place_rate.parquet")
    assert table.columns.tolist() == [
        "event_id",
        "epoch_id",
        "animal",
        "group",
        "start_place_rate_rank",
        "end_place_rate_rank",
    ]
    assert table["event_id"].tolist() == [1, 2, 3]
    assert table["group"].tolist() == ["SWR", "no SWR", "no SWR"]
    # 0 cm -> rank 1/3; 21 cm -> the real place at 30 cm (not the junction at
    # 20 cm), rank 2/3; 41 cm -> 40 cm, rank 1
    np.testing.assert_allclose(table["start_place_rate_rank"], [1 / 3, 1, np.nan])
    np.testing.assert_allclose(table["end_place_rate_rank"], [2 / 3, np.nan, np.nan])

    manifest = json.loads((folder / "manifest.json").read_text())
    record = manifest["epochs"][EPOCH]
    assert record["status"] == "ok"
    assert (record["n_events"], record["n_places"]) == (3, 4)
    assert record["n_interior_junction_bins"] == 1
    assert record["n_events_without_rank"] == 1
    assert (
        record["ground_intensity_min"],
        record["ground_intensity_median"],
        record["ground_intensity_max"],
    ) == (1.0, 2.5, 4.0)
    assert record["epoch_inputs"]["detector"]["sha256"] == detector_sha256
    assert manifest["inputs"]["event_table"]["sha256"] == event_table_sha256
    assert "subset" not in manifest["settings"]
    assert list(manifest["epochs"]) == [EPOCH]  # only the epoch asked for


def test_the_run_refuses_an_event_table_with_an_epoch_not_analyzed(
    run_inputs, monkeypatch
):
    monkeypatch.setattr(script, "analyzed_epoch_ids", lambda: [EPOCH])
    with pytest.raises(ValueError, match="not analyzed"):
        script.main([])
