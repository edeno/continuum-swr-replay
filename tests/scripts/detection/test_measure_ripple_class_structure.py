"""The ripple class-structure analysis reads only what its inputs' runs mark done."""

from __future__ import annotations

import argparse
import json

import pandas as pd
import pytest

import scripts.detection.measure_ripple_class_structure as structure
from continuum_swr_replay.analysis.hippocampus.spike_evidence import (
    SPIKE_EVIDENCE_METRICS,
)
from continuum_swr_replay.analysis.inventory_comparison.feature_run import (
    EVENT_FEATURES_RUN,
)
from continuum_swr_replay.paths import (
    event_features_root,
    event_spike_evidence_dir,
    replay_content_dir,
    table_path,
)
from continuum_swr_replay.runs.io import files_sha256, output_path


def _features(event_id: int, ripple_z_max: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_key": [f"state_space:e:{event_id}"],
            "detector": "state_space",
            "ripple_z_max": ripple_z_max,
            "ripple_z_mean": 1.0,
            "n_spikes": 7,
        }
    )


def _lay_out_inputs() -> None:
    """One event of epoch ``e1``; the features run marks ``e1`` ok and ``e2``
    failed, and ``e2`` left a features table behind holding the same event."""
    event_table = table_path("event_table", replay_content_dir())
    event_table.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "event_id": pd.Series([5], dtype="uint64"),
            "whole_event_speed_eligible": True,
            "ripple_power_z": 1.0,
            "is_swr": False,
            "event_start_s": 1.0,
            "event_end_s": 1.2,
            "mean_abs_ahead_behind_cm": 10.0,
        }
    ).to_parquet(event_table)
    criterion = structure.out_dir() / "criterion"
    criterion.mkdir(parents=True)
    pd.DataFrame(
        {
            "event_id": pd.Series([5], dtype="uint64"),
            "epoch_id": "e1",
            "decode_sha256": "d",
            "ripple_z_max": 2.0,
            "ripple_z_mean": 1.0,
            "ripple_z_criterion": 1.5,
            "rows_hold_swr": False,
        }
    ).to_parquet(criterion / "e1.parquet")
    run = event_features_root() / EVENT_FEATURES_RUN
    run.mkdir(parents=True)
    epochs = {"e1": {"status": "ok"}, "e2": {"status": "failed"}}
    (run / "manifest.json").write_text(json.dumps({"epochs": epochs}))
    _features(5, 2.0).to_parquet(output_path("e1", run, "features"))
    _features(5, 9.0).to_parquet(output_path("e2", run, "features"))
    evidence = event_spike_evidence_dir() / "e1"
    evidence.mkdir(parents=True)
    pd.DataFrame(
        {
            "epoch_id": ["e1"],
            "start_time": 1.0,
            "end_time": 1.2,
            "dll_per_spike": 0.3,
            "group": "nonSWR",
        }
    ).to_parquet(evidence / SPIKE_EVIDENCE_METRICS)


def test_load_events_reads_the_features_of_the_ok_epochs_only(processed_data_root):
    _lay_out_inputs()

    events, record = structure.load_events()

    run = event_features_root() / EVENT_FEATURES_RUN
    assert events["features_z_max"].tolist() == [2.0]
    assert events["n_spikes"].tolist() == [7]
    assert events["dll_per_spike"].tolist() == [0.3]
    assert record["inputs"]["event_features_run"] == {
        "run": EVENT_FEATURES_RUN,
        "n_files": 1,
        "sha256": files_sha256([output_path("e1", run, "features")]),
    }
    assert record["checks"]["peak_differs_from_features_run"] == 0


def test_measure_refuses_a_features_run_without_an_epochs_record(processed_data_root):
    run = event_features_root() / EVENT_FEATURES_RUN
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"settings": {}}))

    with pytest.raises(ValueError, match="no readable run manifest"):
        structure.run_measure(argparse.Namespace(epochs=None, n_workers=1))
