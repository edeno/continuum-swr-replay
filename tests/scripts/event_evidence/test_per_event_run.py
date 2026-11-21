"""The per-event runners' shared epoch loop (records, the gathered manifest,
re-runs and failures) and the held-out runs' fold designs."""

import json

import numpy as np
import pandas as pd
import pytest
import sklearn

from continuum_swr_replay.analysis.event_evidence import tetrode_holdout as th
from scripts.event_evidence import _per_event_run as per_event

HEADER = {
    "produced_by": "runner",
    "provenance": {"commit": "abc"},
    "settings": {"seed": 1},
    "inputs": {"event_table": {"sha256": "x", "path": "/a"}},
}


def _task(epoch_id: str, calls: str) -> dict:
    with open(calls, "a") as handle:
        handle.write(epoch_id + "\n")
    if epoch_id == "bad":
        raise RuntimeError("out of memory")
    return {"status": "ok", "n": len(epoch_id)}


def _run(tmp_path, epochs, overwrite=False):
    per_event.run(
        epochs=epochs,
        task=_task,
        task_args=(str(tmp_path / "calls.txt"),),
        output=tmp_path / "run",
        name="scores",
        header=HEADER,
        overwrite=overwrite,
        max_workers=1,
    )


def test_the_loop_records_each_epoch_gathers_and_skips_done_epochs(tmp_path):
    _run(tmp_path, ["e1", "e22"])
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert manifest["epochs"]["e22"]["n"] == 3
    assert manifest["inputs"] == HEADER["inputs"]
    # a re-run adds an epoch and runs only it
    _run(tmp_path, ["e1", "e22", "e3"])
    calls = (tmp_path / "calls.txt").read_text().split()
    assert sorted(calls) == ["e1", "e22", "e3"]
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert set(manifest["epochs"]) == {"e1", "e22", "e3"}


def test_a_failed_epoch_is_recorded_and_the_run_exits_with_an_error(tmp_path):
    with pytest.raises(SystemExit, match="1 epochs failed"):
        _run(tmp_path, ["e1", "bad"])
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert manifest["epochs"]["bad"]["status"] == "failed"
    assert "out of memory" in manifest["epochs"]["bad"]["error"]
    assert manifest["epochs"]["e1"]["status"] == "ok"


def test_shards_split_the_sorted_epochs():
    epochs = ["e3", "e1", "e2", "e4"]
    assert per_event.select_epochs(epochs, [0], 2) == ["e1", "e3"]
    assert per_event.select_epochs(epochs, [1], 2) == ["e2", "e4"]
    assert per_event.select_epochs(epochs, None, 1) == ["e1", "e2", "e3", "e4"]


def test_the_header_names_the_fold_design_and_its_margin():
    assert per_event.fold_settings("trials") == {
        "fold_design": "trials",
        "fold_classes": "to_well",
        "margin_s": 1.0,
        "max_segment_s": th.MAX_SEGMENT_S,
        "min_trial_bin_share": th.MIN_TRIAL_BIN_SHARE,
        "sklearn_version": sklearn.__version__,
    }
    assert per_event.fold_settings("contiguous") == {
        "fold_design": "contiguous",
        "margin_s": 0.004,
        "margin_bins": 2,
    }
    with pytest.raises(ValueError, match="unknown fold design"):
        per_event.fold_settings("chunks")


def test_epoch_folds_deal_the_trials_and_the_time_after_them_into_folds():
    """Six trials of 1.5 s, to wells 2 and 3 in turn, from 0.5 s into a 10 s
    epoch, the last ending at 9.5 s: eight segments (the time before the first
    trial, the six trials, the time after the last), none longer than the cap."""
    time = np.arange(0.0, 10.0, 0.002)
    trials = pd.DataFrame(
        {
            # out of order: the folds sort the trials by departure
            "movement_start_time": [2.0, 0.5, 3.5, 5.0, 6.5, 8.0],
            "end_time": [3.5, 2.0, 5.0, 6.5, 8.0, 9.5],
            "to_well": [3, 2, 2, 3, 2, 3],
        }
    )
    folds, margin_bins, counts = per_event.epoch_folds(time, trials, "trials")
    starts = np.array([0.0, 0.5, 2.0, 3.5, 5.0, 6.5, 8.0, 9.5])
    classes = np.r_[th.BEFORE_FIRST_TRIAL, 2, 3, 2, 3, 2, 3, th.AFTER_LAST_TRIAL]
    np.testing.assert_array_equal(
        folds, th.trial_folds(time, starts, classes, th.N_FOLDS)
    )
    assert margin_bins == 500
    assert counts == {
        "n_segments": 8,
        "n_groups": 8,
        "trial_bin_share": pytest.approx(0.9),
    }


def test_epoch_folds_of_the_contiguous_design_are_the_default_blocks():
    assert per_event.epoch_folds(np.arange(10.0), None, "contiguous") == (
        None,
        th.MARGIN_BINS,
        {"n_segments": None, "n_groups": None, "trial_bin_share": None},
    )
    with pytest.raises(ValueError, match="unknown fold design"):
        per_event.epoch_folds(np.arange(10.0), None, "chunks")
