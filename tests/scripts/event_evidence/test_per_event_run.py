"""The per-event runners' shared epoch loop (records, the gathered manifest,
re-runs and failures) and the epoch inputs."""

import json

import numpy as np
import pandas as pd
import pytest

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


def _epoch_inputs(tmp_path, monkeypatch, loader_time):
    """``epoch_inputs`` with the decode frame, the loader and the detector
    replaced at their boundaries; the frame's time is 0, 0.002, 0.004 s."""
    import sys
    import types

    from continuum_swr_replay.analysis.event_definition import analyzed_time

    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0]}, index=[0.0, 0.002, 0.004])
    monkeypatch.setattr(analyzed_time, "read_dot_with_swr_source", lambda path: frame)
    monkeypatch.setattr(
        per_event,
        "decode_paths",
        lambda epoch_id, data_dir: types.SimpleNamespace(data_over_time="d"),
    )
    terms = types.ModuleType("terms")
    terms.load_marks_and_training_mask = lambda epoch_id, trials: {"time": loader_time}
    terms.load_detector = lambda epoch_id, data_dir: "detector"
    monkeypatch.setitem(
        sys.modules,
        "continuum_swr_replay.analysis.event_evidence.state_evidence.terms",
        terms,
    )
    return frame


def test_epoch_inputs_read_the_detector_only_when_asked(tmp_path, monkeypatch):
    frame = _epoch_inputs(tmp_path, monkeypatch, np.array([0.0, 0.002, 0.004]))
    epoch = per_event.epoch_inputs("e", tmp_path)
    assert epoch.detector == "detector" and epoch.frame is frame
    np.testing.assert_array_equal(epoch.time, [0.0, 0.002, 0.004])
    assert per_event.epoch_inputs("e", tmp_path, detector=False).detector is None


def test_epoch_inputs_refuse_a_loader_on_another_time_axis(tmp_path, monkeypatch):
    _epoch_inputs(tmp_path, monkeypatch, np.array([0.0, 0.002, 0.005]))
    with pytest.raises(ValueError, match="the loader's time axis is not the decode's"):
        per_event.epoch_inputs("e", tmp_path, detector=False)
