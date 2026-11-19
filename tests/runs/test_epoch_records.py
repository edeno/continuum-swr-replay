"""A run's per-epoch records and the manifest gathered from them."""

import json

import pytest

from continuum_swr_replay.runs import epoch_records as er


def _header(sha="x", path="/checkout/a/table", setting=5, commit="abc"):
    return {
        "produced_by": "runner",
        "settings": {"n_folds": setting},
        "inputs": {"event_table": {"sha256": sha, "path": path}},
        "provenance": {"commit": commit},
    }


def test_an_ok_record_under_the_same_settings_and_inputs_is_not_run_again(tmp_path):
    header = _header()
    er.write_record(tmp_path, "e1", "scores", header, {"status": "ok"})
    er.write_record(tmp_path, "e2", "scores", header, {"status": "failed"})
    epochs = ["e1", "e2", "e3"]
    assert er.epochs_to_run(tmp_path, "scores", epochs, header, False) == ["e2", "e3"]
    assert er.epochs_to_run(tmp_path, "scores", epochs, header, True) == epochs
    # another checkout of the same input is the same run
    moved = _header(path="/checkout/b/table", commit="def")
    assert er.epochs_to_run(tmp_path, "scores", epochs, moved, False) == ["e2", "e3"]
    # other settings or another input are not
    assert (
        er.epochs_to_run(tmp_path, "scores", epochs, _header(setting=3), False)
        == epochs
    )
    assert (
        er.epochs_to_run(tmp_path, "scores", epochs, _header(sha="y"), False) == epochs
    )


def test_the_manifest_gathers_every_record_and_refuses_another_runs(tmp_path):
    er.write_record(tmp_path, "e1", "scores", _header(commit="abc"), {"status": "ok"})
    er.write_record(
        tmp_path, "e2", "scores", _header(commit="def"), {"status": "ok", "n": 3}
    )
    (tmp_path / "e1_other.json").write_text("{}")  # another table's record
    manifest = er.gather_manifest(tmp_path, "scores", _header())
    written = json.loads((tmp_path / er.MANIFEST).read_text())
    assert written == manifest
    assert set(manifest["epochs"]) == {"e1", "e2"}
    assert manifest["epochs"]["e2"]["n"] == 3
    assert manifest["provenance"] == [{"commit": "abc"}, {"commit": "def"}]
    er.write_record(tmp_path, "e3", "scores", _header(sha="y"), {"status": "ok"})
    with pytest.raises(ValueError, match="other settings or inputs"):
        er.gather_manifest(tmp_path, "scores", _header())
