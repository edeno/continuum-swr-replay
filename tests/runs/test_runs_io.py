"""The runners' file writing (``runs.io``): strict JSON, atomic writes, hashes,
and a run's ``ok`` epochs' tables.

``replace_outputs`` and ``parse_run_id`` are tested in ``test_replace_outputs``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.runs.io import (
    array_sha256,
    is_nonempty_file,
    json_ready,
    ok_table_paths,
    read_complete_run,
    read_manifest,
    recorded_output,
    sha256_file,
    write_json,
)
from continuum_swr_replay.tables.io import umask_file_mode, write_atomically


def test_json_ready_gives_strict_json_types():
    """Mappings (a ``MappingProxyType`` too) become dicts with string keys,
    arrays and tuples lists, numpy scalars Python ones, non-finite floats None."""
    value = {
        1: MappingProxyType({"a": (np.int64(2), np.float32("nan"))}),
        "b": np.array([[1.0, np.inf], [-np.inf, 0.5]]),
        "c": [np.bool_(True), np.float64(1.5), None, "x"],
        "d": -math.inf,
    }

    ready = json_ready(value)

    assert ready == {
        "1": {"a": [2, None]},
        "b": [[1.0, None], [None, 0.5]],
        "c": [True, 1.5, None, "x"],
        "d": None,
    }
    assert type(ready["1"]["a"][0]) is int
    assert type(ready["c"][0]) is bool
    json.dumps(ready, allow_nan=False)  # strict JSON: does not raise


def test_json_ready_leaves_other_objects_as_they_are():
    """A Path is not JSON; it is passed through for the writer to refuse or
    convert (``write_json(..., default=str)``)."""
    path = Path("/a/b")
    assert json_ready(path) is path


def test_write_atomically_gives_a_plain_create_mode_and_leaves_no_temporary(
    tmp_path,
):
    """mkstemp makes the temporary file owner-only; the final file gets the
    mode a plain create would have."""
    path = tmp_path / "x_environment.pkl"

    write_atomically(path, lambda temporary: temporary.write_bytes(b"content"))

    assert path.read_bytes() == b"content"
    assert stat.S_IMODE(os.stat(path).st_mode) == umask_file_mode()
    assert [entry.name for entry in tmp_path.iterdir()] == [path.name]


def test_write_atomically_gives_the_writer_a_temporary_beside_the_file(tmp_path):
    """Not ending in the final suffix, so a glob for the final files never
    matches it."""
    path = tmp_path / "x_environment.pkl"
    seen = []

    write_atomically(path, lambda temporary: seen.append(temporary))

    (temporary,) = seen
    assert temporary.parent == tmp_path
    assert temporary.name.startswith(f".{path.name}.")
    assert temporary.suffix == ".tmp"
    assert not temporary.match("*_environment.pkl")


def test_write_atomically_removes_the_temporary_when_the_write_fails(tmp_path):
    path = tmp_path / "out.json"
    path.write_text("the previous run")

    def half_write(temporary: Path) -> None:
        temporary.write_text("trunc")
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        write_atomically(path, half_write)

    assert path.read_text() == "the previous run"
    assert [entry.name for entry in tmp_path.iterdir()] == [path.name]


@pytest.mark.parametrize(
    ("kwargs", "dumps_kwargs"),
    [({}, {}), ({"indent": 2}, {"indent": 2}), ({"indent": 1}, {"indent": 1})],
)
def test_write_json_writes_what_json_dumps_does(tmp_path, kwargs, dumps_kwargs):
    """The runners' manifests compare a saved settings record on resume, so the
    text must be ``json.dumps``' own."""
    content = {"settings": {"b": [1, 2.5, None], "a": "x"}, "epochs": {}}
    path = tmp_path / "sub" / "manifest.json"

    write_json(path, content, **kwargs)

    assert path.read_text() == json.dumps(content, allow_nan=False, **dumps_kwargs)


def test_write_json_sorts_keys_and_ends_the_line_when_asked(tmp_path):
    """The measurement manifests' own format: sorted keys for some, a final
    newline for most."""
    content = {"b": 1, "a": {"d": 2, "c": 3}}
    path = tmp_path / "manifest.json"

    write_json(path, content, indent=2, sort_keys=True, final_newline=True)
    assert path.read_text() == json.dumps(content, indent=2, sort_keys=True) + "\n"
    write_json(path, content, indent=1, final_newline=True)
    assert path.read_text() == json.dumps(content, indent=1) + "\n"


def test_write_json_refuses_non_finite_floats_and_unknown_objects(tmp_path):
    with pytest.raises(ValueError):
        write_json(tmp_path / "a.json", {"x": math.nan})
    with pytest.raises(TypeError):
        write_json(tmp_path / "b.json", {"x": Path("/p")})
    assert not list(tmp_path.iterdir())


def test_write_json_default_converts_unknown_objects(tmp_path):
    path = tmp_path / "c.json"
    write_json(path, {"x": Path("/p")}, indent=1, default=str)
    assert path.read_text() == json.dumps({"x": "/p"}, indent=1)


def test_hashes(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"abc" * 1000)
    assert sha256_file(path, chunk_bytes=7) == hashlib.sha256(b"abc" * 1000).hexdigest()
    values = np.arange(6, dtype=np.int32).reshape(2, 3)
    expected = hashlib.sha256(values.astype(np.float64).tobytes()).hexdigest()
    assert array_sha256(values) == expected
    assert array_sha256(values.T.copy().T) == expected  # C order, whatever the layout


def test_ok_table_paths_are_the_ok_epochs_tables_sorted_and_may_be_none(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "epochs": {
                    "e2": {"status": "ok"},
                    "e3": {"status": "failed"},
                    "e1": {"status": "ok"},
                }
            }
        )
    )
    assert ok_table_paths(tmp_path, "features") == [
        tmp_path / "e1_features.parquet",
        tmp_path / "e2_features.parquet",
    ]
    manifest.write_text(json.dumps({"epochs": {"e3": {"status": "failed"}}}))
    assert ok_table_paths(tmp_path, "features") == []


def test_a_recorded_output_must_exist_with_its_recorded_sha256(tmp_path):
    """A missing file, a changed one and an output recorded without a sha256
    are each refused; the file's sha256 comes back when it matches."""
    path = tmp_path / "e1_features.parquet"
    record = {"outputs": {"features": {"sha256": "0" * 64}}}

    with pytest.raises(ValueError, match="e1_features.parquet is missing"):
        recorded_output(path, record, "features")
    path.write_bytes(b"table")
    sha = sha256_file(path)
    with pytest.raises(ValueError, match=f"sha256 {sha[:12]}, not the 000000000000"):
        recorded_output(path, record, "features")
    for unhashed in ({}, {"outputs": {}}, {"outputs": {"features": "x"}}):
        with pytest.raises(ValueError, match="not the None"):
            recorded_output(path, unhashed, "features")
    record["outputs"]["features"]["sha256"] = sha
    assert recorded_output(path, record, "features") == sha


def test_a_manifest_naming_another_folder_is_refused_only_when_asked(tmp_path):
    """``names_folder``: a run folder copied or renamed from another run is not
    read as this one; without it the run id is not checked (the state-evidence
    run records none)."""
    run_dir = tmp_path / "run0001"
    run_dir.mkdir()
    path = run_dir / "manifest.json"
    path.write_text(json.dumps({"run_id": "run0002", "epochs": {}}))

    assert read_manifest(run_dir)["run_id"] == "run0002"
    with pytest.raises(ValueError, match="run0002.*run0001"):
        read_manifest(run_dir, names_folder=True)
    path.write_text(json.dumps({"epochs": {}}))
    with pytest.raises(ValueError, match="None"):
        read_manifest(run_dir, names_folder=True)
    path.write_text(json.dumps({"run_id": "run0001", "epochs": {}}))
    assert read_manifest(run_dir, names_folder=True) == {
        "run_id": "run0001",
        "epochs": {},
    }


class TestIsNonemptyFile:
    def test_only_a_written_regular_file_counts(self, tmp_path):
        empty = tmp_path / "empty.npy"
        empty.touch()
        written = tmp_path / "written.npy"
        written.write_bytes(b"x")

        assert not is_nonempty_file(tmp_path / "missing.npy")
        assert not is_nonempty_file(empty)
        assert not is_nonempty_file(tmp_path)
        assert is_nonempty_file(written)


def test_read_complete_run_reads_every_epoch_and_refuses_a_failed_one(tmp_path):
    for epoch in ("e1", "e2"):
        pd.DataFrame({"epoch": [epoch, epoch]}).to_parquet(
            tmp_path / f"{epoch}_scores.parquet"
        )
    manifest = tmp_path / "manifest.json"
    epochs = {"e1": {"status": "ok"}, "e2": {"status": "ok"}}
    manifest.write_text(json.dumps({"epochs": epochs}))
    assert read_complete_run(tmp_path, "scores")["epoch"].tolist() == [
        "e1",
        "e1",
        "e2",
        "e2",
    ]
    epochs["e2"] = {"status": "failed"}
    manifest.write_text(json.dumps({"epochs": epochs}))
    with pytest.raises(ValueError, match="1 epochs"):
        read_complete_run(tmp_path, "scores")
