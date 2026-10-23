"""The runners' file writing (``runs.io``): strict JSON, atomic writes, hashes.

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
import pytest

from continuum_swr_replay.runs.io import (
    array_sha256,
    json_ready,
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
