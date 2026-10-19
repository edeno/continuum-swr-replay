"""The runners' shared epoch-output writer and run-id check (``decoding_common``).

``replace_outputs`` leaves an epoch's folder holding exactly the outputs it was
given, or, when a write fails, none of them. ``parse_run_id`` is the argparse
``type`` every runner's run-id option uses.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay import decoding_common
from continuum_swr_replay.decoding_common import parse_run_id, replace_outputs
from continuum_swr_replay.table_io import read_table, sha256_file


def _paths(tmp_path):
    return {
        "events": tmp_path / "sim_events.parquet",
        "observed": tmp_path / "sim_observed.parquet",
        "sidecar": tmp_path / "sim_sidecar.json",
    }


def _frame():
    frame = pd.DataFrame({"start_s": [0.5, 1.5], "label": ["a", "b"]})
    frame.attrs = {"run_id": "r"}
    return frame


def _previous_run(paths):
    for path in paths.values():
        path.write_bytes(b"an earlier run")


def test_writes_the_outputs_and_removes_the_others(tmp_path):
    paths = _paths(tmp_path)
    _previous_run(paths)

    written = replace_outputs(
        paths, {"events": _frame(), "sidecar": {"x": 1.5}, "unlisted": _frame()}
    )

    assert set(written) == {"events", "sidecar"}
    for name, record in written.items():
        assert record == {
            "path": str(paths[name]),
            "sha256": sha256_file(paths[name]),
        }
    events = read_table(paths["events"])
    pd.testing.assert_frame_equal(events, _frame())
    assert events.attrs == {"run_id": "r"}
    assert json.loads(paths["sidecar"].read_text()) == {"x": 1.5}
    assert not paths["observed"].exists()
    assert not (tmp_path / "unlisted").exists()


def test_no_outputs_removes_every_file(tmp_path):
    paths = _paths(tmp_path)
    _previous_run(paths)

    assert replace_outputs(paths, {}) == {}
    assert list(tmp_path.iterdir()) == []


def test_a_failing_write_rolls_back_every_output(tmp_path):
    """The table is written, then the sidecar's write fails (strict JSON has
    no NaN): the error is re-raised and no file of the epoch is left, neither
    the one just written nor the earlier run's."""
    paths = _paths(tmp_path)
    _previous_run(paths)

    with pytest.raises(ValueError, match="JSON compliant"):
        replace_outputs(
            paths,
            {"events": _frame(), "observed": _frame(), "sidecar": {"x": np.nan}},
        )

    assert list(tmp_path.iterdir()) == []


def test_an_interrupted_write_rolls_back_too(tmp_path, monkeypatch):
    """A KeyboardInterrupt mid-set leaves no part set either."""
    paths = _paths(tmp_path)
    _previous_run(paths)
    real_write_table = decoding_common.write_table
    calls = []

    def interrupted_second_write(frame, path):
        calls.append(path)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return real_write_table(frame, path)

    monkeypatch.setattr(decoding_common, "write_table", interrupted_second_write)

    with pytest.raises(KeyboardInterrupt):
        replace_outputs(paths, {"events": _frame(), "observed": _frame()})

    assert calls == [paths["events"], paths["observed"]]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("run_id", ["a1b2c3d4e5f6", "smoke_2026-10-19.v2", "..."])
def test_a_plain_name_is_a_run_id(run_id):
    assert parse_run_id(run_id) == run_id


@pytest.mark.parametrize("run_id", ["../escape", ".", "..", "a/b", "with space", ""])
def test_anything_else_is_refused(run_id):
    with pytest.raises(argparse.ArgumentTypeError, match="must name one folder"):
        parse_run_id(run_id)


def test_argparse_reports_a_bad_run_id_as_a_usage_error(capsys):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=parse_run_id)

    with pytest.raises(SystemExit) as stopped:
        parser.parse_args(["--run-id", "../escape"])

    assert stopped.value.code == 2
    assert "argument --run-id: '../escape' must name one folder" in (
        capsys.readouterr().err
    )
    assert parser.parse_args([]).run_id is None
