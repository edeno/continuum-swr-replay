"""The runners' shared epoch-output writer and run-id check (``runs.io``, ``runs.cli``).

``replace_outputs`` leaves an epoch's folder holding exactly the outputs it was
given, or, when a write fails, none of them. ``parse_run_id`` is the argparse
``type`` every runner's run-id option uses; ``fit_figure_parser`` is the command
line of the scripts that plot stored fits.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.runs import io as runs_io
from continuum_swr_replay.runs.cli import fit_figure_parser, parse_run_id
from continuum_swr_replay.runs.io import replace_outputs, sha256_file
from continuum_swr_replay.tables.io import read_table


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

    written = replace_outputs(paths, {"events": _frame(), "sidecar": {"x": 1.5}})

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


def test_an_output_not_among_the_files_is_refused_before_any_write(tmp_path):
    """A misspelt name would drop that table without a word."""
    paths = _paths(tmp_path)
    _previous_run(paths)

    with pytest.raises(ValueError, match=r"\['evnets'\] are not among"):
        replace_outputs(paths, {"evnets": _frame(), "sidecar": {"x": 1.5}})

    assert all(path.read_bytes() == b"an earlier run" for path in paths.values())


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
    real_write_table = runs_io.write_table
    calls = []

    def interrupted_second_write(frame, path):
        calls.append(path)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return real_write_table(frame, path)

    monkeypatch.setattr(runs_io, "write_table", interrupted_second_write)

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


def test_the_fit_figure_parser_holds_only_the_stale_opt_out():
    parser = fit_figure_parser("Plots a fit.")

    assert parser.description == "Plots a fit."
    assert vars(parser.parse_args([])) == {"allow_stale": False}
    assert parser.parse_args(["--allow-stale"]).allow_stale is True
    with pytest.raises(SystemExit) as stopped:
        parser.parse_args(["--smoke"])
    assert stopped.value.code == 2


def test_every_run_id_option_is_checked():
    """Every ``--…run-id`` option of a supported script names one run folder:
    its argparse ``type`` is ``parse_run_id``, so ``../x`` cannot reach a path."""
    import ast
    from pathlib import Path

    scripts = Path(__file__).resolve().parents[2] / "scripts"
    skipped = scripts / "exploratory"
    options, unchecked = [], []
    for path in sorted(scripts.rglob("*.py")):
        if path.is_relative_to(skipped):
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if not (
                isinstance(node, ast.Call)
                and getattr(node.func, "attr", None) == "add_argument"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and str(node.args[0].value).endswith("run-id")
            ):
                continue
            where = f"{path.relative_to(scripts)}:{node.lineno}"
            options.append(where)
            kind = {k.arg: k.value for k in node.keywords}.get("type")
            if getattr(kind, "id", None) != "parse_run_id":
                unchecked.append(where)

    assert len(options) >= 12
    assert unchecked == []
