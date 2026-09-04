"""Command-line behaviour of the per-stop rate runner.

The runner writes ``models_per_stop_rate/all_models.pkl`` in place, so an
unrecognised flag must abort instead of silently refitting the canonical
artifact. These tests exercise the parser only: the fitting entry point is
replaced so that a run which reaches it is detectable as a failure.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "fit_per_stop_rate_models",
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "rate"
    / "fit_per_stop_rate_models.py",
)
fpsr = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fpsr)


@pytest.fixture()
def no_fitting(monkeypatch):
    """Replace the fitting entry point with a call recorder."""
    calls: list[tuple] = []

    def _record(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("fit_per_stop_rate_hierarchy was called")

    monkeypatch.setattr(fpsr, "fit_per_stop_rate_hierarchy", _record)
    return calls


def test_help_exits_zero_without_fitting(monkeypatch, capsys, no_fitting):
    monkeypatch.setattr(fpsr.sys, "argv", ["fit_per_stop_rate_models.py", "--help"])

    with pytest.raises(SystemExit) as excinfo:
        fpsr.main()

    assert excinfo.value.code == 0
    assert no_fitting == []
    help_text = capsys.readouterr().out
    assert "--max-per-animal" in help_text
    assert "--no-cv" in help_text


def test_unknown_flag_exits_two_without_fitting(monkeypatch, capsys, no_fitting):
    monkeypatch.setattr(
        fpsr.sys, "argv", ["fit_per_stop_rate_models.py", "--not-a-flag"]
    )

    with pytest.raises(SystemExit) as excinfo:
        fpsr.main()

    assert excinfo.value.code == 2
    assert no_fitting == []
    assert "--not-a-flag" in capsys.readouterr().err
