"""End-to-end tests for the golden-baseline freeze/compare safety net.

These scripts are the refactor safety net: freeze the canonical tables, then
byte-diff a candidate against the frozen copy. A regression here is
catastrophic and silent -- a green compare tells a reviewer outputs are
preserved -- so these tests exercise the actual CLI, including the case that
mattered: the baseline is written *under* the candidate root, so a naive
path-prefix dispatch reads the candidate for both sides and always passes.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
_COMPARE = REPO / "scripts/qa/compare_to_golden_baseline.py"


def _stop_frame(value: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "animal": ["chimi", "chimi"],
            "session_date": [20200212, 20200212],
            "epoch": ["02_r1", "02_r1"],
            "stop_id": [10, 11],
            "n_selected_events": [1.0, value],
        }
    )


def _write(root: Path, value: float) -> None:
    from continuum_swr_replay.tables.io import write_table

    root.mkdir(parents=True, exist_ok=True)
    write_table(_stop_frame(value), root / "stop_table")


def _compare(candidate: Path, baseline: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(_COMPARE),
            "--candidate",
            str(candidate),
            "--baseline",
            str(baseline),
            "--tables",
            "stop_table",
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
    )


def test_compare_passes_on_identical_nested_baseline(tmp_path):
    """Baseline nested under the candidate root (the default layout) must still
    compare candidate-vs-baseline, not candidate-vs-candidate."""
    candidate = tmp_path / "replay_content"
    baseline = candidate / "_golden_baseline_v5"
    _write(candidate, 2.0)
    _write(baseline, 2.0)
    result = _compare(candidate, baseline)
    assert result.returncode == 0, result.stderr


def test_compare_detects_drift_with_nested_baseline(tmp_path):
    """The regression this locks: a drifted candidate against a nested baseline
    must fail, not silently pass by reading the candidate for both sides."""
    candidate = tmp_path / "replay_content"
    baseline = candidate / "_golden_baseline_v5"
    _write(baseline, 2.0)
    _write(candidate, 9.0)  # candidate differs from the frozen baseline
    result = _compare(candidate, baseline)
    assert result.returncode != 0
    assert "n_selected_events" in result.stderr


def test_compare_allows_declared_drift(tmp_path):
    """A column named in --expected-drift-columns may differ without failing;
    only undeclared drift fails."""
    candidate = tmp_path / "replay_content"
    baseline = candidate / "_golden_baseline_v5"
    _write(baseline, 2.0)
    _write(candidate, 9.0)
    result = subprocess.run(
        [
            sys.executable,
            str(_COMPARE),
            "--candidate",
            str(candidate),
            "--baseline",
            str(baseline),
            "--tables",
            "stop_table",
            "--expected-drift-columns",
            "n_selected_events",
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert result.returncode == 0, result.stderr


def test_a_frozen_baseline_copies_no_symlink(tmp_path, monkeypatch):
    """A symlink left at an output's old path is not frozen as a second copy."""
    import importlib

    from continuum_swr_replay.paths import CANONICAL_TABLE_BASENAMES, table_path

    freeze = importlib.import_module("scripts.qa.freeze_regression_baseline")
    root = tmp_path / "replay_content"
    per_bin = root / "models_per_bin_rate"
    per_bin.mkdir(parents=True)
    for name in CANONICAL_TABLE_BASENAMES:
        pd.DataFrame({"x": [1]}).to_parquet(table_path(name, root))
    (per_bin / "all_models.pkl").write_bytes(b"fit")
    (per_bin / "qvals_table.parquet").symlink_to(root / "qvals_table.parquet")
    monkeypatch.setattr(freeze, "RC", root)

    out = freeze.freeze("test")

    assert sorted(p.name for p in (out / "models_per_bin_rate").iterdir()) == [
        "all_models.pkl"
    ]
    assert (out / "qvals_table.parquet").is_file()
