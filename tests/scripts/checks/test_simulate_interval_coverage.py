"""The interval-coverage simulation runs and writes one row per setting and scheme."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

_SCRIPT = (
    Path(__file__).resolve().parents[3] / "scripts/checks/simulate_interval_coverage.py"
)
_spec = importlib.util.spec_from_file_location("simulate_interval_coverage", _SCRIPT)
coverage = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = coverage  # the module's dataclass looks itself up
_spec.loader.exec_module(coverage)


def test_the_cohort_has_the_shape_of_the_real_one() -> None:
    animal, day = coverage.cohort()
    assert len(set(animal)) == 5
    assert len(set(day)) == sum(coverage.DAYS_PER_ANIMAL)
    assert 200 <= animal.size <= 300


def test_quick_run_writes_the_table(tmp_path: Path) -> None:
    out = tmp_path / "coverage.csv"
    coverage.main(["--out", str(out), "--quick"])
    table = pd.read_csv(out)
    assert list(table.columns) == [
        "setting",
        "scheme",
        "coverage",
        "mean_width",
        "coverage_se",
        "n_sets",
        "n_draws",
    ]
    assert len(table) == len(coverage.SETTINGS) * len(coverage.SCHEMES)
    assert table.coverage.between(0, 1).all() and (table.mean_width > 0).all()
    assert (table.n_sets == 40).all() and (table.n_draws == 200).all()
