"""The interval-coverage simulation runs and writes one row per setting and scheme."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

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
    bands = tmp_path / "bands.csv"
    coverage.main(["--out", str(out), "--band-out", str(bands), "--quick"])
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
    band_table = pd.read_csv(bands)
    assert list(band_table.columns) == [
        "setting",
        "whole_curve_coverage",
        "point_coverage",
        "mean_width",
        "coverage_se",
        "n_sets",
        "n_draws",
    ]
    assert len(band_table) == len(coverage.BAND_SETTINGS)


@pytest.mark.slow
def test_the_band_holds_the_whole_curve() -> None:
    """300 data sets of each band setting: whole-curve coverage at least 0.93."""
    for setting in coverage.BAND_SETTINGS:
        row = coverage.simulate_band(setting, 300, 1000).iloc[0]
        assert row.whole_curve_coverage >= 0.93, (setting.name, row)


_DATA = Path(__file__).resolve().parent / "data"


def test_the_committed_tables_say_what_the_fitter_doc_says() -> None:
    """docs/design/regression_fitter.md quotes these tables: the two-level
    rule covers at least 95 % in every setting, the third level is up to 41 %
    wider, and the band holds the whole curve in at least 95 %."""
    table = pd.read_csv(_DATA / "interval_coverage_full.csv")
    two_level = table[table.scheme == "animals_sessions"].set_index("setting")
    three_level = table[table.scheme == "animals_days_sessions"].set_index("setting")
    assert (two_level.coverage >= 0.95).all()
    ratio = three_level.mean_width / two_level.mean_width
    assert round(ratio.max() - 1, 2) == 0.41
    assert three_level.coverage.max() == 1.0
    bands = pd.read_csv(_DATA / "band_coverage_full.csv")
    assert len(bands) == len(coverage.BAND_SETTINGS)
    assert (bands.whole_curve_coverage >= 0.95).all()
