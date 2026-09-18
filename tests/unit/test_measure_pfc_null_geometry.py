"""Denominator contracts for the mPFC null-geometry audit."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import continuum_swr_replay.analysis.pfc.load_population as population_loader
from scripts.qa.measure_pfc_null_geometry import (
    OBSERVED_COLUMN,
    PVALUE_COLUMN,
    non_positive_section,
)


def _population(pvalues: list[float], observed: list[float]) -> pd.DataFrame:
    """A population table carrying every class the section reports on."""
    columns: dict[str, list[float]] = {}
    for column in PVALUE_COLUMN.values():
        columns[column] = list(pvalues)
    for column in OBSERVED_COLUMN.values():
        columns[column] = list(observed)
    return pd.DataFrame(columns)


def test_non_positive_section_counts_only_finite_pvalues(monkeypatch):
    population = _population([0.01, np.inf, np.nan], [0.0, -1.0, -1.0])

    monkeypatch.setattr(
        population_loader,
        "load_population_data",
        lambda *_args, **_kwargs: (population, {}, np.array([])),
    )

    _, summary = non_positive_section(Path("unused"), alpha=0.05, min_firing_rate=None)

    assert list(summary["class"]) == list(PVALUE_COLUMN)
    row = summary.loc[summary["class"] == "engagement"].iloc[0]
    assert row["n_units"] == 3
    assert row["n_tested"] == 1
    assert row["n_called"] == 1
    assert row["n_called_with_non_positive_gain"] == 1


def test_a_class_missing_its_statistic_is_named_rather_than_skipped(monkeypatch):
    """Dropping the class would read as "nothing to report" for it."""
    population = _population([0.01], [1.0]).drop(columns=OBSERVED_COLUMN["SWR"])

    monkeypatch.setattr(
        population_loader,
        "load_population_data",
        lambda *_args, **_kwargs: (population, {}, np.array([])),
    )

    with pytest.raises(KeyError, match="SWR"):
        non_positive_section(Path("unused"), alpha=0.05, min_firing_rate=None)
