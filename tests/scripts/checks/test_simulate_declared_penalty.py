"""The declared-penalty simulation: its bins summed to cells, and its table
through the simulation engine and the runner."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.regression import simulation as engine
from continuum_swr_replay.analysis.regression.contrasts import evaluate_contrast
from continuum_swr_replay.analysis.regression.likelihoods import Poisson
from continuum_swr_replay.analysis.regression.record import NoFit
from continuum_swr_replay.analysis.regression.sessions import fit_sessions
from continuum_swr_replay.tables.io import read_table
from scripts.checks import run_simulations
from scripts.checks import simulate_declared_penalty as sim

FIXTURE = (
    Path(__file__).parents[2]
    / "analysis"
    / "regression"
    / "data"
    / "numpy_reference"
    / "simulation_declared_penalty.json"
)


def _fit(data: dict, log_penalty: float):
    pooled = pd.concat(data.values(), ignore_index=True)
    design = sim.design_at(pooled, log_penalty)
    record = fit_sessions(
        design,
        data,
        Poisson(),
        model="m",
        kind="k",
        response="count",
        unit="cell",
        table_sha256={},
    )[0]
    c = sim.contrast_vector(design)
    estimate = evaluate_contrast(c, np.array(record.coef), np.array(record.vcov))
    return record, estimate


def test_summed_bins_fit_as_the_bins() -> None:
    """One session fitted on its bins and on its bins summed over equal times:
    the same edf, contrast and standard error."""
    rng = np.random.default_rng(3)
    visit, time, exposure = sim.visit_bins(sim.durations(rng, sim.N_VISITS))
    mean = exposure * np.exp(sim.CURVES["rise"](time))
    count = rng.poisson(mean * 120.0 / mean.sum()).astype(float)
    unique, summed, cell = sim.cells(time, exposure)
    key = ("a", "20260001", "02_r1")
    bins = pd.DataFrame(
        {sim.COLUMN: time, "exposure": exposure, "count": count, "cell": visit}
    )
    cells = pd.DataFrame(
        {
            sim.COLUMN: unique,
            "exposure": summed,
            "count": np.bincount(cell, weights=count),
            "cell": np.arange(unique.size),
        }
    )
    assert unique.size < time.size / 5
    on_bins, on_cells = _fit({key: bins}, 3.0), _fit({key: cells}, 3.0)
    assert on_bins[0].edf[sim.TERM] == pytest.approx(
        on_cells[0].edf[sim.TERM], abs=1e-8
    )
    np.testing.assert_allclose(on_bins[1], on_cells[1], rtol=1e-7)


def test_planted_contrast() -> None:
    """The Q1 contrast of the rise, a smoothstep to log 3 at 3 s, on the 0.01 s
    grids of [2, 4] and [0, 1]; the flat curve's is zero."""

    def rise(s):
        u = np.minimum(s / 3.0, 1.0)
        return np.log(3.0) * (3 * u**2 - 2 * u**3)

    later = rise(np.linspace(2.0, 4.0, 201)).mean()
    first = rise(np.linspace(0.0, 1.0, 101)).mean()
    assert sim.planted_contrast("rise") == pytest.approx(later - first, rel=1e-12)
    assert sim.planted_contrast("flat") == 0.0


#: The table's columns, in order.
COLUMNS = [
    "target_edf",
    "curve",
    "log_penalty",
    "calibration_median_edf",
    "median_edf",
    "planted",
    "mean_estimate",
    "bias",
    "bias_share",
    "sd",
    "bias_se",
    "coverage",
    "coverage_se",
    "excludes_zero",
    "excludes_zero_se",
    "reject_day",
    "reject_day_se",
    "reject_t_reference",
    "reject_t_reference_se",
    "share_converged",
    "n_sessions_used_min",
    "n_sessions_used_mean",
    "n_data_sets",
    "n_sessions",
    "n_draws",
]


def test_the_table_on_a_tiny_cohort(tmp_path, monkeypatch) -> None:
    """The quick run (five animals of one day of two sessions, two data sets
    per curve), one target, through the runner: one row per curve, the
    columns of the table, the declared penalty's median edf at its target,
    every fit converged, and the same table on a second run."""
    monkeypatch.setattr(sim, "TARGETS", (6.0,))
    run_simulations.main(
        ["--model", "declared_penalty", "--output-dir", str(tmp_path), "--quick"]
    )
    table = read_table(tmp_path / "declared_penalty.parquet")

    assert list(table.columns) == COLUMNS
    assert list(table["curve"]) == ["rise", "flat"]
    assert (table["target_edf"] == 6.0).all()
    assert table["calibration_median_edf"].iloc[0] == pytest.approx(6.0, abs=0.01)
    assert (table["share_converged"] == 1.0).all()
    assert (table["n_data_sets"] == 2).all() and (table["n_sessions"] == 10).all()
    assert (table["n_sessions_used_min"] == 10).all()
    assert (table["n_sessions_used_mean"] == 10.0).all()
    # Five recording days: fewer than six, so the day-level p is NaN and no
    # data set rejects by it.
    assert (table["reject_day"] == 0.0).all()
    for column in ("reject_t_reference", "reject_day", "excludes_zero"):
        assert table[column].between(0.0, 1.0).all()
        assert f"{column}_se" in table
    assert table["planted"].iloc[1] == 0.0 and np.isnan(table["bias_share"].iloc[1])
    rise = table.iloc[0]
    assert rise["bias_share"] == pytest.approx(rise["bias"] / rise["planted"])
    assert table.attrs["calibration"] == {
        "target_edf_6": {
            "target_edf": 6.0,
            "log_penalty": rise["log_penalty"],
            "calibration_median_edf": rise["calibration_median_edf"],
        }
    }
    again = run_simulations.run("declared_penalty", quick=True)
    pd.testing.assert_frame_equal(again, table, check_like=True)


def test_the_contrast_vector_gives_the_planted_contrast() -> None:
    """The least-squares projection of the planted rise on the smooth's
    basis (with an intercept, which the smooth's constraint leaves out), on a
    0.01 s grid over the term's range: ``contrast_vector`` applied to it is
    the planted contrast to 0.02. A contrast that lost its reference window
    (the mean over 0 to 1 s) would be off by about 0.10."""
    grid = np.arange(0.0, 20.0 + 1e-9, 0.01)
    design = sim.design_at(pd.DataFrame({sim.COLUMN: grid, "exposure": 1.0}), 0.0)
    columns = design.term_columns[sim.TERM]
    X = design.build(pd.DataFrame({sim.COLUMN: grid, "exposure": 1.0})).X[:, columns]
    basis = np.column_stack([np.ones(grid.size), X])
    projected = np.linalg.lstsq(basis, sim.CURVES["rise"](grid), rcond=None)[0]
    coef = np.zeros(len(design.coef_names))
    coef[columns] = projected[1:]
    planted = sim.planted_contrast("rise")
    assert sim.contrast_vector(design) @ coef == pytest.approx(planted, abs=0.02)


def test_a_session_without_a_converged_fit_is_counted_and_warned(
    monkeypatch,
) -> None:
    """One session's fit not converged in every data set: it leaves the
    estimate, ``n_sessions_used_*`` count 9 of the 10 sessions,
    ``share_converged`` is 0.9, and the runner warns for each curve."""
    real = engine.fit_sessions

    def one_failed(*args, **kwargs):
        records = real(*args, **kwargs)
        first = records[0]
        records[0] = replace(
            first,
            status="step limit reached",
            no_fit=NoFit.OTHER_STATUS,
            coef=[np.nan] * len(first.coef),
        )
        return records

    monkeypatch.setattr(engine, "fit_sessions", one_failed)
    monkeypatch.setattr(sim, "TARGETS", (6.0,))
    table = run_simulations.run("declared_penalty", quick=True)
    assert (table["n_sessions_used_min"] == 9).all()
    assert (table["share_converged"] == 0.9).all()
    assert (table["n_sessions"] == 10).all()
    warnings = table.attrs["warnings"]
    assert len(warnings) == 4
    assert warnings[0] == (
        "warning: scenario rise, model target_edf_6, estimand Q1: 10.0 % of the "
        "unit fits tried did not converge"
    )
    assert warnings[1].startswith(
        "warning: scenario rise, model target_edf_6, estimand Q1: 10.0 % of the "
        "units of the fit have no value (other_status 10.0 %)"
    )


@pytest.mark.slow
def test_matches_the_fixed_seed_fixture() -> None:
    """The quick run's table at every target equals the fixed-seed fixture,
    to the tolerance of the solver's port (``1e-11 + 1e-9 |b|``)."""
    stored = json.loads(FIXTURE.read_text())["table"]
    table = json.loads(
        run_simulations.run("declared_penalty", quick=True).to_json(orient="records")
    )
    assert len(table) == len(stored) == 2 * len(sim.TARGETS)
    for got, want in zip(table, stored, strict=True):
        assert got.keys() == want.keys()
        for column, value in want.items():
            if isinstance(value, float):
                assert got[column] == pytest.approx(
                    value, rel=1e-9, abs=1e-11, nan_ok=True
                ), column
            else:
                assert got[column] == value, column
