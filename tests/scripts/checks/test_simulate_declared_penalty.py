"""The declared-penalty simulation: its bins summed to cells, and its table."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.regression.contrasts import evaluate_contrast
from continuum_swr_replay.analysis.regression.likelihoods import Poisson
from continuum_swr_replay.analysis.regression.sessions import fit_sessions
from continuum_swr_replay.tables.io import read_table
from scripts.checks import simulate_declared_penalty as sim

TINY = sim.Config(
    days_per_animal=(1,) * 5, sessions_per_day=(2, 2), n_data_sets=2, n_draws=200
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


def test_the_table_on_a_tiny_cohort(tmp_path, monkeypatch) -> None:
    """Five animals of one day of two sessions, two data sets per curve, one
    target: one row per curve, the declared penalty's median edf at its
    target, every fit converged, and the same table on a second run."""
    monkeypatch.setattr(sim, "TARGETS", (6.0,))
    monkeypatch.setattr(sim, "QUICK", TINY)
    sim.main(["--output-dir", str(tmp_path), "--quick"])
    table = read_table(tmp_path / f"{sim.OUTPUT_NAME}.parquet")

    assert list(table["curve"]) == ["rise", "flat"]
    assert (table["target_edf"] == 6.0).all()
    assert table["calibration_median_edf"].iloc[0] == pytest.approx(6.0, abs=0.01)
    assert (table["share_converged"] == 1.0).all()
    assert (table["n_data_sets"] == 2).all() and (table["n_sessions"] == 10).all()
    assert table["planted"].iloc[1] == 0.0 and np.isnan(table["bias_share"].iloc[1])
    rise = table.iloc[0]
    assert rise["bias_share"] == pytest.approx(rise["bias"] / rise["planted"])
    pd.testing.assert_frame_equal(sim.run(TINY), table, check_like=True)
