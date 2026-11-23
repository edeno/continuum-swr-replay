"""Planted scenarios of the mPFC unit models, on a synthetic session's design."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.pfc import unit_scenarios as scenarios
from continuum_swr_replay.analysis.pfc.design import symmetric_lags
from continuum_swr_replay.analysis.pfc.unit_design import (
    Kernel,
    KernelPenalty,
    build_unit_design,
)

FS = 500.0
LAGS = symmetric_lags(0.25, FS)


@pytest.fixture
def design():
    """40 s, every bin fit; SWRs at 5, 15, 25 s, replay at 10 and 30 s with
    marks -1 and 1: every event is more than 0.5 s from the others."""
    time = np.arange(20_000) / FS
    fit_mask = pd.Series(True, index=pd.Index(time, name="time"))
    events = {
        "SWR": pd.DataFrame({"start_time": [5.0, 15.0, 25.0]}),
        "non_local_no_SWR": pd.DataFrame(
            {"start_time": [10.0, 30.0], "mark_z": [-1.0, 1.0]}
        ),
    }
    return build_unit_design(
        events,
        fit_mask,
        kernels=[
            Kernel("SWR"),
            Kernel("non_local_no_SWR"),
            Kernel("non_local_no_SWR", "mark_z"),
        ],
        lags=LAGS,
        sampling_frequency=FS,
        penalty=KernelPenalty("declared", log_penalty=2.0),
    )


def test_no_modulation_is_each_unit_at_its_rate(design):
    planted = scenarios.no_modulation(design, np.random.default_rng(0), 4, FS)
    np.testing.assert_allclose(planted.mu, np.tile(planted.rate_hz / FS, (20_000, 1)))
    assert all(f == 0 for f in planted.modulated_fraction.values())
    assert all((p == 0).all() for p in planted.peaks.values())


def test_a_planted_curve_adds_to_the_log_rate_at_each_lag(design):
    planted = scenarios.planted_share(
        design, np.random.default_rng(1), 6, FS, kernel="k(SWR)", share=0.5
    )
    assert planted.modulated["k(SWR)"].sum() == 3
    assert planted.modulated_fraction["k(SWR)"] == 0.5
    log_ratio = np.log(planted.mu) - np.log(planted.rate_hz / FS)
    onset = int(5.0 * FS)
    np.testing.assert_allclose(
        log_ratio[onset + LAGS], planted.curves["k(SWR)"], atol=1e-12
    )
    # one modulated unit is suppressed, its peak the curve's negative
    peaks = planted.peaks["k(SWR)"][planted.modulated["k(SWR)"]]
    np.testing.assert_allclose(
        np.sort(np.abs(peaks)), scenarios.REALISTIC_GAIN, rtol=1e-3
    )
    assert (peaks < 0).sum() == 1


def test_a_curve_in_the_basis_is_the_design_times_its_coefficients(design):
    rng = np.random.default_rng(2)
    beta = rng.normal(0, 0.3, design.spline_basis.shape[1])
    planted = scenarios.kernel_by_kind_and_mark(
        design,
        rng,
        1,
        FS,
        curves={"k(non_local_no_SWR:mark_z)": design.spline_basis @ beta},
    )
    coef = np.zeros(design.X.shape[1])
    coef[0] = np.log(planted.rate_hz[0] / FS)
    coef[design.kernel_columns["k(non_local_no_SWR:mark_z)"]] = beta
    np.testing.assert_allclose(np.log(planted.mu[:, 0]), design.X @ coef, atol=1e-10)
    # the mark's sign: a -1 mark plants the curve negated
    onset = int(10.0 * FS)
    np.testing.assert_allclose(
        np.log(planted.mu[onset + LAGS, 0] / (planted.rate_hz[0] / FS)),
        -(design.spline_basis @ beta),
        atol=1e-10,
    )


def test_low_rate_units_cycle_through_the_low_rates(design):
    planted = scenarios.low_rate_units(
        design, np.random.default_rng(3), 6, FS, kernel="k(SWR)"
    )
    np.testing.assert_array_equal(planted.rate_hz, [0.2, 0.5, 1.0, 0.2, 0.5, 1.0])
    assert planted.modulated["k(SWR)"].sum() == 3


def test_a_kernel_from_a_peri_event_rate(design):
    lag = design.lag_time
    # a sharp 10 ms step of x3 at +20 ms on a 4 Hz unit: outside the basis
    rate = np.where((lag >= 0.02) & (lag < 0.03), 12.0, 4.0)[:, None]
    planted = scenarios.outside_basis_kernel(
        design, FS, kernel="k(SWR)", peri_event_rate=rate
    )
    assert planted.rate_hz[0] == pytest.approx(4.0)
    curve = planted.curves["k(SWR)"][:, 0]
    assert lag[np.argmax(curve)] == pytest.approx(0.025, abs=0.002)
    assert 0 < curve.max() < np.log(3.0)
    with pytest.raises(ValueError):
        scenarios.outside_basis_kernel(
            design, FS, kernel="k(SWR)", peri_event_rate=rate * 0
        )


def test_gains_differ_between_animals_and_sessions():
    animals = np.array(["a"] * 4 + ["b"] * 4 + ["c"] * 4)
    gains, planted = scenarios.animal_session_gains(
        np.random.default_rng(4), animals, animal_sd=0.2, session_sd=0.0
    )
    assert gains.shape == (12,)
    np.testing.assert_allclose(gains[:4], planted["a"])
    assert planted["population"] == pytest.approx(
        np.mean([planted["a"], planted["b"], planted["c"]])
    )


def test_every_scenario_says_what_it_validates_and_what_fails():
    names = [s.name for s in scenarios.SCENARIOS]
    assert len(set(names)) == len(names) == 6
    assert all(s.validates and s.failure for s in scenarios.SCENARIOS)
    assert set(names) <= set(scenarios.__all__)


def test_a_curve_for_a_kernel_the_design_lacks_is_refused(design):
    with pytest.raises(ValueError, match="lacks"):
        scenarios.expected_counts(
            design, np.ones(1), {"k(MUA)": np.zeros((LAGS.size, 1))}, FS
        )
