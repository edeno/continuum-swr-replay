"""Peak and onset of peri-event curves, and the population timing."""

import numpy as np
import pytest

from continuum_swr_replay.analysis.pfc.timing import (
    peak_and_onset,
    population_timing,
    sign_aligned_mean,
)

# The report window: +/-200 ms at 500 Hz.
LAG_TIME = np.arange(-100, 101) / 500.0


def _box(start_s: float, stop_s: float, height: float) -> np.ndarray:
    return np.where((LAG_TIME >= start_s) & (LAG_TIME <= stop_s), height, 0.0)


def _bump(center_s: float, width_s: float, height: float) -> np.ndarray:
    return height * np.exp(-0.5 * ((LAG_TIME - center_s) / width_s) ** 2)


def test_peak_and_onset_on_planted_curves():
    box = _box(0.020, 0.100, 0.8) + _bump(0.060, 0.010, 0.1)
    dip = _bump(0.050, 0.020, -0.6)
    edge = _box(-0.2, 0.0, 0.5) + _bump(-0.1, 0.01, 0.1)
    curves = np.column_stack([box, dip, edge, np.full_like(LAG_TIME, np.nan)])

    result = peak_and_onset(curves, LAG_TIME)

    assert result.onset_lag[0] == pytest.approx(0.020)
    assert 0.020 <= result.peak_lag[0] <= 0.100
    assert result.peak[1] < 0
    assert result.peak_lag[1] == pytest.approx(0.050)
    # Half of a Gaussian's height is reached 1.1774 SD before its centre.
    assert result.onset_lag[1] == pytest.approx(0.050 - 1.1774 * 0.020, abs=0.002)
    assert result.onset_at_edge[2]
    assert result.onset_lag[2] == pytest.approx(-0.2)
    assert np.isnan(result.peak[3]) and np.isnan(result.onset_lag[3])
    assert not result.onset_at_edge[3]


def test_onset_ignores_a_same_sign_bump_before_a_gap():
    curve = _bump(-0.100, 0.010, 0.6) + _bump(0.050, 0.010, 1.0)

    result = peak_and_onset(curve, LAG_TIME)

    assert result.peak_lag[0] == pytest.approx(0.050)
    assert result.onset_lag[0] > 0.0


def test_onset_stops_at_a_sign_change():
    curve = _bump(-0.020, 0.010, -0.8) + _bump(0.030, 0.015, 1.0)

    result = peak_and_onset(curve, LAG_TIME)

    assert result.peak_lag[0] == pytest.approx(0.030, abs=0.002)
    assert result.onset_lag[0] > -0.020


def test_zero_curve_has_no_peak():
    result = peak_and_onset(np.zeros_like(LAG_TIME), LAG_TIME)

    assert np.isnan(result.peak[0]) and np.isnan(result.onset_lag[0])


def test_sign_aligned_mean_flips_suppressed_units():
    curve = _bump(0.040, 0.020, 0.7)
    curves = np.column_stack([curve, -curve])
    peaks = peak_and_onset(curves, LAG_TIME).peak

    np.testing.assert_allclose(sign_aligned_mean(curves, peaks), curve)


def test_population_timing_bootstrap_is_seeded_and_brackets_the_estimate():
    rng = np.random.default_rng(1)
    centers = rng.normal(0.050, 0.010, size=60)
    curves = np.column_stack([_bump(c, 0.020, 0.5) for c in centers])
    peaks = peak_and_onset(curves, LAG_TIME).peak
    groups = np.repeat(np.arange(12), 5)

    first = population_timing(curves, peaks, groups, LAG_TIME, n_boot=200, seed=3)
    second = population_timing(curves, peaks, groups, LAG_TIME, n_boot=200, seed=3)

    assert first.peak_lag_interval == second.peak_lag_interval
    assert first.onset_lag_interval == second.onset_lag_interval
    assert first.n_units == 60 and first.n_groups == 12
    low, high = first.peak_lag_interval
    assert low <= first.peak_lag <= high
    low, high = first.onset_lag_interval
    assert low <= first.onset_lag <= high
    assert first.onset_lag < first.peak_lag


def test_population_timing_with_no_units_is_nan():
    result = population_timing(
        np.empty((len(LAG_TIME), 0)), np.empty(0), np.empty(0), LAG_TIME
    )

    assert np.isnan(result.peak_lag) and result.n_units == 0


def test_rise_onset_measures_from_the_window_start():
    # A curve sitting at 0.6 from the window start that rises to 1.2 at +50 ms:
    # from zero the run reaches the edge; from the start level it begins near
    # the half-way point of the rise.
    rise = 0.6 + _bump(0.050, 0.020, 0.6)
    from_zero = peak_and_onset(rise, LAG_TIME)
    from_start = peak_and_onset(rise, LAG_TIME, baseline="window_start")

    assert from_zero.onset_at_edge[0]
    assert from_start.onset_lag[0] == pytest.approx(0.050 - 1.1774 * 0.020, abs=0.004)
    assert from_start.peak_lag[0] == from_zero.peak_lag[0]
