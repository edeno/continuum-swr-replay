"""Peak and onset of peri-event curves, and the population timing."""

import numpy as np
import pytest

from continuum_swr_replay.analysis.pfc.timing import (
    peak_and_onset,
    population_timing,
    response_size,
    sign_aligned_mean,
    steepest_approach_lag,
    timing_rule_calibration,
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
    animals = np.repeat(np.arange(3), 20)

    first = population_timing(
        curves, peaks, groups, animals, LAG_TIME, n_boot=200, seed=3
    )
    second = population_timing(
        curves, peaks, groups, animals, LAG_TIME, n_boot=200, seed=3
    )

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
        np.empty((len(LAG_TIME), 0)), np.empty(0), np.empty(0), np.empty(0), LAG_TIME
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


def test_population_timing_weights_units_not_sessions():
    # Session a: 3 units peaking at +20 ms; session b: 1 unit peaking at +100 ms.
    # The unit-weighted mean curve peaks near +20 ms, not half way.
    curves = np.column_stack(
        [_bump(0.020, 0.010, 1.0)] * 3 + [_bump(0.100, 0.010, 1.0)]
    )
    peaks = peak_and_onset(curves, LAG_TIME).peak
    groups = np.array(["a", "a", "a", "b"])
    animals = np.array(["x", "x", "x", "y"])

    result = population_timing(
        curves, peaks, groups, animals, LAG_TIME, n_boot=50, seed=0
    )

    assert result.peak_lag == pytest.approx(0.020)
    assert result.n_groups == 2
    # Resamples drawing only animal y (session b) peak at +100 ms.
    assert result.peak_lag_interval[1] == pytest.approx(0.100)


def test_population_timing_resamples_the_units_within_a_session():
    # One animal with one session: resampling the sessions alone would give
    # every replicate the same curve. The units differ (+10 to +50 ms), so
    # resampling them inside the session spreads the peak lag.
    curves = np.column_stack(
        [_bump(center, 0.005, 1.0) for center in (0.010, 0.020, 0.030, 0.040, 0.050)]
    )
    peaks = peak_and_onset(curves, LAG_TIME).peak
    result = population_timing(
        curves, peaks, np.zeros(5), np.zeros(5), LAG_TIME, n_boot=200, seed=0
    )

    low, high = result.peak_lag_interval
    assert low < result.peak_lag < high


def test_reference_window_baseline_and_size():
    # A dip of depth 0.6 centred at +50 ms on an offset of -0.4: from a
    # reference window before the dip the onset is at the dip's half depth and
    # the size is the dip's depth, not the offset plus the dip.
    curve = -0.4 + _bump(0.050, 0.020, -0.6)
    from_window = peak_and_onset(curve, LAG_TIME, baseline=(-0.2, -0.15))
    size_zero = response_size(curve, LAG_TIME)
    size_window = response_size(curve, LAG_TIME, baseline=(-0.2, -0.15))

    assert from_window.onset_lag[0] == pytest.approx(0.050 - 1.1774 * 0.020, abs=0.004)
    assert size_zero[0] == pytest.approx(-1.0, abs=1e-6)
    assert size_window[0] == pytest.approx(-0.6, abs=1e-3)
    with pytest.raises(ValueError):
        peak_and_onset(curve, LAG_TIME, baseline=(1.0, 2.0))


def test_steepest_approach_ignores_a_constant_offset():
    # A Gaussian dip is steepest one SD before its trough.
    dip = _bump(0.080, 0.025, -0.5)

    first = steepest_approach_lag(dip, LAG_TIME)
    second = steepest_approach_lag(dip - 0.3, LAG_TIME)

    assert first[0] == pytest.approx(0.055, abs=0.002)
    assert first[0] == second[0]


# Five lags, 2 ms apart: -4, -2, 0, +2, +4 ms.
CALIBRATION_LAGS = np.array([-0.004, -0.002, 0.0, 0.002, 0.004])
# Peak +2 ms, onset -2 ms; a decrease at -2 ms off a positive first lag; a
# peak at -2 ms whose first-maximum tie picks the earlier lag.
CALIBRATION_CURVES = np.column_stack(
    [
        [0.0, 1.0, 1.0, 2.0, 0.0],
        [0.5, -1.0, -1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 0.0, 0.0],
    ]
)
# Half-maximum onset from zero at 0 ms, from the first lag at +2 ms.
CALIBRATION_TRUTH = np.array([0.4, 0.4, 0.6, 1.0, 0.0])


def test_timing_rule_calibration_reads_units_population_and_truth():
    calibration = timing_rule_calibration(
        CALIBRATION_CURVES, CALIBRATION_LAGS, CALIBRATION_TRUTH
    )

    assert calibration["fraction_onset_before_zero"] == pytest.approx(1.0)
    assert calibration["fraction_peak_before_zero"] == pytest.approx(2 / 3)
    assert calibration["fraction_peak_sign_negative"] == pytest.approx(1 / 3)
    rules = calibration["rules"]
    assert list(rules) == [
        "peak",
        "onset_zero",
        "onset_rise",
        "steepest",
        "size_zero",
        "size_rise",
    ]
    # Over units: peaks +2, -2, -2 ms; sizes 2, -1, 1 (from zero) and 2,
    # -1.5, 1 (from the first lag).
    assert rules["peak"]["unit_median"] == pytest.approx(-2.0)
    assert rules["peak"]["unit_p2_5"] == pytest.approx(-2.0)
    assert rules["peak"]["unit_p97_5"] == pytest.approx(1.8)
    assert rules["size_zero"]["unit_median"] == pytest.approx(1.0)
    assert rules["size_zero"]["unit_p2_5"] == pytest.approx(-0.9)
    assert rules["size_rise"]["unit_p2_5"] == pytest.approx(-1.375)
    assert rules["steepest"]["unit_median"] == pytest.approx(-3.0)
    # The sign-aligned mean is [-1/6, 1, 1, 2/3, 0]: peak -2 ms, size 1 from
    # zero and 7/6 from its first lag.
    expected = {
        "peak": (-2.0, 2.0),
        "onset_zero": (-2.0, 0.0),
        "onset_rise": (-2.0, 2.0),
        "steepest": (-3.0, 1.0),
        "size_zero": (1.0, 1.0),
        "size_rise": (7 / 6, 0.6),
    }
    for name, (population, truth) in expected.items():
        assert rules[name]["population"] == pytest.approx(population), name
        assert rules[name]["truth"] == pytest.approx(truth), name
        assert rules[name]["population_bias"] == pytest.approx(population - truth)
        assert rules[name]["unit_bias"] == pytest.approx(
            rules[name]["unit_median"] - truth
        )


def test_timing_rule_calibration_without_truth_and_with_a_reference():
    calibration = timing_rule_calibration(
        CALIBRATION_CURVES,
        CALIBRATION_LAGS,
        None,
        reference_lags=(-0.004, -0.002),
    )

    rules = calibration["rules"]
    # Nothing planted: a size's truth is zero, a lag has none.
    assert rules["onset_zero"]["truth"] is None
    assert rules["onset_zero"]["population_bias"] is None
    assert rules["size_zero"]["truth"] == 0.0
    assert rules["size_zero"]["population_bias"] == pytest.approx(1.0)
    # Reference level (-1/6 + 1) / 2 = 5/12 under the mean curve's peak of 1.
    assert rules["size_reference"]["population"] == pytest.approx(7 / 12)
    assert rules["onset_reference"]["population"] == pytest.approx(-2.0)


def test_timing_rule_calibration_reads_a_planted_decrease_sign_aligned():
    rules = timing_rule_calibration(
        -CALIBRATION_CURVES[:, [0]], CALIBRATION_LAGS, -CALIBRATION_TRUTH
    )["rules"]

    assert rules["size_zero"]["truth"] == pytest.approx(1.0)
    assert rules["size_zero"]["population"] == pytest.approx(2.0)
    assert rules["size_zero"]["unit_median"] == pytest.approx(-2.0)
    assert rules["size_zero"]["unit_bias"] == pytest.approx(-3.0)
