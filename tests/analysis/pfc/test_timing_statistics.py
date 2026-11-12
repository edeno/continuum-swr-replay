"""The timing statistics of the units called for an event class, on step curves
whose peak, onset and steepest step can be read off by hand."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.pfc.timing import (
    DIRECTIONS,
    called_response_timing,
    called_response_timing_by_animal,
    paired_peak_lag_statistics,
    peak_and_onset,
    peak_before_onset_fraction,
)
from continuum_swr_replay.analysis.shared_statistics import (
    nested_bootstrap_draw,
    nested_codes,
)

# The report window: +/-200 ms at 500 Hz.
LAG_MS = 2 * np.arange(-100, 101)
LAG_TIME = LAG_MS / 1000.0


def _steps(base: float, first: tuple[int, float], second: tuple[int, float]):
    """``base``, then ``first[1]`` from ``first[0]`` ms, then ``second[1]`` from
    ``second[0]`` ms."""
    curve = np.full(LAG_MS.shape, base)
    for start_ms, level in (first, second):
        curve[LAG_MS >= start_ms] = level
    return curve


def _spike(at_ms: int, height: float) -> np.ndarray:
    return height * (LAG_MS == at_ms)


# Each unit's curve, with what the rules read off it (ms):
#
#                                              peak  onset  rise onset  steepest
# 0  0, 0.7 from 20, 1.0 from 60                 60     20          20        19
# 1  0, 0.6 from -100, 1.1 from -20             -20   -100        -100      -101
# 2  1.0, 1.2 from 20, 1.6 from 100             100   -200         100        99
# 3  0, 0.6 from 160, 1.4 from 200              200    200         200       199
# 4  minus unit 2's curve (a decrease)          100   -200         100        99
# 5  not called: 5.0 at -150 only
# 6  called, but zero everywhere: it has no peak, so it has no direction
#
# The onset is the earliest lag of the run, ending at the peak, on which the
# curve stays at least half the peak (unit 2 never falls below 0.8, so its run
# reaches the window edge). The rise onset asks for half way from the value at
# -200 ms to the peak (unit 2: 1.3, first reached at 100 ms). The steepest
# approach is the midpoint of the largest step before the peak.
UNIT_2 = _steps(1.0, (20, 1.2), (100, 1.6))
CURVES = np.column_stack(
    [
        _steps(0.0, (20, 0.7), (60, 1.0)),
        _steps(0.0, (-100, 0.6), (-20, 1.1)),
        UNIT_2,
        _steps(0.0, (160, 0.6), (200, 1.4)),
        -UNIT_2,
        _spike(-150, 5.0),
        np.zeros(LAG_MS.shape),
    ]
)
CALLED = np.array([True, True, True, True, True, False, True])
ANIMALS = np.array(["a", "a", "b", "b", "b", "a", "a"])
#: Each recording day is of one animal.
DAYS = np.array(["a1", "a2", "b1", "b2", "b2", "a1", "a1"])
#: Matched-control curves: constant, but unit 3's steps from 0.3 to 0.9 at event
#: onset, and unit 2 has none.
CONTROL = np.outer(
    np.ones(LAG_MS.shape), np.array([0.1, 0.1, np.nan, 0.3, 0.2, 0.0, 0.0])
)
CONTROL[LAG_MS >= 0, 3] = 0.9

# The mean curve of the increases (units 0-3) is, by segment:
#   before -100 ms: 0.25   -100 to -20: 0.4   -20 to 20: 0.525   20 to 60: 0.75
#   60 to 100: 0.825   100 to 160: 0.925   160 to 200: 1.075   at 200: 1.275
# Animal a's (units 0, 1, one a day): 0, 0.3, 0.55 from -20, 0.9 from 20, 1.05
# from 60. Animal b's (units 2, 3, one a day): 0.5, 0.6 from 20, 0.8 from 100,
# 1.1 from 160, 1.5 at 200.


@pytest.fixture(scope="module")
def statistics() -> pd.DataFrame:
    return called_response_timing(
        CURVES, CONTROL, LAG_TIME, CALLED, DAYS, ANIMALS, n_boot=200, seed=0
    )[0]


def test_directions_are_the_signs_of_the_peak():
    assert DIRECTIONS == {"enhanced": 1.0, "suppressed": -1.0}


def test_called_response_timing_summarises_units_by_direction(statistics):
    assert list(statistics.index) == ["enhanced", "suppressed"]
    enhanced = statistics.loc["enhanced"]
    assert enhanced["n_units"] == 4
    # peak lags -20, 60, 100, 200
    assert enhanced["peak_lag_median_ms"] == pytest.approx(80)
    assert enhanced["peak_lag_quartile_low_ms"] == pytest.approx(40)
    assert enhanced["peak_lag_quartile_high_ms"] == pytest.approx(125)
    assert enhanced["peak_before_onset_fraction"] == 0.25
    # onsets -200, -100, 20, 200
    assert enhanced["onset_median_ms"] == pytest.approx(-40)
    assert enhanced["onset_before_onset_fraction"] == 0.5
    # unit 2's run reaches -200 ms; unit 3's peak is at +200 ms
    assert enhanced["onset_at_edge_fraction"] == 0.25
    assert enhanced["peak_at_edge_fraction"] == 0.25
    # rise onsets -100, 20, 100, 200; steepest steps -101, 19, 99, 199
    assert enhanced["rise_onset_median_ms"] == pytest.approx(60)
    assert enhanced["steepest_median_ms"] == pytest.approx(59)

    suppressed = statistics.loc["suppressed"]
    assert suppressed["n_units"] == 1
    assert suppressed["peak_lag_median_ms"] == pytest.approx(100)
    assert suppressed["peak_before_onset_fraction"] == 0.0
    assert suppressed["onset_median_ms"] == pytest.approx(-200)
    assert suppressed["onset_before_onset_fraction"] == 1.0
    assert suppressed["onset_at_edge_fraction"] == 1.0
    assert suppressed["peak_at_edge_fraction"] == 0.0
    assert suppressed["rise_onset_median_ms"] == pytest.approx(100)
    assert suppressed["steepest_median_ms"] == pytest.approx(99)


def test_called_response_timing_reads_the_sign_aligned_mean_curve(statistics):
    enhanced = statistics.loc["enhanced"]
    assert enhanced["population_peak_ms"] == pytest.approx(200)
    # half the peak is 0.6375: the mean curve is above it from 20 ms (0.75)
    assert enhanced["population_onset_ms"] == pytest.approx(20)
    # half way from 0.25 to 1.275 is 0.7625: above it from 60 ms (0.825)
    assert enhanced["population_rise_onset_ms"] == pytest.approx(60)
    # its largest step, 0.225, is the one into 20 ms
    assert enhanced["population_steepest_ms"] == pytest.approx(19)
    assert enhanced["population_size"] == pytest.approx(1.275)
    assert enhanced["population_rise_size"] == pytest.approx(1.275 - 0.25)
    assert enhanced["population_pre_event_level_fraction"] == pytest.approx(
        0.25 / 1.275
    )
    # Unit 2 has no control curve, so the control comparison is over units 0,
    # 1 and 3 on both sides. Their mean curve peaks at 200 ms, at
    # (1.0 + 1.1 + 1.4) / 3 (all four units' is 1.275 there); their control
    # curves average (0.1 + 0.1 + 0.9) / 3 at 200 ms and (0.1 + 0.1 + 0.3) / 3
    # at -200 ms.
    assert enhanced["population_size_vs_control"] == pytest.approx(3.5 / 3 - 1.1 / 3)
    assert enhanced["population_control_pre_event_level_fraction"] == pytest.approx(
        0.5 / 3.5
    )

    # the one decrease, signed by its peak, is unit 2's curve
    suppressed = statistics.loc["suppressed"]
    assert suppressed["population_peak_ms"] == pytest.approx(100)
    assert suppressed["population_onset_ms"] == pytest.approx(-200)
    assert suppressed["population_rise_onset_ms"] == pytest.approx(100)
    assert suppressed["population_steepest_ms"] == pytest.approx(99)
    assert suppressed["population_size"] == pytest.approx(1.6)
    assert suppressed["population_rise_size"] == pytest.approx(0.6)
    assert suppressed["population_pre_event_level_fraction"] == pytest.approx(1 / 1.6)
    # its control curve, 0.2, signed the same way is -0.2
    assert suppressed["population_size_vs_control"] == pytest.approx(1.8)
    assert suppressed["population_control_pre_event_level_fraction"] == 0.0


def test_called_response_timing_intervals_come_from_the_nested_bootstrap(
    statistics,
):
    # The replicates are the shared nested draws of the increases (units 0-3):
    # the animals, each drawn animal's days, each drawn day's units, seed 0.
    # Each replicate's curve is the mean of the drawn units' curves.
    animal, day = nested_codes(ANIMALS[:4], DAYS[:4])
    rng = np.random.default_rng(0)
    counts = np.stack(
        [nested_bootstrap_draw(animal, day, rng).rows for _ in range(200)]
    )
    replicates = (counts @ CURVES[:, :4].T) / counts.sum(axis=1)[:, None]
    read = peak_and_onset(replicates.T, LAG_TIME)
    expected = {
        "population_peak_ms": read.peak_lag * 1e3,
        "population_onset_ms": read.onset_lag * 1e3,
        "population_size": read.peak,
    }
    enhanced = statistics.loc["enhanced"]
    for name, values in expected.items():
        # two animals (a, b): the value +/- t(0.975, 1) sqrt(2) SD(replicates)
        half = 12.7062047362 * np.sqrt(2) * np.std(values, ddof=1)
        assert enhanced[f"{name}_ci_low"] == pytest.approx(enhanced[name] - half), name
        assert enhanced[f"{name}_ci_high"] == pytest.approx(enhanced[name] + half), name
    # the decreases are one unit of one animal: no t quantile, no interval
    suppressed = statistics.loc["suppressed"]
    for name in expected:
        assert np.isnan(suppressed[f"{name}_ci_low"]), name
        assert np.isnan(suppressed[f"{name}_ci_high"]), name


def test_called_response_timing_counts_animals_whose_own_curve_starts_early(statistics):
    enhanced = statistics.loc["enhanced"]
    # a: onset -20, rise onset -20, steepest 19; b: 100, 160, 199
    assert enhanced["n_animals"] == 2
    assert enhanced["n_animals_onset_before_event"] == 1
    assert enhanced["n_animals_rise_onset_before_event"] == 1
    assert enhanced["n_animals_steepest_before_event"] == 0
    # b's one decrease: onset -200, rise onset 100, steepest 99
    suppressed = statistics.loc["suppressed"]
    assert suppressed["n_animals"] == 1
    assert suppressed["n_animals_onset_before_event"] == 1
    assert suppressed["n_animals_rise_onset_before_event"] == 0
    assert suppressed["n_animals_steepest_before_event"] == 0


def test_called_response_timing_by_animal():
    by_animal = called_response_timing_by_animal(
        CURVES, LAG_TIME, CALLED, DAYS, ANIMALS
    )
    assert list(by_animal.index) == [
        ("enhanced", "a"),
        ("enhanced", "b"),
        ("suppressed", "b"),
    ]
    assert list(by_animal.columns) == ["onset_ms", "rise_onset_ms", "steepest_ms"]
    np.testing.assert_allclose(by_animal["onset_ms"], [-20, 100, -200])
    np.testing.assert_allclose(by_animal["rise_onset_ms"], [-20, 160, 100])
    np.testing.assert_allclose(by_animal["steepest_ms"], [19, 199, 99])


def test_an_animals_own_curve_weighs_its_days_not_its_units():
    # Unit 0's curve twice on day x and unit 3's once on day y, all of one
    # animal. The mean of the two days' curves is 0.35 from 20 ms, 0.5 from
    # 60, 0.8 from 160 and 1.2 at 200: above half its peak (0.6) from 160 ms.
    # The mean of the three units (0.47, 0.67, 0.87, 1.13) would be from 60 ms.
    curves = CURVES[:, [0, 0, 3]]
    by_animal = called_response_timing_by_animal(
        curves,
        LAG_TIME,
        np.ones(3, dtype=bool),
        np.array(["x", "x", "y"]),
        np.array(["a", "a", "a"]),
    )
    assert by_animal.loc[("enhanced", "a"), "onset_ms"] == pytest.approx(160)


def test_a_direction_without_units_has_only_its_count():
    called = CALLED & (np.arange(len(CALLED)) != 4)
    statistics, by_animal = called_response_timing(
        CURVES, CONTROL, LAG_TIME, called, DAYS, ANIMALS, n_boot=10
    )
    suppressed = statistics.loc["suppressed"]
    assert suppressed["n_units"] == 0
    assert suppressed.drop("n_units").isna().all()
    assert statistics.loc["enhanced", "n_units"] == 4
    # the per-animal timing returned beside the statistics is that of the same units
    pd.testing.assert_frame_equal(
        by_animal,
        called_response_timing_by_animal(CURVES, LAG_TIME, called, DAYS, ANIMALS),
    )
    assert list(by_animal.index.get_level_values("direction")) == ["enhanced"] * 2
    nothing = called_response_timing_by_animal(
        CURVES, LAG_TIME, np.zeros(len(CALLED), dtype=bool), DAYS, ANIMALS
    )
    assert nothing.empty and list(nothing.index.names) == ["direction", "animal"]


# Four spikes on a +/-100 ms window, one per unit: at its first lag, at its last
# lag, one sample before the last, and (the tallest) at event onset. A spike's
# peak, onset and rise onset are all at the spike.
NARROW_LAG_MS = 2 * np.arange(-50, 51)
SPIKES = np.column_stack(
    [
        height * (NARROW_LAG_MS == at_ms)
        for at_ms, height in ((-100, 1.0), (100, 1.0), (98, 1.0), (0, 1.5))
    ]
)


def _spike_statistics(animals: list[str]) -> pd.DataFrame:
    return called_response_timing(
        SPIKES,
        np.zeros(SPIKES.shape),
        NARROW_LAG_MS / 1000.0,
        np.ones(4, dtype=bool),
        np.array([f"{animal}1" for animal in animals]),
        np.array(animals),
        n_boot=10,
    )[0]


def test_a_peak_at_either_end_of_the_lags_given_is_at_the_edge():
    # the edge is where the given lags end (here +/-100 ms), on both sides
    enhanced = _spike_statistics(["a"] * 4).loc["enhanced"]
    assert enhanced["peak_at_edge_fraction"] == 0.5


def test_a_lag_at_event_onset_is_not_before_it():
    enhanced = _spike_statistics(["b", "b", "b", "a"]).loc["enhanced"]
    # of the four peaks, and the four onsets, only the one at -100 ms is before
    # event onset: the one at 0 ms is not
    assert enhanced["peak_before_onset_fraction"] == 0.25
    assert enhanced["onset_before_onset_fraction"] == 0.25
    # Animal a has the spike at event onset alone: its onset and its rise onset
    # are 0 ms, which is not before the event; the step into the spike, its
    # steepest approach, is at -1 ms. Animal b's mean curve has three equal
    # spikes and peaks at the first of them, -100 ms, the first lag: its two
    # onsets are there, and a peak at the first lag has no approach.
    assert enhanced["n_animals"] == 2
    assert enhanced["n_animals_onset_before_event"] == 1
    assert enhanced["n_animals_rise_onset_before_event"] == 1
    assert enhanced["n_animals_steepest_before_event"] == 1


def test_the_mean_curve_has_one_peak_the_lag_of_its_largest_absolute_value():
    # Two units, each with its own peak (+1.0) at another lag, and both at
    # -0.8 at -50 ms and at -0.4 at -200 ms. Their mean is 0.5 at 50 and at
    # 100 ms and -0.8 at -50 ms: its largest absolute value is the dip. Every
    # statistic of the mean curve's peak is read there, not at its largest
    # value (0.5, at 50 ms).
    dip = _spike(-50, -0.8) + _spike(-200, -0.4)
    curves = np.column_stack([_spike(50, 1.0) + dip, _spike(100, 1.0) + dip])
    statistics = called_response_timing(
        curves,
        np.full(curves.shape, -0.2),
        LAG_TIME,
        np.ones(2, dtype=bool),
        np.array(["d1", "d1"]),
        np.array(["a", "a"]),
        n_boot=10,
    )[0].loc["enhanced"]
    assert statistics["population_peak_ms"] == pytest.approx(-50)
    assert statistics["population_size"] == pytest.approx(-0.8)
    assert statistics["population_pre_event_level_fraction"] == pytest.approx(0.5)
    assert statistics["population_size_vs_control"] == pytest.approx(-0.8 + 0.2)
    assert statistics["population_control_pre_event_level_fraction"] == (
        pytest.approx(0.25)
    )


def test_a_pre_event_level_on_the_other_side_of_zero_is_reported_as_zero():
    # the curve is -0.3 at -200 ms and peaks at +1.0: the share, -0.3, is
    # clipped to zero; so is the control's, -0.2
    curves = (_spike(50, 1.0) + _spike(-200, -0.3))[:, None]
    statistics = called_response_timing(
        curves,
        np.full(curves.shape, -0.2),
        LAG_TIME,
        np.ones(1, dtype=bool),
        np.array(["d1"]),
        np.array(["a"]),
        n_boot=10,
    )[0].loc["enhanced"]
    assert statistics["population_pre_event_level_fraction"] == 0.0
    assert statistics["population_control_pre_event_level_fraction"] == 0.0


def test_units_without_a_usable_control_curve_have_no_control_comparison():
    statistics = called_response_timing(
        CURVES,
        np.full(CONTROL.shape, np.nan),
        LAG_TIME,
        CALLED,
        DAYS,
        ANIMALS,
        n_boot=10,
    )[0]
    enhanced = statistics.loc["enhanced"]
    assert np.isnan(enhanced["population_size_vs_control"])
    assert np.isnan(enhanced["population_control_pre_event_level_fraction"])
    assert enhanced["population_size"] == pytest.approx(1.275)


def test_paired_peak_lag_statistics_compare_units_called_for_both_with_one_sign():
    first = np.column_stack(
        [
            _spike(20, 1.0),
            _spike(40, -1.0),
            _spike(0, 1.0),
            _spike(10, 1.0),
            _spike(-50, 1.0),
            _spike(100, 2.0),
        ]
    )
    second = np.column_stack(
        [
            _spike(60, 1.0),  # 40 ms later
            _spike(30, -1.0),  # 10 ms earlier
            _spike(100, -1.0),  # the opposite sign: not paired
            _spike(80, 1.0),  # not called for the second class: not paired
            _spike(-30, 1.0),  # 20 ms later
            _spike(100, 1.0),  # the same lag: not later
        ]
    )
    everyone = np.ones(6, dtype=bool)
    second_called = np.array([True, True, True, False, True, True])
    result = paired_peak_lag_statistics(
        first, second, LAG_TIME, everyone, second_called
    )
    assert result["n_paired"] == 4
    # differences 40, -10, 20 and 0 (their mean is 12.5)
    assert result["difference_median_ms"] == pytest.approx(10)
    assert result["second_later_fraction"] == 0.5
    # the difference is the second class's lag minus the first's
    swapped = paired_peak_lag_statistics(
        second, first, LAG_TIME, second_called, everyone
    )
    assert swapped["difference_median_ms"] == pytest.approx(-10)
    assert swapped["second_later_fraction"] == 0.25

    nobody = paired_peak_lag_statistics(
        first, second, LAG_TIME, everyone, np.zeros(6, dtype=bool)
    )
    assert nobody["n_paired"] == 0
    assert np.isnan(nobody["difference_median_ms"])
    assert np.isnan(nobody["second_later_fraction"])


def test_peak_before_onset_fraction_is_over_curves_with_a_peak():
    curves = np.column_stack(
        [
            _spike(-50, 1.0),
            _spike(50, -1.0),
            _spike(10, 1.0),
            _spike(0, 1.0),  # at onset: not before it
            np.full(LAG_MS.shape, np.nan),  # no curve
            np.zeros(LAG_MS.shape),  # no peak
        ]
    )
    assert peak_before_onset_fraction(curves, LAG_TIME) == 0.25
