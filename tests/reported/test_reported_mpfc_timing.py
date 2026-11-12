"""The mPFC summary's timing values on a synthetic population."""

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.pfc.load_population import HELDOUT_P_COLUMNS
from continuum_swr_replay.reported.mpfc import TAGS, timing_values
from continuum_swr_replay.reported.values import value_factory

LAG_TIME = np.arange(-100, 101) / 500.0


def _bump(center_s: float, height: float) -> np.ndarray:
    return height * np.exp(-0.5 * ((LAG_TIME - center_s) / 0.02) ** 2)


@pytest.fixture
def population_and_curves():
    rng = np.random.default_rng(0)
    n = 40
    population = pd.DataFrame(
        {
            "nwb_file_name": np.repeat(
                ["chimi20200212_.nwb", "wilbur20210329_.nwb"], n // 2
            ),
            "epoch_name": np.tile(["02_r1", "04_r2"], n // 2),
            "animal": np.repeat(["chimi", "wilbur"], n // 2),
        }
    )
    # Every fourth unit is not called for the replay classes; for all SWRs it is
    # every fifth instead, so swapping the p columns changes the counts.
    for name in ("non_local_SWR", "non_local_no_SWR"):
        population[HELDOUT_P_COLUMNS[name]] = np.where(np.arange(n) % 4 == 3, 0.5, 0.01)
    population[HELDOUT_P_COLUMNS["SWR"]] = np.where(np.arange(n) % 5 == 4, 0.5, 0.01)
    signs = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    swr_centers = rng.normal(0.040, 0.005, n)
    curves = {
        "swr": np.column_stack(
            [_bump(c + 0.040, s) for c, s in zip(swr_centers, signs, strict=True)]
        ),
        "non_local_swr": np.column_stack(
            [_bump(c, s) for c, s in zip(swr_centers, signs, strict=True)]
        ),
        "non_local_no_swr": np.column_stack(
            [_bump(c + 0.020, s) for c, s in zip(swr_centers, signs, strict=True)]
        ),
        "matched_control": np.column_stack(
            [_bump(c, 0.1) for c in rng.uniform(-0.15, 0.15, n)]
        ),
    }

    def rule(population, bias):
        return {"population": population, "population_bias": bias}

    calibration = {
        "scenarios": [
            {
                "window": window,
                "scenario": "response",
                "shape": "boxcar",
                "gain": gain,
                "true_onset_ms": onset,
                "n_units": 200,
                "fraction_onset_before_zero": 0.2,
                "fraction_peak_sign_negative": 0.0 if gain > 0 else 1.0,
                "rules": {
                    "onset_zero": rule(onset + 4.0, 4.0),
                    "onset_rise": rule(onset + 2.0, 2.0),
                    "steepest": rule(onset, 0.0),
                    "peak": rule(onset + 50, 0.0),
                    "size_zero": rule(0.8, 0.1),
                    "size_rise": rule(0.75, 0.05),
                },
            }
            for window in ("canonical", "wide")
            for gain in (0.7, -0.7)
            for onset in (0.0, 50.0)
        ]
        + [
            {
                "window": "canonical",
                "scenario": "stop_offset",
                "shape": "boxcar",
                "gain": 0.0,
                "true_onset_ms": 0.0,
                "n_units": 200,
                "fraction_onset_before_zero": 0.5,
                "fraction_peak_sign_negative": 0.95,
                "rules": {
                    "onset_zero": rule(-200.0, None),
                    "onset_rise": rule(130.0, None),
                    "steepest": rule(120.0, None),
                    "peak": rule(150.0, None),
                    "size_zero": rule(0.19, 0.19),
                    "size_rise": rule(0.01, 0.01),
                },
            }
        ]
    }
    return population, curves, calibration


def test_timing_values_keys_kinds_and_directions(population_and_curves):
    population, curves, calibration = population_and_curves
    make = value_factory(timing_values, "synthetic", TAGS["modulation"])

    values = {
        v.key: v for v in timing_values(population, curves, LAG_TIME, calibration, make)
    }

    for word in ("swr", "swr_replay", "non_swr_replay"):
        for direction in ("enhanced", "suppressed"):
            stem = f"{word}_{direction}"
            # Even units are enhanced, odd suppressed; see the fixture's p-values.
            expected = 16 if word == "swr" else 20 if direction == "enhanced" else 10
            assert values[f"timing_units_{stem}"].value == expected
            center = {"swr": 80, "swr_replay": 40, "non_swr_replay": 60}[word]
            peak_ms = values[f"timing_peak_lag_median_ms_{stem}"].value
            assert peak_ms == pytest.approx(center, abs=6)
            assert (
                values[f"timing_onset_median_ms_{stem}"].value
                < values[f"timing_peak_lag_median_ms_{stem}"].value
            )
            low = values[f"timing_population_onset_ms_ci_low_{stem}"].value
            high = values[f"timing_population_onset_ms_ci_high_{stem}"].value
            assert low <= values[f"timing_population_onset_ms_{stem}"].value <= high
            assert values[f"timing_population_onset_animals_{stem}"].value == 2
            # The bumps start near zero, so the rise onset is the half-max onset.
            assert values[f"timing_population_rise_onset_ms_{stem}"].value == (
                pytest.approx(values[f"timing_population_onset_ms_{stem}"].value, abs=4)
            )
            assert values[f"timing_peak_at_edge_fraction_{stem}"].value == 0.0
            for animal in ("chimi", "wilbur"):
                assert f"timing_population_rise_onset_ms_{stem}_{animal}" in values
            assert (
                values[f"timing_population_rise_onset_animals_negative_{stem}"].value
                == 0
            )
    assert values["timing_peak_lag_difference_median_ms"].value == pytest.approx(
        20, abs=2
    )
    assert values["timing_non_swr_later_fraction"].value == 1.0
    assert values["timing_calibration_bias_ms_onset_zero_boxcar_enhanced"].value == 4.0
    assert (
        values["timing_calibration_bias_ms_onset_rise_boxcar_suppressed"].value == 2.0
    )
    assert "timing_calibration_bias_ms_peak_boxcar_enhanced" not in values
    assert values["timing_calibration_stop_offset_suppressed_fraction"].value == 0.95
    assert values["timing_calibration_stop_offset_rise_size"].value == 0.01
    for stem in ("swr_replay_suppressed", "non_swr_replay_enhanced"):
        assert f"timing_population_steepest_ms_{stem}" in values
        assert values[f"timing_population_rise_size_{stem}"].value > 0
        assert f"timing_population_size_vs_control_{stem}" in values
    for value in values.values():
        assert not any(ch.isdigit() for ch in value.key)
        if value.kind == "fraction":
            assert 0.0 <= value.value <= 1.0


def test_timing_values_without_calibration_omit_its_keys(population_and_curves):
    population, curves, _ = population_and_curves
    make = value_factory(timing_values, "synthetic", TAGS["modulation"])

    keys = {v.key for v in timing_values(population, curves, LAG_TIME, None, make)}

    assert not any(key.startswith("timing_calibration") for key in keys)
    assert "timing_matched_control_peak_before_onset_fraction" in keys


def test_rise_onset_is_measured_from_each_unit_window_start(population_and_curves):
    """A constant offset of each unit's curve (its level at -200 ms, with the
    peak's sign) moves the half-maximum onset measured from zero earlier, but
    not the rise onset measured from the curve's value at -200 ms."""
    population, curves, _ = population_and_curves
    make = value_factory(timing_values, "synthetic", TAGS["modulation"])
    rng = np.random.default_rng(1)
    shifted = dict(curves)
    for key in ("swr", "non_local_swr", "non_local_no_swr"):
        sign = np.sign(
            curves[key][np.argmax(np.abs(curves[key]), axis=0), :].diagonal()
        )
        offset = sign * rng.uniform(0.7, 0.9, curves[key].shape[1])
        shifted[key] = curves[key] + offset[None, :]

    plain = {
        v.key: v.value for v in timing_values(population, curves, LAG_TIME, None, make)
    }
    moved = {
        v.key: v.value for v in timing_values(population, shifted, LAG_TIME, None, make)
    }
    for word in ("swr", "swr_replay", "non_swr_replay"):
        for direction in ("enhanced", "suppressed"):
            stem = f"{word}_{direction}"
            for rise, onset in (
                (
                    f"timing_population_rise_onset_ms_{stem}",
                    f"timing_population_onset_ms_{stem}",
                ),
                (
                    f"timing_rise_onset_median_ms_{stem}",
                    f"timing_onset_median_ms_{stem}",
                ),
            ):
                # measured from -200 ms: the offset is removed
                assert moved[rise] == pytest.approx(plain[onset], abs=4)
                # measured from zero: the offset widens the half-maximum run
                assert moved[onset] < plain[onset] - 10


def test_every_onset_description_states_the_half_maximum_rule(population_and_curves):
    """The onset of every class and direction is the half-maximum onset, and its
    description says so: not the name of another statistic."""
    population, curves, _ = population_and_curves
    make = value_factory(timing_values, "synthetic", TAGS["modulation"])
    values = timing_values(population, curves, LAG_TIME, None, make)
    onsets = [
        v
        for v in values
        if v.key.startswith(("timing_onset_median_ms_", "timing_population_onset_ms_"))
        and "_ci_" not in v.key
    ]
    assert len({v.key for v in onsets}) > 2
    for value in onsets:
        assert "the half-maximum onset: the earliest lag" in value.description, (
            value.key,
            value.description,
        )


def test_descriptions_state_the_clip_and_what_is_resampled(population_and_curves):
    population, curves, _ = population_and_curves
    make = value_factory(timing_values, "synthetic", TAGS["modulation"])
    values = {v.key: v for v in timing_values(population, curves, LAG_TIME, None, make)}
    fractions = [v for key, v in values.items() if "pre_event_level_fraction" in key]
    assert len(fractions) == 12
    for value in fractions:
        assert "clipped to the range 0 to 1" in value.description, value.key
    # the intervals resample recording days (NWB files), not sessions
    intervals = [v for key, v in values.items() if "_ci_low_" in key]
    assert len(intervals) == 36
    for value in intervals:
        assert "recording-day resamples" in value.description, value.key
