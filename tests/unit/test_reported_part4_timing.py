"""Part 4's timing values on a synthetic population."""

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.pfc.load_population import HELDOUT_P_COLUMNS
from continuum_swr_replay.reported.part4 import _value_factory, timing_values

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
    for name in ("SWR", "non_local_SWR", "non_local_no_SWR"):
        population[HELDOUT_P_COLUMNS[name]] = np.where(np.arange(n) % 4 == 3, 0.5, 0.01)
    signs = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    swr_centers = rng.normal(0.040, 0.005, n)
    curves = {
        "swr": np.column_stack(
            [_bump(c, s) for c, s in zip(swr_centers, signs, strict=True)]
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
    calibration = {
        "scenarios": [
            {
                "shape": "boxcar",
                "gain": gain,
                "true_onset_ms": onset,
                "true_peak_ms": onset + 50,
                "onset_bias_ms_median": 4.0,
                "fraction_onset_before_zero": 0.2,
                "apparent_peak_ms": {"median": 48.0},
                "n_units": 200,
            }
            for gain in (0.7, -0.7)
            for onset in (0.0, 50.0)
        ]
    }
    return population, curves, calibration


def test_timing_values_keys_kinds_and_directions(population_and_curves):
    population, curves, calibration = population_and_curves
    make = _value_factory("synthetic", ("20",), ("paper",))

    values = {
        v.key: v for v in timing_values(population, curves, LAG_TIME, calibration, make)
    }

    for word in ("swr", "swr_replay", "non_swr_replay"):
        for direction in ("enhanced", "suppressed"):
            stem = f"{word}_{direction}"
            # Every fourth unit (all odd, so all suppressed) is not called.
            assert values[f"timing_units_{stem}"].value == (
                20 if direction == "enhanced" else 10
            )
            center = 40 + (20 if word == "non_swr_replay" else 0)
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
    assert values["timing_peak_lag_difference_median_ms"].value == pytest.approx(
        20, abs=2
    )
    assert values["timing_non_swr_later_fraction"].value == 1.0
    assert values["timing_calibration_onset_bias_ms_boxcar_enhanced"].value == 4.0
    assert "timing_calibration_onset_bias_ms_boxcar_suppressed" in values
    for value in values.values():
        assert not any(ch.isdigit() for ch in value.key)
        if value.kind == "fraction":
            assert 0.0 <= value.value <= 1.0


def test_timing_values_without_calibration_omit_its_keys(population_and_curves):
    population, curves, _ = population_and_curves
    make = _value_factory("synthetic", ("20",), ("paper",))

    keys = {v.key for v in timing_values(population, curves, LAG_TIME, None, make)}

    assert not any(key.startswith("timing_calibration") for key in keys)
    assert "timing_matched_control_peak_before_onset_fraction" in keys
