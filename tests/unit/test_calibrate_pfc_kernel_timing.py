"""The kernel-timing calibration recovers the order of planted onsets."""

import json

import numpy as np
import pytest

from scripts.pfc.calibrate_pfc_kernel_timing import (
    event_onsets,
    main,
    response_shape,
)


def test_response_shapes_start_at_the_onset_and_peak_at_one():
    lag_time = np.arange(-100, 101) / 500.0
    for shape in ("boxcar", "alpha"):
        h = response_shape(shape, lag_time, 0.020)
        assert np.all(h[lag_time < 0.020] == 0)
        assert h.max() == pytest.approx(1.0)


def test_event_onsets_respect_the_spacing():
    onsets = event_onsets(50_000, 60, np.random.default_rng(0))

    assert len(onsets) == 60
    assert np.diff(onsets).min() >= 300


@pytest.mark.slow
def test_calibration_quick_recovers_a_late_onset(tmp_path, monkeypatch):
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    output = tmp_path / "calibration.json"

    main(["--quick", "--n-bins", "60000", "--n-events", "60", "--output", str(output)])

    record = json.loads(output.read_text())
    by_onset = {s["true_onset_ms"]: s for s in record["scenarios"]}
    assert set(by_onset) == {0.0, 50.0}
    early = by_onset[0.0]["apparent_onset_ms"]["median"]
    late = by_onset[50.0]["apparent_onset_ms"]["median"]
    assert late - early == pytest.approx(50.0, abs=15.0)
    for scenario in record["scenarios"]:
        assert scenario["fraction_peak_sign_correct"] >= 0.8
        assert {
            "unit_onset_bias_ms_median",
            "population_rise_onset_bias_ms",
            "apparent_peak_ms",
            "n_units",
        } <= set(scenario)
