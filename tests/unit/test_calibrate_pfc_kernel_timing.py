"""The kernel-timing calibration recovers the order of planted onsets."""

import json

import numpy as np
import pytest

from scripts.pfc.calibrate_pfc_kernel_timing import (
    event_onsets,
    main,
    response_shape,
    stretch_offset_and_onsets,
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


def test_stretch_offset_crowds_events_into_low_stretches():
    log_offset, onsets = stretch_offset_and_onsets(
        100_000, 50, np.random.default_rng(1)
    )

    assert len(onsets) == 50 and np.diff(onsets).min() >= 300
    assert np.mean(log_offset[onsets] < 0) == pytest.approx(0.8)
    assert set(np.unique(log_offset)) == {0.0, -0.3}


@pytest.mark.slow
def test_calibration_quick_recovers_a_late_onset(tmp_path, monkeypatch):
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    output = tmp_path / "calibration.json"

    main(["--quick", "--n-bins", "60000", "--n-events", "60", "--output", str(output)])

    record = json.loads(output.read_text())
    by_onset = {s["true_onset_ms"]: s for s in record["scenarios"]}
    assert set(by_onset) == {0.0, 50.0}
    early = by_onset[0.0]["rules"]["onset_zero"]["population"]
    late = by_onset[50.0]["rules"]["onset_zero"]["population"]
    assert late - early == pytest.approx(50.0, abs=15.0)
    for scenario in record["scenarios"]:
        assert scenario["fraction_peak_sign_negative"] <= 0.2
        assert {"onset_zero", "onset_rise", "steepest", "peak", "size_zero"} <= set(
            scenario["rules"]
        )
        assert scenario["rules"]["onset_rise"]["population_bias"] is not None
