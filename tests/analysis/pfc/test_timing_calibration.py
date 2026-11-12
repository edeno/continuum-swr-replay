"""The simulated responses the kernel-timing calibration plants, and its scenarios."""

import numpy as np
import pytest

from continuum_swr_replay.analysis.pfc.timing_calibration import (
    event_onsets,
    response_shape,
    scenario_list,
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


def test_the_scenarios_cross_windows_shapes_gains_and_onsets():
    scenarios = scenario_list(quick=False)
    response = [s for s in scenarios if s["scenario"] == "response"]
    offsets = [s for s in scenarios if s["scenario"] != "response"]

    # 2 windows x 2 shapes x 3 gains x 3 onsets, then 2 offset scenarios per window
    assert len(response) == 36
    assert len({tuple(sorted(s.items())) for s in response}) == 36
    assert sorted((s["window"], s["scenario"]) for s in offsets) == [
        ("canonical", "stop_offset"),
        ("canonical", "stop_offset_plus_response"),
        ("wide", "stop_offset"),
        ("wide", "stop_offset_plus_response"),
    ]
    assert all(s["gain"] == -0.7 and s["onset_ms"] == 0.0 for s in offsets)
    assert scenario_list(quick=True) == [
        {
            "window": "canonical",
            "scenario": "response",
            "shape": "boxcar",
            "gain": 0.7,
            "onset_ms": onset,
        }
        for onset in (0.0, 50.0)
    ]
