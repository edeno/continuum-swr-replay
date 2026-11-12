"""The kernel-timing calibration recovers the order of planted onsets."""

import json

import pytest

from scripts.pfc.calibrate_pfc_kernel_timing import main


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
