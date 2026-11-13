"""``measure_decoder_quality.measure_epoch``: the rows it writes for one decode. The
measures are tested in ``tests/analysis/decode_quality/test_quality_metrics_over_sessions.py``."""

from __future__ import annotations

import hashlib
import pickle

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.decode_quality.quality_metrics import (
    compute_calibration_curve,
    session_decoder_quality,
)
from scripts.decoding import measure_decoder_quality as script

IDENTIFIER = "j1620210706_02_r1_HPC"


def _frame() -> pd.DataFrame:
    local = np.array([0.9, 0.8, 0.2, 0.95, 0.6, 0.1])
    return pd.DataFrame(
        {
            "animal": "j16",
            "day": 20210706,
            "epoch": "02_r1",
            "head_speed": [10.0, 10.0, 1.0, 1.0, 1.0, 12.0],
            "Local_prob": local,
            "No-Spike_prob": 0.0,
            "Non-Local_Continuous_prob": 1.0 - local,
            "Non-Local_Fragmented_prob": 0.0,
            "ahead_behind_distance": [1.0, -3.0, 40.0, 2.0, -5.0, 60.0],
            "in_excluded_time": False,
        },
        index=np.arange(6) / 500.0,
    )


def test_an_epoch_s_rows_carry_its_id_and_its_decode_s_hash(tmp_path, monkeypatch):
    path = tmp_path / f"{IDENTIFIER}_80_data_over_time.pkl"
    raw = pickle.dumps(_frame())
    path.write_bytes(raw)
    # the SWR source is tested elsewhere; the frame already has its column
    monkeypatch.setattr(script.analysed_time, "with_swr_source", lambda dot: dot)

    sessions, calibration = script.measure_epoch(path)

    sha256 = hashlib.sha256(raw).hexdigest()
    assert sessions["epoch_id"].tolist() == [IDENTIFIER]
    assert sessions["decode_sha256"].tolist() == [sha256]
    pd.testing.assert_frame_equal(
        sessions.drop(columns=["epoch_id", "decode_sha256"]),
        session_decoder_quality(_frame()),
    )
    assert set(calibration["epoch_id"]) == {IDENTIFIER}
    pd.testing.assert_frame_equal(
        calibration.drop(columns=["epoch_id", "decode_sha256"]),
        compute_calibration_curve(_frame()),
    )
