"""The gamma windows around event starts are made as Figure 2g's windows are."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.paths import EVENT_GAMMA_DIR
from scripts.event_properties import measure_gamma_trigger_windows as runner

EPOCH = "chimi20200212_02_r1_HPC"
TIME = np.arange(6) * 0.002


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"animal": "chimi", "day": 20200212, "epoch": "02_r1"},
        index=pd.Index(TIME, name="time"),
    )


def _write_trace(root, time: np.ndarray) -> None:
    folder = root / EVENT_GAMMA_DIR
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "time": time,
            "slow_gamma_z": np.arange(6, dtype=np.float32),
            "fast_gamma_z": -np.arange(6, dtype=np.float32),
        }
    ).to_parquet(folder / f"{EPOCH}_gamma_z.parquet")


def test_the_gamma_columns_join_the_frame_and_go_through_event_start_windows(
    processed_data_root, monkeypatch
):
    """The epoch's per-bin gamma is put on its decode frame, and the windows
    are ``triggered_analysis.event_start_windows`` of that frame (the function
    behind Figure 2g, which leaves out a window that touches excluded time)."""
    _write_trace(processed_data_root, TIME)
    seen = {}

    def windows(frame, event_types, signals):
        seen.update(frame=frame, event_types=event_types, signals=signals)
        return pd.DataFrame({"kind": ["overlap"], "signal": ["slow_gamma_z"]})

    monkeypatch.setattr(runner, "get_event_types", lambda dot: {"events": dot})
    monkeypatch.setattr(runner, "event_start_windows", windows)
    monkeypatch.setattr(runner, "frame_event_counts", lambda types: {"total": 3})

    rows, record = runner.measure(_frame(), np.ones(6, dtype=bool))

    assert rows["signal"].tolist() == ["slow_gamma_z"]
    assert seen["signals"] == ("slow_gamma_z", "fast_gamma_z")
    np.testing.assert_array_equal(seen["frame"]["slow_gamma_z"], np.arange(6))
    np.testing.assert_array_equal(seen["frame"]["fast_gamma_z"], -np.arange(6))
    assert record["status"] == "ok" and record["n_events"] == {"total": 3}
    assert len(record["gamma_z_sha256"]) == 64


def test_a_gamma_trace_on_another_time_grid_is_refused(processed_data_root):
    _write_trace(processed_data_root, TIME + 0.001)
    with pytest.raises(ValueError, match="time grid is not the decode frame's"):
        runner.measure(_frame(), np.ones(6, dtype=bool))
