"""Regression test for ``get_signal_windows`` nearest-sample selection.

``np.searchsorted`` returns the *insertion* index — the next sample
to the right of the query time. For an off-grid event time that
falls between two samples, that's not the closest sample. With
samples every 2 ms and an event at 0.0025 s, the centre used to be
0.004 s instead of the nearer 0.002 s.

Fix: compare distances to the bracketing samples and pick the
closer one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.hippocampus.triggered_analysis import (
    get_signal_windows,
)


def _data_frame_2ms_grid() -> pd.DataFrame:
    times = np.arange(0.0, 0.020, 0.002)  # 0, 2, 4, …, 18 ms
    return pd.DataFrame(
        {
            "animal": ["m"] * len(times),
            "day": [20200101] * len(times),
            "epoch": ["02_r1"] * len(times),
            "signal": np.arange(len(times), dtype=float),  # 0..9
        },
        index=times,
    )


def _events_at(time_s: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "animal": ["m"],
            "day": [20200101],
            "epoch": ["02_r1"],
            "event_time": [time_s],
        }
    )


def _center_window_values(time_s: float) -> np.ndarray:
    """Run ``get_signal_windows`` and return the 3-sample centred window."""
    arr = get_signal_windows(
        data=_data_frame_2ms_grid(),
        event_times=_events_at(time_s),
        signal_cols=["signal"],
        time_col="event_time",
        window_sec=0.004,  # ±2 ms ⇒ 3 samples (left, center, right)
        sampling_frequency=500.0,
    )
    # Shape (n_events=1, n_time=3, n_signals=1).
    return arr.values[0, :, 0]


class TestNearestSampleCenter:
    def test_event_closer_to_left_sample_centers_on_left(self):
        # Event at 0.0025 s; left = 0.002 (Δ=0.0005), right = 0.004 (Δ=0.0015).
        # Centre on the LEFT (sample value 1) ⇒ window [0, 1, 2].
        np.testing.assert_array_equal(
            _center_window_values(0.0025), np.array([0.0, 1.0, 2.0])
        )

    def test_event_exactly_on_sample(self):
        # Exact match ⇒ that sample is the centre.
        # Sample at 0.004 has value 2 ⇒ window [1, 2, 3].
        np.testing.assert_array_equal(
            _center_window_values(0.004), np.array([1.0, 2.0, 3.0])
        )

    def test_event_closer_to_right_sample_centers_on_right(self):
        # Event at 0.0035 s; left = 0.002 (Δ=0.0015), right = 0.004 (Δ=0.0005).
        # Centre on the RIGHT (sample value 2) ⇒ window [1, 2, 3].
        np.testing.assert_array_equal(
            _center_window_values(0.0035), np.array([1.0, 2.0, 3.0])
        )
