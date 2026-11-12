"""The per-bin summary figure's descriptive time course and its support."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

_SPEC = importlib.util.spec_from_file_location(
    "plot_per_bin_rate_summary",
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "rate"
    / "plot_per_bin_rate_summary.py",
)
ppbrs = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ppbrs)


def _clocked_stops() -> pd.DataFrame:
    """Two stops of one session: [10, 12] holds excluded time [10.5, 11.0];
    [20, 21] holds none. Family A's exposure is the span less excluded time."""
    return pd.DataFrame(
        {
            "animal": "a",
            "session_date": 1,
            "epoch": "02_r1",
            "stop_id": [1, 2],
            "stop_start": [10.0, 20.0],
            "stop_end": [12.0, 21.0],
            "immobility_duration_s": [1.5, 1.0],
        }
    )


def test_the_time_course_clock_is_the_stops_span_less_excluded_time(monkeypatch):
    """The empirical rate's clock is time since the stop began, so a stop's
    exposure in each lag bin is its span there less the excluded seconds in
    it -- not its first ``immobility_duration_s`` seconds."""
    excluded = pd.DataFrame(
        {
            "animal": ["a"],
            "session_date": [1],
            "epoch": ["02_r1"],
            "start_time": [10.5],
            "end_time": [11.0],
        }
    )
    monkeypatch.setattr(
        ppbrs, "excluded_intervals_by_session", lambda sessions: excluded
    )
    onsets = pd.DataFrame(
        {"event_type": "SWR", "stop_id": [1, 1], "t_since_stop_onset": [0.2, 1.6]}
    )

    tc, rate = ppbrs._timecourse(_clocked_stops(), onsets, "swr", dt=0.5, tmax=2.0)

    # exposure per 0.5 s lag bin: [0, .5) both stops; [.5, 1) stop 1 is excluded,
    # stop 2 is not; [1, 1.5) and [1.5, 2) stop 1 alone
    np.testing.assert_allclose(tc, [0.25, 0.75, 1.25, 1.75])
    np.testing.assert_allclose(rate, [1 / 1.0, 0 / 0.5, 0 / 0.5, 1 / 0.5])
    t, surviving = ppbrs._survival(_clocked_stops(), tmax=2.0, dt=0.5)
    np.testing.assert_allclose(surviving, [1.0, 1.0, 0.5, 0.5])
