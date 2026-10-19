"""``compare_fits.outcome_boundary_shift_cohort`` predicts the shifted stops
from the design's own bins: over the stop's span, less the bins excluded time
(and the history window) drops. Offline: the script is loaded from its file and
its table read and excluded time are replaced."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from continuum_swr_replay.analysis.replay import per_bin_design

_SPEC = importlib.util.spec_from_file_location(
    "compare_fits",
    Path(__file__).resolve().parents[2] / "scripts" / "qa" / "compare_fits.py",
)
compare_fits = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(compare_fits)


def _stops() -> pd.DataFrame:
    """Two stops of 3 s. Stop 1's outcome becomes known 2.5 s in, and excluded
    time [10.0, 10.9] takes 0.9 s from its start, so its exposure (2.1 s) ends
    before the boundary while its bins, over the whole span, still run past it.
    Stop 2's outcome becomes known 1 s in, with nothing excluded."""
    return pd.DataFrame(
        {
            "animal": "a",
            "session_date": 1,
            "epoch": "02_r1",
            "stop_id": [1, 2],
            "stop_start": [10.0, 20.0],
            "stop_end": [13.0, 23.0],
            "immobility_duration_s": [2.1, 3.0],
            "outcome_known_time_s": [12.5, 21.0],
            "last_well_arrival_time_s": [9.0, 19.0],
            "occupancy_state": "at_well",
        }
    )


def test_a_stop_with_a_bin_past_its_boundary_shifts_whatever_its_exposure(
    monkeypatch,
):
    monkeypatch.setattr(compare_fits, "read_table", lambda path: _stops())
    monkeypatch.setattr(
        per_bin_design,
        "excluded_intervals_by_session",
        lambda sessions: pd.DataFrame(
            {
                "animal": ["a"],
                "session_date": [1],
                "epoch": ["02_r1"],
                "start_time": [10.0],
                "end_time": [10.9],
            }
        ),
    )

    table = compare_fits.outcome_boundary_shift_cohort(Path("stops"))

    row = table.set_index("occupancy_state").loc["all"]
    assert row["n_crossing_boundary"] == 2
    assert row["n_shifted"] == 2
