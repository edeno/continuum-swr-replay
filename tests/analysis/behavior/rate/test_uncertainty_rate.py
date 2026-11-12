"""Tests for analysis/behavior/rate/uncertainty_rate.py's session selection."""

from __future__ import annotations

import pandas as pd

from continuum_swr_replay.analysis.behavior.rate.uncertainty_rate import (
    drop_exposure_sessions,
)
from continuum_swr_replay.data_loaders import epoch_register


def _register(path, rows: list[tuple[str, str, str, str, bool]]) -> None:
    """``REGISTER_PATH`` contents: (nwb_file_name, epoch, status, reason,
    first_exposure) per entry, with evidence wherever a reason needs it."""
    pd.DataFrame(
        [
            [name, epoch, status, reason, "x" if reason else "", str(flag)]
            for name, epoch, status, reason, flag in rows
        ],
        columns=list(epoch_register.COLUMNS),
    ).to_csv(path, index=False)


def test_the_register_s_first_exposure_sessions_are_dropped_and_no_other(
    tmp_path, monkeypatch
):
    """The register names the session, not its place in the stop table.

    ``a``'s exposure (20200101 02_r1) goes and its same-day 04_r2 stays; ``b``'s
    exposure is excluded from the register, so the stop table lacks it and
    ``b``'s first stop-table session (20200105) stays, as peanut's does.
    """
    path = tmp_path / "epoch_register.csv"
    _register(
        path,
        [
            ("a20200101_.nwb", "02_r1", "analysed", "", True),
            ("a20200101_.nwb", "04_r2", "analysed", "", False),
            ("a20200102_.nwb", "02_r1", "analysed", "", False),
            ("b20200104_.nwb", "02_r1", "excluded", "NO_TRIAL_INTERVALS", True),
            ("b20200105_.nwb", "02_r1", "analysed", "", False),
            ("b20200106_.nwb", "02_r1", "analysed", "", False),
        ],
    )
    monkeypatch.setattr(epoch_register, "REGISTER_PATH", path)
    sessions = [
        ("a", 20200101, "02_r1"),
        ("a", 20200101, "04_r2"),
        ("a", 20200102, "02_r1"),
        ("b", 20200105, "02_r1"),
        ("b", 20200106, "02_r1"),
    ]
    stop = pd.DataFrame(
        [(*session, stop_id) for stop_id, session in enumerate(sessions * 2)],
        columns=["animal", "session_date", "epoch", "stop_id"],
    )

    out = drop_exposure_sessions(stop)

    assert out["stop_id"].tolist() == [1, 2, 3, 4, 6, 7, 8, 9]
    assert out.columns.equals(stop.columns)
