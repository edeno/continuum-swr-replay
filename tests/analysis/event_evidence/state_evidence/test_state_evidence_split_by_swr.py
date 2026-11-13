"""The Non-Local SWR split of the held-out state-evidence windows."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_evidence.state_evidence import (
    split_non_local_by_swr,
)


def _windows() -> pd.DataFrame:
    """Two immobile Non-Local Continuous bouts (bins 2-4 and 10-12), one Local
    bout, and a matched baseline window for each Non-Local bout."""
    rows = [
        ("bout", "Non-Local Continuous", 2, 4, np.nan),
        ("bout", "Non-Local Continuous", 10, 12, np.nan),
        ("baseline", "Non-Local Continuous", np.nan, np.nan, 2),
        ("baseline", "Non-Local Continuous", np.nan, np.nan, 10),
        ("bout", "Local", 20, 25, np.nan),
        ("calibration", "Non-Local Continuous", np.nan, np.nan, np.nan),
    ]
    return pd.DataFrame(
        rows, columns=["kind", "state", "bout_s", "bout_e", "matched_bout_s"]
    ).assign(regime="immobile")


def test_a_bout_touching_an_swr_bin_and_its_matched_window_are_labelled_swr():
    swr = np.zeros(30, dtype=bool)
    swr[4] = True  # the last bin of the first bout
    out = split_non_local_by_swr(_windows(), swr)
    assert len(out) == 4  # Local and calibration rows are left out
    by_start = out.assign(
        key=np.where(out["kind"] == "bout", out["bout_s"], out["matched_bout_s"])
    ).set_index(["kind", "key"])["state"]
    assert by_start[("bout", 2)] == "Non-Local Continuous | SWR"
    assert by_start[("baseline", 2)] == "Non-Local Continuous | SWR"
    assert by_start[("bout", 10)] == "Non-Local Continuous | no SWR"
    assert by_start[("baseline", 10)] == "Non-Local Continuous | no SWR"


def test_an_swr_bin_just_outside_the_bout_does_not_count():
    swr = np.zeros(30, dtype=bool)
    swr[[1, 5]] = True
    out = split_non_local_by_swr(_windows(), swr)
    assert set(out["swr"]) == {"no SWR"}


def test_a_matched_window_without_its_bout_raises():
    windows = _windows()
    windows.loc[3, "matched_bout_s"] = 7
    with pytest.raises(ValueError, match="matched window has no bout"):
        split_non_local_by_swr(windows, np.zeros(30, dtype=bool))


def test_duplicate_bout_keys_raise():
    windows = _windows()
    windows.loc[1, ["bout_s", "bout_e"]] = [2, 4]
    with pytest.raises(ValueError, match="duplicate bout keys"):
        split_non_local_by_swr(windows, np.zeros(30, dtype=bool))
