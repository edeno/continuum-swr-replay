"""The brain-state reference and the counts by signal count only the time
inside the epoch's well visits. The measures are tested in
``tests/analysis/event_properties``; here, that each script takes the epoch's
visits and records what it kept and left out."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import scripts.event_properties.measure_brain_state_reference as brain_state
import scripts.event_properties.measure_nonlocal_by_signal as by_signal
from continuum_swr_replay.analysis.event_properties.nonlocal_by_signal import (
    nonlocal_counts,
)

IDENTIFIER = "j1620210706_02_r1_HPC"


def _frame() -> pd.DataFrame:
    """Ten immobile bins at 2 ms, quiet rest throughout; the last four non-local
    with high ripple power and high theta."""
    time = np.arange(10) * 0.002
    loud = np.arange(10) >= 6
    return pd.DataFrame(
        {
            "animal": "j16",
            "day": 20210706,
            "epoch": "02_r1",
            "head_speed": 1.0,
            "in_excluded_time": False,
            "non_local_event": 0,
            "SWR_event": 0,
            "MUA_event": 0,
            "theta_power_zscore": np.where(loud, 5.0, -1.0),
            "SWR_zscore": np.where(loud, 3.1, 0.1),
            "MUA_zscore": 0.0,
            "Non-Local_Continuous_prob": np.where(loud, 0.9, 0.1),
            "Non-Local_Fragmented_prob": 0.0,
        },
        index=pd.Index(time, name="time"),
    )


@pytest.fixture
def one_visit(monkeypatch):
    """The epoch's one well visit holds the first six bins (0 to 0.012 s, its
    final poke-out at the seventh bin, which is outside)."""
    for script in (brain_state, by_signal):
        monkeypatch.setattr(
            script.analyzed_time,
            "read_verified_dot_with_swr_source",
            lambda path, sha256: _frame(),
        )
        monkeypatch.setattr(script.inventory_run, "decode_path", lambda i: i)
        monkeypatch.setattr(
            script,
            "well_visit_times",
            lambda identifier: (np.array([0.0]), np.array([0.012])),
        )


def test_the_brain_state_rest_is_the_rest_inside_the_visits(one_visit):
    row = brain_state._epoch((IDENTIFIER, "sha"))
    assert row["n_rest_bins"] == 6
    assert row["n_rest_bins_outside_visit"] == 4
    assert row["theta_rest_median"] == -1.0
    assert row["decode_sha256"] == "sha"


def test_the_counts_by_signal_count_the_bins_inside_the_visits(one_visit):
    counts, kept = by_signal._epoch((IDENTIFIER, "sha"))
    assert kept == {"n_immobile_bins_in_visit": 6, "n_immobile_bins_outside_visit": 4}
    ripple = nonlocal_counts(counts, "ripple_power")
    assert ripple["n_bins"].sum() == 6
    # the non-local bins of high power are all outside the visit
    assert ripple["n_nonlocal"].sum() == 0
    assert ripple.loc[ripple["signal_low"] == 3.0, "n_bins"].sum() == 0
    assert counts["epoch_id"].eq(IDENTIFIER).all()
