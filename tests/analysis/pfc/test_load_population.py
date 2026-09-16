"""Tests for analysis/pfc/load_population.py.

The historical ``df["animal"] = df["nwb_file_name"].str.split("2").str[0]``
returned ``"j1"`` for j16 sessions (e.g. ``j1620210706_.nwb``) because
the date YYYYMMDD that follows the animal name often contains a ``"2"``
early on. Pin the parse_nwb_filename-based replacement.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.pfc.load_population import (
    HELDOUT_P_COLUMNS,
    _extract_animal,
    load_population_data,
)


@pytest.mark.parametrize(
    "nwb_file_name,expected",
    [
        ("chimi20200212_.nwb", "chimi"),
        ("peanut20201125_.nwb", "peanut"),
        ("wilbur20210329_.nwb", "wilbur"),
        ("senor20201018_.nwb", "senor"),
        # Critical case: j16 session — the date 20210706 contains "2"
        # at position 1, so the legacy str.split("2")[0] returned "j1".
        ("j1620210706_.nwb", "j16"),
    ],
)
def test_extract_animal(nwb_file_name, expected):
    assert _extract_animal(nwb_file_name) == expected


def _write_bundle(directory, epoch_identifier, suffix, firing_rate):
    """Write one session's modulation bundle under the cohort's suffix."""
    n_lag_times = 201
    modulation = np.zeros((n_lag_times, 1))
    frame = pd.DataFrame(
        {
            "nwb_file_name": ["chimi20200212_.nwb"],
            "immobility_avg_firing_rate": [firing_rate],
            "SWR: heldout_deviance": [100.0],
            "min_heldout_model": ["SWR"],
            **{column: [0.025] for column in HELDOUT_P_COLUMNS.values()},
        }
    )
    base = directory / f"{epoch_identifier}_mPFC_modulation{suffix}"
    frame.to_pickle(base.with_suffix(".pkl"))
    for name in (
        "SWR_modulation",
        "no_SWR_modulation",
        "non_local_SWR_modulation",
        "non_local_no_SWR_modulation",
        "matched_control_modulation",
    ):
        np.save(base.with_suffix(f".{name}.npy"), modulation)


def test_only_the_requested_cohorts_bundles_are_pooled(tmp_path):
    # Two cohorts side by side in one directory, distinguishable only by the
    # suffix. Pooling them would double the population and mix two rules.
    _write_bundle(tmp_path, "chimi20200212_02_r1", "", firing_rate=10.0)
    _write_bundle(tmp_path, "chimi20200212_02_r1", "_allstops", firing_rate=20.0)

    canonical, _, _ = load_population_data(tmp_path, min_firing_rate=0.0)
    sensitivity, _, _ = load_population_data(
        tmp_path, min_firing_rate=0.0, artifact_suffix="_allstops"
    )

    assert len(canonical) == 1
    assert canonical["immobility_avg_firing_rate"].iloc[0] == 10.0
    assert len(sensitivity) == 1
    assert sensitivity["immobility_avg_firing_rate"].iloc[0] == 20.0


def test_a_missing_cohort_bundle_is_named_in_the_error(tmp_path):
    _write_bundle(tmp_path, "chimi20200212_02_r1", "", firing_rate=10.0)

    with pytest.raises(FileNotFoundError, match="_allstops"):
        load_population_data(tmp_path, artifact_suffix="_allstops")
