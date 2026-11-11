"""Part 4's SWRs-without-replay values on a synthetic extended population."""

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.pfc.load_population import HELDOUT_P_COLUMNS
from continuum_swr_replay.analysis.pfc.modulation import THREE_CLASS_MODEL
from continuum_swr_replay.reported.mpfc import (
    SWR_WITHOUT_REPLAY_MIN_EVENTS,
    TAGS,
    _value_factory,
    swr_without_replay_values,
)

LAG_TIME = np.arange(-100, 101) / 500.0


@pytest.fixture
def extended():
    n = 40
    rng = np.random.default_rng(3)
    population = pd.DataFrame(
        {
            "nwb_file_name": np.repeat(
                ["chimi20200212_.nwb", "wilbur20210329_.nwb"], n // 2
            ),
            "epoch_name": "02_r1",
            "animal": np.repeat(["chimi", "wilbur"], n // 2),
            # Units 0-9 called for both, 10-19 for SWRs without replay only,
            # 20-29 for SWR-coupled replay only, 30-39 for neither.
            "SWR_no_non_local: p_heldout": np.where(np.arange(n) < 20, 0.01, 0.5),
            HELDOUT_P_COLUMNS["non_local_SWR"]: np.where(
                (np.arange(n) < 10) | ((np.arange(n) >= 20) & (np.arange(n) < 30)),
                0.01,
                0.5,
            ),
            "SWR_no_non_local: peak_log_modulation": rng.normal(0, 1, n),
            "non_local_SWR_3class: peak_log_modulation": rng.normal(0, 1, n),
            f"{THREE_CLASS_MODEL} vs. non_local_SWR + non_local_no_SWR: heldout_delta_dev": np.where(
                np.arange(n) % 4 == 0, 1.0, -1.0
            ),
            "min_heldout_model_three_class": np.where(
                np.arange(n) % 5 == 0, THREE_CLASS_MODEL, "constant"
            ),
        }
    )
    bump = np.exp(-0.5 * ((LAG_TIME - 0.05) / 0.02) ** 2)
    curves = {"swr_no_non_local": np.outer(bump, np.where(np.arange(n) % 2, -1.0, 1.0))}
    sidecars = [
        {
            "nwb_file_name": "chimi20200212_.nwb",
            "epoch_name": "02_r1",
            "n_swr_no_non_local": 4,
            "n_swr_with_non_local": 40,
            "n_swr_no_non_local_unknown": 1,
        },
        {
            "nwb_file_name": "wilbur20210329_.nwb",
            "epoch_name": "02_r1",
            "n_swr_no_non_local": 30,
            "n_swr_with_non_local": 60,
            "n_swr_no_non_local_unknown": 0,
        },
    ]
    return population, curves, sidecars


def test_counts_overlap_and_both_subsets(extended):
    population, curves, sidecars = extended
    make = _value_factory("synthetic", TAGS["paper"])

    values = {
        v.key: v.value
        for v in swr_without_replay_values(population, curves, LAG_TIME, sidecars, make)
    }

    assert values["swr_no_replay_events"] == 34
    assert values["swr_no_replay_events_fraction"] == pytest.approx(34 / 135)
    assert values["swr_no_replay_sessions_min_events"] == 1
    assert values["swr_no_replay_overlap_both_all"] == 10
    assert values["swr_no_replay_overlap_swr_no_replay_only_all"] == 10
    assert values["swr_no_replay_overlap_swr_replay_only_all"] == 10
    assert values["swr_no_replay_overlap_neither_all"] == 10
    assert values["swr_no_replay_called_fraction_all"] == 0.5
    # Only wilbur's session has at least the floor of events; its units are
    # 20-39, none called for SWRs without replay.
    assert SWR_WITHOUT_REPLAY_MIN_EVENTS <= 30
    assert values["swr_no_replay_units_min_events"] == 20
    assert values["swr_no_replay_called_fraction_min_events"] == 0.0
    assert values["swr_no_replay_called_fraction_all_chimi"] == 1.0
    assert values["swr_no_replay_adds_to_split_fraction_all"] == 0.25
    assert values["swr_no_replay_timing_units_enhanced_all"] == 10
    assert values["swr_no_replay_timing_peak_ms_enhanced_all"] == pytest.approx(50)
    for key in values:
        assert not any(ch.isdigit() for ch in key)
