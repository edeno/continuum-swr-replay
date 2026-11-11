"""The other detectors' events against the non-local events, on a run small enough
to check by hand."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.inventory_comparison import (
    nonlocal_comparison as comparison,
)
from continuum_swr_replay.analysis.inventory_comparison.features import (
    STATE_SHARE_COLUMNS,
)

LOCAL, NO_SPIKE, NON_LOCAL, UNCERTAIN = STATE_SHARE_COLUMNS


def _events() -> pd.DataFrame:
    """Kay events as (animal, matched, local, no-spike, non-local, uncertain)."""
    rows = [
        # overlapping no non-local event: mostly Local, mostly No-Spike, mixed, unknown
        ("a", False, 0.8, 0.1, 0.0, 0.1),
        ("a", False, 0.2, 0.6, 0.0, 0.2),
        ("b", False, 0.4, 0.4, 0.0, 0.2),
        ("b", False, np.nan, np.nan, np.nan, np.nan),
        # overlapping a non-local event
        ("a", True, 0.0, 0.0, 1.0, 0.0),
        ("a", True, 0.2, 0.0, 0.6, 0.2),
        ("b", True, 0.0, 0.4, 0.4, 0.2),
        ("b", True, 0.0, 0.0, 1.0, 0.0),
    ]
    frame = pd.DataFrame(
        rows, columns=["animal", "matched", LOCAL, NO_SPIKE, NON_LOCAL, UNCERTAIN]
    )
    return frame.assign(detector="kay")


def test_other_detector_statistics_split_events_by_overlap_and_state():
    result = comparison.other_detector_statistics(_events()).loc["kay"]
    assert result["n_events"] == 8
    assert result["no_nonlocal_fraction"] == 0.5
    # of the three events with state shares that overlap no non-local event
    assert result["no_nonlocal_mostly_local_fraction"] == pytest.approx(1 / 3)
    assert result["no_nonlocal_mostly_no_spike_fraction"] == pytest.approx(1 / 3)
    assert result["no_nonlocal_mixed_fraction"] == pytest.approx(1 / 3)
    assert result["overlap_nonlocal_share_median"] == pytest.approx(0.8)
    assert result["overlap_local_share_median"] == 0.0
    assert result["overlap_any_local_fraction"] == 0.25
    assert result["overlap_any_no_spike_fraction"] == 0.25
    assert result["overlap_all_nonlocal_fraction"] == 0.5
    # each animal has half its events overlapping no non-local event
    assert result["no_nonlocal_fraction_animal_min"] == 0.5
    assert result["no_nonlocal_fraction_animal_max"] == 0.5


def _write_run(run_dir, report_dir) -> None:
    """One epoch: non-local events [0, 1] and [2, 3]; Kay covers [0.5, 1] and
    [10, 11]; MUA covers [2, 2.25]; the Widloski detector covers [2, 3]."""
    run_dir.mkdir()
    report_dir.mkdir()
    features = pd.DataFrame(
        {
            "detector": [
                "state_space",
                "state_space",
                "kay",
                "kay",
                "mua",
                "memoryless",
            ],
            "event_start_s": [0.0, 2.0, 0.5, 10.0, 2.0, 2.0],
            "event_end_s": [1.0, 3.0, 1.0, 11.0, 2.25, 3.0],
        }
    )
    features.to_parquet(run_dir / "a_features.parquet")
    membership = pd.DataFrame(
        {
            "detector": "state_space",
            "other_detector": ["kay", "kay", "kay", "memoryless", "memoryless"],
            "membership": ["matched", "unmatched", "unknown", "unmatched", "matched"],
        }
    )
    membership.to_parquet(run_dir / "a_pair_membership.parquet")
    pd.DataFrame(
        {"detector": ["state_space", "kay"], "exposure_s": [7200.0, 100.0]}
    ).to_csv(report_dir / "counts_by_animal.csv", index=False)
    pd.DataFrame(
        {
            "cell": ["sequence_matched_by_no_trigger", "another_cell"],
            "detector": "state_space",
            "n_events": [10, 99],
            "n_unmatched": [4, 99],
            "n_unknown": [2, 99],
        }
    ).to_csv(report_dir / "cells_by_animal.csv", index=False)


def test_left_out_seconds_subtract_each_covering_from_the_nonlocal_time(tmp_path):
    _write_run(tmp_path / "run", tmp_path / "report")
    seconds = comparison.left_out_seconds(tmp_path / "run")
    assert seconds == {
        "nonlocal": 2.0,
        "kay": 1.5,
        "outside_kay": 1.5,
        "outside_swr": 1.5,
        "outside_trigger": 1.25,
        "outside_widloski": 1.0,
    }


def test_left_out_statistics_count_events_time_and_rates(tmp_path):
    _write_run(tmp_path / "run", tmp_path / "report")
    result = comparison.left_out_statistics(tmp_path / "run", tmp_path / "report")
    assert result["rest_hours"] == 2.0
    assert result["nonlocal_time_s"] == 2.0
    assert result["nonlocal_time_outside_trigger_fraction"] == 0.625
    # one of the two events with a known membership against Kay is unmatched
    assert result["nonlocal_events_outside_kay"] == 1
    assert result["nonlocal_events_outside_kay_fraction"] == 0.5
    assert result["nonlocal_events_outside_kay_per_hour"] == 0.5
    assert result["nonlocal_events_outside_widloski"] == 1
    # the comparison's cell: 4 unmatched of the 10 - 2 with a known membership
    assert result["nonlocal_events_outside_trigger"] == 4
    assert result["nonlocal_events_outside_trigger_fraction"] == 0.5
    assert result["nonlocal_events_outside_trigger_per_hour"] == 2.0
