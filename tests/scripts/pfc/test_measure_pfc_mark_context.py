"""The marked-kernel mark-context audit reads one row per event, per group."""

import numpy as np
import pandas as pd
import pytest

from scripts.pfc.measure_pfc_mark_context import (
    group_context_table,
    load_current_arm_marks,
    summarize,
)


def _synthetic_event_arm_table() -> pd.DataFrame:
    """Two shuffle groups of unequal size, plus non-current-arm rows.

    The first group (``is_swr`` false) holds four events whose durations
    double -- evenly spaced in log -- while ``arm_fraction`` falls in equal
    steps, so its correlation is exactly ``-1``. The second holds three. Every
    event also carries a non-current-arm row. One extra in-transit event
    carries a current-arm row, and one extra ``mixed`` event has no current arm
    (``is_current_arm`` missing, as the canonical table stores it, a float
    column); none of these must reach the statistics, because the event-grain
    at-well cohort is strictly ``at_well`` and a missing flag is not a current
    arm.
    """
    events = pd.DataFrame(
        {
            "event_id": np.arange(7),
            "animal": ["chimi"] * 7,
            "session_date": [20200212] * 7,
            "epoch": ["02_r1"] * 7,
            "is_swr": [False, False, False, False, True, True, True],
            "arm_fraction": [0.8, 0.6, 0.4, 0.2, 0.1, 0.5, 0.9],
            "occupancy_well_arm": [0, 0, 1, 1, 2, 2, 3],
            "event_start_s": [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "event_end_s": [0.02, 10.04, 20.08, 30.16, 40.02, 50.04, 60.08],
            "occupancy_state": ["at_well"] * 7,
        }
    )
    decoys = events.assign(arm_fraction=0.0, is_current_arm=0.0)
    off_well = events.iloc[[0, 0]].assign(
        event_id=[7, 8],
        arm_fraction=0.0,
        occupancy_state=["in_transit", "mixed"],
        event_start_s=[70.0, 80.0],
        event_end_s=[70.02, 80.02],
        is_current_arm=[1.0, np.nan],
    )
    return pd.concat(
        [events.assign(is_current_arm=1.0), decoys, off_well], ignore_index=True
    )


def test_groups_are_session_by_replay_class_and_drop_other_arms(tmp_path):
    path = tmp_path / "event_arm_table.parquet"
    _synthetic_event_arm_table().to_parquet(path)

    marks = load_current_arm_marks(path)
    table = group_context_table(marks)
    summary = summarize(table)

    assert "is_current_arm" not in marks.columns
    assert "event_id" not in marks.columns
    assert len(marks) == 7
    assert (marks["arm_fraction"] > 0).all()
    assert int(summary.loc[0, "n_groups"]) == 2
    assert int(summary.loc[0, "min_group_size"]) == 3
    # Durations 20/40/80/160 ms against a fraction falling in equal steps.
    non_swr = table.loc[~table["is_swr"]].iloc[0]
    assert non_swr["r_log_duration"] == pytest.approx(-1.0)


def test_a_constant_mark_gives_no_correlation_rather_than_a_zero(tmp_path):
    frame = _synthetic_event_arm_table()
    frame.loc[frame["is_swr"] & frame["is_current_arm"].eq(1.0), "arm_fraction"] = 0.5
    path = tmp_path / "event_arm_table.parquet"
    frame.to_parquet(path)

    table = group_context_table(load_current_arm_marks(path))

    swr = table.loc[table["is_swr"]].iloc[0]
    assert np.isnan(swr["r_log_duration"])
    assert np.isnan(swr["r2_occupancy_well_arm"])
