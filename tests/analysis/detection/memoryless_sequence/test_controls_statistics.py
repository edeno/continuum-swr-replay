"""The statistics of the controls sweep's summary, on summaries small enough to check by hand."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.detection.memoryless_sequence import (
    controls_statistics as stats,
)

OURS, SLIDING = "state_space_canonical", "sliding_published"


def _controls(rows: list[tuple]) -> pd.DataFrame:
    """A controls summary from (block, measure, condition, level, detector, value)."""
    return pd.DataFrame(
        rows,
        columns=["summary", "measure", "condition", "level", "detector", "value"],
    )


def _cells(measure: str, condition: str, detector: str, by_level: dict) -> list[tuple]:
    return [
        ("parts", measure, condition, level, detector, value)
        for level, value in by_level.items()
    ]


CONTROLS_ROWS = [
    ("no_replay", "immobile_minutes", "no_replay", np.nan, np.nan, 1.5),
    ("no_replay", "n_events", "no_replay", np.nan, "sliding_accepted", 0.0),
    ("no_replay", "n_events", "no_replay", np.nan, SLIDING, 0.0),
    ("no_replay", "n_events", "no_replay", np.nan, OURS, 0.0),
    ("no_replay", "events_per_minute", "no_replay", np.nan, OURS, 7.0),
    ("provenance", "n_seeds", np.nan, np.nan, np.nan, 4.0),
    ("provenance", "n_recordings", np.nan, np.nan, np.nan, 8.0),
    ("control_events", "called_rate", "placeless_event", np.nan, OURS, 0.9),
    ("control_events", "called_rate", "placeless_event", 0.08, OURS, 0.8),
    ("control_events", "called_rate", "placeless_event", np.nan, SLIDING, 0.1),
    ("control_events", "called_rate", "jumping_event", np.nan, OURS, 1.0),
    ("control_events", "fragmented_share", "placeless_event", 0.08, OURS, 0.6),
    ("control_events", "fragmented_share", "placeless_event", 0.2, OURS, 0.75),
    ("control_events", "fragmented_share", "placeless_event", 0.5, OURS, 0.7),
    ("control_events", "fragmented_share", "jumping_event", 0.08, OURS, 0.1),
    # another output's share at one duration, and a row pooled over durations
    ("control_events", "fragmented_share", "placeless_event", 0.2, SLIDING, 0.01),
    ("control_events", "fragmented_share", "placeless_event", np.nan, OURS, 0.99),
    ("control_events", "continuous_share", "placeless_event", 0.08, OURS, 0.05),
    # jumps of 0, 20, 40 and 80 cm, listed out of order
    *_cells("segmentation_spanning", "j1", OURS, {20: 0.9, 0: 0.5, 80: 0.4, 40: 0.2}),
    *_cells("segmentation_spanning", "j2", OURS, {0: 0.7, 20: 0.7, 40: 0.0, 80: 0.2}),
    *_cells("segmentation_spanning", "j1", SLIDING, {0: 0.0, 20: 0.0, 40: 0.0, 80: 1}),
    *_cells("segmentation_one_per_part", "j1", OURS, {0: 0, 20: 0.1, 40: 0.8, 80: 0.6}),
    # a chain at gaps of 10 and 100 ms
    *_cells("part_recovery", "j1", OURS, {0: 1.0, 20: 0.8, 40: 0.6, 80: 0.4}),
    *_cells("part_recovery", "c1", OURS, {0.01: 0.2, 0.1: 0.6}),
    *_cells("part_recovery_alone", "j1", OURS, {0: 1.0, 20: 1.0, 40: 1.0, 80: 1.0}),
    *_cells("part_recovery_alone", "c1", OURS, {0.01: 0.4, 0.1: 0.6}),
    *_cells("part_recovery", "j1", SLIDING, {0: 0.0, 20: 0.0, 40: 0.0, 80: 0.0}),
    *_cells("part_recovery", "c1", SLIDING, {0.01: 0.0, 0.1: 0.0}),
    ("parts", "n_trials", "j1", np.nan, OURS, 40.0),
    ("parts", "n_trials", "c1", np.nan, OURS, 20.0),
    ("parts", "n_trials", "j2", np.nan, OURS, 1000.0),
    ("parts", "n_trials_fewer_in_context", "j1", np.nan, OURS, 5.0),
    ("parts", "n_trials_fewer_in_context", "c1", np.nan, OURS, 2.0),
    ("parts", "n_trials_more_in_context", "j1", np.nan, OURS, 1.0),
    ("parts", "n_trials_more_in_context", "c1", np.nan, OURS, 0.0),
]
CONTROLS = _controls(CONTROLS_ROWS)


def test_controls_value_reads_one_row_at_a_level_or_pooled_over_levels():
    def called(level):
        return stats.controls_value(
            CONTROLS, "control_events", "called_rate", "placeless_event", level, OURS
        )

    assert called(None) == 0.9
    assert called(0.08) == 0.8


def test_controls_value_refuses_no_row_and_several_rows():
    with pytest.raises(ValueError, match="0 rows"):
        stats.controls_value(
            CONTROLS, "control_events", "called_rate", "placeless_event", 0.3, OURS
        )
    twice = pd.concat([CONTROLS, CONTROLS.iloc[[7]]], ignore_index=True)
    with pytest.raises(ValueError, match="2 rows"):
        stats.controls_value(
            twice, "control_events", "called_rate", "placeless_event", None, OURS
        )


def test_no_replay_statistics_bound_the_rate_by_the_rule_of_three():
    result = stats.no_replay_statistics(CONTROLS)
    assert result["minutes"] == 1.5
    assert result["n_recordings"] == 8
    assert result["n_outputs"] == 3
    # 3 events over 1.5 minutes
    assert result["rate_bound_per_minute"] == pytest.approx(2.0)


@pytest.mark.parametrize("count", [2.0, np.nan])
def test_no_replay_statistics_refuse_an_output_that_reported_an_event(count):
    # a missing count is not a count of none
    rows = [
        (*row[:5], count) if row[1] == "n_events" and row[4] == SLIDING else row
        for row in CONTROLS_ROWS
    ]
    with pytest.raises(ValueError, match="3 outputs were run, 1 reported events"):
        stats.no_replay_statistics(_controls(rows))


def test_no_replay_statistics_refuse_a_summary_where_no_output_was_run():
    counts = (CONTROLS["summary"] == "no_replay") & (CONTROLS["measure"] == "n_events")
    with pytest.raises(ValueError, match="0 outputs were run"):
        stats.no_replay_statistics(CONTROLS[~counts])


def test_no_replay_statistics_refuse_two_counts_for_one_output():
    twice = pd.concat([CONTROLS, CONTROLS.iloc[[2]]], ignore_index=True)
    assert twice.iloc[-1]["measure"] == "n_events"
    with pytest.raises(ValueError, match="several rows for one detector output"):
        stats.no_replay_statistics(twice)


@pytest.mark.parametrize("measure", ["immobile_minutes", "n_recordings"])
def test_no_replay_statistics_refuse_two_rows_of_the_minutes_or_recordings(measure):
    row = CONTROLS[CONTROLS["measure"] == measure]
    assert len(row) == 1
    twice = pd.concat([CONTROLS, row.assign(value=99.0)], ignore_index=True)
    with pytest.raises(ValueError, match=f"{measure}: 2 rows"):
        stats.no_replay_statistics(twice)
    with pytest.raises(ValueError, match=f"{measure}: 0 rows"):
        stats.no_replay_statistics(CONTROLS[CONTROLS["measure"] != measure])


def test_control_event_share_range_is_over_one_outputs_shares_at_the_event_durations():
    result = stats.control_event_share_range(
        CONTROLS, "placeless_event", "fragmented_share", OURS
    )
    assert result == {"min": 0.6, "max": 0.75}


def test_control_event_share_range_refuses_a_duration_the_output_has_no_one_share_at():
    # the sliding window has a share at one of the three durations only
    with pytest.raises(ValueError, match="0 rows"):
        stats.control_event_share_range(
            CONTROLS, "placeless_event", "fragmented_share", SLIDING
        )
    twice = pd.concat(
        [CONTROLS, CONTROLS[CONTROLS["value"] == 0.75]], ignore_index=True
    )
    with pytest.raises(ValueError, match="2 rows"):
        stats.control_event_share_range(
            twice, "placeless_event", "fragmented_share", OURS
        )


def test_control_event_share_range_refuses_a_condition_with_no_share():
    with pytest.raises(ValueError, match="at any control event duration"):
        stats.control_event_share_range(
            CONTROLS, "no_such_event", "fragmented_share", OURS
        )


def test_parts_by_level_is_the_mean_over_conditions_at_each_level():
    result = stats.parts_by_level(CONTROLS, ("j1", "j2"), "segmentation_spanning", OURS)
    # j1 and j2 at 0, 20, 40, 80 cm: (0.5, 0.7), (0.9, 0.7), (0.2, 0.0), (0.4, 0.2)
    assert list(result.index) == [0, 20, 40, 80]
    assert result.to_numpy() == pytest.approx([0.6, 0.8, 0.1, 0.3])
    # one condition: its own values, here a chain's at gaps of 10 and 100 ms
    recovery = stats.parts_by_level(CONTROLS, ("c1",), "part_recovery", OURS)
    assert recovery.to_dict() == {0.01: 0.2, 0.1: 0.6}


def test_parts_by_level_refuses_a_condition_without_a_row_at_a_level():
    # c1 was not run at the jump sizes
    with pytest.raises(ValueError, match="0 rows"):
        stats.parts_by_level(CONTROLS, ("j1", "c1"), "part_recovery", OURS)


def test_carving_statistics_pool_conditions_at_each_level():
    result = stats.carving_statistics(CONTROLS, ("j1", "j2"), "spanning", OURS)
    # means of j1 and j2 at 0, 20, 40, 80 cm: 0.6, 0.8, 0.1, 0.3
    assert result["min"] == pytest.approx(0.1)
    assert result["max"] == pytest.approx(0.8)
    assert result["at_lowest_level"] == pytest.approx(0.6)
    assert result["at_highest_level"] == pytest.approx(0.3)


def test_carving_statistics_of_one_condition_are_its_own_shares():
    result = stats.carving_statistics(CONTROLS, ("j1",), "spanning", OURS)
    assert result == {
        "min": 0.2,
        "max": 0.9,
        "at_lowest_level": 0.5,
        "at_highest_level": 0.4,
    }


def test_part_recovery_statistics_average_cells_and_sum_trials():
    result = stats.part_recovery_statistics(CONTROLS, ("j1", "c1"), OURS)
    # six cells, four of j1 and two of c1: (1.0 + 0.8 + 0.6 + 0.4 + 0.2 + 0.6) / 6,
    # not the mean of the two conditions' means (0.55)
    assert result["recovery_in_event"] == pytest.approx(0.6)
    # (4 * 1.0 + 0.4 + 0.6) / 6
    assert result["recovery_alone"] == pytest.approx(5 / 6)
    assert result["n_trials"] == 60
    assert result["n_trials_fewer_in_event"] == 7
    assert result["n_trials_more_in_event"] == 1
