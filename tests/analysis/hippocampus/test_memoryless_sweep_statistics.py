"""The statistics of the memoryless sweeps' summaries, on summaries small enough to check by hand."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.hippocampus.memoryless_sequence import (
    sweep_statistics as stats,
)

# --- the injected-replay sweep --------------------------------------------------


def _sweep(rows: list[tuple]) -> pd.DataFrame:
    """A replay sweep summary from (block, measure, duration_s, value) rows."""
    return pd.DataFrame(rows, columns=["summary", "measure", "duration_s", "value"])


# durations are listed out of order: every by-duration statistic sorts them
SWEEP_ROWS = [
    ("overall", "n_cells", np.nan, 200.0),
    ("overall", "sliding_matches", np.nan, 0.2),
    # 3 of 200 and 29 of 200; as floats, 0.145 * 200 is just short of 29
    ("overall", "found_by_sliding_window_only", np.nan, 0.015),
    ("overall", "found_by_sliding_window_only_before_shuffle", np.nan, 0.145),
    ("overall_in_domain", "n_cells", np.nan, 50.0),
    ("overall_in_domain", "sliding_matches", np.nan, 0.6),
    ("shuffle_pass_rate_by_duration", "sliding_jump_pass", 0.2, 0.9),
    ("shuffle_pass_rate_by_duration", "sliding_jump_pass", 0.04, 0.3),
    ("shuffle_pass_rate_by_duration", "sliding_jump_pass", 0.1, 0.6),
    ("shuffle_pass_rate_by_duration", "sliding_spread_pass", 0.02, 0.7),
    ("shuffle_pass_rate_by_duration", "sliding_spread_pass", 0.5, 0.99),
    ("timing_by_duration", "sliding_start_error_ms_median", 0.1, -12.0),
    ("timing_by_duration", "sliding_start_error_ms_median", 0.04, -25.0),
    ("timing_by_duration", "sliding_start_error_ms_median", 0.2, -30.0),
    ("timing_by_duration", "sliding_end_error_ms_median", 0.2, 55.0),
    ("timing_by_duration", "sliding_end_error_ms_median", 0.04, 40.0),
    ("timing_by_duration", "sliding_end_error_ms_median", 0.1, 18.0),
    ("timing_by_duration", "sliding_end_error_ms_q25", 0.04, 5.0),
    ("timing_by_duration", "sliding_end_error_ms_q25", 0.1, 6.0),
    ("timing_by_duration", "sliding_end_error_ms_q25", 0.2, 7.0),
    ("timing_by_duration", "sliding_end_error_ms_q75", 0.04, 60.0),
    ("timing_by_duration", "sliding_end_error_ms_q75", 0.1, 70.0),
    ("timing_by_duration", "sliding_end_error_ms_q75", 0.2, 65.0),
    # the largest state-space error is an early start, 4 ms
    ("timing_by_duration", "state_space_start_error_ms_median", 0.04, -4.0),
    ("timing_by_duration", "state_space_start_error_ms_median", 0.1, 2.0),
    ("timing_by_duration", "state_space_start_error_ms_median", 0.2, 1.0),
    ("timing_by_duration", "state_space_end_error_ms_median", 0.04, 3.0),
    ("timing_by_duration", "state_space_end_error_ms_median", 0.1, -1.0),
    ("timing_by_duration", "state_space_end_error_ms_median", 0.2, 2.0),
    ("median_duration_ratio_by_duration", "sliding_duration_ratio", 0.2, 1.25),
    ("median_duration_ratio_by_duration", "sliding_duration_ratio", 0.04, 2.5),
    ("median_duration_ratio_by_duration", "sliding_duration_ratio", 0.1, 1.5),
    ("median_duration_ratio_by_duration", "state_space_duration_ratio", 0.02, 9.0),
]
SWEEP = _sweep(SWEEP_ROWS)


def test_sweep_measures_are_one_block_by_measure():
    overall = stats.sweep_measures(SWEEP, "overall")
    assert overall["sliding_matches"] == 0.2
    assert stats.sweep_measures(SWEEP, "overall_in_domain")["sliding_matches"] == 0.6
    assert list(overall.index) == [
        "n_cells",
        "sliding_matches",
        "found_by_sliding_window_only",
        "found_by_sliding_window_only_before_shuffle",
    ]


def test_sweep_grid_statistics_turn_shares_of_the_grid_into_counts():
    result = stats.sweep_grid_statistics(SWEEP)
    assert result["n_replays"] == 200
    assert result["domain_fraction"] == pytest.approx(0.25)
    assert result["n_sliding_only"] == 3
    assert result["n_sliding_only_before_test"] == 29


def test_sweep_grid_statistics_refuse_a_share_that_is_no_whole_count():
    # 0.0125 of 200 replays is 2.5 replays
    rows = [
        (block, measure, duration, 0.0125)
        if measure == "found_by_sliding_window_only"
        else (block, measure, duration, value)
        for block, measure, duration, value in SWEEP_ROWS
    ]
    with pytest.raises(ValueError, match="not a whole count"):
        stats.sweep_grid_statistics(_sweep(rows))


def test_sweep_jump_pass_statistics_read_the_shortest_and_longest_durations():
    result = stats.sweep_jump_pass_statistics(SWEEP)
    assert result["shortest_duration_s"] == 0.04
    assert result["shortest_fraction"] == 0.3
    assert result["longest_duration_s"] == 0.2
    assert result["longest_fraction"] == 0.9


def test_sweep_timing_statistics_are_in_ms_and_signed_as_early_and_late():
    result = stats.sweep_timing_statistics(SWEEP)
    assert result["state_space_edge_error_max_ms"] == 4.0
    # start errors -25, -12, -30 ms at 40, 100, 200 ms
    assert result["sliding_start_early_min_ms"] == 12.0
    assert result["sliding_start_early_max_ms"] == 30.0
    assert result["shortest_start_early_ms"] == 25.0
    # end errors 40, 18, 55 ms
    assert result["sliding_end_late_min_ms"] == 18.0
    assert result["sliding_end_late_max_ms"] == 55.0
    assert result["shortest_end_late_ms"] == 40.0
    assert result["sliding_end_late_upper_quartile_max_ms"] == 70.0
    # 40 ms replays reported 2.5 times as long
    assert result["shortest_reported_duration_ms"] == pytest.approx(100.0)


# --- the controls ---------------------------------------------------------------

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
    ("bursts", "called_rate", "placeless_burst", np.nan, OURS, 0.9),
    ("bursts", "called_rate", "placeless_burst", 0.08, OURS, 0.8),
    ("bursts", "called_rate", "placeless_burst", np.nan, SLIDING, 0.1),
    ("bursts", "called_rate", "jumping_burst", np.nan, OURS, 1.0),
    ("bursts", "fragmented_share", "placeless_burst", 0.08, OURS, 0.6),
    ("bursts", "fragmented_share", "placeless_burst", 0.2, OURS, 0.75),
    ("bursts", "fragmented_share", "placeless_burst", 0.5, OURS, 0.7),
    ("bursts", "fragmented_share", "jumping_burst", 0.08, OURS, 0.1),
    ("bursts", "continuous_share", "placeless_burst", 0.08, OURS, 0.05),
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
            CONTROLS, "bursts", "called_rate", "placeless_burst", level, OURS
        )

    assert called(None) == 0.9
    assert called(0.08) == 0.8


def test_controls_value_refuses_no_row_and_several_rows():
    with pytest.raises(ValueError, match="0 rows"):
        stats.controls_value(
            CONTROLS, "bursts", "called_rate", "placeless_burst", 0.3, OURS
        )
    twice = pd.concat([CONTROLS, CONTROLS.iloc[[7]]], ignore_index=True)
    with pytest.raises(ValueError, match="2 rows"):
        stats.controls_value(
            twice, "bursts", "called_rate", "placeless_burst", None, OURS
        )


def test_no_replay_statistics_bound_the_rate_by_the_rule_of_three():
    result = stats.no_replay_statistics(CONTROLS)
    assert result["minutes"] == 1.5
    assert result["n_recordings"] == 8
    assert result["n_outputs"] == 3
    assert result["n_outputs_with_events"] == 0
    # 3 events over 1.5 minutes
    assert result["rate_bound_per_minute"] == pytest.approx(2.0)


def test_no_replay_statistics_count_the_outputs_that_reported_an_event():
    rows = [
        (*row[:5], 2.0) if row[1] == "n_events" and row[4] == SLIDING else row
        for row in CONTROLS_ROWS
    ]
    result = stats.no_replay_statistics(_controls(rows))
    assert (result["n_outputs"], result["n_outputs_with_events"]) == (3, 1)


def test_burst_share_range_is_over_one_conditions_burst_durations():
    result = stats.burst_share_range(CONTROLS, "placeless_burst", "fragmented_share")
    assert result == {"min": 0.6, "max": 0.75}


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
