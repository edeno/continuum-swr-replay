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


def _without(block: str, measure: str, duration_s: float) -> pd.DataFrame:
    """``SWEEP`` less one measure's row at one duration."""
    row = (
        (SWEEP["summary"] == block)
        & (SWEEP["measure"] == measure)
        & (SWEEP["duration_s"] == duration_s)
    )
    assert row.sum() == 1
    return SWEEP[~row]


def _with_no_duration(block: str, measure: str, duration_s: float) -> pd.DataFrame:
    """``SWEEP`` with one measure's row at one duration left without a duration."""
    row = (
        (SWEEP["summary"] == block)
        & (SWEEP["measure"] == measure)
        & (SWEEP["duration_s"] == duration_s)
    )
    assert row.sum() == 1
    changed = SWEEP.copy()
    changed.loc[row, "duration_s"] = np.nan
    return changed


def test_sweep_jump_pass_statistics_refuse_a_row_with_no_duration():
    # sorted, a missing duration would come last and be read as the longest
    changed = _with_no_duration(
        "shuffle_pass_rate_by_duration", "sliding_jump_pass", 0.04
    )
    with pytest.raises(ValueError, match="no duration_s"):
        stats.sweep_jump_pass_statistics(changed)


def test_sweep_jump_pass_statistics_refuse_a_summary_without_the_block():
    with pytest.raises(ValueError, match="has no row"):
        stats.sweep_jump_pass_statistics(SWEEP[SWEEP["summary"] == "overall"])


def test_sweep_timing_statistics_are_in_ms_and_signed_as_early_and_late():
    result = stats.sweep_timing_statistics(SWEEP)
    assert result["shortest_duration_s"] == 0.04
    assert result["longest_duration_s"] == 0.2
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


@pytest.mark.parametrize(
    ("block", "measure"),
    [
        ("timing_by_duration", "sliding_start_error_ms_median"),
        ("timing_by_duration", "sliding_end_error_ms_median"),
        ("timing_by_duration", "sliding_end_error_ms_q75"),
        ("timing_by_duration", "state_space_start_error_ms_median"),
        ("timing_by_duration", "state_space_end_error_ms_median"),
        ("median_duration_ratio_by_duration", "sliding_duration_ratio"),
    ],
)
def test_sweep_timing_statistics_refuse_measures_at_different_durations(block, measure):
    # without its 40 ms row, a measure's first value is the 100 ms one
    with pytest.raises(ValueError, match="not all at the same durations"):
        stats.sweep_timing_statistics(_without(block, measure, 0.04))


def test_sweep_timing_statistics_refuse_a_row_with_no_duration():
    changed = _with_no_duration(
        "timing_by_duration", "sliding_end_error_ms_median", 0.1
    )
    with pytest.raises(ValueError, match="no duration_s"):
        stats.sweep_timing_statistics(changed)


def test_sweep_timing_statistics_do_not_skip_a_missing_error():
    row = (SWEEP["measure"] == "sliding_end_error_ms_median") & (
        SWEEP["duration_s"] == 0.1
    )
    changed = SWEEP.copy()
    changed.loc[row, "value"] = np.nan
    result = stats.sweep_timing_statistics(changed)
    assert np.isnan(result["sliding_end_late_min_ms"])
    assert np.isnan(result["sliding_end_late_max_ms"])


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
    # another output's share at one duration, and a row pooled over durations
    ("bursts", "fragmented_share", "placeless_burst", 0.2, SLIDING, 0.01),
    ("bursts", "fragmented_share", "placeless_burst", np.nan, OURS, 0.99),
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


def test_burst_share_range_is_over_one_outputs_shares_at_the_burst_durations():
    result = stats.burst_share_range(
        CONTROLS, "placeless_burst", "fragmented_share", OURS
    )
    assert result == {"min": 0.6, "max": 0.75}


def test_burst_share_range_refuses_a_duration_the_output_has_no_one_share_at():
    # the sliding window has a share at one of the three durations only
    with pytest.raises(ValueError, match="0 rows"):
        stats.burst_share_range(
            CONTROLS, "placeless_burst", "fragmented_share", SLIDING
        )
    twice = pd.concat(
        [CONTROLS, CONTROLS[CONTROLS["value"] == 0.75]], ignore_index=True
    )
    with pytest.raises(ValueError, match="2 rows"):
        stats.burst_share_range(twice, "placeless_burst", "fragmented_share", OURS)


def test_burst_share_range_refuses_a_condition_with_no_share():
    with pytest.raises(ValueError, match="at any burst duration"):
        stats.burst_share_range(CONTROLS, "no_such_burst", "fragmented_share", OURS)


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


# --- the held-out check ---------------------------------------------------------


def _heldout(rows: list[tuple]) -> pd.DataFrame:
    """A held-out summary from (block, measure, fold, estimator, detector,
    event_class, truth, value) rows."""
    return pd.DataFrame(
        rows,
        columns=[
            "summary",
            "measure",
            "fold",
            "estimator",
            "detector",
            "event_class",
            "truth",
            "value",
        ],
    )


def _by_class(
    fold: str,
    event_class: str,
    truth: str,
    measures: dict[str, float],
    estimator: str = "state_space",
    detector: str = "state_space",
) -> list[tuple]:
    return [
        ("by_class", measure, fold, estimator, detector, event_class, truth, value)
        for measure, value in measures.items()
    ]


REPLAY = "continuous_replay"
HELDOUT_ROWS = [
    # fold A: 2 of 4 informative ours-only replays and 12 of 16 shared ones have
    # support; 4 of 10 and 16 of 20 are informative
    *_by_class(
        "A",
        "ours_only",
        REPLAY,
        {
            "n_events_all": 10,
            "informative_share": 0.4,
            "n_events": 4,
            "support_rate": 0.5,
        },
    ),
    *_by_class(
        "A",
        "shared",
        REPLAY,
        {
            "n_events_all": 20,
            "informative_share": 0.8,
            "n_events": 16,
            "support_rate": 0.75,
        },
    ),
    # fold B: 3 of 5 and 12 of 15; 5 of 10 and 15 of 30 informative
    *_by_class(
        "B",
        "ours_only",
        REPLAY,
        {
            "n_events_all": 10,
            "informative_share": 0.5,
            "n_events": 5,
            "support_rate": 0.6,
        },
    ),
    *_by_class(
        "B",
        "shared",
        REPLAY,
        {
            "n_events_all": 30,
            "informative_share": 0.5,
            "n_events": 15,
            "support_rate": 0.8,
        },
    ),
    # fold C: no ours-only replay is informative
    *_by_class(
        "C",
        "ours_only",
        REPLAY,
        {
            "n_events_all": 3,
            "informative_share": 0.0,
            "n_events": 0,
            "support_rate": np.nan,
        },
    ),
    # the sliding window's content of the same events: 1 of 4 and 8 of 16 in
    # fold A, 2 of 5 and 3 of 15 in fold B
    *_by_class(
        "A",
        "ours_only",
        REPLAY,
        {"n_events": 4, "support_rate": 0.25},
        "sliding_window",
    ),
    *_by_class(
        "A", "shared", REPLAY, {"n_events": 16, "support_rate": 0.5}, "sliding_window"
    ),
    *_by_class(
        "B", "ours_only", REPLAY, {"n_events": 5, "support_rate": 0.4}, "sliding_window"
    ),
    *_by_class(
        "B", "shared", REPLAY, {"n_events": 15, "support_rate": 0.2}, "sliding_window"
    ),
    # the shared events as the sliding window bounds them are not counted
    *_by_class(
        "A",
        "shared",
        REPLAY,
        {"n_events": 100, "support_rate": 0.1},
        detector="sliding_window",
    ),
    # another truth: place-less bursts, 0 of 3 with support in fold A and 1 of 2
    # in fold B; at the floor of 2 spikes, 1 of 5 and 2 of 4
    *_by_class(
        "A",
        "ours_only",
        "no_content",
        {
            "n_events": 3,
            "support_rate": 0.0,
            "n_events_floor2": 5,
            "support_rate_floor2": 0.2,
        },
    ),
    *_by_class(
        "B",
        "ours_only",
        "no_content",
        {
            "n_events": 2,
            "support_rate": 0.5,
            "n_events_floor2": 4,
            "support_rate_floor2": 0.5,
        },
    ),
    (
        "error_rates",
        "false_support_rate",
        "A",
        "state_space",
        "state_space",
        "all",
        "",
        0.0,
    ),
    ("agreement", "estimator_agreement", "A", "both", "both", "all", "", 0.99),
    ("agreement", "estimator_agreement", "B", "both", "both", "all", "", 0.9),
    ("agreement", "estimator_agreement", "C", "both", "both", "all", "", 1.0),
    ("agreement", "n_informative_under_both", "A", "both", "both", "all", "", 0.5),
]
HELDOUT = _heldout(HELDOUT_ROWS)
EITHER = ("ours_only", "shared")


def test_pooled_heldout_rate_weights_each_class_by_its_informative_events():
    result = stats.pooled_heldout_rate(HELDOUT, "A", EITHER, REPLAY)
    # (2 + 12) of (4 + 16), not the mean of 0.5 and 0.75
    assert result["rate"] == pytest.approx(0.7)
    assert result["n_events"] == 20


def test_pooled_heldout_rate_of_one_class_is_that_class_rate():
    result = stats.pooled_heldout_rate(HELDOUT, "B", ("shared",), REPLAY)
    assert result["rate"] == pytest.approx(0.8)
    assert result["n_events"] == 15


def test_pooled_informative_share_weights_each_class_by_all_its_events():
    result = stats.pooled_heldout_rate(
        HELDOUT, "A", EITHER, REPLAY, measure="informative_share"
    )
    # (4 + 16) of (10 + 20)
    assert result["rate"] == pytest.approx(2 / 3)
    assert result["n_events"] == 30


def test_pooled_heldout_rate_reads_the_named_estimator_and_floor():
    sliding = stats.pooled_heldout_rate(
        HELDOUT, "A", EITHER, REPLAY, estimator="sliding_window"
    )
    # (1 + 8) of (4 + 16)
    assert sliding["rate"] == pytest.approx(0.45)
    loose = stats.pooled_heldout_rate(
        HELDOUT, "A", ("ours_only",), "no_content", suffix="_floor2"
    )
    assert loose["rate"] == pytest.approx(0.2)
    assert loose["n_events"] == 5


def test_pooled_heldout_rate_is_nan_when_no_event_counts():
    none_informative = stats.pooled_heldout_rate(HELDOUT, "C", ("ours_only",), REPLAY)
    assert np.isnan(none_informative["rate"])
    assert none_informative["n_events"] == 0
    no_rows = stats.pooled_heldout_rate(HELDOUT, "C", ("shared",), REPLAY)
    assert np.isnan(no_rows["rate"])
    assert no_rows["n_events"] == 0


def test_heldout_fold_statistics_range_over_folds_and_count_events():
    result = stats.heldout_fold_statistics(HELDOUT, "AB", EITHER, REPLAY)
    # fold A: 14 of 20; fold B: 15 of 20
    assert result["rate_min"] == pytest.approx(0.7)
    assert result["rate_max"] == pytest.approx(0.75)
    assert result["rate_spread"] == pytest.approx(0.05)
    assert result["n_events"] == 40
    assert result["n_supported"] == 29
    assert result["n_folds_with_support"] == 2


def test_heldout_fold_statistics_find_the_lowest_rate_in_a_later_fold():
    result = stats.heldout_fold_statistics(
        HELDOUT, "AB", EITHER, REPLAY, estimator="sliding_window"
    )
    # fold A: 9 of 20; fold B: 5 of 20
    assert result["rate_min"] == pytest.approx(0.25)
    assert result["rate_max"] == pytest.approx(0.45)


def test_heldout_fold_statistics_count_supported_events_and_folds():
    main = stats.heldout_fold_statistics(HELDOUT, "AB", ("ours_only",), "no_content")
    assert (main["n_events"], main["n_supported"]) == (5, 1)
    # fold A's three informative events have no support
    assert main["n_folds_with_support"] == 1
    loose = stats.heldout_fold_statistics(
        HELDOUT, "AB", ("ours_only",), "no_content", suffix="_floor2"
    )
    assert (loose["n_events"], loose["n_supported"]) == (9, 3)


def test_heldout_fold_statistics_leave_out_a_fold_with_no_counted_event():
    result = stats.heldout_fold_statistics(HELDOUT, "ABC", ("ours_only",), REPLAY)
    assert (result["n_events"], result["n_supported"]) == (9, 5)
    assert result["n_folds_with_support"] == 2
    assert np.isnan(result["rate_spread"])


def test_lowest_estimator_agreement_is_over_folds():
    assert stats.lowest_estimator_agreement(HELDOUT) == 0.9


def test_heldout_event_classes_are_those_of_the_by_class_rows():
    assert stats.heldout_event_classes(HELDOUT) == {"ours_only", "shared"}
    with_comparator = _heldout(
        HELDOUT_ROWS + _by_class("A", "comparator_only", REPLAY, {"n_events_all": 1})
    )
    assert stats.heldout_event_classes(with_comparator) == {
        "ours_only",
        "shared",
        "comparator_only",
    }
