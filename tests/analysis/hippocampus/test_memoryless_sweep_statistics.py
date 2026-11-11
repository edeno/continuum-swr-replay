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
