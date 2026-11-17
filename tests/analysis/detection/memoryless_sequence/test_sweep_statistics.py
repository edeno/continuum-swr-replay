"""The statistics of the replay sweep's summary, on summaries small enough to check by hand."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.detection.memoryless_sequence import (
    sweep_statistics as stats,
)


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


def test_pooled_timing_statistics_are_the_blocks_values_by_measure():
    summary = pd.DataFrame(
        {
            "summary": ["timing_pooled", "timing_pooled", "overall"],
            "measure": ["n_replays", "sliding_end_error_ms_max", "n_cells"],
            "value": [2.0, 150.0, 4.0],
        }
    )
    assert stats.sweep_pooled_timing_statistics(summary) == {
        "n_replays": 2.0,
        "sliding_end_error_ms_max": 150.0,
    }


@pytest.mark.parametrize("value", [None, float("nan")])
def test_pooled_timing_statistics_refuse_an_absent_block_or_a_missing_value(value):
    rows = [] if value is None else [("timing_pooled", "n_replays", value)]
    summary = pd.DataFrame(rows, columns=["summary", "measure", "value"])
    with pytest.raises(ValueError, match="timing_pooled"):
        stats.sweep_pooled_timing_statistics(summary)


def test_timing_exceedance_is_a_table_by_threshold():
    summary = pd.DataFrame(
        [
            ("timing_exceedance", f"{column}_error_beyond_{threshold}ms", share)
            for threshold, share in ((100, 0.2), (0, 1.0))
            for column in (
                "sliding_start",
                "sliding_end",
                "state_space_start",
                "state_space_end",
            )
        ],
        columns=["summary", "measure", "value"],
    )
    shares = stats.sweep_timing_exceedance(summary)

    assert shares.index.tolist() == [0, 100]
    assert shares["state_space_end"].tolist() == [1.0, 0.2]
    assert sorted(shares.columns) == [
        "sliding_end",
        "sliding_start",
        "state_space_end",
        "state_space_start",
    ]


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [("timing_exceedance", "sliding_start_error_beyond_soon", 0.5)],
        [
            ("timing_exceedance", "sliding_start_error_beyond_0ms", 1.0),
            ("timing_exceedance", "sliding_start_error_beyond_5ms", 0.5),
            ("timing_exceedance", "sliding_end_error_beyond_0ms", 1.0),
        ],
    ],
)
def test_timing_exceedance_refuses_an_absent_or_incomplete_block(rows):
    summary = pd.DataFrame(rows, columns=["summary", "measure", "value"])
    with pytest.raises(ValueError, match="timing_exceedance"):
        stats.sweep_timing_exceedance(summary)


def test_position_statistics_are_the_blocks_values_by_measure():
    summary = pd.DataFrame(
        {
            "summary": ["position_pooled", "position_pooled", "overall"],
            "measure": ["sliding_lead_cm_median", "sliding_speed_ratio_median", "n"],
            "value": [-2.0, 0.8, 4.0],
        }
    )
    assert stats.sweep_position_statistics(summary) == {
        "sliding_lead_cm_median": -2.0,
        "sliding_speed_ratio_median": 0.8,
    }
    with pytest.raises(ValueError, match="position_pooled"):
        stats.sweep_position_statistics(summary[summary["summary"] == "overall"])


def _by_speed(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(
        [("position_by_speed", *row) for row in rows],
        columns=["summary", "measure", "speed_m_per_s", "value"],
    )


def test_position_by_speed_is_a_table_slowest_first():
    by_speed = stats.sweep_position_by_speed(
        _by_speed(
            [
                ("sliding_position_error_cm", 5.0, 9.0),
                ("sliding_position_error_cm", 0.0, 1.4),
                ("state_space_position_error_cm", 0.0, 1.3),
                ("state_space_position_error_cm", 5.0, 1.5),
            ]
        )
    )
    assert by_speed.index.tolist() == [0.0, 5.0]
    assert by_speed["sliding_position_error_cm"].tolist() == [1.4, 9.0]


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [("sliding_position_error_cm", np.nan, 1.0)],
        [
            ("sliding_position_error_cm", 0.0, 1.4),
            ("sliding_position_error_cm", 5.0, 9.0),
            ("state_space_position_error_cm", 0.0, 1.3),
        ],
    ],
)
def test_position_by_speed_refuses_an_absent_or_incomplete_block(rows):
    with pytest.raises(ValueError, match="position_by_speed"):
        stats.sweep_position_by_speed(_by_speed(rows))
