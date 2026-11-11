"""Replay calls decided by windows outside every piece, on a table small enough to check by hand."""

from __future__ import annotations

import pandas as pd

from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.event_table import (
    outside_window_call_statistics,
)

THRESHOLD_CM = 12.0

CANDIDATES = pd.DataFrame(
    [
        # a replay call with no window outside its pieces
        (True, 0, 20.0, True),
        # decided by its outside window, and significant
        (True, 1, 5.0, True),
        # decided by its outside windows (its pieces sit at the threshold, not
        # above it), and not significant
        (True, 2, 12.0, False),
        # an outside window, but the pieces clear the threshold alone
        (True, 1, 30.0, True),
        # pieces below the threshold, but no outside window to decide the call
        (True, 0, 5.0, True),
        # not a replay call
        (False, 1, 5.0, False),
    ],
    columns=[
        "is_replay",
        "n_kept_windows_outside_pieces",
        "pieces_dispersion_cm",
        "shuffle_pass",
    ],
)


def test_outside_window_call_statistics_count_the_calls_the_windows_decide():
    result = outside_window_call_statistics(CANDIDATES, THRESHOLD_CM)
    assert result == {
        "n_replay_calls": 5,
        "n_decided_by_outside_window": 2,
        "n_decided_passing_shuffle": 1,
    }


def test_outside_window_call_statistics_of_no_candidates_are_zero():
    result = outside_window_call_statistics(CANDIDATES.iloc[:0], THRESHOLD_CM)
    assert result == {
        "n_replay_calls": 0,
        "n_decided_by_outside_window": 0,
        "n_decided_passing_shuffle": 0,
    }
