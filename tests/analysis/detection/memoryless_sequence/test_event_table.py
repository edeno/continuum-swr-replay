"""Replay calls decided by windows outside every piece, on a table small enough to check by hand."""

from __future__ import annotations

import pandas as pd

from continuum_swr_replay.analysis.detection.memoryless_sequence.event_table import (
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
        # decided: the pieces sit above the threshold by less than the
        # detector's tolerance, which the detector reads as at the threshold
        (True, 1, 12.0 * (1 + 1e-12), True),
        # decided: the pieces have no dispersion, which is not above a threshold
        (True, 3, float("nan"), False),
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
        "n_replay_calls": 7,
        "n_decided_by_outside_window": 4,
        "n_decided_passing_shuffle": 2,
    }


def test_pieces_clearly_above_the_threshold_decide_the_call_themselves():
    # 1e-6 above the threshold is beyond the detector's tolerance
    above = CANDIDATES.assign(pieces_dispersion_cm=12.0 * (1 + 1e-6))
    result = outside_window_call_statistics(above, THRESHOLD_CM)
    assert result["n_replay_calls"] == 7
    assert result["n_decided_by_outside_window"] == 0


def test_outside_window_call_statistics_of_no_candidates_are_zero():
    result = outside_window_call_statistics(CANDIDATES.iloc[:0], THRESHOLD_CM)
    assert result == {
        "n_replay_calls": 0,
        "n_decided_by_outside_window": 0,
        "n_decided_passing_shuffle": 0,
    }
