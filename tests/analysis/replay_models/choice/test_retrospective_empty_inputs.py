"""Regression tests for ``contains_path_to_next`` empty-input handling.

The previous chained
``Series.groupby(...).agg(...).rename(columns=...)`` raised
``TypeError: Series.rename() got an unexpected keyword argument
'columns'`` when ``event_sequence_table`` was empty. The retrospective
runner caught any exception and silently dropped the whole session,
which made an entire session of valid retrospective events disappear
just because its sequence table happened to be empty.

The fix uses named aggregation (and a no-op short-circuit on empty
input) so the result is always a DataFrame.
"""

from __future__ import annotations

import pandas as pd

from continuum_swr_replay.analysis.event_tables.retrospective_metrics import (
    contains_path_to_next,
)


class TestContainsPathToNextEmptyInputs:
    def test_events_present_but_no_sequence_rows_does_not_raise(self):
        ev = pd.DataFrame(
            {
                "animal": ["m"],
                "session_date": [1],
                "epoch": ["e"],
                "event_id": [1],
                "occupancy_well_arm": [0],
                "next_arm": [1],
            }
        )
        seq = pd.DataFrame(
            columns=[
                "animal",
                "session_date",
                "epoch",
                "event_id",
                "t_in_event",
                "decoded_arm_map",
            ]
        )
        # Must not raise.
        out = contains_path_to_next(ev, seq)
        assert "contains_path_to_next" in out.columns
        # With no sequence rows, the path is not covered.
        assert bool(out["contains_path_to_next"].iloc[0]) is False
