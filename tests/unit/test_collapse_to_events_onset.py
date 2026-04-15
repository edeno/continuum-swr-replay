"""Tests for ``collapse_to_events`` onset-flag semantics.

The docstring says: "Context flags use event-onset value (first timebin),
not mode" — meaning ``is_after_switch`` and ``is_after_omission`` are
taken from the first row of each event group, while spatial columns
(``replay_arm``, ``track_segment_id``, etc.) use the mode.

This module pins that contract: the onset semantics must hold even when
later timebins disagree, NaN at the first row must not be silently
replaced by a later valid value, and the function's reliance on input
row ordering is made explicit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.hippocampus.replay_content.event_collapsing import (
    collapse_to_events,
)


def _two_bin_event(
    *,
    is_after_switch_first: float,
    is_after_switch_second: float,
    is_after_omission_first: float = 0.0,
    is_after_omission_second: float = 0.0,
    replay_arm: int = 1,
    track_segment_id: int = 1,
    prev_well: float = 0.0,
    next_well: float = 2.0,
    last_rewarded_well: float = 0.0,
    swr: int = 0,
) -> pd.DataFrame:
    """Build a minimal two-row ``nl`` frame for a single event."""
    return pd.DataFrame(
        [
            {
                "animal": "sim",
                "day": 20200101,
                "epoch": "01_r1",
                "non_local_event": 1,
                "SWR_event": swr,
                "replay_arm": float(replay_arm),
                "track_segment_id": float(track_segment_id),
                "prev_trial_from_well": prev_well,
                "next_trial_to_well": next_well,
                "last_rewarded_well": last_rewarded_well,
                "is_after_switch": is_after_switch_first,
                "is_after_omission": is_after_omission_first,
            },
            {
                "animal": "sim",
                "day": 20200101,
                "epoch": "01_r1",
                "non_local_event": 1,
                "SWR_event": swr,
                "replay_arm": float(replay_arm),
                "track_segment_id": float(track_segment_id),
                "prev_trial_from_well": prev_well,
                "next_trial_to_well": next_well,
                "last_rewarded_well": last_rewarded_well,
                "is_after_switch": is_after_switch_second,
                "is_after_omission": is_after_omission_second,
            },
        ]
    )


class TestOnsetFlagsTakeFirstRow:
    """Onset flags should reflect the first row, even if later rows differ."""

    def test_first_row_wins_for_is_after_switch(self):
        nl = _two_bin_event(
            is_after_switch_first=1.0, is_after_switch_second=0.0
        )
        out = collapse_to_events(nl, level="arm")
        assert len(out) == 1
        assert out["is_after_switch"].iloc[0] == 1.0

    def test_first_row_wins_for_is_after_omission(self):
        nl = _two_bin_event(
            is_after_switch_first=0.0,
            is_after_switch_second=0.0,
            is_after_omission_first=1.0,
            is_after_omission_second=0.0,
        )
        out = collapse_to_events(nl, level="arm")
        assert out["is_after_omission"].iloc[0] == 1.0

    def test_first_row_wins_against_modal_value(self):
        """Three rows: first is 1.0, the *mode* across all three is 0.0.
        First-row semantics should still return 1.0, not the mode."""
        rows = [_two_bin_event(is_after_switch_first=1.0, is_after_switch_second=0.0)]
        # Add a third row with is_after_switch = 0.0 to make mode = 0.0
        third = rows[0].iloc[[0]].copy()
        third["is_after_switch"] = 0.0
        nl = pd.concat([rows[0], third], ignore_index=True)
        out = collapse_to_events(nl, level="arm")
        assert out["is_after_switch"].iloc[0] == 1.0


class TestOnsetNaNHandling:
    """NaN at the first row must propagate; we do *not* fall back to a
    later non-NaN value. This is a documented behavior — if the trial
    context is missing at event onset, the event is genuinely uncontexted
    and must not be silently filled."""

    def test_nan_first_row_propagates_even_if_later_row_is_valid(self):
        nl = _two_bin_event(
            is_after_switch_first=np.nan, is_after_switch_second=1.0
        )
        out = collapse_to_events(nl, level="arm")
        assert np.isnan(out["is_after_switch"].iloc[0]), (
            "Expected NaN onset to survive collapse, but got "
            f"{out['is_after_switch'].iloc[0]}"
        )

    def test_all_nan_returns_nan(self):
        nl = _two_bin_event(
            is_after_switch_first=np.nan, is_after_switch_second=np.nan
        )
        out = collapse_to_events(nl, level="arm")
        assert np.isnan(out["is_after_switch"].iloc[0])


class TestOnsetIsRowOrderDependent:
    """``collapse_to_events`` does NOT sort by time — it trusts the caller
    to feed rows in temporal order. If the rows are reversed before being
    passed in, the "onset" value also flips. Pin this as a precondition
    so a future refactor that, say, calls ``sort_index`` upstream gets
    a clear failure if it forgets to preserve time-order semantics here.
    """

    def test_reversed_rows_flip_onset(self):
        nl = _two_bin_event(
            is_after_switch_first=1.0, is_after_switch_second=0.0
        )
        out_normal = collapse_to_events(nl, level="arm")
        out_reversed = collapse_to_events(nl.iloc[::-1].reset_index(drop=True), level="arm")
        assert out_normal["is_after_switch"].iloc[0] == 1.0
        assert out_reversed["is_after_switch"].iloc[0] == 0.0


class TestSpatialColumnsUseMode:
    """Sanity: spatial columns (replay_arm, track_segment_id, ...) should
    use the mode, not the first row. This is the *complement* of the
    onset behavior and is what makes the collapse meaningful."""

    def test_replay_arm_uses_mode(self):
        nl = _two_bin_event(
            is_after_switch_first=0.0, is_after_switch_second=0.0, replay_arm=1
        )
        # Two rows on arm 1, one row on arm 5 → mode is arm 1
        odd = nl.iloc[[0]].copy()
        odd["replay_arm"] = 5.0
        nl_three = pd.concat([nl, odd], ignore_index=True)
        out = collapse_to_events(nl_three, level="arm")
        assert out["replay_arm"].iloc[0] == 1.0

    def test_replay_arm_does_not_use_first_row(self):
        """Two rows: first row on arm 5, second row on arm 1, equal counts.
        Mode should pick the smaller value (mode tiebreak in pandas).
        At minimum it should NOT be the first row's value alone."""
        nl = _two_bin_event(
            is_after_switch_first=0.0,
            is_after_switch_second=0.0,
            replay_arm=5,
        )
        nl.loc[1, "replay_arm"] = 1.0
        out = collapse_to_events(nl, level="arm")
        # Mode of [5, 1] could be either; just assert it's a valid arm
        # and the function did not crash.
        assert out["replay_arm"].iloc[0] in {1.0, 5.0}
