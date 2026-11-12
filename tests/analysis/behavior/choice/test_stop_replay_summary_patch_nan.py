"""Regression test for ``stop_replay_summary`` patch-level NaN handling.

The patch-level fractions sum ``arm_fraction`` across arms whose
patch label matches the role's patch. The pre-fix groupby used
default ``.sum()`` (``skipna=True``), so a target-patch group with
every arm NaN produced 0.0 instead of NaN. The user reproduced
``frac_on_current = NaN`` but ``frac_on_current_patch = 0.0`` —
missing patch content was indistinguishable from confidently zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.behavior.choice.learning import stop_replay_summary


def _toy_inputs() -> dict[str, pd.DataFrame]:
    """One stop, one event. Target patch (1) arms 0/1/6 all NaN,
    other arms have explicit zero."""
    sess = {"animal": "m", "session_date": 1, "epoch": "e"}
    target_arms = {0, 1, 6}  # patch 1
    eat = pd.DataFrame(
        [
            {
                **sess,
                "event_id": 1,
                "stop_id": 10,
                "candidate_arm": arm,
                "arm_fraction": np.nan if arm in target_arms else 0.0,
                # ``stop_replay_summary`` reads ``recent_reward_rate``
                # to pick the per-event "best" arm; supply zeros so
                # the helper finds *some* well arm with a finite rate.
                "reward_fraction_candidate_arm_last10_visits": 0.0,
            }
            for arm in range(9)
        ]
    )
    et = pd.DataFrame(
        [
            {
                **sess,
                "event_id": 1,
                "stop_id": 10,
                "occupancy_well_arm": 0,  # patch 1 — the NaN patch
                "prev_arm": 0,
                "next_arm": 0,
                "last_rewarded_arm": 0,
                "is_swr": False,
                "is_mua": False,
            }
        ]
    )
    st = pd.DataFrame(
        [
            {
                **sess,
                "stop_id": 10,
                "occupancy_well_arm": 0,
                "prev_arm": 0,
                "last_rewarded_arm": 0,
                "is_rewarded_outcome": 0.0,
                "did_last_completed_trip_switch_patch": 0.0,
                "is_stay": 0.0,
                "is_swr_stop": False,
                "is_mua_stop": False,
                "n_selected_events": 1,
                "immobility_duration_s": 1.0,
            }
        ]
    )
    return {"event_arm_table": eat, "event_table": et, "stop_table": st}


class TestStopReplaySummaryPatchNaN:
    def test_all_nan_target_patch_propagates_nan_not_zero(self):
        tables = _toy_inputs()
        out = stop_replay_summary(
            stop_table=tables["stop_table"],
            event_table=tables["event_table"],
            event_arm_table=tables["event_arm_table"],
        )
        # ``frac_on_current`` is the arm-level value (already NaN
        # because arm 0's arm_fraction is NaN). The patch-level
        # value must also be NaN — the sum over patch-1 arms,
        # which are all NaN.
        row = out.iloc[0]
        assert pd.isna(row["frac_on_current"]), (
            f"arm-level ``frac_on_current`` should be NaN; got {row['frac_on_current']}"
        )
        assert pd.isna(row["frac_on_current_patch"]), (
            "patch-level ``frac_on_current_patch`` returned "
            f"{row['frac_on_current_patch']} for an event whose "
            "target-patch arm fractions are all NaN — should "
            "propagate NaN, not 0.0."
        )
