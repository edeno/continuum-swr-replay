"""Regression tests for NaN propagation in replay aggregations.

Two helpers used to fabricate real zeros from missing data via
pandas' default ``skipna=True`` aggregations:

- ``stop_patch_replay`` (``analysis.replay.switch_analysis``): when
  ``predict_arm_fractions`` drops an event by setting
  ``residual_frac = NaN``, the per-event sum of
  ``residual_frac × (1 - is_current_patch)`` returned 0.0 and the
  per-stop mean carried that 0.0 forward.
- ``chosen_patch_fraction`` (``analysis.replay.retrospective``): when
  every target-patch arm has ``arm_fraction = NaN``, the sum of
  target-patch masses returned 0.0 instead of NaN.

Both now use ``skipna=False`` so missing data propagates instead of
becoming a confident zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.replay.retrospective import (
    chosen_patch_fraction,
)
from continuum_swr_replay.analysis.replay.switch_analysis import (
    stop_patch_replay,
)


class TestStopPatchReplayNaNPropagation:
    def _eat_one_event_all_nan_residuals(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        # 1 stop, 1 event, 9 arms. ``residual_frac`` is NaN on every
        # arm — emulates an event that ``predict_arm_fractions``
        # dropped under the new event-wise NaN filter.
        eat = pd.DataFrame(
            [
                {
                    "animal": "m",
                    "day": 20200101,
                    "epoch": "02_r1",
                    "event_id": 1,
                    "stop_id": 10,
                    "candidate_arm": arm,
                    "arm_fraction": 1.0 / 9,
                    "residual_frac": np.nan,
                    "current_patch": 1,
                }
                for arm in range(9)
            ]
        )
        et = pd.DataFrame(
            [
                {
                    "animal": "m",
                    "day": 20200101,
                    "epoch": "02_r1",
                    "event_id": 1,
                    "current_patch": 1,
                }
            ]
        )
        return eat, et

    def test_missing_current_patch_yields_nan_for_both_summaries(self):
        # User repro: ``current_patch`` is NaN for the only event,
        # so "current vs non-current" is undefined. The pre-fix code
        # made ``is_cp == False`` everywhere (NaN == anything is
        # False), reporting ``replay_fraction_current_patch == 0.0`` and
        # piling the entire replay fraction into ``replay_fraction_noncurrent``.
        eat = pd.DataFrame(
            [
                {
                    "animal": "m",
                    "day": 20200101,
                    "epoch": "02_r1",
                    "event_id": 1,
                    "stop_id": 10,
                    "candidate_arm": arm,
                    "arm_fraction": 1.0 / 9,
                    "current_patch": np.nan,
                }
                for arm in range(9)
            ]
        )
        et = pd.DataFrame(
            [
                {
                    "animal": "m",
                    "day": 20200101,
                    "epoch": "02_r1",
                    "event_id": 1,
                    "current_patch": np.nan,
                }
            ]
        )
        out = stop_patch_replay(eat, et)
        # Both patch-relative summaries undefined when current_patch
        # is missing.
        assert out["replay_fraction_current_patch"].isna().all(), (
            "replay_fraction_current_patch should be NaN when current_patch "
            f"is missing; got {out['replay_fraction_current_patch'].tolist()}"
        )
        assert out["replay_fraction_noncurrent"].isna().all(), (
            f"replay_fraction_noncurrent collapsed to "
            f"{out['replay_fraction_noncurrent'].tolist()} for an event with "
            "missing current_patch — should be NaN, not the full "
            "replay fraction routed to non-current."
        )

    def test_all_nan_residuals_propagate_nan_to_per_stop_value(self):
        eat, et = self._eat_one_event_all_nan_residuals()
        out = stop_patch_replay(eat, et)
        # The crucial property: ``replay_fraction_residual_noncurrent``
        # must be NaN, not 0.0, for a stop whose only event was
        # dropped under the event-wise residual filter.
        assert out["replay_fraction_residual_noncurrent"].isna().all(), (
            "stop_patch_replay materialised a fabricated 0.0 for a "
            f"stop whose only event had all-NaN residuals: "
            f"{out['replay_fraction_residual_noncurrent'].tolist()}"
        )


class TestChosenPatchFractionNaNPropagation:
    def test_all_nan_target_arm_fractions_yield_nan_event_value(self):
        # 1 event, 9 arms. next_patch=1 (covers arms 0,1,6 per
        # TRACK_SEGMENT_TO_PATCH). All target-patch ``arm_fraction``
        # values are NaN — emulates an event with no real-arm bins
        # contributing to the chosen patch's reconstruction.
        target_arms = {0, 1, 6}
        eat = pd.DataFrame(
            [
                {
                    "animal": "m",
                    "day": 20200101,
                    "epoch": "02_r1",
                    "event_id": 1,
                    "candidate_arm": arm,
                    # NaN on target-patch arms; arbitrary value on others.
                    "arm_fraction": np.nan if arm in target_arms else 0.0,
                }
                for arm in range(9)
            ]
        )
        et = pd.DataFrame(
            [
                {
                    "animal": "m",
                    "day": 20200101,
                    "epoch": "02_r1",
                    "event_id": 1,
                    "next_patch": 1,
                }
            ]
        )
        out = chosen_patch_fraction(et, eat)
        # Single event → single row. Must be NaN, not 0.0.
        assert len(out) == 1
        assert pd.isna(out["chosen_patch_fraction"].iloc[0]), (
            "chosen_patch_fraction returned "
            f"{out['chosen_patch_fraction'].iloc[0]} for an event whose "
            "target-patch arm fractions are all NaN; should propagate NaN"
        )

    def test_missing_target_arm_rows_yield_nan(self):
        # Event_arm_table has no rows for the target patch (e.g. only
        # rows for arms outside patch 1). The pre-fix form set every
        # non-target row to 0.0 and ``sum`` returned 0.0 — visually
        # identical to "we observed zero replay on the chosen patch".
        # The fix returns NaN because no target-patch row was present
        # at all.
        eat = pd.DataFrame(
            [
                {
                    "animal": "m",
                    "day": 20200101,
                    "epoch": "02_r1",
                    "event_id": 1,
                    "candidate_arm": arm,
                    # Only arms 4, 5 (patch 3) — no patch-1 rows at all.
                    "arm_fraction": 0.5,
                }
                for arm in (4, 5)
            ]
        )
        et = pd.DataFrame(
            [
                {
                    "animal": "m",
                    "day": 20200101,
                    "epoch": "02_r1",
                    "event_id": 1,
                    "next_patch": 1,  # patch 1 has no rows merged in
                }
            ]
        )
        out = chosen_patch_fraction(et, eat)
        assert pd.isna(out["chosen_patch_fraction"].iloc[0]), (
            "chosen_patch_fraction returned "
            f"{out['chosen_patch_fraction'].iloc[0]} for an event "
            "with no target-patch rows in event_arm_table; "
            "should propagate NaN, not 0.0."
        )

    def test_empty_event_arm_table_yields_nan(self):
        eat = pd.DataFrame(
            columns=[
                "animal", "day", "epoch", "event_id", "candidate_arm", "arm_fraction",
            ]
        )
        et = pd.DataFrame(
            [
                {
                    "animal": "m",
                    "day": 20200101,
                    "epoch": "02_r1",
                    "event_id": 1,
                    "next_patch": 1,
                }
            ]
        )
        out = chosen_patch_fraction(et, eat)
        # Single event → single row. Must be NaN, not 0.0.
        assert len(out) == 1
        assert pd.isna(out["chosen_patch_fraction"].iloc[0])

    def test_known_zero_replay_still_returns_zero(self):
        # All arms in the target patch have arm_fraction=0.0
        # (non-NaN). The function must still return 0.0 — the fix
        # propagates NaN only when the data is missing.
        target_arms = {0, 1, 6}
        eat = pd.DataFrame(
            [
                {
                    "animal": "m",
                    "day": 20200101,
                    "epoch": "02_r1",
                    "event_id": 1,
                    "candidate_arm": arm,
                    "arm_fraction": 0.0 if arm in target_arms else 0.5,
                }
                for arm in range(9)
            ]
        )
        et = pd.DataFrame(
            [
                {
                    "animal": "m",
                    "day": 20200101,
                    "epoch": "02_r1",
                    "event_id": 1,
                    "next_patch": 1,
                }
            ]
        )
        out = chosen_patch_fraction(et, eat)
        assert out["chosen_patch_fraction"].iloc[0] == 0.0
