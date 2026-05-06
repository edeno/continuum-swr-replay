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
                    "arm": arm,
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

    def test_all_nan_residuals_propagate_nan_to_per_stop_value(self):
        eat, et = self._eat_one_event_all_nan_residuals()
        out = stop_patch_replay(eat, et)
        # The crucial property: ``replay_frac_residual_noncurrent``
        # must be NaN, not 0.0, for a stop whose only event was
        # dropped under the event-wise residual filter.
        assert out["replay_frac_residual_noncurrent"].isna().all(), (
            "stop_patch_replay materialised a fabricated 0.0 for a "
            f"stop whose only event had all-NaN residuals: "
            f"{out['replay_frac_residual_noncurrent'].tolist()}"
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
                    "arm": arm,
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
                    "arm": arm,
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
