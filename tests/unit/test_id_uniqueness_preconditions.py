"""Regression tests for ID-uniqueness preconditions.

The replay-content helpers groupby ``event_id`` / ``stop_id`` to
compute softmax-normalized fractions, per-stop means, and
current-vs-noncurrent patch sums. Those operations silently produce
wrong results — fractions summing above 1, dropped stops, sessions
mixed together — when the IDs collide across sessions. The canonical
hashed IDs (``hash_event_id`` / ``hash_stop_id``) mix session keys
into the hash so collisions don't happen, but direct callers can
still pass session-local IDs by mistake. The helpers must catch that
precondition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay.choice_prediction import (
    build_choice_dataset,
    predict_arm_fractions,
)
from continuum_swr_replay.analysis.replay.switch_analysis import (
    stop_patch_replay,
)
from continuum_swr_replay.analysis.replay.tables import (
    build_event_patch_table,
)


def _two_session_event_arm_table_colliding_event_ids() -> pd.DataFrame:
    """Two sessions, both using event_id=1 — what session-local IDs look like."""
    rows = []
    for animal, day in (("a", 20200101), ("b", 20200202)):
        for arm in range(9):
            rows.append(
                {
                    "animal": animal,
                    "day": day,
                    "epoch": "02_r1",
                    "event_id": 1,  # collision across sessions
                    "stop_id": 1,
                    "arm": arm,
                    "arm_fraction": 1.0 / 9,
                    "visit_recency": 0.0,
                    "cumulative_reward": 0.0,
                    "recent_reward_rate": 0.0,
                    "dist_to_arm": 0.0,
                    "is_same_patch": 0,
                    "current_patch": 1,
                }
            )
    return pd.DataFrame(rows)


class TestPredictArmFractionsPrecondition:
    def test_session_local_event_ids_raise(self):
        eat = _two_session_event_arm_table_colliding_event_ids()
        fit = {"coef_visit_recency": 0.0}
        with pytest.raises(ValueError, match="globally unique across sessions"):
            predict_arm_fractions(eat, fit, predictors=["visit_recency"])


class TestBuildChoiceDatasetPrecondition:
    def _stop_table(self) -> pd.DataFrame:
        # Same stop_id reused across two sessions.
        rows = []
        for animal, day in (("a", 20200101), ("b", 20200202)):
            rows.append(
                {
                    "animal": animal,
                    "day": day,
                    "epoch": "02_r1",
                    "stop_id": 1,
                    "has_replay": True,
                    "next_arm": 0,
                    "current_arm": 1,
                    "n_events": 1,
                    "reward_outcome": "rewarded",
                }
            )
        return pd.DataFrame(rows)

    def test_session_local_stop_ids_raise(self):
        eat = _two_session_event_arm_table_colliding_event_ids()
        et = pd.DataFrame()  # unused after early validation
        st = self._stop_table()
        with pytest.raises(ValueError, match="globally unique across sessions"):
            build_choice_dataset(et, eat, st)


class TestStopPatchReplayPrecondition:
    def test_session_local_event_ids_raise(self):
        eat = _two_session_event_arm_table_colliding_event_ids()
        et = pd.DataFrame(
            {
                "event_id": [1],
                "current_patch": [1],
            }
        )
        with pytest.raises(ValueError, match="globally unique across sessions"):
            stop_patch_replay(eat, et)


class TestBuildEventPatchTablePrecondition:
    def test_session_local_event_ids_raise(self):
        # Two sessions, both reusing event_id=1. Without the
        # precondition, the per-(event_id, patch) groupby silently
        # fuses across sessions and the merges to event_table
        # produce a Cartesian product (a tiny repro produced 48 rows
        # instead of 6).
        eat = _two_session_event_arm_table_colliding_event_ids()
        # Add an arm_count_hard column the patch table aggregates.
        eat["arm_count_hard"] = 1.0
        eat["is_same_patch"] = 0
        eat["visit_recency"] = 0.0
        eat["cumulative_reward"] = 0.0
        et = (
            eat[["animal", "day", "epoch", "event_id"]]
            .drop_duplicates()
            .assign(
                last_rewarded_arm=np.nan,
                prev_patch=np.nan,
                is_after_switch=0,
            )
        )
        with pytest.raises(ValueError, match="globally unique across sessions"):
            build_event_patch_table(et, eat)


class TestPreconditionPassesOnUniqueIds:
    """When IDs are globally unique, the validators must not fire — they
    are only there to catch the cross-session-collision case."""

    def _unique_id_event_arm_table(self) -> pd.DataFrame:
        # Two sessions, distinct event_ids and stop_ids.
        rows = []
        eid = 0
        for animal, day, eid_offset, sid_offset in (
            ("a", 20200101, 0, 0),
            ("b", 20200202, 100, 100),
        ):
            for arm in range(9):
                rows.append(
                    {
                        "animal": animal,
                        "day": day,
                        "epoch": "02_r1",
                        "event_id": 1 + eid_offset,
                        "stop_id": 1 + sid_offset,
                        "arm": arm,
                        "arm_fraction": 1.0 / 9,
                        "visit_recency": 0.0,
                        "cumulative_reward": 0.0,
                        "recent_reward_rate": 0.0,
                        "dist_to_arm": 0.0,
                        "is_same_patch": 0,
                        "current_patch": 1,
                    }
                )
            eid += 1
        return pd.DataFrame(rows)

    def test_predict_arm_fractions_accepts_unique_ids(self):
        eat = self._unique_id_event_arm_table()
        fit = {"coef_visit_recency": 0.0}
        out = predict_arm_fractions(eat, fit, predictors=["visit_recency"])
        # Each event's predicted_frac sums to 1 within its 9 arms.
        per_event_sum = out.groupby("event_id")["predicted_frac"].sum()
        np.testing.assert_allclose(per_event_sum.to_numpy(), 1.0, atol=1e-9)
