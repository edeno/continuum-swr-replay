"""Regression test for ``build_event_patch_table`` NaN preservation.

The patch-level ``reward_count_candidate_arm_last10_known_trials_patch`` aggregation must keep NaN
when no arms in the patch have a known cumulative reward. The default
``pandas.Series.sum()`` collapses an all-NaN group to ``0.0``
(``skipna=True`` by default), which would silently turn unknown
reward history into a real zero — keeping those events in the
complete-case fits for the P3 patch model that conditions on
``reward_count_candidate_arm_last10_known_trials_patch``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay.tables_arm_patch import (
    build_event_patch_table,
)


def _make_event_arm_table(cumulative_reward_values):
    """Build a 9-arm × 1-event arm table covering all three patches.

    Patch mapping (from ``data_loaders.constants.TRACK_SEGMENT_TO_PATCH``):
    arms 0,1,6 → patch 1; arms 2,3,7 → patch 2; arms 4,5,8 → patch 3.
    """
    arms = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    n_arms = len(arms)
    return pd.DataFrame(
        {
            "event_id": [1] * n_arms,
            "candidate_arm": arms,
            "arm_count_hard": np.zeros(n_arms, dtype=float),
            "is_same_patch_as_current": np.zeros(n_arms, dtype=int),
            "reward_count_candidate_arm_last10_known_trials": cumulative_reward_values,
            "visit_recency_trials": np.full(n_arms, np.nan),
            "is_candidate_arm_never_visited": np.full(n_arms, np.nan),
        }
    )


def _make_event_table():
    return pd.DataFrame(
        {
            "animal": ["test"],
            "session_date": [20200101],
            "epoch": ["01_r1"],
            "event_id": [1],
            "n_nonlocal_bins": [4],
            "n_undecoded_bins": [4],
            "last_rewarded_arm": [np.nan],
            "prev_patch": [np.nan],
            "did_last_completed_trip_switch_patch": [0],
            "is_swr": [False],
        }
    )


class TestEventPatchTableCumulativeRewardNaN:
    def test_all_nan_patch_stays_nan(self):
        """If every arm in a patch has NaN cumulative_reward, the patch sum
        must remain NaN — not collapse to 0.0."""
        eat = _make_event_arm_table([np.nan] * 9)
        et = _make_event_table()
        out = build_event_patch_table(et, eat)
        # All three patches inherit all-NaN inputs → all NaN outputs.
        assert (
            out["reward_count_candidate_arm_last10_known_trials_patch"].isna().all()
        ), (
            "expected all-NaN, got "
            f"{out['reward_count_candidate_arm_last10_known_trials_patch'].tolist()}"
        )

    def test_partial_nan_patch_is_rejected(self):
        """Partial leaf support is corruption, not a patch-level estimand.

        Canonical history is either known for every leaf or unavailable for
        the event.  Silently summing a subset would turn missing inputs into a
        plausible patch count and make the result depend on which row vanished.
        """
        cum = [0.5, np.nan, np.nan, np.nan, 1.0, 2.0, 0.25, np.nan, 4.0]
        eat = _make_event_arm_table(cum)
        et = _make_event_table()
        with pytest.raises(ValueError, match="partial leaf support"):
            build_event_patch_table(et, eat)


def test_patch_recency_and_novelty_use_leaf_arms_only():
    """A structural trunk zero must not overwrite leaf recency/novelty."""
    eat = _make_event_arm_table(np.zeros(9))
    # Patch arms: (0,1,6), (2,3,7), (4,5,8). Trunk values are structural 0.
    eat["visit_recency_trials"] = [5, 7, 2, 9, 4, 6, 0, 0, 0]
    eat["is_candidate_arm_never_visited"] = [1, 1, 0, 1, 0, 0, 0, 0, 0]

    out = build_event_patch_table(_make_event_table(), eat).set_index("patch")

    np.testing.assert_array_equal(
        out.loc[[1, 2, 3], "visit_recency_trials_patch"].to_numpy(), [5, 2, 4]
    )
    np.testing.assert_array_equal(
        out.loc[[1, 2, 3], "is_patch_never_visited"].to_numpy(), [1, 0, 0]
    )


def test_missing_event_swr_label_is_not_fabricated():
    event_table = _make_event_table().drop(columns="is_swr")
    event_arm_table = _make_event_arm_table(np.zeros(9))
    with pytest.raises(ValueError, match="refusing to fabricate non-SWR labels"):
        build_event_patch_table(event_table, event_arm_table)
