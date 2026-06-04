"""Regression test for ``build_event_patch_table`` NaN preservation.

The patch-level ``cumulative_reward_patch`` aggregation must keep NaN
when no arms in the patch have a known cumulative reward. The default
``pandas.Series.sum()`` collapses an all-NaN group to ``0.0``
(``skipna=True`` by default), which would silently turn unknown
reward history into a real zero — keeping those events in the
complete-case fits for the P3 patch model that conditions on
``cumulative_reward_patch``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.replay.tables import build_event_patch_table


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
            "arm": arms,
            "arm_count_hard": np.zeros(n_arms, dtype=float),
            "is_same_patch": np.zeros(n_arms, dtype=int),
            "cumulative_reward": cumulative_reward_values,
            "visit_recency": np.full(n_arms, np.nan),
        }
    )


def _make_event_table():
    return pd.DataFrame(
        {
            "event_id": [1],
            "last_rewarded_arm": [np.nan],
            "prev_patch": [np.nan],
            "is_after_switch": [0],
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
        assert out["cumulative_reward_patch"].isna().all(), (
            f"expected all-NaN, got {out['cumulative_reward_patch'].tolist()}"
        )

    def test_partial_nan_patch_sums_known_arms(self):
        """A patch with a mix of known and unknown rewards should sum only
        the known values (default ``skipna=True`` semantics, preserved by
        ``min_count=1``)."""
        # arms 0,1,6 → patch 1: 0.5 + NaN + 0.25 = 0.75
        # arms 2,3,7 → patch 2: NaN + NaN + NaN  = NaN
        # arms 4,5,8 → patch 3: 1.0 + 2.0 + 4.0  = 7.0
        cum = [0.5, np.nan, np.nan, np.nan, 1.0, 2.0, 0.25, np.nan, 4.0]
        eat = _make_event_arm_table(cum)
        et = _make_event_table()
        out = build_event_patch_table(et, eat).set_index("patch")
        assert out.loc[1, "cumulative_reward_patch"] == pytest_approx(0.75)
        assert np.isnan(out.loc[2, "cumulative_reward_patch"])
        assert out.loc[3, "cumulative_reward_patch"] == pytest_approx(7.0)


def pytest_approx(value, tol=1e-9):
    """Local approx helper to avoid importing pytest at module top
    where it's not strictly needed (pytest is the runner anyway,
    but keeping the dependency surface minimal)."""
    import pytest

    return pytest.approx(value, abs=tol)
