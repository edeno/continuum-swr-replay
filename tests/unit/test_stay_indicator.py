"""One definition of ``is_stay``, shared by the Replay and ContFrag builders.

``is_stay`` was written out five times -- twice per event as a scalar over
arms, three times per stop as a vectorized comparison over patches -- and the
five copies had to agree on two things that are easy to get wrong: that the
comparison is between *patches* (two arms of the same patch are a stay), and
that an unknown side propagates ``NaN`` rather than reading as a switch.
These tests pin both, and check the two builders still land on the same answer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay._helpers import (
    arm_to_patch,
    stay_indicator,
    stay_indicator_from_patches,
)
from continuum_swr_replay.data_loaders.constants import TRACK_SEGMENT_TO_PATCH


def test_same_patch_is_a_stay_even_across_different_arms():
    """Arms 0 and 1 are both patch 1: leaving one for the other is a stay."""
    assert TRACK_SEGMENT_TO_PATCH[0] == TRACK_SEGMENT_TO_PATCH[1] == 1
    assert stay_indicator(1, 0) == 1.0
    assert stay_indicator(0, 0) == 1.0


def test_different_patch_is_a_switch():
    assert TRACK_SEGMENT_TO_PATCH[2] == 2
    assert stay_indicator(2, 0) == 0.0
    assert stay_indicator(4, 0) == 0.0


@pytest.mark.parametrize(
    ("next_arm", "current_arm"),
    [(np.nan, 0), (0, np.nan), (np.nan, np.nan), (None, 0), (0, None)],
)
def test_unknown_arm_propagates_nan(next_arm, current_arm):
    """An unobserved side is not evidence of a switch."""
    assert np.isnan(stay_indicator(next_arm, current_arm))


def test_unknown_patch_propagates_nan():
    assert np.isnan(stay_indicator_from_patches(np.nan, 1.0))
    assert np.isnan(stay_indicator_from_patches(1.0, np.nan))


def test_scalar_returns_a_python_float_not_a_zero_d_array():
    out = stay_indicator(0, 0)
    assert isinstance(out, float)
    assert not isinstance(out, np.ndarray)


def test_vectorized_matches_the_scalar_elementwise():
    next_arm = np.array([1.0, 2.0, np.nan, 0.0, 8.0])
    current_arm = np.array([0.0, 0.0, 0.0, np.nan, 5.0])
    vectorized = stay_indicator_from_patches(
        np.array([arm_to_patch(a) for a in next_arm]),
        np.array([arm_to_patch(a) for a in current_arm]),
    )
    scalar = np.array(
        [stay_indicator(n, c) for n, c in zip(next_arm, current_arm, strict=True)]
    )
    np.testing.assert_array_equal(vectorized, scalar)


def test_empty_input_returns_an_empty_float_array():
    out = stay_indicator_from_patches(np.array([]), np.array([]))
    assert isinstance(out, np.ndarray)
    assert out.shape == (0,)
    assert out.dtype == float


def _build_frames(next_arm, current_arm):
    """The two shapes the builders hand the indicator, from one arm fixture.

    ``per_bout`` is the stop-grain path (patch arrays, computed once for the
    ``current_patch`` / ``next_patch`` columns and reused); ``per_event`` is
    the event-grain path (one scalar per record, straight from the arms).
    """
    cur_patch = np.array([arm_to_patch(a) for a in current_arm])
    nxt_patch = np.array([arm_to_patch(a) for a in next_arm])
    per_bout = stay_indicator_from_patches(nxt_patch, cur_patch)
    per_event = np.array(
        [stay_indicator(n, c) for n, c in zip(next_arm, current_arm, strict=True)]
    )
    return per_bout, per_event


def test_replay_and_contfrag_grains_agree_on_one_fixture():
    """Stop-grain and event-grain paths give identical ``is_stay`` values.

    ContFrag and Replay each build both grains; the four call sites reduce to
    these two shapes, so agreeing here is agreeing everywhere.
    """
    fixture = pd.DataFrame(
        {
            # leaf/leaf same patch, leaf/leaf other patch, trunk of the same
            # patch, an off-well (NaN) current arm, an unknown next arm.
            "current_arm": [0.0, 0.0, 0.0, np.nan, 3.0],
            "next_arm": [1.0, 2.0, 6.0, 1.0, np.nan],
        }
    )
    per_bout, per_event = _build_frames(
        fixture["next_arm"].to_numpy(), fixture["current_arm"].to_numpy()
    )
    np.testing.assert_array_equal(per_bout, per_event)
    np.testing.assert_array_equal(per_bout, np.array([1.0, 0.0, 1.0, np.nan, np.nan]))
