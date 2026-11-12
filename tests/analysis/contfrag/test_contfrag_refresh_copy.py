"""The ContFrag covariate refresh copies posterior aggregates by key, not by row.

``_copy_by_key`` moves the posterior-derived columns of a preserved bundle onto
rebuilt rows. The rebuilt rows need not come in the bundle's order, so a
positional copy would give an event another event's posterior.
"""

from __future__ import annotations

import pandas as pd
import pytest

from continuum_swr_replay.analysis.contfrag.prepare_data import _copy_by_key


def _source() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SWR_event": [1, 1, 2, 2],
            "candidate_arm": [0, 1, 0, 1],
            "arm_posterior_mass": [0.25, 0.75, 0.6, 0.4],
            "n_timebins": [99, 99, 99, 99],
        }
    )


def test_values_follow_their_key_whatever_the_row_order():
    target = pd.DataFrame(
        {"SWR_event": [2, 1, 2, 1], "candidate_arm": [1, 0, 0, 1], "n_timebins": 4}
    )

    copied = _copy_by_key(
        target,
        _source(),
        keys=["SWR_event", "candidate_arm"],
        columns=("arm_posterior_mass",),
    )

    assert copied["arm_posterior_mass"].tolist() == [0.4, 0.25, 0.6, 0.75]
    # only the named columns move; the target's own columns stay
    assert copied["n_timebins"].tolist() == [4, 4, 4, 4]
    assert "arm_posterior_mass" not in target


@pytest.mark.parametrize(
    ("target_keys", "match"),
    [
        ([(1, 0), (1, 1), (2, 0)], "missing=0, extra=1"),
        ([(1, 0), (1, 1), (2, 0), (2, 1), (3, 0)], "missing=1, extra=0"),
        ([(1, 0), (1, 0), (2, 0), (2, 1)], "unique keys"),
    ],
    ids=["source-has-more", "target-has-more", "duplicate-key"],
)
def test_the_key_sets_must_match_exactly(target_keys, match):
    target = pd.DataFrame(target_keys, columns=["SWR_event", "candidate_arm"])

    with pytest.raises(ValueError, match=match):
        _copy_by_key(
            target,
            _source(),
            keys=["SWR_event", "candidate_arm"],
            columns=("arm_posterior_mass",),
        )


def test_a_column_the_source_lacks_is_refused():
    target = _source()[["SWR_event", "candidate_arm"]]

    with pytest.raises(ValueError, match="multimodal_fraction"):
        _copy_by_key(
            target,
            _source(),
            keys=["SWR_event", "candidate_arm"],
            columns=("multimodal_fraction",),
        )
