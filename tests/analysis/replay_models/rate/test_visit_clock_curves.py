"""Model-free rates on the visit clocks and the stop-end counts, by hand."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay_models.rate.visit_clock_curves import (
    stop_end_kind_counts,
    visit_clock_rate_curves,
)


def _bins() -> pd.DataFrame:
    """Six design bins of two animals and one kind; clock bins [0, 1), [1, 2)."""
    return pd.DataFrame(
        {
            "kind": ["SWR"] * 6,
            "animal": ["a", "a", "a", "a", "b", "b"],
            "time_since_visit_start_s": [0.2, 0.9, 1.0, 2.5, 1.5, np.nan],
            "exposure": [0.5, 0.25, 0.5, 1.0, 0.2, 0.3],
            "count": [1.0, 0.0, 2.0, 1.0, 1.0, 1.0],
        }
    )


def test_rates_per_clock_bin() -> None:
    curves = visit_clock_rate_curves(
        _bins(), clock="time_since_visit_start_s", edges=np.array([0.0, 1.0, 2.0])
    )
    assert list(curves.columns) == [
        "kind",
        "animal",
        "clock",
        "clock_bin_start_s",
        "clock_bin_end_s",
        "n_starts",
        "at_risk_s",
        "rate",
    ]
    a = curves[curves["animal"] == "a"]
    # [0, 1): 1 start in 0.75 s; [1, 2): the bin at 1.0 s, 2 starts in 0.5 s;
    # the bin at 2.5 s lies past the last edge
    assert a["n_starts"].tolist() == [1, 2]
    np.testing.assert_allclose(a["at_risk_s"], [0.75, 0.5])
    np.testing.assert_allclose(a["rate"], [1 / 0.75, 4.0])
    # animal b: an empty first clock bin (no at-risk time: NaN rate) and a bin
    # with a missing clock left out
    b = curves[curves["animal"] == "b"]
    assert b["n_starts"].tolist() == [0, 1]
    np.testing.assert_allclose(b["at_risk_s"], [0.0, 0.2])
    assert np.isnan(b["rate"].iloc[0])
    assert b["rate"].iloc[1] == pytest.approx(5.0)
    assert (curves["clock"] == "time_since_visit_start_s").all()
    assert curves["clock_bin_start_s"].tolist() == [0.0, 1.0, 0.0, 1.0]


def test_edges_must_ascend() -> None:
    with pytest.raises(ValueError, match="ascending"):
        visit_clock_rate_curves(
            _bins(), clock="time_since_visit_start_s", edges=np.array([1.0, 0.0])
        )


def test_stop_end_kind_counts() -> None:
    stops = pd.DataFrame(
        {
            "animal": ["a", "a", "a", "a", "b"],
            "stop_end_kind": [
                "leaves",
                "small_movement",
                "small_movement",
                "outside_visit",
                "leaves",
            ],
        }
    )
    counts = stop_end_kind_counts(stops).set_index(["animal", "stop_end_kind"])
    assert counts["n_stops"].to_dict() == {
        ("a", "leaves"): 1,
        ("a", "outside_visit"): 1,
        ("a", "small_movement"): 2,
        ("b", "leaves"): 1,
    }
    assert counts.loc[("a", "small_movement"), "share"] == pytest.approx(0.5)
    assert counts.loc[("b", "leaves"), "share"] == pytest.approx(1.0)
