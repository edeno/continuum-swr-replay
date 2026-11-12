"""Tests for analysis/behavior/fit_comparability.py: which sessions of a rung pair compare."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.behavior.fit_comparability import (
    pair_rung_fits,
    why_not_comparable,
)


def test_why_not_comparable_names_the_columns_each_session_differs_on():
    paired = pd.DataFrame(
        {
            "n_bins_p": [10, 10, 10, 10, 10],
            "n_bins_c": [10, 11, 10, np.nan, 10],
            "time_clock_p": ["stop", "stop", "stop", "stop", None],
            "time_clock_c": ["stop", "arrival", "arrival", "stop", "arrival"],
            # carried by one rung only: not checked
            "n_stops_c": [1, 2, 3, 4, 5],
        },
        index=[3, 5, 7, 9, 11],
    )
    differs = why_not_comparable(paired, ("_p", "_c"))
    assert list(differs.columns) == ["n_bins", "time_clock"]
    assert differs.index.tolist() == [3, 5, 7, 9, 11]
    # a missing count differs; a clock one rung does not state does not
    assert differs["n_bins"].tolist() == [False, True, False, True, False]
    assert differs["time_clock"].tolist() == [False, True, True, False, False]
    assert (~differs.any(axis=1)).tolist() == [True, False, False, False, True]


def test_why_not_comparable_checks_nothing_when_no_column_is_carried_by_both():
    paired = pd.DataFrame({"aic_p": [1.0, 2.0], "aic_c": [0.0, 3.0]})
    differs = why_not_comparable(paired, ("_p", "_c"))
    assert differs.shape == (2, 0)
    assert not differs.any(axis=1).any()


def _rung(aic, **columns):
    return pd.DataFrame(
        {"animal": ["a", "a", "b"], "date": ["d1", "d2", "d1"], "aic": aic, **columns}
    )


def test_pair_rung_fits_pairs_the_sessions_both_rungs_fit_in_the_parent_s_order():
    parent = _rung([10.0, 20.0, 30.0], n_bins=[5, 5, 5])
    child = _rung([9.0, 25.0, 1.0], n_bins=[5, 6, 5]).iloc[[2, 0]]
    paired, differs = pair_rung_fits(parent, child, ["aic"], on=["animal", "date"])
    assert paired.columns.tolist() == [
        "animal",
        "date",
        "aic_parent",
        "n_bins_parent",
        "aic_child",
        "n_bins_child",
    ]
    assert paired[["animal", "date"]].values.tolist() == [["a", "d1"], ["b", "d1"]]
    assert (paired["aic_child"] - paired["aic_parent"]).tolist() == [-1.0, -29.0]
    assert differs.columns.tolist() == ["n_bins"]
    assert not differs.any(axis=1).any()


def test_pair_rung_fits_carries_optional_columns_both_hold_then_comparability_ones():
    parent = _rung([1.0, 2.0, 3.0], n_events=[4, 4, 4], estimator="x", note="p")
    child = _rung([1.0, 2.0, 3.0], n_events=[4, 3, 4], estimator="x")
    paired, differs = pair_rung_fits(
        parent, child, ["aic"], optional=["note", "estimator"], on=["animal", "date"]
    )
    # ``note`` is held by the parent only; ``n_events`` is carried unasked.
    assert [c for c in paired if c.endswith("_parent")] == [
        "aic_parent",
        "estimator_parent",
        "n_events_parent",
    ]
    assert differs.any(axis=1).tolist() == [False, True, False]


def test_pair_rung_fits_refuses_two_fits_of_one_session():
    parent = _rung([1.0, 2.0, 3.0])
    with pytest.raises(pd.errors.MergeError):
        pair_rung_fits(
            parent, pd.concat([parent, parent]), ["aic"], on=["animal", "date"]
        )
