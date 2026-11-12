"""Tests for analysis/behavior/fit_comparability.py: which sessions of a rung pair compare."""

from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.behavior.fit_comparability import why_not_comparable


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
