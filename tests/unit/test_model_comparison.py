"""Tests for analysis/replay/model_comparison.py.

Pins the behavior the paper's statistical tables rely on: Wilcoxon
signed-rank on delta-AIC must not be silently invalidated when the
mismatched-n_events guard sets some sessions to NaN.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.replay.model_comparison import (
    summarize_comparison,
)


def _make_comparison_df(daic_values):
    """Minimal comparison_df with the columns summarize_comparison reads."""
    n = len(daic_values)
    return pd.DataFrame(
        {
            "animal": ["a"] * n,
            "day": list(range(n)),
            "epoch": ["02_r1"] * n,
            "delta_aic": daic_values,
            "delta_bic": daic_values,
            "lr_stat": [5.0] * n,
            "lr_pvalue": [0.01] * n,
            "df_diff": [1] * n,
        }
    )


def test_summarize_comparison_drops_nan_before_wilcoxon():
    """scipy.stats.wilcoxon defaults to nan_policy='propagate' and
    returns NaN for any NaN in the input. summarize_comparison must
    drop NaN sessions before calling wilcoxon so the omnibus p-value
    is valid whenever at least five sessions are comparable."""
    comparisons = _make_comparison_df(
        [-1.0, -2.0, np.nan, -0.5, -1.5, -2.5]
    )
    result = summarize_comparison(comparisons, "M0", "M1")
    assert not np.isnan(result["wilcoxon_p"]), (
        "wilcoxon_p was NaN — NaN sessions were not dropped before wilcoxon"
    )


def test_summarize_comparison_nan_when_fewer_than_five_valid_sessions():
    """Existing behavior: wilcoxon needs >=5 sessions. After dropping
    NaN, if fewer than five remain, the result is NaN (not crash)."""
    comparisons = _make_comparison_df([-1.0, np.nan, np.nan, -0.5])
    result = summarize_comparison(comparisons, "M0", "M1")
    assert np.isnan(result["wilcoxon_p"])
