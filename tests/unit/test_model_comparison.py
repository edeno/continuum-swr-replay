"""Tests for analysis/replay/model_comparison.py.

Covers the behavior the paper's statistical tables rely on:
- compare_nested_models: ΔAIC / ΔBIC / LR-stat arithmetic, mismatched-
  n_events invalidation, schema acceptance for both ll and llf columns
- summarize_comparison: empty input, NaN-drop before Wilcoxon, the <5
  sessions threshold, negative LR handling
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.replay.model_comparison import (
    compare_nested_models,
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


def _make_fit_results(*, ll, aic, bic, n_params, n_events=100, n_sessions=1, ll_col="llf"):
    """One row per session with the columns compare_nested_models needs."""
    return pd.DataFrame(
        {
            "animal": ["a"] * n_sessions,
            "day": list(range(n_sessions)),
            "epoch": ["02_r1"] * n_sessions,
            ll_col: [ll] * n_sessions,
            "aic": [aic] * n_sessions,
            "bic": [bic] * n_sessions,
            "n_params": [n_params] * n_sessions,
            "n_events": [n_events] * n_sessions,
        }
    )


# ── compare_nested_models ──


class TestCompareNestedModels:
    def test_delta_aic_and_lr_stat_arithmetic(self):
        """Parent AIC=100, child AIC=90 → delta_aic = 10 (child wins).
        Parent ll=-50, child ll=-44 → lr_stat = -2*(-50-(-44)) = 12.
        df_diff = 4-2 = 2."""
        parent = _make_fit_results(ll=-50, aic=100, bic=110, n_params=2)
        child = _make_fit_results(ll=-44, aic=90, bic=105, n_params=4)
        cmp = compare_nested_models(parent, child)
        row = cmp.iloc[0]
        assert row["delta_aic"] == 10.0
        assert row["delta_bic"] == 5.0
        assert row["lr_stat"] == 12.0
        assert row["df_diff"] == 2

    def test_negative_lr_stat_is_clipped_for_pvalue(self):
        """If the child fits WORSE than the parent (numerical
        convergence quirk; nested models can't have lower likelihood
        in theory), the raw lr_stat is negative. The chi2.sf call
        clips lower=0 so the resulting p-value is 1.0 (no evidence
        for the child)."""
        parent = _make_fit_results(ll=-44, aic=90, bic=105, n_params=2)
        child = _make_fit_results(ll=-50, aic=104, bic=119, n_params=4)
        cmp = compare_nested_models(parent, child)
        row = cmp.iloc[0]
        assert row["lr_stat"] < 0
        # Clipped to 0 inside chi2.sf → p = 1.0
        assert row["lr_pvalue"] == 1.0

    def test_mismatched_n_events_invalidates_metrics(self):
        """If parent and child were fit on different event counts
        (caller forgot to pre-filter to common complete-case events),
        delta_aic / delta_bic / lr_stat / lr_pvalue must be NaN —
        their values would be uninterpretable otherwise."""
        parent = _make_fit_results(ll=-50, aic=100, bic=110, n_params=2, n_events=100)
        child = _make_fit_results(ll=-44, aic=90, bic=105, n_params=4, n_events=80)
        cmp = compare_nested_models(parent, child)
        row = cmp.iloc[0]
        assert np.isnan(row["delta_aic"])
        assert np.isnan(row["delta_bic"])
        assert np.isnan(row["lr_stat"])
        assert np.isnan(row["lr_pvalue"])

    def test_matched_n_events_keeps_metrics_finite(self):
        parent = _make_fit_results(ll=-50, aic=100, bic=110, n_params=2, n_events=100)
        child = _make_fit_results(ll=-44, aic=90, bic=105, n_params=4, n_events=100)
        cmp = compare_nested_models(parent, child)
        assert cmp["delta_aic"].notna().all()

    def test_accepts_legacy_ll_column(self):
        """Old binomial path used ``ll`` instead of ``llf``."""
        parent = _make_fit_results(ll=-50, aic=100, bic=110, n_params=2, ll_col="ll")
        child = _make_fit_results(ll=-44, aic=90, bic=105, n_params=4, ll_col="ll")
        cmp = compare_nested_models(parent, child)
        assert cmp["lr_stat"].iloc[0] == 12.0

    def test_empty_inputs_return_empty_frame(self):
        parent = _make_fit_results(ll=-50, aic=100, bic=110, n_params=2, n_sessions=0)
        child = _make_fit_results(ll=-44, aic=90, bic=105, n_params=4, n_sessions=0)
        cmp = compare_nested_models(parent, child)
        assert len(cmp) == 0

    def test_sessions_present_in_only_one_table_are_dropped(self):
        """Inner-merge semantics: a session that fitted only the parent
        (or only the child) is silently dropped from the comparison."""
        parent = pd.DataFrame(
            {
                "animal": ["a", "a"],
                "day": [1, 2],
                "epoch": ["02_r1", "02_r1"],
                "llf": [-50, -55],
                "aic": [100, 110],
                "bic": [110, 120],
                "n_params": [2, 2],
                "n_events": [100, 100],
            }
        )
        child = pd.DataFrame(
            {
                "animal": ["a"],
                "day": [1],
                "epoch": ["02_r1"],
                "llf": [-44],
                "aic": [90],
                "bic": [105],
                "n_params": [4],
                "n_events": [100],
            }
        )
        cmp = compare_nested_models(parent, child)
        assert len(cmp) == 1
        assert cmp.iloc[0]["day"] == 1


# ── summarize_comparison ──


class TestSummarizeComparison:
    def test_empty_input_returns_zero_session_summary(self):
        result = summarize_comparison(pd.DataFrame(), "M0", "M1")
        assert result["n_sessions"] == 0
        assert result["parent"] == "M0"
        assert result["child"] == "M1"

    def test_drops_nan_before_wilcoxon(self):
        """scipy.stats.wilcoxon defaults to nan_policy='propagate' and
        returns NaN for any NaN in the input. summarize_comparison must
        drop NaN sessions before calling wilcoxon so the omnibus p-value
        is valid whenever at least five sessions are comparable."""
        comparisons = _make_comparison_df(
            [-1.0, -2.0, np.nan, -0.5, -1.5, -2.5]
        )
        result = summarize_comparison(comparisons, "M0", "M1")
        assert not np.isnan(result["wilcoxon_p"])

    def test_nan_when_fewer_than_five_valid_sessions(self):
        """wilcoxon needs >=5 sessions. After dropping NaN, if fewer
        than five remain, the result is NaN (not a crash)."""
        comparisons = _make_comparison_df([-1.0, np.nan, np.nan, -0.5])
        result = summarize_comparison(comparisons, "M0", "M1")
        assert np.isnan(result["wilcoxon_p"])

    def test_counts_child_better_aic_threshold(self):
        """child_better_aic counts sessions with delta_aic > 2."""
        comparisons = _make_comparison_df([5.0, 3.0, 1.0, -1.0, -5.0])
        result = summarize_comparison(comparisons, "M0", "M1")
        assert result["child_better_aic"] == 2  # 5.0 and 3.0 are > 2
        assert result["parent_better_aic"] == 1  # only -5.0 is < -2

    def test_lrt_significant_count_at_5_percent(self):
        """lrt_significant counts sessions with lr_pvalue < 0.05."""
        df = pd.DataFrame(
            {
                "animal": ["a"] * 5,
                "day": list(range(5)),
                "epoch": ["02_r1"] * 5,
                "delta_aic": [1.0] * 5,
                "delta_bic": [1.0] * 5,
                "lr_stat": [5.0] * 5,
                "lr_pvalue": [0.01, 0.02, 0.06, 0.10, 0.50],
                "df_diff": [1] * 5,
            }
        )
        result = summarize_comparison(df, "M0", "M1")
        assert result["lrt_significant"] == 2

    def test_n_sessions_counts_full_input_not_just_non_nan(self):
        """``n_sessions`` reports input length (unchanged from prior
        behavior). It is the wilcoxon test that uses the NaN-dropped
        subset, not this counter."""
        comparisons = _make_comparison_df(
            [-1.0, -2.0, np.nan, -0.5, -1.5, -2.5]
        )
        result = summarize_comparison(comparisons, "M0", "M1")
        assert result["n_sessions"] == 6
