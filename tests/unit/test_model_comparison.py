"""Tests for analysis/replay/model_comparison.py.

Covers the behavior the paper's statistical tables rely on:
- compare_nested_models: ΔAIC / ΔBIC / LR-stat arithmetic, mismatched-
  n_events invalidation
- summarize_comparison: empty input, NaN-drop before Wilcoxon, the <5
  sessions threshold, negative LR handling
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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
            "session_date": list(range(n)),
            "epoch": ["02_r1"] * n,
            "delta_aic": daic_values,
            "delta_bic": daic_values,
            "lr_stat": [5.0] * n,
            "lr_pvalue": [0.01] * n,
            "df_diff": [1] * n,
        }
    )


def _make_fit_results(*, ll, aic, bic, n_params, n_events=100, n_sessions=1):
    """One row per session with the columns compare_nested_models needs."""
    return pd.DataFrame(
        {
            "animal": ["a"] * n_sessions,
            "session_date": list(range(n_sessions)),
            "epoch": ["02_r1"] * n_sessions,
            "llf": [ll] * n_sessions,
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
                "session_date": [1, 2],
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
                "session_date": [1],
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
        assert cmp.iloc[0]["session_date"] == 1


# ── summarize_comparison ──


class TestSummarizeComparison:
    def test_empty_input_returns_zero_session_summary(self):
        result = summarize_comparison(pd.DataFrame(), "M0", "M1")
        assert result["n_sessions"] == 0
        assert result["parent"] == "M0"
        assert result["child"] == "M1"

    def test_drops_nan_before_wilcoxon(self):
        """scipy.stats.wilcoxon defaults to nan_policy='propagate' and
        returns NaN for any NaN in the input. A NaN session is left out, so
        the omnibus p-value is valid whenever six sessions are comparable:
        six of one sign give the exact p 2 / 2 ** 6."""
        comparisons = _make_comparison_df([-1.0, -2.0, np.nan, -0.5, -1.5, -2.5, -3.0])
        result = summarize_comparison(comparisons, "M0", "M1")
        assert result["wilcoxon_p"] == pytest.approx(2 / 2**6)

    def test_nan_when_fewer_than_six_valid_sessions(self):
        """Five sessions cannot reach p < 0.05 (their smallest p is 2 / 2 ** 5):
        after dropping NaN, if fewer than six remain, the result is NaN."""
        comparisons = _make_comparison_df([-1.0, np.nan, -0.5, -2.0, -1.5, -2.5])
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
                "session_date": list(range(5)),
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
        comparisons = _make_comparison_df([-1.0, -2.0, np.nan, -0.5, -1.5, -2.5])
        result = summarize_comparison(comparisons, "M0", "M1")
        assert result["n_sessions"] == 6


def test_mismatched_n_bins_invalidates():
    """Per-bin rungs fit on different bin counts are not comparable.

    AIC / BIC / deviance all scale with the number of rows, so a rung fit to
    fewer bins wins mechanically. ``n_bins`` (Family B) and ``n_stops``
    (Family A) are guarded exactly like ``n_events``.
    """

    def rows(*, aic, bic, ll, n_params, n_bins, n_stops):
        return pd.DataFrame(
            {
                "animal": ["a"],
                "session_date": [1],
                "epoch": ["02_r1"],
                "llf": [ll],
                "aic": [aic],
                "bic": [bic],
                "n_params": [n_params],
                "n_bins": [n_bins],
                "n_stops": [n_stops],
            }
        )

    parent = rows(aic=100, bic=110, ll=-50, n_params=2, n_bins=7387816, n_stops=1000)
    child = rows(aic=90, bic=105, ll=-44, n_params=4, n_bins=6466795, n_stops=1000)
    invalid = compare_nested_models(parent, child).iloc[0]
    for col in ("delta_aic", "delta_bic", "lr_stat", "lr_pvalue"):
        assert np.isnan(invalid[col]), f"{col} must be NaN on mismatched n_bins"

    # A stop-count mismatch alone (Family A) invalidates too.
    child_stops = rows(aic=90, bic=105, ll=-44, n_params=4, n_bins=7387816, n_stops=940)
    assert np.isnan(compare_nested_models(parent, child_stops).iloc[0]["delta_aic"])

    # Matched bins and stops leave the arithmetic intact.
    child_ok = rows(aic=90, bic=105, ll=-44, n_params=4, n_bins=7387816, n_stops=1000)
    valid = compare_nested_models(parent, child_ok).iloc[0]
    assert valid["delta_aic"] == 10.0
    assert valid["delta_bic"] == 5.0
    assert valid["lr_stat"] == 12.0


# ── Fit-scale metadata ──


class TestEstimatorTravelsWithTheComparison:
    """A delta-AIC is only readable next to the scale it was computed at.

    On the fraction response an event is one observation; on the counts it is
    ~32 autocorrelated timebins. The summary therefore carries the estimator and
    the response column, and refuses a step whose two rungs disagree on either.
    """

    @staticmethod
    def _rungs(estimator="conditional_poisson_fraction", response="arm_fraction"):
        parent = _make_fit_results(
            ll=-10.0, aic=24.0, bic=26.0, n_params=2, n_sessions=6
        )
        child = _make_fit_results(ll=-7.0, aic=20.0, bic=23.0, n_params=3, n_sessions=6)
        for frame in (parent, child):
            frame["estimator"] = estimator
            frame["response_col"] = response
        return parent, child

    def test_summary_reports_the_estimator_and_response(self):
        parent, child = self._rungs()

        result = summarize_comparison(compare_nested_models(parent, child), "M0", "M1")

        assert result["estimator"] == "conditional_poisson_fraction"
        assert result["response_col"] == "arm_fraction"
        assert result["delta_aic_median"] == 4.0

    def test_summary_of_a_count_ladder_reports_the_count_estimator(self):
        parent, child = self._rungs("conditional_poisson_count", "arm_count_hard")

        result = summarize_comparison(compare_nested_models(parent, child), "M0", "M1")

        assert result["estimator"] == "conditional_poisson_count"
        assert result["response_col"] == "arm_count_hard"

    def test_rungs_fit_at_different_scales_are_refused(self):
        parent, child = self._rungs()
        child["estimator"] = "conditional_poisson_count"
        child["response_col"] = "arm_count_hard"
        comparison = compare_nested_models(parent, child)

        with pytest.raises(ValueError, match="different estimator values"):
            summarize_comparison(comparison, "M0", "M1")

    def test_legacy_frames_without_the_metadata_report_none(self):
        parent = _make_fit_results(
            ll=-10.0, aic=24.0, bic=26.0, n_params=2, n_sessions=6
        )
        child = _make_fit_results(ll=-7.0, aic=20.0, bic=23.0, n_params=3, n_sessions=6)

        result = summarize_comparison(compare_nested_models(parent, child), "M0", "M1")

        assert result["estimator"] is None
        assert result["response_col"] is None

    def test_empty_comparison_still_carries_the_metadata_keys(self):
        result = summarize_comparison(pd.DataFrame(), "M0", "M1")

        assert result["estimator"] is None
        assert result["response_col"] is None


# ── likelihood constants ──


def _fit_with_constants(*, aic, bic, ll, includes_constants, estimator):
    frame = _make_fit_results(ll=ll, aic=aic, bic=bic, n_params=2)
    frame["llf_includes_constants"] = includes_constants
    frame["estimator"] = estimator
    frame["response_col"] = "arm_count_hard"
    return frame


def test_mismatched_likelihood_constants_invalidate_the_comparison():
    """A full likelihood minus a conditional one measures the constant, not the rung.

    The Dirichlet-multinomial keeps the multinomial coefficient the statsmodels
    conditional likelihood drops, so differencing their AICs would report that
    coefficient as a model improvement.
    """
    parent = _fit_with_constants(
        aic=100,
        bic=110,
        ll=-50,
        includes_constants=False,
        estimator="conditional_poisson_count",
    )
    child = _fit_with_constants(
        aic=90,
        bic=105,
        ll=-44,
        includes_constants=True,
        estimator="dirichlet_multinomial",
    )

    row = compare_nested_models(parent, child).iloc[0]

    assert np.isnan(row["delta_aic"])
    assert np.isnan(row["delta_bic"])
    assert np.isnan(row["lr_stat"])
    assert np.isnan(row["lr_pvalue"])
    assert bool(row["llf_includes_constants_parent"]) is False
    assert bool(row["llf_includes_constants_child"]) is True


def test_agreeing_likelihood_constants_leave_the_comparison_alone():
    parent = _fit_with_constants(
        aic=100,
        bic=110,
        ll=-50,
        includes_constants=True,
        estimator="dirichlet_multinomial",
    )
    child = _fit_with_constants(
        aic=90,
        bic=105,
        ll=-44,
        includes_constants=True,
        estimator="dirichlet_multinomial",
    )

    comparison = compare_nested_models(parent, child)

    assert comparison["delta_aic"].iloc[0] == 10.0
    assert comparison["lr_stat"].iloc[0] == 12.0
    summary = summarize_comparison(comparison, "M0", "M1")
    assert bool(summary["llf_includes_constants"]) is True
    assert summary["estimator"] == "dirichlet_multinomial"


def test_a_rung_that_records_nothing_is_unknown_not_different():
    """Legacy artifacts predate the flag; absence must not NaN their comparisons."""
    parent = _make_fit_results(ll=-50, aic=100, bic=110, n_params=2)
    child = _make_fit_results(ll=-44, aic=90, bic=105, n_params=4)

    comparison = compare_nested_models(parent, child)

    assert comparison["delta_aic"].iloc[0] == 10.0
    assert (
        summarize_comparison(comparison, "M0", "M1")["llf_includes_constants"] is None
    )


def test_constants_alone_invalidate_a_pair_that_agrees_on_everything_else():
    """The flag is guarded on its own, not as a proxy for the estimator name.

    Two rungs can report the same estimator and the same response and still
    differ here -- a fit whose likelihood was rebased to include the
    multinomial coefficient against one that was not -- and the ΔAIC would
    then be that constant.
    """
    parent = _fit_with_constants(
        aic=100,
        bic=110,
        ll=-50,
        includes_constants=False,
        estimator="dirichlet_multinomial",
    )
    child = _fit_with_constants(
        aic=90,
        bic=105,
        ll=-44,
        includes_constants=True,
        estimator="dirichlet_multinomial",
    )

    comparison = compare_nested_models(parent, child)

    assert (
        comparison["estimator_parent"].iloc[0] == comparison["estimator_child"].iloc[0]
    )
    assert np.isnan(comparison["delta_aic"].iloc[0])
    assert np.isnan(comparison["lr_stat"].iloc[0])
    # And the summary refuses the step outright: one value cannot be reported.
    with pytest.raises(ValueError, match="llf_includes_constants"):
        summarize_comparison(comparison, "M0", "M1")


def _per_bin_fit(*, aic, bic, ll, n_bins, time_clock):
    """One per-bin rung's per-session row, tagged with the clock it was fit on."""
    frame = _make_fit_results(ll=ll, aic=aic, bic=bic, n_params=7)
    frame = frame.drop(columns=["n_events"])
    frame["n_bins"] = n_bins
    frame["time_clock"] = time_clock
    return frame


def test_cross_clock_comparison_is_invalidated_and_says_which_clocks(caplog):
    """An arrival rung is fit on the arrival window, so it is not the stop ladder's child.

    The row-count guard would also catch this (the window is a strict subset of
    the frozen cohort), but the clock is what makes the reason legible, so it
    travels with the comparison and invalidates it on its own.
    """
    parent = _per_bin_fit(aic=100, bic=110, ll=-50, n_bins=1000, time_clock="stop")
    child = _per_bin_fit(aic=90, bic=105, ll=-44, n_bins=900, time_clock="arrival")

    with caplog.at_level("WARNING"):
        row = compare_nested_models(parent, child).iloc[0]

    assert np.isnan(row["delta_aic"])
    assert np.isnan(row["delta_bic"])
    assert np.isnan(row["lr_stat"])
    assert np.isnan(row["lr_pvalue"])
    assert row["time_clock_parent"] == "stop"
    assert row["time_clock_child"] == "arrival"
    assert "different time clocks" in caplog.text


def test_same_clock_comparison_is_untouched():
    parent = _per_bin_fit(aic=100, bic=110, ll=-50, n_bins=900, time_clock="arrival")
    child = _per_bin_fit(aic=90, bic=105, ll=-44, n_bins=900, time_clock="arrival")

    row = compare_nested_models(parent, child).iloc[0]

    assert row["delta_aic"] == 10
    assert np.isfinite(row["lr_stat"])


def _cohort_fit(*, aic, bic, ll, n_stops, stop_cohort):
    """One rate rung's per-session row, tagged with the stop population it was fit on."""
    frame = _make_fit_results(ll=ll, aic=aic, bic=bic, n_params=7)
    frame = frame.drop(columns=["n_events"])
    frame["n_stops"] = n_stops
    frame["stop_cohort"] = stop_cohort
    return frame


def test_cross_cohort_comparison_is_invalidated_and_says_why(caplog):
    """An at-well fit and an all-stops fit are about different populations.

    Their row counts usually differ too, but a comparison built from equal-sized
    subsets would slip past that guard; the cohort says WHY the step is refused.
    """
    parent = _cohort_fit(aic=100, bic=110, ll=-50, n_stops=900, stop_cohort="at_well")
    child = _cohort_fit(aic=90, bic=105, ll=-44, n_stops=900, stop_cohort="all_stops")

    with caplog.at_level("WARNING"):
        row = compare_nested_models(parent, child).iloc[0]

    assert np.isnan(row["delta_aic"])
    assert np.isnan(row["delta_bic"])
    assert np.isnan(row["lr_stat"])
    assert np.isnan(row["lr_pvalue"])
    assert row["stop_cohort_parent"] == "at_well"
    assert row["stop_cohort_child"] == "all_stops"
    assert "different stop populations" in caplog.text


def test_same_cohort_comparison_is_untouched():
    parent = _cohort_fit(aic=100, bic=110, ll=-50, n_stops=900, stop_cohort="at_well")
    child = _cohort_fit(aic=90, bic=105, ll=-44, n_stops=900, stop_cohort="at_well")

    row = compare_nested_models(parent, child).iloc[0]

    assert row["delta_aic"] == 10
    assert np.isfinite(row["lr_stat"])
