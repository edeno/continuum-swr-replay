"""The stay/leave role summaries read over strata, on tables small enough to check by hand."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.behavior.choice.learning import (
    fit_status_counts,
    fit_stay_leave_roles,
    pooled_gee_role_summary,
    role_content_column,
    smallest_adjusted_p,
)

NAN = np.nan


def test_the_pooled_gee_summary_corrects_across_the_converged_roles():
    pooled = pd.DataFrame(
        {
            "role": ["current", "prev", "best", "last_rewarded"],
            "status": ["ok", "ok", "fit_error_ValueError", "ok"],
            "p_inter": [0.01, 0.04, 0.001, NAN],
        }
    )
    summary = pooled_gee_role_summary(pooled)
    assert list(summary["role"]) == ["current", "prev", "last_rewarded"]
    # Three converged roles; the one without a p is not a test, so the family
    # is two: 0.01 * 2 / 1, 0.04 * 2 / 2. The failed fit's 0.001 is not used.
    np.testing.assert_allclose(summary["p_fdr"], [0.02, 0.04, NAN])


def test_the_pooled_gee_summary_of_a_stratum_where_no_fit_converged_is_empty():
    pooled = pd.DataFrame(
        {
            "role": ["current", "prev"],
            "status": ["too_few_stops", "fit_error_ValueError"],
            "p_inter": [NAN, 0.001],
        }
    )
    summary = pooled_gee_role_summary(pooled)
    assert list(summary.columns) == ["role", "p_fdr"]
    assert len(summary) == 0
    # such a stratum is no candidate for the smallest adjusted p
    other = pd.DataFrame({"role": ["best"], "p_fdr": [0.4]})
    assert smallest_adjusted_p({"mua": summary, "swr": other}) == {
        "p_fdr": 0.4,
        "stratum": "swr",
        "role": "best",
    }


def test_the_smallest_adjusted_p_needs_one_finite_p():
    summaries = {
        "swr": pd.DataFrame({"role": ["a", "b"], "p_fdr": [NAN, NAN]}),
        "mua": pd.DataFrame({"role": [], "p_fdr": []}),
    }
    with pytest.raises(ValueError, match="no role of any stratum has a finite"):
        smallest_adjusted_p(summaries)


def test_the_smallest_adjusted_p_names_its_stratum_and_role():
    summaries = {
        # an untestable role has no adjusted p and is not a candidate
        "swr": pd.DataFrame({"role": ["a", "b"], "p_fdr": [NAN, 0.2]}),
        "mua": pd.DataFrame({"role": ["c", "d"], "p_fdr": [0.05, 0.9]}),
        "all": pd.DataFrame({"role": ["b", "z"], "p_fdr": [0.3, 0.05]}),
    }
    # 0.05 twice: the stratum first in alphabetical order has it
    assert smallest_adjusted_p(summaries) == {
        "p_fdr": 0.05,
        "stratum": "all",
        "role": "z",
    }
    summaries["swr"].loc[0, "p_fdr"] = 0.01
    assert smallest_adjusted_p(summaries) == {
        "p_fdr": 0.01,
        "stratum": "swr",
        "role": "a",
    }


def test_fit_status_counts_the_rows_and_the_converged_fits():
    # a session skipped before fitting (too few stops) has a row too
    fits = pd.DataFrame(
        {"status": ["ok", "ok", "too_few_stops_3", "fit_error", "ok", "not_converged"]}
    )
    assert fit_status_counts(fits) == {"n_fits": 6, "n_converged": 3}


def test_a_role_s_content_column_names_its_stratum():
    assert role_content_column("current", "all") == "frac_on_current"
    assert role_content_column("best_patch", "non_swr") == "frac_on_best_patch_non_swr"


def test_the_role_fits_read_each_role_s_own_stops_and_skip_a_missing_role():
    """Two roles carry content; ``current`` has it on 3 of 4 stops, ``prev`` on 2.

    ``min_stops`` is above both counts, so no session is fit, but each role's
    row says how many stops it read, and every other role is skipped.
    """
    summary = pd.DataFrame(
        {
            "animal": "a",
            "session_date": 1,
            "epoch": "02_r1",
            "stop_id": [0, 1, 2, 3],
            "is_stay": [1.0, 0.0, 1.0, 0.0],
            "is_omitted_outcome": [0.0, 1.0, 1.0, 0.0],
            "frac_on_current": [0.5, NAN, 0.2, 0.1],
            "frac_on_prev": [NAN, 0.4, NAN, 0.3],
        }
    )

    per_session, pooled = fit_stay_leave_roles(
        summary, content_stratum="all", stop_cohort="at_well", min_stops=10
    )

    assert list(per_session["role"]) == ["current", "prev"]
    assert list(per_session["n_stops"]) == [3, 2]
    assert per_session["beta_inter"].isna().all()
    assert set(per_session["stop_cohort"]) == {"at_well"}
    assert list(per_session["content_col"]) == ["frac_on_current", "frac_on_prev"]
    assert list(pooled["role"]) == ["current", "prev"]
    assert set(pooled["content_stratum"]) == {"all"}


def test_no_role_column_fits_nothing():
    per_session, pooled = fit_stay_leave_roles(
        pd.DataFrame({"animal": ["a"], "session_date": [1], "epoch": ["e"]}),
        content_stratum="swr",
        stop_cohort="at_well",
    )
    assert per_session.empty
    assert pooled.empty
