"""The stay/leave role summaries read over strata, on tables small enough to check by hand."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay.learning import (
    fit_status_counts,
    pooled_gee_role_summary,
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
    # Three converged roles, the one without a p entering as 1:
    # 0.01 * 3 / 1, 0.04 * 3 / 2, 1 * 3 / 3. The failed fit's 0.001 is not used.
    assert list(summary["p_fdr"]) == pytest.approx([0.03, 0.06, 1.0])


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
