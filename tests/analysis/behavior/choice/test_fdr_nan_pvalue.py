"""Regression test: cross-session FDR must not be poisoned
by a NaN p-value from a single role.

``learning.cross_session_role_summary`` runs one signed-rank test per role and
corrects across roles. A role with no p (an all-zero β_inter sample, or too few
sessions) has ``p = nan``. ``multipletests`` returns an all-NaN adjusted-p
vector when *any* input is NaN, so a single untested role would contaminate
every other role's reported FDR.

The correction leaves the NaN out of the family and passes it through. These
tests drive the shipped function, not a copy of it, so the regression cannot
come back through an edit the test does not see.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests

from continuum_swr_replay.analysis.behavior.choice.learning import (
    ROLES,
    cross_session_role_summary,
)

#: Enough sessions for a role to be testable at all.
N_SESSIONS = 8


def _coefs(per_role_betas: dict[str, np.ndarray]) -> pd.DataFrame:
    """Per-session interaction coefficients for the named roles."""
    rows = []
    for role, betas in per_role_betas.items():
        for session, beta in enumerate(betas):
            rows.append(
                {
                    "animal": "a",
                    "session_date": 1,
                    "epoch": f"{session:02d}_r1",
                    "role": role,
                    "beta_inter": beta,
                    "status": "ok",
                }
            )
    return pd.DataFrame(rows)


def _betas(shift: float) -> np.ndarray:
    """A sample whose Wilcoxon p-value falls with ``shift``."""
    return np.linspace(0.1, 1.0, N_SESSIONS) + shift


class TestStage6FdrNanPropagation:
    def test_nan_pvalue_does_not_poison_other_roles(self):
        # An all-zero sample has no non-zero difference to rank: no p-value,
        # and it must not erase the other roles'.
        coefs = _coefs(
            {
                "current": _betas(0.0),
                "prev": np.zeros(N_SESSIONS),
                "last_rewarded": _betas(0.5),
            }
        )
        summary = cross_session_role_summary(coefs).set_index("role")

        assert np.isnan(summary.loc["prev", "p"])
        assert np.isnan(summary.loc["prev", "p_fdr"])
        # The roles that were testable keep real adjusted p-values.
        assert np.isfinite(summary.loc["current", "p_fdr"])
        assert np.isfinite(summary.loc["last_rewarded", "p_fdr"])
        # Sanity: the raw call this function replaced still NaN-poisons, so
        # the regression is still being exercised.
        poisoned = multipletests(
            [summary.loc["current", "p"], np.nan, summary.loc["last_rewarded", "p"]],
            method="fdr_bh",
        )
        assert np.all(np.isnan(poisoned[1]))

    def test_correction_is_bh_over_the_testable_roles_only(self):
        coefs = _coefs(
            {
                "current": _betas(0.0),
                "prev": _betas(0.5),
                # Three sessions are below the six-session floor, so this role
                # is neither tested nor part of the correction's denominator.
                "best": _betas(0.0)[:3],
                # Eight sessions, none with a non-zero difference: untested too.
                "last_rewarded": np.zeros(N_SESSIONS),
            }
        )
        summary = cross_session_role_summary(coefs).set_index("role")

        for untested, n_sessions in [("best", 3), ("last_rewarded", N_SESSIONS)]:
            assert summary.loc[untested, "status"] == "too_few_sessions"
            assert summary.loc[untested, "n_sessions"] == n_sessions
            # An untested role has no median (its sessions' would be 0.229 and 0).
            assert np.isnan(summary.loc[untested, "median_beta_inter"])
            assert np.isnan(summary.loc[untested, "p_fdr"])
        # A tested role's median is its sessions' (0.1 .. 1.0 in eight steps).
        assert summary.loc["current", "median_beta_inter"] == pytest.approx(0.55)
        tested = ["current", "prev"]
        _, expected, _, _ = multipletests(
            summary.loc[tested, "p"].to_numpy(dtype=float), method="fdr_bh"
        )
        np.testing.assert_allclose(summary.loc[tested, "p_fdr"], expected)

    def test_every_role_is_reported_even_with_no_fits(self):
        summary = cross_session_role_summary(
            pd.DataFrame(columns=["role", "beta_inter", "status"])
        )
        assert list(summary["role"]) == list(ROLES)
        assert set(summary["status"]) == {"too_few_sessions"}
        assert summary["p_fdr"].isna().all()

    def test_only_converged_sessions_count(self):
        coefs = _coefs({"current": _betas(0.5)})
        failed = coefs.copy()
        failed["status"] = "not_converged"
        failed["beta_inter"] = 99.0

        summary = pd.concat([coefs, failed], ignore_index=True)
        both = cross_session_role_summary(summary).set_index("role")
        only_ok = cross_session_role_summary(coefs).set_index("role")

        assert both.loc["current", "n_sessions"] == N_SESSIONS
        assert both.loc["current", "median_beta_inter"] == pytest.approx(
            only_ok.loc["current", "median_beta_inter"]
        )
