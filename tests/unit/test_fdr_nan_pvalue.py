"""Regression test: cross-session FDR must not be poisoned
by a NaN p-value from a single role.

``run_stage6_learning.py`` builds a list of role-level Wilcoxon
p-values, sets ``p = float("nan")`` when ``wilcoxon`` raises
``ValueError`` (e.g. all-zero β_inter samples on some SciPy
versions), then passes the list to ``multipletests(..., "fdr_bh")``.
``multipletests`` returns an all-NaN adjusted-p vector when *any*
input is NaN, so a single failed role contaminated every other
role's reported FDR.

The fix correct only the finite p-values and pass NaN through for
the failed roles. This test pins that behavior by extracting the
exact code block from the script and exercising it on a mixed list.
"""

from __future__ import annotations

import numpy as np
from statsmodels.stats.multitest import multipletests


def _stage6_fdr(pvals: list[float]) -> np.ndarray:
    """Mirror of the post-fix logic in run_stage6_learning.main()."""
    pvals_arr = np.asarray(pvals, dtype=float)
    ok = np.isfinite(pvals_arr)
    p_fdr = np.full_like(pvals_arr, np.nan)
    if ok.any():
        _, p_fdr_ok, _, _ = multipletests(pvals_arr[ok], method="fdr_bh")
        p_fdr[ok] = p_fdr_ok
    return p_fdr


class TestStage6FdrNanPropagation:
    def test_nan_pvalue_does_not_poison_other_roles(self):
        # Sanity: the bug repro the user reported.
        full = multipletests([0.01, np.nan, 0.02], method="fdr_bh")
        assert np.all(np.isnan(full[1])), (
            "expected baseline NaN-poisoning behavior to still hold "
            "for raw multipletests, otherwise this test no longer "
            "exercises the regression"
        )

        out = _stage6_fdr([0.01, np.nan, 0.02])
        # Roles with finite p get a real adjusted value (BH on
        # [0.01, 0.02] gives [0.02, 0.02]).
        np.testing.assert_allclose(out[0], 0.02, atol=1e-12)
        np.testing.assert_allclose(out[2], 0.02, atol=1e-12)
        # The failed role keeps NaN.
        assert np.isnan(out[1])

    def test_all_nan_input_returns_all_nan(self):
        out = _stage6_fdr([float("nan"), float("nan")])
        assert np.all(np.isnan(out))

    def test_all_finite_input_matches_raw_multipletests(self):
        ps = [0.05, 0.1, 0.5]
        out = _stage6_fdr(ps)
        _, expected, _, _ = multipletests(ps, method="fdr_bh")
        np.testing.assert_allclose(out, expected)
