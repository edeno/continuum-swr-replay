"""Regression test: theta_power z-score must use ``nan_policy="omit"``.

The default ``scipy.stats.zscore`` (``nan_policy="propagate"``) returns
all-NaN whenever any input value is NaN. ``_nearest`` in the data-
aggregation paths can leave NaN at any timebin where no LFP sample
fell within tolerance, so the propagate default would make the entire
session's ``theta_power_zscore`` NaN — silently suppressing theta
event classification downstream in ``quality_metrics.compute_non_local_event_context``.
This test pins both call sites at the source level so a future
revision can't accidentally drop the kwarg.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import zscore

REPO_ROOT = Path(__file__).resolve().parents[2]
HPC_AGG = REPO_ROOT / "src" / "continuum_swr_replay" / "analysis" / "hippocampus" / "data_aggregation.py"
CONTFRAG_AGG = REPO_ROOT / "src" / "continuum_swr_replay" / "analysis" / "contfrag" / "data_aggregation.py"


def _theta_zscore_call(source: str) -> str:
    """Extract the ``zscore(...)`` block following ``theta_power_zscore``."""
    marker = '"theta_power_zscore": zscore('
    start = source.index(marker) + len('"theta_power_zscore": ')
    # Walk balanced parens.
    depth = 0
    for i, ch in enumerate(source[start:], start=start):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return source[start : i + 1]
    raise AssertionError("unterminated zscore(...) call")


class TestThetaPowerZScoreNanPolicy:
    def test_scipy_default_propagates_nan_to_full_series(self):
        """Document why we pin ``nan_policy="omit"``: the default
        ``"propagate"`` zeroes out the entire output even when only
        one input bin is NaN."""
        out = zscore(np.array([1.0, 2.0, np.nan, 4.0]))
        assert np.all(np.isnan(out)), (
            "scipy.stats.zscore default behavior changed — verify the "
            "data-aggregation paths still need nan_policy='omit'."
        )

    def test_nan_policy_omit_preserves_finite_z_scores(self):
        """``nan_policy="omit"`` is the contract we depend on."""
        out = zscore(np.array([1.0, 2.0, np.nan, 4.0]), nan_policy="omit")
        # Three finite z-scores around the omitted-NaN mean/std.
        finite_mask = ~np.isnan(out)
        assert finite_mask.sum() == 3
        assert np.allclose(out[finite_mask].mean(), 0.0, atol=1e-12)

    def test_hippocampus_data_aggregation_uses_nan_policy_omit(self):
        call = _theta_zscore_call(HPC_AGG.read_text())
        assert 'nan_policy="omit"' in call, (
            f"hippocampus/data_aggregation.py theta_power_zscore call "
            f"is missing nan_policy='omit':\n{call}"
        )

    def test_contfrag_data_aggregation_uses_nan_policy_omit(self):
        call = _theta_zscore_call(CONTFRAG_AGG.read_text())
        assert 'nan_policy="omit"' in call, (
            f"contfrag/data_aggregation.py theta_power_zscore call "
            f"is missing nan_policy='omit':\n{call}"
        )
