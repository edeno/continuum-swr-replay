"""The per-bin summarizer's reward kernel must be the figure's estimator.

The reward-history features are strongly collinear, so reconstructing a kernel
from across-session median coefficients and taking the median of per-session
reconstructions are different numbers. The summarizer table, the figure and the
prose in ``docs/per_bin_rate_results.md`` must all report the second one.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.replay._helpers import (
    raised_cosine_history_basis,
)
from continuum_swr_replay.analysis.replay.model_specs import (
    REWARD_HISTORY_COLUMNS,
    REWARD_HISTORY_N_BASIS,
    REWARD_HISTORY_N_LAGS,
)

_SPEC = importlib.util.spec_from_file_location(
    "summarize_per_bin_rate_models",
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "rate"
    / "summarize_per_bin_rate_models.py",
)
spbrm = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(spbrm)


def _two_sessions() -> pd.DataFrame:
    """Two sessions whose basis coefficients differ enough to separate estimators."""
    return pd.DataFrame(
        {
            "animal": ["a", "b"],
            f"coef_{REWARD_HISTORY_COLUMNS[0]}": [1.4, 0.2],
            f"coef_{REWARD_HISTORY_COLUMNS[1]}": [-2.6, -0.1],
            f"coef_{REWARD_HISTORY_COLUMNS[2]}": [1.5, 0.3],
        }
    )


def _expected_median_of_per_session_kernels(sessions: pd.DataFrame) -> np.ndarray:
    """Reconstruct each session independently, then take the per-lag median."""
    basis = raised_cosine_history_basis(REWARD_HISTORY_N_LAGS, REWARD_HISTORY_N_BASIS)
    weights = basis / basis.sum(axis=1, keepdims=True)
    per_session = [
        np.exp(
            np.array([row[f"coef_{c}"] for c in REWARD_HISTORY_COLUMNS], dtype=float)
            @ weights
        )
        for _, row in sessions.iterrows()
    ]
    return np.median(np.vstack(per_session), axis=0)


def test_kernel_is_the_median_of_per_session_reconstructions():
    sessions = _two_sessions()

    kernel = spbrm.reward_kernel_rate_ratios(sessions)

    np.testing.assert_allclose(
        kernel, _expected_median_of_per_session_kernels(sessions), rtol=1e-12
    )
    assert kernel.shape == (REWARD_HISTORY_N_LAGS,)


def _three_sessions() -> pd.DataFrame:
    """Three sessions whose per-column medians come from three different rows.

    The vector of marginal median coefficients is then a session that was never
    fit, which is exactly the failure mode the estimator choice avoids.
    """
    return pd.DataFrame(
        {
            "animal": ["a", "b", "c"],
            f"coef_{REWARD_HISTORY_COLUMNS[0]}": [2.0, 0.1, -1.0],
            f"coef_{REWARD_HISTORY_COLUMNS[1]}": [-0.2, 1.0, -3.0],
            f"coef_{REWARD_HISTORY_COLUMNS[2]}": [-1.5, 2.0, 0.3],
        }
    )


def test_kernel_differs_from_reconstructing_median_coefficients():
    """Guard the estimator: the two orders are not interchangeable."""
    sessions = _three_sessions()
    basis = raised_cosine_history_basis(REWARD_HISTORY_N_LAGS, REWARD_HISTORY_N_BASIS)
    weights = basis / basis.sum(axis=1, keepdims=True)
    median_coefs = np.array(
        [sessions[f"coef_{c}"].median() for c in REWARD_HISTORY_COLUMNS]
    )

    kernel = spbrm.reward_kernel_rate_ratios(sessions)
    from_median_coefs = np.exp(median_coefs @ weights)

    np.testing.assert_allclose(
        kernel, _expected_median_of_per_session_kernels(sessions), rtol=1e-12
    )
    assert abs(kernel[0] - from_median_coefs[0]) > 0.05


def test_sessions_missing_the_block_do_not_contribute():
    """A session with no fitted reward block is dropped, not read as zero."""
    sessions = _two_sessions()
    unfit = sessions.iloc[[0]].copy()
    unfit[f"coef_{REWARD_HISTORY_COLUMNS[1]}"] = np.nan

    np.testing.assert_allclose(
        spbrm.reward_kernel_rate_ratios(pd.concat([sessions, unfit])),
        spbrm.reward_kernel_rate_ratios(sessions),
        rtol=1e-12,
    )
