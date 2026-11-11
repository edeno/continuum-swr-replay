"""The per-bin summarizer's reward kernel must be the figure's estimator.

The reward-history features are strongly collinear, so reconstructing a kernel
from across-session median coefficients and taking the median of per-session
reconstructions are different numbers. The summarizer table and the figure must
both report the second one.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.replay.model_specs import (
    N_HISTORY_BASIS,
    N_HISTORY_LAGS,
    REWARD_HISTORY_COLUMNS,
    REWARD_HISTORY_N_BASIS,
    REWARD_HISTORY_N_LAGS,
)
from continuum_swr_replay.analysis.replay.per_bin_design import (
    raised_cosine_history_basis,
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


# The basis columns the fixture's history block carries.
HIST = [f"hist_rcos_{k}" for k in range(N_HISTORY_BASIS)]


def _history_sessions() -> pd.DataFrame:
    """Two sessions' raised-cosine self-history coefficients."""
    return pd.DataFrame(
        {
            "animal": ["a", "b"],
            "coef_hist_rcos_0": [-2.0, -1.2],
            "coef_hist_rcos_1": [0.4, 0.1],
            "coef_hist_rcos_2": [0.9, 0.6],
            "coef_hist_rcos_3": [0.2, -0.1],
        }
    )


def test_self_history_kernel_uses_the_unnormalized_basis():
    """``hist_rcos_k`` is a plain projection, so its kernel is ``coef @ B``.

    The reward block divides each bump by its span because its features are
    bump-weighted fractions; dividing here would shrink every self-history lag
    by the same wrong factor, so the two reconstructions must differ.
    """
    sessions = _history_sessions()
    basis = raised_cosine_history_basis(N_HISTORY_LAGS, N_HISTORY_BASIS)
    columns = [f"coef_hist_rcos_{k}" for k in range(N_HISTORY_BASIS)]
    expected = np.median(
        np.exp(sessions[columns].to_numpy(dtype=float) @ basis), axis=0
    )

    kernel = spbrm.self_history_rate_ratios(sessions, HIST)

    np.testing.assert_allclose(kernel, expected, rtol=1e-12)
    normalized = np.median(
        np.exp(
            sessions[columns].to_numpy(dtype=float)
            @ (basis / basis.sum(axis=1, keepdims=True))
        ),
        axis=0,
    )
    assert not np.allclose(kernel, normalized)


def test_self_history_sessions_missing_the_block_do_not_contribute():
    """A session that did not fit the history block is dropped, not zero-filled."""
    sessions = _history_sessions()
    with_gap = pd.concat(
        [sessions, sessions.iloc[[0]].assign(coef_hist_rcos_1=np.nan)],
        ignore_index=True,
    )

    np.testing.assert_allclose(
        spbrm.self_history_rate_ratios(with_gap, HIST),
        spbrm.self_history_rate_ratios(sessions, HIST),
        rtol=1e-12,
    )
