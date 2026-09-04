"""The summary figure's reward kernel must be the summarizer's estimator.

Panel C of the six-panel per-bin summary and the kernel figure drawn by
``plot_per_bin_rate_models.py`` report the same quantity, so they must compute
it the same way: reconstruct each session's kernel from that session's own basis
coefficients, then take the per-lag median. The three raised-cosine bumps are
strongly collinear, so reconstructing from across-session median coefficients
gives a different curve -- one belonging to no fitted session.
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
    "plot_per_bin_rate_summary",
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "rate"
    / "plot_per_bin_rate_summary.py",
)
ppbrs = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ppbrs)


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

    kernel = ppbrs.reward_kernel_rate_ratios(sessions)

    np.testing.assert_allclose(
        kernel, _expected_median_of_per_session_kernels(sessions), rtol=1e-12
    )
    assert kernel.shape == (REWARD_HISTORY_N_LAGS,)


def test_kernel_differs_from_reconstructing_median_coefficients():
    """Guard the estimator: the two orders are not interchangeable.

    The per-column medians here come from three different sessions, so the
    median coefficient vector is a session that was never fit.
    """
    sessions = pd.DataFrame(
        {
            "animal": ["a", "b", "c"],
            f"coef_{REWARD_HISTORY_COLUMNS[0]}": [2.0, 0.1, -1.0],
            f"coef_{REWARD_HISTORY_COLUMNS[1]}": [-0.2, 1.0, -3.0],
            f"coef_{REWARD_HISTORY_COLUMNS[2]}": [-1.5, 2.0, 0.3],
        }
    )
    basis = raised_cosine_history_basis(REWARD_HISTORY_N_LAGS, REWARD_HISTORY_N_BASIS)
    weights = basis / basis.sum(axis=1, keepdims=True)
    median_coefs = np.array(
        [sessions[f"coef_{c}"].median() for c in REWARD_HISTORY_COLUMNS]
    )

    kernel = ppbrs.reward_kernel_rate_ratios(sessions)
    from_median_coefs = np.exp(median_coefs @ weights)

    np.testing.assert_allclose(
        kernel, _expected_median_of_per_session_kernels(sessions), rtol=1e-12
    )
    assert abs(kernel[0] - from_median_coefs[0]) > 0.05


def test_the_summary_panel_and_the_summarizer_agree():
    """Panel C and ``summarize_per_bin_rate_models.py`` report one number."""
    spec = importlib.util.spec_from_file_location(
        "summarize_per_bin_rate_models",
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "rate"
        / "summarize_per_bin_rate_models.py",
    )
    spbrm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(spbrm)
    sessions = _two_sessions()

    np.testing.assert_allclose(
        ppbrs.reward_kernel_rate_ratios(sessions),
        spbrm.reward_kernel_rate_ratios(sessions),
        rtol=1e-12,
    )


def test_sessions_missing_the_block_do_not_contribute():
    """A session with no fitted reward block is dropped, not read as zero."""
    sessions = _two_sessions()
    unfit = sessions.iloc[[0]].copy()
    unfit[f"coef_{REWARD_HISTORY_COLUMNS[1]}"] = np.nan

    np.testing.assert_allclose(
        ppbrs.reward_kernel_rate_ratios(pd.concat([sessions, unfit])),
        ppbrs.reward_kernel_rate_ratios(sessions),
        rtol=1e-12,
    )
