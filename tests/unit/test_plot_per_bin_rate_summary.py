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

from continuum_swr_replay.analysis.replay.model_specs import (
    REWARD_HISTORY_COLUMNS,
    REWARD_HISTORY_N_BASIS,
    REWARD_HISTORY_N_LAGS,
)
from continuum_swr_replay.analysis.replay.per_bin_design import (
    raised_cosine_history_basis,
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


def _clocked_stops() -> pd.DataFrame:
    """Two stops of one session: [10, 12] holds excluded time [10.5, 11.0];
    [20, 21] holds none. Family A's exposure is the span less excluded time."""
    return pd.DataFrame(
        {
            "animal": "a",
            "session_date": 1,
            "epoch": "02_r1",
            "stop_id": [1, 2],
            "stop_start": [10.0, 20.0],
            "stop_end": [12.0, 21.0],
            "immobility_duration_s": [1.5, 1.0],
        }
    )


def test_the_time_course_clock_is_the_stops_span_less_excluded_time(monkeypatch):
    """The empirical rate's clock is time since the stop began, so a stop's
    exposure in each lag bin is its span there less the excluded seconds in
    it -- not its first ``immobility_duration_s`` seconds."""
    excluded = pd.DataFrame(
        {
            "animal": ["a"],
            "session_date": [1],
            "epoch": ["02_r1"],
            "start_time": [10.5],
            "end_time": [11.0],
        }
    )
    monkeypatch.setattr(
        ppbrs, "excluded_intervals_by_session", lambda sessions: excluded
    )
    onsets = pd.DataFrame(
        {"event_type": "SWR", "stop_id": [1, 1], "t_since_stop_onset": [0.2, 1.6]}
    )

    tc, rate = ppbrs._timecourse(_clocked_stops(), onsets, "swr", dt=0.5, tmax=2.0)

    # exposure per 0.5 s lag bin: [0, .5) both stops; [.5, 1) stop 1 is excluded,
    # stop 2 is not; [1, 1.5) and [1.5, 2) stop 1 alone
    np.testing.assert_allclose(tc, [0.25, 0.75, 1.25, 1.75])
    np.testing.assert_allclose(rate, [1 / 1.0, 0 / 0.5, 0 / 0.5, 1 / 0.5])
    t, surviving = ppbrs._survival(_clocked_stops(), tmax=2.0, dt=0.5)
    np.testing.assert_allclose(surviving, [1.0, 1.0, 0.5, 0.5])
