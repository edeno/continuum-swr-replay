"""Without a penalty the shared fitter gives the current fitters' coefficients.

On a small synthetic per-bin design the Poisson likelihood equals the current
per-bin fitter (statsmodels' GLM) on ``M7r_time_histcos_rew``; on a small
synthetic content design the grouped softmax equals the current content fitter
(``fit_conditional_poisson``) on ``M0a``. The comparisons are those of
``scripts/checks/compare_regression_to_current_fitters.py``, which runs them
on real sessions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.replay_models.model_specs import (
    MODEL_HIERARCHY,
    MODEL_PER_BIN_RATE,
)
from scripts.checks.compare_regression_to_current_fitters import (
    content_coefficients,
    largest_relative_difference,
    per_bin_coefficients,
)


def _per_bin_design(n_stops: int = 300, seed: int = 0) -> pd.DataFrame:
    """15 ms bins of stops with the columns of ``M7r_time_histcos_rew``.

    Stops last from 0.3 to 4 s (the spline's range); the log rate rises with
    the stop clock ``tc``, doubles after a reward (``reward_history_rcos_0``),
    and follows the self-history columns; every tenth stop has an incomplete
    reward history (its components zero).
    """
    rng = np.random.default_rng(seed)
    n_bins = rng.integers(20, 267, size=n_stops)
    stop_id = np.repeat(np.arange(n_stops), n_bins)
    binj = np.arange(n_bins.sum()) - np.repeat(np.cumsum(n_bins) - n_bins, n_bins)
    tc = (binj + 0.5) * 0.015
    incomplete = (stop_id % 10 == 0).astype(float)
    per_stop = rng.uniform(size=(n_stops, 2))
    reward = np.where(incomplete == 1, 0.0, (stop_id % 2).astype(float))
    rcos = np.where(incomplete[:, None] == 1, 0.0, per_stop[stop_id])
    history = rng.exponential(0.3, size=(stop_id.size, 4))
    log_rate = (
        np.log(0.03)
        + 0.6 * np.sin(tc)
        + np.log(2.0) * reward
        + 0.3 * rcos[:, 0]
        - 0.4 * history[:, 0]
        + 0.2 * history[:, 2]
    )
    exposure = np.full(stop_id.size, 0.015)
    count = rng.poisson(np.exp(log_rate)).astype(int)
    frame = pd.DataFrame(
        {
            "stop_id": stop_id,
            "binj": binj,
            "tc": tc,
            "exposure": exposure,
            "count": count,
            "reward_history_rcos_0": reward,
            "reward_history_rcos_1": rcos[:, 0],
            "reward_history_rcos_2": rcos[:, 1],
            "reward_history_incomplete": incomplete,
        }
    )
    for j in range(4):
        frame[f"offhist_rcos_{j}"] = history[:, j]
    return frame


def _content_design(
    n_events: int = 120, n_arms: int = 9, seed: int = 1
) -> pd.DataFrame:
    """Events with a share on each of nine arms and the arm indicators of ``M0a``.

    The shares are Dirichlet draws around arm probabilities that favor the
    first arms; the rows are out of event order, as a table may hold them.
    """
    rng = np.random.default_rng(seed)
    arm = np.tile(np.arange(n_arms), n_events)
    log_weight = np.linspace(0.8, -0.8, n_arms)
    probability = np.exp(log_weight) / np.exp(log_weight).sum()
    shares = np.vstack([rng.dirichlet(15.0 * probability) for _ in range(n_events)])
    frame = pd.DataFrame(
        {
            "event_id": np.repeat(np.arange(n_events), n_arms),
            "arm_fraction": shares.ravel(),
        }
    )
    for a in range(n_arms - 1):
        frame[f"is_candidate_arm_{a}"] = (arm == a).astype(float)
    return frame.sample(frac=1.0, random_state=0).reset_index(drop=True)


def test_per_bin_rung_equals_current_fitter() -> None:
    design = _per_bin_design()
    ours, current, left_out = per_bin_coefficients(
        design, MODEL_PER_BIN_RATE["M7r_time_histcos_rew"]
    )
    assert left_out.empty
    assert len(ours) == 15  # intercept, six spline, four reward, four history columns
    assert largest_relative_difference(ours, current) < 1e-6


def test_content_rung_equals_current_fitter() -> None:
    design = _content_design()
    ours, current = content_coefficients(design, MODEL_HIERARCHY["M0a"].predictors)
    assert len(ours) == 8
    assert largest_relative_difference(ours, current) < 1e-6
