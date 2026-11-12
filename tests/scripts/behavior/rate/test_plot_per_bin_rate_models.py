"""The per-bin ladder's reward-kernel figure draws what the summary reports."""

from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.behavior.model_specs import (
    REWARD_HISTORY_COLUMNS,
    REWARD_HISTORY_N_BASIS,
    REWARD_HISTORY_N_LAGS,
)
from continuum_swr_replay.analysis.behavior.per_bin_design import (
    reward_kernel_weights,
)
from scripts.behavior.rate import plot_per_bin_rate_models as mod

ANIMALS = ("a", "b", "c", "d", "e")
N_SESSIONS = 20
LAG_ONE_WEIGHTS = reward_kernel_weights(REWARD_HISTORY_N_LAGS, REWARD_HISTORY_N_BASIS)[
    :, 0
]


def _fits(outcome: str, coefs: np.ndarray, separated: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "model": mod.MODEL,
            "outcome": outcome,
            "animal": [ANIMALS[i % 5] for i in range(N_SESSIONS)],
            "session_date": [f"d{i}" for i in range(N_SESSIONS)],
            "epoch": "02_r1",
            "separated_terms": separated,
            "converged": True,
        }
    )
    for k, term in enumerate(REWARD_HISTORY_COLUMNS):
        frame[f"coef_{term}"] = coefs[:, k]
        frame[f"sd_{term}"] = 0.1
    return frame


def test_the_dissociation_panel_draws_the_paired_lag_one_kernels(
    processed_data_root, monkeypatch
):
    """The non-SWR stream has the larger first basis coefficient but, offset by
    the second, the smaller lag-1 kernel in every session: the panel reads the
    kernel, 100% positive, where the first coefficient would read 0%. One
    session's reward term is separated in the SWR-coupled stream alone, so 19
    sessions are paired."""
    rng = np.random.default_rng(0)
    swr_coupled = np.column_stack(
        [1.0 + rng.normal(0, 0.05, N_SESSIONS), np.zeros((N_SESSIONS, 2))]
    )
    no_swr = np.column_stack(
        [
            2.0 + rng.normal(0, 0.05, N_SESSIONS),
            np.full(N_SESSIONS, -12.0),
            np.zeros(N_SESSIONS),
        ]
    )
    assert (swr_coupled @ LAG_ONE_WEIGHTS > no_swr @ LAG_ONE_WEIGHTS).all()
    assert (swr_coupled[:, 0] < no_swr[:, 0]).all()
    separated = [""] * N_SESSIONS
    separated[3] = REWARD_HISTORY_COLUMNS[2]
    m7 = pd.concat(
        [
            _fits("swr", swr_coupled, [""] * N_SESSIONS),
            _fits("non_local_swr", swr_coupled, separated),
            _fits("non_local_no_swr", no_swr, [""] * N_SESSIONS),
        ],
        ignore_index=True,
    )
    figures = []
    monkeypatch.setattr(mod.plt, "close", figures.append)
    mod.figure_dir("rate").mkdir(parents=True)

    mod.plot_reward_kernel(m7)

    panel = figures[0].axes[1]
    # 19 positive differences: the exact two-sided p is 2 / 2 ** 19
    assert panel.get_title().splitlines()[1] == (
        "NL-SWR > NL-no-SWR: 100% of 19, 5/5 animals, p=3.8e-06"
    )
    # one line per paired session, joining its two kernels
    session_lines = [line for line in panel.lines if list(line.get_xdata()) == [1, 2]]
    assert len(session_lines) == 19
    kept = [i for i in range(N_SESSIONS) if i != 3]
    np.testing.assert_allclose(
        sorted(line.get_ydata()[0] for line in session_lines),
        sorted(np.exp(swr_coupled[kept] @ LAG_ONE_WEIGHTS)),
    )
