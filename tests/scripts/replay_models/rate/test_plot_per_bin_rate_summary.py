"""The per-bin ladder's summary figure says which panels read gated sessions."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay_models.model_specs import (
    REWARD_HISTORY_COLUMNS,
)
from scripts.replay_models.rate import plot_per_bin_rate_summary as mod

N_SESSIONS = 10


@pytest.fixture
def full_rung_fits() -> pd.DataFrame:
    """The full rung's reward coefficients for every population, no term gated."""
    rng = np.random.default_rng(0)
    frames = []
    for outcome in mod.POPS4:
        frame = pd.DataFrame(
            {
                "model": mod.FULL,
                "outcome": outcome,
                "animal": [("a", "b", "c", "d", "e")[i % 5] for i in range(N_SESSIONS)],
                "session_date": [f"d{i}" for i in range(N_SESSIONS)],
                "epoch": "02_r1",
                "separated_terms": "",
                "converged": True,
            }
        )
        for term in REWARD_HISTORY_COLUMNS:
            frame[f"coef_{term}"] = rng.normal(0.2, 0.1, N_SESSIONS)
            frame[f"sd_{term}"] = 0.1
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


@pytest.mark.parametrize(
    "panel", [mod._panel_reward1, mod._panel_reward_kernel, mod._panel_covar]
)
def test_the_gated_reward_panels_say_so(full_rung_fits, panel):
    fig, ax = plt.subplots()
    panel(ax, full_rung_fits)
    assert "gated sessions" in ax.get_title()
    plt.close(fig)
