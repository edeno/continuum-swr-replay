"""The behavior summary's SWR-power values read each covariate's own column."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.reported.behavior import coupling
from tests.support._behavior_fits import ANIMALS, values_by_key

NAN = np.nan


def test_power_values_read_each_covariate_s_own_column():
    rising = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    coefs = pd.DataFrame(
        {
            "animal": [*ANIMALS, "e"],
            "session_date": list(range(7)),
            "epoch": "02_r1",
            "known_reward_outcome_lag_1": rising,
            "rpe_postoutcome": [-c for c in rising],
            # one session has no coefficient for this covariate
            "patch_switch_fraction_last5_completed_trips": [2.0] * 6 + [NAN],
        }
    )
    values = values_by_key(coupling.power_values(coefs, "source"))
    assert values["power_sessions"] == 7
    assert values["power_reward"] == pytest.approx(0.4)
    assert values["power_rpe"] == pytest.approx(-0.4)
    assert values["power_exploration"] == 2.0
    assert values["power_reward_animals_positive"] == 5
    assert values["power_rpe_animals_positive"] == 0
    assert values["power_reward_p"] == pytest.approx(2 / 2**7)
    assert values["power_exploration_p"] == pytest.approx(2 / 2**6)
