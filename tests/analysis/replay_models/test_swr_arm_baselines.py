"""SWR-specific arm baselines in the pooled content contrasts, on simulated data.

SWR and non-SWR events here favor the current arm by exactly the same amount;
SWR events only put more of their content on the three trunk arms. With one set
of arm baselines shared by both kinds of event, that trunk shift loads on the
only SWR-specific term, ``swr_x_is_current_arm``, which reads negative. With
the SWR-specific baselines the pooled rungs now carry, it reads zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay_models.fitting import fit_conditional_poisson
from continuum_swr_replay.analysis.replay_models.model_specs import (
    ARMFE,
    SWR_ARMFE,
    SWR_CONTRAST_CONTENT_ARM_LADDER,
    leaf_only_predictors,
)

N_ARMS = 9
TRUNKS = (6, 7, 8)
CURRENT_LOGIT = 1.0
SWR_TRUNK_LOGIT = 1.0


def _events(n_events: int = 600, seed: int = 0) -> pd.DataFrame:
    """Long frame, one row per event x arm, with the expected content fraction."""
    rng = np.random.default_rng(seed)
    current = rng.integers(0, 6, n_events)
    is_swr = rng.random(n_events) < 0.5
    arm = np.tile(np.arange(N_ARMS), n_events)
    event = np.repeat(np.arange(n_events), N_ARMS)
    is_current = (arm == np.repeat(current, N_ARMS)).astype(float)
    swr = np.repeat(is_swr, N_ARMS).astype(float)
    logit = CURRENT_LOGIT * is_current + SWR_TRUNK_LOGIT * swr * np.isin(arm, TRUNKS)
    weight = np.exp(logit).reshape(n_events, N_ARMS)
    fraction = (weight / weight.sum(axis=1, keepdims=True)).ravel()
    frame = pd.DataFrame(
        {
            "event_id": event,
            "candidate_arm": arm,
            "arm_fraction": fraction,
            "is_current_arm": is_current,
            "swr_x_is_current_arm": swr * is_current,
        }
    )
    for k, column in enumerate(ARMFE):
        frame[column] = (arm == k).astype(float)
        frame[f"swr_x_{column}"] = swr * frame[column]
    return frame


def _swr_current(predictors: list[str]) -> float:
    fit = fit_conditional_poisson(
        _events(), predictors=predictors, response_col="arm_fraction"
    )
    assert fit is not None
    return fit["coef_swr_x_is_current_arm"]


def test_a_shared_baseline_reads_a_trunk_shift_as_off_the_current_arm():
    shared = [*ARMFE, "is_current_arm", "swr_x_is_current_arm"]
    assert _swr_current(shared) < -0.2


def test_swr_specific_baselines_leave_the_current_arm_contrast_at_zero():
    own = [*ARMFE, *SWR_ARMFE, "is_current_arm", "swr_x_is_current_arm"]
    assert _swr_current(own) == pytest.approx(0.0, abs=1e-6)


def test_every_pooled_rung_carries_the_swr_arm_baselines():
    for name, (predictors, scope) in SWR_CONTRAST_CONTENT_ARM_LADDER.items():
        has = set(SWR_ARMFE) <= set(predictors)
        assert has == (scope == "pooled"), name


def test_a_leaf_only_fit_keeps_the_leaf_baselines_and_their_swr_twins():
    kept = leaf_only_predictors([*ARMFE, *SWR_ARMFE])
    assert kept == [*ARMFE[:5], *SWR_ARMFE[:5]]
