"""Cohort gating and separation flagging in the is_swr-contrast rate runner.

Two guards:

* ``_mask_post_outcome_covariates`` -- a stop that starts *before* the trial's
  outcome is known cannot carry a post-outcome covariate, so those values are
  NaN'd (and later removed by the ladder's cohort freeze) rather than being fit
  as if the animal already knew the outcome.
* ``_fit`` -- the separation flag is judged on the effect scale, so a
  small-SD design column with an ordinary effect is not mistaken for a
  quasi-separated fit just because its per-unit coefficient is large.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.rate import fit_swr_contrast_rate as mod


@pytest.fixture
def stop_frame() -> pd.DataFrame:
    """Three stops: one starts before its outcome is known, two do not."""
    return pd.DataFrame(
        {
            "stop_id": [1, 2, 3],
            "stop_start": [10.0, 20.0, 30.0],
            # stop 2 starts 1 s before the outcome is known; stop 3 starts
            # exactly at it (not before -> keeps its post-outcome values).
            "outcome_known_time": [9.5, 21.0, 30.0],
            "occupancy_state": ["at_well", "at_well", "in_transit"],
            "rpe_postoutcome": [0.4, -0.6, 0.2],
            "reward_belief_mean_update_l1_all_ports_postoutcome": [0.1, 0.2, 0.3],
            "relative_value_switch_minus_stay_preoutcome": [1.0, 2.0, 3.0],
        }
    )


def test_post_outcome_gate_nans_only_pre_outcome_stops(stop_frame):
    gated, counts = mod._mask_post_outcome_covariates(stop_frame)

    assert gated["rpe_postoutcome"].tolist()[0] == 0.4
    assert np.isnan(gated["rpe_postoutcome"].iloc[1])
    assert gated["rpe_postoutcome"].iloc[2] == 0.2
    update = gated["reward_belief_mean_update_l1_all_ports_postoutcome"]
    assert update.iloc[0] == 0.1
    assert np.isnan(update.iloc[1])
    assert update.iloc[2] == 0.3
    # Pre-outcome covariates are untouched on the same rows.
    assert gated["relative_value_switch_minus_stay_preoutcome"].tolist() == [
        1.0,
        2.0,
        3.0,
    ]
    assert counts["at_well"] == 1
    assert counts.get("in_transit", 0) == 0


def test_post_outcome_gate_does_not_mutate_the_input(stop_frame):
    mod._mask_post_outcome_covariates(stop_frame)
    assert stop_frame["rpe_postoutcome"].tolist() == [0.4, -0.6, 0.2]


def test_post_outcome_gate_requires_the_outcome_clock(stop_frame):
    with pytest.raises(ValueError, match="outcome_known_time"):
        mod._mask_post_outcome_covariates(stop_frame.drop(columns="outcome_known_time"))


def test_small_sd_predictor_with_ordinary_effect_is_not_flagged_separated():
    """|coef| = 12 on a column with SD 0.03 is a 0.36-log-unit effect, not separation.

    The legacy ``|coef| >= 10`` rule flagged exactly this design; the effect-scale
    rule (``bse * SD``) does not.
    """
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.normal(0.0, 0.03, n)
    exposure = np.full(n, 0.015)
    rate = np.exp(np.log(2.0) + 12.0 * x)
    design = pd.DataFrame(
        {
            "stop_id": np.repeat(np.arange(n // 10), 10),
            "exposure": exposure,
            "x": x,
            "count": rng.poisson(rate * exposure),
        }
    ).astype({"count": float})

    row, reason = mod._fit(design, "x", ["x"])

    assert reason is None
    assert row["coef_x"] == pytest.approx(12.0, abs=3.0)
    # The design column's SD is small, so the per-unit coefficient is large ...
    assert abs(row["coef_x"]) >= 10.0
    # ... but the effect-scale standard error is far below the separation bound.
    assert row["bse_x"] * x.std() < 1.0
    assert row["separated"] is False
    assert row["converged"] is True
