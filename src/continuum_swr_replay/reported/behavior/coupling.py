"""The behavior summary's SWR-power values (question 3.8): how a non-local
event's ripple power depends on the task. The statistics are computed in
``analysis.replay_models.coefficient_statistics``."""

from __future__ import annotations

import pandas as pd

from continuum_swr_replay.analysis.replay_models.coefficient_statistics import (
    coefficient_statistics,
)
from continuum_swr_replay.reported.behavior.common import TAGS
from continuum_swr_replay.reported.values import (
    Value,
    value_factory,
)

#: The SWR-power model's covariates the behavior summary reports: key word -> column.
POWER_COVARIATES = {
    "reward": "known_reward_outcome_lag_1",
    "rpe": "rpe_postoutcome",
    "exploration": "patch_switch_fraction_last5_completed_trips",
}


# --- the SWR-power mark -----------------------------------------------------------


def power_values(coefs: pd.DataFrame, source: str) -> list[Value]:
    """The SWR-coupling mark: per-session OLS of event ripple power on
    standardized covariates, summarized as the fitting script prints it (median,
    share positive, animals whose median is positive, signed-rank p)."""
    make = value_factory(power_values, source, TAGS["ripple_power"])
    values = [
        make(
            "power_sessions",
            len(coefs),
            "count",
            "count",
            "sessions in the SWR-power model",
        )
    ]
    for word, column in POWER_COVARIATES.items():
        stats = coefficient_statistics(coefs, column)
        values += [
            make(
                f"power_{word}",
                stats["median"],
                "quantity",
                "sig2s",
                f"{column}: median per-session coefficient on ripple_power_z",
                "SD of power per SD",
            ),
            make(
                f"power_{word}_animals_positive",
                stats["n_animals_positive"],
                "count",
                "count",
                f"{column}: animals with a positive median",
            ),
            make(
                f"power_{word}_p",
                stats["signed_rank_p"],
                "quantity",
                "pval",
                f"{column}: signed-rank p",
            ),
        ]
    return values
