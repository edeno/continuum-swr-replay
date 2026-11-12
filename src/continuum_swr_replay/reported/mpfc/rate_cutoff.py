"""The mpfc summary's firing-rate check: a few main values recomputed over the
units at or above a 1 Hz cutoff."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.pfc import population_statistics
from continuum_swr_replay.reported.mpfc.population import (
    called_fraction_values,
    overlap_and_effect_values,
)
from continuum_swr_replay.reported.mpfc.swr_without_replay import (
    swr_without_replay_values,
)
from continuum_swr_replay.reported.values import (
    Value,
)

CHECK_FIRING_RATE_HZ = 1.0
"""The cutoff the main values used before 2026-10-29, kept as a check; the full
table over cutoffs is ``scripts/pfc/part4_rate_threshold_sensitivity.py``'s CSV."""

#: The check keys quoted at :data:`CHECK_FIRING_RATE_HZ`, by their main key.
CHECK_KEYS = (
    "units",
    "swr_replay_called_fraction",
    "non_swr_replay_called_fraction",
    "engagement_called_fraction",
    "overlap_odds_ratio",
    "overlap_rank_correlation",
    "swr_no_replay_called_fraction_all",
)


# --- the 1 Hz check ----------------------------------------------------------------


def rate_cutoff_check_values(
    population: pd.DataFrame,
    extended: tuple[pd.DataFrame, dict, np.ndarray, list] | None,
    make: Callable[..., Value],
) -> list[Value]:
    """A few main values recomputed with a :data:`CHECK_FIRING_RATE_HZ` cutoff.

    The same functions as the main values, on the population restricted to
    units at or above the cutoff (``immobility_avg_firing_rate``); only the keys
    of :data:`CHECK_KEYS` are kept, renamed ``check_one_hz_<key>``.
    """
    gated, _ = population_statistics.units_at_or_above_rate(
        population, CHECK_FIRING_RATE_HZ
    )
    computed = [
        make(
            "units",
            len(gated),
            "count",
            "count",
            "mPFC units in the population (a usable fit and finite curves)",
        )
    ]
    computed += called_fraction_values(gated, make)
    computed += overlap_and_effect_values(gated, make)
    if extended is not None:
        ext_population, ext_curves, ext_lag, ext_sidecars = extended
        ext_gated, ext_gated_curves = population_statistics.units_at_or_above_rate(
            ext_population, CHECK_FIRING_RATE_HZ, ext_curves
        )
        computed += swr_without_replay_values(
            ext_gated, ext_gated_curves, ext_lag, ext_sidecars, make
        )
    values = []
    for value in computed:
        if value.key in CHECK_KEYS:
            values.append(
                replace(
                    value,
                    key=f"check_one_hz_{value.key}",
                    description=f"check, units at or above "
                    f"{CHECK_FIRING_RATE_HZ:g} Hz on the fitted bins: "
                    + value.description,
                )
            )
    return values
