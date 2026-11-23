"""Synthetic visit bins for the visit rate model's tests.

:func:`synthetic_visit_rows` lays the in-visit at-risk bins of one session as
the cohort step returns them (``visit_rate.cohort_rows``): visits of random
length, each one or two immobile stretches (with a 0.3 s movement between
them), bins of ``bin_s`` from each stretch's start, the three clocks at the
bin center, and an outcome history with ten known outcomes in every bin. The
counts are zero; a test draws them from a scenario's expected counts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay_models.rate.visit_rate import (
    ARRIVAL,
    LEAVING,
    REWARD,
    STRETCH,
    CohortRows,
)
from continuum_swr_replay.analysis.replay_models.visit_design import (
    REWARD_LAG_COLUMNS,
)

#: The movement between two stretches of a visit (s).
GAP_S = 0.3


def synthetic_visit_rows(
    key: tuple[str, str, str],
    *,
    n_visits: int,
    seed: int,
    bin_s: float = 0.05,
    p_reward: float = 0.6,
    two_stretch_share: float = 0.5,
    length_s: tuple[float, float] = (1.5, 8.0),
) -> pd.DataFrame:
    """One session's synthetic in-visit bins (module docstring).

    Parameters
    ----------
    key : tuple of str
        (animal, session_date, epoch).
    n_visits : int
    seed : int
    bin_s : float
        Bin width (s).
    p_reward : float
        The chance that a visit is rewarded.
    two_stretch_share : float
        The share of visits with two stretches; 0 makes the stretch clock the
        leaving clock in every bin.
    length_s : tuple of float
        The range of the visits' lengths (uniform).

    Returns
    -------
    pd.DataFrame
        One row per bin.
    """
    rng = np.random.default_rng(seed)
    outcomes = (rng.random(n_visits + len(REWARD_LAG_COLUMNS)) < p_reward).astype(float)
    parts = []
    for visit in range(n_visits):
        length = rng.uniform(*length_s)
        if rng.random() < two_stretch_share:
            cut = rng.uniform(0.3, length - GAP_S - 0.3)
            stretches = [(0.0, cut), (cut + GAP_S, length)]
        else:
            stretches = [(0.0, length)]
        history = outcomes[visit : visit + len(REWARD_LAG_COLUMNS)][::-1]
        for index, (start, end) in enumerate(stretches):
            left = np.arange(start, end - 1e-9, bin_s)
            right = np.minimum(left + bin_s, end)
            center = (left + right) / 2
            part = pd.DataFrame(
                {
                    "well_visit_id": visit,
                    "visit_stretch_index": index,
                    "bin_start_s": 100.0 * visit + left,
                    "bin_end_s": 100.0 * visit + right,
                    "exposure": right - left,
                    "count": 0.0,
                    ARRIVAL: center,
                    LEAVING: length - center,
                    STRETCH: end - center,
                }
            )
            for column, outcome in zip(REWARD_LAG_COLUMNS, history, strict=True):
                part[column] = outcome
            parts.append(part)
    rows = pd.concat(parts, ignore_index=True)
    rows[REWARD] = rows[REWARD_LAG_COLUMNS[0]]
    rows["visit_context"] = "in_visit"
    rows["well_visit_id"] = pd.array(rows["well_visit_id"], dtype="Int64")
    for column, value in zip(("animal", "session_date", "epoch"), key, strict=True):
        rows.insert(0, column, value)
    return rows[["animal", "session_date", "epoch", *rows.columns[3:]]]


def as_cohort(rows: pd.DataFrame, kind: str) -> CohortRows:
    """Synthetic rows as the cohort step gives them (both models, no counts)."""
    return CohortRows(kind, {"A": rows, "D": rows}, {})


@pytest.fixture
def make_visit_rows():
    """:func:`synthetic_visit_rows`, for a test to call with its own seed."""
    return synthetic_visit_rows


@pytest.fixture
def cohort_of():
    """:func:`as_cohort`."""
    return as_cohort
