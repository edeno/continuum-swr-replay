"""Synthetic visit bins for the visit rate model's tests.

``visit_rate_simulation.synthetic_visit_rows`` lays the in-visit at-risk bins
of one session as the cohort step returns them (``visit_rate.cohort_rows``):
visits of random length, each one or two immobile stretches (with a 0.3 s
movement between them), bins of ``bin_s`` from each stretch's start, the three
clocks at the bin center, and an outcome history with ten known outcomes in
every bin. The counts are zero; a test draws them from a scenario's expected
counts.
"""

from __future__ import annotations

import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay_models.rate.visit_rate import (
    CohortRows,
    model_d_rows,
)
from continuum_swr_replay.analysis.replay_models.rate.visit_rate_simulation import (
    synthetic_visit_rows,
)

__all__ = ["as_cohort", "synthetic_visit_rows"]


def as_cohort(rows: pd.DataFrame, kind: str) -> CohortRows:
    """Synthetic rows as the cohort step gives them: model A's are the rows,
    model D's are ``model_d_rows`` of them, with its counts."""
    rows_d, counts = model_d_rows(rows)
    return CohortRows(kind, {"A": rows, "D": rows_d}, counts)


@pytest.fixture
def make_visit_rows():
    """:func:`synthetic_visit_rows`, for a test to call with its own seed."""
    return synthetic_visit_rows


@pytest.fixture
def cohort_of():
    """:func:`as_cohort`."""
    return as_cohort
