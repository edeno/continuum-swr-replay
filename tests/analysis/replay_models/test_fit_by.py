"""The one declaration of the units the behavior models are fitted at."""

from __future__ import annotations

from typing import get_args

from continuum_swr_replay.analysis.replay_models.fit_by import BEHAVIOR_FIT_BY
from continuum_swr_replay.tables.session_keys import FitBy


def test_behavior_units_are_day_and_animal():
    """Decided by the lead, 2026-11-24 09:55: both the day and the animal
    unit, in that order; each a value of ``FitBy``."""
    assert BEHAVIOR_FIT_BY == ("day", "animal")
    assert all(fit_by in get_args(FitBy) for fit_by in BEHAVIOR_FIT_BY)
