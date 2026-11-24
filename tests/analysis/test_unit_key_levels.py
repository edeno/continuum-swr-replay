"""shared_statistics repeats the number of key levels of each unit of the fit,
because it imports nothing from the package; it must agree with the one
declaration in tables.session_keys."""

from __future__ import annotations

from typing import get_args

from continuum_swr_replay.analysis import shared_statistics
from continuum_swr_replay.tables.session_keys import FIT_BY_COLUMNS, FitBy


def test_unit_key_levels_agree_with_fit_by_columns():
    levels = shared_statistics._UNIT_KEY_LEVELS
    assert set(levels) == set(FIT_BY_COLUMNS) == set(get_args(FitBy))
    for fit_by, columns in FIT_BY_COLUMNS.items():
        assert levels[fit_by] == len(columns)
