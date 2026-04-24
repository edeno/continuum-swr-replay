"""Tests for analysis/pfc/load_population.py.

The historical ``df["animal"] = df["nwb_file_name"].str.split("2").str[0]``
returned ``"j1"`` for j16 sessions (e.g. ``j1620210706_.nwb``) because
the date YYYYMMDD that follows the animal name often contains a ``"2"``
early on. Pin the parse_nwb_filename-based replacement.
"""
from __future__ import annotations

import pytest

from continuum_swr_replay.analysis.pfc.load_population import _extract_animal


@pytest.mark.parametrize(
    "nwb_file_name,expected",
    [
        ("chimi20200212_.nwb", "chimi"),
        ("peanut20201125_.nwb", "peanut"),
        ("wilbur20210329_.nwb", "wilbur"),
        ("senor20201018_.nwb", "senor"),
        # Critical case: j16 session — the date 20210706 contains "2"
        # at position 1, so the legacy str.split("2")[0] returned "j1".
        ("j1620210706_.nwb", "j16"),
    ],
)
def test_extract_animal(nwb_file_name, expected):
    assert _extract_animal(nwb_file_name) == expected
