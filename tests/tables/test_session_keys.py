"""The session identity has one home, sourced from the registry row keys."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from continuum_swr_replay.tables.covariate_registry import row_keys
from continuum_swr_replay.tables.session_keys import (
    SESSION_COLUMNS,
    SESSION_COLUMNS_DAY,
    n_sessions,
    session_key_frame,
    session_key_set,
    subset_max_per_animal,
)


def test_session_columns_sourced_from_registry():
    assert SESSION_COLUMNS == row_keys(table="event")[:3]
    assert SESSION_COLUMNS == ("animal", "session_date", "epoch")


def test_day_variant_for_hippocampus_grain():
    assert SESSION_COLUMNS_DAY == ("animal", "day", "epoch")


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "animal": ["a", "a", "b"],
            "session_date": [1, 1, 2],
            "epoch": ["e", "e", "e"],
        }
    )


def test_helper_return_types():
    df = _frame()
    assert isinstance(session_key_set(df), set)
    assert session_key_set(df) == {("a", 1, "e"), ("b", 2, "e")}
    assert isinstance(n_sessions(df), int)
    assert n_sessions(df) == 2
    out = session_key_frame(df)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == list(SESSION_COLUMNS)
    assert len(out) == 2


def test_missing_columns_raise():
    with pytest.raises(KeyError, match="session columns"):
        session_key_set(pd.DataFrame({"animal": ["a"]}))


_MODULE_LEVEL_SESSION_CONST = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(\s*:\s*[^=]+?)?\s*=\s*[\[\(]\s*"
    r'"animal"\s*,\s*"(session_date|day)"\s*,\s*"epoch"\s*[\]\)]\s*$'
)


def test_no_module_level_session_constant_defined_elsewhere():
    """The session tuple has one module-level definition (session_keys), keyed
    once in covariate_registry._ROW_KEYS. No other module may define a top-level
    constant equal to the bare 3-element session tuple.

    Scope: module-level constants (column 0) only. Inline local variables,
    function-default arguments, and composite keys that append event columns are
    usages, not definitions, and are out of scope for this check.
    """
    repo = Path(__file__).resolve().parents[2]
    offenders = []
    for directory in ("src", "scripts"):
        for path in (repo / directory).rglob("*.py"):
            if path.name in {"session_keys.py", "covariate_registry.py"}:
                continue
            if "figures/archive" in str(path):
                continue
            for number, line in enumerate(path.read_text().splitlines(), start=1):
                if _MODULE_LEVEL_SESSION_CONST.match(line):
                    offenders.append(f"{path.relative_to(repo)}:{number}")
    assert not offenders, (
        "module-level session constant defined outside session_keys.py: "
        + ", ".join(offenders)
    )


_LOCAL_SESSION_ALIAS = re.compile(
    r"^_?SESSION_COL(S|UMNS)(_DAY)?\s*(:\s*[^=]+?)?\s*=\s*\S"
)


def test_no_local_session_key_definitions():
    """No module may bind its own name for the session key.

    ``test_no_module_level_session_constant_defined_elsewhere`` catches a
    *retyped* tuple; this catches the softer copy -- a module-level
    ``SESSION_COLS = list(SESSION_COLUMNS)`` alias. Fifteen modules carried one,
    which meant fifteen names for one concept and fifteen places a reader had to
    check that the alias really was the canonical key. Import
    ``SESSION_COLUMNS`` / ``SESSION_COLUMNS_DAY`` and use it under its own name.
    """
    repo = Path(__file__).resolve().parents[2]
    offenders = []
    for directory in ("src", "scripts"):
        for path in (repo / directory).rglob("*.py"):
            if path.name == "session_keys.py":
                continue
            for number, line in enumerate(path.read_text().splitlines(), start=1):
                if _LOCAL_SESSION_ALIAS.match(line):
                    offenders.append(f"{path.relative_to(repo)}:{number}")
    assert not offenders, (
        "local session-key alias defined outside session_keys.py: "
        + ", ".join(offenders)
    )


def test_session_key_matches_columns_day():
    """The event builder's ``_SessionKey`` fields mirror ``SESSION_COLUMNS_DAY``.

    ``_SessionKey`` is a local scalar record (the package has no shared one),
    so this pins its field names/order to the canonical day-grain column tuple;
    a future ``day``->``session_date`` rename that missed it would fail here.
    """
    from continuum_swr_replay.analysis.behavior.tables.tables_event import _SessionKey

    assert _SessionKey._fields == SESSION_COLUMNS_DAY


def test_subset_max_per_animal_keeps_each_animal_s_first_sessions():
    """Unsorted, an animal's first sessions are those that appear first in the
    frame; sorted, the first by session key. The kept rows stay in the frame's
    order, every row of a kept session is kept, and an animal with fewer
    sessions than the cap keeps them all."""
    frame = pd.DataFrame(
        {
            "animal": ["b", "a", "a", "b", "a", "b", "c"],
            "session_date": [2, 3, 1, 1, 3, 2, 5],
            "epoch": ["02_r1"] * 7,
            "row": [0, 1, 2, 3, 4, 5, 6],
        }
    )

    unsorted = subset_max_per_animal(frame, 1)
    assert unsorted["row"].tolist() == [0, 1, 4, 5, 6]
    assert unsorted.index.tolist() == [0, 1, 2, 3, 4]

    by_key = subset_max_per_animal(frame, 1, sort_sessions=True)
    assert by_key["row"].tolist() == [2, 3, 6]
