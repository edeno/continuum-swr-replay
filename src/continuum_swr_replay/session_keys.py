"""The natural key for one recording session: (animal, session_date, epoch).

This tuple identifies a session everywhere in the pipeline -- it is the join
key for every table and the group key for every per-session fit. Defining it
once here, backed by the covariate registry's row-key authority, keeps a rename
(as ``day`` -> ``session_date`` once was) a one-line change instead of a hunt
through twenty files.

The hippocampus event table names its date column ``day`` rather than
``session_date``; that column name is load-bearing for that grain, so
``SESSION_COLUMNS_DAY`` serves it without forcing a rename.
"""

from __future__ import annotations

import pandas as pd

# The canonical session columns, in join order. These are the first three
# segments of every table's registered row key
# (``covariate_registry._ROW_KEYS``); that equality is enforced by
# ``tests/unit/test_session_keys.py`` rather than imported here, because
# importing ``covariate_registry`` at module load would pull in the heavy
# ``analysis.replay`` package (tables -> covariates) and those modules import
# this one -- a cycle. A literal plus a test is the one definition without it.
SESSION_COLUMNS: tuple[str, ...] = ("animal", "session_date", "epoch")

# The hippocampus event table's date column is ``day``, not ``session_date``.
SESSION_COLUMNS_DAY: tuple[str, ...] = ("animal", "day", "epoch")

__all__ = [
    "SESSION_COLUMNS",
    "SESSION_COLUMNS_DAY",
    "session_key_set",
    "n_sessions",
    "session_key_frame",
]


def _require(frame: pd.DataFrame, columns: tuple[str, ...]) -> list[str]:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"frame missing session columns {missing}")
    return list(columns)


def session_key_set(
    frame: pd.DataFrame, *, columns: tuple[str, ...] = SESSION_COLUMNS
) -> set[tuple]:
    """The set of distinct session keys present in ``frame`` (order-independent)."""
    cols = _require(frame, columns)
    return set(frame.loc[:, cols].itertuples(index=False, name=None))


def n_sessions(
    frame: pd.DataFrame, *, columns: tuple[str, ...] = SESSION_COLUMNS
) -> int:
    """Number of distinct sessions in ``frame``."""
    return len(session_key_set(frame, columns=columns))


def session_key_frame(
    frame: pd.DataFrame, *, columns: tuple[str, ...] = SESSION_COLUMNS
) -> pd.DataFrame:
    """The distinct session-key rows of ``frame``, columns in canonical order."""
    cols = _require(frame, columns)
    return frame.loc[:, cols].drop_duplicates().reset_index(drop=True)
