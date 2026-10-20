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

from continuum_swr_replay.tables.covariate_registry import row_keys

# The canonical session columns, in join order: the first three segments of
# every table's registered row key (``covariate_registry._ROW_KEYS``).
SESSION_COLUMNS: tuple[str, ...] = row_keys(table="event")[:3]

# The hippocampus event table's date column is ``day``, not ``session_date``.
SESSION_COLUMNS_DAY: tuple[str, ...] = ("animal", "day", "epoch")

__all__ = [
    "SESSION_COLUMNS",
    "SESSION_COLUMNS_DAY",
    "session_key_set",
    "n_sessions",
    "session_key_frame",
    "subset_max_per_animal",
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


def subset_max_per_animal(
    frame: pd.DataFrame, n: int, *, sort_sessions: bool = False
) -> pd.DataFrame:
    """Keep only the rows of the first ``n`` sessions per animal (smoke mode).

    Parameters
    ----------
    frame : pd.DataFrame
        Rows carrying the ``SESSION_COLUMNS``.
    n : int
        Sessions kept per animal.
    sort_sessions : bool
        If False, an animal's first sessions are the first to appear in
        ``frame``'s row order; if True, the first in sorted session-key order.

    Returns
    -------
    pd.DataFrame
        The kept rows, in ``frame``'s order, with a fresh index.
    """
    keep = frame[list(SESSION_COLUMNS)].drop_duplicates()
    if sort_sessions:
        keep = keep.sort_values(list(SESSION_COLUMNS))
    keep = keep.groupby("animal", sort=False).head(n)
    return frame.merge(keep, on=list(SESSION_COLUMNS))
