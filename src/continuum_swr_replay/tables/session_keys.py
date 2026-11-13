"""The natural key for one recording session: (animal, session_date, epoch).

This tuple identifies a session everywhere in the pipeline -- it is the join
key for every table and the group key for every per-session fit. Defining it
once here, backed by the covariate registry's row-key authority, keeps a rename
(as ``day`` -> ``session_date`` once was) a one-line change instead of a hunt
through twenty files.

The hippocampus event table names its date column ``day`` rather than
``session_date``; that column name is load-bearing for that grain, so
``SESSION_COLUMNS_DAY`` serves it without forcing a rename.

The table builders and the analyses that group by an id share three helpers
here: :func:`present_session_columns` (the session columns a frame carries),
:func:`require_session_unique_ids` (an id names one session) and
:func:`one_session` (the one epoch a per-epoch frame holds).
"""

from __future__ import annotations

import pandas as pd

from continuum_swr_replay.tables.columns import require_columns
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
    "one_session",
    "present_session_columns",
    "require_session_unique_ids",
]


def session_key_set(
    frame: pd.DataFrame, *, columns: tuple[str, ...] = SESSION_COLUMNS
) -> set[tuple]:
    """The set of distinct session keys present in ``frame`` (order-independent)."""
    require_columns(frame, columns, "the frame")
    return set(frame.loc[:, list(columns)].itertuples(index=False, name=None))


def n_sessions(
    frame: pd.DataFrame, *, columns: tuple[str, ...] = SESSION_COLUMNS
) -> int:
    """Number of distinct sessions in ``frame``."""
    return len(session_key_set(frame, columns=columns))


def session_key_frame(
    frame: pd.DataFrame, *, columns: tuple[str, ...] = SESSION_COLUMNS
) -> pd.DataFrame:
    """The distinct session-key rows of ``frame``, columns in canonical order."""
    require_columns(frame, columns, "the frame")
    return frame.loc[:, list(columns)].drop_duplicates().reset_index(drop=True)


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


def one_session(frame: pd.DataFrame, what: str) -> tuple:
    """The ``(animal, day, epoch)`` of a frame that holds one epoch.

    Parameters
    ----------
    frame : pd.DataFrame
        A per-epoch frame (a decode frame) carrying :data:`SESSION_COLUMNS_DAY`.
    what : str
        What reads the frame, for the error.

    Returns
    -------
    tuple
        The frame's one ``(animal, day, epoch)``, as stored.

    Raises
    ------
    ValueError
        If the frame holds rows of more than one epoch, or no row.
    """
    sessions = frame[list(SESSION_COLUMNS_DAY)].drop_duplicates()
    if len(sessions) != 1:
        raise ValueError(f"{what} reads one epoch at a time, not {len(sessions)}")
    return tuple(sessions.iloc[0])


def present_session_columns(df: pd.DataFrame) -> list[str]:
    """Session-key columns present in ``df``."""
    return [c for c in SESSION_COLUMNS if c in df.columns]


def require_session_unique_ids(
    df: pd.DataFrame, id_col: str, *, helper_name: str
) -> None:
    """Raise if ``id_col`` values are not unique across sessions.

    The canonical hashed event_id / stop_id columns mix session keys
    into the hash, so each ID maps to exactly one
    ``(animal, day, epoch)`` tuple. Helpers that do per-ID groupbys
    (softmax / mean / sum) silently produce wrong results — fractions
    summing above 1, dropped stops, cross-session averages — when
    that invariant is violated. This guard catches session-local IDs
    early with a clear message instead.

    No-op when the session columns aren't present (e.g. tests that
    pass minimal frames without animal/day/epoch).
    """
    missing = [c for c in SESSION_COLUMNS if c not in df.columns]
    if missing or id_col not in df.columns:
        return
    per_id_session_count = (
        df.groupby(id_col)[list(SESSION_COLUMNS)].nunique().max(axis=1)
    )
    bad = per_id_session_count[per_id_session_count > 1]
    if len(bad):
        sample = bad.index[: min(3, len(bad))].tolist()
        raise ValueError(
            f"{helper_name}: {id_col!r} must be globally unique across "
            f"sessions, but {len(bad)} value(s) appear in multiple "
            f"(animal, day, epoch) tuples (e.g. {sample}). Use the "
            "canonical hashed IDs from "
            "``tables.event_ids.hash_event_id`` / "
            "``hash_stop_id`` (or the ContFrag equivalents) before "
            "calling this helper."
        )
