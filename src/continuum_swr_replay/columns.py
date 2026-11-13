"""The one check that a table carries the columns a step reads.

A foundation module that imports nothing from the package, so every layer can
call it, the loaders included; ``tables.columns`` re-exports it.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def require_columns(frame: pd.DataFrame, columns: Iterable[str], table: str) -> None:
    """Refuse a table that lacks a column the caller reads.

    Parameters
    ----------
    frame : pd.DataFrame
        The table.
    columns : iterable of str
        The columns the caller reads.
    table : str
        The table's name, for the message.

    Raises
    ------
    ValueError
        If ``frame`` lacks any of ``columns``; the message names ``table`` and
        the missing columns, sorted.
    """
    missing = sorted(column for column in set(columns) if column not in frame)
    if missing:
        raise ValueError(f"{table} lacks required column(s) {missing}")
