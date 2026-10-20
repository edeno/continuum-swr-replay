"""The PFC analyses' entry point to the shared well-anchored stop cohort.

The hippocampal rate and content families are read on one population -- the
stops the animal spends at a well (``analysis/replay/cohort_selection.py``,
``select_stop_cohort``). The PFC unit GLMs describe the same events and so must
be read on the same population; this module is the one place that loads it for a
single session, so no PFC script re-derives "at a well" from position.

The session key is the canonical ``(animal, session_date, epoch)`` triple, which
the PFC runners hold as ``nwb_file_name`` / ``epoch_name`` instead; the
translation is done here once.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from continuum_swr_replay.analysis.replay.cohort_selection import select_stop_cohort
from continuum_swr_replay.data_loaders.utils import parse_nwb_filename
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR
from continuum_swr_replay.tables.io import parquet_path, read_table, sha256_file

__all__ = ["DEFAULT_STOP_TABLE_PATH", "load_session_stop_cohort"]

#: The canonical stop table every family's cohort is cut from.
DEFAULT_STOP_TABLE_PATH = PROCESSED_DATA_DIR / "replay_content" / "stop_table.parquet"


def load_session_stop_cohort(
    nwb_file_name: str,
    epoch_name: str,
    *,
    cohort: str = "at_well",
    stop_table_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load one session's stops and restrict them to ``cohort``.

    Parameters
    ----------
    nwb_file_name : str
        NWB file name, e.g. ``"chimi20200212_.nwb"``.
    epoch_name : str
        Epoch name, e.g. ``"02_r1"``.
    cohort : {"at_well", "all_stops", "in_transit"}
        Population to keep, passed through to
        :func:`~continuum_swr_replay.analysis.replay.cohort_selection.select_stop_cohort`.
        ``at_well`` is canonical and, at the stop grain, means *well-anchored*:
        ``occupancy_state`` of ``at_well`` or ``mixed``.
    stop_table_path : str or Path, optional
        Stop table to read; defaults to :data:`DEFAULT_STOP_TABLE_PATH`.

    Returns
    -------
    stops : pandas.DataFrame
        The session's kept stops, with a fresh integer index. ``stop_start`` and
        ``stop_end`` are absolute seconds on ``position_info.index``'s clock.
    report : dict
        ``select_stop_cohort``'s report, plus the session key
        (``animal``/``session_date``/``epoch``), the ``stop_table`` path read and
        its ``stop_table_sha256``. Suitable for a provenance sidecar as-is.

    Raises
    ------
    FileNotFoundError
        If no stop table exists at ``stop_table_path``.
    KeyError
        If the table lacks ``occupancy_state`` or the session columns.
    ValueError
        If the session has no stops at all -- a missing session is a data
        problem, not an empty cohort, and fitting the whole epoch instead would
        silently reintroduce the population this restriction exists to remove.
    """
    path = DEFAULT_STOP_TABLE_PATH if stop_table_path is None else Path(stop_table_path)
    animal, session_date = parse_nwb_filename(nwb_file_name)

    stop_table = read_table(path)
    missing = [
        column
        for column in ("animal", "session_date", "epoch")
        if column not in stop_table.columns
    ]
    if missing:
        raise KeyError(f"stop table at {path} is missing session column(s) {missing}")

    is_session = (
        (stop_table["animal"] == animal)
        & (stop_table["session_date"] == session_date)
        & (stop_table["epoch"] == epoch_name)
    )
    session_stops = stop_table.loc[is_session]
    if session_stops.empty:
        raise ValueError(
            f"no stops for ({animal}, {session_date}, {epoch_name}) in {path}; "
            "the session is absent from the canonical stop table, so its stop "
            "cohort cannot be resolved"
        )

    stops, report = select_stop_cohort(session_stops, cohort=cohort)
    resolved = parquet_path(path) if parquet_path(path).exists() else path
    report = {
        **report,
        "animal": animal,
        "session_date": int(session_date),
        "epoch": epoch_name,
        "stop_table": str(resolved),
        "stop_table_sha256": sha256_file(resolved),
    }
    return stops, report
