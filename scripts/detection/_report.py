"""What the two detection-comparison reports share.

``report_inventory_comparison.py`` and ``report_event_features.py``: the pending
marking of the memoryless inventories, the caption and quantile wording,
and reading a run's tables. Sessions are pooled within an animal by
``analysis.inventory_comparison.pooling``.

Not a script. The reports import it as ``scripts.detection._report``, after
putting the repo root on ``sys.path``.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.inventory_comparison.pooling import (
    session_names,
)
from continuum_swr_replay.analysis.inventory_comparison.schema import (
    MEMORYLESS,
    MEMORYLESS_CRITERIA,
    typed_table,
)
from continuum_swr_replay.runs.io import ok_table_paths, read_manifest
from continuum_swr_replay.tables.io import read_table
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS_DAY

_SESSION = list(SESSION_COLUMNS_DAY)

PENDING = "pending the Phase 2 pilot checks"
PENDING_DETECTORS = frozenset({MEMORYLESS, MEMORYLESS_CRITERIA})
PENDING_SENTENCE = (
    "Every row or column with a memoryless or memoryless_criteria inventory is "
    f"marked {PENDING} (in tables, the pending column; in figures, †)."
)
QUANTILES: Mapping[str, float] = MappingProxyType(
    {"q10": 0.1, "q25": 0.25, "median": 0.5, "q75": 0.75, "q90": 0.9}
)
QUANTILE_COLUMNS = (
    "Each distribution is given as its count of finite values (_n), its 10th, "
    "25th, 50th, 75th and 90th percentiles and its maximum."
)


def caption_with_pending(*sentences: str) -> str:
    """The sentences as one caption, closed by :data:`PENDING_SENTENCE`."""
    return " ".join((*sentences, PENDING_SENTENCE))


def read_run(
    run_dir: Path, table_dtypes: Mapping[str, Mapping[str, object]]
) -> tuple[dict, dict[str, pd.DataFrame]]:
    """The run's manifest and every table of its ``ok`` epochs, concatenated.

    The manifest and its ``ok`` epochs' tables are read by
    ``runs.io.read_manifest`` and ``runs.io.ok_table_paths``.

    Parameters
    ----------
    run_dir : Path
    table_dtypes : mapping
        Each table's columns and dtypes, as the runner that wrote the run
        declares them (``run_inventory_comparison.TABLE_DTYPES``, say).

    Returns
    -------
    manifest : dict
    tables : dict of str to DataFrame
        By name, with the runner's columns and dtypes.
    """
    manifest = read_manifest(run_dir)
    tables = {
        name: typed_table(map(read_table, ok_table_paths(run_dir, name)), dtypes)
        for name, dtypes in table_dtypes.items()
    }
    return manifest, tables


def with_pending(
    frame: pd.DataFrame,
    columns: Sequence[str] = (),
    groups: Mapping[str, Mapping[str, Collection[str]]] = MappingProxyType({}),
) -> pd.DataFrame:
    """``frame`` with ``pending``: :data:`PENDING` on rows a memoryless
    inventory enters, else empty.

    Parameters
    ----------
    frame : DataFrame
    columns : sequence of str
        Columns naming a detector or cohort.
    groups : mapping
        Column -> {value: the detectors it stands for}, for a column naming a
        set of detectors, such as a comparison or a cell.
    """
    marked = np.zeros(len(frame), dtype=bool)
    for column in columns:
        marked |= frame[column].isin(PENDING_DETECTORS).to_numpy()
    for column, members in groups.items():
        pending = {
            value
            for value, detectors in members.items()
            if PENDING_DETECTORS & set(detectors)
        }
        marked |= frame[column].isin(pending).to_numpy()
    return frame.assign(pending=np.where(marked, PENDING, ""))


def per_session(frame: pd.DataFrame) -> pd.DataFrame:
    """``frame`` with each row's session name after the session columns."""
    return frame.assign(session=session_names(frame))[
        [*_SESSION, "session", *(c for c in frame.columns if c not in _SESSION)]
    ]
