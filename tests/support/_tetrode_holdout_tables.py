"""A minimal table of the held-out tetrode run, with every column the runner
writes (``tetrode_holdout.RUN_COLUMNS``), for the tests of its readers."""

from __future__ import annotations

import pandas as pd

from continuum_swr_replay.analysis.event_evidence.tetrode_holdout import RUN_COLUMNS

#: The runner's keys, the last columns of :data:`RUN_COLUMNS`; the columns
#: before them are ``holdout_scores``'.
RUNNER_KEYS = ("event_id", "kind", "group", "epoch_id", "animal")

SCORE_COLUMNS = RUN_COLUMNS[: -len(RUNNER_KEYS)]
"""The columns ``holdout_scores`` returns, in order."""

_TEXT = {"kind": "event", "group": "no SWR", "epoch_id": "e", "animal": "a"}
_INTEGER = {
    "window_id",
    "tetrode",
    "n_bins",
    "n_scored_bins",
    "n_spikes",
    "n_information_spikes",
    "n_null_bins",
    "n_null_spikes",
    "null_window_id",
    "event_id",
}


def run_table(**columns) -> pd.DataFrame:
    """A table with every column of :data:`RUN_COLUMNS`, in order, then any
    other column given: the given values (a scalar or one value per row), the
    rest a placeholder (``kind`` ``"event"``, ``group`` ``"no SWR"``,
    ``epoch_id`` ``"e"``, ``animal`` ``"a"``, 0 for a count or an id, 0.0 for a
    score). The rows are as many as the longest given list (one without)."""
    n_rows = max(
        (len(v) for v in columns.values() if not isinstance(v, (str, int, float))),
        default=1,
    )
    defaults = {
        name: _TEXT.get(name, 0 if name in _INTEGER else 0.0) for name in RUN_COLUMNS
    }
    values = {**defaults, **columns}
    ordered = [*RUN_COLUMNS, *(c for c in columns if c not in RUN_COLUMNS)]
    return pd.DataFrame(
        {name: values[name] for name in ordered}, index=pd.RangeIndex(n_rows)
    )
