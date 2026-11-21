"""Fail-closed guard on the ``--stop-only`` qvals bootstrap.

``prepare_replay_content_data.build_stop_table_scope`` writes straight to the
canonical ``replay_content`` root, outside the staged manifest gate. These
tests pin the way that can corrupt the atomic event/stop set: the bootstrap
downgrading a complete table to a split-less one.
"""

from __future__ import annotations

import pandas as pd
import pytest

from continuum_swr_replay.tables.io import write_table
from scripts.tables.prepare_replay_content_data import (
    _refuse_to_clobber_complete_stop_table,
)


def _complete_stop_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stop_id": [1, 2],
            "n_selected_events": [1, 0],
            "n_nonlocal_swr_events": [1, 0],
            "n_nonlocal_non_swr_events": [0, 0],
        }
    )


def _bootstrap_stop_table() -> pd.DataFrame:
    """What build_stop_table produces before the split is added."""
    return pd.DataFrame({"stop_id": [1, 2], "n_selected_events": [1, 0]})


def test_bootstrap_refuses_to_overwrite_a_complete_stop_table(tmp_path):
    write_table(_complete_stop_table(), tmp_path / "stop_table")

    with pytest.raises(FileExistsError, match="complete stop table"):
        _refuse_to_clobber_complete_stop_table(tmp_path)


def test_bootstrap_allows_writing_over_a_split_less_table(tmp_path):
    """A staging dir holding an earlier bootstrap (no split) is fine to reuse."""
    write_table(_bootstrap_stop_table(), tmp_path / "stop_table")

    _refuse_to_clobber_complete_stop_table(tmp_path)  # must not raise


def test_bootstrap_allows_writing_into_an_empty_dir(tmp_path):
    _refuse_to_clobber_complete_stop_table(tmp_path)  # must not raise
