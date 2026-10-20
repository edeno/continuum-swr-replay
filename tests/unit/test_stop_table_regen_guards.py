"""Fail-closed guards on the two stop-table writers that hit the prod root.

Both ``prepare_data.build_stop_table_scope`` (the ``--stop-only`` qvals
bootstrap) and ``scripts/tables/regenerate_stop_table.py`` write straight to
the canonical ``replay_content`` root, outside the staged manifest gate. These
tests pin the two ways that can corrupt the atomic event/stop set: the
bootstrap downgrading a complete table to a split-less one, and the
regeneration silently dropping a session so the stop table no longer covers the
event table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay.prepare_data import (
    _refuse_to_clobber_complete_stop_table,
)
from continuum_swr_replay.tables.io import write_table
from scripts.tables.regenerate_stop_table import _assert_matches_event_table


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


def _stop_and_event(session_keys, stop_ids):
    stop = pd.DataFrame(
        {
            "animal": [k[0] for k in session_keys],
            "session_date": [k[1] for k in session_keys],
            "epoch": [k[2] for k in session_keys],
            "stop_id": stop_ids,
        }
    )
    # Every event points at a stop; a stop with no events is stop-only.
    event = stop.iloc[:1].copy()
    return stop, event


def test_regen_parity_passes_when_scopes_match():
    keys = [("chimi", 20200212, "02_r1"), ("chimi", 20200212, "02_r1")]
    stop, event = _stop_and_event(keys, [10, 11])
    _assert_matches_event_table(stop, event)  # must not raise


def test_regen_parity_rejects_a_missing_session():
    stop = pd.DataFrame(
        {
            "animal": ["chimi"],
            "session_date": [20200212],
            "epoch": ["02_r1"],
            "stop_id": [10],
        }
    )
    event = pd.DataFrame(
        {
            "animal": ["chimi", "j16"],
            "session_date": [20200212, 20210701],
            "epoch": ["02_r1", "04_r2"],
            "stop_id": [10, 99],
        }
    )

    with pytest.raises(ValueError, match="does not match event_table"):
        _assert_matches_event_table(stop, event)


def test_regen_parity_rejects_an_orphan_event_stop_id():
    """Same sessions, but an event references a stop_id the rebuild dropped."""
    stop = pd.DataFrame(
        {
            "animal": ["chimi"],
            "session_date": [20200212],
            "epoch": ["02_r1"],
            "stop_id": [10],
        }
    )
    event = pd.DataFrame(
        {
            "animal": ["chimi"],
            "session_date": [20200212],
            "epoch": ["02_r1"],
            "stop_id": [999],
        }
    )

    with pytest.raises(ValueError, match="absent from the rebuilt"):
        _assert_matches_event_table(stop, event)


def test_regen_parity_rejects_duplicate_stop_id():
    stop = pd.DataFrame(
        {
            "animal": ["chimi", "chimi"],
            "session_date": [20200212, 20200212],
            "epoch": ["02_r1", "02_r1"],
            "stop_id": [10, 10],
        }
    )
    event = stop.iloc[:1].copy()

    with pytest.raises(ValueError, match="unique nonmissing stop_id"):
        _assert_matches_event_table(stop, event)


def test_regen_parity_ignores_nan_event_stop_ids():
    """event_table stop_id has no nulls in practice, but guard tolerates them."""
    stop = pd.DataFrame(
        {
            "animal": ["chimi"],
            "session_date": [20200212],
            "epoch": ["02_r1"],
            "stop_id": [10],
        }
    )
    event = pd.DataFrame(
        {
            "animal": ["chimi", "chimi"],
            "session_date": [20200212, 20200212],
            "epoch": ["02_r1", "02_r1"],
            "stop_id": [10, np.nan],
        }
    )

    _assert_matches_event_table(stop, event)  # NaN dropped, 10 present -> OK
