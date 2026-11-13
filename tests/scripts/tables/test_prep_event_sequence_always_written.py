"""``prepare_replay_content_data.run_prepare_data`` always writes
``event_sequence_table``, even when no session contributed sequence rows.

The retrospective-alignment pass and ``write_regeneration_manifest.py`` read
``event_sequence_table`` unconditionally, so a run in which no session had a
posterior must still publish it: an empty frame with the canonical columns of
:data:`_EVENT_SEQUENCE_TABLE_COLUMNS`, which the retrospective helpers treat as
a legitimate input.
"""

from __future__ import annotations

import argparse
import inspect

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_tables import prepare_data
from continuum_swr_replay.analysis.event_tables._common import (
    _EVENT_SEQUENCE_TABLE_COLUMNS,
)
from continuum_swr_replay.tables.io import read_table, write_table
from scripts.tables import prepare_replay_content_data as prepare_cli

SESSION = {"animal": "test", "session_date": 20200101, "epoch": "01_r1"}


def _session_tables() -> tuple[list, list, list, list]:
    """One session's event, event-arm and stop tables, and no sequence rows.

    One non-local event in one stop, with the nine arm rows the patch
    projection needs (arms 0, 1, 6 are patch 1; 2, 3, 7 patch 2; 4, 5, 8
    patch 3).
    """
    events = pd.DataFrame(
        [
            {
                **SESSION,
                "event_id": 1,
                "stop_id": 1,
                "n_nonlocal_bins": 4,
                "n_undecoded_bins": 4,
                "last_rewarded_arm": np.nan,
                "prev_patch": np.nan,
                "did_last_completed_trip_switch_patch": 0,
                "is_swr": False,
            }
        ]
    )
    arms = range(9)
    event_arms = pd.DataFrame(
        {
            "event_id": 1,
            "candidate_arm": list(arms),
            "arm_count_hard": 0.0,
            "is_same_patch_as_current": 0,
            "reward_count_candidate_arm_last10_known_trials": np.nan,
            "visit_recency_trials": np.nan,
            "is_candidate_arm_never_visited": np.nan,
        }
    )
    stops = pd.DataFrame([{**SESSION, "stop_id": 1, "n_selected_events": 1}])
    return [events], [event_arms], [stops], []


@pytest.fixture
def published(monkeypatch, tmp_path):
    """Run ``run_prepare_data`` on that session and return its output folder.

    The Spyglass-backed steps (finding sessions, reading the contingency
    table, building each session, the pooled standardization) are replaced;
    the assembly and the publish loop run as written. Tables are written with
    ``write_table``, without the canonical-schema validation the one-event
    tables here could not pass.
    """
    monkeypatch.setattr(
        prepare_cli, "find_sessions", lambda max_per_animal: [{"session_id": "x"}]
    )
    monkeypatch.setattr(prepare_cli, "load_contingency", lambda path: None)
    monkeypatch.setattr(
        prepare_cli,
        "_build_all_session_tables",
        lambda sessions, contingency, *, from_cache, t0: _session_tables(),
    )
    monkeypatch.setattr(
        prepare_cli,
        "standardize_event_arm",
        lambda table, contingency, reference_path: table,
    )
    monkeypatch.setattr(
        inspect.getmodule(prepare_cli.publish_replay_tables),
        "publish_swr_table",
        lambda frame, path, *, table, basename: write_table(frame, path / basename),
    )
    # run_prepare_data points the behavioral cache at the output folder.
    monkeypatch.setattr(prepare_data, "SESSION_CACHE_DIR", None)
    args = argparse.Namespace(
        output_dir=tmp_path,
        behavior_cache_dir=None,
        contingency_dir=None,
        test=False,
        max_per_animal=None,
        animal=None,
        from_cache=True,
        cache_only=False,
        stop_only=False,
    )
    prepare_cli.run_prepare_data(args)
    return tmp_path


def test_an_empty_event_sequence_table_is_published(published):
    sequence = read_table(published / "event_sequence_table")

    assert len(sequence) == 0
    assert list(sequence.columns) == list(_EVENT_SEQUENCE_TABLE_COLUMNS)


def test_the_other_tables_are_published_beside_it(published):
    for name in ("event_table", "event_arm_table", "event_patch_table", "stop_table"):
        assert len(read_table(published / name)) > 0, name


def test_empty_canonical_frame_roundtrips_through_storage(tmp_path):
    """An empty frame with the canonical schema survives write_table ->
    read_table with its column order intact: what the retrospective pass
    receives when no session contributed sequences."""
    cols = list(_EVENT_SEQUENCE_TABLE_COLUMNS)
    df = pd.DataFrame(columns=cols)
    path = write_table(df, tmp_path / "event_sequence_table")
    loaded = read_table(path)
    assert list(loaded.columns) == cols
    assert len(loaded) == 0
    for required in ("event_id", "decoded_arm_map", "nonlocal_prob"):
        assert required in loaded.columns
