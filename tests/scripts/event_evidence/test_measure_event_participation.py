"""The participation run to its manifest on a tiny synthetic epoch: every event has
its row, each control lies inside one well visit, the events left without one are
counted per group, and the settings and inputs record the subset, the control rule,
the event table and the session-cache entries. The decode frame, the spikes and the
visits are stubbed; the window rule itself is tested in
``tests/analysis/event_evidence/test_participation.py``."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_definition.cohort_selection import (
    ANALYZED_SUBSET,
)
from continuum_swr_replay.analysis.event_evidence.participation import CONTROL_RULE
from continuum_swr_replay.paths import (
    EVENT_PARTICIPATION_DIR,
    processed_data_dir,
    replay_content_dir,
    table_path,
)
from continuum_swr_replay.runs.io import files_sha256, sha256_file
from scripts.event_evidence import _per_event_run
from scripts.event_evidence import measure_event_participation as script
from tests.support._session_cache_fixtures import write_session_cache_entries

EPOCH = "j1620210706_02_r1_HPC"
TIME = np.round(np.arange(500) * 0.002, 6)
VISITS = (np.array([0.1]), np.array([0.6]))
"""The epoch's one well visit: bins 50 to 299."""


def _table() -> pd.DataFrame:
    """Two short events inside the visit and a 51-bin one on the track after it."""
    return pd.DataFrame(
        {
            "animal": "j16",
            "session_date": 20210706,
            "epoch": "02_r1",
            "event_id": pd.Series([1, 2, 3], dtype="uint64"),
            "event_start_s": [0.2, 0.45, 0.8],
            "event_end_s": [0.21, 0.46, 0.9],
            "is_swr": [True, False, False],
            "visit_context": ["in_visit", "in_visit", "track_center"],
        }
    )


def _frame() -> pd.DataFrame:
    """Local everywhere; the events' bins (100-105, 225-230, 400-450) in an event.

    With the 0.1 s margin, Local rest is bins 0-49, 156-174 and 281-349: inside
    the visit the longest stretch is 19 bins, and the only one of 51 bins (281-349)
    runs out of the visit at bin 300.
    """
    in_event = np.zeros(TIME.size, dtype=int)
    for first, last in ((100, 105), (225, 230), (400, 450)):
        in_event[first : last + 1] = 1
    return pd.DataFrame(
        {
            "Local_prob": 0.95,
            "SWR_event": 0,
            "non_local_event": in_event,
        },
        index=pd.Index(TIME, name="time"),
    )


@pytest.fixture
def epoch_inputs(processed_data_root, monkeypatch):
    """The event table, the epoch's cache entry, and the stubbed reads."""
    path = table_path("event_table", replay_content_dir())
    path.parent.mkdir(parents=True, exist_ok=True)
    _table().to_parquet(path)
    monkeypatch.setattr(script, "read_swr_table", lambda path: _table())
    monkeypatch.setattr(script, "read_dot_with_swr_source", lambda path: _frame())
    monkeypatch.setattr(
        script, "immobile_analyzed_bins", lambda frame: np.ones(len(frame), bool)
    )
    monkeypatch.setattr(script, "well_visit_times", lambda identifier: VISITS)
    monkeypatch.setattr(
        script, "load_spike_times", lambda identifier: [TIME[::3], TIME[1::7]]
    )
    monkeypatch.setattr(_per_event_run, "ProcessPoolExecutor", ThreadPoolExecutor)
    return sha256_file(path), write_session_cache_entries([EPOCH])


def test_the_controls_lie_inside_a_visit_and_the_events_without_one_are_counted(
    epoch_inputs,
):
    event_table_sha256, entries = epoch_inputs
    script.main([])
    folder = processed_data_dir() / EVENT_PARTICIPATION_DIR
    table = pd.read_parquet(folder / f"{EPOCH}_participation.parquet")
    # every event has its row, whatever its label; the events come first
    assert table["kind"].tolist() == ["event"] * 3 + ["control"] * 2
    assert table["event_id"].tolist()[:3] == [1, 2, 3]
    controls = table[table["kind"] == "control"]
    assert controls["event_id"].tolist() == [1, 2]
    assert controls["group"].tolist() == ["SWR", "no SWR"]
    assert (controls["n_bins"] == 6).all()
    # each control lies inside the visit, in the 19-bin stretch of rest there
    first = np.searchsorted(TIME, controls["start_s"].to_numpy())
    last = first + controls["n_bins"].to_numpy() - 1
    assert ((first >= 156) & (last <= 174) | (first >= 281) & (last <= 299)).all()

    manifest = json.loads((folder / "manifest.json").read_text())
    record = manifest["epochs"][EPOCH]
    # the 51-bin event has no window inside the visit; Local rest anywhere had one
    assert record["n_without_control"] == {"SWR": 0, "no SWR": 1}
    assert record["n_without_control_any_local_rest"] == {"SWR": 0, "no SWR": 0}
    assert manifest["settings"]["subset"] == ANALYZED_SUBSET
    assert manifest["settings"]["control_rule"] == CONTROL_RULE
    assert manifest["inputs"]["event_table"]["sha256"] == event_table_sha256
    assert manifest["inputs"]["session_cache"]["sha256"] == files_sha256(entries)


def test_a_run_is_repeated_when_the_visits_change(epoch_inputs, monkeypatch):
    _, entries = epoch_inputs
    script.main([])
    folder = processed_data_dir() / EVENT_PARTICIPATION_DIR
    written = (folder / f"{EPOCH}_participation.parquet").read_bytes()
    calls = []
    monkeypatch.setattr(
        script,
        "epoch_table",
        lambda *args: calls.append(args) or (pd.DataFrame({"kind": []}), {}),
    )
    script.main([])
    assert calls == []  # the same table and visits: the epoch is reused
    assert (folder / f"{EPOCH}_participation.parquet").read_bytes() == written
    entries[0].write_bytes(b"other visits")
    script.main([])
    assert len(calls) == 1
