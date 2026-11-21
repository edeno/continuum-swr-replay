"""Tests for the event-onset table builder (``timebin_tables``).

Covers the sparse per-onset table (:func:`build_event_onset_table`) on a
synthetic session with two immobility bouts separated by a movement gap:

* the onsets of each event type are counted,
* ``t_since_pump`` carries a pump across the movement gap into the next bout,
* ``isi_within_stop`` resets at the stop boundary (first onset of a bout is NaN),
* an absent event column (older decodes without ``MUA_event``) is tolerated,
* a session with no immobility returns the empty schema.

It also covers ``scripts/tables/build_timebin_tables.py --output-dir``, which
sends the table to a staging directory instead of ``replay_content/``. The
script is imported offline with ``spyglass`` stubbed, as
``test_decoding_save_dir.py`` does, and its ``find_sessions``, ``build_one``,
``validate`` and ``publish_canonical`` are replaced, so nothing is read from or
written to ``Processed-Data``.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_tables.timebin_tables import (
    build_event_onset_table,
)
from continuum_swr_replay.paths import replay_content_dir

# All sources require whole-event head speed to be finite and < 4 cm/s.
# Non-Local onsets additionally require every event bin's probability > 0.8.


@pytest.fixture
def two_bout_dot() -> tuple[pd.DataFrame, np.ndarray]:
    """A 50-row session: bout 0 (t 0.00-0.19), movement gap, bout 1 (0.30-0.49).

    Events placed so that:
      - non_local onsets at t = 0.02, 0.10 (bout 0) and 0.32 (bout 1) -> 3 onsets,
      - one SWR onset (t = 0.05) and one MUA onset (t = 0.08), both in bout 0,
      - a single pump at t = 0.05 (in bout 0), used to test gap-bridging.
    Returns ``(dot, pump_times)``.
    """
    t = np.round(np.arange(50) * 0.01, 2)  # 0.00 .. 0.49
    head_speed = np.full(50, 1.0)  # immobile by default
    head_speed[20:30] = 10.0  # movement gap separates the two bouts

    swr = np.zeros(50, dtype=int)
    nl = np.zeros(50, dtype=int)
    mua = np.zeros(50, dtype=int)
    nlprob = np.zeros(50)

    # bout 0 events
    nl[2:5] = 1  # onset t=0.02
    nl[10:13] = 2  # onset t=0.10
    nlprob[2:5] = 0.9
    nlprob[10:13] = 0.9
    swr[5:7] = 1  # onset t=0.05
    mua[8:9] = 1  # onset t=0.08
    # bout 1 event
    nl[32:34] = 3  # onset t=0.32
    nlprob[32:34] = 0.9

    dot = pd.DataFrame(
        {
            "animal": "x",
            "day": 1,
            "epoch": "02_r1",
            "head_speed": head_speed,
            "SWR_event": swr,
            "non_local_event": nl,
            "MUA_event": mua,
            "Non-Local_Continuous_prob": nlprob,
            "Non-Local_Fragmented_prob": 0.0,
        },
        index=t,
    )
    return dot, np.array([0.05])


def test_onset_counts(two_bout_dot):
    """One row per onset of each event type."""
    dot, pumps = two_bout_dot
    onset = build_event_onset_table(dot, pumps)

    sparse_counts = onset["event_type"].value_counts()
    assert sparse_counts.get("non_local", 0) == 3
    assert sparse_counts.get("SWR", 0) == 1
    assert sparse_counts.get("MUA", 0) == 1


def test_t_since_pump_bridges_movement_gap(two_bout_dot):
    """A pump in bout 0 carries forward to an onset in bout 1 across the gap."""
    dot, pumps = two_bout_dot  # pump at t=0.05
    onset = build_event_onset_table(dot, pumps)
    bout1 = onset[
        (onset.event_type == "non_local") & np.isclose(onset.onset_time, 0.32)
    ]
    assert len(bout1) == 1
    # 0.32 - 0.05 = 0.27, even though a movement gap intervenes.
    assert bout1["t_since_pump"].iloc[0] == pytest.approx(0.27)


def test_isi_resets_at_stop_boundary(two_bout_dot):
    """``isi_within_stop`` is NaN for the first onset of each stop (no cross-bout
    carry), and the within-bout gap otherwise."""
    dot, pumps = two_bout_dot
    onset = build_event_onset_table(dot, pumps)
    nl = onset[onset.event_type == "non_local"].sort_values("onset_time")
    isi = dict(zip(np.round(nl.onset_time, 2), nl.isi_within_stop, strict=True))
    assert np.isnan(isi[0.02])  # first onset of bout 0
    assert isi[0.10] == pytest.approx(0.08)  # within bout 0
    assert np.isnan(isi[0.32])  # first onset of bout 1 -> reset, not 0.22 from 0.10


def test_missing_mua_column_tolerated(two_bout_dot):
    """An older decode without MUA_event yields no MUA onsets, not a KeyError."""
    dot, pumps = two_bout_dot
    dot = dot.drop(columns=["MUA_event"])
    onset = build_event_onset_table(dot, pumps)
    assert "MUA" not in set(onset["event_type"])
    # the other event types are unaffected.
    assert int((onset["event_type"] == "non_local").sum()) == 3


def test_empty_session_returns_schema(two_bout_dot):
    """No immobility -> an empty table with the documented columns (not a crash)."""
    dot, pumps = two_bout_dot
    dot = dot.assign(head_speed=10.0)  # never immobile -> no bouts
    onset = build_event_onset_table(dot, pumps)
    assert len(onset) == 0
    for col in ("stop_id", "event_type", "onset_time", "t_since_pump"):
        assert col in onset.columns


#: Modules that bind the Spyglass stub when imported under it.
_STUB_BOUND_MODULES = (
    "continuum_swr_replay.data_loaders.bandit_task",
    "scripts.tables.build_timebin_tables",
)
#: Two synthetic sessions: no real decode has these names.
_SESSIONS = [
    {"nwb_file_name": "testrat20990101_.nwb", "epoch_name": "02_r1"},
    {"nwb_file_name": "testrat20990101_.nwb", "epoch_name": "04_r2"},
]


@pytest.fixture
def timebin_script(monkeypatch):
    """``build_timebin_tables.py``, imported offline with ``spyglass`` stubbed.

    The script imports ``load_data``, whose ``bandit_task`` imports
    ``spyglass.common.PositionIntervalMap`` at import time, which opens a
    DataJoint connection. The modules imported here that bind the stub are
    forgotten afterward, so a later import gets the real ones.
    """
    common = types.ModuleType("spyglass.common")
    common.PositionIntervalMap = type("PositionIntervalMap", (), {})
    monkeypatch.setitem(sys.modules, "spyglass", types.ModuleType("spyglass"))
    monkeypatch.setitem(sys.modules, "spyglass.common", common)

    data_loaders = importlib.import_module("continuum_swr_replay.data_loaders")
    fresh = [name for name in _STUB_BOUND_MODULES if name not in sys.modules]
    load_data_was_cached = "load_data" in vars(data_loaders)

    yield importlib.import_module("scripts.tables.build_timebin_tables")

    for name in fresh:
        sys.modules.pop(name, None)
        parent, _, child = name.rpartition(".")
        vars(sys.modules[parent]).pop(child, None)
    if not load_data_was_cached:
        vars(data_loaders).pop("load_data", None)


def _small_table(session: dict) -> pd.DataFrame:
    """Stand in for ``build_one``: one onset row for `session`."""
    key = {
        "animal": "testrat",
        "session_date": "20990101",
        "epoch": session["epoch_name"],
    }
    return pd.DataFrame([{**key, "event_type": "SWR"}])


def test_output_dir_receives_the_table(
    timebin_script, monkeypatch, tmp_path, processed_data_root
):
    """``--output-dir`` receives the table; without it it goes to
    ``paths.replay_content_dir()``.

    The sessions are synthetic, ``build_one`` returns small frames instead of
    reading a decode and calling ``load_data``, ``validate`` is a no-op, and
    ``publish_swr_table`` records the directory it is given instead of writing,
    refusing any directory outside ``tmp_path``. For the default case the data
    root is ``processed_data_root``, under ``tmp_path``.
    """
    script = timebin_script
    monkeypatch.setattr(script, "find_sessions", lambda max_per_animal: _SESSIONS)
    monkeypatch.setattr(script, "build_one", _small_table)
    monkeypatch.setattr(script, "validate", lambda onset: None)
    published: list[tuple[Path, str]] = []

    def record(
        frame: pd.DataFrame,
        path: Path,
        *,
        table: str,
        basename: str,
    ) -> Path:
        if not Path(path).resolve().is_relative_to(tmp_path.resolve()):
            raise AssertionError(f"refusing to publish outside the test folder: {path}")
        published.append((Path(path), table))
        return Path(path) / f"{basename}.parquet"

    monkeypatch.setattr(script, "publish_swr_table", record)

    script.main(["--output-dir", str(tmp_path)])
    assert published == [(tmp_path, "event_onset")]

    canonical = replay_content_dir()
    assert canonical == processed_data_root / "replay_content"
    published.clear()
    script.main([])
    assert published == [(canonical, "event_onset")]
