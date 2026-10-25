"""Tests for the per-bin timebin builders (``timebin_tables``).

Covers the sparse per-onset table (:func:`build_event_onset_table`) and the
dense per-(stop, bin) table (:func:`build_stop_timebin_table`) on a synthetic
session with two immobility bouts separated by a movement gap:

* onset counts agree between the sparse and dense tables,
* ``t_since_pump`` carries a pump across the movement gap into the next bout,
* ``isi_within_stop`` resets at the stop boundary (first onset of a bout is NaN),
* an absent event column (older decodes without ``MUA_event``) is tolerated,
* a session with no immobility returns the empty schema.

It also covers ``scripts/tables/build_timebin_tables.py --output-dir``, which
sends both tables to a staging directory instead of ``replay_content/``. The
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

from continuum_swr_replay.analysis.replay.timebin_tables import (
    build_event_onset_table,
    build_stop_timebin_table,
)
from continuum_swr_replay.paths import REPLAY_CONTENT_DIR

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


def test_onset_counts_match_dense(two_bout_dot):
    """Total sparse onsets per event type == summed dense per-bin counts."""
    dot, pumps = two_bout_dot
    onset = build_event_onset_table(dot, pumps)
    dense = build_stop_timebin_table(dot, pumps, bin_s=0.01)

    sparse_counts = onset["event_type"].value_counts()
    assert sparse_counts.get("non_local", 0) == 3
    assert sparse_counts.get("SWR", 0) == 1
    assert sparse_counts.get("MUA", 0) == 1
    # the dense table counts the same onsets, just bucketed into bins.
    assert int(dense["n_nonlocal_events"].sum()) == 3
    assert int(dense["n_swr_events"].sum()) == 1
    assert int(dense["n_mua_events"].sum()) == 1


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
    """An older decode without MUA_event yields zero MUA counts, not a KeyError."""
    dot, pumps = two_bout_dot
    dot = dot.drop(columns=["MUA_event"])
    onset = build_event_onset_table(dot, pumps)
    dense = build_stop_timebin_table(dot, pumps, bin_s=0.01)
    assert "MUA" not in set(onset["event_type"])
    assert int(dense["n_mua_events"].sum()) == 0
    # the other event types are unaffected.
    assert int(dense["n_nonlocal_events"].sum()) == 3


def test_onsets_in_excluded_bins_are_counted_not_lost(two_bout_dot, monkeypatch):
    """A kept event whose onset bin touches excluded time.

    The event lies wholly outside excluded time, so the sparse table keeps its
    onset (later bins' history needs it). Its bin touches excluded time, so the
    dense table drops the bin. The dense table counts such onsets per event
    type in ``attrs``, and sparse rows = dense sum + that count.
    """
    from continuum_swr_replay.analysis.replay import timebin_tables

    dot, pumps = two_bout_dot
    # The SWR onset at t = 0.05 lies outside [0.041, 0.0474], in the 20-ms bin
    # [0.04, 0.06], which touches it; the other onsets' bins do not.
    monkeypatch.setattr(
        timebin_tables,
        "excluded_intervals_of",
        lambda frame: pd.DataFrame({"start_time": [0.041], "end_time": [0.0474]}),
    )
    onset = build_event_onset_table(dot, pumps)
    dense = build_stop_timebin_table(dot, pumps, bin_s=0.02)

    in_excluded = dense.attrs[timebin_tables.ONSETS_IN_EXCLUDED_BINS_ATTR]
    assert in_excluded == {"SWR": 1, "non_local": 0, "MUA": 0}
    for event_type, count_col in (
        ("SWR", "n_swr_events"),
        ("non_local", "n_nonlocal_events"),
        ("MUA", "n_mua_events"),
    ):
        n_onsets = int((onset["event_type"] == event_type).sum())
        assert n_onsets == int(dense[count_col].sum()) + in_excluded[event_type]


def test_no_excluded_time_counts_no_onsets_in_excluded_bins(two_bout_dot):
    """Without excluded time every onset is in a kept bin."""
    from continuum_swr_replay.analysis.replay import timebin_tables

    dot, pumps = two_bout_dot
    dense = build_stop_timebin_table(dot, pumps, bin_s=0.01)
    assert dense.attrs[timebin_tables.ONSETS_IN_EXCLUDED_BINS_ATTR] == {
        "SWR": 0,
        "non_local": 0,
        "MUA": 0,
    }


def test_validate_counts_onsets_in_excluded_bins(timebin_script):
    """The script's check: onset rows = dense sum + onsets in excluded bins."""
    onset = pd.DataFrame(
        {
            "event_type": ["SWR", "SWR", "non_local"],
            "t_since_stop_onset": [0.1, 0.2, 0.3],
        }
    )
    dense = pd.DataFrame(
        {
            "n_swr_events": [1],
            "n_nonlocal_events": [1],
            "n_mua_events": [0],
            "t_since_stop_onset": [0.0],
            "is_reward_stop": [False],
            "t_since_pump": [np.nan],
        }
    )
    timebin_script.validate(onset, dense, {"SWR": 1, "non_local": 0, "MUA": 0})
    with pytest.raises(AssertionError, match="SWR"):
        timebin_script.validate(onset, dense, {"SWR": 0, "non_local": 0, "MUA": 0})


def test_empty_session_returns_schema(two_bout_dot):
    """No immobility -> empty tables with the documented columns (not a crash)."""
    dot, pumps = two_bout_dot
    dot = dot.assign(head_speed=10.0)  # never immobile -> no bouts
    onset = build_event_onset_table(dot, pumps)
    dense = build_stop_timebin_table(dot, pumps, bin_s=0.01)
    assert len(onset) == 0
    assert len(dense) == 0
    for col in ("stop_id", "event_type", "onset_time", "t_since_pump"):
        assert col in onset.columns
    for col in (
        "stop_id",
        "n_swr_events",
        "n_nonlocal_events",
        "n_mua_events",
        "head_speed",
    ):
        assert col in dense.columns


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
    forgotten afterwards, so a later import gets the real ones.
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


def _small_tables(session: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stand in for ``build_one``: one onset row and one stop bin for `session`."""
    key = {
        "animal": "testrat",
        "session_date": "20990101",
        "epoch": session["epoch_name"],
    }
    onset = pd.DataFrame([{**key, "event_type": "SWR"}])
    dense = pd.DataFrame([{**key, "n_swr_events": 1}])
    return onset, dense


def test_output_dir_receives_both_tables(timebin_script, monkeypatch, tmp_path):
    """``--output-dir`` receives both tables; without it they go to ``RC``.

    The sessions are synthetic, ``build_one`` returns small frames instead of
    reading a decode and calling ``load_data``, ``validate`` is a no-op, and
    ``publish_swr_table`` records the directory it is given instead of writing,
    refusing any directory outside ``tmp_path``. For the default case ``RC`` is
    pointed at a folder under ``tmp_path``, after checking its real value.
    """
    script = timebin_script
    assert script.RC == REPLAY_CONTENT_DIR
    monkeypatch.setattr(script, "find_sessions", lambda max_per_animal: _SESSIONS)
    monkeypatch.setattr(script, "build_one", _small_tables)
    monkeypatch.setattr(script, "validate", lambda onset, dense, in_excluded: None)
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
    assert published == [(tmp_path, "event_onset"), (tmp_path, "stop_timebin")]

    canonical = tmp_path / "canonical"
    monkeypatch.setattr(script, "RC", canonical)
    published.clear()
    script.main([])
    assert published == [(canonical, "event_onset"), (canonical, "stop_timebin")]


def test_published_dense_table_keeps_its_stamp_and_sums_the_excluded_onsets(
    timebin_script, monkeypatch, tmp_path
):
    """Each session's count of onsets in excluded bins is taken off its frame,
    so the concat keeps the frames' shared stamp, and the published dense table
    carries the sum over sessions."""
    from continuum_swr_replay.analysis.replay.timebin_tables import (
        ONSETS_IN_EXCLUDED_BINS_ATTR,
    )

    script = timebin_script

    def stamped_tables(session: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
        onset, dense = _small_tables(session)
        dense.attrs["covariate_schema_table"] = "stop_timebin"
        # Counts differ by session: unequal attrs are what make pd.concat drop
        # every attr, the stamp included, unless the count is taken off first.
        n_swr = 1 if session["epoch_name"] == "02_r1" else 2
        dense.attrs[ONSETS_IN_EXCLUDED_BINS_ATTR] = {"SWR": n_swr, "non_local": 0}
        return onset, dense

    monkeypatch.setattr(script, "find_sessions", lambda max_per_animal: _SESSIONS)
    monkeypatch.setattr(script, "build_one", stamped_tables)
    checked: list[dict] = []
    monkeypatch.setattr(
        script,
        "validate",
        lambda onset, dense, in_excluded: checked.append(dict(in_excluded)),
    )
    published: dict[str, pd.DataFrame] = {}

    def record(
        frame: pd.DataFrame,
        path: Path,
        *,
        table: str,
        basename: str,
    ) -> Path:
        published[table] = frame
        return Path(path) / f"{basename}.parquet"

    monkeypatch.setattr(script, "publish_swr_table", record)
    script.main(["--output-dir", str(tmp_path)])

    assert checked == [{"SWR": 3, "non_local": 0}]
    dense = published["stop_timebin"]
    assert dense.attrs["covariate_schema_table"] == "stop_timebin"
    assert dense.attrs[ONSETS_IN_EXCLUDED_BINS_ATTR] == {"SWR": 3, "non_local": 0}
