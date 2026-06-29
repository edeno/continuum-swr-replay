"""Tests for the per-bin timebin builders (``timebin_tables``).

Covers the sparse per-onset table (:func:`build_event_onset_table`) and the
dense per-(stop, bin) table (:func:`build_stop_timebin_table`) on a synthetic
session with two immobility bouts separated by a movement gap:

* onset counts agree between the sparse and dense tables,
* ``t_since_pump`` carries a pump across the movement gap into the next bout,
* ``isi_within_stop`` resets at the stop boundary (first onset of a bout is NaN),
* an absent event column (older decodes without ``MUA_event``) is tolerated,
* a session with no immobility returns the empty schema.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay.timebin_tables import (
    build_event_onset_table,
    build_stop_timebin_table,
)

# Non-local onsets require Non-Local prob > 0.8 and head_speed < 4 cm/s (the
# immobility threshold); SWR/MUA are taken inclusively at any labelled bin.


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
    assert int(dense["n_non_local"].sum()) == 3
    assert int(dense["n_SWR"].sum()) == 1
    assert int(dense["n_MUA"].sum()) == 1


def test_t_since_pump_bridges_movement_gap(two_bout_dot):
    """A pump in bout 0 carries forward to an onset in bout 1 across the gap."""
    dot, pumps = two_bout_dot  # pump at t=0.05
    onset = build_event_onset_table(dot, pumps)
    bout1 = onset[(onset.event_type == "non_local") & np.isclose(onset.onset_time, 0.32)]
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
    assert int(dense["n_MUA"].sum()) == 0
    # the other event types are unaffected.
    assert int(dense["n_non_local"].sum()) == 3


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
    for col in ("stop_id", "n_SWR", "n_non_local", "n_MUA", "head_speed"):
        assert col in dense.columns
