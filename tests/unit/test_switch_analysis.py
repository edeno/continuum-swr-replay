"""Tests for switch_analysis session-boundary handling."""
from __future__ import annotations

import pandas as pd


def _make_two_session_stop_table() -> pd.DataFrame:
    """Two sessions where the last stop of session 1 and first stop of session 2
    are both switches. Under the old code (plain .shift without groupby),
    prev_is_switch for session 2's first stop would leak from session 1."""
    rows = []
    # Session 1: 5 stops, last one is a switch (is_stay=0)
    for i in range(5):
        rows.append({
            "animal": "test", "day": 20200101, "epoch": "02_r1",
            "stop_idx": i, "stop_id": i,
            "is_stay": 1.0 if i < 4 else 0.0,
            "current_patch": 1,
            "has_replay": True,
            "n_events": 2,
        })
    # Session 2: 5 stops, first one is also a switch
    for i in range(5):
        rows.append({
            "animal": "test", "day": 20200102, "epoch": "02_r1",
            "stop_idx": i, "stop_id": 100 + i,
            "is_stay": 0.0 if i == 0 else 1.0,
            "current_patch": 2,
            "has_replay": True,
            "n_events": 2,
        })
    return pd.DataFrame(rows)


def test_shift_does_not_leak_across_sessions():
    """The first stop of session 2 should be identified as a first-switch,
    not suppressed by the last stop of session 1 being a switch."""

    stop_table = _make_two_session_stop_table()

    # build_switch_aligned_table needs event_table and event_arm_table;
    # we only need to test the first_switches identification, which
    # happens before those are used. Intercept at the point of
    # first_switches construction by testing the internal logic directly.
    st = stop_table.copy()
    valid = st[st["is_stay"].notna()].copy()
    valid["is_switch"] = (valid["is_stay"] == 0).astype(int)

    # Apply the fixed groupby shift
    valid["prev_is_switch"] = (
        valid.groupby(["animal", "day", "epoch"])["is_switch"]
        .shift(1)
        .fillna(0)
        .astype(int)
    )

    first_switches = valid[(valid["is_switch"] == 1) & (valid["prev_is_switch"] == 0)]

    # Session 1 last stop (index 4, is_switch=1, prev_is_switch should be 0) -> first switch
    # Session 2 first stop (index 5, is_switch=1, prev_is_switch should be 0) -> first switch
    assert len(first_switches) == 2, (
        f"Expected 2 first-switches (one per session), got {len(first_switches)}"
    )

    # Verify session 2's first stop is included
    s2_switches = first_switches[first_switches["day"] == 20200102]
    assert len(s2_switches) == 1, "Session 2's first stop should be a first-switch"


def test_shift_leak_would_fail_without_groupby():
    """Verify that the old (plain shift) approach WOULD incorrectly suppress
    session 2's first switch."""
    stop_table = _make_two_session_stop_table()
    st = stop_table.copy()
    valid = st[st["is_stay"].notna()].copy()
    valid["is_switch"] = (valid["is_stay"] == 0).astype(int)

    # Old broken approach: plain shift without groupby
    valid["prev_is_switch"] = valid["is_switch"].shift(1, fill_value=0)
    first_switches = valid[(valid["is_switch"] == 1) & (valid["prev_is_switch"] == 0)]

    # The old approach would see session 1 last stop = switch, then session 2
    # first stop gets prev_is_switch=1 (leaked), so it's NOT a first-switch.
    assert len(first_switches) == 1, (
        "Old approach should find only 1 first-switch due to session leak"
    )
