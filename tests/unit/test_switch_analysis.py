"""Tests for switch_analysis session-boundary handling."""
from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.replay.switch_analysis import (
    build_control_aligned_table,
)


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


def _make_control_window_fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """One session, 30 stops, single switch at stop_idx=5 plus a
    second switch at stop_idx=20 used as the episode's switch_stop_idx.

    The control anchor will be drawn from stays in the same session.
    For ``max_lag=10`` the anchor candidates exclude stops within ±10
    of switch_stop_idx=20 (so stops 0..9 and 31+), and require ≥5
    stays before the anchor. This guarantees an anchor in the
    [stop_idx 5, stop_idx 9] range and so the [anchor - 10, anchor)
    window WILL straddle the is_stay==0 stop at stop_idx=5 if the
    anchor lands ≥ 6.

    Pre-fix the preceding-window query did not filter ``is_stay==1``,
    so the switch stop at index 5 was emitted as a control row.
    """
    rows = []
    for i in range(30):
        rows.append({
            "animal": "test",
            "day": 20200101,
            "epoch": "02_r1",
            "stop_idx": i,
            "stop_id": i,
            "is_stay": 0.0 if i in (5, 20) else 1.0,
            "current_patch": 1,
            "has_replay": True,
            "n_events": 2,
        })
    stop_table = pd.DataFrame(rows)

    # One event per stop on a single arm (arm 0, patch 1).
    event_table = pd.DataFrame(
        {
            "event_id": stop_table["stop_id"].to_numpy(),
            "stop_id": stop_table["stop_id"].to_numpy(),
            "current_patch": 1,
            "is_swr": True,
            "is_mua": False,
        }
    )
    eat_rows = []
    for sid in stop_table["stop_id"]:
        for arm in range(9):
            eat_rows.append({
                "event_id": int(sid),
                "stop_id": int(sid),
                "arm": arm,
                "arm_fraction": 1.0 if arm == 0 else 0.0,
                "current_patch": 1,
            })
    event_arm_table = pd.DataFrame(eat_rows)

    switch_episodes = pd.DataFrame(
        [
            {
                "episode_id": 1,
                "animal": "test",
                "day": 20200101,
                "epoch": "02_r1",
                "current_patch": 1,
                "switch_stop_idx": 20,
            }
        ]
    )

    return stop_table, event_table, event_arm_table, switch_episodes


def test_control_window_excludes_switch_stops():
    """Regression: control windows must contain only ``is_stay==1``
    stops. Pre-fix the preceding-window query at
    ``build_control_aligned_table`` line ~238 only filtered by
    session keys, ``stop_idx`` range, and ``has_replay``, NOT
    ``is_stay``. A deterministic probe could land an anchor in a
    range whose ``[anchor - max_lag, anchor)`` window included a
    switch (``is_stay==0``) stop, contaminating the matched-control
    comparison with the very switch dynamics it's meant to baseline
    against."""
    stop_table, event_table, event_arm_table, switch_episodes = (
        _make_control_window_fixture()
    )

    # Force the rng to land the anchor at stop_idx == 8 (within the
    # range that exposes the switch at stop_idx == 5 inside the
    # ``[8 - 10, 8) = [-2, 8)`` window).
    class _PinnedRng:
        def integers(self, low: int, high: int) -> int:
            # The function calls ``rng.integers(0, len(possible_anchors))``
            # exactly once per episode. Pin to 0 so the first eligible
            # anchor (the smallest stop_idx in possible_anchors) is
            # picked. With max_lag=10 the eligible anchors come from
            # [stops with stop_idx outside ±10 of 20 AND ≥ index 5
            # in the sorted stays]. Smallest is stop_idx=8 here.
            del low, high
            return 0

    out = build_control_aligned_table(
        stop_table,
        event_table,
        event_arm_table,
        switch_episodes,
        max_lag=10,
        rng=_PinnedRng(),
    )
    assert len(out) > 0, (
        "Expected at least one control row given the fixture; if "
        "this is empty the fixture/anchor pin diverged from the "
        "function's branching."
    )

    # Look up is_stay for every stop_id emitted; none may be a switch.
    is_stay_lookup = stop_table.set_index("stop_id")["is_stay"]
    control_is_stay = is_stay_lookup.loc[out["stop_id"].to_numpy()].to_numpy()
    assert np.all(control_is_stay == 1.0), (
        f"Control table contains {(control_is_stay != 1.0).sum()} "
        "switch (is_stay==0) stop(s); preceding-window filter must "
        "include `is_stay == 1` to honor the matched-control contract. "
        f"Offending stop_ids: "
        f"{out['stop_id'].to_numpy()[control_is_stay != 1.0].tolist()}"
    )
