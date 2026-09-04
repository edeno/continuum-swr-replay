"""Tests for switch_analysis session-boundary handling."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay.switch_analysis import (
    _detect_first_switches,
    build_control_aligned_table,
    build_switch_aligned_table,
)


def _make_two_session_stop_table() -> pd.DataFrame:
    """Two sessions where the last stop of session 1 and first stop of session 2
    are both switches. Under the old code (plain .shift without groupby),
    prev_is_switch for session 2's first stop would leak from session 1."""
    rows = []
    # Session 1: 5 stops, last one is a switch (is_stay=0)
    for i in range(5):
        rows.append({
            "animal": "test", "session_date": 20200101, "epoch": "02_r1",
            "stop_idx": i, "stop_id": i,
            "is_stay": 1.0 if i < 4 else 0.0,
            "current_patch": 1,
            "has_replay": True,
            "n_selected_events": 2,
        })
    # Session 2: 5 stops, first one is also a switch
    for i in range(5):
        rows.append({
            "animal": "test", "session_date": 20200102, "epoch": "02_r1",
            "stop_idx": i, "stop_id": 100 + i,
            "is_stay": 0.0 if i == 0 else 1.0,
            "current_patch": 2,
            "has_replay": True,
            "n_selected_events": 2,
        })
    return pd.DataFrame(rows)


def _replay_tables_for_stops(
    stop_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build minimal event_table / event_arm_table for every stop with
    ``has_replay``. One event per stop; replay 100% on arm 0 (patch 1)
    so the cross-stop alignment helper has bounds to compute.
    """
    sess_cols = ["animal", "session_date", "epoch"]
    replay_stops = stop_table.loc[stop_table["has_replay"]]
    event_table = pd.DataFrame(
        [
            {
                **{c: row[c] for c in sess_cols},
                "event_id": int(row["stop_id"]) + 10_000,
                "stop_id": int(row["stop_id"]),
                "current_patch": int(row["current_patch"]),
                "is_swr": True,
                "is_mua": False,
            }
            for _, row in replay_stops.iterrows()
        ]
    )
    event_arm_rows = []
    for _, row in replay_stops.iterrows():
        event_id = int(row["stop_id"]) + 10_000
        for arm in range(9):
            event_arm_rows.append(
                {
                    **{c: row[c] for c in sess_cols},
                    "event_id": event_id,
                    "stop_id": int(row["stop_id"]),
                    "candidate_arm": arm,
                    "arm_fraction": 1.0 if arm == 0 else 0.0,
                    "current_patch": int(row["current_patch"]),
                }
            )
    event_arm_table = pd.DataFrame(event_arm_rows)
    return event_table, event_arm_table


def test_shift_does_not_leak_across_sessions():
    """Regression: ``prev_is_switch`` previously used
    ``df.shift(1)`` without ``groupby``, so the boundary between
    session 1's last stop and session 2's first stop leaked — making
    session 2's first stop look like a continuation of session 1's
    final switch and getting suppressed from the first-switch set.

    The cross-session boundary is non-observable through the full
    ``build_switch_aligned_table`` output: session 2's first-switch
    sits at ``stop_idx == 0`` and emits zero preceding-stop rows
    either way. The builder calls the private
    ``_detect_first_switches`` helper, so this test exercises the
    helper directly — a regression that drops the ``groupby`` keys
    or the chronological sort fails the assertion below.
    """
    stop_table = _make_two_session_stop_table()
    first_switches = _detect_first_switches(stop_table)

    assert first_switches is not None
    keys = list(
        first_switches[["animal", "session_date", "epoch", "stop_idx"]]
        .itertuples(index=False, name=None)
    )
    assert ("test", 20200101, "02_r1", 4) in keys, (
        "Session 1's last stop is a switch and must be detected as a "
        "first-switch episode."
    )
    assert ("test", 20200102, "02_r1", 0) in keys, (
        "Session 2's first stop must be its own first-switch — pre-fix "
        "the leaked prev_is_switch from session 1's final switch hid "
        "it from the first-switch set."
    )


def test_shift_does_not_leak_across_sessions_in_builder():
    """Companion to the helper-level test above: confirms the builder
    actually routes through ``_detect_first_switches``. A future
    refactor that re-inlines the logic (and re-introduces the leak)
    would still pass the helper-level test if the builder kept its
    own copy. Build the two-session fixture, call the builder, and
    require session 1's switch (the only one that emits preceding
    rows) to appear in the output."""
    stop_table = _make_two_session_stop_table()
    event_table, event_arm_table = _replay_tables_for_stops(stop_table)

    out = build_switch_aligned_table(
        stop_table, event_table, event_arm_table, max_lag=5
    )

    sess1 = out[
        (out["animal"] == "test")
        & (out["session_date"] == 20200101)
        & (out["epoch"] == "02_r1")
    ]
    assert sorted(sess1["switch_stop_idx"].unique().tolist()) == [4], (
        "Session 1's only switch (stop_idx=4) must drive the episode "
        "in the builder output."
    )


def test_first_switch_detection_sorts_by_stop_idx():
    """Regression: ``prev_is_switch`` walks the current row order, so
    an unsorted ``stop_table`` made consecutive switches register as
    separate first-switch episodes. With true ``stop_idx`` order
    [0=stay, 1=stay, 2=switch, 3=switch] but rows shuffled into
    [3, 0, 1, 2], the pre-fix builder emitted episodes for both
    stop_idx=3 and stop_idx=2; only stop_idx=2 should be a
    first-switch in that run. The fix sorts by
    ``[animal, session_date, epoch, stop_idx]`` before computing
    ``prev_is_switch``.
    """
    # Same minimal event_table / event_arm_table fixture as the
    # patch-relative tests below — replay tied to stop 0 so each
    # detected first-switch episode adds at least one preceding row.
    stop_rows = [
        {"animal": "test", "session_date": 20200101, "epoch": "02_r1",
         "stop_idx": 3, "stop_id": 3, "is_stay": 0.0,
         "current_patch": 1, "has_replay": False, "n_selected_events": 0},
        {"animal": "test", "session_date": 20200101, "epoch": "02_r1",
         "stop_idx": 0, "stop_id": 0, "is_stay": 1.0,
         "current_patch": 1, "has_replay": True, "n_selected_events": 1},
        {"animal": "test", "session_date": 20200101, "epoch": "02_r1",
         "stop_idx": 1, "stop_id": 1, "is_stay": 1.0,
         "current_patch": 1, "has_replay": False, "n_selected_events": 0},
        {"animal": "test", "session_date": 20200101, "epoch": "02_r1",
         "stop_idx": 2, "stop_id": 2, "is_stay": 0.0,
         "current_patch": 1, "has_replay": False, "n_selected_events": 0},
    ]
    stop_table = pd.DataFrame(stop_rows)

    event_table = pd.DataFrame([
        {"animal": "test", "session_date": 20200101, "epoch": "02_r1",
         "event_id": 100, "stop_id": 0, "current_patch": 1,
         "is_swr": True, "is_mua": False},
    ])
    event_arm_table = pd.DataFrame([
        {"animal": "test", "session_date": 20200101, "epoch": "02_r1",
         "event_id": 100, "stop_id": 0, "candidate_arm": arm,
         "arm_fraction": 1.0 if arm == 0 else 0.0, "current_patch": 1}
        for arm in range(9)
    ])

    out = build_switch_aligned_table(
        stop_table, event_table, event_arm_table, max_lag=5
    )
    unique_switches = sorted(out["switch_stop_idx"].unique().tolist())
    assert unique_switches == [2], (
        f"Expected only stop_idx=2 as first-switch, got "
        f"{unique_switches} — consecutive switches were "
        "double-counted under unsorted input."
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
            "session_date": 20200101,
            "epoch": "02_r1",
            "stop_idx": i,
            "stop_id": i,
            "is_stay": 0.0 if i in (5, 20) else 1.0,
            "current_patch": 1,
            "has_replay": True,
            "n_selected_events": 2,
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
                "candidate_arm": arm,
                "arm_fraction": 1.0 if arm == 0 else 0.0,
                "current_patch": 1,
            })
    event_arm_table = pd.DataFrame(eat_rows)

    switch_episodes = pd.DataFrame(
        [
            {
                "episode_id": 1,
                "animal": "test",
                "session_date": 20200101,
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


def _make_patch_relative_fixture():
    """Replay at stop 0 lives entirely on arm 0 (patch 1). Stops 1, 2
    are stay stops on patch 2. Stop 3 is the switch (is_stay=0,
    current_patch=2). Pre-fix the switch-aligned row for stop 0
    reported ``replay_fraction_current_patch=1.0`` against the switch's
    current_patch=2, even though that replay landed on patch 1.
    """
    stop_table = pd.DataFrame([
        {"animal": "a", "session_date": 1, "epoch": "e1", "stop_idx": 0, "stop_id": 0,
         "is_stay": 1.0, "current_patch": 1, "has_replay": True, "n_selected_events": 1},
        {"animal": "a", "session_date": 1, "epoch": "e1", "stop_idx": 1, "stop_id": 1,
         "is_stay": 1.0, "current_patch": 2, "has_replay": False, "n_selected_events": 0},
        {"animal": "a", "session_date": 1, "epoch": "e1", "stop_idx": 2, "stop_id": 2,
         "is_stay": 1.0, "current_patch": 2, "has_replay": False, "n_selected_events": 0},
        {"animal": "a", "session_date": 1, "epoch": "e1", "stop_idx": 3, "stop_id": 3,
         "is_stay": 0.0, "current_patch": 2, "has_replay": False, "n_selected_events": 0},
    ])
    event_table = pd.DataFrame([
        {"event_id": 100, "stop_id": 0, "current_patch": 1,
         "is_swr": True, "is_mua": False},
    ])
    event_arm_table = pd.DataFrame([
        {"event_id": 100, "stop_id": 0, "candidate_arm": arm,
         "arm_fraction": 1.0 if arm == 0 else 0.0, "current_patch": 1}
        for arm in range(9)
    ])
    return stop_table, event_table, event_arm_table


def test_switch_aligned_replay_relative_to_switch_patch():
    """Regression: ``build_switch_aligned_table`` previously aggregated
    replay relative to each replay stop's OWN current_patch via
    ``stop_patch_replay``, then stamped the SWITCH's current_patch
    onto the lag rows. A replay that landed on the previous patch in
    a multi-patch run was silently re-labeled as "replay on the
    switch patch". Fix recomputes per-stop, per-patch fractions and
    looks up by the switch's target patch."""
    stop_table, event_table, event_arm_table = _make_patch_relative_fixture()
    out = build_switch_aligned_table(
        stop_table, event_table, event_arm_table, max_lag=5
    )
    assert len(out) == 1
    row = out.iloc[0]
    # Replay was 100% on arm 0 = patch 1, switch's current_patch is 2.
    # Pre-fix: replay_fraction_current_patch=1.0 (wrong);
    # post-fix: replay_fraction_current_patch=0.0 (correct).
    assert row["replay_fraction_current_patch"] == 0.0, (
        f"replay_fraction_current_patch={row['replay_fraction_current_patch']} — "
        "replay landed on patch 1 but the switch is on patch 2; "
        "fraction-on-switch-patch should be 0.0."
    )
    assert row["replay_fraction_noncurrent"] == 1.0


def test_switch_aligned_replay_attributes_when_replay_matches_switch():
    """Sanity: when the replay actually IS on the switch patch, the
    fraction is 1.0. Differences from the prior test isolate to which
    patch the arm sits in, so this rules out an off-by-one in patch
    indexing."""
    stop_table, event_table, event_arm_table = _make_patch_relative_fixture()
    # Move the replay to arm 2 (patch 2), so it now matches the
    # switch's current_patch=2.
    event_arm_table = event_arm_table.copy()
    event_arm_table.loc[
        event_arm_table["candidate_arm"] == 0, "arm_fraction"
    ] = 0.0
    event_arm_table.loc[
        event_arm_table["candidate_arm"] == 2, "arm_fraction"
    ] = 1.0
    out = build_switch_aligned_table(
        stop_table, event_table, event_arm_table, max_lag=5
    )
    row = out.iloc[0]
    assert row["replay_fraction_current_patch"] == 1.0
    assert row["replay_fraction_noncurrent"] == 0.0


def test_control_aligned_replay_relative_to_switch_patch():
    """Regression: ``build_control_aligned_table`` had the same
    conceptual mismatch — control replay fractions were aggregated
    relative to each control stop's own current_patch but stored
    under the switch episode's patch label, breaking the matched-
    control contract. Fix uses the same per-stop, per-patch helper as
    the switch-aligned path.
    """
    # Build a session where the replay-bearing control stop sits
    # within the [anchor - max_lag, anchor) window. Layout:
    #   stops 0..3    — leading stays, no replay (4 stays)
    #   stop  4       — replay stop, on patch 1 (1 stay, 5 total)
    #   stops 5..14   — more leading stays, no replay (10 stays,
    #                   15 total) — the anchor will be drawn from
    #                   ``session_stays.iloc[5:]`` (= stop_idx 5..14)
    # All stops are on patch 1; the matched switch episode has
    # current_patch=2 (a different patch) and switch_stop_idx well
    # outside the ±max_lag exclusion zone.
    rows = []
    for i in range(4):
        rows.append({
            "animal": "a", "session_date": 1, "epoch": "e1",
            "stop_idx": i, "stop_id": 1000 + i,
            "is_stay": 1.0, "current_patch": 1,
            "has_replay": False, "n_selected_events": 0,
        })
    rows.append({
        "animal": "a", "session_date": 1, "epoch": "e1",
        "stop_idx": 4, "stop_id": 0,
        "is_stay": 1.0, "current_patch": 1,
        "has_replay": True, "n_selected_events": 1,
    })
    for i in range(5, 15):
        rows.append({
            "animal": "a", "session_date": 1, "epoch": "e1",
            "stop_idx": i, "stop_id": 1000 + i,
            "is_stay": 1.0, "current_patch": 1,
            "has_replay": False, "n_selected_events": 0,
        })
    stop_table = pd.DataFrame(rows)
    event_table = pd.DataFrame([
        {"event_id": 200, "stop_id": 0, "current_patch": 1,
         "is_swr": True, "is_mua": False},
    ])
    event_arm_table = pd.DataFrame([
        {"event_id": 200, "stop_id": 0, "candidate_arm": arm,
         "arm_fraction": 1.0 if arm == 0 else 0.0, "current_patch": 1}
        for arm in range(9)
    ])
    # Matched switch episode: current_patch=2, switch_stop_idx=200
    # (far outside ±max_lag, so all stays in the session are
    # eligible anchors).
    switch_episodes = pd.DataFrame([
        {"episode_id": 1, "animal": "a", "session_date": 1, "epoch": "e1",
         "current_patch": 2, "switch_stop_idx": 200},
    ])

    # Pin the rng so ``possible_anchors.iloc[5]`` is picked — that
    # corresponds to stop_idx=10 in the fixture, whose [0, 10) window
    # straddles the replay stop at stop_idx=4.
    class _PinnedRng:
        def integers(self, low, high):  # noqa: ARG002
            return 5

    out = build_control_aligned_table(
        stop_table,
        event_table,
        event_arm_table,
        switch_episodes,
        max_lag=10,
        rng=_PinnedRng(),
    )
    assert len(out) >= 1
    # Find the row for stop_id 0 (the replay-bearing control stop).
    row = out[out["stop_id"] == 0].iloc[0]
    # Replay was on patch 1; matched switch's current_patch=2.
    # Post-fix: fraction-on-switch-patch is 0.0, not 1.0.
    assert row["replay_fraction_current_patch"] == 0.0, (
        f"Control row replay_fraction_current_patch="
        f"{row['replay_fraction_current_patch']} — replay landed on "
        "patch 1 but the matched switch episode's current_patch is "
        "2; fraction should be 0.0 to honor the matched-control "
        "contract."
    )
    assert row["replay_fraction_noncurrent"] == 1.0
    assert row["current_patch"] == 2  # Switch episode's patch is preserved.


def _make_two_session_collision_fixture(replay_arms: tuple[int, int]):
    """Two sessions, both reusing ``stop_id=0`` (session-local IDs).
    Each session's lone replay event lands entirely on a different
    arm. Without a session-uniqueness guard, the per-stop, per-patch
    helper averages across sessions (replay_fraction_current_patch=0.5
    in both rows even though each session's true value is 0.0 or 1.0).
    """
    rows_st = []
    rows_et = []
    rows_eat = []
    for animal, session_date, arm in (("a", 1, replay_arms[0]), ("b", 2, replay_arms[1])):
        rows_st.extend([
            {"animal": animal, "session_date": session_date, "epoch": "e1",
             "stop_idx": 0, "stop_id": 0, "is_stay": 1.0,
             "current_patch": 1, "has_replay": True, "n_selected_events": 1},
            {"animal": animal, "session_date": session_date, "epoch": "e1",
             "stop_idx": 1, "stop_id": 1, "is_stay": 1.0,
             "current_patch": 2, "has_replay": False, "n_selected_events": 0},
            {"animal": animal, "session_date": session_date, "epoch": "e1",
             "stop_idx": 2, "stop_id": 2, "is_stay": 1.0,
             "current_patch": 2, "has_replay": False, "n_selected_events": 0},
            {"animal": animal, "session_date": session_date, "epoch": "e1",
             "stop_idx": 3, "stop_id": 3, "is_stay": 0.0,
             "current_patch": 2, "has_replay": False, "n_selected_events": 0},
        ])
        eid = 100 if animal == "a" else 101
        rows_et.append({
            "animal": animal, "session_date": session_date, "epoch": "e1",
            "event_id": eid, "stop_id": 0, "current_patch": 1,
            "is_swr": True, "is_mua": False,
        })
        for a in range(9):
            rows_eat.append({
                "animal": animal, "session_date": session_date, "epoch": "e1",
                "event_id": eid, "stop_id": 0, "candidate_arm": a,
                "arm_fraction": 1.0 if a == arm else 0.0,
                "current_patch": 1,
            })
    return (
        pd.DataFrame(rows_st),
        pd.DataFrame(rows_et),
        pd.DataFrame(rows_eat),
    )


def test_switch_aligned_session_local_stop_ids_raise():
    """Regression: ``_stop_replay_per_patch`` must reject
    cross-session collisions on ``stop_id`` / ``event_id``. Without
    that guard, the per-stop reduction silently averages across
    sessions whenever a session-local ID is reused — the user
    reproduced this with two sessions both using ``stop_id=0``,
    where the per-patch fractions came back as the cross-session
    mean (0.5) instead of each session's true value (0.0 or 1.0).
    """
    st, et, eat = _make_two_session_collision_fixture(replay_arms=(0, 2))
    with pytest.raises(ValueError, match="must be globally unique across sessions"):
        build_switch_aligned_table(st, et, eat, max_lag=5)


def test_switch_aligned_globally_unique_ids_partition_per_session():
    """Sanity: with globally-unique stop_id / event_id (the canonical
    hashed-ID contract), the helper must NOT mix across sessions —
    each session's row reflects its own replay arm. Differs from the
    failing test above only in that the ID values are session-disjoint.
    """
    st, et, eat = _make_two_session_collision_fixture(replay_arms=(0, 2))
    # Reassign IDs so they're globally unique.
    is_b = (st["animal"] == "b").to_numpy()
    st.loc[is_b, "stop_id"] += 100
    is_b_et = (et["animal"] == "b").to_numpy()
    et.loc[is_b_et, "stop_id"] += 100
    et.loc[is_b_et, "event_id"] += 100
    is_b_eat = (eat["animal"] == "b").to_numpy()
    eat.loc[is_b_eat, "stop_id"] += 100
    eat.loc[is_b_eat, "event_id"] += 100

    out = build_switch_aligned_table(st, et, eat, max_lag=5)
    assert len(out) == 2
    # session a's replay was on arm 0 (patch 1), switch on patch 2.
    # session b's replay was on arm 2 (patch 2), switch on patch 2.
    by_animal = out.set_index("animal")
    assert by_animal.loc["a", "replay_fraction_current_patch"] == 0.0
    assert by_animal.loc["a", "replay_fraction_noncurrent"] == 1.0
    assert by_animal.loc["b", "replay_fraction_current_patch"] == 1.0
    assert by_animal.loc["b", "replay_fraction_noncurrent"] == 0.0


def test_stop_replay_per_patch_fallback_merge_still_validates_stop_id():
    """Regression: the uniqueness guard must run AFTER the
    ``stop_id`` fallback merge from ``event_table``. The previous
    placement checked ``stop_id`` against ``event_arm_table`` first,
    which is a no-op when ``stop_id`` isn't a column there — so a
    session-local ``stop_id`` arriving via the fallback merge slipped
    through and the per-stop reduction silently averaged across
    sessions (probe: two sessions with globally-unique ``event_id``
    but reused ``stop_id=0`` and no ``stop_id`` in
    ``event_arm_table`` → ``replay_fraction_current_patch=0.5`` for both
    sessions instead of session-specific values).
    """
    from continuum_swr_replay.analysis.replay.switch_analysis import (
        _stop_replay_per_patch,
    )

    eat = pd.DataFrame(
        [
            *(
                {
                    "animal": "a", "session_date": 1, "epoch": "e1",
                    "event_id": 100, "candidate_arm": a,
                    "arm_fraction": 1.0 if a == 0 else 0.0,
                }
                for a in range(9)
            ),
            *(
                {
                    "animal": "b", "session_date": 2, "epoch": "e1",
                    "event_id": 101, "candidate_arm": a,
                    "arm_fraction": 1.0 if a == 2 else 0.0,
                }
                for a in range(9)
            ),
        ]
    )
    et = pd.DataFrame(
        [
            {"animal": "a", "session_date": 1, "epoch": "e1",
             "event_id": 100, "stop_id": 0},
            {"animal": "b", "session_date": 2, "epoch": "e1",
             "event_id": 101, "stop_id": 0},
        ]
    )
    with pytest.raises(ValueError, match="must be globally unique across sessions"):
        _stop_replay_per_patch(eat, et, fraction_col="arm_fraction")
