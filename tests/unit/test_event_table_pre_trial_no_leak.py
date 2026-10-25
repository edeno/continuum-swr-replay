"""Regression test for event-table pre-trial trial-context leakage.

The replay/contfrag event tables doc-claim event-level trial context.
The implementation used ``safe_first`` (drop NaN before .iloc[0]) to
extract those values, which silently surfaces the *next* trial's
context for events whose first timebin fell before any trial began —
e.g. an event spanning t=0..2 on a session whose first trial starts
at t=1 inherited trial 1's ``is_rewarded_outcome`` / ``occupancy_well_arm`` /
``is_rewarded_outcome`` even though t=0 had no trial active.

Fix: replace ``safe_first`` with NaN-strict ``strict_first`` for the
trial-derived fields so a before-trial event surfaces missing context
explicitly. The stop-table builder already uses the strict semantics
for the same hazard (see ``test_stop_table_bout_start_no_leak.py``).
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.replay.tables_event import (
    build_event_table as _build_event_table,
)
from tests.unit._port_experience_fixtures import port_experiences_from_trials


def build_event_table(dot, env, trials, graph, **kwargs):
    return _build_event_table(
        dot,
        env,
        trials,
        port_experiences_from_trials(trials, prepend_initial=True),
        graph,
        **kwargs,
    )


def _two_well_track_graph() -> nx.Graph:
    """Minimal bandit-style graph with one leaf arm (arm 0) for clean
    ``current_arm`` extraction during the event."""
    g = nx.Graph()
    for u, v, eid in (
        (0, 6, 0),
        (1, 6, 1),
        (2, 7, 2),
        (3, 7, 3),
        (4, 8, 4),
        (5, 8, 5),
        (6, 9, 6),
        (7, 9, 7),
        (8, 9, 8),
    ):
        g.add_edge(u, v, edge_id=eid, distance=1.0)
    return g


class _EnvStub:
    """Minimal Environment stand-in (mirrors test_replay_tables.py)."""

    class _Df:
        _arm_lp = np.arange(18, dtype=float)
        _arm_eid = np.repeat(np.arange(9), 2).astype(int)
        _junc_lp = np.array([100.0, 101.0, 102.0])
        _junc_eid = np.array([-1, -1, -1])
        d = {
            "linear_position": np.concatenate([_arm_lp, _junc_lp]),
            "edge_id": np.concatenate([_arm_eid, _junc_eid]),
        }

        def __getitem__(self, k):
            return self.d[k]

    place_bin_centers_nodes_df_ = _Df()


def _pre_trial_event_dot() -> pd.DataFrame:
    """Single non-local event spanning t=0..2.0 — entirely before the
    first trial start at t=2.5. Every bin is immobile and has high
    Non-Local_Continuous_prob so it passes ``_in_event_mask``.
    Modal ``track_segment_id`` is NaN (no spatial info), so the
    ``current_arm`` fallback drops to ``current_well``, which is
    derived from the trial context — exactly the leak path under test.
    """
    times = np.linspace(0.0, 2.0, 11)
    return pd.DataFrame(
        {
            "animal": ["m"] * len(times),
            "day": [20200101] * len(times),
            "epoch": ["02_r1"] * len(times),
            "non_local_event": [1] * len(times),
            "SWR_event": [0] * len(times),
            "MUA_event": [0] * len(times),
            "Non-Local_Continuous_prob": [0.9] * len(times),
            "Non-Local_Fragmented_prob": [0.0] * len(times),
            "head_speed": [0.0] * len(times),
            # NaN map_estimate / track_segment_id forces fallback to
            # current_well for ``current_arm`` — that's the column
            # whose leak the user reproduced.
            "map_estimate": [np.nan] * len(times),
            "track_segment_id": [np.nan] * len(times),
        },
        index=times,
    )


def _trials_starting_after_event() -> pd.DataFrame:
    # Single rewarded trial starting at t=2.5, after the event window
    # closes at t=2.0. Every dot timebin in the event has no trial
    # active, so add_trial_context fills NaN for the entire event.
    return pd.DataFrame(
        {
            "movement_start_time": [2.5],
            "end_time": [3.0],
            "from_well": [1],
            "to_well": [0],
            "is_reward": [True],
            "is_patch_change": [False],
        }
    )


class TestEventStartContextNoLeak:
    def test_event_entirely_before_first_trial_yields_nan_context(self):
        """User repro: an event whose first (and every) timebin
        precedes the first trial used to inherit
        ``is_rewarded_outcome=1.0`` and
        ``current_arm=0.0`` from the not-yet-started trial. Strict
        first-row extraction must surface NaN instead.
        """
        dot = _pre_trial_event_dot()
        trials = _trials_starting_after_event()
        graph = _two_well_track_graph()
        env = _EnvStub()
        out = build_event_table(
            dot,
            env,
            trials,
            graph,
            speed_threshold=4.0,
            nonlocal_threshold=0.5,
        )
        assert len(out) == 1, f"expected 1 event row, got {len(out)}"
        row = out.iloc[0]

        for col in (
            "is_rewarded_outcome",
            "did_last_completed_trip_switch_patch",
            "occupancy_well_arm",
            "prev_arm",
            "next_arm",
            "last_rewarded_arm",
            "current_patch",
            "prev_patch",
            "next_patch",
        ):
            assert pd.isna(row[col]), (
                f"{col}={row[col]!r} leaked from a future trial — "
                "an event entirely before any trial start should have "
                "NaN trial-derived context."
            )
        # The collapsed legacy outcome spellings must never reappear as columns.
        assert not {"is_after_reward", "is_after_omission", "reward_outcome"} & set(
            out.columns
        )

    def test_event_starting_before_trial_does_not_inherit_future_context(self):
        """Half-overlapping case: event spans t=0..2.0, first trial
        starts at t=1.0. The first event bin still has no trial
        active, so the strict extractor returns NaN. ``safe_first``
        would have skipped the leading NaN bins and surfaced trial 1's
        context.
        """
        dot = _pre_trial_event_dot()
        trials = pd.DataFrame(
            {
                "movement_start_time": [1.0],
                "end_time": [2.5],
                "from_well": [1],
                "to_well": [0],
                "is_reward": [True],
                "is_patch_change": [False],
            }
        )
        graph = _two_well_track_graph()
        env = _EnvStub()
        out = build_event_table(
            dot,
            env,
            trials,
            graph,
            speed_threshold=4.0,
            nonlocal_threshold=0.5,
        )
        assert len(out) == 1
        row = out.iloc[0]
        # The event's first timebin (t=0) is before the trial — strict
        # first-row extraction propagates NaN even though later bins
        # in the event have valid trial context.
        assert pd.isna(row["is_rewarded_outcome"]), (
            f"is_rewarded_outcome leaked: {row['is_rewarded_outcome']!r}"
        )
        assert pd.isna(row["occupancy_well_arm"])
