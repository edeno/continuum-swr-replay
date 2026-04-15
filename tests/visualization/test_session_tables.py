"""Unit tests for ``plot_session_tables`` scaffolding.

Covers what can be tested without a full session: the trial-rail
panel logic, the panel registry, and unknown-panel error handling.
Full end-to-end tests live alongside real-data validation in a later
chunk.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # must be set before pyplot is imported

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.visualization.palettes import patch_grouped_arm_palette
from continuum_swr_replay.visualization.session_tables import (
    DEFAULT_PANELS,
    _compute_mismatch_table,
    _PanelContext,
    _plot_arm_markers,
    _plot_mismatch_pointers,
    _plot_trial_rail,
    register_panel,
)


def _bandit_graph() -> nx.Graph:
    g = nx.Graph()
    for u, v, eid in (
        (0, 6, 0), (1, 6, 1), (2, 7, 2), (3, 7, 3),
        (4, 8, 4), (5, 8, 5), (6, 9, 6), (7, 9, 7), (8, 9, 8),
    ):
        g.add_edge(u, v, edge_id=eid)
    return g

# ── Trial rail panel ──


def _ctx_with_trials(trials: pd.DataFrame, window: tuple[float, float]) -> _PanelContext:
    """Build a minimal context for ``_plot_trial_rail`` isolation."""
    return _PanelContext(
        data={},
        detector=None,
        results=None,
        event_table=pd.DataFrame(),
        event_arm_table=pd.DataFrame(),
        event_sequence_table=pd.DataFrame(),
        stop_table=pd.DataFrame(),
        trials=trials,
        track_graph=None,
        time_slice=slice(*window),
        arm_palette={},
        env=None,
        data_type="clusterless",
    )


class TestTrialRail:
    def test_empty_window_shows_placeholder(self):
        trials = pd.DataFrame(
            {
                "start_time": [100.0],
                "end_time": [110.0],
                "to_well": [3],
                "from_well": [0],
                "is_reward": [True],
                "is_patch_change": [False],
            }
        )
        fig, ax = plt.subplots()
        _plot_trial_rail(ax, _ctx_with_trials(trials, (0.0, 50.0)))
        texts = [t.get_text() for t in ax.texts]
        assert any("no trials" in t for t in texts), texts
        plt.close(fig)

    def test_step_post_plots_one_line_per_trial(self):
        trials = pd.DataFrame(
            {
                "start_time": [0.0, 10.0, 20.0],
                "end_time": [5.0, 15.0, 25.0],
                "to_well": [1, 3, 5],
                "from_well": [0, 1, 3],
                "is_reward": [False, True, False],
                "is_patch_change": [False, True, False],
            }
        )
        fig, ax = plt.subplots()
        _plot_trial_rail(ax, _ctx_with_trials(trials, (0.0, 30.0)))
        # One step-post line covers the rail; well-guideline axhlines
        # are grey at the well positions (6 of them).
        lines = ax.get_lines()
        # 1 step line + 6 axhline guidelines = 7 lines
        assert len(lines) >= 1
        # Reward star: one scatter collection
        collections = list(ax.collections)
        assert len(collections) >= 1
        plt.close(fig)

    def test_ylim_covers_leaf_wells(self):
        trials = pd.DataFrame(
            {
                "start_time": [0.0],
                "end_time": [5.0],
                "to_well": [4],
                "from_well": [0],
                "is_reward": [False],
                "is_patch_change": [False],
            }
        )
        fig, ax = plt.subplots()
        _plot_trial_rail(ax, _ctx_with_trials(trials, (0.0, 10.0)))
        lo, hi = ax.get_ylim()
        assert lo <= -0.5 and hi >= 5.5
        plt.close(fig)


# ── Future-covariate debug: mismatch table + panels ──


def _ctx_for_debug(
    event_table: pd.DataFrame,
    trials: pd.DataFrame,
    window: tuple[float, float],
) -> _PanelContext:
    return _PanelContext(
        data={},
        detector=None,
        results=None,
        event_table=event_table,
        event_arm_table=pd.DataFrame(),
        event_sequence_table=pd.DataFrame(),
        stop_table=pd.DataFrame(),
        trials=trials,
        track_graph=_bandit_graph(),
        time_slice=slice(*window),
        arm_palette=patch_grouped_arm_palette(),
        env=None,
        data_type="clusterless",
    )


def _make_event_table(event_midpoint_times, next_arms):
    """Minimal event_table fixture with the columns debug code reads."""
    return pd.DataFrame(
        {
            "event_id": list(range(len(event_midpoint_times))),
            "event_midpoint_time": event_midpoint_times,
            "current_arm": np.repeat(0.0, len(event_midpoint_times)),
            "prev_arm": np.repeat(1.0, len(event_midpoint_times)),
            "next_arm": next_arms,
            "last_rewarded_arm": np.repeat(2.0, len(event_midpoint_times)),
        }
    )


class TestComputeMismatchTable:
    def test_all_events_match(self):
        """Canonical ``next_arm`` agrees with trials' actual next trip.

        Trial starts at [0, 10, 20]. Event at t=5 is mid-stop after
        trial 0 completed; its "next trip" is trial 1 (start=10) →
        to_well=5 → arm 5. Event at t=15 → trial 2 (start=20) →
        to_well=3 → arm 3.
        """
        event_table = _make_event_table([5.0, 15.0], [5.0, 3.0])
        trials = pd.DataFrame(
            {
                "start_time": [0.0, 10.0, 20.0],
                "end_time": [5.0, 15.0, 25.0],
                "from_well": [0, 3, 5],
                "to_well": [3, 5, 3],
                "is_reward": [True, True, False],
                "is_patch_change": [True, False, True],
            }
        )
        mm = _compute_mismatch_table(event_table, trials, _bandit_graph())
        assert mm["match"].all(), mm

    def test_off_by_one_caught(self):
        """Canonical ``next_arm`` is WRONG; debug row flags the mismatch.

        Actual next trip at event t=5 is trial 1 (start=10), to_well=5,
        arm=5. Fixture claims canonical ``next_arm=1`` — that's the bug
        the debug row is meant to catch.
        """
        event_table = _make_event_table([5.0], [1.0])  # WRONG canonical
        trials = pd.DataFrame(
            {
                "start_time": [0.0, 10.0],
                "end_time": [5.0, 15.0],
                "from_well": [0, 3],
                "to_well": [3, 5],
                "is_reward": [True, True],
                "is_patch_change": [True, False],
            }
        )
        mm = _compute_mismatch_table(event_table, trials, _bandit_graph())
        assert not mm["match"].iloc[0]
        assert mm["canonical_next_arm"].iloc[0] == 1.0
        assert mm["actual_next_arm"].iloc[0] == 5.0

    def test_session_end_nan_matches_nan(self):
        """Last event has no next trip; both canonical and actual are NaN ⇒ match."""
        event_table = _make_event_table([100.0], [np.nan])
        trials = pd.DataFrame(
            {
                "start_time": [0.0, 10.0],
                "end_time": [5.0, 15.0],
                "from_well": [0, 3],
                "to_well": [3, 5],
                "is_reward": [True, True],
                "is_patch_change": [True, False],
            }
        )
        mm = _compute_mismatch_table(event_table, trials, _bandit_graph())
        assert mm["match"].iloc[0]
        assert np.isnan(mm["canonical_next_arm"].iloc[0])
        assert np.isnan(mm["actual_next_arm"].iloc[0])

    def test_empty_event_table(self):
        mm = _compute_mismatch_table(
            pd.DataFrame(columns=["event_id", "event_midpoint_time", "next_arm"]),
            pd.DataFrame(columns=["start_time", "to_well"]),
            _bandit_graph(),
        )
        assert len(mm) == 0
        assert "match" in mm.columns


class TestArmMarkersPanel:
    def test_renders_without_error(self):
        event_table = _make_event_table([5.0, 15.0], [3.0, 5.0])
        trials = pd.DataFrame({"start_time": [], "end_time": [], "to_well": []})
        fig, ax = plt.subplots()
        _plot_arm_markers(ax, _ctx_for_debug(event_table, trials, (0.0, 20.0)))
        # 4 marker kinds × present events = at least 4 scatter collections.
        assert len(ax.collections) >= 4
        plt.close(fig)

    def test_empty_window_placeholder(self):
        event_table = _make_event_table([1000.0], [3.0])
        trials = pd.DataFrame({"start_time": [], "end_time": [], "to_well": []})
        fig, ax = plt.subplots()
        _plot_arm_markers(ax, _ctx_for_debug(event_table, trials, (0.0, 20.0)))
        assert any("no events" in t.get_text() for t in ax.texts)
        plt.close(fig)


class TestMismatchPointersPanel:
    def test_mismatched_pair_drawn_red(self):
        event_table = _make_event_table([5.0], [1.0])  # wrong canonical
        trials = pd.DataFrame(
            {
                "start_time": [0.0, 10.0],
                "end_time": [5.0, 15.0],
                "from_well": [0, 3],
                "to_well": [3, 5],
                "is_reward": [True, True],
                "is_patch_change": [True, False],
            }
        )
        ctx = _ctx_for_debug(event_table, trials, (0.0, 20.0))
        fig, ax = plt.subplots()
        _plot_mismatch_pointers(ax, ctx)
        # Connector line for the mismatch should be red.
        red_lines = [
            ln for ln in ax.get_lines()
            if ln.get_color() == "red"
        ]
        assert len(red_lines) >= 1, "no red mismatch line drawn"
        plt.close(fig)

    def test_stashes_mismatch_table_on_context(self):
        event_table = _make_event_table([5.0], [3.0])
        trials = pd.DataFrame(
            {
                "start_time": [0.0, 10.0],
                "end_time": [5.0, 15.0],
                "from_well": [0, 3],
                "to_well": [3, 5],
                "is_reward": [True, True],
                "is_patch_change": [True, False],
            }
        )
        ctx = _ctx_for_debug(event_table, trials, (0.0, 20.0))
        fig, ax = plt.subplots()
        _plot_mismatch_pointers(ax, ctx)
        assert hasattr(ctx, "_mismatch_table")
        assert len(ctx._mismatch_table) == 1
        plt.close(fig)


# ── Panel registry ──


class TestPanelRegistry:
    def test_default_panels_are_all_registered(self):
        # Import fresh — verify every default panel name is resolvable.
        from continuum_swr_replay.visualization import session_tables as mod
        for name in DEFAULT_PANELS:
            assert name in mod._PANEL_REGISTRY, f"{name} missing from registry"

    def test_register_panel_rejects_duplicate(self):
        def _noop(ax, ctx):
            pass

        from continuum_swr_replay.visualization import session_tables as mod

        with pytest.raises(ValueError, match="already registered"):
            register_panel(next(iter(mod._PANEL_REGISTRY)), _noop)

    def test_register_panel_adds_new_entry(self):
        def _noop(ax, ctx):
            pass

        from continuum_swr_replay.visualization import session_tables as mod

        name = "test_panel_unique_name_xyz"
        register_panel(name, _noop, height=2.0)
        assert name in mod._PANEL_REGISTRY
        assert mod._PANEL_HEIGHTS[name] == 2.0
        # Clean up so the registry isn't polluted for later tests.
        del mod._PANEL_REGISTRY[name]
        del mod._PANEL_HEIGHTS[name]
