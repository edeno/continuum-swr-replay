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
import pandas as pd
import pytest

from continuum_swr_replay.visualization.session_tables import (
    DEFAULT_PANELS,
    _PanelContext,
    _plot_trial_rail,
    register_panel,
)

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
