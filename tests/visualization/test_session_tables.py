"""Unit tests for ``plot_session_tables`` scaffolding.

Covers what can be tested without a full session: the trial-rail
panel logic, the panel registry, and unknown-panel error handling.
Full end-to-end tests live alongside real-data validation in a later
chunk.
"""

from __future__ import annotations

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")  # must be set before pyplot is imported

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba

from continuum_swr_replay.analysis.replay._helpers import (
    single_arm_from_well,
)
from continuum_swr_replay.analysis.replay.covariates import (
    add_trial_context as _add_trial_context,
)
from continuum_swr_replay.visualization.palettes import patch_grouped_arm_palette
from continuum_swr_replay.visualization.session_tables import (
    DEFAULT_PANELS,
    _PanelContext,
    _plot_arm_markers,
    _plot_mismatch_pointers,
    _plot_trial_rail,
    plot_session_tables,
    register_panel,
)
from continuum_swr_replay.visualization.session_tables import (
    _compute_mismatch_table as _production_compute_mismatch_table,
)
from tests.unit._port_experience_fixtures import port_experiences_from_trials


def _experiences(trials: pd.DataFrame) -> pd.DataFrame:
    return port_experiences_from_trials(trials, prepend_initial=True)


def add_trial_context(dot: pd.DataFrame, trials: pd.DataFrame) -> pd.DataFrame:
    return _add_trial_context(dot, trials, _experiences(trials))


def _compute_mismatch_table(
    event_table: pd.DataFrame, trials: pd.DataFrame, track_graph
) -> pd.DataFrame:
    return _production_compute_mismatch_table(
        event_table, trials, _experiences(trials), track_graph
    )


def _bandit_graph() -> nx.Graph:
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
        g.add_edge(u, v, edge_id=eid)
    return g


# ── Trial rail panel ──


def _ctx_with_trials(
    trials: pd.DataFrame, window: tuple[float, float]
) -> _PanelContext:
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
        port_experiences=_experiences(trials),
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
                "is_patch_change": [True],
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
                "is_patch_change": [False, True, True],
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
                "is_patch_change": [True],
            }
        )
        fig, ax = plt.subplots()
        _plot_trial_rail(ax, _ctx_with_trials(trials, (0.0, 10.0)))
        lo, hi = ax.get_ylim()
        assert lo <= -0.5 and hi >= 5.5
        plt.close(fig)

    def test_intertrial_window_retains_last_completed_destination(self):
        """The rail remains at the last well between completed trips."""
        trials = pd.DataFrame(
            {
                "start_time": [0.0, 10.0],
                "end_time": [5.0, 15.0],
                "to_well": [1, 3],
                "from_well": [0, 1],
                "is_reward": [False, False],
                "is_patch_change": [False, True],
            }
        )
        fig, ax = plt.subplots()
        _plot_trial_rail(ax, _ctx_with_trials(trials, (6.0, 9.0)))

        rail_lines = [line for line in ax.lines if line.get_color() == "#333333"]
        assert len(rail_lines) == 1
        x = np.asarray(rail_lines[0].get_xdata(), dtype=float)
        y = np.asarray(rail_lines[0].get_ydata(), dtype=float)
        assert x.min() <= 6.0 and x.max() >= 9.0
        assert np.all(y == 1.0)
        assert not any("no trials" in text.get_text() for text in ax.texts)
        plt.close(fig)

    @staticmethod
    def _timed_trials() -> pd.DataFrame:
        """Trips whose start, arrival, and interval-end times are distinct."""
        return pd.DataFrame(
            {
                "start_time": [0.0, 10.0],
                "goal_arrival_time": [6.0, np.nan],
                "end_time": [8.0, 18.0],
                "from_well": [0, 1],
                "to_well": [1, 3],
                "is_reward": [True, True],
                "is_patch_change": [False, True],
            }
        )

    def test_dwell_rail_is_blank_during_transit_and_updates_at_arrival(self):
        """Well location is visible only after each trip completes."""
        fig, ax = plt.subplots()
        _plot_trial_rail(ax, _ctx_with_trials(self._timed_trials(), (0.0, 20.0)))

        rails = [line for line in ax.lines if line.get_color() == "#333333"]
        assert len(rails) == 2
        segments = [
            (
                np.asarray(line.get_xdata(), dtype=float).tolist(),
                np.asarray(line.get_ydata(), dtype=float).tolist(),
            )
            for line in rails
        ]
        assert segments == [([6.0, 10.0], [1.0, 1.0]), ([18.0, 20.0], [3.0, 3.0])]
        plt.close(fig)

    def test_reward_star_uses_goal_arrival_with_end_time_fallback(self):
        fig, ax = plt.subplots()
        _plot_trial_rail(ax, _ctx_with_trials(self._timed_trials(), (0.0, 20.0)))

        offsets = np.vstack(
            [
                np.asarray(collection.get_offsets(), dtype=float)
                for collection in ax.collections
                if len(collection.get_offsets())
            ]
        )
        offsets = offsets[np.argsort(offsets[:, 0])]

        np.testing.assert_allclose(
            offsets,
            np.array([[6.0, 1.0], [18.0, 3.0]]),
            rtol=0,
            atol=0,
        )
        plt.close(fig)

    def test_patch_switch_tick_uses_trip_completion_time(self):
        fig, ax = plt.subplots()
        _plot_trial_rail(ax, _ctx_with_trials(self._timed_trials(), (0.0, 20.0)))

        ticks = [line for line in ax.lines if line.get_color() == "#9467bd"]
        assert len(ticks) == 1
        assert np.asarray(ticks[0].get_xdata(), dtype=float) == pytest.approx(
            [18.0, 18.0]
        )
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
        port_experiences=_experiences(trials),
        track_graph=_bandit_graph(),
        time_slice=slice(*window),
        arm_palette=patch_grouped_arm_palette(),
        env=None,
        data_type="clusterless",
    )


def _make_event_table(event_midpoint_times, next_arms, *, event_start_times=None):
    """Minimal event_table fixture with the columns debug code reads."""
    if event_start_times is None:
        event_start_times = event_midpoint_times
    return pd.DataFrame(
        {
            "event_id": list(range(len(event_midpoint_times))),
            "event_start_s": event_start_times,
            "event_midpoint_s": event_midpoint_times,
            "occupancy_well_arm": np.repeat(0.0, len(event_midpoint_times)),
            "prev_arm": np.repeat(1.0, len(event_midpoint_times)),
            "next_arm": next_arms,
            "last_rewarded_arm": np.repeat(2.0, len(event_midpoint_times)),
        }
    )


class TestComputeMismatchTable:
    def test_exact_parity_with_production_trial_context_and_arm_mapping(self):
        """Audit reconstruction cannot drift from the production helpers.

        One vector covers shuffled input, pre-first context, both sides of a
        trial boundary, the boundary itself, and the no-next-trip session tail.
        Categorical wells/arms must agree exactly; there is no floating-point
        tolerance to choose for these quantities.
        """
        trials = pd.DataFrame(
            {
                # Deliberately shuffled: production sorts before searchsorted.
                "start_time": [20.0, 0.0, 10.0],
                "goal_arrival_time": [24.0, 4.0, np.nan],
                "end_time": [25.0, 5.0, 15.0],
                "from_well": [1, 0, 0],
                "to_well": [2, 0, 1],
                "is_reward": [False, True, False],
                "is_patch_change": [True, False, False],
            }
        )
        event_starts = np.array([-1.0, 0.0, 9.999, 10.0, 19.999, 20.0, 30.0])
        production = add_trial_context(
            pd.DataFrame(index=pd.Index(event_starts, name="time")),
            trials,
        )
        expected_wells = production["next_well"].to_numpy(dtype=float)
        expected_arms = np.array(
            [single_arm_from_well(well, _bandit_graph()) for well in expected_wells]
        )
        event_table = pd.DataFrame(
            {
                "event_id": np.arange(len(event_starts)),
                "event_start_s": event_starts,
                "event_midpoint_s": event_starts + 0.001,
                "next_arm": expected_arms,
            }
        )

        mismatch = _compute_mismatch_table(event_table, trials, _bandit_graph())

        np.testing.assert_array_equal(
            mismatch["actual_next_to_well"].to_numpy(dtype=float), expected_wells
        )
        np.testing.assert_array_equal(
            mismatch["actual_next_arm"].to_numpy(dtype=float), expected_arms
        )
        assert mismatch["match"].all()

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
                "is_patch_change": [True, True, True],
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
                "is_patch_change": [True, True],
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
                "is_patch_change": [True, True],
            }
        )
        mm = _compute_mismatch_table(event_table, trials, _bandit_graph())
        assert mm["match"].iloc[0]
        assert np.isnan(mm["canonical_next_arm"].iloc[0])
        assert np.isnan(mm["actual_next_arm"].iloc[0])

    def test_trial_boundary_straddle_uses_event_start_context(self):
        """Canonical trial context is defined at the event's first time bin."""
        event_table = _make_event_table([10.001], [1.0], event_start_times=[9.96])
        trials = pd.DataFrame(
            {
                "start_time": [0.0, 10.0, 20.0],
                "end_time": [5.0, 15.0, 25.0],
                "from_well": [0, 0, 1],
                "to_well": [0, 1, 2],
                "is_reward": [False, False, False],
                "is_patch_change": [False, False, True],
            }
        )

        mm = _compute_mismatch_table(event_table, trials, _bandit_graph())

        assert mm["actual_next_arm"].iloc[0] == 1.0
        assert mm["match"].iloc[0]

    def test_unsorted_trials_are_sorted_before_future_lookup(self):
        event_table = _make_event_table([15.0], [2.0], event_start_times=[15.0])
        trials = pd.DataFrame(
            {
                "start_time": [20.0, 0.0, 10.0],
                "end_time": [25.0, 5.0, 15.0],
                "from_well": [1, 0, 0],
                "to_well": [2, 0, 1],
                "is_reward": [False, False, False],
                "is_patch_change": [True, False, False],
            }
        )

        mm = _compute_mismatch_table(event_table, trials, _bandit_graph())

        assert mm["actual_next_arm"].iloc[0] == 2.0
        assert mm["match"].iloc[0]

    def test_event_before_first_trial_has_no_future_context(self):
        """Match add_trial_context: no current trial means next_well is unknown."""
        event_table = _make_event_table([-0.5], [np.nan], event_start_times=[-1.0])
        trials = pd.DataFrame(
            {
                "start_time": [0.0, 10.0],
                "end_time": [5.0, 15.0],
                "from_well": [0, 1],
                "to_well": [1, 2],
                "is_reward": [False, False],
                "is_patch_change": [False, True],
            }
        )

        mm = _compute_mismatch_table(event_table, trials, _bandit_graph())

        assert np.isnan(mm["actual_next_arm"].iloc[0])
        assert mm["match"].iloc[0]

    def test_empty_event_table(self):
        mm = _compute_mismatch_table(
            pd.DataFrame(
                columns=[
                    "event_id",
                    "event_start_s",
                    "event_midpoint_s",
                    "next_arm",
                ]
            ),
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
                "is_patch_change": [True, True],
            }
        )
        ctx = _ctx_for_debug(event_table, trials, (0.0, 20.0))
        fig, ax = plt.subplots()
        _plot_mismatch_pointers(ax, ctx)
        # Connector line for the mismatch should be red.
        red_lines = [ln for ln in ax.get_lines() if ln.get_color() == "red"]
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
                "is_patch_change": [True, True],
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


# ── Covariate panels: behavioral model, arm/patch/stop grain ──


def _ctx_for_covariates(
    *,
    event_arm_table: pd.DataFrame | None = None,
    event_table: pd.DataFrame | None = None,
    event_patch_table: pd.DataFrame | None = None,
    stop_table: pd.DataFrame | None = None,
    window: tuple[float, float] = (0.0, 10.0),
) -> _PanelContext:
    empty = pd.DataFrame()
    return _PanelContext(
        data={},
        detector=None,
        results=None,
        event_table=empty if event_table is None else event_table,
        event_arm_table=empty if event_arm_table is None else event_arm_table,
        event_sequence_table=empty,
        stop_table=empty if stop_table is None else stop_table,
        trials=empty,
        port_experiences=_experiences(empty),
        track_graph=None,
        time_slice=slice(*window),
        arm_palette=patch_grouped_arm_palette(),
        env=None,
        data_type="clusterless",
        event_patch_table=event_patch_table,
    )


def _arm_table(values: dict[str, list[float]], n_events: int = 1) -> pd.DataFrame:
    """One event x nine arms, with caller-supplied covariate columns."""
    n = 9 * n_events
    base = {
        "event_id": np.repeat(np.arange(n_events), 9),
        "candidate_arm": np.tile(np.arange(9), n_events),
        "event_start_s": np.repeat(np.arange(n_events, dtype=float), 9) + 1.0,
        "event_end_s": np.repeat(np.arange(n_events, dtype=float), 9) + 1.5,
    }
    for col, vals in values.items():
        assert len(vals) == n, f"{col}: expected {n} values"
        base[col] = np.asarray(vals, dtype=float)
    return pd.DataFrame(base)


def _rectangles_centered_on(ax, row: float):
    """Rectangle artists whose vertical midpoint is ``row`` in data units."""
    return [
        patch
        for patch in ax.patches
        if np.isclose(patch.get_y() + patch.get_height() / 2.0, row)
    ]


def _same_rgb(actual, expected) -> bool:
    return bool(
        np.allclose(np.asarray(to_rgba(actual))[:3], np.asarray(to_rgba(expected))[:3])
    )


def _colorbar_mappable(fig, main_ax):
    """Return the sole quantitative colorbar's mappable for one panel."""
    colorbar_axes = [ax for ax in fig.axes if ax is not main_ax]
    assert len(colorbar_axes) == 1
    colorbar = getattr(colorbar_axes[0], "_colorbar", None)
    assert colorbar is not None
    return colorbar.mappable


def _horizontal_segments_for_color(ax, color: str):
    """Horizontal ``(x0, x1, y)`` segments represented by real artists."""
    segments: list[tuple[float, float, float]] = []
    for line in ax.lines:
        if not _same_rgb(line.get_color(), color):
            continue
        vertices = np.asarray(line.get_path().vertices, dtype=float)
        for left, right in zip(vertices[:-1], vertices[1:], strict=True):
            if (
                np.isfinite(left).all()
                and np.isfinite(right).all()
                and np.isclose(left[1], right[1])
                and not np.isclose(left[0], right[0])
            ):
                segments.append(
                    (
                        float(min(left[0], right[0])),
                        float(max(left[0], right[0])),
                        float(left[1]),
                    )
                )
    for collection in ax.collections:
        if not isinstance(collection, LineCollection):
            continue
        colors = collection.get_colors()
        if len(colors) == 0 or not any(_same_rgb(c, color) for c in colors):
            continue
        for vertices in collection.get_segments():
            vertices = np.asarray(vertices, dtype=float)
            if (
                len(vertices) == 2
                and np.isfinite(vertices).all()
                and np.isclose(vertices[0, 1], vertices[1, 1])
            ):
                segments.append(
                    (
                        float(min(vertices[:, 0])),
                        float(max(vertices[:, 0])),
                        float(vertices[0, 1]),
                    )
                )
    return segments


def _public_plot_inputs() -> dict:
    """Complete minimal public-API input using a real xarray time axis."""
    event_table = pd.DataFrame(
        {
            "event_id": [0],
            "event_start_s": [1.0],
            "event_end_s": [1.5],
            "event_midpoint_s": [1.25],
            "next_arm": [1.0],
        }
    )
    trials = pd.DataFrame(
        {
            "start_time": [0.0, 2.0],
            "end_time": [0.5, 3.0],
            "from_well": [0, 0],
            "to_well": [0, 1],
            "is_reward": [False, False],
            "is_patch_change": [False, False],
        }
    )
    return {
        "data": {},
        "detector": SimpleNamespace(environments=[None]),
        "results": xr.Dataset(coords={"time": np.array([0.0, 10.0])}),
        "event_table": event_table,
        "event_arm_table": _arm_table(
            {"reward_belief_mean_candidate_arm_preoutcome": [0.5] * 9}
        ),
        "event_sequence_table": pd.DataFrame(),
        "stop_table": pd.DataFrame(),
        "trials": trials,
        "port_experiences": _experiences(trials),
        "track_graph": _bandit_graph(),
    }


class TestBehavioralModelPanels:
    def test_q_arm_uses_fixed_probability_scale(self):
        """q_arm is a Beta mean, so the scale must be pinned to [0, 1]."""
        from continuum_swr_replay.visualization import session_tables as mod

        # All values low: an inferred scale would stretch them to full range.
        eat = _arm_table(
            {"reward_belief_mean_candidate_arm_preoutcome": [0.1] * 8 + [0.2]}
        )
        fig, ax = plt.subplots()
        mod._plot_reward_belief_mean_candidate_arm(
            ax, _ctx_for_covariates(event_arm_table=eat)
        )
        (cbar_ax,) = [a for a in fig.axes if a is not ax]
        assert cbar_ax.get_ylim() == (0.0, 1.0)
        plt.close(fig)

    def test_arm_uncertainty_scale_is_pinned_to_beta_variance_max(self):
        """This fitted model keeps alpha,beta >= 1, so variance is <= 1/12."""
        from continuum_swr_replay.visualization import session_tables as mod

        eat = _arm_table(
            {"reward_belief_variance_candidate_arm_preoutcome": [0.001] * 9}
        )
        fig, ax = plt.subplots()
        mod._plot_reward_belief_variance_candidate_arm(
            ax, _ctx_for_covariates(event_arm_table=eat)
        )
        (cbar_ax,) = [a for a in fig.axes if a is not ax]
        assert cbar_ax.get_ylim() == pytest.approx((0.0, 1.0 / 12.0))
        plt.close(fig)

    def test_q_arm_and_p_reward_arm_share_a_scale(self):
        """Nominal vs believed are only comparable by eye on one scale."""
        from continuum_swr_replay.visualization import session_tables as mod

        eat = _arm_table(
            {
                "reward_belief_mean_candidate_arm_preoutcome": [0.3] * 9,
                "task_reward_probability_candidate_arm": [0.8] * 9,
            }
        )
        lims = []
        for plotter in (
            mod._plot_reward_belief_mean_candidate_arm,
            mod._plot_task_reward_probability_candidate_arm,
        ):
            fig, ax = plt.subplots()
            plotter(ax, _ctx_for_covariates(event_arm_table=eat))
            (cbar_ax,) = [a for a in fig.axes if a is not ax]
            lims.append(cbar_ax.get_ylim())
            plt.close(fig)
        assert lims[0] == lims[1]

    def test_probability_covariates_share_cividis_unit_interval_mapping(self):
        """Comparable probabilities use one perceptually uniform mapping."""
        from continuum_swr_replay.visualization import session_tables as mod

        values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.0, 0.0, 0.0]
        eat = _arm_table(
            {
                "reward_belief_mean_candidate_arm_preoutcome": values,
                "task_reward_probability_candidate_arm": values,
                "reward_fraction_candidate_arm_last10_visits": values,
            }
        )
        mappings = []
        for plotter in (
            mod._plot_reward_belief_mean_candidate_arm,
            mod._plot_task_reward_probability_candidate_arm,
            mod._plot_recent_reward_rate,
        ):
            fig, ax = plt.subplots()
            plotter(ax, _ctx_for_covariates(event_arm_table=eat))
            mappable = _colorbar_mappable(fig, ax)
            mappings.append(
                (mappable.cmap.name, mappable.norm.vmin, mappable.norm.vmax)
            )
            plt.close(fig)

        assert mappings == [("cividis", 0.0, 1.0)] * 3

    def test_trunks_are_structural_na_not_zero_beliefs(self):
        """A trunk has no reward port/Beta posterior despite its stored zero."""
        from continuum_swr_replay.visualization import session_tables as mod

        eat = _arm_table({"reward_belief_mean_candidate_arm_preoutcome": [0.0] * 9})
        fig, ax = plt.subplots()
        mod._plot_reward_belief_mean_candidate_arm(
            ax, _ctx_for_covariates(event_arm_table=eat)
        )

        leaf_zero = _rectangles_centered_on(ax, 0)
        trunk_na = _rectangles_centered_on(ax, 6)
        assert len(leaf_zero) == len(trunk_na) == 1
        assert not np.allclose(
            leaf_zero[0].get_facecolor(), trunk_na[0].get_facecolor()
        )
        plt.close(fig)

    def test_missing_leaf_belief_has_an_explicit_cell(self):
        from continuum_swr_replay.visualization import session_tables as mod

        eat = _arm_table(
            {
                "reward_belief_mean_candidate_arm_preoutcome": [np.nan]
                + [0.25] * 5
                + [0.0] * 3
            }
        )
        fig, ax = plt.subplots()
        mod._plot_reward_belief_mean_candidate_arm(
            ax, _ctx_for_covariates(event_arm_table=eat)
        )

        missing = _rectangles_centered_on(ax, 0)
        observed = _rectangles_centered_on(ax, 1)
        assert len(missing) == len(observed) == 1
        assert not np.allclose(missing[0].get_facecolor(), observed[0].get_facecolor())
        plt.close(fig)


class TestDivergingCenter:
    def test_center_makes_limits_symmetric(self):
        """visit_recency_trials_event_leaf_centered must put 0 at the colormap midpoint."""
        from continuum_swr_replay.visualization import session_tables as mod

        # Asymmetric data: -1 .. +9. Symmetric limits should be -9 .. +9.
        eat = _arm_table(
            {"visit_recency_trials_event_leaf_centered": [-1.0] + [9.0] * 8}
        )
        fig, ax = plt.subplots()
        mod._plot_visit_recency_trials_event_leaf_centered(
            ax, _ctx_for_covariates(event_arm_table=eat)
        )
        (cbar_ax,) = [a for a in fig.axes if a is not ax]
        lo, hi = cbar_ax.get_ylim()
        assert lo == pytest.approx(-hi)
        assert hi == pytest.approx(9.0)
        plt.close(fig)

    def test_center_survives_all_one_sided_data(self):
        """TwoSlopeNorm would raise here; symmetric widening must not."""
        from continuum_swr_replay.visualization import session_tables as mod

        eat = _arm_table({"visit_recency_trials_event_leaf_centered": [5.0] * 9})
        fig, ax = plt.subplots()
        mod._plot_visit_recency_trials_event_leaf_centered(
            ax, _ctx_for_covariates(event_arm_table=eat)
        )
        (cbar_ax,) = [a for a in fig.axes if a is not ax]
        lo, hi = cbar_ax.get_ylim()
        assert lo == pytest.approx(-5.0) and hi == pytest.approx(5.0)
        plt.close(fig)

    def test_all_zero_data_remain_symmetric_and_use_neutral_color(self):
        from continuum_swr_replay.visualization import session_tables as mod

        eat = _arm_table({"visit_recency_trials_event_leaf_centered": [0.0] * 9})
        fig, ax = plt.subplots()
        mod._plot_visit_recency_trials_event_leaf_centered(
            ax, _ctx_for_covariates(event_arm_table=eat)
        )

        (cbar_ax,) = [a for a in fig.axes if a is not ax]
        lo, hi = cbar_ax.get_ylim()
        assert lo < 0.0 < hi
        assert lo == pytest.approx(-hi)
        neutral = plt.get_cmap("coolwarm")(0.5)
        assert len(ax.patches) == 9
        for arm in range(6):
            (patch,) = _rectangles_centered_on(ax, arm)
            assert np.allclose(patch.get_facecolor(), neutral)
        (trunk,) = _rectangles_centered_on(ax, 6)
        assert not np.allclose(trunk.get_facecolor(), neutral)
        plt.close(fig)

    def test_scale_is_stable_across_windows_using_full_session_table(self):
        """Window selection must not recolor the same scientific quantity."""
        from continuum_swr_replay.visualization import session_tables as mod

        event_one = [-2.0, -1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0, 0.0]
        event_two = [-10.0, -5.0, 0.0, 5.0, 10.0, 4.0, 0.0, 0.0, 0.0]
        eat = _arm_table(
            {"visit_recency_trials_event_leaf_centered": [*event_one, *event_two]},
            n_events=2,
        )
        limits = []
        for window in ((0.5, 1.6), (1.9, 2.6)):
            fig, ax = plt.subplots()
            mod._plot_visit_recency_trials_event_leaf_centered(
                ax,
                _ctx_for_covariates(event_arm_table=eat, window=window),
            )
            mappable = _colorbar_mappable(fig, ax)
            limits.append((mappable.norm.vmin, mappable.norm.vmax))
            plt.close(fig)

        assert limits == [(-10.0, 10.0), (-10.0, 10.0)]

    def test_recency_uses_disclosed_robust_limits_for_sentinel_outlier(self):
        """A never-visited sentinel must not wash out ordinary recency."""
        from continuum_swr_replay.visualization import session_tables as mod

        values = [-2.0, -1.0, 0.0, 1.0, 2.0, 100.0, 0.0, 0.0, 0.0]
        eat = _arm_table({"visit_recency_trials_event_leaf_centered": values})
        fig, ax = plt.subplots()
        mod._plot_visit_recency_trials_event_leaf_centered(
            ax, _ctx_for_covariates(event_arm_table=eat)
        )

        mappable = _colorbar_mappable(fig, ax)
        assert mappable.norm.vmin == pytest.approx(-mappable.norm.vmax)
        assert 2.0 < mappable.norm.vmax < 100.0
        (cbar_ax,) = [axis for axis in fig.axes if axis is not ax]
        assert cbar_ax._colorbar.extend == "max"
        assert "saturation = clipped" in " ".join(text.get_text() for text in ax.texts)
        plt.close(fig)


class TestArmIndicatorPanels:
    def test_zero_is_visible_and_distinct_from_missing(self):
        """A measured 0 is a model row; NaN means the event is excluded."""
        from continuum_swr_replay.visualization import session_tables as mod

        eat = _arm_table({"is_current_arm": [1.0, 0.0] + [np.nan] * 7})
        fig, ax = plt.subplots()
        mod._plot_is_current(ax, _ctx_for_covariates(event_arm_table=eat))

        one = _rectangles_centered_on(ax, 0)
        zero = _rectangles_centered_on(ax, 1)
        missing = _rectangles_centered_on(ax, 2)
        assert len(one) == len(zero) == len(missing) == 1
        assert not np.allclose(one[0].get_facecolor(), zero[0].get_facecolor())
        assert not np.allclose(zero[0].get_facecolor(), missing[0].get_facecolor())
        plt.close(fig)

    def test_missing_source_reference_overlays_stored_zero_design(self):
        """Canonical builders store all-zero indicators for a NaN reference."""
        from continuum_swr_replay.visualization import session_tables as mod

        missing_ref = _arm_table(
            {"is_current_arm": [0.0] * 9, "occupancy_well_arm": [np.nan] * 9}
        )
        valid_ref = _arm_table(
            {"is_current_arm": [0.0] * 9, "occupancy_well_arm": [5.0] * 9}
        )
        colors = []
        for eat in (missing_ref, valid_ref):
            fig, ax = plt.subplots()
            mod._plot_is_current(ax, _ctx_for_covariates(event_arm_table=eat))
            cell = _rectangles_centered_on(ax, 0)
            assert len(cell) == 1
            colors.append(cell[0].get_facecolor())
            plt.close(fig)
        assert not np.allclose(colors[0], colors[1])

    def test_empty_window_shows_placeholder(self):
        from continuum_swr_replay.visualization import session_tables as mod

        eat = _arm_table({"is_prev_arm": [1.0] * 9})
        ctx = _ctx_for_covariates(event_arm_table=eat, window=(100.0, 200.0))
        fig, ax = plt.subplots()
        mod._plot_is_prev(ax, ctx)
        assert any("no events" in t.get_text() for t in ax.texts)
        assert len(ax.patches) == 0
        plt.close(fig)


class TestPatchPanels:
    @staticmethod
    def _patch_tables(values: dict[str, list[float]]):
        ept = pd.DataFrame(
            {
                "event_id": [0, 0, 0],
                "patch": [1, 2, 3],
                **values,
            }
        )
        et = pd.DataFrame(
            {"event_id": [0], "event_start_s": [1.0], "event_end_s": [1.5]}
        )
        return ept, et

    def test_missing_patch_table_raises_actionable_error(self):
        from continuum_swr_replay.visualization import session_tables as mod

        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="event_patch_table"):
            mod._plot_reward_belief_mean_candidate_arm_preoutcome_patch(
                ax, _ctx_for_covariates()
            )
        plt.close(fig)

    def test_unknown_patch_column_names_the_column(self):
        from continuum_swr_replay.visualization import session_tables as mod

        column = "reward_belief_mean_candidate_arm_preoutcome_patch"
        ept, et = self._patch_tables({column: [0.1, 0.2, 0.3]})
        ctx = _ctx_for_covariates(
            event_patch_table=ept.drop(columns=[column]), event_table=et
        )
        fig, ax = plt.subplots()
        with pytest.raises(KeyError, match=column):
            mod._plot_reward_belief_mean_candidate_arm_preoutcome_patch(ax, ctx)
        plt.close(fig)

    def test_three_patch_rows_are_drawn(self):
        from continuum_swr_replay.visualization import session_tables as mod

        column = "reward_belief_mean_candidate_arm_preoutcome_patch"
        ept, et = self._patch_tables({column: [0.1, 0.5, 0.9]})
        ctx = _ctx_for_covariates(event_patch_table=ept, event_table=et)
        fig, ax = plt.subplots()
        mod._plot_reward_belief_mean_candidate_arm_preoutcome_patch(ax, ctx)
        assert len(ax.patches) == 3
        assert ax.get_ylim() == (0.5, 3.5)
        plt.close(fig)

    def test_patch_indicator_draws_valid_zeros(self):
        from continuum_swr_replay.visualization import session_tables as mod

        ept, et = self._patch_tables({"is_current_patch": [1.0, 0.0, 0.0]})
        ctx = _ctx_for_covariates(event_patch_table=ept, event_table=et)
        fig, ax = plt.subplots()
        mod._plot_is_current_patch(ax, ctx)
        assert len(ax.patches) == 3
        one = _rectangles_centered_on(ax, 1)
        zero = _rectangles_centered_on(ax, 2)
        assert len(one) == len(zero) == 1
        assert not np.allclose(one[0].get_facecolor(), zero[0].get_facecolor())
        plt.close(fig)

    def test_patch_indicator_marks_missing_source_reference(self):
        from continuum_swr_replay.visualization import session_tables as mod

        missing_ept, missing_et = self._patch_tables(
            {"is_current_patch": [0.0, 0.0, 0.0]}
        )
        missing_et["current_patch"] = np.nan
        valid_ept, valid_et = self._patch_tables({"is_current_patch": [1.0, 0.0, 0.0]})
        valid_et["current_patch"] = 1.0

        colors = []
        for ept, et in ((missing_ept, missing_et), (valid_ept, valid_et)):
            fig, ax = plt.subplots()
            mod._plot_is_current_patch(
                ax,
                _ctx_for_covariates(event_patch_table=ept, event_table=et),
            )
            patch_two = _rectangles_centered_on(ax, 2)
            assert len(patch_two) == 1
            colors.append(patch_two[0].get_facecolor())
            plt.close(fig)
        assert not np.allclose(colors[0], colors[1])

    def test_orphan_patch_event_is_rejected(self):
        from continuum_swr_replay.visualization import session_tables as mod

        column = "reward_belief_mean_candidate_arm_preoutcome_patch"
        ept, et = self._patch_tables({column: [0.1, 0.5, 0.9]})
        orphan = pd.DataFrame({"event_id": [99], "patch": [1], column: [0.4]})
        ept = pd.concat([ept, orphan], ignore_index=True)
        ctx = _ctx_for_covariates(event_patch_table=ept, event_table=et)
        fig, ax = plt.subplots()
        try:
            with pytest.raises(ValueError):
                mod._plot_reward_belief_mean_candidate_arm_preoutcome_patch(ax, ctx)
        finally:
            plt.close(fig)

    def test_missing_patch_row_is_rejected(self):
        from continuum_swr_replay.visualization import session_tables as mod

        ept, et = self._patch_tables(
            {"reward_belief_mean_candidate_arm_preoutcome_patch": [0.1, 0.5, 0.9]}
        )
        ctx = _ctx_for_covariates(
            event_patch_table=ept[ept["patch"] != 3], event_table=et
        )
        fig, ax = plt.subplots()
        try:
            with pytest.raises(ValueError):
                mod._plot_reward_belief_mean_candidate_arm_preoutcome_patch(ax, ctx)
        finally:
            plt.close(fig)

    def test_duplicate_event_patch_row_is_rejected(self):
        from continuum_swr_replay.visualization import session_tables as mod

        ept, et = self._patch_tables(
            {"reward_belief_mean_candidate_arm_preoutcome_patch": [0.1, 0.5, 0.9]}
        )
        ept = pd.concat([ept, ept.iloc[[0]]], ignore_index=True)
        ctx = _ctx_for_covariates(event_patch_table=ept, event_table=et)
        fig, ax = plt.subplots()
        try:
            with pytest.raises(ValueError):
                mod._plot_reward_belief_mean_candidate_arm_preoutcome_patch(ax, ctx)
        finally:
            plt.close(fig)


class TestStopGrainPanels:
    @staticmethod
    def _stop_table(n_stops: int = 2) -> pd.DataFrame:
        from continuum_swr_replay.analysis.replay.model_specs import N_REWARD_LAGS

        cols = {
            "stop_start": np.arange(n_stops, dtype=float),
            "stop_end": np.arange(n_stops, dtype=float) + 0.5,
            "reward_fraction_last10_known_trials": np.full(n_stops, 0.4),
            "did_last_completed_trip_switch_patch": np.array([1.0, 0.0][:n_stops]),
        }
        for j in range(1, N_REWARD_LAGS + 1):
            # Rewarded at odd lags only, and NaN on the second stop's lag 10
            # so the ragged-history case is exercised.
            cols[f"known_reward_outcome_lag_{j}"] = np.full(n_stops, float(j % 2))
        cols["known_reward_outcome_lag_10"] = np.array([0.0, np.nan][:n_stops])
        return pd.DataFrame(cols)

    def test_reward_lag_kernel_draws_distinct_cells_for_reward_and_omission(self):
        from continuum_swr_replay.visualization import session_tables as mod

        st = self._stop_table(n_stops=1)
        fig, ax = plt.subplots()
        mod._plot_reward_lag_kernel(ax, _ctx_for_covariates(stop_table=st))
        # Every observed lag gets a cell; outcome is encoded by distinct colors.
        assert len(ax.patches) == 10
        rewarded = _rectangles_centered_on(ax, 1)
        omitted = _rectangles_centered_on(ax, 2)
        assert len(rewarded) == len(omitted) == 1
        assert not np.allclose(rewarded[0].get_facecolor(), omitted[0].get_facecolor())
        # Lag axis is inverted so lag 1 sits at the top.
        assert ax.get_ylim()[0] > ax.get_ylim()[1]
        plt.close(fig)

    def test_nan_lag_is_distinct_from_zero(self):
        from continuum_swr_replay.visualization import session_tables as mod

        st = self._stop_table(n_stops=2)
        fig, ax = plt.subplots()
        mod._plot_reward_lag_kernel(
            ax, _ctx_for_covariates(stop_table=st, window=(0.0, 10.0))
        )
        # All twenty cells are explicit; lag 10 distinguishes zero from missing.
        assert len(ax.patches) == 20
        lag10 = _rectangles_centered_on(ax, 10)
        assert len(lag10) == 2
        lag10 = sorted(lag10, key=lambda patch: patch.get_x())
        assert lag10[0].get_x() == pytest.approx(st["stop_start"].iloc[0])
        assert lag10[0].get_width() == pytest.approx(
            st["stop_end"].iloc[0] - st["stop_start"].iloc[0]
        )
        assert not np.allclose(lag10[0].get_facecolor(), lag10[1].get_facecolor())
        plt.close(fig)

    def test_reward_rate_segments_match_stops_without_bridging_gaps(self):
        from continuum_swr_replay.visualization import session_tables as mod

        st = pd.DataFrame(
            {
                "stop_start": [0.0, 10.0, 20.0],
                "stop_end": [2.0, 12.0, 22.0],
                "reward_fraction_last10_known_trials": [0.2, 0.8, np.nan],
            }
        )
        fig, ax = plt.subplots()
        mod._plot_reward_rate_trial_k10(
            ax, _ctx_for_covariates(stop_table=st, window=(0.0, 22.0))
        )

        segments = sorted(_horizontal_segments_for_color(ax, "#8c564b"))
        assert segments == pytest.approx([(0.0, 2.0, 0.2), (10.0, 12.0, 0.8)])
        missing = [
            patch for patch in ax.patches if patch.get_x() == pytest.approx(20.0)
        ]
        assert len(missing) == 1
        assert missing[0].get_width() == pytest.approx(2.0)
        plt.close(fig)

    def test_missing_stop_column_raises(self):
        from continuum_swr_replay.visualization import session_tables as mod

        st = self._stop_table().drop(columns=["reward_fraction_last10_known_trials"])
        fig, ax = plt.subplots()
        with pytest.raises(KeyError, match="reward_fraction_last10_known_trials"):
            mod._plot_reward_rate_trial_k10(ax, _ctx_for_covariates(stop_table=st))
        plt.close(fig)

    def test_is_after_switch_distinguishes_true_and_false_stops(self):
        from continuum_swr_replay.visualization import session_tables as mod

        st = self._stop_table(n_stops=2)
        fig, ax = plt.subplots()
        mod._plot_last_completed_trip_switch(ax, _ctx_for_covariates(stop_table=st))
        assert len(ax.patches) == 2
        cells = sorted(ax.patches, key=lambda patch: patch.get_x())
        assert not np.allclose(cells[0].get_facecolor(), cells[1].get_facecolor())
        plt.close(fig)

    def test_is_after_switch_marks_missing_stop(self):
        from continuum_swr_replay.visualization import session_tables as mod

        st = self._stop_table(n_stops=2)
        st.loc[1, "did_last_completed_trip_switch_patch"] = np.nan
        fig, ax = plt.subplots()
        mod._plot_last_completed_trip_switch(ax, _ctx_for_covariates(stop_table=st))

        cells = sorted(ax.patches, key=lambda patch: patch.get_x())
        assert len(cells) == 2
        assert cells[1].get_hatch() == "///"
        plt.close(fig)


class TestPanelGroupsAndRegistryIntegrity:
    def test_every_registered_panel_has_a_height(self):
        from continuum_swr_replay.visualization import session_tables as mod

        assert set(mod._PANEL_REGISTRY) == set(mod._PANEL_HEIGHTS)

    def test_panel_groups_resolve(self):
        from continuum_swr_replay.visualization import session_tables as mod

        for group in (
            mod.BEHAVIORAL_MODEL_PANELS,
            mod.ARM_COVARIATE_PANELS,
            mod.PATCH_COVARIATE_PANELS,
            mod.RATE_COVARIATE_PANELS,
        ):
            assert group, "panel group must not be empty"
            for name in group:
                assert name in mod._PANEL_REGISTRY, name

    def test_default_panels_exclude_covariate_panels(self):
        """A bare call stays a detector-style overview."""
        from continuum_swr_replay.visualization import session_tables as mod

        covariates = set(mod.ARM_COVARIATE_PANELS) | set(mod.PATCH_COVARIATE_PANELS)
        assert not (set(DEFAULT_PANELS) & covariates)

    def test_every_fitted_predictor_is_reachable(self):
        """Guard against a new model predictor landing with no way to see it."""
        from continuum_swr_replay.analysis.replay import model_specs as ms
        from continuum_swr_replay.visualization import session_tables as mod

        registries = [
            obj
            for name in dir(ms)
            if isinstance(obj := getattr(ms, name), dict)
            and obj
            and all(hasattr(v, "predictors") for v in obj.values())
        ]
        predictors = {
            p for reg in registries for s in reg.values() for p in s.predictors
        }
        panels = set(mod._PANEL_REGISTRY)

        def reachable(p: str) -> bool:
            # Documented exclusions: see the module docstring.
            return (
                p in panels
                or p.startswith("is_candidate_arm_")  # drawn by arm_markers
                or p
                == "did_last_completed_trip_switch_patch"  # -> is_after_switch_stop
                or p.startswith("known_reward_outcome_lag_")  # -> reward-lag kernel
                or p.startswith("reward_history_")  # basis of the drawn lag vector
                or p.startswith("swr_x_")  # base panel x is_swr row
                or p.endswith("_z")  # affine rescale of the raw column
                or p.startswith("replay_fraction")  # choice grain, not a table input
            )

        unreachable = sorted(p for p in predictors if not reachable(p))
        assert not unreachable, f"fitted predictors with no panel: {unreachable}"


class TestPublicPlotSessionTables:
    def test_real_canonical_builders_feed_public_plot_unchanged(self):
        """Arm, patch, and stop artists consume canonical builder columns.

        This is intentionally a small integration test, not a mock-based call
        assertion: the same production builders used by ``prepare_data`` create
        all plotted tables, and the resulting artists are checked against those
        tables' values and time intervals.
        """
        from continuum_swr_replay.analysis.replay.tables_arm_patch import (
            build_event_arm_table,
            build_event_patch_table,
        )
        from continuum_swr_replay.analysis.replay.tables_event import build_event_table
        from continuum_swr_replay.analysis.replay.tables_stop import build_stop_table
        from tests.unit.test_replay_tables import (
            _bandit_graph as production_test_graph,
        )
        from tests.unit.test_replay_tables import (
            _EnvStub,
            _tiny_dot_with_two_events,
        )

        dot = _tiny_dot_with_two_events()
        env = _EnvStub()
        graph = production_test_graph()
        trials = pd.DataFrame(
            {
                "start_time": [0.0, 4.0, 8.0],
                "goal_arrival_time": [0.8, 4.8, 8.8],
                "end_time": [1.0, 5.0, 9.0],
                "from_well": [0, 0, 1],
                "to_well": [0, 1, 2],
                "from_patch": [1, 1, 1],
                "to_patch": [1, 1, 2],
                "is_reward": [True, False, True],
                "is_patch_change": [False, False, True],
            }
        )

        port_experiences = _experiences(trials)
        stop_table = build_stop_table(dot, trials, port_experiences, graph)
        event_table = build_event_table(dot, env, trials, port_experiences, graph)
        contingency = pd.DataFrame({"stop_id": stop_table["stop_id"]})
        for arm in range(6):
            contingency[f"task_reward_probability_port_{arm}"] = 0.1 * (arm + 1)
            contingency[f"reward_belief_mean_port_{arm}"] = 0.05 * (arm + 1)
            contingency[f"reward_belief_variance_port_{arm}"] = 0.005 * (arm + 1)
        event_arm_table = build_event_arm_table(
            event_table,
            dot,
            env,
            trials,
            port_experiences,
            graph,
            contingency=contingency,
        )
        event_patch_table = build_event_patch_table(event_table, event_arm_table)
        event_sequence_table = pd.DataFrame()

        frames = {
            "event_table": event_table,
            "event_arm_table": event_arm_table,
            "event_patch_table": event_patch_table,
            "event_sequence_table": event_sequence_table,
            "stop_table": stop_table,
            "trials": trials,
        }
        before = {name: frame.copy(deep=True) for name, frame in frames.items()}

        panels = (
            "reward_belief_mean_candidate_arm_preoutcome",
            "reward_belief_mean_candidate_arm_preoutcome_patch",
            "reward_fraction_last10_known_trials",
        )
        fig, mismatch = plot_session_tables(
            data={"environment": env},
            detector=None,
            results=xr.Dataset(coords={"time": dot.index.to_numpy(dtype=float)}),
            event_table=event_table,
            event_arm_table=event_arm_table,
            event_patch_table=event_patch_table,
            event_sequence_table=event_sequence_table,
            stop_table=stop_table,
            trials=trials,
            port_experiences=port_experiences,
            track_graph=graph,
            time_window=(float(dot.index.min()), float(dot.index.max())),
            panels=panels,
        )

        for name, frame in frames.items():
            pd.testing.assert_frame_equal(frame, before[name])

        arm_ax, patch_ax, rate_ax = fig.axes[: len(panels)]
        colorbar_mappables = [
            ax._colorbar.mappable
            for ax in fig.axes
            if getattr(ax, "_colorbar", None) is not None
        ]
        assert len(colorbar_mappables) == 2

        first_event_id = event_table["event_id"].iloc[0]
        arm_row = event_arm_table[
            event_arm_table["event_id"].eq(first_event_id)
            & event_arm_table["candidate_arm"].eq(2)
        ].iloc[0]
        arm_cell = min(
            _rectangles_centered_on(arm_ax, 2), key=lambda cell: cell.get_x()
        )
        assert arm_cell.get_x() == pytest.approx(arm_row["event_start_s"])
        assert arm_cell.get_width() == pytest.approx(
            arm_row["event_end_s"] - arm_row["event_start_s"]
        )
        arm_map = colorbar_mappables[0]
        assert np.allclose(
            arm_cell.get_facecolor(),
            arm_map.cmap(
                arm_map.norm(
                    float(arm_row["reward_belief_mean_candidate_arm_preoutcome"])
                )
            ),
        )

        patch_row = event_patch_table[
            event_patch_table["event_id"].eq(first_event_id)
            & event_patch_table["patch"].eq(1)
        ].iloc[0]
        event_row = event_table[event_table["event_id"].eq(first_event_id)].iloc[0]
        patch_cell = min(
            _rectangles_centered_on(patch_ax, 1), key=lambda cell: cell.get_x()
        )
        assert patch_cell.get_x() == pytest.approx(event_row["event_start_s"])
        assert patch_cell.get_width() == pytest.approx(
            event_row["event_end_s"] - event_row["event_start_s"]
        )
        patch_map = colorbar_mappables[1]
        assert np.allclose(
            patch_cell.get_facecolor(),
            patch_map.cmap(
                patch_map.norm(
                    float(
                        patch_row["reward_belief_mean_candidate_arm_preoutcome_patch"]
                    )
                )
            ),
        )

        expected_rate_segments = sorted(
            (
                float(row.stop_start),
                float(row.stop_end),
                float(row.reward_fraction_last10_known_trials),
            )
            for row in stop_table.itertuples()
            if np.isfinite(row.reward_fraction_last10_known_trials)
        )
        actual_rate_segments = sorted(
            _horizontal_segments_for_color(rate_ax, "#8c564b")
        )
        assert actual_rate_segments == pytest.approx(expected_rate_segments)
        assert mismatch["match"].all()
        plt.close(fig)

    def test_covariate_panel_smoke_uses_public_entry_point(self):
        fig, mismatch = plot_session_tables(
            **_public_plot_inputs(),
            time_window=(0.0, 4.0),
            panels=("reward_belief_mean_candidate_arm_preoutcome",),
        )

        assert fig.axes[0].get_ylabel() == "Beta belief q\npre-outcome"
        assert fig.axes[0].get_xlim() == pytest.approx((0.0, 4.0))
        assert len(fig.axes[0].patches) >= 9
        assert len(mismatch) == 1
        assert mismatch["match"].all()
        plt.close(fig)

    @pytest.mark.parametrize("time_window", [(20.0, 30.0), (8.0, 2.0)])
    def test_invalid_time_window_raises(self, time_window):
        with pytest.raises(ValueError, match="time_window"):
            fig, _ = plot_session_tables(
                **_public_plot_inputs(),
                time_window=time_window,
                panels=("reward_belief_mean_candidate_arm_preoutcome",),
            )
            plt.close(fig)

    @pytest.mark.parametrize(
        "panels", [(), "reward_belief_mean_candidate_arm_preoutcome"]
    )
    def test_invalid_panels_raise(self, panels):
        with pytest.raises((TypeError, ValueError), match="panels"):
            plot_session_tables(**_public_plot_inputs(), panels=panels)

    def test_behavior_only_plot_accepts_dataarray_without_detector(self):
        inputs = _public_plot_inputs()
        inputs["detector"] = None
        inputs["data"] = {"environment": None}
        inputs["results"] = xr.DataArray(
            np.zeros(2),
            coords={"time": np.array([0.0, 10.0])},
            dims=("time",),
        )

        fig, mismatch = plot_session_tables(
            **inputs,
            time_window=(0.0, 4.0),
            panels=("reward_belief_mean_candidate_arm_preoutcome",),
        )

        assert mismatch["match"].all()
        plt.close(fig)

    def test_relative_time_reference_and_supplied_labels_are_used(self):
        """Epoch timestamps remain readable in a narrow audit window."""
        reference = 1_581_637_647.0
        inputs = _public_plot_inputs()
        inputs["results"] = xr.Dataset(
            coords={"time": reference + np.array([0.0, 10.0])}
        )
        for frame_name, columns in (
            ("event_table", ("event_start_s", "event_end_s", "event_midpoint_s")),
            ("event_arm_table", ("event_start_s", "event_end_s")),
            ("trials", ("start_time", "end_time")),
            (
                "port_experiences",
                ("arrival_time", "departure_time", "outcome_known_time"),
            ),
        ):
            frame = inputs[frame_name].copy()
            frame.loc[:, list(columns)] = frame.loc[:, list(columns)] + reference
            inputs[frame_name] = frame

        xlabel = "Time from audit start [s]"
        title = "Behavioral covariate audit"
        fig, _ = plot_session_tables(
            **inputs,
            time_window=(reference, reference + 4.0),
            panels=("reward_belief_mean_candidate_arm_preoutcome",),
            time_reference=reference,
            xlabel=xlabel,
            title=title,
        )
        fig.canvas.draw()

        ax = fig.axes[0]
        labels = [tick.get_text() for tick in ax.get_xticklabels() if tick.get_text()]
        parsed = np.array([float(label.replace("−", "-")) for label in labels])
        positions = ax.get_xticks()[: len(parsed)]
        np.testing.assert_allclose(parsed, positions - reference, rtol=0, atol=1e-9)
        assert 0.0 in parsed
        assert np.allclose(np.diff(parsed), 0.5)
        reference_lines = [
            line for line in ax.lines if line.get_label() == "t = 0 reference"
        ]
        assert len(reference_lines) == 1
        np.testing.assert_allclose(reference_lines[0].get_xdata(), reference)
        assert max(map(len, labels)) <= 6
        assert ax.get_xlabel() == xlabel
        assert fig._suptitle is not None
        assert fig._suptitle.get_text() == title
        plt.close(fig)

    def test_arm_heatmaps_have_separator_and_one_figure_level_semantic_key(self):
        """Leaf/trunk and missingness semantics are visible without repetition."""
        inputs = _public_plot_inputs()
        values = [0.1, 0.25, np.nan, 0.5, 0.75, 0.9, 0.0, 0.0, 0.0]
        inputs["event_arm_table"] = _arm_table(
            {
                "reward_belief_mean_candidate_arm_preoutcome": values,
                "task_reward_probability_candidate_arm": values,
                "reward_fraction_candidate_arm_last10_visits": values,
            }
        )
        panels = (
            "reward_belief_mean_candidate_arm_preoutcome",
            "task_reward_probability_candidate_arm",
            "reward_fraction_candidate_arm_last10_visits",
        )

        fig, _ = plot_session_tables(
            **inputs,
            time_window=(0.0, 4.0),
            panels=panels,
        )

        main_axes = fig.axes[: len(panels)]
        for ax in main_axes:
            separators = [
                line
                for line in ax.lines
                if np.allclose(np.asarray(line.get_ydata(), dtype=float), 5.5)
            ]
            assert len(separators) == 1

        axis_semantics = [
            text.get_text().lower()
            for ax in main_axes
            for text in ax.texts
            if any(
                token in text.get_text().lower()
                for token in ("event-sampled", "missing", "trunk placeholder")
            )
        ]
        assert axis_semantics == []

        figure_semantics = [text.get_text() for text in fig.texts]
        figure_semantics.extend(
            text.get_text() for legend in fig.legends for text in legend.get_texts()
        )
        semantic_key = " ".join(figure_semantics).lower()
        assert "event-sampled" in semantic_key
        assert "missing" in semantic_key
        # The trunk key must say the rows are FIT as 0.0, not that they are
        # excluded: production writes 0.0 there so the fitter keeps the event.
        assert "trunk placeholder" in semantic_key and "0.0" in semantic_key
        plt.close(fig)

    def test_context_flags_draw_undefined_distinctly_from_false(self):
        """NaN context is undefined, not observed-False.

        ``build_event_table`` assigns NaN to the outcome/transition flags at
        session boundaries. Drawing that as blank makes the first stop of every
        session read as "all four flags observed False".
        """
        from continuum_swr_replay.visualization.session_tables import (
            _MISSING_CELL_COLOR,
        )

        inputs = _public_plot_inputs()
        inputs["event_table"] = inputs["event_table"].assign(
            is_rewarded_outcome=[np.nan],  # both outcome complements undefined
            did_last_completed_trip_switch_patch=[0.0],
            is_swr=[0.0],
            is_mua=[0.0],
        )

        fig, _ = plot_session_tables(
            **inputs, time_window=(0.0, 4.0), panels=("context_flags",)
        )
        ax = fig.axes[0]

        missing_rgba = to_rgba(_MISSING_CELL_COLOR, alpha=0.85)
        missing = [
            p for p in ax.patches if np.allclose(p.get_facecolor(), missing_rgba)
        ]
        assert len(missing) == 2, (
            "an unknown outcome makes both exact complements undefined"
        )
        assert all(patch.get_hatch() == "///" for patch in missing)
        # Three cells total: two undefined outcome complements and one derived
        # stayed-in-patch True. The remaining observed-False flags stay blank.
        assert len(ax.patches) == 3
        plt.close(fig)

    def test_shared_colorbar_never_overlaps_a_sibling_colorbar(self):
        """A shared bar spanning non-adjacent axes would be drawn over.

        ``q_arm``/``p_reward_arm`` share a colorbar group; ``arm_uncertainty``
        has its own 0-1/12 scale. Ordered between them, its bar landed inside
        the shared 0-1 bar, so belief variance read against a probability axis.
        """
        inputs = _public_plot_inputs()
        values = [0.05] * 9
        inputs["event_arm_table"] = _arm_table(
            {
                "reward_belief_mean_candidate_arm_preoutcome": values,
                "task_reward_probability_candidate_arm": values,
                "reward_belief_variance_candidate_arm_preoutcome": values,
            }
        )
        # Deliberately non-adjacent: the grouped pair straddles a third panel.
        panels = (
            "reward_belief_mean_candidate_arm_preoutcome",
            "reward_belief_variance_candidate_arm_preoutcome",
            "task_reward_probability_candidate_arm",
        )

        fig, _ = plot_session_tables(**inputs, time_window=(0.0, 4.0), panels=panels)
        fig.canvas.draw()

        cbar_axes = fig.axes[len(panels) :]
        assert len(cbar_axes) >= 2
        boxes = [ax.get_position() for ax in cbar_axes]
        for i, first in enumerate(boxes):
            for second in boxes[i + 1 :]:
                overlap = (
                    first.x0 < second.x1
                    and second.x0 < first.x1
                    and first.y0 < second.y1
                    and second.y0 < first.y1
                )
                assert not overlap, "colorbars must not be drawn on top of each other"
        plt.close(fig)

    def test_continuous_cell_width_floor_scales_with_the_window(self):
        """A fixed absolute floor renders sub-pixel on a full-session window.

        The binary panels already use a window-relative floor, so a fixed one
        here made continuous rows blank while binary rows beside them stayed
        visible -- under a legend reading "blank = no event".
        """
        inputs = _public_plot_inputs()
        inputs["results"] = xr.Dataset(coords={"time": np.array([0.0, 4000.0])})
        inputs["event_table"] = pd.DataFrame(
            {
                "event_id": [0],
                "event_start_s": [100.0],
                "event_end_s": [100.15],  # 0.15 s event, as in real data
                "event_midpoint_s": [100.075],
                "next_arm": [1.0],
            }
        )
        inputs["event_arm_table"] = _arm_table(
            {"reward_belief_mean_candidate_arm_preoutcome": [0.5] * 9}
        ).assign(event_start_s=100.0, event_end_s=100.15)

        window = (0.0, 4000.0)
        fig, _ = plot_session_tables(
            **inputs,
            time_window=window,
            panels=("reward_belief_mean_candidate_arm_preoutcome",),
        )
        expected = (window[1] - window[0]) * 0.002

        widths = {round(p.get_width(), 6) for p in fig.axes[0].patches}
        assert widths == {round(expected, 6)}, (
            "cell width must be floored relative to the window, not at 1e-3"
        )
        plt.close(fig)

    def test_semantic_key_and_title_stay_inside_the_canvas(self):
        """The key is laid out, not anchored over the top panel.

        An explicit ``bbox_to_anchor`` above the axes is not managed by
        constrained_layout, so the key covered the first panel and pushed the
        suptitle off-figure -- clipping it in any inline render.
        """
        inputs = _public_plot_inputs()
        fig, _ = plot_session_tables(
            **inputs,
            time_window=(0.0, 4.0),
            panels=("reward_belief_mean_candidate_arm_preoutcome",),
            title="Session audit",
        )
        fig.canvas.draw()

        (legend,) = fig.legends
        legend_box = legend.get_window_extent().transformed(fig.transFigure.inverted())
        top_axes_box = fig.axes[0].get_position()
        assert legend_box.y0 >= top_axes_box.y1, "key must not overlap the top panel"
        assert legend_box.y1 <= 1.0, "key must stay inside the canvas"

        title_box = fig._suptitle.get_window_extent().transformed(
            fig.transFigure.inverted()
        )
        assert title_box.y1 <= 1.0, "suptitle must stay inside the canvas"
        plt.close(fig)
