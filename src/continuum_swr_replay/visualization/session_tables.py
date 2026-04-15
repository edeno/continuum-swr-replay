"""Session-tables inspection figure.

``plot_session_tables`` stacks a time-aligned multi-panel view of one
session so the user can trace the transformation from raw decoder
output through the four canonical tables. It is the canonical-tables
analog of :func:`plot_detector`: same layout conventions, a superset
of the core panels, plus overlays and optional rows that expose each
step of the derivation chain.

See ``plans/2026_04_15_session_tables_viz_plan.md`` for the full
design. This chunk lands the scaffold and core panels only — the
canonical-table-aware panels (arm markers, mismatch pointers, content
heatmaps, scalar rows) arrive in later chunks.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from continuum_swr_replay.data_loaders.constants import WELL_PATCH_MAPPING
from continuum_swr_replay.parameters import COLORS

from .palettes import patch_grouped_arm_palette
from .plots import (
    plot_acausal_probabilities,
    plot_conditional_non_local_posterior,
    plot_multiunit_firing,
    plot_ripple_consensus_trace_zscore,
    plot_speed,
    plot_spike_times,
)

# Core panels (always available). Later chunks register additional
# panel names in this registry.
DEFAULT_PANELS: tuple[str, ...] = (
    "swr_lfp",
    "mua",
    "spike_times",
    "state_probs",
    "conditional_posterior",
    "speed",
    "trial_rail",
)


@dataclass
class _PanelContext:
    """Context object passed to every panel helper."""

    data: Mapping[str, Any]
    detector: Any
    results: xr.Dataset
    event_table: pd.DataFrame
    event_arm_table: pd.DataFrame
    event_sequence_table: pd.DataFrame
    stop_table: pd.DataFrame
    trials: pd.DataFrame
    time_slice: slice
    arm_palette: dict[int, tuple[float, float, float, float]]
    env: Any
    data_type: str


# ── Core panels (thin adapters over existing plot_detector helpers) ──


def _plot_swr_lfp(ax: Axes, ctx: _PanelContext) -> None:
    plot_ripple_consensus_trace_zscore(ctx.data, ctx.time_slice, ax)


def _plot_mua(ax: Axes, ctx: _PanelContext) -> None:
    plot_multiunit_firing(
        ctx.data, ctx.time_slice, ax, plot_multiunit_zscore=True
    )


def _plot_spike_times(ax: Axes, ctx: _PanelContext) -> None:
    plot_spike_times(
        ctx.data, ctx.time_slice, ctx.detector, ctx.env, ctx.results, ax, ctx.data_type
    )


def _plot_state_probs(ax: Axes, ctx: _PanelContext) -> None:
    plot_acausal_probabilities(
        ctx.results, ctx.time_slice, ax, plot_thresholds=True, plot_local=True
    )


def _plot_conditional_posterior(ax: Axes, ctx: _PanelContext) -> None:
    plot_conditional_non_local_posterior(
        ctx.data, ctx.time_slice, ctx.results, ctx.env, ctx.detector, ax
    )


def _plot_speed(ax: Axes, ctx: _PanelContext) -> None:
    plot_speed(ctx.data, ctx.time_slice, ax)


# ── New: trial rail ──


def _plot_trial_rail(ax: Axes, ctx: _PanelContext) -> None:
    """Ground-truth animal location over time, from ``trials``.

    Step-post plot of ``to_well`` over ``end_time``: at each trial's
    completion the rail jumps to the new well and stays there until
    the next trial ends. Reward trials get a star; patch-change trial
    segments get a thicker line.

    This panel is the **independent reference** against which the
    canonical tables' ``next_arm`` / ``next_well`` claims are
    checked. It's deliberately built directly from ``trials``, not
    from any canonical-table column, so an alignment bug in the
    pipeline cannot hide behind the plot.
    """
    trials = ctx.trials
    lo, hi = ctx.time_slice.start, ctx.time_slice.stop
    windowed = trials[(trials["end_time"] >= lo) & (trials["start_time"] <= hi)]
    if len(windowed) == 0:
        ax.text(
            0.5,
            0.5,
            "no trials in window",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="grey",
        )
        ax.set_ylabel("Trial\nto_well")
        ax.set_ylim(-0.5, 5.5)
        return

    end_times = windowed["end_time"].to_numpy()
    to_wells = windowed["to_well"].to_numpy(dtype=float)
    is_reward = windowed["is_reward"].to_numpy(dtype=bool)
    is_patch_change = windowed["is_patch_change"].to_numpy(dtype=bool)

    # Step-post: (t0, y0) → (t1, y0) → (t1, y1) → ...
    # Use ``drawstyle="steps-post"`` on ``plot`` for compact form.
    ax.plot(
        end_times,
        to_wells,
        drawstyle="steps-post",
        color="#333333",
        linewidth=1.2,
        clip_on=False,
    )

    # Reward stars — drawn on top of the step plot at each rewarded
    # trial's end time / to_well.
    if is_reward.any():
        ax.scatter(
            end_times[is_reward],
            to_wells[is_reward],
            marker="*",
            s=70,
            facecolor="gold",
            edgecolor="black",
            linewidths=0.5,
            zorder=5,
            clip_on=False,
        )

    # Patch-change markers — a colored tick at the x-axis, since the
    # change is a trip-level fact not a positional one.
    if is_patch_change.any():
        for t in end_times[is_patch_change]:
            ax.axvline(t, color=COLORS["SWR"], linewidth=0.5, alpha=0.4)

    # Per-well horizontal guidelines so step levels are easy to read.
    for well in sorted(WELL_PATCH_MAPPING):
        ax.axhline(well, color="lightgrey", linewidth=0.3, zorder=0)

    ax.set_ylabel("Trial\nto_well")
    ax.set_ylim(-0.5, 5.5)
    ax.set_yticks(sorted(WELL_PATCH_MAPPING))


# ── Panel registry ──


_PANEL_REGISTRY: dict[str, Callable[[Axes, _PanelContext], None]] = {
    "swr_lfp": _plot_swr_lfp,
    "mua": _plot_mua,
    "spike_times": _plot_spike_times,
    "state_probs": _plot_state_probs,
    "conditional_posterior": _plot_conditional_posterior,
    "speed": _plot_speed,
    "trial_rail": _plot_trial_rail,
}

# Relative height weight per panel — inherits the ``plot_detector``
# template (posterior and raster get weight 3).
_PANEL_HEIGHTS: dict[str, float] = {
    "swr_lfp": 1.0,
    "mua": 1.0,
    "spike_times": 3.0,
    "state_probs": 1.0,
    "conditional_posterior": 3.0,
    "speed": 1.0,
    "trial_rail": 1.0,
}


def register_panel(
    name: str,
    plotter: Callable[[Axes, _PanelContext], None],
    *,
    height: float = 1.0,
) -> None:
    """Register a new panel plotter under ``name``.

    Later viz chunks (arm markers, mismatch pointers, content
    heatmaps, scalar rows) add entries here so ``plot_session_tables``
    can dispatch them by name in the ``panels`` kwarg.
    """
    if name in _PANEL_REGISTRY:
        raise ValueError(f"panel {name!r} already registered")
    _PANEL_REGISTRY[name] = plotter
    _PANEL_HEIGHTS[name] = height


# ── Public entry point ──


def plot_session_tables(
    data: Mapping[str, Any],
    detector: Any,
    results: xr.Dataset,
    event_table: pd.DataFrame,
    event_arm_table: pd.DataFrame,
    event_sequence_table: pd.DataFrame,
    stop_table: pd.DataFrame,
    *,
    trials: pd.DataFrame | None = None,
    time_window: tuple[float, float] | None = None,
    panels: Sequence[str] = DEFAULT_PANELS,
    figsize: tuple[float, float] = (12.0, 9.0),
    data_type: str = "clusterless",
    save_path: Path | None = None,
) -> tuple[Figure, pd.DataFrame]:
    """Multi-panel time-aligned view of one session's canonical tables.

    Parameters
    ----------
    data : mapping
        ``load_data``-style dict. Must contain ``position_info``,
        ``hse_times``, ``ripple_times``, ``multiunit_rate_zscore``,
        ``zscored_ripple_consensus_trace`` — the same keys
        :func:`plot_detector` consumes. ``trials`` and ``track_graph``
        may be here or passed separately.
    detector, results
        Fitted detector and decoding results — same roles as in
        :func:`plot_detector`.
    event_table, event_arm_table, event_sequence_table, stop_table
        Canonical tables produced by
        :mod:`continuum_swr_replay.analysis.replay.tables`.
    trials : pd.DataFrame, optional
        Trial DataFrame; falls back to ``data['trials']`` when absent.
    time_window : (float, float), optional
        ``(start_time, end_time)`` in seconds. Default: full session.
    panels : sequence of str
        Panel names, top-to-bottom order. See :data:`DEFAULT_PANELS`.
        Unknown names raise :class:`KeyError`.
    figsize, data_type, save_path : passthrough.

    Returns
    -------
    fig : matplotlib.figure.Figure
    mismatch_table : pd.DataFrame
        Currently empty — the future-covariate mismatch detector is
        landed in a later chunk. Returned as part of the stable
        contract so downstream callers can already accept the
        two-tuple shape.
    """
    if trials is None:
        trials = data.get("trials")
        if trials is None:
            raise KeyError(
                "trials not provided and not present in ``data``; "
                "pass ``trials=`` explicitly."
            )

    unknown = [p for p in panels if p not in _PANEL_REGISTRY]
    if unknown:
        raise KeyError(
            f"unknown panel name(s): {unknown}. "
            f"known: {sorted(_PANEL_REGISTRY)}"
        )

    t_start = float(results.time.values[0])
    t_end = float(results.time.values[-1])
    if time_window is not None:
        t_start = max(t_start, float(time_window[0]))
        t_end = min(t_end, float(time_window[1]))
    time_slice = slice(t_start, t_end)

    env = detector.environments[0]
    ctx = _PanelContext(
        data=data,
        detector=detector,
        results=results,
        event_table=event_table,
        event_arm_table=event_arm_table,
        event_sequence_table=event_sequence_table,
        stop_table=stop_table,
        trials=trials,
        time_slice=time_slice,
        arm_palette=patch_grouped_arm_palette(),
        env=env,
        data_type=data_type,
    )

    heights = [_PANEL_HEIGHTS.get(p, 1.0) for p in panels]
    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=figsize,
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": heights},
        dpi=100,
    )
    # ``plt.subplots`` returns a single Axes when n=1; normalize to
    # an array so ``zip(panels, axes)`` works uniformly.
    axes = np.atleast_1d(axes)

    for panel_name, ax in zip(panels, axes, strict=True):
        _PANEL_REGISTRY[panel_name](ax, ctx)

    axes[-1].set_xlabel("Time [s]")
    axes[-1].set_xlim(t_start, t_end)
    sns.despine(offset=5)

    # Stable return type: (fig, empty mismatch_table) — populated by
    # the chunk that lands the mismatch-pointer panel.
    mismatch_table = pd.DataFrame(
        columns=[
            "event_id",
            "event_midpoint_time",
            "canonical_next_arm",
            "actual_next_to_well",
            "actual_next_arm",
            "match",
        ]
    )

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, mismatch_table
