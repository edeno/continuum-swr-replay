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

from continuum_swr_replay.analysis.replay.spatial_mapping import well_to_arm_set
from continuum_swr_replay.data_loaders.constants import (
    TRACK_SEGMENT_TO_PATCH,
    WELL_PATCH_MAPPING,
)
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

_ALL_ARMS: tuple[int, ...] = tuple(sorted(TRACK_SEGMENT_TO_PATCH.keys()))

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
    track_graph: Any
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


# ── Future-covariate debug: arm markers + mismatch pointers ──


def _well_to_single_arm(well: float, track_graph: Any) -> float:
    """Scalar arm for a well node, NaN if well is NaN or not in graph.

    Mirrors the ``_single_arm_from_well`` helper in
    ``analysis.replay.tables``; we duplicate the tiny helper here so
    the visualization package does not take a hard dependency on the
    tables module. On the real bandit graph each well is degree-1 so
    ``well_to_arm_set`` returns a 1-element set.
    """
    if well is None or (isinstance(well, float) and np.isnan(well)):
        return np.nan
    arms = well_to_arm_set(int(well), track_graph)
    if not arms:
        return np.nan
    return float(min(arms))


def _compute_mismatch_table(
    event_table: pd.DataFrame,
    trials: pd.DataFrame,
    track_graph: Any,
) -> pd.DataFrame:
    """Compare canonical ``next_arm`` against trials' actual next trip.

    For every event, the independent ground truth is: the first trial
    whose ``start_time`` is strictly greater than ``event_midpoint_time``.
    Its ``to_well`` is mapped to an arm via
    :func:`well_to_arm_set`. The canonical value comes from
    ``event_table['next_arm']`` — same one downstream GLMs consume.

    ``match`` is ``True`` when the canonical and actual arms are equal
    OR when both are ``NaN`` (legitimate session end: the last event
    has no next trip). Any other combination is a mismatch and will be
    drawn in red by :func:`_plot_mismatch_pointers`.
    """
    if len(event_table) == 0:
        return pd.DataFrame(
            columns=[
                "event_id",
                "event_midpoint_time",
                "canonical_next_arm",
                "actual_next_to_well",
                "actual_next_arm",
                "match",
            ]
        )

    event_times = event_table["event_midpoint_time"].to_numpy()
    start_times = (
        trials["start_time"].to_numpy() if len(trials) > 0 else np.array([])
    )
    to_wells = trials["to_well"].to_numpy() if len(trials) > 0 else np.array([])

    actual_to_well = np.full(len(event_times), np.nan)
    if start_times.size > 0:
        next_idx = np.searchsorted(start_times, event_times, side="right")
        valid = next_idx < start_times.size
        actual_to_well[valid] = to_wells[next_idx[valid]]

    actual_arm = np.array(
        [_well_to_single_arm(w, track_graph) for w in actual_to_well]
    )
    canonical = event_table["next_arm"].to_numpy(dtype=float)

    both_nan = np.isnan(canonical) & np.isnan(actual_arm)
    equal = (canonical == actual_arm) & ~np.isnan(canonical) & ~np.isnan(actual_arm)
    match = both_nan | equal

    return pd.DataFrame(
        {
            "event_id": event_table["event_id"].to_numpy(),
            "event_midpoint_time": event_times,
            "canonical_next_arm": canonical,
            "actual_next_to_well": actual_to_well,
            "actual_next_arm": actual_arm,
            "match": match,
        }
    )


def _plot_arm_markers(ax: Axes, ctx: _PanelContext) -> None:
    """Canonical arm references per event.

    At each event's ``event_midpoint_time``, four markers in the
    arm-palette color:

    - ``current_arm`` — filled circle
    - ``prev_arm`` — upward triangle ▲
    - ``next_arm`` — downward triangle ▼ (**post-hoc**)
    - ``last_rewarded_arm`` — star ★

    Markers are skipped where the corresponding column is NaN. Y-axis
    is the 9-arm index space grouped visually by the patch palette.
    """
    lo, hi = ctx.time_slice.start, ctx.time_slice.stop
    et = ctx.event_table
    in_window = (et["event_midpoint_time"] >= lo) & (et["event_midpoint_time"] <= hi)
    et = et.loc[in_window]

    for arm in _ALL_ARMS:
        ax.axhline(arm, color="lightgrey", linewidth=0.3, zorder=0)

    if len(et) == 0:
        ax.text(
            0.5, 0.5, "no events in window",
            transform=ax.transAxes, ha="center", va="center", color="grey",
        )
    else:
        times = et["event_midpoint_time"].to_numpy()
        palette = ctx.arm_palette
        marker_spec = (
            ("current_arm", "o", 36),
            ("prev_arm", "^", 36),
            ("next_arm", "v", 36),
            ("last_rewarded_arm", "*", 60),
        )
        for col, marker, size in marker_spec:
            arms = et[col].to_numpy(dtype=float)
            mask = ~np.isnan(arms)
            if not mask.any():
                continue
            colors = [palette.get(int(a), (0.5, 0.5, 0.5, 1.0)) for a in arms[mask]]
            ax.scatter(
                times[mask],
                arms[mask],
                marker=marker,
                s=size,
                c=colors,
                edgecolor="black",
                linewidths=0.3,
                zorder=3,
                clip_on=False,
            )

    ax.set_ylabel("Arm\ncurr/prev/\nnext/rew")
    ax.set_ylim(-0.5, 8.5)
    ax.set_yticks(list(_ALL_ARMS))


def _plot_mismatch_pointers(ax: Axes, ctx: _PanelContext) -> None:
    """Canonical ``next_arm`` versus trials' actual next trip.

    For every event in the window: a ▼ at the canonical
    ``event_table['next_arm']`` and a × at the arm derived independently
    from ``trials`` (the ``to_well`` of the first trial whose
    ``start_time`` is strictly greater than ``event_midpoint_time``).
    A vertical connector between them — **red when the two disagree**,
    thin grey when they match.

    ``(NaN, NaN)`` — the legitimate session-end case where the last
    event has no next trip — is treated as a match.

    The underlying mismatch table is computed once per ``ctx`` and
    stashed on the context so the caller can retrieve it from the
    public entry point.
    """
    lo, hi = ctx.time_slice.start, ctx.time_slice.stop
    mismatch = _compute_mismatch_table(
        ctx.event_table, ctx.trials, ctx.track_graph
    )
    in_window = (
        (mismatch["event_midpoint_time"] >= lo)
        & (mismatch["event_midpoint_time"] <= hi)
    )
    view = mismatch.loc[in_window]

    for arm in _ALL_ARMS:
        ax.axhline(arm, color="lightgrey", linewidth=0.3, zorder=0)

    if len(view) == 0:
        ax.text(
            0.5, 0.5, "no events in window",
            transform=ax.transAxes, ha="center", va="center", color="grey",
        )
    else:
        times = view["event_midpoint_time"].to_numpy()
        canonical = view["canonical_next_arm"].to_numpy(dtype=float)
        actual = view["actual_next_arm"].to_numpy(dtype=float)
        is_match = view["match"].to_numpy(dtype=bool)

        # Small vertical offset so the canonical ▼ and actual × stay
        # visually distinguishable even when they're at the same arm.
        # Without it, "everything matches" (the expected healthy case)
        # reads as "nothing is plotted".
        dy = 0.18

        # Connector lines first so markers sit on top. We always draw
        # a connector — thin grey for matches (short segment of length
        # 2*dy), bold red for mismatches (spans the arm gap + offsets).
        for t, c_arm, a_arm, match in zip(times, canonical, actual, is_match, strict=True):
            if np.isnan(c_arm) or np.isnan(a_arm):
                continue
            color = "lightgrey" if match else "red"
            lw = 0.5 if match else 1.6
            ax.plot(
                [t, t], [c_arm - dy, a_arm + dy],
                color=color, linewidth=lw, zorder=2, clip_on=False,
            )

        palette = ctx.arm_palette
        c_mask = ~np.isnan(canonical)
        if c_mask.any():
            ax.scatter(
                times[c_mask],
                canonical[c_mask] - dy,
                marker="v",
                s=36,
                c=[palette.get(int(a), (0.5, 0.5, 0.5, 1.0)) for a in canonical[c_mask]],
                edgecolor="black",
                linewidths=0.3,
                zorder=3,
                clip_on=False,
                label="canonical next_arm",
            )
        a_mask = ~np.isnan(actual)
        if a_mask.any():
            ax.scatter(
                times[a_mask],
                actual[a_mask] + dy,
                marker="x",
                s=40,
                c=[palette.get(int(a), (0.5, 0.5, 0.5, 1.0)) for a in actual[a_mask]],
                linewidths=1.0,
                zorder=4,
                clip_on=False,
                label="actual next_arm (from trials)",
            )

    ax.set_ylabel("next_arm\ncanonical\nvs actual")
    ax.set_ylim(-0.5, 8.5)
    ax.set_yticks(list(_ALL_ARMS))

    # Stash the full (unfiltered) mismatch table on the context so the
    # public entry point can return it without recomputing.
    ctx.__dict__.setdefault("_mismatch_table", mismatch)


# ── Panel registry ──


_PANEL_REGISTRY: dict[str, Callable[[Axes, _PanelContext], None]] = {
    "swr_lfp": _plot_swr_lfp,
    "mua": _plot_mua,
    "spike_times": _plot_spike_times,
    "state_probs": _plot_state_probs,
    "conditional_posterior": _plot_conditional_posterior,
    "speed": _plot_speed,
    "trial_rail": _plot_trial_rail,
    "arm_markers": _plot_arm_markers,
    "mismatch_pointers": _plot_mismatch_pointers,
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
    "arm_markers": 1.4,
    "mismatch_pointers": 1.4,
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
    track_graph: Any = None,
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
    if track_graph is None:
        track_graph = data.get("track_graph")
        if track_graph is None:
            raise KeyError(
                "track_graph not provided and not present in ``data``; "
                "pass ``track_graph=`` explicitly."
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
        track_graph=track_graph,
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
    # Suppress the ``+1.625935e9`` offset notation — session times are
    # seconds-since-epoch; absolute values read fine and the offset
    # notation clutters the axis without aiding interpretation.
    axes[-1].ticklabel_format(useOffset=False, style="plain", axis="x")
    sns.despine(offset=5)

    # If the mismatch-pointers panel ran it stashed the full table on
    # ctx. Otherwise compute once here so callers get the same shape
    # regardless of panel selection (useful for headless CI / audit).
    mismatch_table = getattr(
        ctx, "_mismatch_table", None
    )
    if mismatch_table is None:
        mismatch_table = _compute_mismatch_table(event_table, trials, track_graph)

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, mismatch_table
