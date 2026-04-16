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

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

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
    # Arm-label overlay — annotate the y-axis with arm IDs at the
    # mean linear position of each arm's place bins so the reader
    # can tell "this posterior blob is on arm 3" without mental
    # arithmetic. Mirrors the ``plot_graph_as_1D`` annotations in
    # ``plot_replay_content_time_slice.py``.
    bin_df = ctx.env.place_bin_centers_nodes_df_
    edge_ids = np.asarray(bin_df["edge_id"])
    lin_pos = np.asarray(bin_df["linear_position"])
    palette = ctx.arm_palette
    t_left = ctx.time_slice.start
    for arm in _ALL_ARMS:
        mask = edge_ids == arm
        if not mask.any():
            continue
        y_center = float(lin_pos[mask].mean())
        color = palette.get(arm, (0.3, 0.3, 0.3, 1.0))
        ax.text(
            t_left, y_center, f"a{arm}",
            fontsize=7, va="center", ha="right",
            color=color, fontweight="bold", clip_on=False,
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

    # Patch-change markers — purple ticks (deliberately NOT the SWR
    # orange, which would collide with SWR shading in panels 1–2).
    if is_patch_change.any():
        for t in end_times[is_patch_change]:
            ax.axvline(t, color="#9467bd", linewidth=0.7, alpha=0.5)

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

    ax.set_ylabel("Arm\nIdentity")
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
        # Half-NaN events (one side has a next_arm, the other doesn't
        # — legitimate at session boundaries but still a mismatch)
        # get a short red stub around the side that does have a value
        # so the red-highlights-disagreement cue is preserved.
        for t, c_arm, a_arm, match in zip(times, canonical, actual, is_match, strict=True):
            c_nan, a_nan = np.isnan(c_arm), np.isnan(a_arm)
            if c_nan and a_nan:
                continue
            if c_nan ^ a_nan:
                present = c_arm if not c_nan else a_arm
                ax.plot(
                    [t, t], [present - dy, present + dy],
                    color="red", linewidth=1.6, zorder=2, clip_on=False,
                )
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

    ax.set_ylabel("Next-Arm\nCheck")
    ax.set_ylim(-0.5, 8.5)
    ax.set_yticks(list(_ALL_ARMS))

    # Legend — explain the two markers and the connector color.
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([], [], marker="v", color="grey", markersize=6,
               markeredgecolor="black", markeredgewidth=0.3,
               linestyle="none", label="canonical next_arm"),
        Line2D([], [], marker="x", color="grey", markersize=6,
               markeredgewidth=1.0,
               linestyle="none", label="actual (from trials)"),
        Line2D([], [], color="lightgrey", linewidth=1.5, label="match"),
        Line2D([], [], color="red", linewidth=1.5, label="MISMATCH"),
    ]
    ax.legend(
        handles=legend_handles,
        bbox_to_anchor=(1.02, 1), loc="upper left",
        fontsize=7, framealpha=0.7,
    )

    # Stash the full (unfiltered) mismatch table on the context so the
    # public entry point can return it without recomputing.
    ctx.__dict__.setdefault("_mismatch_table", mismatch)


# ── Content panels: posterior → per-arm content heatmaps ──


def _plot_arm_posterior_heatmap(ax: Axes, ctx: _PanelContext) -> None:
    """Per-timebin arm-marginal posterior as a 9×T heatmap.

    Reads ``event_sequence_table.arm_posterior_{0..8}`` and draws a
    ``pcolormesh`` per event (so gaps between events stay white
    instead of smeared). Overlays the per-timebin MAP arm
    (``decoded_arm``) as black dots so the reader can tell where the
    MAP landed on each timebin.

    This is the direct arm-marginal counterpart of
    :func:`plot_conditional_non_local_posterior`: the raw posterior
    collapsed to 9 arms via
    :func:`summaries.posterior_to_arm_marginal`. A bug in that
    collapse or in ``arm_posterior_mass`` downstream would surface as
    a heatmap that disagrees with the position-bin panel above it.
    """
    lo, hi = ctx.time_slice.start, ctx.time_slice.stop
    est = ctx.event_sequence_table
    view = est[(est["time"] >= lo) & (est["time"] <= hi)]

    for arm in _ALL_ARMS:
        ax.axhline(arm, color="lightgrey", linewidth=0.3, zorder=0)

    if len(view) == 0:
        ax.text(
            0.5, 0.5, "no in-event timebins in window",
            transform=ax.transAxes, ha="center", va="center", color="grey",
        )
        ax.set_ylabel("arm\nposterior")
        ax.set_ylim(-0.5, 8.5)
        ax.set_yticks(list(_ALL_ARMS))
        return

    arm_cols = [f"arm_posterior_{a}" for a in _ALL_ARMS]
    edges_a = np.arange(len(_ALL_ARMS) + 1) - 0.5

    # Derive a fallback ``dt`` from the global time diffs so single-
    # timebin events still get a visible cell (1/sampling_frequency on
    # the bandit data, ~2 ms). Without this, single-timebin events
    # drop out of the heatmap but the MAP-dot overlay still draws
    # them, leaving a confusing lone dot with no background.
    all_times = view["time"].to_numpy()
    diffs = np.diff(all_times)
    dt_fallback = float(np.median(diffs[diffs > 0])) if (diffs > 0).any() else 0.002

    # Draw one pcolormesh per event — matplotlib doesn't interpolate
    # across discontinuous x, so per-event calls keep gaps white.
    for _event_id, rows in view.groupby("event_id", sort=False):
        times = rows["time"].to_numpy()
        if times.size < 1:
            continue
        if times.size == 1:
            dt = dt_fallback
        else:
            dt = float(np.median(np.diff(times)))
        edges_t = np.concatenate([times - dt / 2, [times[-1] + dt / 2]])
        posterior = rows[arm_cols].to_numpy().T  # (n_arms, T)
        ax.pcolormesh(
            edges_t, edges_a, posterior, cmap="bone_r", vmin=0.0, vmax=1.0,
            shading="flat", zorder=1,
        )

    # MAP-arm overlay (decoded_arm_map per timebin).
    ax.scatter(
        view["time"].to_numpy(),
        view["decoded_arm_map"].to_numpy(),
        s=4, c="black", marker=".", zorder=4, clip_on=False,
    )

    ax.set_ylabel("arm\nposterior")
    ax.set_ylim(-0.5, 8.5)
    ax.set_yticks(list(_ALL_ARMS))


def _draw_event_arm_heatmap(
    ax: Axes,
    ctx: _PanelContext,
    *,
    value_col: str,
    cmap_name: str,
    vmin: float | None,
    vmax: float | None,
    label: str,
) -> None:
    """Shared: per-(event, arm) value drawn as colored rectangles.

    ``event_arm_table`` has 9 rows per event; we draw each as a
    rectangle spanning ``[event_start, event_end]`` at arm height ±
    0.5. Color is ``cmap_name`` normalized by ``(vmin, vmax)`` — when
    either is ``None`` it is inferred from the data in the active
    time window so the palette uses the full range available.
    """
    lo, hi = ctx.time_slice.start, ctx.time_slice.stop
    # ``event_arm_table`` already carries the broadcast
    # ``event_start`` / ``event_end`` per row (from Phase B4 canonical
    # schema), so no merge with ``event_table`` is needed.
    eat = ctx.event_arm_table
    view = eat[(eat["event_start"] <= hi) & (eat["event_end"] >= lo)]

    for arm in _ALL_ARMS:
        ax.axhline(arm, color="lightgrey", linewidth=0.3, zorder=0)

    if len(view) == 0:
        ax.text(
            0.5, 0.5, "no events in window",
            transform=ax.transAxes, ha="center", va="center", color="grey",
        )
        ax.set_ylabel(label)
        ax.set_ylim(-0.5, 8.5)
        ax.set_yticks(list(_ALL_ARMS))
        return

    values = view[value_col].to_numpy(dtype=float)
    finite = np.isfinite(values)
    if vmin is None:
        vmin = float(values[finite].min()) if finite.any() else 0.0
    if vmax is None:
        vmax = float(values[finite].max()) if finite.any() else 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)

    # Iterate rows via numpy columns to avoid .iterrows() overhead.
    arms = view["arm"].to_numpy()
    starts = view["event_start"].to_numpy()
    ends = view["event_end"].to_numpy()
    for start, end, arm, val in zip(starts, ends, arms, values, strict=True):
        if not np.isfinite(val):
            continue
        rect = Rectangle(
            (start, arm - 0.5),
            width=max(end - start, 1e-3),  # avoid zero-width on tiny events
            height=1.0,
            facecolor=cmap(norm(val)),
            edgecolor="none",
            zorder=1,
        )
        ax.add_patch(rect)

    # Colorbar inside the axis as a narrow tall strip on the right.
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    ax.figure.colorbar(sm, ax=ax, shrink=0.7, pad=0.01, aspect=10)

    ax.set_ylabel(label)
    ax.set_ylim(-0.5, 8.5)
    ax.set_yticks(list(_ALL_ARMS))


def _plot_arm_count_hard(ax: Axes, ctx: _PanelContext) -> None:
    _draw_event_arm_heatmap(
        ax, ctx, value_col="arm_count_hard",
        cmap_name="viridis", vmin=0, vmax=None,
        label="arm_count\nhard",
    )


def _plot_arm_count_soft(ax: Axes, ctx: _PanelContext) -> None:
    _draw_event_arm_heatmap(
        ax, ctx, value_col="arm_count_soft",
        cmap_name="viridis", vmin=0, vmax=None,
        label="arm_count\nsoft",
    )


def _plot_arm_posterior_mass(ax: Axes, ctx: _PanelContext) -> None:
    _draw_event_arm_heatmap(
        ax, ctx, value_col="arm_posterior_mass",
        cmap_name="viridis", vmin=0, vmax=None,
        label="arm_post.\nmass",
    )


def _plot_arm_fraction(ax: Axes, ctx: _PanelContext) -> None:
    _draw_event_arm_heatmap(
        ax, ctx, value_col="arm_fraction",
        cmap_name="viridis", vmin=0.0, vmax=1.0,
        label="arm\nfraction",
    )


# ── Per-arm predictor panels (trials → per-arm event-arm row) ──


def _plot_visit_recency(ax: Axes, ctx: _PanelContext) -> None:
    _draw_event_arm_heatmap(
        ax, ctx, value_col="visit_recency",
        cmap_name="viridis_r", vmin=1, vmax=None,
        label="visit\nrecency",
    )


def _plot_cumulative_reward(ax: Axes, ctx: _PanelContext) -> None:
    _draw_event_arm_heatmap(
        ax, ctx, value_col="cumulative_reward",
        cmap_name="viridis", vmin=0, vmax=None,
        label="cumul.\nreward",
    )


def _plot_recent_reward_rate(ax: Axes, ctx: _PanelContext) -> None:
    _draw_event_arm_heatmap(
        ax, ctx, value_col="recent_reward_rate",
        cmap_name="RdYlGn", vmin=0.0, vmax=1.0,
        label="recent\nreward\nrate",
    )


def _plot_nominal_reward_prob(ax: Axes, ctx: _PanelContext) -> None:
    """Programmed reward probability per arm.

    Currently all NaN — the contingency-block loader that populates
    this column was intentionally deferred in Phase B. The panel
    still draws (as an empty heatmap) so the viz stays layout-stable
    once the column is wired.
    """
    _draw_event_arm_heatmap(
        ax, ctx, value_col="nominal_reward_prob",
        cmap_name="RdYlGn", vmin=0.0, vmax=1.0,
        label="nominal\np(R)",
    )


def _plot_dist_to_arm(ax: Axes, ctx: _PanelContext) -> None:
    _draw_event_arm_heatmap(
        ax, ctx, value_col="dist_to_arm",
        cmap_name="viridis_r", vmin=0, vmax=None,
        label="dist to\narm",
    )


def _plot_arm_dist_rank(ax: Axes, ctx: _PanelContext) -> None:
    _draw_event_arm_heatmap(
        ax, ctx, value_col="arm_dist_rank",
        cmap_name="viridis_r", vmin=1, vmax=len(_ALL_ARMS),
        label="arm\ndist rank",
    )


# ── Stacked-area arm mass (composition over time) ──


def _plot_arm_mass_stackplot(ax: Axes, ctx: _PanelContext) -> None:
    """Per-timebin arm-posterior composition as a stacked area.

    Each arm's band height is proportional to its posterior mass at
    that timebin; the total always sums to 1. Colored by the patch-
    grouped palette. More readable than the pcolormesh heatmap for
    answering "which arm dominated?" since the visual weight is
    directly proportional.

    Gaps between events are left white by drawing one stackplot
    per event (same strategy as ``_plot_arm_posterior_heatmap``).
    """
    lo, hi = ctx.time_slice.start, ctx.time_slice.stop
    est = ctx.event_sequence_table
    view = est[(est["time"] >= lo) & (est["time"] <= hi)]

    ax.set_ylabel("arm mass\n(stacked)")
    ax.set_ylim(0, 1.01)
    ax.set_yticks([0, 0.5, 1])

    if len(view) == 0:
        ax.text(
            0.5, 0.5, "no in-event timebins in window",
            transform=ax.transAxes, ha="center", va="center", color="grey",
        )
        return

    arm_cols = [f"arm_posterior_{a}" for a in _ALL_ARMS]
    palette = ctx.arm_palette
    colors = [palette.get(a, (0.5, 0.5, 0.5, 1.0)) for a in _ALL_ARMS]

    for _event_id, rows in view.groupby("event_id", sort=False):
        times = rows["time"].to_numpy()
        if times.size < 2:
            continue
        mass = rows[arm_cols].to_numpy().T  # (n_arms, T)
        ax.stackplot(
            times, mass, colors=colors,
            edgecolor="none", alpha=0.95,
        )


# ── Replay fraction by role ──


# Role-specific shapes and desaturated colors for replay_frac_by_role.
# Deliberately DIFFERENT from the arm_markers panel (which uses
# ○ ▲ ▼ ★ with the arm-palette). The visual vocabulary separation
# signals "panel 9 = magnitude by role, panel 8 = arm identity".
_ROLE_SPEC = (
    ("current", "is_current", "s", "#708090"),       # square, slate
    ("prev", "is_prev", "D", "#5f9ea0"),             # diamond, cadet blue
    ("next (post-hoc)", "is_next", "P", "#cd5c5c"),  # plus, indian red
    ("last_rewarded", "is_last_rewarded", "h", "#daa520"),  # hexagon, goldenrod
)


def _plot_replay_frac_by_role(ax: Axes, ctx: _PanelContext) -> None:
    """Per-event arm_fraction at each of four canonical arm roles.

    For each event: look up the ``arm_fraction`` of the row where
    ``is_current == 1``, ``is_prev == 1``, ``is_next == 1``, and
    ``is_last_rewarded == 1``. Plot four stems per event at
    ``event_midpoint_time`` so the reader can immediately answer
    "how much replay went to the future destination vs. where the
    animal came from vs. where it is now?"

    This is the per-event analog of the plan's Stage 1 descriptive
    variables (``replay_frac_current`` / ``replay_frac_next`` /
    ``replay_frac_prev`` / ``replay_frac_last_rewarded``).
    """
    lo, hi = ctx.time_slice.start, ctx.time_slice.stop
    eat = ctx.event_arm_table
    et = ctx.event_table
    in_window = (
        (et["event_midpoint_time"] >= lo)
        & (et["event_midpoint_time"] <= hi)
    )
    et_w = et.loc[in_window]

    ax.set_ylabel("Replay Frac\n(by role)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 0.5, 1])
    ax.axhline(1.0 / len(_ALL_ARMS), color="grey", linewidth=0.5,
               linestyle="--", zorder=0, label="uniform 1/N")

    if len(et_w) == 0:
        ax.text(
            0.5, 0.5, "no events in window",
            transform=ax.transAxes, ha="center", va="center", color="grey",
        )
        return

    # Small x-offset per role so stems don't overlap.
    times = et_w["event_midpoint_time"].to_numpy()
    dt = float(np.median(np.abs(np.diff(times)))) if len(times) > 1 else 1.0
    offsets = np.linspace(-0.08 * dt, 0.08 * dt, len(_ROLE_SPEC))

    for (label, indicator_col, marker, color), x_off in zip(
        _ROLE_SPEC, offsets, strict=True
    ):
        fracs = np.full(len(et_w), np.nan)
        for i, (_, ev_row) in enumerate(et_w.iterrows()):
            arm_rows = eat[eat["event_id"] == ev_row["event_id"]]
            role_row = arm_rows[arm_rows[indicator_col] == 1]
            if len(role_row) == 1:
                fracs[i] = float(role_row["arm_fraction"].iloc[0])
        finite = np.isfinite(fracs)
        if not finite.any():
            continue
        ax.vlines(
            times[finite] + x_off, 0, fracs[finite],
            color=color, linewidth=0.6, zorder=2,
        )
        ax.scatter(
            times[finite] + x_off, fracs[finite],
            s=24, marker=marker, color=color,
            edgecolor="black", linewidths=0.3,
            zorder=3, clip_on=False, label=label,
        )

    ax.legend(
        bbox_to_anchor=(1.02, 1), loc="upper left",
        fontsize=7, framealpha=0.7,
    )


# ── Event / stop activity panels ──


_EVENT_TYPE_COLORS = {
    (True, True): "#5e4fa2",    # both SWR and MUA — deep purple
    (True, False): COLORS["SWR"],   # SWR only
    (False, True): COLORS["MUA"],   # MUA only
    (False, False): "#808080",  # neither — grey
}


def _plot_event_bars(ax: Axes, ctx: _PanelContext) -> None:
    """One bar per event, spanning ``event_start → event_end``.

    Height = ``n_nonlocal_bins`` (event size). Fill color encodes
    event-type membership:

    - SWR only → orange (``COLORS["SWR"]``)
    - MUA only → purple (``COLORS["MUA"]``)
    - Both → deep purple
    - Neither → grey

    Reads directly from ``event_table``. Lets the user see event
    density, duration, and type at a glance without having to
    correlate across panels.
    """
    lo, hi = ctx.time_slice.start, ctx.time_slice.stop
    et = ctx.event_table
    view = et[(et["event_start"] <= hi) & (et["event_end"] >= lo)]

    ax.set_ylabel("n_nonlocal\nbins")

    if len(view) == 0:
        ax.text(
            0.5, 0.5, "no events in window",
            transform=ax.transAxes, ha="center", va="center", color="grey",
        )
        ax.set_ylim(0, 1)
        return

    starts = view["event_start"].to_numpy()
    ends = view["event_end"].to_numpy()
    heights = view["n_nonlocal_bins"].to_numpy(dtype=float)
    is_swr = view["is_swr"].to_numpy(dtype=bool)
    is_mua = view["is_mua"].to_numpy(dtype=bool)

    for start, end, height, swr, mua in zip(
        starts, ends, heights, is_swr, is_mua, strict=True
    ):
        color = _EVENT_TYPE_COLORS[(bool(swr), bool(mua))]
        rect = Rectangle(
            (start, 0.0),
            width=max(end - start, 1e-3),
            height=height,
            facecolor=color,
            edgecolor="none",
            zorder=2,
        )
        ax.add_patch(rect)

    ax.set_ylim(0, float(heights.max()) * 1.1 if heights.size else 1.0)


def _plot_stop_activity(ax: Axes, ctx: _PanelContext) -> None:
    """Per-stop activity bars.

    For every stop in window: a vertical stem at the stop's center
    from 0 to ``n_events``. A filled marker at the top means the
    stop had at least one non-local event (``has_replay=True``);
    hollow means zero-event. Small 'S' / 'M' letter labels above the
    stem flag ``is_swr_stop`` / ``is_mua_stop``.

    Zero-event stops are the hurdle model's Part-1 inputs; their
    visual presence here is itself a sanity check that
    ``build_stop_table`` is emitting them.
    """
    lo, hi = ctx.time_slice.start, ctx.time_slice.stop
    st = ctx.stop_table
    view = st[(st["stop_start"] <= hi) & (st["stop_end"] >= lo)]

    ax.set_ylabel("n_events\nper stop")
    ax.axhline(0, color="grey", linewidth=0.3, zorder=0)

    if len(view) == 0:
        ax.text(
            0.5, 0.5, "no stops in window",
            transform=ax.transAxes, ha="center", va="center", color="grey",
        )
        ax.set_ylim(0, 1)
        return

    mids = 0.5 * (
        view["stop_start"].to_numpy() + view["stop_end"].to_numpy()
    )
    n_events = view["n_events"].to_numpy(dtype=float)
    has_replay = view["has_replay"].to_numpy(dtype=bool)
    is_swr_stop = view["is_swr_stop"].to_numpy(dtype=bool)
    is_mua_stop = view["is_mua_stop"].to_numpy(dtype=bool)

    ax.vlines(mids, 0, n_events, color="black", linewidth=0.5, zorder=2)
    # Two-pass scatter so "has_replay" stops are filled and
    # zero-event stops are hollow but both use the same axis.
    filled = has_replay
    if filled.any():
        ax.scatter(
            mids[filled], n_events[filled],
            s=18, c="#3F51B5", edgecolor="black", linewidths=0.5,
            zorder=3, clip_on=False,
        )
    if (~filled).any():
        ax.scatter(
            mids[~filled], n_events[~filled],
            s=18, facecolor="white", edgecolor="black", linewidths=0.5,
            zorder=3, clip_on=False,
        )

    ymax = float(n_events.max()) * 1.15 if n_events.size else 1.0
    for x, swr, mua, n in zip(mids, is_swr_stop, is_mua_stop, n_events, strict=True):
        y = max(n, 0.05 * ymax) + 0.05 * ymax
        if swr:
            ax.text(
                x, y, "S",
                ha="center", va="bottom", fontsize=7,
                color=COLORS["SWR"], weight="bold",
            )
        if mua:
            ax.text(
                x, y + 0.08 * ymax, "M",
                ha="center", va="bottom", fontsize=7,
                color=COLORS["MUA"], weight="bold",
            )

    ax.set_ylim(0, ymax if ymax > 0 else 1.0)


def _plot_context_flags(ax: Axes, ctx: _PanelContext) -> None:
    """Event-level boolean context flags as a binary heatmap.

    Six flag rows (``is_after_reward``, ``is_after_omission``,
    ``is_after_switch``, ``is_after_stay``, ``is_swr``, ``is_mua``)
    displayed as filled cells per event: colored when True, white
    when False. The heatmap encoding is more legible than ``|``
    ticks at this density because flag co-occurrence within a single
    event reads as a filled column rather than overlapping marks.
    """
    lo, hi = ctx.time_slice.start, ctx.time_slice.stop
    et = ctx.event_table
    view = et[
        (et["event_midpoint_time"] >= lo)
        & (et["event_midpoint_time"] <= hi)
    ].sort_values("event_midpoint_time")

    flag_specs = (
        ("is_after_reward", "#2ca02c"),
        ("is_after_omission", "#d62728"),
        ("is_after_switch", "#9467bd"),
        ("is_after_stay", "#8c564b"),
        ("is_swr", COLORS["SWR"]),
        ("is_mua", COLORS["MUA"]),
    )
    flag_names = [name for name, _ in flag_specs]
    ax.set_yticks(range(len(flag_specs)))
    ax.set_yticklabels(flag_names, fontsize=7)
    ax.set_ylim(-0.5, len(flag_specs) - 0.5)
    ax.set_ylabel("Context\nFlags")

    if len(view) == 0:
        ax.text(
            0.5, 0.5, "no events in window",
            transform=ax.transAxes, ha="center", va="center", color="grey",
        )
        return

    n_events = len(view)
    starts = view["event_start"].to_numpy()
    ends = view["event_end"].to_numpy()

    for row_idx, (flag, color) in enumerate(flag_specs):
        vals = np.asarray(view[flag].to_numpy(), dtype=float)
        for j in range(n_events):
            if vals[j] == 1.0:
                rect = Rectangle(
                    (starts[j], row_idx - 0.4),
                    width=max(ends[j] - starts[j], (hi - lo) * 0.003),
                    height=0.8,
                    facecolor=color, edgecolor="none", alpha=0.85,
                    zorder=2,
                )
                ax.add_patch(rect)


# ── Scalar per-event panels ──


def _draw_event_scalar_stem(
    ax: Axes,
    ctx: _PanelContext,
    *,
    value_col: str,
    ylabel: str,
    ylim: tuple[float, float] | None = None,
    baseline: float = 0.0,
    marker_color: str = "#3F51B5",
) -> None:
    """Stem plot of a per-event scalar against time.

    One value per event at ``event_midpoint_time``: a thin vertical
    line from ``baseline`` to the value, plus a dot at the top. Works
    for any column of ``event_table`` whose dtype is numeric.
    """
    lo, hi = ctx.time_slice.start, ctx.time_slice.stop
    et = ctx.event_table
    in_window = (
        (et["event_midpoint_time"] >= lo)
        & (et["event_midpoint_time"] <= hi)
    )
    view = et.loc[in_window]

    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.axhline(baseline, color="grey", linewidth=0.3, zorder=0)
    ax.set_ylabel(ylabel)

    if len(view) == 0:
        ax.text(
            0.5, 0.5, "no events in window",
            transform=ax.transAxes, ha="center", va="center", color="grey",
        )
        return

    times = view["event_midpoint_time"].to_numpy()
    values = view[value_col].to_numpy(dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        ax.text(
            0.5, 0.5, f"{value_col}: all NaN in window",
            transform=ax.transAxes, ha="center", va="center", color="grey",
        )
        return
    ax.vlines(
        times[finite], baseline, values[finite],
        color="black", linewidth=0.5, zorder=2,
    )
    ax.scatter(
        times[finite], values[finite],
        s=12, c=marker_color, zorder=3, clip_on=False,
    )


def _plot_posterior_peak_mean(ax: Axes, ctx: _PanelContext) -> None:
    _draw_event_scalar_stem(
        ax, ctx,
        value_col="posterior_peak_mean",
        ylabel="posterior\npeak_mean",
        ylim=(1 / len(_ALL_ARMS) - 0.05, 1.05),
        baseline=1 / len(_ALL_ARMS),
    )


def _plot_posterior_entropy_mean(ax: Axes, ctx: _PanelContext) -> None:
    _draw_event_scalar_stem(
        ax, ctx,
        value_col="posterior_entropy_mean",
        ylabel="posterior\nentropy_mean",
        ylim=(-0.05, float(np.log(len(_ALL_ARMS))) + 0.1),
        baseline=0.0,
    )


def _plot_hpd_widths(ax: Axes, ctx: _PanelContext) -> None:
    """Three HPD widths (50/80/95) overlaid on one axis.

    Line thickness increases with coverage so the heavier coverage
    level is visually "outside" the lighter one, matching the
    monotonicity invariant ``width_50 ≤ width_80 ≤ width_95``.
    A mismatch with the invariant would read as a crossover.
    """
    lo, hi = ctx.time_slice.start, ctx.time_slice.stop
    et = ctx.event_table
    in_window = (
        (et["event_midpoint_time"] >= lo)
        & (et["event_midpoint_time"] <= hi)
    )
    view = et.loc[in_window].sort_values("event_midpoint_time")

    ax.set_ylim(0.8, len(_ALL_ARMS) + 0.2)
    ax.set_ylabel("HPD width\n50/80/95")

    if len(view) == 0:
        ax.text(
            0.5, 0.5, "no events in window",
            transform=ax.transAxes, ha="center", va="center", color="grey",
        )
        return

    times = view["event_midpoint_time"].to_numpy()
    specs = (
        ("posterior_hpd_width_50", 0.8, "#a6cee3", "50"),
        ("posterior_hpd_width_80", 1.3, "#1f78b4", "80"),
        ("posterior_hpd_width_95", 1.8, "#08306b", "95"),
    )
    for col, lw, color, label in specs:
        vals = view[col].to_numpy(dtype=float)
        ax.plot(
            times, vals, color=color, linewidth=lw, marker="o",
            markersize=3, zorder=3, clip_on=False, label=label,
        )
    ax.legend(
        bbox_to_anchor=(1.02, 1), loc="upper left",
        fontsize=7, framealpha=0.7,
    )


def _plot_n_nonlocal_bins(ax: Axes, ctx: _PanelContext) -> None:
    _draw_event_scalar_stem(
        ax, ctx,
        value_col="n_nonlocal_bins",
        ylabel="n_nonlocal\nbins",
        ylim=None,
        baseline=0.0,
    )


def _plot_time_since_stop(ax: Axes, ctx: _PanelContext) -> None:
    _draw_event_scalar_stem(
        ax, ctx,
        value_col="time_since_stop",
        ylabel="time_since\nstop [s]",
        ylim=None,
        baseline=0.0,
    )


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
    "arm_posterior_heatmap": _plot_arm_posterior_heatmap,
    "arm_count_hard": _plot_arm_count_hard,
    "arm_count_soft": _plot_arm_count_soft,
    "arm_posterior_mass": _plot_arm_posterior_mass,
    "arm_fraction": _plot_arm_fraction,
    "posterior_peak_mean": _plot_posterior_peak_mean,
    "posterior_entropy_mean": _plot_posterior_entropy_mean,
    "hpd_widths": _plot_hpd_widths,
    "n_nonlocal_bins": _plot_n_nonlocal_bins,
    "time_since_stop": _plot_time_since_stop,
    "visit_recency": _plot_visit_recency,
    "cumulative_reward": _plot_cumulative_reward,
    "recent_reward_rate": _plot_recent_reward_rate,
    "nominal_reward_prob": _plot_nominal_reward_prob,
    "dist_to_arm": _plot_dist_to_arm,
    "arm_dist_rank": _plot_arm_dist_rank,
    "arm_mass_stackplot": _plot_arm_mass_stackplot,
    "replay_frac_by_role": _plot_replay_frac_by_role,
    "event_bars": _plot_event_bars,
    "stop_activity": _plot_stop_activity,
    "context_flags": _plot_context_flags,
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
    "arm_posterior_heatmap": 2.0,
    "arm_count_hard": 1.4,
    "arm_count_soft": 1.4,
    "arm_posterior_mass": 1.4,
    "arm_fraction": 1.4,
    "posterior_peak_mean": 1.0,
    "posterior_entropy_mean": 1.0,
    "hpd_widths": 1.0,
    "n_nonlocal_bins": 1.0,
    "time_since_stop": 1.0,
    "visit_recency": 1.4,
    "cumulative_reward": 1.4,
    "recent_reward_rate": 1.4,
    "nominal_reward_prob": 1.4,
    "dist_to_arm": 1.4,
    "arm_dist_rank": 1.4,
    "arm_mass_stackplot": 1.4,
    "replay_frac_by_role": 1.2,
    "event_bars": 1.2,
    "stop_activity": 1.2,
    "context_flags": 1.4,
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
    axes[-1].ticklabel_format(useOffset=False, style="plain", axis="x")

    # Vertical event-span guides across ALL panels so the reader can
    # correlate "what is the state probability here?" with "what is
    # the arm fraction here?" without manually tracking x-coordinates.
    et_windowed = event_table[
        (event_table["event_start"] <= t_end)
        & (event_table["event_end"] >= t_start)
    ]
    for _, ev_row in et_windowed.iterrows():
        for ax in axes:
            ax.axvspan(
                ev_row["event_start"],
                ev_row["event_end"],
                alpha=0.04,
                color="black",
                zorder=0,
            )

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
