"""Response and decoder-state panels: core detector rows, trial rail, derivation-debug rows, content heatmaps, event/stop activity and per-event scalars."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

from continuum_swr_replay.analysis.behavior.covariates import (
    trial_arrival_times,
    trial_context_at_times,
)
from continuum_swr_replay.analysis.behavior.spatial_mapping import single_arm_from_well
from continuum_swr_replay.data_loaders.constants import ALL_ARMS as _ALL_ARMS
from continuum_swr_replay.data_loaders.constants import WELL_PATCH_MAPPING
from continuum_swr_replay.intervals import sample_step
from continuum_swr_replay.parameters import DECODE_SAMPLE_S
from continuum_swr_replay.tables.columns import require_columns
from continuum_swr_replay.visualization.palettes import COLORS

from ..plots import (
    plot_acausal_probabilities,
    plot_conditional_non_local_posterior,
    plot_multiunit_firing,
    plot_ripple_consensus_trace_zscore,
    plot_speed,
    plot_spike_times,
)
from ._common import (
    _MISSING_CELL_COLOR,
    _draw_event_arm_heatmap,
    _PanelContext,
    _validate_binary,
)

#: The trial columns the trial panels and the next-arm audit read.
_TRIAL_COLUMNS = (
    "movement_start_time",
    "end_time",
    "from_well",
    "to_well",
    "is_reward",
    "is_patch_change",
)

_MISMATCH_COLUMNS: tuple[str, ...] = (
    "event_id",
    "event_start_s",
    "event_midpoint_s",
    "canonical_next_arm",
    "actual_next_to_well",
    "actual_next_arm",
    "match",
)


# ── Core panels (thin adapters over existing plot_detector helpers) ──


def _plot_swr_lfp(ax: Axes, ctx: _PanelContext) -> None:
    # ``ripple_times`` here is the caller's SWR source intervals
    # (``plot_session_tables(swr_intervals=...)``), never load_data's.
    plot_ripple_consensus_trace_zscore(ctx.data, ctx.time_slice, ax)


def _plot_mua(ax: Axes, ctx: _PanelContext) -> None:
    plot_multiunit_firing(ctx.data, ctx.time_slice, ax, plot_multiunit_zscore=True)


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
        ctx.data, ctx.time_slice, ctx.results, ctx.env, ax
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
            t_left,
            y_center,
            f"a{arm}",
            fontsize=7,
            va="center",
            ha="right",
            color=color,
            fontweight="bold",
            clip_on=False,
        )


def _plot_speed(ax: Axes, ctx: _PanelContext) -> None:
    plot_speed(ctx.data, ctx.time_slice, ax)
    from continuum_swr_replay.parameters import DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC

    ax.axhline(
        DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
        color="gray",
        linestyle="--",
        linewidth=0.8,
        zorder=5,
    )


# ── New: trial rail ──


def _plot_trial_rail(ax: Axes, ctx: _PanelContext) -> None:
    """Production-aligned behavioral location and completed-trip context.

    Dark horizontal segments show dwell intervals at a well. Light shading
    marks ``[movement_start_time, destination_first_poke_in_time)`` transit
    intervals, during which
    destination and outcome are not exposed as current context. Reward stars
    and purple patch-switch ticks occur at the completion cutoff (goal arrival,
    with the established per-row ``end_time`` fallback).
    """
    trials = ctx.trials
    lo, hi = ctx.time_slice.start, ctx.time_slice.stop
    require_columns(trials, _TRIAL_COLUMNS, "trials")
    if len(trials) == 0:
        ax.text(
            0.5,
            0.5,
            "no trials in window",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="gray",
        )
        ax.set_ylabel("behavioral\nlocation")
        ax.set_ylim(-0.5, 5.5)
        return

    # keep the trial ids: the port-experience stream names trips by them
    ordered = trials.sort_values("movement_start_time", kind="stable")
    start_times = ordered["movement_start_time"].to_numpy(dtype=float)
    if not np.isfinite(start_times).all() or np.any(np.diff(start_times) < 0):
        raise ValueError(
            "trials.movement_start_time must be finite and chronologically sortable"
        )

    # Validate using the same helper as the canonical table builders.
    trial_context_at_times(np.array([lo]), ordered, ctx.port_experiences)
    completion_times = trial_arrival_times(ordered)

    drew_interval = False
    for trial_idx, (_, row) in enumerate(ordered.iterrows()):
        transit_start = max(float(row["movement_start_time"]), lo)
        transit_end = min(float(completion_times[trial_idx]), hi)
        if transit_end > transit_start:
            ax.axvspan(
                transit_start,
                transit_end,
                color="#bdbdbd",
                alpha=0.22,
                linewidth=0,
                zorder=0,
            )
            drew_interval = True

        next_start = (
            float(start_times[trial_idx + 1]) if trial_idx + 1 < len(ordered) else hi
        )
        dwell_start = max(float(completion_times[trial_idx]), lo)
        dwell_end = min(next_start, hi)
        if dwell_end > dwell_start:
            well = float(row["to_well"])
            ax.plot(
                [dwell_start, dwell_end],
                [well, well],
                color="#333333",
                linewidth=1.2,
                solid_capstyle="butt",
                clip_on=False,
            )
            drew_interval = True

    if not drew_interval:
        ax.text(
            0.5,
            0.5,
            "no trials in window",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="gray",
        )
        ax.set_ylabel("behavioral\nlocation")
        ax.set_ylim(-0.5, 5.5)
        return

    # Reward markers use the same arrival cutoffs as last-rewarded, recency,
    # cumulative-reward, and recent-reward-rate production covariates.
    reward_times = completion_times
    reward_mask = (
        ordered["is_reward"].to_numpy(dtype=bool)
        & (reward_times >= lo)
        & (reward_times <= hi)
    )
    if reward_mask.any():
        ax.scatter(
            reward_times[reward_mask],
            ordered.loc[reward_mask, "to_well"].to_numpy(dtype=float),
            marker="*",
            s=70,
            facecolor="gold",
            edgecolor="black",
            linewidths=0.5,
            zorder=5,
            clip_on=False,
        )

    # Patch-change outcome becomes past context at trip completion.
    switch_mask = (
        ordered["is_patch_change"].to_numpy(dtype=bool)
        & (completion_times >= lo)
        & (completion_times <= hi)
    )
    if switch_mask.any():
        for t in completion_times[switch_mask]:
            ax.axvline(t, color="#9467bd", linewidth=0.7, alpha=0.5)

    # Per-well horizontal guidelines so step levels are easy to read.
    for well in sorted(WELL_PATCH_MAPPING):
        ax.axhline(well, color="lightgray", linewidth=0.3, zorder=0)

    ax.set_ylabel("behavioral\nlocation")
    ax.set_ylim(-0.5, 5.5)
    ax.set_yticks(sorted(WELL_PATCH_MAPPING))


# ── Future-covariate debug: arm markers + mismatch pointers ──


def _compute_mismatch_table(
    event_table: pd.DataFrame,
    trials: pd.DataFrame,
    port_experiences: pd.DataFrame,
    track_graph: Any,
) -> pd.DataFrame:
    """Compare canonical ``next_arm`` against trials' actual next trip.

    Canonical trial context is sampled from the event's first retained
    timebin, so the audit is evaluated at ``event_start_s`` (not the display
    midpoint). The recomputation delegates to the production
    :func:`trial_context_at_times` and :func:`single_arm_from_well` helpers;
    boundary, stable-sort, pre-first-trial, and session-tail semantics therefore
    cannot drift into a second visualization-only implementation.

    ``event_midpoint_s`` is retained solely as the display coordinate
    used by :func:`_plot_mismatch_pointers`.

    ``match`` is ``True`` when the canonical and actual arms are equal
    OR when both are ``NaN`` (legitimate session end: the last event
    has no next trip). Any other combination is a mismatch and will be
    drawn in red by :func:`_plot_mismatch_pointers`.
    """
    require_columns(
        event_table,
        ("event_id", "event_start_s", "event_midpoint_s", "next_arm"),
        "event_table",
    )
    if len(event_table) == 0:
        return pd.DataFrame(columns=_MISMATCH_COLUMNS)
    require_columns(trials, _TRIAL_COLUMNS, "trials")

    event_starts = event_table["event_start_s"].to_numpy(dtype=float)
    event_midpoints = event_table["event_midpoint_s"].to_numpy(dtype=float)
    actual_to_well = trial_context_at_times(event_starts, trials, port_experiences)[
        "next_well"
    ].to_numpy(dtype=float)
    actual_arm = np.array(
        [single_arm_from_well(well, track_graph) for well in actual_to_well]
    )
    canonical = event_table["next_arm"].to_numpy(dtype=float)

    both_nan = np.isnan(canonical) & np.isnan(actual_arm)
    equal = (canonical == actual_arm) & ~np.isnan(canonical) & ~np.isnan(actual_arm)
    match = both_nan | equal

    return pd.DataFrame(
        {
            "event_id": event_table["event_id"].to_numpy(),
            "event_start_s": event_starts,
            "event_midpoint_s": event_midpoints,
            "canonical_next_arm": canonical,
            "actual_next_to_well": actual_to_well,
            "actual_next_arm": actual_arm,
            "match": match,
        }
    )


def _plot_arm_markers(ax: Axes, ctx: _PanelContext) -> None:
    """Canonical arm references per event.

    At each event's ``event_midpoint_s``, four markers in the
    arm-palette color:

    - ``occupancy_well_arm`` — filled circle
    - ``prev_arm`` — upward triangle ▲
    - ``next_arm`` — downward triangle ▼ (**post-hoc**)
    - ``last_rewarded_arm`` — star ★

    Markers are skipped where the corresponding column is NaN. Y-axis
    is the 9-arm index space grouped visually by the patch palette.
    """
    lo, hi = ctx.time_slice.start, ctx.time_slice.stop
    et = ctx.event_table
    in_window = (et["event_midpoint_s"] >= lo) & (et["event_midpoint_s"] <= hi)
    et = et.loc[in_window]

    for arm in _ALL_ARMS:
        ax.axhline(arm, color="lightgray", linewidth=0.3, zorder=0)

    if len(et) == 0:
        ax.text(
            0.5,
            0.5,
            "no events in window",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="gray",
        )
    else:
        times = et["event_midpoint_s"].to_numpy()
        palette = ctx.arm_palette
        marker_spec = (
            ("occupancy_well_arm", "o", 36),
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

    # Legend for the four marker types.
    from matplotlib.lines import Line2D as _L2D

    ax.legend(
        handles=[
            _L2D(
                [],
                [],
                marker="o",
                color="gray",
                markersize=6,
                markeredgecolor="black",
                markeredgewidth=0.3,
                linestyle="none",
                label="current",
            ),
            _L2D(
                [],
                [],
                marker="^",
                color="gray",
                markersize=6,
                markeredgecolor="black",
                markeredgewidth=0.3,
                linestyle="none",
                label="prev",
            ),
            _L2D(
                [],
                [],
                marker="v",
                color="gray",
                markersize=6,
                markeredgecolor="black",
                markeredgewidth=0.3,
                linestyle="none",
                label="next (post-hoc)",
            ),
            _L2D(
                [],
                [],
                marker="*",
                color="gray",
                markersize=8,
                markeredgecolor="black",
                markeredgewidth=0.3,
                linestyle="none",
                label="last rewarded",
            ),
        ],
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=7,
        framealpha=0.7,
    )
    ax.set_ylabel("Arm\nIdentity")
    ax.set_ylim(-0.5, 8.5)
    ax.set_yticks(list(_ALL_ARMS))


def _plot_mismatch_pointers(ax: Axes, ctx: _PanelContext) -> None:
    """Canonical ``next_arm`` versus trials' actual next trip.

    For every event in the window: a ▼ at the canonical
    ``event_table['next_arm']`` and a × at the arm recomputed from ``trials``
    through the production event-start context rule. Markers are displayed at
    ``event_midpoint_s`` for alignment with the other event panels; that
    midpoint is not used to select the next trial.
    A vertical connector between them — **red when the two disagree**,
    thin gray when they match.

    ``(NaN, NaN)`` — the legitimate session-end case where the last
    event has no next trip — is treated as a match.

    The underlying mismatch table is computed once per ``ctx`` and
    stashed on the context so the caller can retrieve it from the
    public entry point.
    """
    lo, hi = ctx.time_slice.start, ctx.time_slice.stop
    mismatch = _compute_mismatch_table(
        ctx.event_table, ctx.trials, ctx.port_experiences, ctx.track_graph
    )
    in_window = (mismatch["event_midpoint_s"] >= lo) & (
        mismatch["event_midpoint_s"] <= hi
    )
    view = mismatch.loc[in_window]

    for arm in _ALL_ARMS:
        ax.axhline(arm, color="lightgray", linewidth=0.3, zorder=0)

    if len(view) == 0:
        ax.text(
            0.5,
            0.5,
            "no events in window",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="gray",
        )
    else:
        times = view["event_midpoint_s"].to_numpy()
        canonical = view["canonical_next_arm"].to_numpy(dtype=float)
        actual = view["actual_next_arm"].to_numpy(dtype=float)
        is_match = view["match"].to_numpy(dtype=bool)

        # Small x-jitter so ▼ (canonical) and × (actual) don't
        # overplot when they agree, but both sit at the SAME y so
        # agreement reads as "same row" and disagreement reads as
        # "different rows connected by a red line".
        dt_jitter = (ctx.time_slice.stop - ctx.time_slice.start) * 0.003

        # Connector lines first so markers sit on top.
        for t, c_arm, a_arm, match in zip(
            times, canonical, actual, is_match, strict=True
        ):
            c_nan, a_nan = np.isnan(c_arm), np.isnan(a_arm)
            if c_nan and a_nan:
                continue
            if c_nan ^ a_nan:
                present = c_arm if not c_nan else a_arm
                ax.plot(
                    [t, t],
                    [present - 0.15, present + 0.15],
                    color="red",
                    linewidth=1.6,
                    zorder=2,
                    clip_on=False,
                )
                continue
            if not match:
                ax.plot(
                    [t, t],
                    [c_arm, a_arm],
                    color="red",
                    linewidth=1.6,
                    zorder=2,
                    clip_on=False,
                )

        palette = ctx.arm_palette
        c_mask = ~np.isnan(canonical)
        if c_mask.any():
            ax.scatter(
                times[c_mask] - dt_jitter,
                canonical[c_mask],
                marker="v",
                s=36,
                c=[
                    palette.get(int(a), (0.5, 0.5, 0.5, 1.0)) for a in canonical[c_mask]
                ],
                edgecolor="black",
                linewidths=0.3,
                zorder=3,
                clip_on=False,
            )
        a_mask = ~np.isnan(actual)
        if a_mask.any():
            ax.scatter(
                times[a_mask] + dt_jitter,
                actual[a_mask],
                marker="x",
                s=40,
                c=[palette.get(int(a), (0.5, 0.5, 0.5, 1.0)) for a in actual[a_mask]],
                linewidths=1.0,
                zorder=4,
                clip_on=False,
            )

    ax.set_ylabel("Next-Arm\nCheck")
    ax.set_ylim(-0.5, 8.5)
    ax.set_yticks(list(_ALL_ARMS))

    # Legend — explain the two markers and the connector color.
    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D(
            [],
            [],
            marker="v",
            color="gray",
            markersize=6,
            markeredgecolor="black",
            markeredgewidth=0.3,
            linestyle="none",
            label="canonical next_arm",
        ),
        Line2D(
            [],
            [],
            marker="x",
            color="gray",
            markersize=6,
            markeredgewidth=1.0,
            linestyle="none",
            label="actual (from trials)",
        ),
        Line2D([], [], color="lightgray", linewidth=1.5, label="match"),
        Line2D([], [], color="red", linewidth=1.5, label="MISMATCH"),
    ]
    ax.legend(
        handles=legend_handles,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=7,
        framealpha=0.7,
    )

    # Stash the full (unfiltered) mismatch table on the context so the
    # public entry point can return it without recomputing.
    if ctx._mismatch_table is None:
        ctx._mismatch_table = mismatch


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
        ax.axhline(arm, color="lightgray", linewidth=0.3, zorder=0)

    if len(view) == 0:
        ax.text(
            0.5,
            0.5,
            "no in-event timebins in window",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="gray",
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
    dt_fallback = (
        float(np.median(diffs[diffs > 0])) if (diffs > 0).any() else DECODE_SAMPLE_S
    )

    # Draw one pcolormesh per event — matplotlib doesn't interpolate
    # across discontinuous x, so per-event calls keep gaps white.
    for _event_id, rows in view.groupby("event_id", sort=False):
        times = rows["time"].to_numpy()
        if times.size < 1:
            continue
        if times.size == 1:
            dt = dt_fallback
        else:
            dt = sample_step(times)
        edges_t = np.concatenate([times - dt / 2, [times[-1] + dt / 2]])
        posterior = rows[arm_cols].to_numpy().T  # (n_arms, T)
        ax.pcolormesh(
            edges_t,
            edges_a,
            posterior,
            cmap="bone_r",
            vmin=0.0,
            vmax=1.0,
            shading="flat",
            zorder=1,
        )

    # MAP-arm overlay (decoded_arm_map per timebin).
    ax.scatter(
        view["time"].to_numpy(),
        view["decoded_arm_map"].to_numpy(),
        s=4,
        c="black",
        marker=".",
        zorder=4,
        clip_on=False,
    )

    ax.set_ylabel("arm\nposterior")
    ax.set_ylim(-0.5, 8.5)
    ax.set_yticks(list(_ALL_ARMS))


def _plot_arm_count_hard(ax: Axes, ctx: _PanelContext) -> None:
    _draw_event_arm_heatmap(
        ax,
        ctx,
        value_col="arm_count_hard",
        cmap_name="viridis",
        vmin=0,
        vmax=None,
        label="MAP arm count\n[decoded bins]",
    )


def _plot_arm_count_soft(ax: Axes, ctx: _PanelContext) -> None:
    _draw_event_arm_heatmap(
        ax,
        ctx,
        value_col="arm_count_soft",
        cmap_name="viridis",
        vmin=0,
        vmax=None,
        label="soft arm count\n[effective bins]",
    )


def _plot_arm_posterior_mass(ax: Axes, ctx: _PanelContext) -> None:
    _draw_event_arm_heatmap(
        ax,
        ctx,
        value_col="arm_posterior_mass",
        cmap_name="viridis",
        vmin=0,
        vmax=None,
        label="arm post. mass\n[effective bins]",
    )


def _plot_arm_fraction(ax: Axes, ctx: _PanelContext) -> None:
    _draw_event_arm_heatmap(
        ax,
        ctx,
        value_col="arm_fraction",
        cmap_name="viridis",
        vmin=0.0,
        vmax=1.0,
        label="MAP arm frac.\n[real-arm bins]",
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

    ax.set_ylabel("Arm Mass\n(proportion)")
    ax.set_ylim(0, 1.01)
    ax.set_yticks([0, 0.5, 1])

    if len(view) == 0:
        ax.text(
            0.5,
            0.5,
            "no in-event timebins in window",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="gray",
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
            times,
            mass,
            colors=colors,
            edgecolor="none",
            alpha=0.95,
        )


# ── Replay fraction by role ──


# Role-specific shapes and desaturated colors for replay_frac_by_role.
# Deliberately DIFFERENT from the arm_markers panel (which uses
# ○ ▲ ▼ ★ with the arm-palette). The visual vocabulary separation
# signals "panel 9 = magnitude by role, panel 8 = arm identity".
_ROLE_SPEC = (
    ("current", "is_current_arm", "s", "#708090"),  # square, slate
    ("prev", "is_prev_arm", "D", "#5f9ea0"),  # diamond, cadet blue
    ("next (post-hoc)", "is_next_arm", "P", "#cd5c5c"),  # plus, indian red
    ("last_rewarded", "is_last_rewarded_arm", "h", "#daa520"),  # hexagon, goldenrod
)


def _plot_replay_frac_by_role(ax: Axes, ctx: _PanelContext) -> None:
    """Per-event arm_fraction at each of four canonical arm roles.

    For each event: look up the ``arm_fraction`` of the row where
    ``is_current == 1``, ``is_prev == 1``, ``is_next == 1``, and
    ``is_last_rewarded == 1``. Plot four stems per event at
    ``event_midpoint_s`` so the reader can immediately answer
    "how much replay went to the future destination vs. where the
    animal came from vs. where it is now?"

    This is the per-event analog of the descriptive
    variables (``replay_frac_current`` / ``replay_frac_next`` /
    ``replay_frac_prev`` / ``replay_frac_last_rewarded``).
    """
    lo, hi = ctx.time_slice.start, ctx.time_slice.stop
    eat = ctx.event_arm_table
    et = ctx.event_table
    in_window = (et["event_midpoint_s"] >= lo) & (et["event_midpoint_s"] <= hi)
    et_w = et.loc[in_window]

    ax.set_ylabel("Replay Frac\n(by role)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 0.5, 1])
    ax.axhline(
        1.0 / len(_ALL_ARMS),
        color="gray",
        linewidth=0.5,
        linestyle="--",
        zorder=0,
        label="uniform 1/N",
    )

    if len(et_w) == 0:
        ax.text(
            0.5,
            0.5,
            "no events in window",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="gray",
        )
        return

    # Small x-offset per role so stems don't overlap.
    times = et_w["event_midpoint_s"].to_numpy()
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
            times[finite] + x_off,
            0,
            fracs[finite],
            color=color,
            linewidth=0.6,
            zorder=2,
        )
        ax.scatter(
            times[finite] + x_off,
            fracs[finite],
            s=24,
            marker=marker,
            color=color,
            edgecolor="black",
            linewidths=0.3,
            zorder=3,
            clip_on=False,
            label=label,
        )

    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=7,
        framealpha=0.7,
    )


# ── Event / stop activity panels ──


_EVENT_TYPE_COLORS = {
    (True, True): "#5e4fa2",  # both SWR and MUA — deep purple
    (True, False): COLORS["SWR"],  # SWR only
    (False, True): COLORS["MUA"],  # MUA only
    (False, False): "#808080",  # neither — gray
}


def _plot_event_bars(ax: Axes, ctx: _PanelContext) -> None:
    """One bar per event, spanning ``event_start_s → event_end_s``.

    Height = ``n_nonlocal_bins`` (event size). Fill color encodes
    event-type membership:

    - SWR only → orange (``COLORS["SWR"]``)
    - MUA only → purple (``COLORS["MUA"]``)
    - Both → deep purple
    - Neither → gray

    Reads directly from ``event_table``. Lets the user see event
    density, duration, and type at a glance without having to
    correlate across panels.
    """
    lo, hi = ctx.time_slice.start, ctx.time_slice.stop
    et = ctx.event_table
    view = et[(et["event_start_s"] <= hi) & (et["event_end_s"] >= lo)]

    ax.set_ylabel("n_nonlocal\nbins")

    if len(view) == 0:
        ax.text(
            0.5,
            0.5,
            "no events in window",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="gray",
        )
        ax.set_ylim(0, 1)
        return

    starts = view["event_start_s"].to_numpy()
    ends = view["event_end_s"].to_numpy()
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
    from 0 to ``n_selected_events``. A filled marker at the top means the
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

    ax.set_ylabel("selected events\nper stop")
    ax.axhline(0, color="gray", linewidth=0.3, zorder=0)

    if len(view) == 0:
        ax.text(
            0.5,
            0.5,
            "no stops in window",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="gray",
        )
        ax.set_ylim(0, 1)
        return

    mids = 0.5 * (view["stop_start"].to_numpy() + view["stop_end"].to_numpy())
    n_events = view["n_selected_events"].to_numpy(dtype=float)
    has_replay = view["has_replay"].to_numpy(dtype=bool)
    is_swr_stop = view["is_swr_stop"].to_numpy(dtype=bool)
    is_mua_stop = view["is_mua_stop"].to_numpy(dtype=bool)

    ax.vlines(mids, 0, n_events, color="black", linewidth=0.5, zorder=2)
    # Two-pass scatter so "has_replay" stops are filled and
    # zero-event stops are hollow but both use the same axis.
    filled = has_replay
    if filled.any():
        ax.scatter(
            mids[filled],
            n_events[filled],
            s=18,
            c="#3F51B5",
            edgecolor="black",
            linewidths=0.5,
            zorder=3,
            clip_on=False,
        )
    if (~filled).any():
        ax.scatter(
            mids[~filled],
            n_events[~filled],
            s=18,
            facecolor="white",
            edgecolor="black",
            linewidths=0.5,
            zorder=3,
            clip_on=False,
        )

    ymax = float(n_events.max()) * 1.15 if n_events.size else 1.0
    for x, swr, mua, n in zip(mids, is_swr_stop, is_mua_stop, n_events, strict=True):
        y = max(n, 0.05 * ymax) + 0.05 * ymax
        if swr:
            ax.text(
                x,
                y,
                "S",
                ha="center",
                va="bottom",
                fontsize=7,
                color=COLORS["SWR"],
                weight="bold",
            )
        if mua:
            ax.text(
                x,
                y + 0.08 * ymax,
                "M",
                ha="center",
                va="bottom",
                fontsize=7,
                color=COLORS["MUA"],
                weight="bold",
            )

    ax.set_ylim(0, ymax if ymax > 0 else 1.0)


def _plot_context_flags(ax: Axes, ctx: _PanelContext) -> None:
    """Event-level boolean context flags as a binary heatmap.

    Six flag rows (rewarded, omitted, switched patch, stayed in patch,
    ``is_swr``, ``is_mua``)
    displayed as filled cells per event: colored when True, white
    when False. The heatmap encoding is more legible than ``|``
    ticks at this density because flag co-occurrence within a single
    event reads as a filled column rather than overlapping marks.

    Undefined is drawn as a distinct hatched gray cell, never as white.
    ``build_event_table`` assigns ``NaN`` to the outcome/transition flags at
    session boundaries, so the first stop of every session has genuinely
    undefined trial context; rendering that as white would read as "all four
    context flags observed False" — i.e. as evidence the covariates are
    miscoded rather than absent.
    """
    lo, hi = ctx.time_slice.start, ctx.time_slice.stop
    et = ctx.event_table
    view = et[
        (et["event_midpoint_s"] >= lo) & (et["event_midpoint_s"] <= hi)
    ].sort_values("event_midpoint_s")

    # Canonical persisted tables store one primitive per exact complement.
    # Derive the display-only complements without changing missingness.
    view = view.assign(
        is_omitted_outcome=1.0 - view["is_rewarded_outcome"],
        did_last_completed_trip_stay_in_patch=(
            1.0 - view["did_last_completed_trip_switch_patch"]
        ),
    )

    flag_specs = (
        ("is_rewarded_outcome", "#2ca02c"),
        ("is_omitted_outcome", "#d62728"),
        ("did_last_completed_trip_switch_patch", "#9467bd"),
        ("did_last_completed_trip_stay_in_patch", "#8c564b"),
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
            0.5,
            0.5,
            "no events in window",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="gray",
        )
        return

    require_columns(view, flag_names, "event_table")
    n_events = len(view)
    starts = view["event_start_s"].to_numpy(dtype=float)
    ends = view["event_end_s"].to_numpy(dtype=float)
    min_width = max((hi - lo) * 0.003, 1e-6)

    for row_idx, (flag, color) in enumerate(flag_specs):
        vals = np.asarray(view[flag].to_numpy(), dtype=float)
        _validate_binary(vals, f"event_table['{flag}']")
        for j in range(n_events):
            if vals[j] == 1.0:
                style = {"facecolor": color, "edgecolor": "none", "hatch": None}
            elif not np.isfinite(vals[j]):
                # False stays blank (the panel's established encoding), so
                # undefined must not share that blank.
                style = {
                    "facecolor": _MISSING_CELL_COLOR,
                    "edgecolor": "#666666",
                    "hatch": "///",
                }
            else:
                continue
            ax.add_patch(
                Rectangle(
                    (starts[j], row_idx - 0.4),
                    width=max(ends[j] - starts[j], min_width),
                    height=0.8,
                    alpha=0.85,
                    zorder=2,
                    linewidth=0.4 if style["edgecolor"] != "none" else 0.0,
                    **style,
                )
            )


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

    One value per event at ``event_midpoint_s``: a thin vertical
    line from ``baseline`` to the value, plus a dot at the top. Works
    for any column of ``event_table`` whose dtype is numeric.
    """
    lo, hi = ctx.time_slice.start, ctx.time_slice.stop
    et = ctx.event_table
    in_window = (et["event_midpoint_s"] >= lo) & (et["event_midpoint_s"] <= hi)
    view = et.loc[in_window]

    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.axhline(baseline, color="gray", linewidth=0.3, zorder=0)
    ax.set_ylabel(ylabel)

    if len(view) == 0:
        ax.text(
            0.5,
            0.5,
            "no events in window",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="gray",
        )
        return

    times = view["event_midpoint_s"].to_numpy()
    values = view[value_col].to_numpy(dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        ax.text(
            0.5,
            0.5,
            f"{value_col}: all NaN in window",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="gray",
        )
        return
    ax.vlines(
        times[finite],
        baseline,
        values[finite],
        color="black",
        linewidth=0.5,
        zorder=2,
    )
    ax.scatter(
        times[finite],
        values[finite],
        s=12,
        c=marker_color,
        zorder=3,
        clip_on=False,
    )


def _plot_posterior_peak_mean(ax: Axes, ctx: _PanelContext) -> None:
    _draw_event_scalar_stem(
        ax,
        ctx,
        value_col="posterior_peak_mean",
        ylabel="posterior\npeak_mean",
        ylim=(1 / len(_ALL_ARMS) - 0.05, 1.05),
        baseline=1 / len(_ALL_ARMS),
    )


def _plot_posterior_entropy_mean(ax: Axes, ctx: _PanelContext) -> None:
    _draw_event_scalar_stem(
        ax,
        ctx,
        value_col="posterior_entropy_mean",
        ylabel="posterior entropy\n[nats]",
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
    in_window = (et["event_midpoint_s"] >= lo) & (et["event_midpoint_s"] <= hi)
    view = et.loc[in_window].sort_values("event_midpoint_s")

    ax.set_ylim(0.8, len(_ALL_ARMS) + 0.2)
    ax.set_ylabel("HPD width\n50/80/95")

    if len(view) == 0:
        ax.text(
            0.5,
            0.5,
            "no events in window",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="gray",
        )
        return

    times = view["event_midpoint_s"].to_numpy()
    specs = (
        ("posterior_hpd_arm_count_mean_coverage_50pct", 0.8, "#a6cee3", "50"),
        ("posterior_hpd_arm_count_mean_coverage_80pct", 1.3, "#1f78b4", "80"),
        ("posterior_hpd_arm_count_mean_coverage_95pct", 1.8, "#08306b", "95"),
    )
    for col, lw, color, label in specs:
        vals = view[col].to_numpy(dtype=float)
        ax.plot(
            times,
            vals,
            color=color,
            linewidth=lw,
            marker="o",
            markersize=3,
            zorder=3,
            clip_on=False,
            label=label,
        )
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=7,
        framealpha=0.7,
    )


def _plot_n_nonlocal_bins(ax: Axes, ctx: _PanelContext) -> None:
    _draw_event_scalar_stem(
        ax,
        ctx,
        value_col="n_nonlocal_bins",
        ylabel="n_nonlocal\nbins",
        ylim=None,
        baseline=0.0,
    )


def _plot_time_since_stop(ax: Axes, ctx: _PanelContext) -> None:
    _draw_event_scalar_stem(
        ax,
        ctx,
        value_col="time_since_stop_start_s",
        ylabel="time_since\nstop [s]",
        ylim=None,
        baseline=0.0,
    )
