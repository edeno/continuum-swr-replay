"""Patch-grain fitted-covariate panels (the P0-P4 ladder's design matrix)."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable

from continuum_swr_replay.tables.columns import require_columns

from ._common import (
    _MISSING_CELL_COLOR,
    _PROBABILITY_CMAP,
    _ZERO_CELL_COLOR,
    _add_interval_cells,
    _overlay_missing_references,
    _PanelContext,
    _validate_binary,
)

# ── Patch-grain panels (the P0–P4 ladder's design matrix) ──
#
# ``event_patch_table`` carries three rows per event but, unlike
# ``event_arm_table``, does NOT broadcast ``event_start_s`` / ``event_end_s``.
# Timing therefore has to come from ``event_table`` via a validated
# ``event_id`` join, which the helper below prepares once per figure.

_PATCHES: tuple[int, ...] = (1, 2, 3)


def _patch_view_with_times(ctx: _PanelContext, value_col: str) -> pd.DataFrame:
    """Return aligned, windowed patch rows with cached event intervals.

    Patch panels are diagnostics for a canonical three-row-per-event table,
    so a lossy inner join would be especially dangerous: orphan rows and
    missing events could disappear while leaving a plausible-looking plot.
    Validate the one-to-three alignment before joining and cache the complete
    joined table for all patch panels in the same figure.
    """
    if ctx.event_patch_table is None:
        raise ValueError(
            "patch-grain panels require ``event_patch_table``; pass it to "
            "plot_session_tables(event_patch_table=...)"
        )
    ept = ctx.event_patch_table
    require_columns(ept, [value_col], "event_patch_table")
    if ctx._event_patch_with_times is None:
        require_columns(ept, ("event_id", "patch"), "event_patch_table")
        require_columns(
            ctx.event_table,
            ("event_id", "event_start_s", "event_end_s"),
            "event_table",
        )
        if ept[["event_id", "patch"]].isna().any().any():
            raise ValueError("event_patch_table event_id and patch must be non-missing")
        if ctx.event_table["event_id"].isna().any():
            raise ValueError("event_table.event_id must be non-missing")
        if ctx.event_table["event_id"].duplicated().any():
            raise ValueError("event_table.event_id must be unique for patch alignment")
        if ept.duplicated(["event_id", "patch"]).any():
            raise ValueError("event_patch_table has duplicate (event_id, patch) rows")

        patch_values = set(ept["patch"].tolist())
        if not patch_values.issubset(set(_PATCHES)):
            raise ValueError(
                "event_patch_table.patch must contain only canonical patches "
                f"{list(_PATCHES)}; found {sorted(patch_values)}"
            )
        bad_events = [
            event_id
            for event_id, rows in ept.groupby("event_id", sort=False, dropna=False)
            if len(rows) != len(_PATCHES) or set(rows["patch"]) != set(_PATCHES)
        ]
        if bad_events:
            raise ValueError(
                "event_patch_table must contain exactly patches 1, 2, and 3 "
                f"for every event; invalid event_id(s): {bad_events[:10]}"
            )

        event_ids = set(ctx.event_table["event_id"].tolist())
        patch_event_ids = set(ept["event_id"].tolist())
        orphan = sorted(patch_event_ids - event_ids)
        missing = sorted(event_ids - patch_event_ids)
        if orphan or missing:
            raise ValueError(
                "event_patch_table/event_table event_id alignment failed: "
                f"orphan patch event(s)={orphan[:10]}, missing patch event(s)={missing[:10]}"
            )

        context_columns = [
            column
            for column in ("current_patch", "prev_patch", "last_rewarded_arm")
            if column in ctx.event_table.columns
        ]
        times = ctx.event_table[
            ["event_id", "event_start_s", "event_end_s", *context_columns]
        ]
        # Event-level context is authoritative here; avoid suffixing if an
        # external patch table happens to carry stale broadcast copies.
        patch_rows = ept.drop(
            columns=["event_start_s", "event_end_s", *context_columns],
            errors="ignore",
        )
        ctx._event_patch_with_times = patch_rows.merge(
            times,
            on="event_id",
            how="left",
            validate="many_to_one",
        )

    merged = ctx._event_patch_with_times
    require_columns(merged, [value_col], "event_patch_table")
    lo, hi = ctx.time_slice.start, ctx.time_slice.stop
    return merged[(merged["event_start_s"] <= hi) & (merged["event_end_s"] >= lo)]


def _draw_event_patch_heatmap(
    ax: Axes,
    ctx: _PanelContext,
    *,
    value_col: str,
    cmap_name: str,
    vmin: float | None,
    vmax: float | None,
    label: str,
    binary_color: str | None = None,
    reference_cols: str | Sequence[str] | None = None,
) -> None:
    """Shared: per-(event, patch) value as three stacked colored rows.

    Mirrors :func:`_draw_event_arm_heatmap` at patch grain. When
    ``binary_color`` is given the value is treated as a 0/1 indicator and
    drawn as presence/absence instead of through a colormap. Explicit zero
    gets a pale cell; missing values get a darker hatched cell.
    """
    view = _patch_view_with_times(ctx, value_col)

    for patch in _PATCHES:
        ax.axhline(patch, color="lightgray", linewidth=0.3, zorder=0)
    ax.set_ylabel(label)
    ax.set_ylim(0.5, 3.5)
    ax.set_yticks(list(_PATCHES))

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

    lo, hi = ctx.time_slice.start, ctx.time_slice.stop
    values = view[value_col].to_numpy(dtype=float)
    if binary_color is not None:
        values = _overlay_missing_references(values, view, reference_cols)
    patches = view["patch"].to_numpy()
    starts = view["event_start_s"].to_numpy()
    ends = view["event_end_s"].to_numpy()

    if binary_color is None:
        finite = np.isfinite(values)
        if vmin is None:
            vmin = float(values[finite].min()) if finite.any() else 0.0
        if vmax is None:
            vmax = float(values[finite].max()) if finite.any() else 1.0
        if vmax <= vmin:
            vmax = vmin + 1.0
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap(cmap_name)

        colors = [cmap(norm(value)) for value in values[finite]]
        _add_interval_cells(
            ax,
            starts[finite],
            ends[finite],
            patches[finite],
            colors,
            min_width=max((hi - lo) * 0.002, 1e-6),
        )
    else:
        _validate_binary(values, f"event_patch_table.{value_col}")
        finite = np.isfinite(values)
        present = finite & (values == 1.0)
        absent = finite & (values == 0.0)
        _add_interval_cells(
            ax,
            starts[present],
            ends[present],
            patches[present],
            [binary_color] * int(present.sum()),
            min_width=max((hi - lo) * 0.002, 1e-6),
            edgecolors="#bdbdbd",
            alpha=0.9,
        )
        # Keep explicit zero visible without conflating it with NaN. Batching
        # the pale background cells also keeps binary patch panels lightweight.
        _add_interval_cells(
            ax,
            starts[absent],
            ends[absent],
            patches[absent],
            [_ZERO_CELL_COLOR] * int(absent.sum()),
            min_width=max((hi - lo) * 0.002, 1e-6),
            edgecolors="#bdbdbd",
            alpha=0.9,
        )
    missing = ~np.isfinite(values)
    _add_interval_cells(
        ax,
        starts[missing],
        ends[missing],
        patches[missing],
        [_MISSING_CELL_COLOR] * int(missing.sum()),
        min_width=max((hi - lo) * 0.002, 1e-6),
        edgecolors="#666666",
        hatch="///",
    )

    if binary_color is None:
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        ax.figure.colorbar(sm, ax=ax, shrink=0.7, pad=0.01, aspect=10)


def _plot_reward_belief_mean_candidate_arm_preoutcome_patch(
    ax: Axes, ctx: _PanelContext
) -> None:
    """Event-sampled behavioral-model value aggregated to patch."""
    _draw_event_patch_heatmap(
        ax,
        ctx,
        value_col="reward_belief_mean_candidate_arm_preoutcome_patch",
        cmap_name=_PROBABILITY_CMAP,
        vmin=0.0,
        vmax=1.0,
        label="mean leaf q\nper patch",
    )


def _plot_task_reward_probability_candidate_arm_patch(
    ax: Axes, ctx: _PanelContext
) -> None:
    """Event-sampled programmed reward probability aggregated to patch."""
    _draw_event_patch_heatmap(
        ax,
        ctx,
        value_col="task_reward_probability_candidate_arm_patch",
        cmap_name=_PROBABILITY_CMAP,
        vmin=0.0,
        vmax=1.0,
        label="mean leaf p(R)\nper patch",
    )


def _plot_reward_fraction_candidate_arm_last10_visits_patch(
    ax: Axes, ctx: _PanelContext
) -> None:
    """Mean recent reward rate across the patch's two reward-site arms."""
    _draw_event_patch_heatmap(
        ax,
        ctx,
        value_col="reward_fraction_candidate_arm_last10_visits_patch",
        cmap_name=_PROBABILITY_CMAP,
        vmin=0.0,
        vmax=1.0,
        label="mean leaf reward fraction\nprior ≤10 arm visits",
    )


def _plot_reward_count_candidate_arm_last10_known_trials_patch(
    ax: Axes, ctx: _PanelContext
) -> None:
    """Rewards on this patch over the last ten completed trials.

    A count of recent rewarded exposure, not a rate: it is the sum over
    the patch's arms of a last-ten-*trial* reward count, so it shares no
    denominator with the visit-window reward-rate terms.
    """
    _draw_event_patch_heatmap(
        ax,
        ctx,
        value_col="reward_count_candidate_arm_last10_known_trials_patch",
        cmap_name="viridis",
        vmin=0,
        vmax=None,
        label="patch reward count\nprior ≤10 trials",
    )


def _plot_is_current_patch(ax: Axes, ctx: _PanelContext) -> None:
    _draw_event_patch_heatmap(
        ax,
        ctx,
        value_col="is_current_patch",
        cmap_name="",
        vmin=None,
        vmax=None,
        label="is_current\npatch",
        binary_color="#1f77b4",
        reference_cols="current_patch",
    )


def _plot_is_last_rewarded_noncurrent_patch(ax: Axes, ctx: _PanelContext) -> None:
    _draw_event_patch_heatmap(
        ax,
        ctx,
        value_col="is_last_rewarded_noncurrent_patch",
        cmap_name="",
        vmin=None,
        vmax=None,
        label="is_last_rew.\npatch elsew.",
        binary_color="#2ca02c",
        reference_cols=("last_rewarded_arm", "current_patch"),
    )


def _plot_is_prev_patch(ax: Axes, ctx: _PanelContext) -> None:
    """Previous patch indicator (used by the patch-switch ladder)."""
    _draw_event_patch_heatmap(
        ax,
        ctx,
        value_col="is_prev_patch",
        cmap_name="",
        vmin=None,
        vmax=None,
        label="is_prev\npatch",
        binary_color="#ff7f0e",
        reference_cols="prev_patch",
    )
