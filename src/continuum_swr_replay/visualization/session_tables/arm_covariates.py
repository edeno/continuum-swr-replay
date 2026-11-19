"""Arm-grain fitted-covariate panels (per-arm predictors, behavioral-model beliefs, role indicators)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from matplotlib.axes import Axes

from continuum_swr_replay.data_loaders.constants import ALL_ARMS as _ALL_ARMS
from continuum_swr_replay.tables.columns import require_columns

from ._common import (
    _ANNOTATION_FONTSIZE,
    _MISSING_CELL_COLOR,
    _PROBABILITY_CMAP,
    _TRUNK_ARMS,
    _ZERO_CELL_COLOR,
    _add_interval_cells,
    _draw_event_arm_heatmap,
    _format_arm_axis,
    _overlay_missing_references,
    _PanelContext,
    _validate_binary,
)


def _draw_event_arm_indicator(
    ax: Axes,
    ctx: _PanelContext,
    *,
    value_col: str,
    color: str,
    label: str,
    reference_cols: str | Sequence[str] | None = None,
) -> None:
    """Shared: binary per-(event, arm) predictor as filled/empty cells.

    The 0/1 spatial-role predictors (``is_current``, ``is_prev``,
    ``is_last_rewarded_elsewhere``) read far better as presence/absence
    than through a continuous colormap, where "0" would still take a
    saturated color at the low end of the scale. Explicit zero is pale;
    ``NaN`` is dark-gray and hatched. Missing source references are given the
    same semantic overlay even when the canonical fitted design deliberately
    stores zeros; plotting never changes those fitted rows.
    """
    lo, hi = ctx.time_slice.start, ctx.time_slice.stop
    eat = ctx.event_arm_table
    require_columns(
        eat,
        ("event_start_s", "event_end_s", "candidate_arm", value_col),
        "event_arm_table",
    )
    view = eat[(eat["event_start_s"] <= hi) & (eat["event_end_s"] >= lo)]

    _format_arm_axis(ax, label)

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

    values = view[value_col].to_numpy(dtype=float)
    values = _overlay_missing_references(values, view, reference_cols)
    _validate_binary(values, f"event_arm_table.{value_col}")
    finite = np.isfinite(values)
    arms = view["candidate_arm"].to_numpy(dtype=float)
    starts = view["event_start_s"].to_numpy(dtype=float)
    ends = view["event_end_s"].to_numpy(dtype=float)
    colors = [color if value == 1.0 else _ZERO_CELL_COLOR for value in values[finite]]
    _add_interval_cells(
        ax,
        starts[finite],
        ends[finite],
        arms[finite],
        colors,
        min_width=max((hi - lo) * 0.002, 1e-6),
        edgecolors="#bdbdbd",
        alpha=0.9,
    )
    missing = ~finite
    _add_interval_cells(
        ax,
        starts[missing],
        ends[missing],
        arms[missing],
        [_MISSING_CELL_COLOR] * int(missing.sum()),
        min_width=max((hi - lo) * 0.002, 1e-6),
        edgecolors="#666666",
        hatch="///",
    )


# ── Per-arm predictor panels (trials → per-arm event-arm row) ──


def _plot_visit_recency_trials(ax: Axes, ctx: _PanelContext) -> None:
    """Trials since the last destination visit; trunks are not destinations."""
    _draw_event_arm_heatmap(
        ax,
        ctx,
        value_col="visit_recency_trials",
        cmap_name="viridis_r",
        vmin=1,
        vmax=None,
        label="visit recency\n[trials ago]",
        structural_na_rows=_TRUNK_ARMS,
    )
    ax.text(
        0.005,
        0.02,
        "never visited = completed trials + 1\n(explicit novelty row)",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=_ANNOTATION_FONTSIZE,
        color="0.35",
    )


def _plot_cumulative_reward(ax: Axes, ctx: _PanelContext) -> None:
    """Per-arm reward count among up to ten prior completed trials."""
    _draw_event_arm_heatmap(
        ax,
        ctx,
        value_col="reward_count_candidate_arm_last10_known_trials",
        cmap_name="viridis",
        vmin=0,
        vmax=None,
        label="reward count\nprior ≤10 trials",
        structural_na_rows=_TRUNK_ARMS,
    )


def _plot_recent_reward_rate(ax: Axes, ctx: _PanelContext) -> None:
    """Reward fraction over up to ten visits to each leaf arm.

    A never-visited leaf is intentionally encoded as zero by the fitted design.
    Trunk rows are held out of the quantitative colormap because they are not
    reward sites — but they are *not* held out of the fit: production writes
    0.0 there so the conditional-Poisson fitter does not drop the whole
    nine-arm event. The masking is a reading aid, not an exclusion.
    """
    _draw_event_arm_heatmap(
        ax,
        ctx,
        value_col="reward_fraction_candidate_arm_last10_visits",
        cmap_name=_PROBABILITY_CMAP,
        vmin=0.0,
        vmax=1.0,
        label="reward fraction\nprior ≤10 arm visits",
        structural_na_rows=_TRUNK_ARMS,
        colorbar_label="reward fraction",
        colorbar_ticks=(0.0, 0.5, 1.0),
    )
    ax.text(
        0.005,
        0.02,
        "leaf value 0 may encode never visited",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=_ANNOTATION_FONTSIZE,
        color="0.35",
    )


def _plot_task_reward_probability_candidate_arm(ax: Axes, ctx: _PanelContext) -> None:
    """Programmed (nominal) reward probability per arm.

    Sampled once for each event from the active contingency block (per-arm
    ``task_reward_probability_port_*``); equals
    ``task_reward_probability_chosen_arm`` at the occupied arm.
    Trunk arms are not reward sites, so they are masked here rather than
    colored — but the model-ready table encodes their placeholder as 0.0 and
    fits it; the mask is a reading aid, not an exclusion.
    """
    _draw_event_arm_heatmap(
        ax,
        ctx,
        value_col="task_reward_probability_candidate_arm",
        cmap_name=_PROBABILITY_CMAP,
        vmin=0.0,
        vmax=1.0,
        label="programmed p(R)\nassigned trial",
        structural_na_rows=_TRUNK_ARMS,
        colorbar_group="behavioral_probability",
        colorbar_label="probability",
        colorbar_ticks=(0.0, 0.5, 1.0),
    )


def _plot_distance_to_candidate_arm(ax: Axes, ctx: _PanelContext) -> None:
    _draw_event_arm_heatmap(
        ax,
        ctx,
        value_col="distance_to_candidate_arm_cm",
        cmap_name="viridis_r",
        vmin=0,
        vmax=None,
        label="arm distance\n[graph units]",
    )


def _plot_arm_dist_rank(ax: Axes, ctx: _PanelContext) -> None:
    _draw_event_arm_heatmap(
        ax,
        ctx,
        value_col="candidate_arm_distance_rank",
        cmap_name="viridis_r",
        vmin=1,
        vmax=len(_ALL_ARMS),
        label="arm\ndist rank",
    )


def _make_recency_spline_panel(column: str, index: int):
    """Panel for one recency-spline basis column.

    The basis is zero outside its identified support (visited leaves that are
    neither the current nor the previous arm), so those rows read as structural
    zeros rather than missing data -- the same convention the trunk rows use in
    the linear-recency panel above.
    """

    def _plot(ax: Axes, ctx: _PanelContext) -> None:
        _draw_event_arm_heatmap(
            ax,
            ctx,
            value_col=column,
            cmap_name="coolwarm",
            vmin=None,
            vmax=None,
            center=0.0,
            label=f"recency spline\nbasis {index} [centered]",
            structural_na_rows=_TRUNK_ARMS,
            scale_across_table=True,
            robust_symmetric_quantile=0.95,
            colorbar_label="basis value from event mean",
        )

    return _plot


def _plot_visit_recency_trials_event_leaf_centered(
    ax: Axes, ctx: _PanelContext
) -> None:
    """Event-local centered visit recency — the column M4+ actually fits.

    At each event, the mean over visited leaf arms is subtracted from leaf-arm
    recency. Never-visited leaves are excluded from that mean and represented
    separately by ``arm_never_visited``; trunks remain structural zero. The
    diverging map therefore shows recency relative to the event's visited-leaf
    mean, not relative to a future-dependent session mean.
    """
    _draw_event_arm_heatmap(
        ax,
        ctx,
        value_col="visit_recency_trials_event_leaf_centered",
        cmap_name="coolwarm",
        vmin=None,
        vmax=None,
        center=0.0,
        label="visit recency\ncentered [trials]",
        structural_na_rows=_TRUNK_ARMS,
        scale_across_table=True,
        robust_symmetric_quantile=0.95,
        colorbar_label="trials from event visited-leaf mean",
    )
    ax.text(
        0.005,
        0.02,
        "session 95% color range; saturation = clipped\n"
        "time-local censored values included before centering",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=_ANNOTATION_FONTSIZE,
        color="0.35",
    )


# ── Behavioral-model (beta-bernoulli) per-arm belief panels ──
#
# The candidate-arm reward-belief mean and variance come from the SAME
# per-leaf Beta belief state, so a value+uncertainty read stays
# construct-coherent. Both are distinct from the experimenter's programmed
# task reward probability; adjacent panels expose that nominal-vs-believed
# contrast.


def _plot_reward_belief_mean_candidate_arm(ax: Axes, ctx: _PanelContext) -> None:
    """Behavioral-model value per arm (Beta posterior mean).

    Bounded [0, 1] like ``task_reward_probability_candidate_arm`` and drawn on the same colormap
    and scale so the two panels are directly comparable by eye. The value
    is sampled once for each event; the colored event-width cell does not
    imply within-event updating. Trunk arms are not reward sites, so they are
    masked here rather than colored — but the model-ready table encodes their
    placeholder as 0.0 and fits it; the mask is a reading aid, not an
    exclusion.
    """
    _draw_event_arm_heatmap(
        ax,
        ctx,
        value_col="reward_belief_mean_candidate_arm_preoutcome",
        cmap_name=_PROBABILITY_CMAP,
        vmin=0.0,
        vmax=1.0,
        label="Beta belief q\npre-outcome",
        structural_na_rows=_TRUNK_ARMS,
        colorbar_group="behavioral_probability",
        colorbar_label="probability",
        colorbar_ticks=(0.0, 0.5, 1.0),
    )


def _plot_reward_belief_variance_candidate_arm(ax: Axes, ctx: _PanelContext) -> None:
    """Behavioral-model belief uncertainty per arm (Beta posterior variance).

    The variance counterpart of
    :func:`_plot_reward_belief_mean_candidate_arm` from the same belief
    state. In this fitted model, alpha and beta remain at least 1 and
    forgetting pulls toward Beta(1, 1), so posterior variance is bounded by
    1/12. The scale is pinned there rather than inferred -- an inferred ceiling
    would make a low-uncertainty window look falsely uncertain. Values are
    sampled once per event. Trunk arms have no reward belief, so they are
    masked here rather than drawn as certain zeros — but the model-ready table
    does encode 0.0 there and fits it; the mask is a reading aid, not an
    exclusion.
    """
    _draw_event_arm_heatmap(
        ax,
        ctx,
        value_col="reward_belief_variance_candidate_arm_preoutcome",
        cmap_name="magma",
        vmin=0.0,
        vmax=1.0 / 12.0,
        label="Beta variance\npre-outcome [p²]",
        structural_na_rows=_TRUNK_ARMS,
        colorbar_label="variance [p²]",
        colorbar_ticks=(0.0, 1.0 / 24.0, 1.0 / 12.0),
    )


# ── Binary per-arm role indicators (fitted spatial predictors) ──


def _plot_is_current(ax: Axes, ctx: _PanelContext) -> None:
    _draw_event_arm_indicator(
        ax,
        ctx,
        value_col="is_current_arm",
        color="#1f77b4",
        label="is_current_arm",
        reference_cols="occupancy_well_arm",
    )


def _plot_is_prev(ax: Axes, ctx: _PanelContext) -> None:
    _draw_event_arm_indicator(
        ax,
        ctx,
        value_col="is_prev_arm",
        color="#ff7f0e",
        label="is_prev_arm",
        reference_cols="prev_arm",
    )


def _plot_is_last_rewarded_elsewhere(ax: Axes, ctx: _PanelContext) -> None:
    _draw_event_arm_indicator(
        ax,
        ctx,
        value_col="is_last_rewarded_noncurrent_arm",
        color="#2ca02c",
        label="is_last_rew.\nelsewhere",
        reference_cols=("last_rewarded_arm", "occupancy_well_arm"),
    )


def _plot_arm_never_visited(ax: Axes, ctx: _PanelContext) -> None:
    """Explicit novel-leaf indicator paired with censored visit recency."""
    _draw_event_arm_indicator(
        ax,
        ctx,
        value_col="is_candidate_arm_never_visited",
        color="#7b3294",
        label="arm never\nvisited",
    )
