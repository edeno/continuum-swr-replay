"""Session-tables inspection figure.

``plot_session_tables`` stacks a time-aligned multi-panel view of one
session so the user can trace the transformation from raw decoder
output through the four canonical tables. It is the canonical-tables
analog of :func:`plot_detector`: same layout conventions, a superset
of the core panels, plus overlays and optional rows that expose each
step of the derivation chain.

Panels fall into four groups, dispatched by name through the ``panels``
argument. :data:`DEFAULT_PANELS` is a deliberately small
``plot_detector``-style overview; **every covariate panel is opt-in**, so
a bare call shows none of them:

* **Responses and decoder state** — ``arm_count_hard`` / ``arm_count_soft``
  / ``arm_fraction`` / ``arm_posterior_mass``, posterior heatmaps, and the
  derivation-debug rows (``arm_markers``, ``mismatch_pointers``).
* **Arm-grain fitted covariates** (:data:`ARM_COVARIATE_PANELS`) — the
  arm-grain predictors, spanning the M ladder plus terms fit only by the
  content-contrast (``arm_dist_rank``) and choice (``cumulative_reward``)
  models. It is *not* a faithful M-ladder design matrix in either
  direction: those two are in no M rung, and the ``arm_fe_*`` dummies
  carried by every rung from M0a have no panel (they are arm identity,
  which is already the y-axis). :data:`BEHAVIORAL_MODEL_PANELS` is the
  narrower belief group — ``q_arm`` and ``arm_uncertainty`` from the
  beta-bernoulli behavioral model, alongside the programmed
  ``p_reward_arm`` they are read against.
* **Patch-grain fitted covariates** (:data:`PATCH_COVARIATE_PANELS`) — the
  P-ladder predictors, plus ``is_prev_patch`` (patch-switch ladder only)
  and the ``*_patch`` belief pair (``Pv`` only); ``visit_recency_patch``
  has no panel. These require ``event_patch_table``.
* **Stop-grain fitted covariates** (:data:`RATE_COVARIATE_PANELS`) — the
  Family-A rate design matrix, keyed on stops rather than events.

Note that ``p_reward_arm`` (and ``p_reward_arm_patch``) is the
experimenter's *programmed* reward probability, while ``q_arm`` (and
``q_arm_patch``) is the *fitted behavioral model's* belief. They are
different quantities on the same scale; plotting them adjacent is the
point.

Three families of fitted predictors are deliberately **not** given their
own panels, because a panel would carry no information the reader cannot
already read off an existing one:

* ``swr_x_*`` interactions are the elementwise product of ``is_swr`` with
  a base covariate. Read them as the base panel masked by the ``is_swr``
  row of ``context_flags``.
* ``*_z`` columns (e.g. ``arm_uncertainty_z``) are affine rescalings of
  the raw column that is plotted; the spatial/temporal pattern is
  identical and only the color-bar units would differ.
* ``replay_frac`` / ``replay_frac_swr`` / ``replay_frac_non_swr`` are
  choice-grain columns built by ``build_choice_dataset``, not by the
  canonical table builders, so they are not reachable from this
  function's inputs. ``replay_frac_by_role`` covers the same quantity
  summarized by arm role.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
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
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FixedFormatter, FixedLocator, MaxNLocator

from continuum_swr_replay.analysis.replay import (
    single_arm_from_well,
    trial_context_at_times,
    trial_cutoff_times,
)
from continuum_swr_replay.analysis.replay.model_specs import (
    N_REWARD_LAGS,
    RECENCY_SPLINE_COLUMNS,
)
from continuum_swr_replay.data_loaders.constants import (
    ALL_ARMS as _ALL_ARMS,
)
from continuum_swr_replay.data_loaders.constants import (
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

_LEAF_ARMS: frozenset[int] = frozenset(range(6))
_TRUNK_ARMS: frozenset[int] = frozenset(set(_ALL_ARMS) - _LEAF_ARMS)
_COLLECTION_THRESHOLD = 128
_ZERO_CELL_COLOR = "#f2f2f2"
_MISSING_CELL_COLOR = "#969696"
_STRUCTURAL_NA_COLOR = "#e3e3e3"
_PROBABILITY_CMAP = "cividis"
_ANNOTATION_FONTSIZE = 7
_REWARDED_COLOR = "#0072B2"

_MISMATCH_COLUMNS: tuple[str, ...] = (
    "event_id",
    "event_start",
    "event_midpoint_time",
    "canonical_next_arm",
    "actual_next_to_well",
    "actual_next_arm",
    "match",
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
class _ColorbarRequest:
    """One deferred colorbar shared by one or more comparable panels."""

    mappable: ScalarMappable
    axes: list[Axes]
    label: str | None = None
    ticks: tuple[float, ...] | None = None
    extend: str = "neither"


@dataclass
class _PanelContext:
    """Context object passed to every panel helper."""

    data: Mapping[str, Any]
    detector: Any
    results: xr.Dataset | xr.DataArray
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
    # Optional because the patch ladder is a separate model family: every
    # pre-existing panel and caller works without it. Defaulted last so
    # adding it does not break positional construction of this dataclass.
    event_patch_table: pd.DataFrame | None = None
    # Prepared lazily because most figures do not request a patch panel. A
    # full patch-group figure otherwise repeats the same validated timing join
    # six times.
    _event_patch_with_times: pd.DataFrame | None = field(
        default=None, init=False, repr=False
    )
    _mismatch_table: pd.DataFrame | None = field(default=None, init=False, repr=False)
    defer_colorbars: bool = False
    _colorbar_requests: dict[str, _ColorbarRequest] = field(
        default_factory=dict, init=False, repr=False
    )


def _require_columns(
    frame: pd.DataFrame,
    frame_name: str,
    columns: Sequence[str],
) -> None:
    """Raise an actionable error when a plotting input lacks columns."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{frame_name} has no column(s) {missing}")


def _validate_binary(values: np.ndarray, value_name: str) -> None:
    """Fail rather than silently drawing corrupt finite indicators as true."""
    finite = values[np.isfinite(values)]
    invalid = finite[~np.isin(finite, (0.0, 1.0))]
    if invalid.size:
        raise ValueError(
            f"{value_name} must contain only 0, 1, or NaN; "
            f"found {np.unique(invalid).tolist()}"
        )


def _add_colorbar(
    ax: Axes,
    ctx: _PanelContext,
    mappable: ScalarMappable,
    *,
    group: str | None,
    label: str | None,
    ticks: tuple[float, ...] | None,
    extend: str = "neither",
) -> None:
    """Draw now or register a shared colorbar for the public figure."""
    if not ctx.defer_colorbars:
        ax.figure.colorbar(
            mappable,
            ax=ax,
            shrink=0.7,
            pad=0.01,
            aspect=10,
            label=label,
            ticks=ticks,
            extend=extend,
        )
        return

    key = group if group is not None else f"axis:{id(ax)}"
    request = ctx._colorbar_requests.get(key)
    if request is None:
        ctx._colorbar_requests[key] = _ColorbarRequest(
            mappable=mappable,
            axes=[ax],
            label=label,
            ticks=ticks,
            extend=extend,
        )
        return

    old_norm = request.mappable.norm
    new_norm = mappable.norm
    compatible = (
        request.mappable.cmap.name == mappable.cmap.name
        and np.isclose(old_norm.vmin, new_norm.vmin)
        and np.isclose(old_norm.vmax, new_norm.vmax)
        and request.ticks == ticks
        and request.label == label
        and request.extend == extend
    )
    if not compatible:
        raise ValueError(f"colorbar group {key!r} mixes incompatible scales")
    request.axes.append(ax)


def _format_arm_axis(ax: Axes, label: str) -> None:
    """Apply the shared nine-arm geometry and leaf/trunk distinction."""
    for arm in _ALL_ARMS:
        ax.axhline(arm, color="lightgrey", linewidth=0.3, zorder=0)
    ax.axhline(
        5.5,
        color="0.5",
        linewidth=0.65,
        linestyle="--",
        zorder=0.5,
    )
    ax.set_ylabel(label)
    ax.set_ylim(-0.5, 8.5)
    ax.set_yticks(list(_ALL_ARMS))
    for tick, arm in zip(ax.get_yticklabels(), _ALL_ARMS, strict=True):
        if arm in _TRUNK_ARMS:
            tick.set_color("0.45")


def _overlay_missing_references(
    values: np.ndarray,
    view: pd.DataFrame,
    reference_cols: str | Sequence[str] | None,
) -> np.ndarray:
    """Mark plotted indicators missing when an available source is missing.

    Canonical design builders encode comparisons to missing arm/patch
    references as all-zero indicator rows. That representation remains valid
    and unchanged for fitting; the inspection view overlays the causal source
    missingness so an unknown reference is not mistaken for nine (or three)
    observed negatives. Optional metadata may be absent from small external
    fixtures, in which case the stored indicator values remain authoritative.
    """
    if reference_cols is None:
        return values
    requested = (reference_cols,) if isinstance(reference_cols, str) else reference_cols
    available = [column for column in requested if column in view.columns]
    if not available:
        return values
    reference_missing = view[available].isna().any(axis=1).to_numpy()
    if not reference_missing.any():
        return values
    plotted = values.copy()
    plotted[reference_missing] = np.nan
    return plotted


def _add_interval_cells(
    ax: Axes,
    starts: np.ndarray,
    ends: np.ndarray,
    rows: np.ndarray,
    facecolors: Sequence[Any],
    *,
    min_width: float,
    edgecolors: str | Sequence[Any] = "none",
    alpha: float = 1.0,
    hatch: str | None = None,
) -> None:
    """Draw interval cells, batching large panels into one collection.

    Small views retain individual :class:`~matplotlib.patches.Rectangle`
    artists, which keeps notebook inspection and focused tests convenient.
    Full-session views use one rasterized :class:`PolyCollection`; this avoids
    thousands of Python artist objects per covariate panel without changing
    cell geometry or colors.
    """
    starts = np.asarray(starts, dtype=float)
    ends = np.asarray(ends, dtype=float)
    rows = np.asarray(rows, dtype=float)
    widths = np.maximum(ends - starts, float(min_width))
    colors = list(facecolors)
    if not (len(starts) == len(rows) == len(colors)):
        raise ValueError("interval-cell arrays must have equal lengths")
    if len(starts) == 0:
        return

    if len(starts) <= _COLLECTION_THRESHOLD:
        if isinstance(edgecolors, str):
            edges = [edgecolors] * len(starts)
        else:
            edges = list(edgecolors)
        for start, width, row, facecolor, edgecolor in zip(
            starts, widths, rows, colors, edges, strict=True
        ):
            ax.add_patch(
                Rectangle(
                    (start, row - 0.5),
                    width=width,
                    height=1.0,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    linewidth=0.4 if edgecolor != "none" else 0.0,
                    alpha=alpha,
                    hatch=hatch,
                    zorder=1,
                )
            )
        return

    vertices = [
        (
            (start, row - 0.5),
            (start + width, row - 0.5),
            (start + width, row + 0.5),
            (start, row + 0.5),
        )
        for start, width, row in zip(starts, widths, rows, strict=True)
    ]
    no_edges = isinstance(edgecolors, str) and edgecolors == "none"
    collection = PolyCollection(
        vertices,
        facecolors=colors,
        edgecolors=edgecolors,
        linewidths=0.0 if no_edges else 0.4,
        alpha=alpha,
        hatch=hatch,
        zorder=1,
        rasterized=True,
    )
    ax.add_collection(collection)


# ── Core panels (thin adapters over existing plot_detector helpers) ──


def _plot_swr_lfp(ax: Axes, ctx: _PanelContext) -> None:
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
    from continuum_swr_replay.analysis.constants import (
        DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    )

    ax.axhline(
        DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
        color="grey",
        linestyle="--",
        linewidth=0.8,
        zorder=5,
    )


# ── New: trial rail ──


def _plot_trial_rail(ax: Axes, ctx: _PanelContext) -> None:
    """Production-aligned behavioral location and completed-trip context.

    Dark horizontal segments show dwell intervals at a well. Light shading
    marks ``[start_time, goal_arrival_time)`` transit intervals, during which
    destination and outcome are not exposed as current context. Reward stars
    and purple patch-switch ticks occur at the completion cutoff (goal arrival,
    with the established per-row ``end_time`` fallback).
    """
    trials = ctx.trials
    lo, hi = ctx.time_slice.start, ctx.time_slice.stop
    _require_columns(
        trials,
        "trials",
        (
            "start_time",
            "end_time",
            "from_well",
            "to_well",
            "is_reward",
            "is_patch_change",
        ),
    )
    if len(trials) == 0:
        ax.text(
            0.5,
            0.5,
            "no trials in window",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="grey",
        )
        ax.set_ylabel("behavioral\nlocation")
        ax.set_ylim(-0.5, 5.5)
        return

    ordered = trials.sort_values("start_time", kind="stable").reset_index(drop=True)
    start_times = ordered["start_time"].to_numpy(dtype=float)
    if not np.isfinite(start_times).all() or np.any(np.diff(start_times) < 0):
        raise ValueError(
            "trials.start_time must be finite and chronologically sortable"
        )

    # Validate using the same helper as the canonical table builders.
    trial_context_at_times(np.array([lo]), ordered)
    completion_times = trial_cutoff_times(ordered)
    if not np.isfinite(completion_times).all():
        raise ValueError("canonical trial cutoff times must be finite")

    drew_interval = False
    for trial_idx, row in ordered.iterrows():
        transit_start = max(float(row["start_time"]), lo)
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
            color="grey",
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
        ax.axhline(well, color="lightgrey", linewidth=0.3, zorder=0)

    ax.set_ylabel("behavioral\nlocation")
    ax.set_ylim(-0.5, 5.5)
    ax.set_yticks(sorted(WELL_PATCH_MAPPING))


# ── Future-covariate debug: arm markers + mismatch pointers ──


def _compute_mismatch_table(
    event_table: pd.DataFrame,
    trials: pd.DataFrame,
    track_graph: Any,
) -> pd.DataFrame:
    """Compare canonical ``next_arm`` against trials' actual next trip.

    Canonical trial context is sampled from the event's first retained
    timebin, so the audit is evaluated at ``event_start`` (not the display
    midpoint). The recomputation delegates to the production
    :func:`trial_context_at_times` and :func:`single_arm_from_well` helpers;
    boundary, stable-sort, pre-first-trial, and session-tail semantics therefore
    cannot drift into a second visualization-only implementation.

    ``event_midpoint_time`` is retained solely as the display coordinate
    used by :func:`_plot_mismatch_pointers`.

    ``match`` is ``True`` when the canonical and actual arms are equal
    OR when both are ``NaN`` (legitimate session end: the last event
    has no next trip). Any other combination is a mismatch and will be
    drawn in red by :func:`_plot_mismatch_pointers`.
    """
    _require_columns(
        event_table,
        "event_table",
        ("event_id", "event_start", "event_midpoint_time", "next_arm"),
    )
    if len(event_table) == 0:
        return pd.DataFrame(columns=_MISMATCH_COLUMNS)
    _require_columns(
        trials,
        "trials",
        (
            "start_time",
            "end_time",
            "from_well",
            "to_well",
            "is_reward",
            "is_patch_change",
        ),
    )

    event_starts = event_table["event_start"].to_numpy(dtype=float)
    event_midpoints = event_table["event_midpoint_time"].to_numpy(dtype=float)
    actual_to_well = trial_context_at_times(event_starts, trials)["next_well"].to_numpy(
        dtype=float
    )
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
            "event_start": event_starts,
            "event_midpoint_time": event_midpoints,
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
            0.5,
            0.5,
            "no events in window",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="grey",
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

    # Legend for the four marker types.
    from matplotlib.lines import Line2D as _L2D

    ax.legend(
        handles=[
            _L2D(
                [],
                [],
                marker="o",
                color="grey",
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
                color="grey",
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
                color="grey",
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
                color="grey",
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
    ``event_midpoint_time`` for alignment with the other event panels; that
    midpoint is not used to select the next trial.
    A vertical connector between them — **red when the two disagree**,
    thin grey when they match.

    ``(NaN, NaN)`` — the legitimate session-end case where the last
    event has no next trip — is treated as a match.

    The underlying mismatch table is computed once per ``ctx`` and
    stashed on the context so the caller can retrieve it from the
    public entry point.
    """
    lo, hi = ctx.time_slice.start, ctx.time_slice.stop
    mismatch = _compute_mismatch_table(ctx.event_table, ctx.trials, ctx.track_graph)
    in_window = (mismatch["event_midpoint_time"] >= lo) & (
        mismatch["event_midpoint_time"] <= hi
    )
    view = mismatch.loc[in_window]

    for arm in _ALL_ARMS:
        ax.axhline(arm, color="lightgrey", linewidth=0.3, zorder=0)

    if len(view) == 0:
        ax.text(
            0.5,
            0.5,
            "no events in window",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="grey",
        )
    else:
        times = view["event_midpoint_time"].to_numpy()
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
            color="grey",
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
            color="grey",
            markersize=6,
            markeredgewidth=1.0,
            linestyle="none",
            label="actual (from trials)",
        ),
        Line2D([], [], color="lightgrey", linewidth=1.5, label="match"),
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
        ax.axhline(arm, color="lightgrey", linewidth=0.3, zorder=0)

    if len(view) == 0:
        ax.text(
            0.5,
            0.5,
            "no in-event timebins in window",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="grey",
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


def _draw_event_arm_heatmap(
    ax: Axes,
    ctx: _PanelContext,
    *,
    value_col: str,
    cmap_name: str,
    vmin: float | None,
    vmax: float | None,
    label: str,
    center: float | None = None,
    structural_na_rows: frozenset[int] = frozenset(),
    scale_across_table: bool = False,
    robust_symmetric_quantile: float | None = None,
    colorbar_group: str | None = None,
    colorbar_label: str | None = None,
    colorbar_ticks: tuple[float, ...] | None = None,
) -> None:
    """Shared: per-(event, arm) value drawn as colored rectangles.

    ``event_arm_table`` has 9 rows per event; we draw each event-level
    sample as a rectangle spanning ``[event_start, event_end]`` at arm
    height ± 0.5. The rectangle communicates event support, not a value
    continuously re-evaluated throughout the event. Color is ``cmap_name``
    normalized by ``(vmin, vmax)`` — when either is ``None`` it is inferred
    from the data in the active time window unless ``scale_across_table`` is
    true. The latter is appropriate for centered covariates whose model units
    are session-wide: changing the inspection window must not recolor the same
    value.

    ``center`` makes the scale diverging: the inferred limits are
    widened to be symmetric about that value, so a diverging colormap's
    neutral midpoint lands on it. Symmetric widening is used rather than
    ``TwoSlopeNorm`` because the latter raises when the windowed data
    happens to fall entirely on one side of the center.

    ``robust_symmetric_quantile`` optionally sets the symmetric half-range to
    a quantile of the absolute, session-wide deviations from ``center``.
    Saturated colors and colorbar extensions then disclose clipped outliers.
    This is useful for recency, whose never-visited sentinel can otherwise
    erase nearly all visible variation.
    """
    lo, hi = ctx.time_slice.start, ctx.time_slice.stop
    # ``event_arm_table`` already carries the broadcast
    # ``event_start`` / ``event_end`` per row (canonical
    # schema), so no merge with ``event_table`` is needed.
    eat = ctx.event_arm_table
    _require_columns(
        eat,
        "event_arm_table",
        ("event_start", "event_end", "arm", value_col),
    )
    view = eat[(eat["event_start"] <= hi) & (eat["event_end"] >= lo)]

    _format_arm_axis(ax, label)

    if len(view) == 0:
        ax.text(
            0.5,
            0.5,
            "no events in window",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="grey",
        )
        return

    values = view[value_col].to_numpy(dtype=float)
    arms = view["arm"].to_numpy(dtype=int)
    structural = np.isin(arms, tuple(structural_na_rows))
    finite = np.isfinite(values) & ~structural
    if scale_across_table:
        scale_values = eat[value_col].to_numpy(dtype=float)
        scale_arms = eat["arm"].to_numpy(dtype=int)
        scale_finite = np.isfinite(scale_values) & ~np.isin(
            scale_arms, tuple(structural_na_rows)
        )
    else:
        scale_values = values
        scale_finite = finite
    if vmin is None:
        vmin = float(scale_values[scale_finite].min()) if scale_finite.any() else 0.0
    if vmax is None:
        vmax = float(scale_values[scale_finite].max()) if scale_finite.any() else 1.0
    colorbar_extend = "neither"
    if robust_symmetric_quantile is not None:
        if center is None:
            raise ValueError("robust_symmetric_quantile requires center")
        if not 0.0 < robust_symmetric_quantile <= 1.0:
            raise ValueError("robust_symmetric_quantile must be in (0, 1]")
    if center is not None:
        if robust_symmetric_quantile is None:
            half = max(abs(vmax - center), abs(center - vmin))
        else:
            deviations = np.abs(scale_values[scale_finite] - center)
            half = (
                float(np.quantile(deviations, robust_symmetric_quantile))
                if deviations.size
                else 0.0
            )
        if half == 0.0:
            # Preserve the requested center even when every value equals it.
            # ``[center, center + 1]`` would incorrectly map zero to one end
            # of a diverging scale rather than its neutral midpoint.
            half = 1.0
        vmin, vmax = center - half, center + half
        if robust_symmetric_quantile is not None and scale_finite.any():
            below = bool((scale_values[scale_finite] < vmin).any())
            above = bool((scale_values[scale_finite] > vmax).any())
            if below and above:
                colorbar_extend = "both"
            elif below:
                colorbar_extend = "min"
            elif above:
                colorbar_extend = "max"
    if vmax <= vmin:
        vmax = vmin + 1.0
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)

    starts = view["event_start"].to_numpy(dtype=float)
    ends = view["event_end"].to_numpy(dtype=float)
    # Window-relative floor, matching every other interval-cell call site. A
    # fixed absolute floor renders sub-pixel on a full-session window (a 0.15 s
    # event over 2400 s is ~0.06 px), so continuous rows would come up blank
    # while the binary rows beside them stayed visible — and the figure legend
    # reads "blank = no event".
    cell_min_width = max((hi - lo) * 0.002, 1e-6)
    observed = np.isfinite(values) & ~structural
    _add_interval_cells(
        ax,
        starts[observed],
        ends[observed],
        arms[observed],
        [cmap(norm(value)) for value in values[observed]],
        min_width=cell_min_width,
    )
    missing = ~np.isfinite(values) & ~structural
    _add_interval_cells(
        ax,
        starts[missing],
        ends[missing],
        arms[missing],
        [_MISSING_CELL_COLOR] * int(missing.sum()),
        min_width=cell_min_width,
        edgecolors="#666666",
        hatch="///",
    )
    _add_interval_cells(
        ax,
        starts[structural],
        ends[structural],
        arms[structural],
        [_STRUCTURAL_NA_COLOR] * int(structural.sum()),
        min_width=cell_min_width,
        edgecolors="#bdbdbd",
        hatch="..",
    )

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    _add_colorbar(
        ax,
        ctx,
        sm,
        group=colorbar_group,
        label=colorbar_label,
        ticks=colorbar_ticks,
        extend=colorbar_extend,
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
    _require_columns(
        eat,
        "event_arm_table",
        ("event_start", "event_end", "arm", value_col),
    )
    view = eat[(eat["event_start"] <= hi) & (eat["event_end"] >= lo)]

    _format_arm_axis(ax, label)

    if len(view) == 0:
        ax.text(
            0.5,
            0.5,
            "no events in window",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="grey",
        )
        return

    values = view[value_col].to_numpy(dtype=float)
    values = _overlay_missing_references(values, view, reference_cols)
    _validate_binary(values, f"event_arm_table.{value_col}")
    finite = np.isfinite(values)
    arms = view["arm"].to_numpy(dtype=float)
    starts = view["event_start"].to_numpy(dtype=float)
    ends = view["event_end"].to_numpy(dtype=float)
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


# ── Per-arm predictor panels (trials → per-arm event-arm row) ──


def _plot_visit_recency(ax: Axes, ctx: _PanelContext) -> None:
    """Trials since the last destination visit; trunks are not destinations."""
    _draw_event_arm_heatmap(
        ax,
        ctx,
        value_col="visit_recency",
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
        value_col="cumulative_reward",
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
        value_col="reward_rate_arm_k10",
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


def _plot_p_reward_arm(ax: Axes, ctx: _PanelContext) -> None:
    """Programmed (nominal) reward probability per arm.

    Sampled once for each event from the active contingency block (per-arm
    ``p_reward_port``); equals ``p_reward_current`` at the current arm.
    Trunk arms are not reward sites, so they are masked here rather than
    colored — but the model-ready table encodes their placeholder as 0.0 and
    fits it; the mask is a reading aid, not an exclusion.
    """
    _draw_event_arm_heatmap(
        ax,
        ctx,
        value_col="p_reward_arm",
        cmap_name=_PROBABILITY_CMAP,
        vmin=0.0,
        vmax=1.0,
        label="programmed p(R)\nassigned trial",
        structural_na_rows=_TRUNK_ARMS,
        colorbar_group="behavioral_probability",
        colorbar_label="probability",
        colorbar_ticks=(0.0, 0.5, 1.0),
    )


def _plot_dist_to_arm(ax: Axes, ctx: _PanelContext) -> None:
    _draw_event_arm_heatmap(
        ax,
        ctx,
        value_col="dist_to_arm",
        cmap_name="viridis_r",
        vmin=0,
        vmax=None,
        label="arm distance\n[graph units]",
    )


def _plot_arm_dist_rank(ax: Axes, ctx: _PanelContext) -> None:
    _draw_event_arm_heatmap(
        ax,
        ctx,
        value_col="arm_dist_rank",
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


def _plot_visit_recency_demeaned(ax: Axes, ctx: _PanelContext) -> None:
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
        value_col="visit_recency_demeaned",
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
# ``q_arm`` and ``arm_uncertainty`` are the mean and variance of the SAME
# per-leaf Beta posterior from the fitted behavioral model, so a
# value+uncertainty read stays construct-coherent. Both are distinct from
# ``p_reward_arm``, which is the experimenter's PROGRAMMED probability --
# plotting them adjacent makes the nominal-vs-believed contrast readable.


def _plot_q_arm(ax: Axes, ctx: _PanelContext) -> None:
    """Behavioral-model value per arm (Beta posterior mean).

    Bounded [0, 1] like ``p_reward_arm`` and drawn on the same colormap
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
        value_col="q_arm",
        cmap_name=_PROBABILITY_CMAP,
        vmin=0.0,
        vmax=1.0,
        label="Beta belief q\npre-outcome",
        structural_na_rows=_TRUNK_ARMS,
        colorbar_group="behavioral_probability",
        colorbar_label="probability",
        colorbar_ticks=(0.0, 0.5, 1.0),
    )


def _plot_arm_uncertainty(ax: Axes, ctx: _PanelContext) -> None:
    """Behavioral-model belief uncertainty per arm (Beta posterior variance).

    The variance counterpart of :func:`_plot_q_arm` from the same
    posterior. In this fitted model, alpha and beta remain at least 1 and
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
        value_col="arm_uncertainty",
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
        value_col="is_current",
        color="#1f77b4",
        label="is_current",
        reference_cols="current_arm",
    )


def _plot_is_prev(ax: Axes, ctx: _PanelContext) -> None:
    _draw_event_arm_indicator(
        ax,
        ctx,
        value_col="is_prev",
        color="#ff7f0e",
        label="is_prev",
        reference_cols="prev_arm",
    )


def _plot_is_last_rewarded_elsewhere(ax: Axes, ctx: _PanelContext) -> None:
    _draw_event_arm_indicator(
        ax,
        ctx,
        value_col="is_last_rewarded_elsewhere",
        color="#2ca02c",
        label="is_last_rew.\nelsewhere",
        reference_cols=("last_rewarded_arm", "current_arm"),
    )


def _plot_arm_never_visited(ax: Axes, ctx: _PanelContext) -> None:
    """Explicit novel-leaf indicator paired with censored visit recency."""
    _draw_event_arm_indicator(
        ax,
        ctx,
        value_col="arm_never_visited",
        color="#7b3294",
        label="arm never\nvisited",
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
            color="grey",
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
    ("current", "is_current", "s", "#708090"),  # square, slate
    ("prev", "is_prev", "D", "#5f9ea0"),  # diamond, cadet blue
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

    This is the per-event analog of the descriptive
    variables (``replay_frac_current`` / ``replay_frac_next`` /
    ``replay_frac_prev`` / ``replay_frac_last_rewarded``).
    """
    lo, hi = ctx.time_slice.start, ctx.time_slice.stop
    eat = ctx.event_arm_table
    et = ctx.event_table
    in_window = (et["event_midpoint_time"] >= lo) & (et["event_midpoint_time"] <= hi)
    et_w = et.loc[in_window]

    ax.set_ylabel("Replay Frac\n(by role)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 0.5, 1])
    ax.axhline(
        1.0 / len(_ALL_ARMS),
        color="grey",
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
            color="grey",
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
            0.5,
            0.5,
            "no events in window",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="grey",
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
            0.5,
            0.5,
            "no stops in window",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="grey",
        )
        ax.set_ylim(0, 1)
        return

    mids = 0.5 * (view["stop_start"].to_numpy() + view["stop_end"].to_numpy())
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

    Six flag rows (``is_after_reward``, ``is_after_omission``,
    ``is_after_switch``, ``is_after_stay``, ``is_swr``, ``is_mua``)
    displayed as filled cells per event: colored when True, white
    when False. The heatmap encoding is more legible than ``|``
    ticks at this density because flag co-occurrence within a single
    event reads as a filled column rather than overlapping marks.

    Undefined is drawn as a distinct hatched grey cell, never as white.
    ``build_event_table`` assigns ``NaN`` to the outcome/transition flags at
    session boundaries, so the first stop of every session has genuinely
    undefined trial context; rendering that as white would read as "all four
    context flags observed False" — i.e. as evidence the covariates are
    miscoded rather than absent.
    """
    lo, hi = ctx.time_slice.start, ctx.time_slice.stop
    et = ctx.event_table
    view = et[
        (et["event_midpoint_time"] >= lo) & (et["event_midpoint_time"] <= hi)
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
            0.5,
            0.5,
            "no events in window",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="grey",
        )
        return

    _require_columns(view, "event_table", flag_names)
    n_events = len(view)
    starts = view["event_start"].to_numpy(dtype=float)
    ends = view["event_end"].to_numpy(dtype=float)
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


# ── Patch-grain panels (the P0–P4 ladder's design matrix) ──
#
# ``event_patch_table`` carries three rows per event but, unlike
# ``event_arm_table``, does NOT broadcast ``event_start`` / ``event_end``.
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
    if value_col not in ept.columns:
        raise KeyError(
            f"event_patch_table has no column {value_col!r}; "
            f"available: {sorted(ept.columns)}"
        )
    if ctx._event_patch_with_times is None:
        _require_columns(ept, "event_patch_table", ("event_id", "patch"))
        _require_columns(
            ctx.event_table,
            "event_table",
            ("event_id", "event_start", "event_end"),
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
            ["event_id", "event_start", "event_end", *context_columns]
        ]
        # Event-level context is authoritative here; avoid suffixing if an
        # external patch table happens to carry stale broadcast copies.
        patch_rows = ept.drop(
            columns=["event_start", "event_end", *context_columns],
            errors="ignore",
        )
        ctx._event_patch_with_times = patch_rows.merge(
            times,
            on="event_id",
            how="left",
            validate="many_to_one",
        )

    merged = ctx._event_patch_with_times
    if value_col not in merged.columns:
        raise KeyError(f"event_patch_table has no column {value_col!r}")
    lo, hi = ctx.time_slice.start, ctx.time_slice.stop
    return merged[(merged["event_start"] <= hi) & (merged["event_end"] >= lo)]


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
        ax.axhline(patch, color="lightgrey", linewidth=0.3, zorder=0)
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
            color="grey",
        )
        return

    lo, hi = ctx.time_slice.start, ctx.time_slice.stop
    values = view[value_col].to_numpy(dtype=float)
    if binary_color is not None:
        values = _overlay_missing_references(values, view, reference_cols)
    patches = view["patch"].to_numpy()
    starts = view["event_start"].to_numpy()
    ends = view["event_end"].to_numpy()

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


def _plot_q_arm_patch(ax: Axes, ctx: _PanelContext) -> None:
    """Event-sampled behavioral-model value aggregated to patch."""
    _draw_event_patch_heatmap(
        ax,
        ctx,
        value_col="q_arm_patch",
        cmap_name=_PROBABILITY_CMAP,
        vmin=0.0,
        vmax=1.0,
        label="mean leaf q\nper patch",
    )


def _plot_p_reward_arm_patch(ax: Axes, ctx: _PanelContext) -> None:
    """Event-sampled programmed reward probability aggregated to patch."""
    _draw_event_patch_heatmap(
        ax,
        ctx,
        value_col="p_reward_arm_patch",
        cmap_name=_PROBABILITY_CMAP,
        vmin=0.0,
        vmax=1.0,
        label="mean leaf p(R)\nper patch",
    )


def _plot_cumulative_reward_patch(ax: Axes, ctx: _PanelContext) -> None:
    """Rewards on this patch over the last ten completed trials.

    A count of recent rewarded exposure, not a rate: it is the sum over
    the patch's arms of a last-ten-*trial* reward count, so it shares no
    denominator with the visit-window reward-rate terms.
    """
    _draw_event_patch_heatmap(
        ax,
        ctx,
        value_col="cumulative_reward_patch",
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


def _plot_is_last_rewarded_patch_elsewhere(ax: Axes, ctx: _PanelContext) -> None:
    _draw_event_patch_heatmap(
        ax,
        ctx,
        value_col="is_last_rewarded_patch_elsewhere",
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


# ── Stop-grain panels (the Family-A rate design matrix) ──
#
# These vary per STOP, not per (event, arm), so they are keyed on
# ``stop_start`` / ``stop_end`` rather than event windows. Rendering them
# on the same time axis is what makes it possible to see that a rate
# covariate is constant across every event inside one stop.


def _stop_view(ctx: _PanelContext, cols: Sequence[str]) -> pd.DataFrame:
    """Validated, chronologically sorted stop rows overlapping the window."""
    st = ctx.stop_table
    _require_columns(st, "stop_table", ("stop_start", "stop_end", *cols))
    starts = st["stop_start"].to_numpy(dtype=float)
    ends = st["stop_end"].to_numpy(dtype=float)
    if not (np.isfinite(starts).all() and np.isfinite(ends).all()):
        raise ValueError("stop_table intervals must be finite")
    if np.any(ends < starts):
        raise ValueError("stop_table.stop_end must be at or after stop_start")
    lo, hi = ctx.time_slice.start, ctx.time_slice.stop
    view = st[(st["stop_start"] <= hi) & (st["stop_end"] >= lo)]
    return view.sort_values("stop_start", kind="stable")


def _plot_reward_lag_kernel(ax: Axes, ctx: _PanelContext) -> None:
    """The full ``Rlag`` design: reward outcome at lags 1–10, per stop.

    One row per lag, one column block per stop. Blue = rewarded, pale =
    observed omission, and dark-gray hatch = insufficient trial history.
    The missing-history cells identify rows excluded from a model that uses
    that lag; this plotting diagnostic does not alter any fitting cohort.
    """
    lag_cols = [f"reward_lag{j}" for j in range(1, N_REWARD_LAGS + 1)]
    view = _stop_view(ctx, lag_cols)

    ax.set_ylabel("reward lag\n(trials back)")
    ax.set_ylim(0.5, N_REWARD_LAGS + 0.5)
    ax.set_yticks([1, 5, 10])
    ax.invert_yaxis()

    if len(view) == 0:
        ax.text(
            0.5,
            0.5,
            "no stops in window",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="grey",
        )
        return

    lo, hi = ctx.time_slice.start, ctx.time_slice.stop
    starts = view["stop_start"].to_numpy()
    ends = view["stop_end"].to_numpy()
    min_width = max((hi - lo) * 0.002, 1e-6)
    for lag_idx, col in enumerate(lag_cols, start=1):
        vals = view[col].to_numpy(dtype=float)
        _validate_binary(vals, f"stop_table.{col}")
        finite = np.isfinite(vals)
        rows = np.full(len(vals), lag_idx, dtype=float)
        colors = [
            _REWARDED_COLOR if value == 1.0 else _ZERO_CELL_COLOR
            for value in vals[finite]
        ]
        _add_interval_cells(
            ax,
            starts[finite],
            ends[finite],
            rows[finite],
            colors,
            min_width=min_width,
            edgecolors="#bdbdbd",
            alpha=0.9,
        )
        missing = ~finite
        _add_interval_cells(
            ax,
            starts[missing],
            ends[missing],
            rows[missing],
            [_MISSING_CELL_COLOR] * int(missing.sum()),
            min_width=min_width,
            edgecolors="#666666",
            hatch="///",
        )


def _plot_reward_rate_trial_k10(ax: Axes, ctx: _PanelContext) -> None:
    """Rewarded fraction among up to ten completed trials, per stop.

    The value averages the available most-recent trials (maximum ten) at
    stop start and is missing only before the first completed trial. R2 uses
    this smoothed history together with ``reward_lag1``; Rlag instead uses
    the ten lag outcomes separately. On their production ladder-wide common
    cohort, R2 is an eight-degree-of-freedom constrained submodel of Rlag; all
    rungs use the exact same stop IDs.
    """
    view = _stop_view(ctx, ["reward_rate_trial_k10"])
    ax.set_ylabel("reward fraction\nprior ≤10 trials")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.0, color="grey", linewidth=0.3, zorder=0)

    if len(view) == 0:
        ax.text(
            0.5,
            0.5,
            "no stops in window",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="grey",
        )
        return

    starts = view["stop_start"].to_numpy(dtype=float)
    ends = view["stop_end"].to_numpy(dtype=float)
    vals = view["reward_rate_trial_k10"].to_numpy(dtype=float)
    finite = np.isfinite(vals)
    missing = ~finite
    if missing.any():
        lo, hi = ctx.time_slice.start, ctx.time_slice.stop
        _add_interval_cells(
            ax,
            np.maximum(starts[missing], lo),
            np.minimum(ends[missing], hi),
            np.full(int(missing.sum()), 0.5),
            [_MISSING_CELL_COLOR] * int(missing.sum()),
            min_width=max((hi - lo) * 0.002, 1e-6),
            edgecolors="#666666",
            alpha=0.35,
            hatch="///",
        )
    if finite.any():
        lo, hi = ctx.time_slice.start, ctx.time_slice.stop
        clipped_starts = np.maximum(starts[finite], lo)
        clipped_ends = np.minimum(ends[finite], hi)
        nonempty = clipped_ends >= clipped_starts
        segments = [
            ((start, value), (end, value))
            for start, end, value in zip(
                clipped_starts[nonempty],
                clipped_ends[nonempty],
                vals[finite][nonempty],
                strict=True,
            )
        ]
        ax.add_collection(
            LineCollection(
                segments,
                colors="#8c564b",
                linewidths=1.5,
                zorder=2,
            )
        )
        mids = 0.5 * (clipped_starts[nonempty] + clipped_ends[nonempty])
        ax.scatter(
            mids,
            vals[finite][nonempty],
            s=10,
            color="#8c564b",
            zorder=3,
        )


def _plot_is_after_switch_stop(ax: Axes, ctx: _PanelContext) -> None:
    """Per-stop patch-switch context with explicit zero/missing states."""
    view = _stop_view(ctx, ["is_after_switch"])
    ax.set_ylabel("is_after\nswitch")
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])

    if len(view) == 0:
        ax.text(
            0.5,
            0.5,
            "no stops in window",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="grey",
        )
        return

    lo, hi = ctx.time_slice.start, ctx.time_slice.stop
    starts = view["stop_start"].to_numpy(dtype=float)
    ends = view["stop_end"].to_numpy(dtype=float)
    vals = view["is_after_switch"].to_numpy(dtype=float)
    _validate_binary(vals, "stop_table.is_after_switch")
    finite = np.isfinite(vals)
    _add_interval_cells(
        ax,
        starts[finite],
        ends[finite],
        np.zeros(int(finite.sum())),
        ["#9467bd" if value == 1.0 else _ZERO_CELL_COLOR for value in vals[finite]],
        min_width=max((hi - lo) * 0.002, 1e-6),
        edgecolors="#bdbdbd",
        alpha=0.9,
    )
    missing = ~finite
    _add_interval_cells(
        ax,
        starts[missing],
        ends[missing],
        np.zeros(int(missing.sum())),
        [_MISSING_CELL_COLOR] * int(missing.sum()),
        min_width=max((hi - lo) * 0.002, 1e-6),
        edgecolors="#666666",
        hatch="///",
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

    One value per event at ``event_midpoint_time``: a thin vertical
    line from ``baseline`` to the value, plus a dot at the top. Works
    for any column of ``event_table`` whose dtype is numeric.
    """
    lo, hi = ctx.time_slice.start, ctx.time_slice.stop
    et = ctx.event_table
    in_window = (et["event_midpoint_time"] >= lo) & (et["event_midpoint_time"] <= hi)
    view = et.loc[in_window]

    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.axhline(baseline, color="grey", linewidth=0.3, zorder=0)
    ax.set_ylabel(ylabel)

    if len(view) == 0:
        ax.text(
            0.5,
            0.5,
            "no events in window",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="grey",
        )
        return

    times = view["event_midpoint_time"].to_numpy()
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
            color="grey",
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
    in_window = (et["event_midpoint_time"] >= lo) & (et["event_midpoint_time"] <= hi)
    view = et.loc[in_window].sort_values("event_midpoint_time")

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
            color="grey",
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
    "reward_rate_arm_k10": _plot_recent_reward_rate,
    "p_reward_arm": _plot_p_reward_arm,
    "dist_to_arm": _plot_dist_to_arm,
    "arm_dist_rank": _plot_arm_dist_rank,
    "arm_mass_stackplot": _plot_arm_mass_stackplot,
    "replay_frac_by_role": _plot_replay_frac_by_role,
    "event_bars": _plot_event_bars,
    "stop_activity": _plot_stop_activity,
    "context_flags": _plot_context_flags,
    # Behavioral-model (beta-bernoulli) beliefs, arm grain.
    "q_arm": _plot_q_arm,
    "arm_uncertainty": _plot_arm_uncertainty,
    # Remaining fitted arm-grain predictors.
    "visit_recency_demeaned": _plot_visit_recency_demeaned,
    **{
        _c: _make_recency_spline_panel(_c, _i + 1)
        for _i, _c in enumerate(RECENCY_SPLINE_COLUMNS)
    },
    "arm_never_visited": _plot_arm_never_visited,
    "is_current": _plot_is_current,
    "is_prev": _plot_is_prev,
    "is_last_rewarded_elsewhere": _plot_is_last_rewarded_elsewhere,
    # Patch ladder (P0–P4) design matrix.
    "q_arm_patch": _plot_q_arm_patch,
    "p_reward_arm_patch": _plot_p_reward_arm_patch,
    "cumulative_reward_patch": _plot_cumulative_reward_patch,
    "is_current_patch": _plot_is_current_patch,
    "is_last_rewarded_patch_elsewhere": _plot_is_last_rewarded_patch_elsewhere,
    "is_prev_patch": _plot_is_prev_patch,
    # Family-A per-stop rate design matrix.
    "reward_lag_kernel": _plot_reward_lag_kernel,
    "reward_rate_trial_k10": _plot_reward_rate_trial_k10,
    "is_after_switch_stop": _plot_is_after_switch_stop,
}

# Panel groups for convenient dispatch — the covariate panels are opt-in
# because ``DEFAULT_PANELS`` stays a detector-style overview.
BEHAVIORAL_MODEL_PANELS: tuple[str, ...] = (
    "p_reward_arm",
    "q_arm",
    "arm_uncertainty",
)
ARM_COVARIATE_PANELS: tuple[str, ...] = (
    "is_current",
    "is_prev",
    "is_last_rewarded_elsewhere",
    "visit_recency_demeaned",
    *RECENCY_SPLINE_COLUMNS,
    "arm_never_visited",
    "reward_rate_arm_k10",
    "cumulative_reward",
    "arm_dist_rank",
    # p_reward_arm and q_arm share a colorbar group, so they must stay
    # adjacent: a shared bar spans the block of axes it serves, and an
    # unrelated panel sitting between them would be drawn over.
    "p_reward_arm",
    "q_arm",
    "arm_uncertainty",
)
PATCH_COVARIATE_PANELS: tuple[str, ...] = (
    "is_current_patch",
    "is_prev_patch",
    "is_last_rewarded_patch_elsewhere",
    "cumulative_reward_patch",
    "p_reward_arm_patch",
    "q_arm_patch",
)
RATE_COVARIATE_PANELS: tuple[str, ...] = (
    "reward_lag_kernel",
    "reward_rate_trial_k10",
    "is_after_switch_stop",
)

_EVENT_INTERVAL_CELL_PANELS = frozenset(
    {
        "arm_count_hard",
        "arm_count_soft",
        "arm_posterior_mass",
        "arm_fraction",
        "visit_recency",
        "cumulative_reward",
        "reward_rate_arm_k10",
        "p_reward_arm",
        "dist_to_arm",
        "arm_dist_rank",
        "visit_recency_demeaned",
    *RECENCY_SPLINE_COLUMNS,
        "arm_never_visited",
        "q_arm",
        "arm_uncertainty",
        "is_current",
        "is_prev",
        "is_last_rewarded_elsewhere",
        *PATCH_COVARIATE_PANELS,
    }
)
_STRUCTURAL_NA_PANELS = frozenset(
    {
        "visit_recency",
        "cumulative_reward",
        "reward_rate_arm_k10",
        "p_reward_arm",
        "visit_recency_demeaned",
    *RECENCY_SPLINE_COLUMNS,
        "q_arm",
        "arm_uncertainty",
    }
)


def _add_figure_semantic_legend(fig: Figure, panel_names: tuple[str, ...]) -> None:
    """Add one compact key for state encodings used across audit panels."""
    selected = set(panel_names)
    has_event_cells = bool(selected & _EVENT_INTERVAL_CELL_PANELS)
    has_discrete_cells = has_event_cells or bool(
        selected
        & {
            "reward_lag_kernel",
            "is_after_switch_stop",
            # Both draw hatched missing-input cells and so need the key; without
            # them a hatched band renders with nothing explaining it.
            "reward_rate_trial_k10",
            "context_flags",
        }
    )
    if not (has_discrete_cells or "trial_rail" in selected):
        return

    handles: list[Any] = []
    if has_event_cells:
        handles.append(
            Patch(
                facecolor=plt.get_cmap(_PROBABILITY_CMAP)(0.65),
                edgecolor="none",
                label="event-sampled value",
            )
        )
    if has_discrete_cells:
        handles.extend(
            [
                Patch(
                    facecolor=_ZERO_CELL_COLOR,
                    edgecolor="#bdbdbd",
                    label="observed 0 / omission",
                ),
                Patch(
                    facecolor=_MISSING_CELL_COLOR,
                    edgecolor="#666666",
                    hatch="///",
                    label="missing input",
                ),
            ]
        )
    if selected & _STRUCTURAL_NA_PANELS:
        handles.append(
            Patch(
                facecolor=_STRUCTURAL_NA_COLOR,
                edgecolor="#bdbdbd",
                hatch="..",
                # Masked for reading, NOT dropped from the fit: production
                # writes 0.0 on trunk rows precisely so the conditional-Poisson
                # fitter does not discard the whole nine-arm event.
                label="trunk placeholder (fit as 0.0)",
            )
        )
    if "reward_lag_kernel" in selected:
        handles.append(
            Patch(
                facecolor=_REWARDED_COLOR,
                edgecolor="#bdbdbd",
                label="rewarded history",
            )
        )
    if "trial_rail" in selected:
        handles.extend(
            [
                Line2D(
                    [],
                    [],
                    marker="*",
                    linestyle="none",
                    markerfacecolor="gold",
                    markeredgecolor="black",
                    label="reward at goal arrival",
                ),
                Line2D(
                    [],
                    [],
                    color="#9467bd",
                    linewidth=1.2,
                    label="patch-switch trial start",
                ),
            ]
        )

    # "outside upper center" is laid out by constrained_layout, which reserves
    # space for the key. An explicit bbox_to_anchor above the axes is not
    # managed, so the key was drawn over the top panel and pushed the suptitle
    # off the canvas.
    legend = fig.legend(
        handles=handles,
        loc="outside upper center",
        ncol=min(4, len(handles)),
        frameon=False,
        fontsize=_ANNOTATION_FONTSIZE,
        title=(
            "Blank time in event×arm rows = no event"
            if has_event_cells
            else "Cell-state key"
        ),
    )
    legend.get_title().set_fontsize(_ANNOTATION_FONTSIZE)


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
    "reward_rate_arm_k10": 1.4,
    "p_reward_arm": 1.4,
    "dist_to_arm": 1.4,
    "arm_dist_rank": 1.4,
    "arm_mass_stackplot": 1.4,
    "replay_frac_by_role": 1.2,
    "event_bars": 1.2,
    "stop_activity": 1.2,
    "context_flags": 1.4,
    "q_arm": 1.4,
    "arm_uncertainty": 1.4,
    "visit_recency_demeaned": 1.4,
    **{_c: 1.4 for _c in RECENCY_SPLINE_COLUMNS},
    "arm_never_visited": 1.4,
    "is_current": 1.4,
    "is_prev": 1.4,
    "is_last_rewarded_elsewhere": 1.4,
    "q_arm_patch": 1.0,
    "p_reward_arm_patch": 1.0,
    "cumulative_reward_patch": 1.0,
    "is_current_patch": 1.0,
    "is_last_rewarded_patch_elsewhere": 1.0,
    "is_prev_patch": 1.0,
    "reward_lag_kernel": 1.6,
    "reward_rate_trial_k10": 1.0,
    "is_after_switch_stop": 0.5,
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


def _validated_result_time(results: xr.Dataset | xr.DataArray) -> np.ndarray:
    """Return a finite, strictly increasing one-dimensional time coordinate."""
    if not isinstance(results, (xr.Dataset, xr.DataArray)):
        raise TypeError("results must be an xarray Dataset or DataArray")
    if "time" not in results.coords:
        raise ValueError("results must have a 'time' coordinate")
    time = np.asarray(results.coords["time"].values, dtype=float)
    if time.ndim != 1 or time.size < 2:
        raise ValueError("results.time must be one-dimensional with at least 2 samples")
    if not np.isfinite(time).all():
        raise ValueError("results.time must contain only finite values")
    if np.any(np.diff(time) <= 0):
        raise ValueError("results.time must be strictly increasing")
    return time


def _validated_panels(panels: Sequence[str]) -> tuple[str, ...]:
    """Materialize and validate the requested panel sequence."""
    if isinstance(panels, (str, bytes)):
        raise TypeError("panels must be a non-empty sequence of panel names")
    try:
        panel_names = tuple(panels)
    except TypeError as exc:
        raise TypeError("panels must be a non-empty sequence of panel names") from exc
    if not panel_names:
        raise ValueError("panels must contain at least one panel name")
    if not all(isinstance(panel, str) for panel in panel_names):
        raise TypeError("every panels entry must be a string")
    unknown = [panel for panel in panel_names if panel not in _PANEL_REGISTRY]
    if unknown:
        raise KeyError(
            f"unknown panel name(s): {unknown}. known: {sorted(_PANEL_REGISTRY)}"
        )
    return panel_names


def _validated_time_window(
    result_time: np.ndarray,
    time_window: tuple[float, float] | None,
) -> tuple[float, float]:
    """Resolve a requested window and reject empty or reversed intersections."""
    result_start = float(result_time[0])
    result_end = float(result_time[-1])
    if time_window is None:
        return result_start, result_end
    try:
        requested_start, requested_end = (float(value) for value in time_window)
    except (TypeError, ValueError) as exc:
        raise ValueError("time_window must be a finite (start, end) pair") from exc
    if not np.isfinite([requested_start, requested_end]).all():
        raise ValueError("time_window bounds must be finite")
    if requested_start >= requested_end:
        raise ValueError("time_window start must be less than its end")
    start = max(result_start, requested_start)
    end = min(result_end, requested_end)
    if start >= end:
        raise ValueError("time_window does not overlap results.time")
    return start, end


def _relative_time_ticks(
    start: float,
    end: float,
    reference: float,
    *,
    nbins: int = 8,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return readable ticks anchored to a relative-time reference.

    Matplotlib's default locator operates on the large absolute epoch-time
    coordinates.  Formatting those ticks after subtracting ``reference`` can
    therefore produce arbitrary offsets such as ``-0.0836782`` and omit the
    scientifically important zero.  Locate ticks in relative coordinates
    first, force zero when it lies in the window, and only then translate the
    positions back to the absolute coordinate system used by the data.
    """
    relative_start = float(start - reference)
    relative_end = float(end - reference)
    locator = MaxNLocator(nbins=nbins, steps=(1.0, 2.0, 2.5, 5.0, 10.0))
    offsets = np.asarray(locator.tick_values(relative_start, relative_end), dtype=float)
    tolerance = max(relative_end - relative_start, 1.0) * 1e-10
    offsets = offsets[
        (offsets >= relative_start - tolerance) & (offsets <= relative_end + tolerance)
    ]
    if (
        relative_start <= 0.0 <= relative_end
        and not np.isclose(offsets, 0.0, atol=tolerance, rtol=0.0).any()
    ):
        offsets = np.append(offsets, 0.0)
    offsets = np.unique(np.where(np.abs(offsets) <= tolerance, 0.0, offsets))
    labels = tuple(f"{offset:g}" for offset in offsets)
    return reference + offsets, labels


def _validate_event_intervals(event_table: pd.DataFrame) -> None:
    """Validate interval columns used by every public figure."""
    _require_columns(
        event_table,
        "event_table",
        (
            "event_id",
            "event_start",
            "event_end",
            "event_midpoint_time",
            "next_arm",
        ),
    )
    if (
        event_table["event_id"].isna().any()
        or event_table["event_id"].duplicated().any()
    ):
        raise ValueError("event_table.event_id must be unique and non-missing")
    starts = event_table["event_start"].to_numpy(dtype=float)
    ends = event_table["event_end"].to_numpy(dtype=float)
    midpoints = event_table["event_midpoint_time"].to_numpy(dtype=float)
    if not (
        np.isfinite(starts).all()
        and np.isfinite(ends).all()
        and np.isfinite(midpoints).all()
    ):
        raise ValueError("event_table event intervals and midpoints must be finite")
    if np.any(ends < starts):
        raise ValueError("event_table.event_end must be at or after event_start")
    if np.any((midpoints < starts) | (midpoints > ends)):
        raise ValueError("event_table.event_midpoint_time must lie inside each event")


def plot_session_tables(
    data: Mapping[str, Any],
    detector: Any,
    results: xr.Dataset | xr.DataArray,
    event_table: pd.DataFrame,
    event_arm_table: pd.DataFrame,
    event_sequence_table: pd.DataFrame,
    stop_table: pd.DataFrame,
    *,
    event_patch_table: pd.DataFrame | None = None,
    trials: pd.DataFrame | None = None,
    track_graph: Any = None,
    time_window: tuple[float, float] | None = None,
    panels: Sequence[str] = DEFAULT_PANELS,
    figsize: tuple[float, float] = (12.0, 9.0),
    data_type: str = "clusterless",
    time_reference: float | None = None,
    xlabel: str | None = None,
    title: str | None = None,
    panel_group_labels: Mapping[str, str] | None = None,
    show_semantic_legend: bool = True,
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
        :func:`plot_detector`. ``results`` may be a Dataset or DataArray as
        long as it has a finite, increasing ``time`` coordinate. Covariate-
        only figures may pass ``detector=None`` and provide
        ``data['environment']``; decoder-dependent panels still require their
        normal fitted-detector inputs.
    event_table, event_arm_table, event_sequence_table, stop_table
        Canonical tables produced by
        :mod:`continuum_swr_replay.analysis.replay.tables`.
    event_patch_table : pd.DataFrame, optional
        Patch-collapsed table from
        :func:`~continuum_swr_replay.analysis.replay.tables.build_event_patch_table`.
        Required only by the patch-grain panels in
        :data:`PATCH_COVARIATE_PANELS`; every other panel ignores it.
    trials : pd.DataFrame, optional
        Trial DataFrame; falls back to ``data['trials']`` when absent.
    time_window : (float, float), optional
        ``(start_time, end_time)`` in seconds. Default: full session.
    panels : sequence of str
        Panel names, top-to-bottom order. See :data:`DEFAULT_PANELS`.
        Unknown names raise :class:`KeyError`.
    time_reference : float, optional
        Display x ticks relative to this absolute timestamp. Scientific table
        lookups remain in absolute seconds; only tick formatting changes.
    xlabel, title : str, optional
        Explicit bottom-axis label and figure title. When ``time_reference`` is
        supplied, the default x label is ``"Time from reference [s]"``.
    panel_group_labels : mapping, optional
        ``{first_panel_name: group_heading}`` labels used to structure long
        audit figures without adding a separate axes row.
    show_semantic_legend : bool
        Add one figure-level key for event support, observed zero, missing
        input, structural trunk N/A, and any reward markers in the panel set.
    figsize, data_type, save_path : passthrough.

    Returns
    -------
    fig : matplotlib.figure.Figure
    mismatch_table : pd.DataFrame
        Comparison between stored canonical ``next_arm`` and trial context
        recomputed at ``event_start`` through the production context and
        well-to-arm helpers. ``event_midpoint_time`` is retained as the
        plotting coordinate.
    """
    panel_names = _validated_panels(panels)
    result_time = _validated_result_time(results)
    t_start, t_end = _validated_time_window(result_time, time_window)
    _validate_event_intervals(event_table)
    if time_reference is not None:
        try:
            time_reference = float(time_reference)
        except (TypeError, ValueError) as exc:
            raise ValueError("time_reference must be a finite timestamp") from exc
        if not np.isfinite(time_reference):
            raise ValueError("time_reference must be a finite timestamp")

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

    time_slice = slice(t_start, t_end)

    env = None
    environments = getattr(detector, "environments", None)
    if environments is not None and len(environments) > 0:
        env = environments[0]
    if env is None:
        env = data.get("environment")
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
        event_patch_table=event_patch_table,
        defer_colorbars=True,
    )

    heights = [_PANEL_HEIGHTS.get(panel, 1.0) for panel in panel_names]
    fig, axes = plt.subplots(
        len(panel_names),
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

    for panel_name, ax in zip(panel_names, axes, strict=True):
        _PANEL_REGISTRY[panel_name](ax, ctx)

    axes_order = {id(ax): index for index, ax in enumerate(axes)}
    for request in ctx._colorbar_requests.values():
        # A shared colorbar is anchored across the full span of the axes it
        # serves, so if those axes are not adjacent it is drawn over whatever
        # sits between them — including that panel's own colorbar, which then
        # reads against the wrong scale. Fall back to per-axes bars instead of
        # rendering a misleading one.
        indices = sorted(axes_order[id(ax)] for ax in request.axes)
        contiguous = indices == list(range(indices[0], indices[0] + len(indices)))
        targets = [request.axes] if contiguous else [[ax] for ax in request.axes]
        for target in targets:
            fig.colorbar(
                request.mappable,
                ax=target,
                shrink=0.7,
                pad=0.01,
                aspect=10,
                label=request.label,
                ticks=request.ticks,
                extend=request.extend,
            )

    if panel_group_labels is not None:
        for panel_name, group_label in panel_group_labels.items():
            if panel_name not in panel_names:
                raise ValueError(
                    f"panel_group_labels key {panel_name!r} is not in panels"
                )
            group_ax = axes[panel_names.index(panel_name)]
            group_ax.set_title(
                str(group_label),
                loc="left",
                fontsize=9,
                fontweight="bold",
                pad=6,
            )

    if time_reference is None:
        axes[-1].ticklabel_format(useOffset=False, style="plain", axis="x")
        default_xlabel = "Time [s]"
    else:
        reference = time_reference
        tick_positions, tick_labels = _relative_time_ticks(t_start, t_end, reference)
        axes[-1].xaxis.set_major_locator(FixedLocator(tick_positions))
        axes[-1].xaxis.set_major_formatter(FixedFormatter(tick_labels))
        if t_start <= reference <= t_end:
            for ax in axes:
                ax.axvline(
                    reference,
                    color="0.25",
                    linestyle=":",
                    linewidth=0.8,
                    alpha=0.8,
                    zorder=4,
                    label="t = 0 reference",
                )
        default_xlabel = "Time from reference [s]"
    axes[-1].set_xlabel(default_xlabel if xlabel is None else xlabel)
    axes[-1].set_xlim(t_start, t_end)
    if title is not None:
        # No explicit y: constrained_layout places the suptitle inside the
        # canvas and above the semantic key. A hard y=1.035 put it off-figure,
        # so the title was clipped in any inline render.
        fig.suptitle(title, fontsize=11, fontweight="bold")
    if show_semantic_legend:
        _add_figure_semantic_legend(fig, panel_names)

    # Vertical event-span guides align continuous/context and stop-grain rows.
    # Event-cell panels already encode support with their own cell widths and
    # deliberately skip the redundant gray overlay.
    et_windowed = event_table[
        (event_table["event_start"] <= t_end) & (event_table["event_end"] >= t_start)
    ]
    if len(et_windowed) > 0:
        event_starts = np.maximum(
            et_windowed["event_start"].to_numpy(dtype=float), t_start
        )
        event_ends = np.minimum(et_windowed["event_end"].to_numpy(dtype=float), t_end)
        # One axes-coordinate collection replaces one ``axvspan`` artist per
        # event and panel. This keeps full-session covariate groups tractable
        # while preserving the same event-span geometry and alpha.
        span_vertices = [
            ((start, 0.0), (end, 0.0), (end, 1.0), (start, 1.0))
            for start, end in zip(event_starts, event_ends, strict=True)
        ]
        for panel_name, ax in zip(panel_names, axes, strict=True):
            if panel_name in _EVENT_INTERVAL_CELL_PANELS:
                # These rows already encode event support with their own cell
                # widths; another gray span only darkens the same geometry.
                continue
            guides = PolyCollection(
                span_vertices,
                facecolors="black",
                edgecolors="none",
                alpha=0.04,
                transform=ax.get_xaxis_transform(),
                zorder=0,
                rasterized=True,
            )
            ax.add_collection(guides, autolim=False)

    sns.despine(offset=5)

    # If the mismatch-pointers panel ran it stashed the full table on
    # ctx. Otherwise compute once here so callers get the same shape
    # regardless of panel selection (useful for headless CI / audit).
    mismatch_table = ctx._mismatch_table
    if mismatch_table is None:
        mismatch_table = _compute_mismatch_table(event_table, trials, track_graph)

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, mismatch_table
