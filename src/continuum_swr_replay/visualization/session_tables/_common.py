"""Shared context, constants and cell/colorbar helpers for the session-tables panels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle

from continuum_swr_replay.data_loaders.constants import ALL_ARMS as _ALL_ARMS
from continuum_swr_replay.data_loaders.constants import TRUNK_ARMS
from continuum_swr_replay.tables.columns import require_columns

_TRUNK_ARMS: frozenset[int] = frozenset(TRUNK_ARMS)
_COLLECTION_THRESHOLD = 128
_ZERO_CELL_COLOR = "#f2f2f2"
_MISSING_CELL_COLOR = "#969696"
_STRUCTURAL_NA_COLOR = "#e3e3e3"
_PROBABILITY_CMAP = "cividis"
_ANNOTATION_FONTSIZE = 7
_REWARDED_COLOR = "#0072B2"


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
    port_experiences: pd.DataFrame
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
        ax.axhline(arm, color="lightgray", linewidth=0.3, zorder=0)
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
    sample as a rectangle spanning ``[event_start_s, event_end_s]`` at arm
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
    # ``event_start_s`` / ``event_end_s`` per row (canonical
    # schema), so no merge with ``event_table`` is needed.
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
    arms = view["candidate_arm"].to_numpy(dtype=int)
    structural = np.isin(arms, tuple(structural_na_rows))
    finite = np.isfinite(values) & ~structural
    if scale_across_table:
        scale_values = eat[value_col].to_numpy(dtype=float)
        scale_arms = eat["candidate_arm"].to_numpy(dtype=int)
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

    starts = view["event_start_s"].to_numpy(dtype=float)
    ends = view["event_end_s"].to_numpy(dtype=float)
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
