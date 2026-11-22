"""The held-out gain panels: a held-out gain per animal and its mean.

Figure 2g (each decoder's gain over the local-only decoder, and the full
decoder's over the SWR-only decoder) and Figure 2's supplement
(``scripts/figures/generate_supp_figure02.py``: the held-out tetrode gain over
local coding per spike; the full decoder's gain outside SWRs, by distance from
the SWRs and by place sampling; and where that gain sits) draw each value the
same way: a dot per animal and, beside them, the animals' mean with its 95 %
interval in black, with a dashed line at zero, where neither side predicts the
spikes better.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.transforms import blended_transform_factory, offset_copy

from continuum_swr_replay.analysis.event_evidence.restriction_totals import (
    OUTSIDE_SWR_SHARES,
    RESTRICTION_FIGURE_FAMILY,
)
from continuum_swr_replay.analysis.event_evidence.state_evidence.windows import (
    SWR_LABELS,
)
from continuum_swr_replay.visualization.palettes import COLORS, readable_text_color
from continuum_swr_replay.visualization.simulated_run_panels import SMALL_SIZE

GROUP_COLORS = {SWR_LABELS[False]: COLORS["No SWR"], SWR_LABELS[True]: COLORS["SWR"]}
"""Each group's color, in the figures' order: no-SWR events, then SWR events."""

REFERENCE_COLOR = "#777777"
"""A reference level (the zero line): a neutral gray, not the No-Spike state's."""

HELDOUT_X = {SWR_LABELS[False]: 0.0, SWR_LABELS[True]: 1.0}
"""Each group's position on a gain axis' x axis."""


def animal_values(row: pd.Series, column: str, scale: float = 1.0) -> pd.Series:
    """A statistics row's per-animal values, indexed by animal."""
    return scale * pd.Series(row[column], index=row["animal_names"])


MEAN_OFFSET = 0.25
"""How far right of a group's dots its mean is drawn, in x units."""

DOT_SPREAD = 0.05
"""The sideways step between an animal's dot and the next (x units), when a
group's dots are spread so that each shows (:func:`spread_offsets`)."""

NEUTRAL_DOTS = "#444444"
"""A dark gray for dots that need no class's color (Figure 2g's differences,
the SWR-only decoder)."""


def spread_offsets(values: np.ndarray, step: float = DOT_SPREAD) -> np.ndarray:
    """Sideways offsets that spread a group's dots symmetrically about its
    position: in order of value, the dots alternate right and left of it (the
    middle one, of an odd number, on it), so that close values part and the
    dots do not read as a slope."""
    n = len(values)
    rank = np.argsort(np.argsort(values))
    offsets = np.where(rank % 2 == 0, step, -step).astype(float)
    if n % 2:
        offsets[rank == n // 2] = 0.0
    return offsets


def draw_animal_gain(
    ax: Axes, x: float, row: pd.Series, color: str, *, spread: bool = False
) -> None:
    """One group's gain at ``x``: a dot per animal in ``color`` and, at
    ``x + MEAN_OFFSET``, the animals' mean with its 95 % interval in black.

    Parameters
    ----------
    ax : Axes
    x : float
    row : pd.Series
        ``animals``, ``animal_names``, ``estimate``, ``ci_low`` and ``ci_high``
        (``likelihood_advantage_statistics``' columns).
    color : str
    spread : bool, optional
        Spread the dots sideways about ``x`` (:func:`spread_offsets`), so that
        animals with close values each show.
    """
    animals = animal_values(row, "animals")
    offsets = spread_offsets(animals.to_numpy()) if spread else np.zeros(animals.size)
    ax.scatter(x + offsets, animals, s=6, color=color, lw=0, zorder=2)
    ax.errorbar(
        x + MEAN_OFFSET, row["estimate"],
        yerr=[[row["estimate"] - row["ci_low"]], [row["ci_high"] - row["estimate"]]],
        fmt="_", color="black", ms=5, lw=0.8, capsize=0,
    )  # fmt: skip


def draw_group_gain(ax: Axes, statistics: pd.DataFrame, title: str) -> None:
    """One gain axis: a gain per spike, by group.

    A dot per animal in the group's color and, beside them, the animals' mean
    with its 95 % interval in black, for no-SWR and SWR events, with a dashed
    line at zero and the axis' title.

    Parameters
    ----------
    ax : Axes
    statistics : pd.DataFrame
        Indexed by group (``"no SWR"``, ``"SWR"``): ``animals``,
        ``animal_names``, ``estimate``, ``ci_low`` and ``ci_high``
        (``likelihood_advantage_statistics``' columns).
    title : str
    """
    for group, color in GROUP_COLORS.items():
        draw_animal_gain(ax, HELDOUT_X[group], statistics.loc[group], color)
    ax.axhline(0, color=REFERENCE_COLOR, lw=0.6, ls="--", zorder=1)
    ax.set_xlim(-0.6, 1.6)
    ax.set_xticks(list(HELDOUT_X.values()), ["No SWR", "SWR"], rotation=45, ha="right")
    ax.set_title(title, fontsize=SMALL_SIZE, pad=3)


def label_gain_axis(
    ax: Axes, ylabel: str, reference: str = "no gain", floor: float = -0.12
) -> None:
    """The y axis of a gain axis (or of axes sharing it): ``ylabel``, a floor
    below zero (``floor``, in the axis' units) and ``reference`` under the zero
    line at the axis' left, clear of the groups' dots and intervals."""
    ax.text(
        0.04, -0.02 * abs(floor) / 0.12, reference,
        transform=ax.get_yaxis_transform(), va="top", fontsize=SMALL_SIZE,
        color=REFERENCE_COLOR,
    )  # fmt: skip
    ax.set_ylim(floor, None)
    ax.set_ylabel(ylabel)


DIFFERENCE_YLABEL = "Held-out log score,\nfull \u2212 SWR-only\n(nats/spike)"
"""The y label of an axis of the full decoder minus the SWR-only decoder."""


def full_better_cue(ax: Axes) -> None:
    """ "\u2191 full better" just above the zero line at the axis' left, in the
    reference gray: which way the full decoder predicts the spikes better."""
    ax.text(
        0.03, 0.0, "\u2191 full better", ha="left", va="bottom",
        transform=blended_transform_factory(ax.transAxes, ax.transData),
        fontsize=SMALL_SIZE, color=REFERENCE_COLOR,
    )  # fmt: skip


def heading_over(fig: Figure, renderer: Any, axes: Sequence[Axes], title: str) -> None:
    """One heading centered over several axes (``axes``, left to right), just
    above the highest of their tops and their own titles and texts, with a thin
    line under it from the first frame's left edge to the last one's right
    edge, so it reads as the heading of them all. Both are drawn by the first
    axes, so that axes' drawn extent, and a panel letter placed from it,
    include them."""
    to_figure = fig.transFigure.inverted()
    boxes = [ax.get_position() for ax in axes]
    top = max(
        [box.y1 for box in boxes]
        + [
            text.get_window_extent(renderer).transformed(to_figure).y1
            for ax in axes
            for text in ([ax.title] if ax.get_title() else []) + list(ax.texts)
        ]
    )
    above = offset_copy(fig.transFigure, fig=fig, y=2, units="points")
    axes[0].plot(
        (boxes[0].x0, boxes[-1].x1), (top, top), transform=above,
        color="black", lw=0.5, clip_on=False,
    )  # fmt: skip
    axes[0].text(
        (boxes[0].x0 + boxes[-1].x1) / 2, top, title,
        transform=offset_copy(fig.transFigure, fig=fig, y=3, units="points"),
        ha="center", va="bottom", fontsize=SMALL_SIZE,
    )  # fmt: skip


SHARE_LABELS = dict(
    zip(OUTSIDE_SWR_SHARES, ("No SWR", "Partly in SWR", "Outside events"), strict=True)
)
"""Each share of the gain outside SWRs, named as panel d of Figure 2 names the
event classes, and the rest outside the events."""
SHARE_COLORS = dict(
    zip(
        OUTSIDE_SWR_SHARES,
        (COLORS["No SWR"], COLORS["SWR partial"], "#999999"),
        strict=True,
    )
)
"""Panel d's colors of the no-SWR events and of the events partly in an SWR,
and a neutral gray for the time outside events."""


def percent(row: pd.Series) -> pd.Series:
    """A statistics row with its values (``animals``, ``estimate``, ``ci_low``,
    ``ci_high``) in percent."""
    scaled = row.copy()
    for column in ("animals", "estimate", "ci_low", "ci_high"):
        scaled[column] = 100 * np.asarray(row[column])
    return scaled


def draw_outside_swr_shares(ax: Axes, outside: pd.DataFrame, ylabel: str) -> None:
    """Where the full decoder's gain over the SWR-only decoder outside SWRs
    sits, in percent of that gain: in the no-SWR events, in the time outside
    the SWR of the events partly in an SWR, and outside the events (the rest;
    per epoch the three add to 100 %).

    The three lie at very different heights, so they share one column: for
    each, a dot per animal (the median over its epochs, spread sideways), the
    animals' mean with its 95 % interval, and its name beside them in its color
    (text no lighter than the reference gray); a dashed line at zero; no x tick
    labels (``restriction_totals.outside_swr_gain_statistics``'
    :data:`~continuum_swr_replay.analysis.event_evidence.restriction_totals.OUTSIDE_SWR_SHARES`).

    Parameters
    ----------
    ax : Axes
    outside : pd.DataFrame
        ``outside_swr_gain_statistics``' rows, indexed by ``family`` and
        ``measure``.
    ylabel : str
    """
    rows = {
        measure: percent(outside.loc[(RESTRICTION_FIGURE_FAMILY, measure)])
        for measure in OUTSIDE_SWR_SHARES
    }
    for measure, row in rows.items():
        color = SHARE_COLORS[measure]
        draw_animal_gain(ax, 0.0, row, color, spread=True)
        text_color = readable_text_color(color)
        if text_color == color and color == SHARE_COLORS[OUTSIDE_SWR_SHARES[-1]]:
            text_color = REFERENCE_COLOR  # the gray dots' name, no lighter than it
        ax.annotate(
            SHARE_LABELS[measure], (MEAN_OFFSET, row["estimate"]),
            xytext=(4, 0), textcoords="offset points", ha="left", va="center",
            fontsize=SMALL_SIZE, color=text_color, annotation_clip=False,
        )  # fmt: skip
    ax.axhline(0, color=REFERENCE_COLOR, lw=0.6, ls="--", zorder=1)
    ax.set_xlim(-0.4, 0.4)
    ax.set_xticks([])
    low = min(min(float(r["animals"].min()), r["ci_low"]) for r in rows.values())
    high = max(max(float(r["animals"].max()), r["ci_high"]) for r in rows.values())
    ax.set_ylim(min(low, 0) - 0.06 * (high - low), high + 0.06 * (high - low))
    ax.set_ylabel(ylabel)
