"""The held-out gain panels: a held-out gain per animal and its mean.

Figure 2 (j, the left-out tetrode's better fit to the decoded place than to
the animal's place and to another event's place; k, the full decoder's gain
over the SWR-only decoder) and Figure 2's supplement
(``scripts/figures/generate_supp_figure02.py``: the fit over the animal's
place split by bins with and without a spike; the full decoder's
gain outside SWRs by distance from the SWRs; the four restriction decoders,
with the SWR-allowed one) draw each value the
same way: a dot per animal and, beside them, the animals' mean with its 95 %
interval in black, with a dashed line at zero, where neither side predicts the
spikes better.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.transforms import blended_transform_factory

from continuum_swr_replay.analysis.event_evidence.state_evidence.windows import (
    SWR_LABELS,
)
from continuum_swr_replay.visualization.palettes import (
    COLORS,
    REFERENCE_COLOR,
    readable_text_color,
)
from continuum_swr_replay.visualization.simulated_run_panels import SMALL_SIZE

GROUP_COLORS = {SWR_LABELS[False]: COLORS["No SWR"], SWR_LABELS[True]: COLORS["SWR"]}
"""Each group's color, in the figures' order: no-SWR events, then SWR events."""
GROUP_NAMES = {SWR_LABELS[False]: "No SWR", SWR_LABELS[True]: "SWR"}
"""The name of each group of :data:`GROUP_COLORS` in a panel's text."""


def name_groups(ax: Axes, x: float, top: float, step: float, ha: str) -> None:
    """The groups' names (:data:`GROUP_NAMES`) in their colors, one under the
    other from ``top`` down by ``step``, at ``x`` aligned by ``ha`` (axes
    fractions)."""
    for k, (group, color) in enumerate(GROUP_COLORS.items()):
        ax.text(
            x, top - step * k, GROUP_NAMES[group], transform=ax.transAxes, ha=ha,
            va="top", fontsize=SMALL_SIZE, color=readable_text_color(color),
        )  # fmt: skip


def animal_values(row: pd.Series, column: str, scale: float = 1.0) -> pd.Series:
    """A statistics row's per-animal values, indexed by animal."""
    return scale * pd.Series(row[column], index=row["animal_names"])


MEAN_OFFSET = 0.25
"""How far right of a group's dots its mean is drawn, in x units."""

DOT_SPREAD = 0.05
"""The sideways step between an animal's dot and the next (x units), when a
group's dots are spread so that each shows (:func:`spread_offsets`)."""

NEUTRAL_DOTS = "#444444"
"""A dark gray for dots that need no class's color (Figure 2k's differences,
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
    ax: Axes,
    x: float,
    row: pd.Series,
    color: str,
    *,
    spread: bool = False,
    hollow: bool = False,
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
    hollow : bool, optional
        Open dots (an outline in ``color``): a variant of the group drawn
        filled beside it.
    """
    animals = animal_values(row, "animals")
    offsets = spread_offsets(animals.to_numpy()) if spread else np.zeros(animals.size)
    if hollow:
        ax.scatter(
            x + offsets, animals, s=7, facecolors="none", edgecolors=color, lw=0.5,
            zorder=2,
        )  # fmt: skip
    else:
        ax.scatter(x + offsets, animals, s=6, color=color, lw=0, zorder=2)
    ax.errorbar(
        x + MEAN_OFFSET, row["estimate"],
        yerr=[[row["estimate"] - row["ci_low"]], [row["ci_high"] - row["estimate"]]],
        fmt="_", color="black", ms=5, lw=0.8, capsize=0,
    )  # fmt: skip


SPLIT_LABELS = {
    "gain_per_spike": "All bins",
    "spiking_gain_per_spike": "Spiking",
    "silent_gain_per_spike": "Silent",
}
"""The held-out tetrode gain and its two parts, by ``tetrode_holdout.window_scores``
column: over every bin, over the bins where the scored tetrode spiked, and over
those where it was silent."""

SPLIT_GROUP_GAP = 0.8
"""The extra x space between the no-SWR and the SWR positions."""


def split_x(group: int, part: int) -> float:
    """The x position of a part (0, 1, 2 in :data:`SPLIT_LABELS`' order) of a
    group's gain (0 no SWR, 1 SWR)."""
    return group * (len(SPLIT_LABELS) + SPLIT_GROUP_GAP) + part


def draw_split_gain(ax: Axes, statistics: dict, title: str) -> None:
    """A gain per spike by group and by part: for no-SWR then SWR events, the
    gain over all bins, over the bins with a spike and over the silent bins
    (:data:`SPLIT_LABELS`), each a dot per animal in the group's color and the
    animals' mean with its 95 % interval in black, the groups named over their
    positions, a dashed line at zero and the axis' title.

    Parameters
    ----------
    ax : Axes
    statistics : dict
        By column of :data:`SPLIT_LABELS`: statistics indexed by group
        (``"no SWR"``, ``"SWR"``) with ``animals``, ``animal_names``,
        ``estimate``, ``ci_low`` and ``ci_high``
        (``likelihood_advantage_statistics``' columns).
    title : str
    """
    ticks, labels = [], []
    for k, (group, color) in enumerate(GROUP_COLORS.items()):
        for j, (column, label) in enumerate(SPLIT_LABELS.items()):
            draw_animal_gain(ax, split_x(k, j), statistics[column].loc[group], color)
            ticks.append(split_x(k, j))
            labels.append(label)
        ax.text(
            split_x(k, 1) + MEAN_OFFSET / 2, 1.0, GROUP_NAMES[group],
            transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=SMALL_SIZE, color=readable_text_color(color),
        )  # fmt: skip
    draw_zero_line(ax)
    ax.set_xlim(-0.5, split_x(1, len(SPLIT_LABELS) - 1) + 0.6)
    ax.set_xticks(ticks, labels, rotation=90)
    ax.set_title(title, fontsize=SMALL_SIZE, pad=3)


def gain_range(rows: Iterable[pd.Series]) -> tuple[float, float]:
    """The lowest and highest of statistics rows' animals and interval bounds
    (``animals``, ``ci_low``, ``ci_high``): what a gain axis must show."""
    rows = list(rows)
    low = min(min(float(np.min(r["animals"])), r["ci_low"]) for r in rows)
    high = max(max(float(np.max(r["animals"])), r["ci_high"]) for r in rows)
    return low, high


def draw_zero_line(ax: Axes) -> None:
    """The dashed line at zero, in the reference gray, under the data."""
    ax.axhline(0, color=REFERENCE_COLOR, lw=0.6, ls="--", zorder=1)


EQUAL_FIT = "equal fit"
"""The name of a zero line where both sides fit the spikes equally well."""


def name_zero_line(
    ax: Axes, text: str, x: float, *, y: float = 0.0, offset_pt: float = 0.0,
    va: str = "bottom",
) -> None:  # fmt: skip
    """``text`` beside the zero line in the reference gray, its left edge at
    ``x`` (axes fraction) and its anchor at ``y`` (data units) moved
    ``offset_pt`` points up, aligned by ``va``: a zero line's name or a cue
    to which side is which."""
    ax.annotate(
        text, (x, y), xycoords=blended_transform_factory(ax.transAxes, ax.transData),
        xytext=(0, offset_pt), textcoords="offset points", ha="left", va=va,
        fontsize=SMALL_SIZE, color=REFERENCE_COLOR,
    )  # fmt: skip


def label_gain_axis(ax: Axes, ylabel: str, floor: float) -> None:
    """The y axis of a gain axis (or of axes sharing it): ``ylabel`` and a
    floor below zero (``floor``, in the axis' units)."""
    ax.set_ylim(floor, None)
    ax.set_ylabel(ylabel)


def reference_y(floor: float) -> float:
    """Where a zero line's name hangs under the line on a gain axis with
    ``floor`` (data units): clear of the groups' dots and intervals."""
    return -0.02 * abs(floor) / 0.12


GAIN_FLOOR = 0.13
"""A gain axis' floor below zero, as a share of its top (Figure 2k, Figure 2's
supplement, panels i and l)."""

DIFFERENCE_YLABEL = "Held-out log score,\nfull \u2212 SWR-only\n(nats/spike)"
"""The y label of an axis of the full decoder minus the SWR-only decoder."""

DIFFERENCE_LABELS = {
    "all": "All\nimmobility",
    "inside_swr": "Inside\nSWRs",
    "outside_swr": "Outside\nSWRs",
}
"""An axis of differences between decoders (Figure 2's supplement, panel l): over all the immobile time, over its bins inside SWRs and
over its bins outside SWRs, left to right; horizontal, on two lines."""
EVENTS_LABELS = {
    "in_events": "In no-SWR\nevents",
    "outside_events": "Outside\nevents",
}
"""The axis of a difference outside SWRs in the held-out events (Figure 2's
supplement, panel l, third axis): inside the no-SWR events detected
from the held-out decode and outside every held-out event, left to right."""
EVENTS_TITLE = "Outside SWRs,\nheld-out events"

DECODER_STYLES = {
    "swr_minus_none_per_spike": ("SWR-\nonly", NEUTRAL_DOTS, False),
    "swr_allowed_minus_none_per_spike": ("SWR-\nallowed", NEUTRAL_DOTS, True),
    "full_minus_none_per_spike": ("Full", COLORS["Non-Local"], False),
}
"""Each decoder's gain over the local-only decoder (Figure 2's supplement,
panel l, first axis), by its measure: its label, its dots' color and
whether they are open. The SWR-only decoder in a neutral dark gray, the
SWR-allowed decoder, a variant of it, in open dots of the same gray, the full
decoder in the Non-Local state's green."""
DECODERS_YLABEL = "Held-out log score\nover local-only\n(nats/spike)"
DECODERS_REFERENCE = "local-only"


def draw_decoder_gain_axis(ax: Axes, rows: Mapping[str, pd.Series]) -> None:
    """Each decoder's gain over the local-only decoder, at 0, 1, ... in the
    order of ``rows`` (statistics rows by measure of :data:`DECODER_STYLES`): a
    dot per animal in the decoder's style (spread sideways) and the animals'
    mean with its 95 % interval, the decoders named under the axis, a dashed
    line at zero labeled as the local-only decoder."""
    for x, (measure, row) in enumerate(rows.items()):
        _, color, hollow = DECODER_STYLES[measure]
        draw_animal_gain(ax, float(x), row, color, spread=True, hollow=hollow)
    draw_zero_line(ax)
    ax.set_xlim(-0.45, len(rows) - 0.5)
    ax.set_xticks(
        [float(x) for x in range(len(rows))], [DECODER_STYLES[m][0] for m in rows]
    )
    top = gain_range(rows.values())[1] * 1.08
    # twice the usual floor, so that the reference's name clears the frame of
    # this short axis
    floor = -2 * GAIN_FLOOR * top
    name_zero_line(ax, DECODERS_REFERENCE, 0.04, y=reference_y(floor), va="top")
    label_gain_axis(ax, DECODERS_YLABEL, floor)
    ax.set_ylim(top=top)


def full_better_cue(ax: Axes) -> None:
    """ "\u2191 full better" just above the zero line at the axis' left, in the
    reference gray: which way the full decoder predicts the spikes better."""
    name_zero_line(ax, "\u2191 full better", 0.03)


def draw_full_minus_swr_axis(
    ax: Axes,
    xs: Sequence[float],
    rows: Sequence[pd.Series],
    top_scale: float,
    *,
    through_means: bool = False,
    hollow: Sequence[bool] | None = None,
    ylabel: str = DIFFERENCE_YLABEL,
    cue: bool = True,
) -> None:
    """An axis of the full decoder minus the SWR-only decoder: at each of ``xs``
    a dot per animal (spread sideways, :data:`NEUTRAL_DOTS`) and the animals'
    mean with its interval (``rows``, statistics rows), a thin line through the
    means when ``through_means``, a dashed line at zero, the y axis from
    :data:`GAIN_FLOOR` below zero (or ``top_scale`` times the lowest dot or
    bound, when lower) to ``top_scale`` times the highest dot or bound, ``ylabel`` (by default :data:`DIFFERENCE_YLABEL`) and, when ``cue``, :func:`full_better_cue`. ``hollow`` marks the columns drawn in open dots (none by default). The x axis is
    the caller's (Figure 2k, Figure 2's supplement, panels i and l)."""
    if hollow is None:
        hollow = [False] * len(xs)
    for x, row, open_dots in zip(xs, rows, hollow, strict=True):
        draw_animal_gain(ax, x, row, NEUTRAL_DOTS, spread=True, hollow=open_dots)
    if through_means:
        ax.plot(
            [x + MEAN_OFFSET for x in xs], [row["estimate"] for row in rows],
            color=REFERENCE_COLOR, lw=0.5, zorder=1,
        )  # fmt: skip
    draw_zero_line(ax)
    low, high = gain_range(rows)
    top = top_scale * high
    # a value below zero sets the floor, with the same margin as the top
    ax.set_ylim(min(-GAIN_FLOOR * top, top_scale * low), top)
    ax.set_ylabel(ylabel)
    if cue:
        full_better_cue(ax)
