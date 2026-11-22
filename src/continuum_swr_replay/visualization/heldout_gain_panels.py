"""The held-out gain panels: a held-out gain per animal and its mean.

Figure 2j (each decoder's gain over the local-only decoder, and the full
decoder's over the SWR-only decoder) and Figure 2's supplement
(``scripts/figures/generate_supp_figure02.py``: the held-out tetrode gain over
local coding per spike; the full decoder's gain outside SWRs by distance from
the SWRs) draw each value the
same way: a dot per animal and, beside them, the animals' mean with its 95 %
interval in black, with a dashed line at zero, where neither side predicts the
spikes better.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.transforms import blended_transform_factory

from continuum_swr_replay.analysis.event_evidence.state_evidence.windows import (
    SWR_LABELS,
)
from continuum_swr_replay.visualization.palettes import COLORS, readable_text_color
from continuum_swr_replay.visualization.simulated_run_panels import SMALL_SIZE

GROUP_COLORS = {SWR_LABELS[False]: COLORS["No SWR"], SWR_LABELS[True]: COLORS["SWR"]}
"""Each group's color, in the figures' order: no-SWR events, then SWR events."""

REFERENCE_COLOR = "#777777"
"""A reference level (the zero line): a neutral gray, not the No-Spike state's."""


def animal_values(row: pd.Series, column: str, scale: float = 1.0) -> pd.Series:
    """A statistics row's per-animal values, indexed by animal."""
    return scale * pd.Series(row[column], index=row["animal_names"])


MEAN_OFFSET = 0.25
"""How far right of a group's dots its mean is drawn, in x units."""

DOT_SPREAD = 0.05
"""The sideways step between an animal's dot and the next (x units), when a
group's dots are spread so that each shows (:func:`spread_offsets`)."""

NEUTRAL_DOTS = "#444444"
"""A dark gray for dots that need no class's color (Figure 2j's differences,
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


SPLIT_LABELS = {
    "gain_per_spike": "All bins",
    "spiking_gain_per_spike": "Bins with\na spike",
    "silent_gain_per_spike": "Silent\nbins",
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
            split_x(k, 1) + MEAN_OFFSET / 2, 1.0, "No SWR" if k == 0 else "SWR",
            transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=SMALL_SIZE, color=readable_text_color(color),
        )  # fmt: skip
    ax.axhline(0, color=REFERENCE_COLOR, lw=0.6, ls="--", zorder=1)
    ax.set_xlim(-0.5, split_x(1, len(SPLIT_LABELS) - 1) + 0.6)
    ax.set_xticks(ticks, labels, rotation=90)
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


GAIN_FLOOR = 0.13
"""A gain axis' floor below zero, as a share of its top (Figure 2j, Figure 2's
supplement, panel j)."""

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
