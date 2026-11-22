"""The held-out gain panels: a gain for each of two kinds of event.

The held-out tetrode supplementary figure (the gain over local coding per spike)
and Figure 2g's second and third axes (the full decoder's gain over the SWR-only
decoder per spike, and the shares of it outside SWRs) draw each value the same
way: a dot per animal in the group's color and, beside them, the animals' mean
with its 95 % interval in black, with a dashed line at zero, where neither side
predicts the spikes better.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from continuum_swr_replay.analysis.event_evidence.state_evidence.windows import (
    SWR_LABELS,
)
from continuum_swr_replay.visualization.palettes import COLORS
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


def draw_animal_gain(ax: Axes, x: float, row: pd.Series, color: str) -> None:
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
    """
    animals = animal_values(row, "animals")
    ax.scatter(np.full(animals.size, x), animals, s=6, color=color, lw=0, zorder=2)
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
