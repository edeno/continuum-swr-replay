"""The held-out gain panels: a gain per spike for no-SWR and SWR events.

Figure 2h (the full decoder's gain over the SWR-only decoder) and the held-out
tetrode supplementary figure (the gain over local coding, held out two ways)
draw each gain the same way: a dot per animal in the group's color and, beside
them, the animals' mean with its 95 % interval in black, with a dashed line at
zero, where neither side predicts the spikes better.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from continuum_swr_replay.visualization.palettes import COLORS
from continuum_swr_replay.visualization.simulated_run_panels import SMALL_SIZE

GROUP_COLORS = {"no SWR": COLORS["No SWR"], "SWR": COLORS["SWR"]}
"""Each group's color, in the figures' order: no-SWR events, then SWR events."""

REFERENCE_COLOR = "#777777"
"""A reference level (the zero line): a neutral gray, not the No-Spike state's."""

HELDOUT_X = {"no SWR": 0.0, "SWR": 1.0}
"""Each group's position on a gain axis' x axis."""


def animal_values(row: pd.Series, column: str, scale: float = 1.0) -> pd.Series:
    """A statistics row's per-animal values, indexed by animal."""
    return scale * pd.Series(row[column], index=row["animal_names"])


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
        x = HELDOUT_X[group]
        event = statistics.loc[group]
        animals = animal_values(event, "animals")
        ax.scatter(np.full(animals.size, x), animals, s=6, color=color, lw=0, zorder=2)
        ax.errorbar(
            x + 0.25, event["estimate"],
            yerr=[[event["estimate"] - event["ci_low"]],
                  [event["ci_high"] - event["estimate"]]],
            fmt="_", color="black", ms=5, lw=0.8, capsize=0,
        )  # fmt: skip
    ax.axhline(0, color=REFERENCE_COLOR, lw=0.6, ls="--", zorder=1)
    ax.set_xlim(-0.6, 1.6)
    ax.set_xticks(list(HELDOUT_X.values()), ["No SWR", "SWR"], rotation=45, ha="right")
    ax.set_title(title, fontsize=SMALL_SIZE, pad=3)


def label_gain_axis(ax: Axes, ylabel: str) -> None:
    """The y axis of a gain axis (or of axes sharing it): ``ylabel``, a floor
    below zero and "no gain" under the zero line at the axis' left, clear of
    the groups' dots and intervals."""
    ax.text(
        0.04, -0.02, "no gain", transform=ax.get_yaxis_transform(), va="top",
        fontsize=SMALL_SIZE, color=REFERENCE_COLOR,
    )  # fmt: skip
    ax.set_ylim(-0.12, None)
    ax.set_ylabel(ylabel)
