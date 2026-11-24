"""Where the non-local time sits in a signal: Figure 2i (ripple power) and its
MUA counterpart on Figure 2's supplement draw it the same way.

Top, how confident Local time and confident non-local time are spread over the
signal (each class's percent of its time in each signal bin; the mean over
animals, ``shared_statistics.animal_mean``, of each animal's mean over its
epochs, in the class's color), the part
of the non-local curve below the threshold shaded and the share of non-local
time below it written; bottom, the share of immobile bins decoded non-local at
each level, one thin line per animal and their mean in bold. A dashed line at
the threshold on both.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea

from continuum_swr_replay.analysis.shared_statistics import animal_mean
from continuum_swr_replay.visualization.palettes import COLORS, readable_text_color
from continuum_swr_replay.visualization.publication_style import SMALL_SIZE

STATE_STYLES = {
    "local": ("Local", COLORS["Local"]),
    "non_local": ("Non-local", COLORS["Non-Local"]),
}
"""Each state class of the top strip: its name and color."""
STRIP_TITLE = (
    ("Local", "local"),
    ("and", None),
    ("non-local", "non_local"),
    ("time", None),
)
"""The top strip's title, word by word, with the state class whose color a
word takes (the strip's two curves are named there, not on the curves)."""


def draw_strip_title(ax: Axes) -> None:
    """:data:`STRIP_TITLE` centered above ``ax``, each class's name in its color."""
    words = [
        TextArea(
            word,
            textprops={
                "fontsize": SMALL_SIZE,
                "color": "black"
                if state is None
                else readable_text_color(STATE_STYLES[state][1]),
            },
        )
        for word, state in STRIP_TITLE
    ]
    ax.add_artist(
        AnchoredOffsetbox(
            loc="lower center",
            child=HPacker(children=words, pad=0, sep=2, align="baseline"),
            frameon=False,
            pad=0.0,
            borderpad=0.25,
            bbox_to_anchor=(0.5, 1.0),
            bbox_transform=ax.transAxes,
        )
    )


SHADE_COLOR = COLORS["All Events"]
"""The non-local curve's part below the threshold, shaded."""


def draw_share_by_signal(
    ax_spread: Axes,
    ax_share: Axes,
    counts: pd.DataFrame,
    *,
    threshold: float,
    threshold_label: str,
    threshold_color: str,
    xlabel: str,
    below_note: str,
    states: pd.DataFrame,
) -> None:
    """Where the non-local time sits in a signal, against the Local time.

    Top, confident Local and confident non-local time spread over the signal
    (``states``), each the mean over animals in the class's color, the classes
    named in their colors in the strip's title (:func:`draw_strip_title`), the
    non-local curve's part below ``threshold`` shaded and ``below_note``
    written in the non-local color (the figure's script makes it from
    ``event_statistics.nonlocal_below_threshold_statistics``, which counts the
    bins beyond the axis too). Bottom, the share of immobile bins decoded
    non-local at each level, with ``threshold_label`` beside the threshold's
    line.

    Parameters
    ----------
    ax_spread, ax_share : Axes
    counts : pd.DataFrame
        ``event_statistics.animal_signal_counts`` of the stored per-epoch counts
        (the bottom axis).
    threshold : float
        z; an edge of the signal's bins.
    threshold_label, threshold_color, xlabel : str
    below_note : str
        The share below the threshold, in words.
    states : pd.DataFrame
        ``state_by_signal.state_distribution`` of the signal: ``state``,
        ``animal``, ``signal_low``, ``signal_high`` and ``percent``.
    """
    drawn = counts[
        np.isfinite(counts["signal_low"]) & np.isfinite(counts["signal_high"])
    ]
    width = (drawn["signal_high"] - drawn["signal_low"]).iloc[0]
    percent = drawn.assign(percent=100 * drawn["share"])
    by_animal = percent.pivot(index="signal_low", columns="animal", values="percent")
    center = by_animal.index + width / 2
    ax_share.plot(center, by_animal.to_numpy(), color="k", lw=0.5, alpha=0.5)
    ax_share.plot(
        center, animal_mean(percent, "signal_low", "percent"), color="k", lw=1.0
    )
    finite = states[
        np.isfinite(states["signal_low"]) & np.isfinite(states["signal_high"])
    ]
    for state, (_, color) in STATE_STYLES.items():
        mean = animal_mean(finite[finite["state"] == state], "signal_low", "percent")
        middle = mean.index + width / 2
        ax_spread.plot(middle, mean.to_numpy(), color=color, lw=1.0)
        if state == "non_local":
            ax_spread.fill_between(
                middle, mean.to_numpy(), where=middle < threshold, color=SHADE_COLOR,
                lw=0,
            )  # fmt: skip
    for ax in (ax_spread, ax_share):
        ax.axvline(threshold, color=threshold_color, lw=0.8, ls="--")
        ax.set_xlim(drawn["signal_low"].min(), drawn["signal_high"].max())
    ax_spread.text(
        0.98, 0.95, below_note, transform=ax_spread.transAxes, ha="right",
        va="top", fontsize=SMALL_SIZE,
        color=readable_text_color(STATE_STYLES["non_local"][1]),
    )  # fmt: skip
    ax_spread.set_ylim(0, None)
    ax_spread.set_ylabel(f"% per\n{width:g} z")
    draw_strip_title(ax_spread)
    ax_spread.tick_params(labelbottom=False)
    ax_share.text(
        threshold + 0.3, 5, threshold_label, fontsize=SMALL_SIZE,
        color=threshold_color, va="bottom",
    )  # fmt: skip
    ax_share.set_ylim(0, 102)
    ax_share.set_yticks([0, 50, 100])
    ax_share.set_xlabel(xlabel)
    ax_share.set_xticks([0, 5, 10])
    ax_share.set_ylabel("% of bins\nnon-local")
