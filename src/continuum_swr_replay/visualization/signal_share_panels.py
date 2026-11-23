"""Where the non-local time sits in a signal: Figure 2h (ripple power) and its
MUA counterpart on Figure 2's supplement draw it the same way.

Top, each animal's non-local time bins spread over the signal (the percent of
them in each signal bin), the part below the threshold shaded under the
animals' mean and its share written; bottom, the share of immobile bins decoded
non-local at each level. One thin line per animal, their mean in bold, a dashed
line at the threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from continuum_swr_replay.visualization.simulated_run_panels import SMALL_SIZE


def draw_share_by_signal(
    ax_spread: Axes,
    ax_share: Axes,
    counts: pd.DataFrame,
    *,
    threshold: float,
    threshold_label: str,
    threshold_color: str,
    shade_color: str,
    xlabel: str,
    below_note: str,
) -> None:
    """Where the non-local time sits in a signal.

    Top, each animal's non-local time bins spread over the signal, the part
    below ``threshold`` shaded under the animals' mean and ``below_note``
    written (the figure's script makes it from
    ``event_statistics.nonlocal_below_threshold_statistics``, which counts the
    bins beyond the axis too). Bottom, the share of immobile bins decoded
    non-local at each level, with ``threshold_label`` beside the threshold's
    line.

    Parameters
    ----------
    ax_spread, ax_share : Axes
    counts : pd.DataFrame
        ``event_statistics.animal_signal_counts`` of the stored per-epoch counts.
    threshold : float
        z; an edge of the signal's bins.
    threshold_label, threshold_color, shade_color, xlabel : str
    below_note : str
        The share below the threshold, in words.
    """
    drawn = counts[
        np.isfinite(counts["signal_low"]) & np.isfinite(counts["signal_high"])
    ]
    width = (drawn["signal_high"] - drawn["signal_low"]).iloc[0]
    for ax, column in ((ax_spread, "nonlocal_fraction"), (ax_share, "share")):
        by_animal = 100 * drawn.pivot(
            index="signal_low", columns="animal", values=column
        )
        center = by_animal.index + width / 2
        mean = by_animal.mean(axis=1)
        ax.plot(center, by_animal.to_numpy(), color="k", lw=0.5, alpha=0.5)
        ax.plot(center, mean, color="k", lw=1.0)
        ax.axvline(threshold, color=threshold_color, lw=0.8, ls="--")
        ax.set_xlim(drawn["signal_low"].min(), drawn["signal_high"].max())
        if ax is ax_spread:
            ax.fill_between(
                center, mean, where=center < threshold, color=shade_color, lw=0
            )
    ax_spread.text(
        0.98, 0.95, below_note, transform=ax_spread.transAxes, ha="right",
        va="top", fontsize=SMALL_SIZE,
    )  # fmt: skip
    ax_spread.set_ylim(0, None)
    ax_spread.set_ylabel(f"% per\n{width:g} z")
    ax_spread.set_title("Non-local time", fontsize=SMALL_SIZE, pad=3)
    ax_spread.tick_params(labelbottom=False)
    ax_share.text(
        threshold + 0.3, 5, threshold_label, fontsize=SMALL_SIZE,
        color=threshold_color, va="bottom",
    )  # fmt: skip
    ax_share.set_ylim(0, 102)
    ax_share.set_yticks([0, 50, 100])
    ax_share.set_xlabel(xlabel)
    ax_share.set_ylabel("% of bins\nnon-local")
