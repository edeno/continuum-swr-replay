"""What the figures of the simulated run share: colours, sizes and drawing helpers.

Figure 1 (``scripts/figures/generate_figure01.py``) and the memoryless
comparison (``scripts/figures/generate_supp_memoryless_comparison.py``) draw the same
simulated run (``analysis.detection.simulated_run``) and read as one pair, so
the state colours, the type sizes, the decoded-position panel and the hand
layout (rows spaced from what they draw, letters on measured edges) are defined
once, here.
"""

from collections.abc import Callable, Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox

from continuum_swr_replay.visualization.figure_utilities import WONG
from continuum_swr_replay.visualization.palettes import COLORS

__all__ = [
    "ANIMAL_POS_COLOR",
    "DATA_LEFT",
    "DATA_RIGHT",
    "FIGURE_WIDTH",
    "LABEL_SIZE",
    "NON_LOCAL_COLOR",
    "NON_LOCAL_SHADE",
    "NON_LOCAL_SHADE_ALPHA",
    "PANEL_LETTER_SIZE",
    "POSITION_MARGIN_CM",
    "SMALL_SIZE",
    "STATE_COLORS",
    "axes_added",
    "drawn_extent",
    "example_badge",
    "letter_panel",
    "plot_posterior",
    "plot_raster",
    "shift_vertically",
    "space_rows_evenly",
]

# Keyed by the figures' name for each state: the detector's name without its
# "Non-Local " prefix.
STATE_COLORS = {
    "Local": WONG[5],
    "No-Spike": "#999999",
    "Continuous": WONG[1],
    "Fragmented": WONG[6],
}
NON_LOCAL_COLOR = COLORS["Non-Local"]
NON_LOCAL_SHADE_ALPHA = 0.13
# The shading of a non-local event.
NON_LOCAL_SHADE = {
    "color": NON_LOCAL_COLOR,
    "alpha": NON_LOCAL_SHADE_ALPHA,
    "linewidth": 0,
}
ANIMAL_POS_COLOR = "magenta"

# Cell Press asks for text of about 6-8 pt at print size.
LABEL_SIZE = 7
SMALL_SIZE = 6
PANEL_LETTER_SIZE = 8

# Inches. ``save_figure`` crops to the drawing and adds 0.1 in on each side, so
# this saves at about 17.3 cm: within Cell Press's 17.4 cm full width.
FIGURE_WIDTH = 6.9
# The data axes of the panels that span the figure, in figure fractions.
DATA_LEFT, DATA_RIGHT = 0.095, 0.985
# Space shown beyond each end of the track in the decoded-position plots.
POSITION_MARGIN_CM = 8.0


def example_badge(color: str) -> dict:
    """The ``bbox`` of a numbered example flag."""
    return {"boxstyle": "circle,pad=0.18", "facecolor": color, "edgecolor": "none"}


def plot_raster(
    ax: Axes,
    spikes: Sequence[np.ndarray],
    *,
    linewidth: float,
    rasterized: bool = False,
) -> None:
    """Spikes of each cell on its own row, the cells numbered from 1.

    Parameters
    ----------
    ax : Axes
    spikes : sequence of np.ndarray
        Each cell's spike times, in the units of the x axis.
    linewidth : float
        Of a spike's tick, in points.
    rasterized : bool, optional
        Draw the ticks as an image, for a raster of many spikes.
    """
    n_neurons = len(spikes)
    ax.eventplot(
        spikes,
        lineoffsets=np.arange(1, n_neurons + 1),
        linelengths=0.85,
        colors="black",
        linewidths=linewidth,
        **({"rasterized": True} if rasterized else {}),
    )
    ax.set_ylim(0.3, n_neurons + 0.7)
    ax.set_yticks([1, n_neurons])


def plot_posterior(
    ax: Axes,
    posterior: np.ndarray,
    position_bins: np.ndarray,
    time: np.ndarray,
    animal_position: np.ndarray,
    *,
    vmax: float,
    linewidth: float,
    dashes: tuple[float, float] | None = None,
) -> None:
    """Posterior over position with the animal's position on top.

    Parameters
    ----------
    ax : Axes
    posterior : np.ndarray, shape (n_time, n_position)
    position_bins : np.ndarray, shape (n_position,)
        Bin centers, in cm, evenly spaced.
    time : np.ndarray, shape (n_time,)
        The x coordinate of each row, on an even grid.
    animal_position : np.ndarray, shape (n_animal,) or (n_time,)
        The animal's position, in cm, drawn against ``time`` when it has one
        value per row.
    vmax : float
        Top of the colour scale.
    linewidth : float
        Of the animal's position, in points.
    dashes : tuple of float, optional
        Dash pattern of the animal's position; solid when None.
    """
    time_step = time[1] - time[0]
    bin_size = position_bins[1] - position_bins[0]
    # Every decoded bin is given to the renderer, which averages the bins that
    # share a pixel ("antialiased"); none is dropped.
    ax.imshow(
        posterior.T,
        origin="lower",
        aspect="auto",
        extent=(
            time[0] - time_step / 2,
            time[-1] + time_step / 2,
            position_bins[0] - bin_size / 2,
            position_bins[-1] + bin_size / 2,
        ),
        cmap="bone_r",
        vmin=0,
        vmax=vmax,
        interpolation="antialiased",
        rasterized=True,
    )
    (line,) = ax.plot(
        time, animal_position, color=ANIMAL_POS_COLOR, linewidth=linewidth
    )
    if dashes:
        line.set_dashes(dashes)
    # Room beyond the track's ends: the animal pauses at an end, and its position
    # would otherwise lie on the edge of the axes and be clipped.
    ax.set_ylim(
        position_bins[0] - POSITION_MARGIN_CM,
        position_bins[-1] + POSITION_MARGIN_CM,
    )


def axes_added(fig: Figure, draw: Callable[[], None]) -> list[Axes]:
    """Run ``draw`` and return the axes it added to ``fig``, in order."""
    before = set(fig.axes)
    draw()
    return [ax for ax in fig.axes if ax not in before]


def drawn_extent(fig: Figure, axes: Sequence[Axes]) -> Bbox:
    """Bounding box of everything ``axes`` draw (labels included), in figure fractions."""
    return Bbox.union([ax.get_tightbbox() for ax in axes]).transformed(
        fig.transFigure.inverted()
    )


def shift_vertically(axes: Sequence[Axes], dy: float) -> None:
    """Move ``axes`` up by ``dy``, a figure fraction."""
    for ax in axes:
        box = ax.get_position(original=True)
        ax.set_position([box.x0, box.y0 + dy, box.width, box.height])


def space_rows_evenly(
    fig: Figure, rows: Sequence[Sequence[Axes]], margin: float = 0.012
) -> list[Bbox]:
    """Move rows of axes so the gaps between what they draw are equal.

    The first row's drawing ends ``margin`` below the figure's top and the last
    row's ``margin`` above its bottom; the figure must already be drawn.

    Parameters
    ----------
    fig : Figure
    rows : sequence of sequences of Axes
        Top row first.
    margin : float, optional
        A figure fraction.

    Returns
    -------
    list of Bbox
        Each row's drawn extent after the move, in figure fractions.
    """
    extents = [drawn_extent(fig, row) for row in rows]
    gap = (1 - 2 * margin - sum(extent.height for extent in extents)) / (len(rows) - 1)
    row_top = 1 - margin
    for row, extent in zip(rows, extents, strict=True):
        shift_vertically(row, row_top - extent.y1)
        row_top -= extent.height + gap
    fig.draw_without_rendering()
    return [drawn_extent(fig, row) for row in rows]


def letter_panel(fig: Figure, letter: str, x: float, y: float) -> None:
    """Write a panel letter with its top left corner at ``(x, y)``, figure fractions."""
    fig.text(
        x,
        y,
        letter,
        fontsize=PANEL_LETTER_SIZE,
        fontweight="bold",
        ha="left",
        va="top",
    )
