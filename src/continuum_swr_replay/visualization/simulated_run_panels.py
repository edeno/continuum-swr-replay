"""What the figures of the simulated run share: colors, sizes and drawing helpers.

Figure 1 (``scripts/figures/generate_figure01.py``) and the memoryless
comparison (``scripts/figures/generate_supp_memoryless_comparison.py``) draw the same
simulated run (``analysis.detection.simulated_run``) and read as one pair, so
the state colors, the type sizes, the decoded-position panel and the hand
layout (rows spaced from what they draw, letters on measured edges) are defined
once, here.
"""

from collections.abc import Callable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.transforms import Bbox

from continuum_swr_replay.visualization.figure_utilities import WONG
from continuum_swr_replay.visualization.palettes import COLORS

__all__ = [
    "axes_added",
    "check_journal_limits",
    "check_placement",
    "drawn_extent",
    "drawn_texts",
    "example_badge",
    "letter_panel",
    "overlapping_panels",
    "overlapping_texts",
    "plot_posterior",
    "plot_raster",
    "posterior_color_limit",
    "space_rows_evenly",
    "ANIMAL_POS_COLOR",
    "DATA_LEFT",
    "DATA_RIGHT",
    "FIGURE_WIDTH",
    "LABEL_SIZE",
    "LINE_WIDTH_LIMITS_PT",
    "MAX_FIGURE_SIZE_CM",
    "NON_LOCAL_COLOR",
    "NON_LOCAL_SHADE",
    "NON_LOCAL_SHADE_ALPHA",
    "SMALL_SIZE",
    "STATE_COLORS",
    "TEXT_SIZE_LIMITS_PT",
]

# Keyed by the figures' name for each state: the detector's name without its
# "Non-Local " prefix. Local and No-Spike are the palette's (``COLORS``), as in
# every other figure. The palette has no color for the two non-local states:
# CONTFRAG_COLORS' Continuous and Fragmented are the ContFrag classifier's, and
# equal Non-Local and SWR.
STATE_COLORS = {
    "Local": COLORS["Local"],
    "No-Spike": COLORS["No-Spike"],
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


def set_publication_style() -> None:
    """The paper figures' matplotlib style: Cell Press text sizes, thin axes and
    ticks, no top or right spines, and no layout engine (the figures place
    their own axes)."""
    plt.rcParams.update(
        {
            "font.size": SMALL_SIZE,
            "axes.labelsize": LABEL_SIZE,
            "axes.titlesize": LABEL_SIZE,
            "xtick.labelsize": SMALL_SIZE,
            "ytick.labelsize": SMALL_SIZE,
            "legend.fontsize": SMALL_SIZE,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "lines.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.constrained_layout.use": False,
        }
    )


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
        Top of the color scale.
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


MAX_FIGURE_SIZE_CM = (17.4, 20.0)
"""Cell Press's widest and tallest figure: a full page width, and a page's height
less room for the legend."""
TEXT_SIZE_LIMITS_PT = (6.0, 8.0)
"""Cell Press's smallest and largest text in a figure at its printed size."""
LINE_WIDTH_LIMITS_PT = (0.5, 1.5)
"""Cell Press's thinnest and thickest line."""


def check_journal_limits(fig: Figure) -> None:
    """Stop rather than save a figure outside Cell Press's figure guidelines.

    Checks the size of the figure as it is saved (cropped to what it draws,
    with the saving margin), the size of every text and the width of every
    line, axis spine and tick. Panel letters, fonts and colors are not
    checked here.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The finished figure.

    Raises
    ------
    ValueError
        If the saved figure is wider or taller than :data:`MAX_FIGURE_SIZE_CM`,
        a text is outside :data:`TEXT_SIZE_LIMITS_PT`, or a line is outside
        :data:`LINE_WIDTH_LIMITS_PT`.
    """
    fig.draw_without_rendering()
    problems = []

    saved = fig.get_tightbbox()
    margin = 2 * float(plt.rcParams["savefig.pad_inches"])
    size_cm = ((saved.width + margin) * 2.54, (saved.height + margin) * 2.54)
    if size_cm[0] > MAX_FIGURE_SIZE_CM[0] or size_cm[1] > MAX_FIGURE_SIZE_CM[1]:
        problems.append(
            f"saved size {size_cm[0]:.2f} x {size_cm[1]:.2f} cm is over "
            f"{MAX_FIGURE_SIZE_CM[0]} x {MAX_FIGURE_SIZE_CM[1]} cm"
        )

    low, high = TEXT_SIZE_LIMITS_PT
    for text in fig.findobj(Text):
        if text.get_visible() and text.get_text().strip():
            if not low <= text.get_fontsize() <= high:
                problems.append(
                    f"text {text.get_text()!r} is {text.get_fontsize():g} pt"
                )

    low, high = LINE_WIDTH_LIMITS_PT
    widths: list[tuple[str, float]] = []
    for line in fig.findobj(Line2D):
        drawn = line.get_linestyle() not in ("None", "", " ")
        if line.get_visible() and drawn and np.size(line.get_xdata()):
            widths.append((f"line {line.get_label()!r}", line.get_linewidth()))
    for collection in fig.findobj(LineCollection):
        if collection.get_visible() and len(collection.get_segments()):
            widths += [
                ("line collection", float(width))
                for width in np.atleast_1d(collection.get_linewidths())
            ]
    for ax in fig.axes:
        widths += [
            (f"{name} spine", spine.get_linewidth())
            for name, spine in ax.spines.items()
            if spine.get_visible()
        ]
    problems += [
        f"{what} is {width:g} pt wide"
        for what, width in widths
        if width > 0 and not low <= width <= high
    ]

    if problems:
        raise ValueError(
            "The figure is outside Cell Press's figure guidelines: "
            + "; ".join(sorted(set(problems)))
        )


def posterior_color_limit(posterior: np.ndarray, time: np.ndarray, events) -> float:
    """Top of a decoded-position color scale: the posterior's 99th percentile
    inside ``events``.

    Outside an event the posterior sits on the animal and is more concentrated;
    it would set a scale on which an event is faint, so on this scale it
    saturates.

    Parameters
    ----------
    posterior : ndarray, shape (n_time, n_position)
    time : ndarray, shape (n_time,)
        The posterior's time axis, in seconds.
    events : iterable of (float, float)
        Start and end of each event, in seconds; both ends are inside it.
    """
    inside = np.zeros(time.size, dtype=bool)
    for start, end in events:
        inside |= (time >= start) & (time <= end)
    return float(np.percentile(posterior[inside], 99))


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


def drawn_texts(fig: Figure) -> list[Text]:
    """The text artists a draw of ``fig`` paints, in drawing order.

    Not every text a figure holds is drawn: a tick label whose tick lies outside
    the view is not. This draws the figure and keeps the texts the renderer is
    handed.
    """
    renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]
    drawn: dict[int, Text] = {}
    draw_text = renderer.draw_text

    def record(gc, x, y, s, prop, angle, ismath=False, mtext=None):
        if mtext is not None and s.strip():
            drawn.setdefault(id(mtext), mtext)
        return draw_text(gc, x, y, s, prop, angle, ismath=ismath, mtext=mtext)

    renderer.draw_text = record
    try:
        fig.draw(renderer)
    finally:
        del renderer.draw_text  # the class's method again
    return list(drawn.values())


def _text_outline(text: Text) -> np.ndarray:
    """The four corners of a text's own rectangle, in display pixels.

    An upright text's rectangle is its window extent. A slanted text's extent is
    the upright box around its slanted rectangle, which reaches well past the
    letters; the rectangle itself has the text's unrotated size, the extent's
    center and the text's angle.
    """
    extent = text.get_window_extent()
    angle = np.deg2rad(text.get_rotation())
    width, height = extent.width, extent.height
    if not np.isclose(np.sin(angle), 0.0):
        rotation = text.get_rotation()
        text.set_rotation(0)
        flat = text.get_window_extent()
        text.set_rotation(rotation)
        width, height = flat.width, flat.height
    elif not np.isclose(np.cos(angle), 1.0):
        angle = 0.0  # upside down: the same rectangle
    corners = 0.5 * np.array(
        [[-width, -height], [width, -height], [width, height], [-width, height]]
    )
    turn = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    center = np.array([0.5 * (extent.x0 + extent.x1), 0.5 * (extent.y0 + extent.y1)])
    return corners @ turn.T + center


def _rectangles_overlap(first: np.ndarray, second: np.ndarray, by: float) -> bool:
    """Whether two rectangles (corners, in order) overlap by more than ``by``
    along every direction: no side of either separates them."""
    for corners in (first, second):
        for side in np.roll(corners, -1, axis=0) - corners:
            normal = np.array([-side[1], side[0]]) / np.hypot(*side)
            a, b = first @ normal, second @ normal
            if min(a.max(), b.max()) - max(a.min(), b.min()) <= by:
                return False
    return True


TEXT_OVERLAP_PT = 0.5
"""Two texts overlap when their rectangles do by more than this, in points: a
text's rectangle has a little room around its letters."""


def overlapping_texts(fig: Figure) -> list[tuple[str, str]]:
    """Each pair of drawn texts of ``fig`` whose rectangles overlap.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The finished figure; it is drawn.

    Returns
    -------
    pairs : list of (str, str)
        The two texts of each pair, in drawing order. A slanted text is its own
        slanted rectangle, not the upright box around it.
    """
    texts = drawn_texts(fig)
    outlines = [_text_outline(text) for text in texts]
    by = TEXT_OVERLAP_PT * fig.dpi / 72.0
    return [
        (texts[i].get_text(), texts[j].get_text())
        for i in range(len(texts))
        for j in range(i + 1, len(texts))
        if _rectangles_overlap(outlines[i], outlines[j], by)
    ]


def overlapping_panels(
    fig: Figure, panels: Mapping[str, Sequence[Axes]]
) -> list[tuple[str, str]]:
    """Each pair of panels whose drawings (labels included) overlap.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The finished figure, already drawn.
    panels : Mapping[str, Sequence[Axes]]
        Each panel's name and its axes.

    Returns
    -------
    pairs : list of (str, str)
        The names of each pair whose :func:`drawn_extent` boxes overlap.
    """
    extents = {name: drawn_extent(fig, axes) for name, axes in panels.items()}
    names = list(extents)
    return [
        (names[i], names[j])
        for i in range(len(names))
        for j in range(i + 1, len(names))
        if extents[names[i]].overlaps(extents[names[j]])
    ]


def check_placement(fig: Figure, panels: Mapping[str, Sequence[Axes]]) -> None:
    """Stop rather than save a figure in which a text runs into another or a
    panel's drawing into another panel's.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The finished figure.
    panels : Mapping[str, Sequence[Axes]]
        Each panel's name and its axes.

    Raises
    ------
    ValueError
        Naming each pair of overlapping texts (:func:`overlapping_texts`) and of
        overlapping panels (:func:`overlapping_panels`).
    """
    problems = [
        f"the texts {first!r} and {second!r} overlap"
        for first, second in overlapping_texts(fig)
    ]
    problems += [
        f"panels {first} and {second} overlap"
        for first, second in overlapping_panels(fig, panels)
    ]
    if problems:
        raise ValueError(
            "The figure's parts run into each other: " + "; ".join(problems)
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
