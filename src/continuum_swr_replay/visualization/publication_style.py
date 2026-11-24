"""The paper figures' style and the checks a figure passes before it is saved.

The type sizes and the matplotlib style every paper figure is drawn with
(:func:`set_publication_style`), Cell Press's limits on a figure's size, text
and lines (:func:`check_journal_limits`), and the check that no text runs into
another and no panel into another (:func:`check_placement`), with the drawn
extents both rest on.
"""

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.transforms import Bbox

__all__ = [
    "check_journal_limits",
    "check_placement",
    "drawn_extent",
    "drawn_texts",
    "overlapping_panels",
    "overlapping_texts",
    "set_publication_style",
    "LABEL_SIZE",
    "LINE_WIDTH_LIMITS_PT",
    "MAX_FIGURE_SIZE_CM",
    "PANEL_LETTER_SIZE",
    "SMALL_SIZE",
    "TEXT_OVERLAP_PT",
    "TEXT_SIZE_LIMITS_PT",
]

# Cell Press asks for text of about 6-8 pt at print size.
LABEL_SIZE = 7
SMALL_SIZE = 6
PANEL_LETTER_SIZE = 8


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
