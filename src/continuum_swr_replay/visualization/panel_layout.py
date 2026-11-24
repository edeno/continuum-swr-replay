"""Where the panels of a multi-panel figure and their letters sit.

A figure states its layout once, as rows of panels, each panel with the left
and right edges it spans (:func:`panel_spans`); the panels' axes are fitted
between those edges (:func:`fit_panel`), and the letters' rows and columns
follow from the same rows (:func:`letter_layout`, :func:`place_letters`).

The letters follow convention 10 of ``docs/development.md``: the letters of a
row sit at one height, just above the highest of the row's drawn tops, and the
letters of a column at one left edge, the leftmost of the column's drawn
extents. Every extent is measured on the drawn figure (``drawn_extent``), so
the figure must be drawn (``fig.draw_without_rendering()``) before a letter is
placed.
"""

from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any, Literal

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib.transforms import Bbox

from continuum_swr_replay.visualization.publication_style import (
    PANEL_LETTER_SIZE,
    drawn_extent,
)

__all__ = [
    "FIT_PASSES",
    "FIT_TOLERANCE_PT",
    "fit_panel",
    "letter_layout",
    "panel_spans",
    "place_letters",
]

FIT_PASSES = 10
"""The most passes :func:`fit_panel` makes: what a panel draws beside its
frames can change as the frames move, so the frames settle over a few passes."""
FIT_TOLERANCE_PT = 0.01
"""The frames have settled when no frame edge moves by more than this in a
pass, in points."""

Edge = str | float
"""A panel's left or right edge in a layout: a column's name or a figure
fraction."""


def panel_spans(
    rows: Sequence[Sequence[tuple[str, Edge, Edge]]],
    edges: Mapping[str, float],
    gap: float = 0.0,
) -> dict[str, tuple[float, float]]:
    """Each panel's left and right edges, from a layout's rows.

    A row is its panels left to right, each ``(panel, left, right)``. A named
    left edge is that column's edge; a named right edge is ``gap`` before that
    column's edge, so the panel ends a gap before the panel that starts there;
    a number is the edge itself.

    Parameters
    ----------
    rows : sequence of sequence of (str, str or float, str or float)
    edges : Mapping[str, float]
        Each column's left edge (figure width).
    gap : float, optional
        Figure width.

    Returns
    -------
    dict[str, tuple[float, float]]
        Each panel's left and right edges (figure width), in the layout's order.
    """

    def edge(value: Edge, before: float) -> float:
        return edges[value] - before if isinstance(value, str) else float(value)

    return {
        panel: (edge(left, 0.0), edge(right, gap))
        for row in rows
        for panel, left, right in row
    }


def letter_layout(
    rows: Sequence[Sequence[tuple[str, Edge, Edge]]], letters: Mapping[str, str]
) -> tuple[list[str], dict[Edge, str]]:
    """The letters' rows and columns, from a layout's rows (:func:`panel_spans`).

    Parameters
    ----------
    rows : sequence of sequence of (str, str or float, str or float)
    letters : Mapping[str, str]
        Each panel's letter.

    Returns
    -------
    rows : list of str
        The letters of each row, left to right.
    columns : dict
        By left edge, the letters of the panels that start there, top to bottom.
    """
    columns: dict[Edge, str] = {}
    for row in rows:
        for panel, left, _ in row:
            columns[left] = columns.get(left, "") + letters[panel]
    return ["".join(letters[panel] for panel, _, _ in row) for row in rows], columns


def _beside_frame(ax: Axes, renderer: Any) -> tuple[float, float]:
    """How far what ``ax`` draws reaches left and right of its frame, in display
    pixels.

    Its axes (tick labels and labels) and every other artist it holds count,
    except what sits wholly above the frame (titles, headings, keys), which may
    reach over the space between panels.
    """
    frame = ax.get_window_extent(renderer)
    boxes = [frame]
    if ax.axison:
        boxes += [axis.get_tightbbox(renderer) for axis in (ax.xaxis, ax.yaxis)]
    for artist in ax.get_default_bbox_extra_artists():
        box = artist.get_tightbbox(renderer)
        if box is not None and box.width > 0 and box.height > 0 and box.y0 < frame.y1:
            boxes.append(box)
    drawn = Bbox.union([box for box in boxes if box is not None])
    return frame.x0 - drawn.x0, drawn.x1 - frame.x1


def fit_panel(
    fig: Figure,
    columns: Sequence[Sequence[Axes]],
    left: float,
    right: float,
    *,
    weights: Sequence[float] | None = None,
    gap: float = 0.0,
    between: Literal["frames", "drawings"] = "frames",
) -> None:
    """Place a panel's columns of axes so that what they draw spans ``left``
    to ``right``.

    The columns stand side by side, left to right; the axes of a column share
    its frame's left edge and width, and keep their heights. Each column takes
    the room its axes draw beside their frames (their tick labels, axis labels
    and the texts beside them, not the titles and keys above them), measured on
    the drawn figure, and the frames share the rest in proportion to
    ``weights``. Between two columns is ``gap``: with ``between="frames"`` a
    share of the frames' mean width (as matplotlib's ``wspace``), from one
    frame to the next, the inner columns' labels sitting in it; with
    ``between="drawings"`` a figure width, from one column's drawing to the
    next's. What a column draws can change as its frame moves (a label placed
    in its axes' coordinates), so the frames are placed again until no edge
    moves by more than :data:`FIT_TOLERANCE_PT`, at most :data:`FIT_PASSES`
    times.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    columns : sequence of sequence of Axes
        The panel's columns, left to right, each its axes.
    left, right : float
        Where the panel's drawing starts and ends (figure width).
    weights : sequence of float, optional
        The frames' widths, in proportion; equal when not given.
    gap : float, optional
    between : {"frames", "drawings"}, optional

    Raises
    ------
    ValueError
        If the frames have not settled after :data:`FIT_PASSES` passes.
    """
    weights = [1.0] * len(columns) if weights is None else list(weights)
    share = [weight / sum(weights) for weight in weights]
    n = len(columns)
    figure_width = fig.get_figwidth() * fig.dpi
    tolerance = FIT_TOLERANCE_PT * fig.dpi / 72 / figure_width
    for _ in range(FIT_PASSES):
        fig.draw_without_rendering()
        renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]
        pads = [
            [max(pad) / figure_width for pad in zip(*sides, strict=True)]
            for sides in (
                [_beside_frame(ax, renderer) for ax in col] for col in columns
            )
        ]
        if between == "drawings":
            spaces = [
                after + gap + before
                for (_, after), (before, _) in zip(pads, pads[1:], strict=False)
            ]
            frames = right - left - pads[0][0] - pads[-1][1] - sum(spaces)
        else:
            frames = (right - left - pads[0][0] - pads[-1][1]) / (1 + gap * (n - 1) / n)
            spaces = [gap * frames / n] * (n - 1)
        x = left + pads[0][0]
        moved = 0.0
        for k, column in enumerate(columns):
            width = frames * share[k]
            for ax in column:
                box = ax.get_position()
                moved = max(moved, abs(box.x0 - x), abs(box.x1 - (x + width)))
                ax.set_position((x, box.y0, width, box.height))
            x += width + (spaces[k] if k < n - 1 else 0.0)
        if moved <= tolerance:
            return
    raise ValueError(
        f"The panel's frames did not settle in {FIT_PASSES} passes: the last moved "
        f"an edge by {moved * figure_width * 72 / fig.dpi:.3g} pt"
    )


def place_letters(
    fig: Figure,
    panels: Mapping[str, Sequence[Axes]],
    rows: Sequence[str] = (),
    columns: Sequence[str] = (),
    *,
    above: float,
    left_of: float = 0.0,
    ceiling: float | None = None,
    floor: float | None = 0.0,
    va: Literal["top", "bottom"] = "top",
    heights: Mapping[str, Sequence[Axes] | float] | None = None,
) -> dict[str, Text]:
    """Write each panel's letter, aligned by row and by column.

    The letters of a row (each string of ``rows``, such as ``"cdef"``) sit
    ``above`` the highest of their panels' drawn tops, at most at ``ceiling``;
    the letters of a column (each string of ``columns``, such as ``"dh"``) sit
    ``left_of`` the leftmost of their panels' drawn extents, at least at
    ``floor``. A letter in no row or no column is placed by its own
    panel. The figure must already be drawn.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    panels : Mapping[str, Sequence[Axes]]
        Each letter's axes; the letters are written in this order.
    rows, columns : sequence of str
        The letters that share a height, and the letters that share a left
        edge.
    above : float
        From a row's highest drawn top to its letters' ``va`` edge (figure
        height).
    left_of : float, optional
        From a column's leftmost drawn extent to its letters' left edge (figure
        width).
    ceiling : float, optional
        The highest a row's letters may sit (figure height); no limit when not
        given.
    floor : float or None, optional
        The leftmost a column's letters may sit (figure width): by default the
        figure's left edge; no limit when None.
    va : {"top", "bottom"}, optional
        The edge of each letter its height places.
    heights : Mapping[str, Sequence[Axes] | float], optional
        By letter, which part of its panel sets its row's height in place of the
        panel's whole drawn extent: the axes whose drawn tops count (none: the
        panel does not count, as when a legend above it would lift the row's
        letters away from the others), or the letter's own height (figure
        height).

    Returns
    -------
    dict[str, matplotlib.text.Text]
        Each letter's text, in the order of ``panels``.
    """
    heights = heights or {}
    extents = {name: drawn_extent(fig, axes) for name, axes in panels.items()}
    rows = [*rows, *(name for name in panels if not any(name in r for r in rows))]
    columns = [
        *columns,
        *(name for name in panels if not any(name in c for c in columns)),
    ]
    y = {}
    for row in rows:
        tops = [
            drawn_extent(fig, axes).y1
            for axes in (heights.get(name, panels[name]) for name in row)
            if not isinstance(axes, Real) and len(axes) > 0
        ]
        for name in row:
            given = heights.get(name)
            if isinstance(given, Real):
                y[name] = float(given)
            else:
                y[name] = max(tops) + above
                if ceiling is not None:
                    y[name] = min(y[name], ceiling)
    x = {}
    for column in columns:
        edge = min(extents[name].x0 for name in column) - left_of
        if floor is not None:
            edge = max(edge, floor)
        x.update(dict.fromkeys(column, edge))
    return {
        name: fig.text(
            x[name],
            y[name],
            name,
            fontsize=PANEL_LETTER_SIZE,
            fontweight="bold",
            ha="left",
            va=va,
        )
        for name in panels
    }
