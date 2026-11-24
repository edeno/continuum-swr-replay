"""Where the panels of a multi-panel figure and their letters sit.

The letters follow convention 10 of ``docs/development.md``: the letters of a
row sit at one height, just above the highest of the row's drawn tops, and the
letters of a column at one left edge, the leftmost of the column's drawn
extents. Every extent is measured on the drawn figure (``drawn_extent``), so
the figure must be drawn (``fig.draw_without_rendering()``) before a letter is
placed.
"""

from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Literal

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.text import Text

from continuum_swr_replay.visualization.simulated_run_panels import (
    PANEL_LETTER_SIZE,
    drawn_extent,
)

__all__ = ["place_letters"]


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
