"""Small drawings of what Figure 2's evidence panels compare.

Each draws on one axes in data units 0 to 10 (x) by 0 to 10 (y), with the
axis hidden, so that a figure places it like any other axes. The names sit at
the left, the strips at the right (from :data:`STRIP_LEFT`), so that the
drawing is short:

- :func:`draw_left_out_test`: the left-out tetrode's spikes under the place
  decoded from the other tetrodes, and the two places they must fit better
  than (the animal's place; the place decoded in another event);
- :func:`draw_restriction_test`: a stretch of time with two SWRs and the two
  decoders compared, the non-local states allowed only in SWRs or at any time;
- :func:`draw_boundary_cases`: how an event's bounds sit against its SWR's,
  one row per case (Figure 2's supplement, the timing panel).

The drawings are schematic: their places, times and spikes are not data.
"""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory

from continuum_swr_replay.visualization.inventory_examples import (
    ANIMAL_POSITION_COLOR,
)
from continuum_swr_replay.visualization.palettes import (
    COLORS,
    REFERENCE_COLOR,
    readable_text_color,
)
from continuum_swr_replay.visualization.publication_style import SMALL_SIZE

STRIP_LEFT, STRIP_RIGHT = 6.5, 10.0
"""The x extent of a drawn strip of :func:`draw_left_out_test` (data units);
the names sit left of it."""
ROWS_LEFT = 0.0
"""Where :func:`draw_restriction_test`'s rows start (data units): their names
sit above them."""
STRIP_HEIGHT = 1.0
"""A strip's height (data units)."""

LEFT_OUT_TEXTS = {
    "decoded": "Decoded place\n(other tetrodes)",
    "spikes": "Left-out\ntetrode's spikes",
    "must": "It must fit better than:",
    "animal": "1  animal's place",
    "other": "2  another event's\n    decoded place",
}
"""The texts of :func:`draw_left_out_test`, by what they name."""

RESTRICTION_TEXTS = {
    "swrs": "SWRs",
    "allowed": "Non-local allowed:",
    "only": "only in SWRs",
    "any": "at any time",
}
"""The texts of :func:`draw_restriction_test`, by what they name."""


def _blank(ax: Axes) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")


def _strip(ax: Axes, y: float) -> None:
    ax.add_patch(
        Rectangle(
            (STRIP_LEFT, y), STRIP_RIGHT - STRIP_LEFT, STRIP_HEIGHT,
            fc="white", ec="black", lw=0.5,
        )
    )  # fmt: skip


def _blob(ax: Axes, y: float, center: float, color: str, sigma: float = 0.3) -> None:
    """A place estimate in a strip: shading that falls off around ``center``."""
    edges = np.linspace(STRIP_LEFT, STRIP_RIGHT, 121)
    width = edges[1] - edges[0]
    for left in edges[:-1]:
        level = float(np.exp(-0.5 * ((left + width / 2 - center) / sigma) ** 2))
        if level > 0.03:
            ax.add_patch(
                Rectangle(
                    (left, y), width, STRIP_HEIGHT, fc=color, ec="none", alpha=level
                )
            )


def _name(ax: Axes, y: float, text: str, **kwargs) -> None:
    ax.text(0.0, y, text, fontsize=SMALL_SIZE, ha="left", va="center", **kwargs)


def draw_left_out_test(ax: Axes) -> None:
    """The left-out tetrode test, as a drawing (texts :data:`LEFT_OUT_TEXTS`).

    Top, a strip with the place decoded from the other tetrodes in this event
    (a dark shading); under it the left-out tetrode's spikes, ticks under the
    shading, joined to it by a double arrow; then the two places the spikes
    must fit better than: a strip with the animal's place (a mark in the
    color of the animal's position) and a strip with the place decoded in
    another event (a gray shading elsewhere). Each is named at the left."""
    _blank(ax)
    texts = LEFT_OUT_TEXTS
    center = 8.4
    _strip(ax, 8.7)
    _blob(ax, 8.7, center, "black", sigma=0.25)
    _name(ax, 9.2, texts["decoded"])
    for position in (8.0, 8.15, 8.3, 8.4, 8.5, 8.6, 8.8):
        ax.plot([position, position], [6.2, 7.1], color="black", lw=0.6)
    ax.plot([STRIP_LEFT, STRIP_RIGHT], [6.15, 6.15], color="black", lw=0.5)
    _name(ax, 6.65, texts["spikes"])
    ax.annotate(
        "", (center, 8.65), (center, 7.15),
        arrowprops={"arrowstyle": "<->", "lw": 0.6, "color": "black",
                    "shrinkA": 0, "shrinkB": 0},
    )  # fmt: skip
    _name(ax, 4.6, texts["must"], style="italic")
    _strip(ax, 2.9)
    ax.plot(
        [STRIP_LEFT + 0.5] * 2, [2.9, 2.9 + STRIP_HEIGHT],
        color=ANIMAL_POSITION_COLOR, lw=1.4,
    )  # fmt: skip
    _name(ax, 3.4, texts["animal"])
    _strip(ax, 0.4)
    _blob(ax, 0.4, 7.6, REFERENCE_COLOR, sigma=0.25)
    _name(ax, 0.9, texts["other"])


SWR_TIMES = ((5.0, 6.1), (8.0, 8.8))
"""The drawn SWRs' starts and ends (data units of :func:`draw_restriction_test`)."""


def draw_restriction_test(ax: Axes) -> None:
    """The decoder comparison, as a drawing (texts :data:`RESTRICTION_TEXTS`).

    A stretch of time with two SWRs (boxes in the SWR color, named at the
    left), and under it, below "Non-local allowed:", two rows, each named
    above it at the right: only in SWRs (the non-local color only under the
    SWRs) and at any time (the whole row in the non-local color)."""
    _blank(ax)
    texts = RESTRICTION_TEXTS
    for start, end in SWR_TIMES:
        ax.add_patch(
            Rectangle((start, 8.8), end - start, 1.2, fc=COLORS["SWR"], ec="none")
        )
    _name(ax, 9.4, texts["swrs"], color=COLORS["SWR"])
    _name(ax, 7.4, texts["allowed"], style="italic")
    for y, name, spans in (
        (4.0, texts["only"], SWR_TIMES),
        (0.4, texts["any"], ((ROWS_LEFT, STRIP_RIGHT),)),
    ):
        ax.add_patch(
            Rectangle(
                (ROWS_LEFT, y), STRIP_RIGHT - ROWS_LEFT, 1.4,
                fc="white", ec="black", lw=0.5,
            )
        )  # fmt: skip
        for start, end in spans:
            ax.add_patch(
                Rectangle(
                    (start, y), end - start, 1.4, fc=COLORS["Non-Local"], ec="none"
                )
            )
        ax.text(
            STRIP_RIGHT, y + 1.6, name, fontsize=SMALL_SIZE, ha="right", va="bottom"
        )


BOUNDARY_KEY = ("SWR", "event")
"""The names of a case drawing's two bars, written over the drawings."""
BOUNDARY_SWR = (7.4, 9.2)
"""The drawn SWR's start and end (data units of :func:`draw_boundary_cases`)."""
BOUNDARY_SHIFT = 0.7
"""How far a drawn event's bound sits from the SWR's when it is earlier or later."""
BOUNDARY_BAR = 0.24
"""The height of a drawn bar (rows are 1 apart)."""


def boundary_case_bounds(start: str, end: str) -> tuple[float, float]:
    """A drawn event's start and end for how its bounds sit against its SWR's
    (``"earlier"``, ``"together"`` or ``"later"`` each, as
    ``event_statistics.TIMING_CLASSES``)."""
    shift = {"earlier": -BOUNDARY_SHIFT, "together": 0.0, "later": BOUNDARY_SHIFT}
    return BOUNDARY_SWR[0] + shift[start], BOUNDARY_SWR[1] + shift[end]


def draw_boundary_cases(ax: Axes, cases: list[tuple[str, str, str]]) -> None:
    """How an event's bounds sit against its SWR's, one row per case: at the
    left the case's name, at the right the SWR as a bar in the SWR color and
    under it the event as a bar in the non-local color, longer or shorter at
    the bound that differs.

    Parameters
    ----------
    ax : Axes
        Drawn in data units 0 to 10 across; row ``k`` at y = ``k`` (the first
        at the top), so that bars drawn on another axes sharing the y axis
        line up with the rows.
    cases : list of (name, start, end)
        Each row's name and its event's start and end class (``"earlier"``,
        ``"together"``, ``"later"``).
    """
    for k, (name, start, end) in enumerate(cases):
        ax.add_patch(
            Rectangle(
                (BOUNDARY_SWR[0], k - BOUNDARY_BAR - 0.03),
                BOUNDARY_SWR[1] - BOUNDARY_SWR[0], BOUNDARY_BAR,
                fc=COLORS["SWR"], ec="none",
            )
        )  # fmt: skip
        left, right = boundary_case_bounds(start, end)
        ax.add_patch(
            Rectangle(
                (left, k + 0.03), right - left, BOUNDARY_BAR,
                fc=COLORS["Non-Local"], ec="none",
            )
        )  # fmt: skip
        ax.text(0.0, k, name, fontsize=SMALL_SIZE, ha="left", va="center")
    # the two bars named in their colors, over the drawings
    middle = sum(BOUNDARY_SWR) / 2
    for name, color, side, shift in (
        (BOUNDARY_KEY[0], COLORS["SWR"], "right", -0.12),
        (BOUNDARY_KEY[1], COLORS["Non-Local"], "left", 0.12),
    ):
        ax.text(
            middle + shift, 1.0, name, ha=side, va="bottom", fontsize=SMALL_SIZE,
            color=readable_text_color(color),
            transform=blended_transform_factory(ax.transData, ax.transAxes),
        )  # fmt: skip
    ax.set_xlim(0, 10)
    ax.set_ylim(len(cases) - 0.5, -0.5)
    ax.axis("off")
