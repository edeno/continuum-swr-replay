"""The drawings of Figure 2's evidence panels: what each names and marks."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from continuum_swr_replay.visualization import evidence_drawings  # noqa: E402
from continuum_swr_replay.visualization.inventory_examples import (  # noqa: E402
    ANIMAL_POSITION_COLOR,
)
from continuum_swr_replay.visualization.palettes import COLORS  # noqa: E402


def _drawn(draw):
    fig, ax = plt.subplots()
    draw(ax)
    fig.canvas.draw()
    return fig, ax


def _filled(ax, color) -> list[Rectangle]:
    return [
        p for p in ax.patches
        if isinstance(p, Rectangle) and np.allclose(p.get_facecolor(), to_rgba(color))
    ]  # fmt: skip


def test_the_left_out_test_names_its_parts_and_draws_its_places():
    """Every text of ``LEFT_OUT_TEXTS`` once; three strips (the decoded place,
    the animal's place, another event's place); the animal's place a mark in
    the animal-position color; the left-out spikes as ticks under the decoded
    place, joined to it by a double arrow; the axis hidden, 0 to 10 each way."""
    fig, ax = _drawn(evidence_drawings.draw_left_out_test)
    texts = [t.get_text() for t in ax.texts if t.get_text()]
    assert sorted(texts) == sorted(evidence_drawings.LEFT_OUT_TEXTS.values())
    strips = [
        p for p in ax.patches
        if isinstance(p, Rectangle) and p.get_facecolor()[:3] == (1.0, 1.0, 1.0)
    ]  # fmt: skip
    assert len(strips) == 3
    marks = [
        line for line in ax.lines
        if np.allclose(to_rgba(line.get_color()), to_rgba(ANIMAL_POSITION_COLOR))
    ]  # fmt: skip
    assert len(marks) == 1
    # the spike ticks, vertical and under the decoded place's center
    ticks = [
        line for line in ax.lines
        if line.get_xdata()[0] == line.get_xdata()[1] and line not in marks
    ]  # fmt: skip
    assert len(ticks) >= 5
    arrows = [c for c in ax.texts if getattr(c, "arrow_patch", None) is not None]
    assert len(arrows) == 1
    assert not ax.axison
    assert ax.get_xlim() == (0, 10) and ax.get_ylim() == (0, 10)
    plt.close(fig)


def test_the_restriction_test_draws_the_swrs_and_the_two_rows():
    """Every text of ``RESTRICTION_TEXTS`` once; the two SWRs in the SWR color;
    the non-local color under each SWR in the first row (only in SWRs) and
    across the whole second row (at any time)."""
    fig, ax = _drawn(evidence_drawings.draw_restriction_test)
    texts = [t.get_text() for t in ax.texts]
    assert sorted(texts) == sorted(evidence_drawings.RESTRICTION_TEXTS.values())
    swrs = _filled(ax, COLORS["SWR"])
    assert [(p.get_x(), p.get_x() + p.get_width()) for p in swrs] == list(
        evidence_drawings.SWR_TIMES
    )
    nonlocal_spans = sorted(
        (p.get_y(), p.get_x(), p.get_x() + p.get_width())
        for p in _filled(ax, COLORS["Non-Local"])
    )
    (whole, *in_swrs) = nonlocal_spans  # the lower row first
    assert (whole[1], whole[2]) == (
        evidence_drawings.ROWS_LEFT,
        evidence_drawings.STRIP_RIGHT,
    )
    assert [(x0, x1) for _, x0, x1 in in_swrs] == list(evidence_drawings.SWR_TIMES)
    assert len({y for y, _, _ in in_swrs}) == 1 and in_swrs[0][0] > whole[0]
    assert not ax.axison
    plt.close(fig)


def test_the_boundary_cases_draw_each_rows_swr_and_event():
    """Each row: its name, the SWR in the SWR color above, the event in the
    non-local color below, with the bound that differs moved by the shift; the
    rows top to bottom."""
    cases = [
        ("Same start and end", "together", "together"),
        ("Ends later", "together", "later"),
        ("Starts earlier", "earlier", "together"),
    ]
    fig, ax = plt.subplots()
    evidence_drawings.draw_boundary_cases(ax, cases)
    assert [t.get_text() for t in ax.texts] == [name for name, _, _ in cases]
    swrs = _filled(ax, COLORS["SWR"])
    events = _filled(ax, COLORS["Non-Local"])
    assert len(swrs) == len(events) == 3
    low, high = evidence_drawings.BOUNDARY_SWR
    shift = evidence_drawings.BOUNDARY_SHIFT
    expected = [(low, high), (low, high + shift), (low - shift, high)]
    for k, (event, bounds) in enumerate(zip(events, expected, strict=True)):
        assert (event.get_x(), event.get_x() + event.get_width()) == pytest.approx(
            bounds
        )
        assert swrs[k].get_y() < event.get_y()  # the SWR above (y down)
        assert k - 0.5 < event.get_y() < k + 0.5
    assert ax.get_ylim() == (2.5, -0.5)
    assert not ax.axison
    plt.close(fig)
