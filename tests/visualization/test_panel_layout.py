"""Where a figure's panels and their letters sit: ``visualization.panel_layout``."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from continuum_swr_replay.visualization.panel_layout import (  # noqa: E402
    place_letters,
)
from continuum_swr_replay.visualization.simulated_run_panels import (  # noqa: E402
    drawn_extent,
)


@pytest.fixture
def grid():
    """A drawn 2 x 3 grid of panels a to f, row by row; c has a two-line title
    (the highest top of its row) and e a two-line y label (the leftmost extent
    of its column)."""
    fig, axes = plt.subplots(2, 3, layout="none")
    panels = dict(zip("abcdef", ([ax] for ax in axes.ravel()), strict=True))
    axes[0, 2].set_title("a title\nof two lines")
    axes[1, 1].set_ylabel("a label\nof two lines")
    fig.draw_without_rendering()
    yield fig, panels
    plt.close(fig)


def _positions(letters):
    return {name: text.get_position() for name, text in letters.items()}


def test_a_rows_letters_share_its_highest_top_and_a_columns_its_leftmost_edge(grid):
    fig, panels = grid
    extent = {name: drawn_extent(fig, axes) for name, axes in panels.items()}
    letters = place_letters(
        fig, panels, ["abc", "def"], ["ad", "be", "cf"], above=0.01, left_of=0.004
    )
    assert list(letters) == list("abcdef")
    at = _positions(letters)
    for row in ("abc", "def"):
        assert {at[n][1] for n in row} == {max(extent[n].y1 for n in row) + 0.01}
    for column in ("ad", "be", "cf"):
        assert {at[n][0] for n in column} == {min(extent[n].x0 for n in column) - 0.004}
    # e's label sets its column's edge, left of b's own
    assert at["b"][0] < extent["b"].x0 - 0.01
    for text in letters.values():
        assert (text.get_ha(), text.get_va()) == ("left", "top")
        assert text.get_fontweight() == "bold"


def test_a_letter_in_no_row_or_column_is_placed_by_its_own_panel(grid):
    fig, panels = grid
    extent = {name: drawn_extent(fig, axes) for name, axes in panels.items()}
    at = _positions(place_letters(fig, panels, ["ab"], ["ad"], above=0.0))
    assert at["a"][1] == at["b"][1] == max(extent["a"].y1, extent["b"].y1)
    assert at["c"] == (extent["c"].x0, extent["c"].y1)
    assert at["e"] == (extent["e"].x0, extent["e"].y1)
    assert at["a"][0] == at["d"][0] == min(extent["a"].x0, extent["d"].x0)


def test_the_named_parts_set_a_rows_height(grid):
    """A panel named with no axes does not count (its title would lift the
    row), a panel named with axes counts by those, and a letter given a height
    sits there."""
    fig, panels = grid
    extent = {name: drawn_extent(fig, axes) for name, axes in panels.items()}
    rows = ["abc", "def"]
    plain = _positions(place_letters(fig, panels, rows, above=0.002))
    assert {plain[n][1] for n in "abc"} == {extent["c"].y1 + 0.002}
    named = _positions(
        place_letters(fig, panels, rows, above=0.002, heights={"c": [], "d": 0.3})
    )
    top = max(extent[n].y1 for n in "ab") + 0.002
    assert top < extent["c"].y1
    assert {named[n][1] for n in "abc"} == {top}
    assert named["d"][1] == 0.3
    assert {named[n][1] for n in "ef"} == {max(extent[n].y1 for n in "ef") + 0.002}


def test_the_letters_stay_under_the_ceiling_and_inside_the_left_edge(grid):
    fig, panels = grid
    letters = place_letters(
        fig, panels, ["abc"], above=0.5, left_of=1.0, ceiling=0.985, va="bottom"
    )
    at = _positions(letters)
    assert {at[n][1] for n in "abc"} == {0.985}
    assert {x for x, _ in at.values()} == {0.0}
    assert {text.get_va() for text in letters.values()} == {"bottom"}


def test_without_a_floor_a_letter_sits_at_its_column_even_past_the_left_edge(grid):
    fig, panels = grid
    extent = drawn_extent(fig, panels["a"])
    letters = place_letters(fig, panels, ["abc"], above=0.0, left_of=1.0, floor=None)
    assert letters["a"].get_position()[0] == extent.x0 - 1.0
