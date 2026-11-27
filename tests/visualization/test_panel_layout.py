"""Where a figure's panels and their letters sit: ``visualization.panel_layout``."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from continuum_swr_replay.visualization import panel_layout  # noqa: E402
from continuum_swr_replay.visualization.panel_layout import (  # noqa: E402
    fit_panel,
    letter_layout,
    panel_spans,
    place_letters,
)
from continuum_swr_replay.visualization.publication_style import drawn_extent


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


def _gaps(fig) -> tuple[float, float]:
    """The letters' distances above and left of their panels, as figure
    fractions of ``fig``."""
    width_in, height_in = fig.get_size_inches()
    return (
        panel_layout.LETTER_GAP_PT / 72 / height_in,
        panel_layout.LETTER_LEFT_PT / 72 / width_in,
    )


def test_a_rows_letters_share_its_highest_top_and_a_columns_its_leftmost_edge(grid):
    """A row's letters' bottom edge stands the gap above its highest drawn top,
    a column's letters' left edge the gap left of its leftmost drawn extent,
    the same distances in points in every figure."""
    fig, panels = grid
    above, left_of = _gaps(fig)
    extent = {name: drawn_extent(fig, axes) for name, axes in panels.items()}
    letters = place_letters(fig, panels, ["abc", "def"], ["ad", "be", "cf"])
    assert list(letters) == list("abcdef")
    at = _positions(letters)
    for row in ("abc", "def"):
        assert {at[n][1] for n in row} == {max(extent[n].y1 for n in row) + above}
    for column in ("ad", "be", "cf"):
        assert {at[n][0] for n in column} == {
            min(extent[n].x0 for n in column) - left_of
        }
    # e's label sets its column's edge, left of b's own
    assert at["b"][0] < extent["b"].x0 - 0.01
    for text in letters.values():
        assert (text.get_ha(), text.get_va()) == ("left", "bottom")
        assert text.get_fontweight() == "bold"
    assert (panel_layout.LETTER_GAP_PT, panel_layout.LETTER_LEFT_PT) == (1.0, 2.0)


def test_a_letter_in_no_row_or_column_is_placed_by_its_own_panel(grid):
    fig, panels = grid
    above, left_of = _gaps(fig)
    extent = {name: drawn_extent(fig, axes) for name, axes in panels.items()}
    at = _positions(place_letters(fig, panels, ["ab"], ["ad"]))
    assert at["a"][1] == at["b"][1] == max(extent["a"].y1, extent["b"].y1) + above
    assert at["e"] == (extent["e"].x0 - left_of, extent["e"].y1 + above)
    assert at["a"][0] == at["d"][0] == min(extent["a"].x0, extent["d"].x0) - left_of


def test_the_named_parts_set_a_rows_height(grid):
    """A panel named with no axes does not count (its title would lift the
    row), and a panel named with axes counts by those."""
    fig, panels = grid
    above, _ = _gaps(fig)
    extent = {name: drawn_extent(fig, axes) for name, axes in panels.items()}
    rows = ["abc", "def"]
    plain = _positions(place_letters(fig, panels, rows))
    assert {plain[n][1] for n in "abc"} == {extent["c"].y1 + above}
    named = _positions(place_letters(fig, panels, rows, heights={"c": []}))
    top = max(extent[n].y1 for n in "ab") + above
    assert top < extent["c"].y1 + above
    assert {named[n][1] for n in "abc"} == {top}


def test_the_letters_stay_on_the_page(grid):
    """A letter's top no higher than the figure's top, its left edge no further
    left than the figure's left edge."""
    fig, panels = grid
    for ax in panels["a"]:
        ax.set_title("a title\nof\nmany\nlines\nreaching\npast\nthe top")
        ax.set_ylabel("a label far to the left", labelpad=60)
    fig.draw_without_rendering()
    letters = place_letters(fig, panels, ["abc"], ["ad"])
    fig.draw_without_rendering()
    figure = fig.bbox
    for name in "abc":
        assert letters[name].get_window_extent().y1 == pytest.approx(figure.y1)
    assert letters["a"].get_position()[0] == 0.0


def _panel(n_columns: int, ylabel: str = "a label"):
    """A figure with one row of ``n_columns`` axes, each with a y label."""
    fig = plt.figure(figsize=(6.0, 2.0), layout="none")
    axes = [fig.add_axes((0.1 + 0.2 * k, 0.2, 0.1, 0.6)) for k in range(n_columns)]
    for ax in axes:
        ax.set_ylabel(ylabel)
    return fig, axes


def test_a_panel_is_fitted_between_its_edges_with_its_widths_and_gap():
    """Between frames, the gap is a share of the frames' mean width and the
    inner labels sit in it; the drawing spans the edges."""
    fig, axes = _panel(3)
    fit_panel(fig, [[ax] for ax in axes], 0.1, 0.9, weights=(2, 1, 1), gap=0.5)
    fig.draw_without_rendering()
    boxes = [ax.get_position() for ax in axes]
    widths = [box.width for box in boxes]
    assert widths[0] == pytest.approx(2 * widths[1]) == pytest.approx(2 * widths[2])
    for first, second in zip(boxes, boxes[1:], strict=False):
        assert second.x0 - first.x1 == pytest.approx(0.5 * sum(widths) / 3)
    extent = drawn_extent(fig, axes)
    assert (extent.x0, extent.x1) == pytest.approx((0.1, 0.9), abs=1e-5)
    plt.close(fig)


def test_between_drawings_each_column_takes_the_room_of_its_labels():
    fig, axes = _panel(2, ylabel="a long\ny label")
    fit_panel(fig, [[ax] for ax in axes], 0.05, 0.95, gap=0.02, between="drawings")
    fig.draw_without_rendering()
    first, second = (drawn_extent(fig, [ax]) for ax in axes)
    assert second.x0 - first.x1 == pytest.approx(0.02, abs=1e-5)
    assert (first.x0, second.x1) == pytest.approx((0.05, 0.95), abs=1e-5)
    plt.close(fig)


def test_a_column_shares_one_frame_and_titles_above_do_not_count():
    """The axes of a column share its left edge and width; a title wider than
    the frame reaches over the edges."""
    fig, axes = _panel(2)
    axes[0].set_position((0.1, 0.55, 0.1, 0.3))
    axes[1].set_position((0.5, 0.15, 0.2, 0.25))
    for ax in axes:
        ax.set_xticks([0.0, 0.5, 1.0])
    axes[0].set_title(
        "a title much wider than its frame, reaching far over", fontsize=10
    )
    fit_panel(fig, [axes], 0.3, 0.7)
    first, second = (ax.get_position() for ax in axes)
    assert (first.x0, first.width) == pytest.approx((second.x0, second.width))
    assert (first.y0, second.height) == pytest.approx((0.55, 0.25))
    fig.draw_without_rendering()
    title = axes[0].title.get_window_extent().transformed(fig.transFigure.inverted())
    assert title.x0 < 0.3 and title.x1 > 0.7
    plt.close(fig)


def test_a_panel_that_does_not_settle_is_refused(monkeypatch):
    fig, axes = _panel(1)
    monkeypatch.setattr(panel_layout, "FIT_PASSES", 1)
    with pytest.raises(ValueError, match="did not settle in 1 passes"):
        fit_panel(fig, [axes], 0.3, 0.7)
    plt.close(fig)


def test_the_spans_and_letters_follow_from_the_layouts_rows():
    rows = (
        (("p", "left", "middle"), ("q", "middle", 0.9)),
        (("r", "left", 0.5),),
    )
    spans = panel_spans(rows, {"left": 0.1, "middle": 0.6}, gap=0.05)
    expected = {"p": (0.1, 0.55), "q": (0.6, 0.9), "r": (0.1, 0.5)}
    assert list(spans) == list(expected)
    for panel, edges in expected.items():
        assert spans[panel] == pytest.approx(edges)
    letter_rows, columns = letter_layout(rows, {"p": "a", "q": "b", "r": "c"})
    assert letter_rows == ["ab", "c"]
    assert columns == {"left": "ac", "middle": "b"}


def test_a_named_ceiling_keeps_the_letters_tops_under_it(grid):
    """A figure at the journal's full height names its highest drawn top as
    the ceiling, so that no letter adds to its height."""
    fig, panels = grid
    top = drawn_extent(fig, panels["c"]).y1
    letters = place_letters(fig, panels, ["abc"], ceiling=top)
    fig.draw_without_rendering()
    to_figure = fig.transFigure.inverted()
    for name in "abc":
        box = letters[name].get_window_extent().transformed(to_figure)
        assert box.y1 == pytest.approx(top)
