"""The checks a paper figure passes: Cell Press's limits and the placement of
its texts and panels."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from continuum_swr_replay.visualization.publication_style import (
    check_journal_limits,  # noqa: E402
)


def _figure(width_in: float = 3.0, fontsize: float = 7.0, linewidth: float = 1.0):
    fig, ax = plt.subplots(figsize=(width_in, 2.0), layout="none")
    ax.plot([0, 1], [0, 1], linewidth=linewidth)
    ax.set_xlabel("Time (s)", fontsize=fontsize)
    ax.tick_params(labelsize=6)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    return fig


def test_a_figure_within_the_limits_passes():
    fig = _figure()
    check_journal_limits(fig)
    plt.close(fig)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"width_in": 10.0}, "saved size"),
        ({"fontsize": 5.0}, "is 5 pt"),
        ({"fontsize": 9.0}, "is 9 pt"),
        ({"linewidth": 0.3}, "0.3 pt wide"),
        ({"linewidth": 2.0}, "2 pt wide"),
    ],
)
def test_a_figure_outside_a_limit_is_refused(change, message):
    fig = _figure(**change)
    with pytest.raises(ValueError, match=message):
        check_journal_limits(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Placement: no text over another, no panel into another
# ---------------------------------------------------------------------------


def _texts(pairs):
    return {frozenset(pair) for pair in pairs}


def test_overlapping_texts_names_the_texts_that_run_into_each_other():
    from continuum_swr_replay.visualization.publication_style import overlapping_texts

    fig, ax = plt.subplots(figsize=(3, 2), layout="none")
    ax.axis("off")
    ax.text(0.50, 0.5, "first", ha="left")
    ax.text(0.53, 0.5, "second", ha="left")
    ax.text(0.05, 0.9, "apart")
    assert _texts(overlapping_texts(fig)) == {frozenset({"first", "second"})}
    plt.close(fig)


def test_slanted_labels_that_do_not_touch_are_not_overlapping():
    """Two slanted tick labels side by side: their upright bounding boxes
    overlap, the texts do not. Slanted the other way into each other, they do."""
    from continuum_swr_replay.visualization.publication_style import overlapping_texts

    # Sizes given here, not left to the defaults another test may have set: the
    # ticks are 0.45 in apart and the labels 9 pt.
    fig = plt.figure(figsize=(1.5, 2), dpi=100, layout="none")
    ax = fig.add_axes([0.3, 0.5, 0.6, 0.4])
    ax.set_yticks([])
    ax.set_xlim(-0.5, 1.5)
    ax.set_xticks(
        [0, 1], ["Continuous", "Fragmented"], rotation=30, ha="right", fontsize=9
    )
    fig.draw_without_rendering()
    first, second = (label.get_window_extent() for label in ax.get_xticklabels())
    assert first.overlaps(second)
    assert overlapping_texts(fig) == []

    ax.get_xticklabels()[1].set_rotation(-30)
    assert _texts(overlapping_texts(fig)) == {frozenset({"Continuous", "Fragmented"})}
    plt.close(fig)


def test_a_tick_label_outside_the_view_is_not_drawn_so_cannot_overlap():
    from continuum_swr_replay.visualization.publication_style import (
        drawn_texts,
        overlapping_texts,
    )

    fig, ax = plt.subplots(figsize=(3, 2), layout="none")
    ax.set_yticks([])
    ax.set_xticks([0, 1, 5], ["zero", "one", "five"])
    ax.set_xlim(0, 1)
    # a text right on the last drawn label, wherever the defaults put it
    fig.draw_without_rendering()
    on_one = ax.get_xticklabels()[1].get_window_extent()
    ax.text(on_one.x0, on_one.y0, "here", transform=None, ha="left", va="bottom")
    assert {text.get_text() for text in drawn_texts(fig)} == {"zero", "one", "here"}
    assert _texts(overlapping_texts(fig)) == {frozenset({"one", "here"})}
    plt.close(fig)


def test_check_placement_names_overlapping_texts_and_panels():
    from continuum_swr_replay.visualization.publication_style import check_placement

    fig = plt.figure(figsize=(4, 2), layout="none")
    left = fig.add_axes([0.10, 0.2, 0.30, 0.6])
    right = fig.add_axes([0.60, 0.2, 0.30, 0.6])
    check_placement(fig, {"a": [left], "b": [right]})  # nothing meets

    right.set_ylabel("a label that is drawn\nwell into the panel before it", x=0)
    right.yaxis.set_label_coords(-1.2, 0.5)
    with pytest.raises(ValueError, match="panels a and b"):
        check_placement(fig, {"a": [left], "b": [right]})
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3, 2), layout="none")
    ax.axis("off")
    ax.text(0.5, 0.5, "first", ha="left")
    ax.text(0.53, 0.5, "second", ha="left")
    with pytest.raises(ValueError, match="'first' and 'second'"):
        check_placement(fig, {"a": [ax]})
    plt.close(fig)
