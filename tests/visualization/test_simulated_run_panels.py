"""The check that a figure is within Cell Press's figure guidelines."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from continuum_swr_replay.visualization.simulated_run_panels import (  # noqa: E402
    check_journal_limits,
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


def test_the_state_colours_are_the_palette_colours():
    from continuum_swr_replay.visualization.palettes import COLORS
    from continuum_swr_replay.visualization.simulated_run_panels import STATE_COLORS

    assert STATE_COLORS["Local"] == COLORS["Local"]
    assert STATE_COLORS["No-Spike"] == COLORS["No-Spike"]


@pytest.mark.parametrize(
    ("color", "expected"),
    [
        ("#add8e6", "black"),
        ("#b0b0b0", "black"),
        ("#1b9e77", "white"),
        ("#D55E00", "white"),
    ],
)
def test_text_on_a_light_fill_is_black_and_on_a_dark_fill_white(color, expected):
    from continuum_swr_replay.visualization.simulated_run_panels import text_color_on

    assert text_color_on(color) == expected


def test_a_light_colour_is_darkened_for_text_and_a_dark_one_kept():
    from colorsys import rgb_to_hls

    from matplotlib.colors import to_rgb

    from continuum_swr_replay.visualization.simulated_run_panels import (
        readable_text_color,
    )

    assert readable_text_color("#1b9e77") == "#1b9e77"
    darkened = readable_text_color("#add8e6")
    hue, lightness, _ = rgb_to_hls(*to_rgb(darkened))
    assert lightness == pytest.approx(0.4, abs=0.01)
    assert hue == pytest.approx(rgb_to_hls(*to_rgb("#add8e6"))[0], abs=0.01)
