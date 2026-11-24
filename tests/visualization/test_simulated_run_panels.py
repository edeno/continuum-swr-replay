"""The colors the figures of the simulated run share."""

from __future__ import annotations


def test_the_state_colors_are_the_palette_colors():
    from continuum_swr_replay.visualization.palettes import COLORS
    from continuum_swr_replay.visualization.simulated_run_panels import STATE_COLORS

    assert STATE_COLORS["Local"] == COLORS["Local"]
    assert STATE_COLORS["No-Spike"] == COLORS["No-Spike"]
