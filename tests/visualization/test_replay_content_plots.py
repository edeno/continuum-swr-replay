"""Axis units of the replay-content hierarchy figure, and the forest plot's stars.

A fraction fit's AIC is a quasi-likelihood quantity at the event scale and a
count fit's is at the timebin scale; the two are not comparable, so the plotted
ΔAIC has to say which one it is. ΔAIC is the child's AIC minus the parent's,
so a negative bar is a step whose child is better.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.colors import to_rgba

from continuum_swr_replay.visualization.palettes import COLORS
from continuum_swr_replay.visualization.replay_content_plots import (
    plot_coefficient_forest,
    plot_model_hierarchy_comparison,
)

COMPARISON = pd.DataFrame(
    {
        "parent": ["M0", "M1"],
        "child": ["M1", "M2"],
        # M0 -> M1: the child's AIC is lower (negative); M1 -> M2: higher.
        "delta_aic_median": [-12.0, 3.0],
        "delta_aic_q25": [-16.0, -1.0],
        "delta_aic_q75": [-8.0, 6.0],
        "n_sessions": [215, 215],
        "child_better_aic": [200, 40],
    }
)


@pytest.mark.parametrize(
    "scale_label,expected",
    [
        (None, "ΔAIC (child − parent)\nnegative = child better"),
        (
            "event-scale (quasi)",
            "ΔAIC (child − parent), event-scale (quasi)\nnegative = child better",
        ),
        (
            "timebin-scale",
            "ΔAIC (child − parent), timebin-scale\nnegative = child better",
        ),
    ],
)
def test_the_axis_names_the_scale_the_fit_was_on(scale_label, expected) -> None:
    _fig, ax = plt.subplots()
    try:
        plot_model_hierarchy_comparison(COMPARISON, ax=ax, scale_label=scale_label)
        assert ax.get_ylabel() == expected
    finally:
        plt.close("all")


def test_a_step_whose_child_is_better_is_drawn_negative_in_the_non_local_colour():
    _fig, ax = plt.subplots()
    try:
        plot_model_hierarchy_comparison(COMPARISON, ax=ax)
        bars = ax.patches
        assert [bar.get_height() for bar in bars] == [-12.0, 3.0]
        assert bars[0].get_facecolor() == to_rgba(COLORS["Non-Local"], 0.7)
        assert bars[1].get_facecolor() == to_rgba(COLORS["SWR"], 0.7)
        (errors,) = ax.collections
        assert [list(seg[:, 1]) for seg in errors.get_segments()] == [
            [-16.0, -8.0],
            [-1.0, 6.0],
        ]
    finally:
        plt.close("all")


def test_the_forest_stars_are_the_corrected_signed_rank_p() -> None:
    # x: eight positive sessions, p = 2 / 2 ** 8. y: three positive, too few
    # for a p, so x is its own family and its q, 0.0078, earns two stars.
    # Were y tested, the family of two would give x q = 0.0156: one star.
    results = pd.DataFrame(
        {
            "animal": ["a", "a", "a", "a", "b", "b", "b", "b"],
            "session_date": list(range(8)),
            "epoch": "02_r1",
            "coef_x": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            "coef_y": [0.1, 0.2, 0.3] + [float("nan")] * 5,
        }
    )
    _fig, ax = plt.subplots()
    try:
        plot_coefficient_forest(results, ["x", "y"], ax=ax)
        assert [text.get_text() for text in ax.texts] == ["**"]
    finally:
        plt.close("all")
