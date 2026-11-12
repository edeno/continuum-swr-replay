"""Axis units of the replay-content hierarchy figure, and the forest plot's stars.

A fraction fit's AIC is a quasi-likelihood quantity at the event scale and a
count fit's is at the timebin scale; the two are not comparable, so the plotted
ΔAIC has to say which one it is.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from continuum_swr_replay.visualization.replay_content_plots import (
    plot_coefficient_forest,
    plot_model_hierarchy_comparison,
)

COMPARISON = pd.DataFrame(
    {
        "parent": ["M0", "M1"],
        "child": ["M1", "M2"],
        "delta_aic_median": [12.0, -3.0],
        "delta_aic_q25": [8.0, -6.0],
        "delta_aic_q75": [16.0, 1.0],
        "n_sessions": [215, 215],
        "child_better_aic": [200, 40],
    }
)


@pytest.mark.parametrize(
    "scale_label,expected",
    [
        (None, "ΔAIC (positive = child better)"),
        (
            "event-scale (quasi)",
            "ΔAIC, event-scale (quasi) (positive = child better)",
        ),
        ("timebin-scale", "ΔAIC, timebin-scale (positive = child better)"),
    ],
)
def test_the_axis_names_the_scale_the_fit_was_on(scale_label, expected) -> None:
    _fig, ax = plt.subplots()
    try:
        plot_model_hierarchy_comparison(COMPARISON, ax=ax, scale_label=scale_label)
        assert ax.get_ylabel() == expected
    finally:
        plt.close("all")


def test_the_forest_stars_are_the_corrected_signed_rank_p() -> None:
    # x: eight positive sessions, p = 2 / 2 ** 8. y: three positive, too few
    # for a p, so x is its own family and its q, 0.0078, earns two stars.
    # Were y tested, the family of two would give x q = 0.0156: one star.
    results = pd.DataFrame(
        {
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
