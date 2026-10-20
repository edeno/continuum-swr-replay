"""The per-bin figure panels leave excluded time out (task 26b, items 2 and 6).

Each panel that summarises bins draws the same thing on a frame with excluded
time as on the same frame with those rows removed, and something else when the
excluded rows are counted. The frames are ``tests/unit/_excluded_time_frames.py``'s.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from continuum_swr_replay.visualization import figure_panels  # noqa: E402
from scripts.content import analyze_content_occupancy  # noqa: E402
from scripts.figures import generate_event_gallery as gallery  # noqa: E402
from scripts.figures import generate_supp_decoder_validation as supp  # noqa: E402
from tests.unit._excluded_time_frames import (  # noqa: E402
    rows_removed,
    two_sessions,
    unfiltered,
)


def _drawn(plot, frame: pd.DataFrame) -> list[np.ndarray]:
    """The bars, histogram bins and lines a panel draws."""
    fig, ax = plt.subplots()
    plot(frame, ax=ax)
    drawn = [
        np.array([p.get_x(), p.get_y(), p.get_width(), p.get_height()])
        for p in ax.patches
    ] + [np.asarray(line.get_ydata(), dtype=float) for line in ax.get_lines()]
    plt.close(fig)
    return drawn


def _same(left: list[np.ndarray], right: list[np.ndarray]) -> bool:
    return len(left) == len(right) and all(
        a.shape == b.shape and np.allclose(a, b, equal_nan=True)
        for a, b in zip(left, right, strict=True)
    )


PANELS = {
    "B2 plot_swr_zscore_by_state": figure_panels.plot_swr_zscore_by_state,
    "B5 plot_state_fractions_by_movement": (
        figure_panels.plot_state_fractions_by_movement
    ),
    "C9 plot_state_probability_by_theta_phase": (
        figure_panels.plot_state_probability_by_theta_phase
    ),
    "B3 supp panel a": lambda frame, ax: supp.plot_non_local_prob_distribution(
        frame, ax
    ),
}


@pytest.mark.parametrize("name", PANELS)
def test_a_panel_draws_what_it_draws_without_the_excluded_rows(name):
    plot = PANELS[name]
    frame = two_sessions()

    drawn = _drawn(plot, frame)

    assert _same(drawn, _drawn(plot, rows_removed(frame)))
    assert not _same(drawn, _drawn(plot, unfiltered(frame)))


@pytest.mark.parametrize(
    "plot",
    [
        figure_panels.plot_swr_zscore_by_state,
        figure_panels.plot_state_fractions_by_movement,
        figure_panels.plot_state_probability_by_theta_phase,
    ],
)
def test_a_panel_refuses_a_frame_not_read_through_the_source(plot):
    frame = two_sessions().drop(columns="in_excluded_time")

    with pytest.raises(KeyError, match="in_excluded_time"):
        _drawn(plot, frame)


STRICT = {
    **PANELS,
    "B8 well positions": lambda frame, ax: (
        analyze_content_occupancy._well_positions(frame)
    ),
}


@pytest.mark.parametrize("name", STRICT)
def test_a_filter_refuses_a_column_that_is_not_boolean(name):
    """The same check as the quality metrics (``analysed_time.analysed_bins``).
    Cast to bool, a None would read as analysed and a NaN as excluded."""
    frame = two_sessions()
    frame["in_excluded_time"] = frame["in_excluded_time"].astype(object)
    frame.iloc[:10, frame.columns.get_loc("in_excluded_time")] = None

    with pytest.raises(TypeError, match="in_excluded_time"):
        _drawn(STRICT[name], frame)


# --------------------------------------------------------------------------
# B8: the reward-well positions of the content-occupancy scan
# --------------------------------------------------------------------------


def test_the_well_positions_leave_excluded_time_out():
    """Each leaf arm's well is the median immobile position on it. On arm 0,
    100 excluded immobile bins at 50 cm would move it from 10 cm."""
    arms = np.repeat(np.arange(6), 60)
    linear = 10.0 + 20.0 * arms
    analysed = pd.DataFrame(
        {
            "head_speed": 1.0,
            "track_segment_id": arms,
            "linear_position": linear,
            "in_excluded_time": False,
        }
    )
    excluded = pd.DataFrame(
        {
            "head_speed": 1.0,
            "track_segment_id": 0,
            "linear_position": np.full(100, 50.0),
            "in_excluded_time": True,
        }
    )
    frame = pd.concat([analysed, excluded], ignore_index=True)

    wells = analyze_content_occupancy._well_positions(frame)

    assert wells == [10.0, 30.0, 50.0, 70.0, 90.0, 110.0]
    assert wells == analyze_content_occupancy._well_positions(analysed)
    assert analyze_content_occupancy._well_positions(unfiltered(frame))[0] == 50.0


# --------------------------------------------------------------------------
# The event gallery shades excluded time in each window
# --------------------------------------------------------------------------

EXCLUDED = pd.DataFrame({"start_time": [1.0, 2.5, 5.0], "end_time": [2.0, 3.0, 6.0]})


@pytest.mark.parametrize(
    ("window", "spans"),
    [
        ((1.5, 2.7), [(1.5, 2.0), (2.5, 2.7)]),  # clipped to the window
        ((0.2, 0.9), []),  # none
        ((5.2, 5.8), [(5.2, 5.8)]),  # the window lies inside one
        ((3.0, 4.0), []),  # touches one at a single instant
        ((0.0, 10.0), [(1.0, 2.0), (2.5, 3.0), (5.0, 6.0)]),
    ],
)
def test_the_gallery_finds_the_excluded_spans_inside_a_window(window, spans):
    assert gallery._excluded_spans_in_window(EXCLUDED, *window) == spans


def test_the_gallery_shades_each_span_behind_the_data():
    fig, axes = plt.subplots(3, 1)
    for ax in axes:
        ax.plot([0.0, 1.0], [0.0, 1.0])

    gallery._shade_excluded(list(axes), [(0.1, 0.2), (0.5, 0.6)])

    for ax in axes:
        bands = [p for p in ax.patches if p.get_label() == gallery.EXCLUDED_LABEL]
        assert len(bands) == 2
        assert all(band.get_zorder() < ax.get_lines()[0].get_zorder() for band in bands)
    plt.close(fig)
