"""Small tables and checks the tests of Figure 2 and its supplement share."""

from __future__ import annotations

import matplotlib.collections
import numpy as np
import pandas as pd
import pytest


def figure02_events() -> pd.DataFrame:
    """Eight events whose shares and median offsets fall half-way between two
    whole numbers: 1 of 8 SWR-contained (12.5 %), 5 of 8 non-SWR (62.5 %); the
    three overlapping events' median start offset is 2.5 ms and their median
    end offset -4.5 ms."""
    nan = np.nan
    return pd.DataFrame(
        {
            "animal": "a",
            "session_date": "d1",
            "epoch": "e1",
            "overlap_class": ["swr_contained", "swr_partial", "swr_partial"]
            + ["non_swr"] * 5,
            "visit_context": "in_visit",
            "whole_event_speed_eligible": True,
            "event_start_s": [0.0025, -0.25, 1.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "event_end_s": [0.25, -0.0045, 2.0, 3.1, 4.1, 5.1, 6.1, 7.1],
            "matched_swr_start": [0.0, -0.125, 0.5, nan, nan, nan, nan, nan],
            "matched_swr_end": [0.5, 0.0, 1.5, nan, nan, nan, nan, nan],
        }
    )


def gain_statistics(no_swr: list[float], swr: list[float]) -> pd.DataFrame:
    """``holdout_statistics``' rows indexed by group, the animals as the test
    gives them, with each group's mean as its estimate and an interval of
    +/- 0.1."""
    rows = {}
    for group, animals in (("no SWR", no_swr), ("SWR", swr)):
        estimate = float(np.mean(animals))
        rows[group] = {
            "animals": np.asarray(animals),
            "animal_names": np.array(list("abcde"[: len(animals)])),
            "estimate": estimate,
            "ci_low": estimate - 0.1,
            "ci_high": estimate + 0.1,
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("group")


def measure_rows(measures: dict[str, list[float]]) -> pd.DataFrame:
    """Statistics rows indexed by family and measure: the filter family's
    per-animal values of each measure, their mean as the estimate and an
    interval of +/- a tenth of the largest value (the two-sided family's
    twice them)."""
    rows = {}
    for family, scale in (("filter", 1.0), ("two_sided", 2.0)):
        for measure, animals in measures.items():
            values = scale * np.asarray(animals)
            half = 0.1 * float(np.max(np.abs(values)))
            rows[(family, measure)] = {
                "animals": values,
                "animal_names": np.array(list("abcde"[: len(values)])),
                "estimate": float(values.mean()),
                "ci_low": float(values.mean()) - half,
                "ci_high": float(values.mean()) + half,
            }
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.names = ["family", "measure"]
    return table


OUTSIDE = [0.029, 0.02, 0.025, 0.038, 0.022]
INSIDE = [-0.02, -0.015, -0.025, -0.03, -0.018]
IN_EVENTS = [0.38, 0.39, 0.23, 0.42, 0.40]
OUTSIDE_EVENTS = [-0.003, -0.004, -0.005, -0.002, -0.004]
"""The full minus the SWR-only decoder per spike of each animal: outside SWRs,
inside SWRs, in the no-SWR held-out events and outside every held-out event."""


def outside_rows() -> pd.DataFrame:
    """``outside_swr_gain_statistics``' rows the panels read."""
    return measure_rows(
        {"gain_per_spike": OUTSIDE, "outside_events_gain_per_spike": OUTSIDE_EVENTS}
    )


def restriction_statistics():
    """``restriction_totals.restriction_figure_statistics``' tables Figure 2k
    reads: the events' row indexed by group, as the window statistics are."""
    from continuum_swr_replay.analysis.event_evidence.restriction_totals import (
        RestrictionFigureStatistics,
    )

    events = measure_rows({"no SWR": IN_EVENTS}).loc["filter"]
    events.index.name = "group"
    return RestrictionFigureStatistics(
        inside=measure_rows({"gain_per_spike": INSIDE}),
        outside=outside_rows(),
        events=events,
    )


GAIN_SPLIT = {
    "gain_per_spike": ([0.52, 0.47, 0.25, 0.43, 0.46], [0.86, 0.8, 0.7, 0.75, 0.9]),
    "spiking_gain_per_spike": ([0.3, 0.2, 0.1, 0.2, 0.25], [0.6, 0.5, 0.4, 0.5, 0.6]),
    "silent_gain_per_spike": ([0.2, 0.25, 0.15, 0.2, 0.2], [0.25, 0.3, 0.3, 0.2, 0.3]),
}
"""The left-out tetrode's gain over the animal's place and its two parts: each
part's animals, no SWR then SWR."""


def dots_at(ax, x: float):
    """The one scatter of ``ax`` whose dots are spread about ``x``."""
    (dots,) = [
        c
        for c in ax.collections
        if isinstance(c, matplotlib.collections.PathCollection)
        and np.allclose(c.get_offsets()[:, 0], x, atol=0.2)
    ]
    return dots


def assert_spread_about(dots, x: float) -> None:
    """The dots sit symmetrically about ``x``, alternating sides by value."""
    xs, ys = dots.get_offsets()[:, 0], dots.get_offsets()[:, 1]
    assert np.mean(xs) == pytest.approx(x)
    by_value = xs[np.argsort(ys)] - x
    # neighbors in value never on the same side, the middle one on x
    assert all(a * b <= 0 for a, b in zip(by_value, by_value[1:], strict=False))
    assert by_value[len(by_value) // 2] == pytest.approx(0.0)


def means_at(ax, x: float) -> list:
    """The y of the lines (the means' markers) beside the dots at ``x``."""
    return [
        line.get_ydata() for line in ax.lines if np.allclose(line.get_xdata(), x + 0.25)
    ]


def has_zero_line(ax) -> bool:
    """Whether ``ax`` draws a dashed line at zero."""
    return any(
        line.get_linestyle() == "--" and np.allclose(line.get_ydata(), 0)
        for line in ax.lines
    )


def texts(ax) -> list[str]:
    """Every text drawn on ``ax``: its texts and annotations."""
    return [t.get_text() for t in ax.texts]
