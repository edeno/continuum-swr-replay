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
