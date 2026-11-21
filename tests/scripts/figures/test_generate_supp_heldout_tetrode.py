"""The held-out tetrode supplementary figure: what it draws and from which values."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from continuum_swr_replay.reported.questions import FIGURES  # noqa: E402
from scripts.figures import generate_supp_heldout_tetrode as figure  # noqa: E402
from tests.scripts.figures.test_figure_script_fixes import (  # noqa: E402
    _gain_statistics,
    assert_draws_group_gain,
)


def test_the_figure_draws_the_held_out_tetrode_gain_by_group():
    """One axis, as Figure 2h had it: a dot per animal and the mean with its
    interval for no-SWR and SWR events, the dashed zero line and "no gain"."""
    statistics = _gain_statistics([0.52, 0.47, 0.25, 0.43, 0.46], [0.86, 0.8, 0.7])
    fig = figure.generate_figure(statistics)
    (ax,) = fig.axes
    assert_draws_group_gain(ax, statistics)
    assert ax.get_title() == "Held-out tetrode"
    assert ax.get_ylabel() == "Held-out gain\nover local (nats/spike)"
    assert ax.get_ylim()[0] == pytest.approx(-0.12)
    figure.check_journal_limits(fig)
    plt.close(fig)


def test_the_figure_is_the_question_lists_held_out_tetrode_figure():
    """Values tagged ``heldout_tetrode`` name this script."""
    assert FIGURES["heldout_tetrode"].script == (
        "scripts/figures/generate_supp_heldout_tetrode.py"
    )
