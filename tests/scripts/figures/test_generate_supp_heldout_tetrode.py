"""The held-out tetrode supplementary figure: what it draws and from which values."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from continuum_swr_replay.reported.questions import FIGURES  # noqa: E402
from scripts.figures import generate_supp_heldout_tetrode as figure  # noqa: E402
from tests.scripts.figures.test_figure_script_fixes import (  # noqa: E402
    _figure02_events,
    _gain_statistics,
    assert_draws_group_gain,
)

TRIAL_FOLD_SETTINGS = {"n_folds": 5, "fold_design": "trials"}


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


def test_the_figure_reads_the_held_out_run_over_the_events_that_start_in_a_visit(
    monkeypatch,
):
    """The stored run (held out in folds of whole trials) scores every event;
    the figure, like the events summary, reads the rows (every tetrode's) of
    the events that start inside a well visit."""
    inside = _figure02_events().assign(event_id=range(8))
    outside = inside.iloc[[3]].assign(event_id=99, visit_context="track_leaf_arm")
    stored = pd.DataFrame({"event_id": [0, 0, *range(1, 8), 99, 99]})
    seen = []
    for name, value in {
        "read_output_manifest": lambda folder: {"settings": TRIAL_FOLD_SETTINGS},
        "read_swr_table": lambda path: pd.concat([inside, outside]),
        "event_table_epochs": lambda table, analyzed: ["e"],
        "analyzed_epoch_ids": lambda: ["e"],
        "read_complete_run": lambda folder, name, epochs: stored,
        "window_scores": lambda table: seen.append(table) or table,
        "holdout_statistics": lambda windows, kind: windows,
    }.items():
        monkeypatch.setattr(figure, name, value)
    figure.read_inputs()
    assert list(seen[0]["event_id"]) == [0, 0, *range(1, 8)]


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"n_folds": 5, "margin_bins": 2}, "fold_design None, not 'trials'"),
        (
            {"n_folds": 5, "fold_design": "contiguous"},
            "fold_design 'contiguous', not 'trials'",
        ),
    ],
    ids=["no fold design", "another design"],
)
def test_the_figure_refuses_a_held_out_run_not_on_trial_folds(
    monkeypatch, settings, message
):
    """A held-out tetrode run of an older fold design is refused before any
    table is read, as the events summary refuses it."""
    monkeypatch.setattr(
        figure, "read_output_manifest", lambda folder: {"settings": settings}
    )

    def read_swr_table(path):
        raise AssertionError("read before the refusal")

    monkeypatch.setattr(figure, "read_swr_table", read_swr_table)
    with pytest.raises(ValueError, match=message):
        figure.read_inputs()
