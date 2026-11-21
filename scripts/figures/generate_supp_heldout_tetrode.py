#!/usr/bin/env python
"""Generate the supplementary figure: the held-out tetrode gain over local coding.

The immobile non-local events of the canonical event table that start inside a
well visit (``cohort_selection.select_visit_context``), split by whether they
overlap an SWR. Each tetrode is left out of the decoder's smoother in turn
and scored at every bin of each event under the other tetrodes' position
distribution against local coding around the animal (time-fold encodings); the
gain is summed over tetrodes and bins and divided by the event's spikes. A dot
per animal for no-SWR and SWR events, the animals' mean with its 95 % interval,
and a line at zero, where local coding predicts the held-out spikes as well.
The line is not a control: the Local-rest controls sit there by construction
(in Local rest the decoded position is the animal's own), so they are in the
events summary and not drawn.

The statistics are ``tetrode_holdout.holdout_statistics`` of the held-out
tetrode run (``scripts/event_evidence/score_tetrode_holdout.py``;
``Processed-Data/event_tetrode_holdout``) over the epochs the events summary
reads: the values its ``*_heldout_*`` keys hold (``paper/reported/events.json``).
It was a panel of Figure 2; Figure 2g now compares the full decoder with the
SWR-only decoder.

Usage
-----
    python scripts/figures/generate_supp_heldout_tetrode.py
"""

import argparse
import logging
import warnings
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import continuum_swr_replay.visualization.figure_utilities as figure_utilities
from continuum_swr_replay.analysis.event_definition.analyzed_time import (
    read_swr_table,
)
from continuum_swr_replay.analysis.event_definition.cohort_selection import (
    select_events_in,
    select_visit_context,
)
from continuum_swr_replay.analysis.event_definition.swr_source import (
    event_table_epochs,
)
from continuum_swr_replay.analysis.event_evidence.tetrode_holdout import (
    holdout_statistics,
    window_scores,
)
from continuum_swr_replay.data_loaders.epoch_register import analyzed_epoch_ids
from continuum_swr_replay.paths import TETRODE_HOLDOUT_NAME
from continuum_swr_replay.reported.events import default_inputs
from continuum_swr_replay.runs.io import read_complete_run
from continuum_swr_replay.visualization.figure_utilities import save_figure
from continuum_swr_replay.visualization.heldout_gain_panels import (
    draw_group_gain,
    label_gain_axis,
)
from continuum_swr_replay.visualization.simulated_run_panels import (
    check_journal_limits,
    check_placement,
    set_publication_style,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%d-%b-%y %H:%M:%S"
)
figure_utilities.set_figure_defaults(context="paper")
set_publication_style()
warnings.filterwarnings("ignore")

# Inches: a third of a column's width is enough for one panel of two groups.
WIDTH = 1.6
HEIGHT = 1.9

TITLE = "Held-out tetrode"
YLABEL = "Held-out gain\nover local (nats/spike)"


def read_inputs() -> pd.DataFrame:
    """The held-out tetrode statistics of the events that start inside a well
    visit, indexed by group, over every analyzed epoch with an event, as the
    events summary reads them."""
    inputs = default_inputs()
    table = read_swr_table(inputs.event_table)
    scores = read_complete_run(
        inputs.tetrode_holdout,
        TETRODE_HOLDOUT_NAME,
        event_table_epochs(table, analyzed_epoch_ids()),
    )
    selected = select_events_in(scores, select_visit_context(table)["event_id"])
    return holdout_statistics(window_scores(selected), "event")


def draw_heldout_tetrode(ax, statistics: pd.DataFrame) -> None:
    """The held-out tetrode gain over local coding per spike, by group
    (``statistics``: :func:`read_inputs`)."""
    draw_group_gain(ax, statistics, TITLE)
    label_gain_axis(ax, YLABEL)


def generate_figure(statistics: pd.DataFrame):
    """Build the figure from :func:`read_inputs`."""
    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))
    fig.subplots_adjust(left=0.36, right=0.97, top=0.88, bottom=0.2)
    draw_heldout_tetrode(ax, statistics)
    fig.draw_without_rendering()
    check_placement(fig, {"a": [ax]})
    return fig


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("paper/figures/supp_heldout_tetrode")
    )
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig = generate_figure(read_inputs())
    check_journal_limits(fig)
    logging.info(f"Saving to {args.output_dir}...")
    save_figure(fig, "supp_heldout_tetrode", base_dir=args.output_dir)
    plt.close("all")


if __name__ == "__main__":
    main()
