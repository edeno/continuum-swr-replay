#!/usr/bin/env python
"""Generate the supplementary figure: slow and fast gamma during SWR and no-SWR events.

The immobile non-local events of the canonical event table that start inside
a well visit (``cohort_selection.select_visit_context``), split by whether
they overlap an SWR; quiet rest is the rest inside a well visit. For each band of ``gamma_power.GAMMA_BANDS`` (a, slow
gamma; b, fast gamma) and each kind of event, every animal's level during the
events, between the levels of the same epochs' quiet-rest and running windows
as long as the kind's events. An event's gamma is a mean over its bins, and a
mean of skewed power over a window sits above the median of single bins, so the
reference is windows of the events' own length.

The animal values are ``event_statistics.animal_signal_levels`` with
``matched``: the values the events summary's ``<band>_animals_*`` counts compare
(``paper/reported/events.json``; the gamma run is
``scripts/event_properties/measure_event_gamma.py``). This was Figure 3's last
panel; ripple power, MUA, theta and head speed around the events are Figure 2's.

Usage
-----
    python scripts/figures/generate_supp_event_gamma.py
"""

import argparse
import logging
import warnings
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import continuum_swr_replay.visualization.figure_utilities as figure_utilities
from continuum_swr_replay.analysis.event_definition.analyzed_time import (
    read_swr_table,
)
from continuum_swr_replay.analysis.event_definition.cohort_selection import (
    select_visit_context,
    subset_staleness,
)
from continuum_swr_replay.analysis.event_properties.event_statistics import (
    animal_signal_levels,
)
from continuum_swr_replay.analysis.event_properties.gamma_power import GAMMA_BANDS
from continuum_swr_replay.data_loaders.epoch_register import analyzed_epoch_ids
from continuum_swr_replay.reported.events import default_inputs, with_brain_state
from continuum_swr_replay.runs.staleness import read_output_manifest
from continuum_swr_replay.visualization.figure_utilities import save_figure
from continuum_swr_replay.visualization.palettes import COLORS
from continuum_swr_replay.visualization.simulated_run_panels import (
    SMALL_SIZE,
    check_journal_limits,
    check_placement,
    drawn_extent,
    letter_panel,
    set_publication_style,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%d-%b-%y %H:%M:%S"
)
figure_utilities.set_figure_defaults(context="paper")
set_publication_style()
warnings.filterwarnings("ignore")

# Inches: a column and a half.
WIDTH = 4.4
HEIGHT = 2.3

# The two kinds of event, named and colored as in Figures 2 and 3.
KINDS = ((False, "No SWR", COLORS["No SWR"]), (True, "SWR", COLORS["SWR"]))
# The rest and running levels: neutral, not decoder-state colors.
C_REST = "white"
C_RUN = "black"
# Each panel: (band, event column, title).
SIGNALS = tuple(
    (band, f"{band}_z", f"{name} gamma ({low:.0f}–{high:.0f} Hz)")
    for name, (band, (low, high)) in zip(
        ("Slow", "Fast"), GAMMA_BANDS.items(), strict=True
    )
)


def read_inputs() -> dict:
    """The events with their gamma, and each epoch's rest and running levels.

    Raises
    ------
    ValueError
        If the brain-state or the gamma reference was not made over the
        analyses' subset (``cohort_selection.subset_staleness``).
    """
    inputs = default_inputs()
    for folder in (inputs.brain_state_reference.parent, inputs.event_gamma):
        note = subset_staleness(read_output_manifest(folder), str(folder))
        if note is not None:
            raise ValueError(note)
    events = select_visit_context(read_swr_table(inputs.event_table))
    events, reference, _ = with_brain_state(events, inputs, len(analyzed_epoch_ids()))
    return {"events": events, "reference": reference}


def draw_gamma(axes, events: pd.DataFrame, reference: pd.DataFrame) -> None:
    """Each animal's level of each band in each kind of event, between the
    levels of quiet-rest (white) and running (black) windows as long as the
    kind's events (``axes``: one per band)."""
    for ax, (band, column, title) in zip(axes, SIGNALS, strict=True):
        levels = animal_signal_levels(events, reference, band, column, matched=True)
        centers = []
        for k, (is_swr, _, color) in enumerate(KINDS):
            x = 3.7 * k + np.arange(3)
            values = levels.loc[is_swr, ["rest", "event", "running"]]
            for _, row in values.iterrows():
                ax.plot(x, row.to_numpy(), color="#999999", lw=0.5, zorder=1)
            for i, dot_color in enumerate((C_REST, color, C_RUN)):
                ax.scatter(
                    np.full(len(values), x[i]), values.iloc[:, i], s=9,
                    color=dot_color, edgecolors="black", linewidths=0.5, zorder=2,
                )  # fmt: skip
            centers.append(x[1])
        ax.axhline(0, color="0.85", lw=0.5, zorder=0)
        ax.set_title(title, fontsize=SMALL_SIZE)
        # One label per kind; the key says which dots are rest and running.
        ax.set_xticks(centers, [name for _, name, _ in KINDS])
        ax.set_xlim(-0.7, centers[-1] + 1.7)
    axes[0].set_ylabel("Power (z)")


def generate_figure(inputs: dict):
    """Build the figure from :func:`read_inputs`."""
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, HEIGHT))
    fig.subplots_adjust(left=0.13, right=0.99, top=0.76, bottom=0.12, wspace=0.35)
    draw_gamma(axes, inputs["events"], inputs["reference"])
    handles = [
        plt.Line2D([], [], marker="o", ls="", ms=3, mfc=color, mec="black", mew=0.5)
        for color in (C_REST, C_RUN)
    ]
    fig.legend(
        handles, ["quiet-rest windows", "running windows"], frameon=False,
        loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, handletextpad=0.1,
        columnspacing=1.5,
    )  # fmt: skip
    fig.draw_without_rendering()
    top = max(drawn_extent(fig, [ax]).y1 for ax in axes) + 0.02
    for ax, letter in zip(axes, "ab", strict=True):
        letter_panel(fig, letter, max(drawn_extent(fig, [ax]).x0 - 0.004, 0.0), top)
    check_placement(fig, {"a": [axes[0]], "b": [axes[1]]})
    return fig


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("paper/figures/supp_event_gamma")
    )
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig = generate_figure(read_inputs())
    check_journal_limits(fig)
    logging.info(f"Saving to {args.output_dir}...")
    save_figure(fig, "supp_event_gamma", base_dir=args.output_dir)
    plt.close("all")


if __name__ == "__main__":
    main()
