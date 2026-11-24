#!/usr/bin/env python
"""Generate Supplementary Figure S3, the supplement to Figure 3: slow and fast
gamma during SWR and no-SWR events, and how fast their position distribution
moves.

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

c — How fast the position distribution moves, no-SWR against SWR events: each
    event's transport-slope speed (``event_speed.transport_slope_speed``; the
    event-speed run, ``scripts/event_properties/measure_event_speed.py``), the
    slope, at least 0, of the earth mover's distance on the track between the
    Continuous-state position distributions of two of its time bins against
    their lag, over lags of 10 to 20 ms. Figure 3d draws the speed of the MAP
    position; this one reads the whole distribution and sets aside the
    bin-to-bin difference that does not grow with the lag. The percent of a
    class's events at or below each speed (the bins of
    ``event_statistics.TRANSPORT_SPEED_BIN_EDGES_CM_S`` summed), a thin line
    per animal and their mean in bold
    (``event_statistics.transport_speed_distribution``), over the events that
    have a speed: one too short for three lags, or with too few time bins that
    are mostly Continuous, has none.

Usage
-----
    python scripts/figures/generate_supp_figure03.py
"""

import argparse
import logging
import warnings
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

import continuum_swr_replay.visualization.figure_utilities as figure_utilities
from continuum_swr_replay.analysis.event_definition.analyzed_time import (
    read_swr_table,
)
from continuum_swr_replay.analysis.event_definition.cohort_selection import (
    select_visit_context,
)
from continuum_swr_replay.analysis.event_properties.event_statistics import (
    TRANSPORT_SPEED_BIN_EDGES_CM_S,
    animal_signal_levels,
    transport_speed_distribution,
)
from continuum_swr_replay.analysis.event_properties.gamma_power import GAMMA_BANDS
from continuum_swr_replay.analysis.event_tables.prepare_data import (
    time_mask_staleness,
)
from continuum_swr_replay.data_loaders.epoch_register import analyzed_epoch_ids
from continuum_swr_replay.reported.events import (
    default_inputs,
    with_brain_state,
    with_event_speed,
)
from continuum_swr_replay.runs.staleness import read_output_manifest
from continuum_swr_replay.visualization.figure_utilities import save_figure
from continuum_swr_replay.visualization.heldout_gain_panels import (
    GROUP_COLORS,
    name_groups,
)
from continuum_swr_replay.visualization.palettes import ANIMAL_LINE_COLOR, COLORS
from continuum_swr_replay.visualization.panel_layout import place_letters
from continuum_swr_replay.visualization.simulated_run_panels import (
    SMALL_SIZE,
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

# Inches: the full page width.
WIDTH = 6.5
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
    """The events with their gamma, each epoch's rest and running levels, and
    the selected events' ``transport_speed_distribution`` (``speed``).

    Raises
    ------
    ValueError
        If the brain-state or the gamma reference was not made over the
        analyses' subset or from the current session cache's well visits
        (``prepare_data.time_mask_staleness``).
    """
    inputs = default_inputs()
    events = select_visit_context(read_swr_table(inputs.event_table))
    selected = events
    events, reference, gamma_reference = with_brain_state(
        events, inputs, len(analyzed_epoch_ids())
    )
    for folder, levels in (
        (inputs.brain_state_reference.parent, reference),
        (inputs.event_gamma, gamma_reference),
    ):
        note = time_mask_staleness(
            read_output_manifest(folder), levels["epoch_id"], str(folder)
        )
        if note is not None:
            raise ValueError(note)
    # panel c: the selected events, as the events summary reads them
    with_speed, _ = with_event_speed(
        selected, inputs.event_speed, len(analyzed_epoch_ids())
    )
    return {
        "events": events,
        "reference": reference,
        "speed": transport_speed_distribution(with_speed),
    }


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
                ax.plot(x, row.to_numpy(), color=ANIMAL_LINE_COLOR, lw=0.5, zorder=1)
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


SPEED_TICKS_CM_S = (0, 500, 1000)


def draw_transport_speed(ax: Axes, distribution: pd.DataFrame) -> None:
    """Panel c, the speed panel: for no-SWR and SWR events, the percent of the
    class's events (with a speed) whose transport-slope speed is at most each
    speed, a thin line per animal and their mean in bold in the class's color
    (``distribution``: :func:`read_inputs`' ``speed``)."""
    for group, color in GROUP_COLORS.items():
        mine = distribution[distribution["group"] == group]
        by_animal = mine.pivot(index="bin_high", columns="animal", values="percent")
        within = by_animal.cumsum()
        edges = within.index.to_numpy()
        ax.plot(edges, within.to_numpy(), color=color, lw=0.5, alpha=0.5)
        ax.plot(edges, within.mean(axis=1).to_numpy(), color=color, lw=1.2)
    name_groups(ax, 0.98, 0.3, 0.1, "right")
    ax.set_xlim(TRANSPORT_SPEED_BIN_EDGES_CM_S[0], TRANSPORT_SPEED_BIN_EDGES_CM_S[-1])
    ax.set_xticks(SPEED_TICKS_CM_S)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 50, 100])
    ax.set_xlabel("Speed (cm/s)")
    ax.set_ylabel("Events at most\nthis fast (%)")
    ax.set_title("Distribution's speed", fontsize=SMALL_SIZE)


def generate_figure(inputs: dict):
    """Build the figure from :func:`read_inputs`."""
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, HEIGHT))
    fig.subplots_adjust(left=0.09, right=0.99, top=0.76, bottom=0.2, wspace=0.45)
    draw_gamma(axes[:2], inputs["events"], inputs["reference"])
    draw_transport_speed(axes[2], inputs["speed"])
    # the key of the gamma panels, centered over them
    middle = (axes[0].get_position().x0 + axes[1].get_position().x1) / 2
    handles = [
        plt.Line2D([], [], marker="o", ls="", ms=3, mfc=color, mec="black", mew=0.5)
        for color in (C_REST, C_RUN)
    ]
    fig.legend(
        handles, ["quiet-rest windows", "running windows"], frameon=False,
        loc="upper center", bbox_to_anchor=(middle, 1.0), ncol=2, handletextpad=0.1,
        columnspacing=1.5,
    )  # fmt: skip
    fig.draw_without_rendering()
    panels = dict(zip("abc", ([ax] for ax in axes), strict=True))
    place_letters(fig, panels, rows=["abc"], above=0.02, left_of=0.004)
    check_placement(fig, panels)
    return fig


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("paper/figures/supp_figure03")
    )
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig = generate_figure(read_inputs())
    check_journal_limits(fig)
    logging.info(f"Saving to {args.output_dir}...")
    save_figure(fig, "supp_figure03", base_dir=args.output_dir)
    plt.close("all")


if __name__ == "__main__":
    main()
