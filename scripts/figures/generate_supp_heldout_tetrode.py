#!/usr/bin/env python
"""Generate the supplementary figure: the held-out tetrode gain, and the gain
outside SWRs.

    a — The held-out tetrode gain over local coding. The immobile non-local
        events of the canonical event table that start inside a well visit
        (``cohort_selection.select_visit_context``), split by whether they
        overlap an SWR. Each tetrode is left out of the decoder's smoother in
        turn and scored at every bin of each event under the other tetrodes'
        position distribution against local coding around the animal
        (encodings held out in folds of whole trials); the gain is summed over
        tetrodes and bins and divided by the event's spikes. A dot per animal
        for no-SWR and SWR events, the animals' mean with its 95 % interval,
        and a line at zero, where local coding predicts the held-out spikes as
        well. The line is not a control: the Local-rest controls sit there by
        construction (in Local rest the decoded position is the animal's own),
        so they are in the events summary and not drawn
        (``tetrode_holdout.holdout_statistics`` of the held-out tetrode run,
        ``scripts/event_evidence/score_tetrode_holdout.py``;
        ``Processed-Data/event_tetrode_holdout``; the events summary's
        ``*_heldout_*`` values).

    b to d — Figure 2g's main result, the full decoder's gain over the SWR-only
    decoder outside SWRs, looked at more closely. Every value is over
    "Immobile, at wells" (the heading over the three): the immobile bins inside
    a well visit, which runs from the animal's first nose-poke at a reward well
    to its last poke-out there. Each is per epoch, summed over its bins and
    divided by the spikes in them (held-out log score, the whole decoder fit
    on folds of whole trials and each fold scored under the fit that did not
    see it); a dot per animal is the median over its epochs, beside the
    animals' mean with its 95 % interval.

    b — That gain over every immobile bin outside SWRs ("Outside SWRs",
        ``restriction_totals.outside_swr_gain_statistics``), then over only
        the bins farther than 100, 250 and 500 ms from every SWR
        (``restriction_totals.outside_swr_subset_statistics``, the
        ``beyond_swr_margin`` parts), a thin line through the means: the gain
        is not SWR edges.
    c — The same gain over every bin outside SWRs, then over only the bins
        whose place (the animal's place bin, under the bin's own fold) has at
        least 10 and at least 60 s of training data (the ``place_sampling``
        parts): the gain is not poorly sampled places.
    d — Where that gain outside SWRs sits, as a share of it per epoch: in the
        events with no SWR, in the time outside the SWR of the events partly in
        one, and outside those events (1 minus the other two, so the three add
        to 100 %; below zero, as the full decoder does slightly worse there),
        each named beside its dots in panel d's colors of Figure 2
        (``outside_swr_gain_statistics``' shares). The events are those
        detected from the held-out decode that start inside a well visit, the
        full decoder's own events. Outside SWRs the SWR-only decoder is the
        local-only decoder, so the full decoder can gain only where it goes
        non-local, which is where its events are: the shares follow from how
        the events are defined, and so describe where the gain sits and do not
        test the events.

Panel a reads the held-out tetrode run over the epochs the events summary reads
and refuses a run not held out in folds of whole trials, as the summary does.
Panels b to d read the held-out event run
(``Processed-Data/event_nonlocal_restriction_heldout_events``, its events and
time sums) by the events summary's reader
(``reported.events.read_heldout_restriction``), and refuse it, as Figure 2g
does, when the restriction run is not the fold fit on folds of whole trials or
was not made from the current decode frames and SWR source, when the held-out
event run was not made from the current restriction run or per-bin facts run,
or when the per-bin facts were not made from the current session cache. Their
values are the events summary's ``restriction_outside_swr_*`` values.

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
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

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
from continuum_swr_replay.analysis.event_evidence.restriction_totals import (
    OUTSIDE_SWR_MARGINS_S,
    PLACE_SAMPLING_THRESHOLDS_S,
    RESTRICTION_FIGURE_FAMILY,
    RESTRICTION_FIGURE_OUTSIDE_SWR,
    epoch_outside_swr_gains,
    outside_swr_gain_statistics,
    outside_swr_subset_statistics,
)
from continuum_swr_replay.analysis.event_evidence.tetrode_holdout import (
    holdout_statistics,
    window_scores,
)
from continuum_swr_replay.data_loaders.epoch_register import analyzed_epoch_ids
from continuum_swr_replay.paths import TETRODE_HOLDOUT_NAME
from continuum_swr_replay.reported.events import (
    default_inputs,
    read_heldout_restriction,
    require_tetrode_holdout_folds,
    restriction_inputs_note,
)
from continuum_swr_replay.reported.values import rel
from continuum_swr_replay.runs.io import read_complete_run, sha256_file
from continuum_swr_replay.runs.staleness import read_output_manifest
from continuum_swr_replay.visualization.figure_utilities import save_figure
from continuum_swr_replay.visualization.heldout_gain_panels import (
    DIFFERENCE_YLABEL,
    MEAN_OFFSET,
    NEUTRAL_DOTS,
    REFERENCE_COLOR,
    draw_animal_gain,
    draw_group_gain,
    draw_outside_swr_shares,
    full_better_cue,
    heading_over,
    label_gain_axis,
)
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

# Inches: the full page width, one row of four panels.
WIDTH = 6.5
HEIGHT = 2.1

TITLE = "Held-out tetrode"
YLABEL = "Held-out gain\nover local (nats/spike)"
HEADING = "Immobile, at wells"
"""The heading over panels b to d: their values are over the immobile time
inside a well visit."""
SHARES_YLABEL = "Share of the\noutside-SWR gain (%)"
"""Panel d's y label: a description of where the gain sits, not a test."""
DISTANCE_LABEL = "at least this far from\nany SWR (ms)"
SAMPLING_LABEL = "training data at the\nanimal's place (s, at least)"
OUTSIDE_LABEL = "Outside\nSWRs"
PARTS_LABEL_DROP = 21
"""How far below the axis a series' parts label starts (points): under the two
lines of "Outside SWRs", so that it can be wider than its columns."""
SERIES_START = 1.3
"""The x position of a series' first part, after the "Outside SWRs" column at 0;
the parts are 1 apart."""
LEFTS = (0.077, 0.31, 0.6, 0.845)
"""Each panel's frame's left edge (figure width)."""
FRAME_WIDTHS = (0.12, 0.18, 0.125, 0.045)
"""Each panel's frame's width (figure width); d's names reach to its right."""
BOTTOM, TOP = 0.3, 0.74
"""The frames' bottom and top (figure height)."""
LETTER_HEIGHT = 0.07
"""A panel letter's height (figure height): the letters' tops sit this far
above the highest drawn top, so that they stand above the heading."""
REFUSED = "the held-out tetrode figure's panels b to d are not drawn"
RERUN = {
    "restriction_inputs": "scripts/event_evidence/score_nonlocal_restriction.py "
    "and the runs made from it",
    "restriction_run": "scripts/event_evidence/detect_heldout_events.py",
    "bin_facts_run": "scripts/event_evidence/detect_heldout_events.py",
    "session_cache": "scripts/event_evidence/measure_bin_facts.py, then "
    "scripts/event_evidence/detect_heldout_events.py",
}
"""The runners to make again when panels b to d refuse their inputs."""


def read_inputs() -> dict:
    """The figure's statistics, as the events summary reads them.

    Returns
    -------
    dict
        ``tetrode``: the held-out tetrode statistics of the events that start
        inside a well visit, indexed by group (a held-out tetrode run not held
        out in folds of whole trials is refused,
        ``reported.events.require_tetrode_holdout_folds``); ``outside``:
        ``outside_swr_gain_statistics`` over the held-out event run, indexed by
        family and measure; ``parts``: ``outside_swr_subset_statistics`` over
        its time sums, indexed by subset, threshold, family and measure.

    Raises
    ------
    ValueError
        If the held-out event run, or what it was made from, is not current
        (:data:`RERUN` names what to make again).
    """
    inputs = default_inputs()
    require_tetrode_holdout_folds(
        read_output_manifest(inputs.tetrode_holdout), rel(inputs.tetrode_holdout)
    )
    table = read_swr_table(inputs.event_table)
    epochs = event_table_epochs(table, analyzed_epoch_ids())
    scores = read_complete_run(inputs.tetrode_holdout, TETRODE_HOLDOUT_NAME, epochs)
    selected = select_events_in(scores, select_visit_context(table)["event_id"])
    heldout = read_heldout_restriction(
        inputs.nonlocal_restriction_heldout_events,
        inputs.nonlocal_restriction,
        inputs.bin_facts,
        epochs,
        sha256_file(inputs.event_table),
    )
    for name, reason in (
        (
            "restriction_inputs",
            restriction_inputs_note(inputs.nonlocal_restriction, epochs),
        ),
        ("restriction_run", heldout.restriction_run),
        ("bin_facts_run", heldout.bin_facts_run),
        ("session_cache", heldout.session_cache),
    ):
        if reason is not None:
            cause = reason.split("; rerun ")[0]
            raise ValueError(f"{REFUSED}: {cause}. Rerun {RERUN[name]}.")
    return {
        "tetrode": holdout_statistics(window_scores(selected), "event"),
        "outside": outside_swr_gain_statistics(
            epoch_outside_swr_gains(heldout.time_sums, heldout.bin_s, heldout.events)
        ).set_index(["family", "measure"]),
        "parts": outside_swr_subset_statistics(
            heldout.time_sums, heldout.bin_s
        ).set_index(["subset", "threshold_s", "family", "measure"]),
    }


def draw_heldout_tetrode(ax: Axes, statistics: pd.DataFrame) -> None:
    """Panel a: the held-out tetrode gain over local coding per spike, by group
    (``statistics``: :func:`read_inputs`' ``tetrode``)."""
    draw_group_gain(ax, statistics, TITLE)
    label_gain_axis(ax, YLABEL)


def draw_outside_swr_series(
    ax: Axes,
    outside: pd.DataFrame,
    parts: pd.DataFrame,
    subset: str,
    thresholds: Sequence[float],
    names: Sequence[str],
    xlabel: str,
) -> None:
    """Panels b and c: the full decoder minus the SWR-only decoder per spike
    over every immobile bin outside SWRs, then over each part of them
    (``subset``, at each of ``thresholds``), a dot per animal and the mean
    with its interval, a thin line through the means, the parts' label under
    their tick labels (``xlabel``), a dashed zero line and which way the full
    decoder is better."""
    rows = [outside.loc[(RESTRICTION_FIGURE_FAMILY, RESTRICTION_FIGURE_OUTSIDE_SWR)]]
    rows += [
        parts.loc[(subset, threshold, RESTRICTION_FIGURE_FAMILY, "gain_per_spike")]
        for threshold in thresholds
    ]
    xs = [0.0, *(SERIES_START + k for k in range(len(thresholds)))]
    for x, row in zip(xs, rows, strict=True):
        draw_animal_gain(ax, x, row, NEUTRAL_DOTS, spread=True)
    ax.plot(
        [x + MEAN_OFFSET for x in xs], [row["estimate"] for row in rows],
        color=REFERENCE_COLOR, lw=0.5, zorder=1,
    )  # fmt: skip
    ax.axhline(0, color=REFERENCE_COLOR, lw=0.6, ls="--", zorder=1)
    ax.set_xlim(-0.45, xs[-1] + 0.6)
    ax.set_xticks(xs, [OUTSIDE_LABEL, *names])
    ax.set_ylabel(DIFFERENCE_YLABEL)
    full_better_cue(ax)
    # the parts' label, centered under their tick labels
    ax.annotate(
        xlabel, (float(np.mean(xs[1:])), 0), xycoords=("data", "axes fraction"),
        xytext=(0, -PARTS_LABEL_DROP), textcoords="offset points", ha="center",
        va="top",
        fontsize=SMALL_SIZE, annotation_clip=False,
    )  # fmt: skip


def shared_limits(axes: Sequence[Axes], floor: float = 0.13) -> None:
    """One y range for panels b and c, from 0 less a floor to a little above
    their highest dot or interval, with no tick label at the frames' top."""
    top = max(ax.get_ylim()[1] for ax in axes) * 1.05
    for ax in axes:
        ax.set_ylim(-floor * top, top)
        ax.set_yticks([t for t in ax.get_yticks() if 0 <= t <= 0.9 * top])


def generate_figure(statistics: dict) -> Figure:
    """Build the figure from :func:`read_inputs`."""
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    axes = [
        fig.add_axes((left, BOTTOM, width, TOP - BOTTOM))
        for left, width in zip(LEFTS, FRAME_WIDTHS, strict=True)
    ]
    ax_a, ax_b, ax_c, ax_d = axes
    draw_heldout_tetrode(ax_a, statistics["tetrode"])
    draw_outside_swr_series(
        ax_b, statistics["outside"], statistics["parts"], "beyond_swr_margin",
        OUTSIDE_SWR_MARGINS_S, [f"{t * 1000:g}" for t in OUTSIDE_SWR_MARGINS_S],
        DISTANCE_LABEL,
    )  # fmt: skip
    draw_outside_swr_series(
        ax_c, statistics["outside"], statistics["parts"], "place_sampling",
        PLACE_SAMPLING_THRESHOLDS_S, [f"{t:g}" for t in PLACE_SAMPLING_THRESHOLDS_S],
        SAMPLING_LABEL,
    )  # fmt: skip
    shared_limits([ax_b, ax_c])
    draw_outside_swr_shares(ax_d, statistics["outside"], SHARES_YLABEL)
    fig.draw_without_rendering()
    # each panel against the others before the heading, which b draws over c
    # and d; after it, the heading against everything as one group's
    check_placement(fig, {"a": [ax_a], "b": [ax_b], "c": [ax_c], "d": [ax_d]})
    heading_over(fig, fig.canvas.get_renderer(), [ax_b, ax_c, ax_d], HEADING)
    fig.draw_without_rendering()
    # the letters of the one row at one height, just above the highest panel
    # (b's, which holds the heading), each at its panel's left edge
    top = max(drawn_extent(fig, [ax]).y1 for ax in axes) + LETTER_HEIGHT
    for ax, letter in zip(axes, "abcd", strict=True):
        letter_panel(fig, letter, max(drawn_extent(fig, [ax]).x0 - 0.004, 0.0), top)
    check_placement(fig, {"a": [ax_a], "b to d": [ax_b, ax_c, ax_d]})
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
