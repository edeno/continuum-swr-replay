#!/usr/bin/env python
"""Generate Supplementary Figure S2, the supplement to Figure 2, its panels in
the order the text cites them: MUA in the No-Spike state, the held-out tetrode
gain, shared content, and the gain outside SWRs.

    a — Supports Figure 2a, where the No-Spike state first appears. Multiunit
        activity (``MUA_zscore``, as Figure 2g draws it) around
        the start of No-Spike periods: the median over periods with the
        interquartile band, a dotted line at the start and a dashed line at the
        MUA level of immobile time outside the periods
        (``no_spike_periods.level_outside_periods``, each epoch's median, then
        the median over epochs; Figure 2g's "between events" level is outside
        every non-local, SWR and MUA event, which does not carry over). A
        period is the non-local events' rule applied to the No-Spike state
        (``no_spike_periods.no_spike_periods``: P(No-Spike) strictly above 0.8
        for at least 7 samples, the animal immobile throughout), wholly outside
        excluded time and starting inside a well visit, read from every
        analyzed epoch's decode frame in one pass; the windows are Figure 2g's
        (0.3 s in all, ``triggered_analysis.get_signal_windows``, a window
        that touches excluded time left out). The decoder enters the No-Spike
        state when the decoded tetrodes fall silent, so the fall in their MUA
        describes the state and is not a test of it.

    b and c — Support Figure 2h, the events driven by several tetrodes. Each
    tetrode is left out of the decoder in turn: the other tetrodes' estimate
    predicts its spikes better than local coding does (b), and in the events
    where that estimate is specific, its activity is where the others point (c).

    b — The held-out tetrode gain over local coding. The immobile non-local
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
    c — Shared content: during an event, do the tetrodes point to the same
        remote place? The same events and run as b, each left-out tetrode
        scored under the other tetrodes' position distribution against no
        place preference (the agreement; above zero only when the left-out
        tetrode's activity is where the others point), summed over tetrodes and
        bins and divided by the event's spikes, as the gain is. The events are
        grouped by their spread, the mean over their left-out tetrodes of the
        mean over the event's bins of the effective width of the other
        tetrodes' distribution, in bins fixed in advance
        (``shared_content.SPREAD_BINS_CM``). A dot per animal for no-SWR and
        SWR events in their colors, beside them the animals' mean with its
        95 % interval, a thin line through each group's means and a dashed line
        at zero, no place preferred (``shared_content.agreement_by_spread_statistics``;
        the events summary's ``*_shared_content_*`` values).

    d and e — Support Figure 2j: its main result, the full decoder's gain over the
    SWR-only decoder outside SWRs, looked at more closely. Every value is over
    the immobile bins inside a well visit, which runs from the animal's first
    nose-poke at a reward well to its last poke-out there. Each is per epoch,
    summed over its bins and divided by the spikes in them (held-out log score, the whole decoder fit
    on folds of whole trials and each fold scored under the fit that did not
    see it); a dot per animal is the median over its epochs, beside the
    animals' mean with its 95 % interval.

    d — That gain over every immobile bin outside SWRs ("Outside SWRs",
        ``restriction_totals.outside_swr_gain_statistics``), then over only
        the bins farther than 100, 250 and 500 ms from every SWR
        (``restriction_totals.outside_swr_subset_statistics``, the
        ``beyond_swr_margin`` parts), a thin line through the means: the gain
        is not SWR edges. The same gain over only the bins whose place has at
        least 10 and at least 60 s of training data (the ``place_sampling``
        parts) is a reported value for the text, not drawn
        (``restriction_outside_swr_sampled_*``).
    e — Where that gain outside SWRs sits, as a share of it per epoch: in the
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

Panels b and c read the held-out tetrode run over the epochs the events
summary reads and refuse a run not held out in folds of whole trials, as the summary does.
Panels d and e read the held-out event run
(``Processed-Data/event_nonlocal_restriction_heldout_events``, its events and
time sums) by the events summary's reader
(``reported.events.read_heldout_restriction``), and refuse it, as Figure 2j
does, when the restriction run is not the fold fit on folds of whole trials or
was not made from the current decode frames and SWR source, when the held-out
event run was not made from the current restriction run or per-bin facts run,
or when the per-bin facts were not made from the current session cache. Their
values are the events summary's ``restriction_outside_swr_*`` values. Panel a
reads every analyzed epoch's decode frame (``Processed-Data``), which takes a
few minutes; it has no reported value.

Usage
-----
    python scripts/figures/generate_supp_figure02.py
"""

import argparse
import logging
import string
import warnings
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.transforms import blended_transform_factory

import continuum_swr_replay.visualization.figure_utilities as figure_utilities
from continuum_swr_replay.analysis.event_definition.analyzed_time import (
    read_dot_with_swr_source,
    read_swr_table,
)
from continuum_swr_replay.analysis.event_definition.cohort_selection import (
    select_events_in,
    select_visit_context,
)
from continuum_swr_replay.analysis.event_definition.no_spike_periods import (
    level_outside_periods,
    no_spike_periods,
)
from continuum_swr_replay.analysis.event_definition.swr_source import (
    event_table_epochs,
)
from continuum_swr_replay.analysis.event_evidence.restriction_totals import (
    OUTSIDE_SWR_MARGINS_S,
    RESTRICTION_FIGURE_FAMILY,
    RESTRICTION_FIGURE_OUTSIDE_SWR,
    epoch_outside_swr_gains,
    outside_swr_gain_statistics,
    outside_swr_subset_statistics,
)
from continuum_swr_replay.analysis.event_evidence.shared_content import (
    SPREAD_BIN_LABELS,
    agreement_by_spread_statistics,
)
from continuum_swr_replay.analysis.event_evidence.tetrode_holdout import (
    holdout_statistics,
    window_scores,
)
from continuum_swr_replay.analysis.event_properties.triggered_analysis import (
    get_signal_windows,
    window_quartiles,
)
from continuum_swr_replay.analysis.event_tables.prepare_data import well_visit_times
from continuum_swr_replay.data_loaders.epoch_register import (
    analyzed_decode_files,
    analyzed_epoch_ids,
)
from continuum_swr_replay.decoding.files import data_over_time_suffix
from continuum_swr_replay.paths import TETRODE_HOLDOUT_NAME, processed_data_dir
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
    GAIN_FLOOR,
    GROUP_COLORS,
    MEAN_OFFSET,
    NEUTRAL_DOTS,
    REFERENCE_COLOR,
    draw_animal_gain,
    draw_group_gain,
    draw_outside_swr_shares,
    full_better_cue,
    label_gain_axis,
    spread_offsets,
)
from continuum_swr_replay.visualization.palettes import COLORS, readable_text_color
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

# Inches: the full page width, two rows: a and b, then c, d and e.
WIDTH = 6.5
HEIGHT = 4.2

TITLE = "Held-out tetrode"
YLABEL = "Held-out gain\nover local (nats/spike)"
SHARED_YLABEL = "Held-out agreement\n(nats/spike)"
SHARED_XLABEL = "Spread of the other tetrodes'\nestimate (cm)"
SHARED_OFFSET = 0.2
"""Panel c: each group's dots sit this far either side of its spread bin; the
mean with its interval sits a further :data:`SHARED_MEAN_OFFSET` out."""
SHARED_MEAN_OFFSET = 0.11
SHARES_YLABEL = "Share of the\noutside-SWR gain (%)"
"""Panel e's y label: a description of where the gain sits, not a test."""
DISTANCE_LABEL = "at least this far from\nany SWR (ms)"
OUTSIDE_LABEL = "Outside\nSWRs"
PARTS_LABEL_DROP = 21
"""How far below the axis a series' parts label starts (points): under the two
lines of "Outside SWRs", so that it can be wider than its columns."""
SERIES_START = 1.3
"""The x position of a series' first part, after the "Outside SWRs" column at 0;
the parts are 1 apart."""
PANELS = (
    "no_spike_mua",
    "heldout_gain",
    "shared_content",
    "outside_distance",
    "outside_shares",
)
"""The panels in the order the text cites them; their letters follow from it."""
LETTERS = dict(zip(PANELS, string.ascii_lowercase, strict=False))
ROWS = (
    ((0.6, 0.9), ("no_spike_mua", "heldout_gain", "shared_content")),
    ((0.155, 0.41), ("outside_distance", "outside_shares")),
)
"""Each row's frames' bottom and top (figure height) and its panels, left to
right."""
FRAMES = {
    "no_spike_mua": (0.075, 0.155),
    "heldout_gain": (0.34, 0.1),
    "shared_content": (0.56, 0.42),
    "outside_distance": (0.075, 0.2),
    "outside_shares": (0.41, 0.045),
}
"""Each panel's frame's left edge and width (figure width); the shares' names
reach to the frame's right."""
LETTER_HEIGHT = 0.035
"""A panel letter's height (figure height): the letters' tops sit this far
above the highest drawn top of their row."""
NO_SPIKE_SIGNAL = "MUA_zscore"
TRIGGER_WINDOW_S = 0.3
"""Panel a's window around each start (s, in all), Figure 2g's: the periods
last a few tens of ms."""
NO_SPIKE_COLOR = COLORS["No-Spike"]
REFUSED = (
    f"Figure 2's supplement: panels {LETTERS['outside_distance']} and "
    f"{LETTERS['outside_shares']} are not drawn"
)
RERUN = {
    "restriction_inputs": "scripts/event_evidence/score_nonlocal_restriction.py "
    "and the runs made from it",
    "restriction_run": "scripts/event_evidence/detect_heldout_events.py",
    "bin_facts_run": "scripts/event_evidence/detect_heldout_events.py",
    "session_cache": "scripts/event_evidence/measure_bin_facts.py, then "
    "scripts/event_evidence/detect_heldout_events.py",
}
"""The runners to make again when the panels of the gain outside SWRs refuse
their inputs."""


def read_inputs() -> dict:
    """The figure's statistics, as the events summary reads them.

    Returns
    -------
    dict
        ``tetrode``: the held-out tetrode statistics of the events that start
        inside a well visit, indexed by group (a held-out tetrode run not held
        out in folds of whole trials is refused,
        ``reported.events.require_tetrode_holdout_folds``); ``shared``: their
        agreement by spread bin (``agreement_by_spread_statistics``); ``outside``:
        ``outside_swr_gain_statistics`` over the held-out event run, indexed by
        family and measure; ``parts``: ``outside_swr_subset_statistics`` over
        its time sums, indexed by subset, threshold, family and measure;
        ``no_spike``: :func:`read_no_spike_trace`.

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
    windows = window_scores(selected)
    return {
        "tetrode": holdout_statistics(windows, "event"),
        "shared": agreement_by_spread_statistics(windows[windows["kind"] == "event"]),
        "outside": outside_swr_gain_statistics(
            epoch_outside_swr_gains(heldout.time_sums, heldout.bin_s, heldout.events)
        ).set_index(["family", "measure"]),
        "parts": outside_swr_subset_statistics(
            heldout.time_sums, heldout.bin_s
        ).set_index(["subset", "threshold_s", "family", "measure"]),
        "no_spike": read_no_spike_trace(),
    }


def read_no_spike_trace() -> dict:
    """Panel a's trace: MUA around the start of every analyzed epoch's No-Spike
    periods, in one pass over the decode frames (the SWR source's reader).

    Returns
    -------
    dict
        ``time`` (s from the start), ``q25``, ``median`` and ``q75`` over the
        periods with a window (``triggered_analysis.window_quartiles``);
        ``level``: the median over epochs of each epoch's MUA level outside its
        periods; ``periods``: the periods with a window, by animal;
        ``median_duration``: their median duration (s).
    """
    suffix = data_over_time_suffix()
    windows, levels, counts, durations = [], [], {}, []
    for path in analyzed_decode_files(processed_data_dir(), suffix, require_all=True):
        frame = read_dot_with_swr_source(path)
        arrival, departure = well_visit_times(path.name.removesuffix(suffix))
        periods = no_spike_periods(frame, arrival, departure)
        levels.append(
            level_outside_periods(frame, periods, arrival, departure, NO_SPIKE_SIGNAL)
        )
        if periods.empty:
            continue
        session = frame[["animal", "day", "epoch"]].iloc[0]
        epoch = get_signal_windows(
            frame, periods.assign(**session.to_dict()), [NO_SPIKE_SIGNAL],
            window_sec=TRIGGER_WINDOW_S,
        )  # fmt: skip
        windows.append(epoch.sel(signal=NO_SPIKE_SIGNAL).values)
        counts[session["animal"]] = (
            counts.get(session["animal"], 0) + epoch.sizes["event"]
        )
        durations.append(periods["duration"].to_numpy())
    stacked = np.concatenate(windows)
    return {
        "time": epoch["time"].values,
        **window_quartiles(stacked),
        "level": float(np.nanmedian(levels)),
        "periods": pd.Series(counts, name="periods"),
        "median_duration": float(np.median(np.concatenate(durations))),
    }


def draw_heldout_tetrode(ax: Axes, statistics: pd.DataFrame) -> None:
    """Panel b: the held-out tetrode gain over local coding per spike, by group
    (``statistics``: :func:`read_inputs`' ``tetrode``)."""
    draw_group_gain(ax, statistics, TITLE)
    label_gain_axis(ax, YLABEL)


def draw_shared_content(ax: Axes, statistics: pd.DataFrame) -> None:
    """Panel c: the held-out agreement per spike by spread bin, for no-SWR and
    SWR events (``statistics``: :func:`read_inputs`' ``shared``). Each group's
    dots (one per animal) in its color on its side of the bin, the animals'
    mean with its 95 % interval in black farther out, a thin line through the
    group's means, the group's name beside its first bin, a dashed line at zero
    and which way means shared content."""
    bins = np.arange(len(SPREAD_BIN_LABELS))
    for side, (group, color) in zip((-1, 1), GROUP_COLORS.items(), strict=True):
        rows = [statistics.loc[(label, group)] for label in SPREAD_BIN_LABELS]
        for x, row in zip(bins + side * SHARED_OFFSET, rows, strict=True):
            animals = np.asarray(row["animals"], dtype=float)
            ax.scatter(
                x + spread_offsets(animals, 0.03), animals, s=5, color=color, lw=0,
                zorder=2,
            )  # fmt: skip
        means = bins + side * (SHARED_OFFSET + SHARED_MEAN_OFFSET)
        estimates = np.array([row["estimate"] for row in rows])
        ax.errorbar(
            means, estimates,
            yerr=[estimates - [r["ci_low"] for r in rows],
                  [r["ci_high"] for r in rows] - estimates],
            fmt="_", color="black", ms=4, lw=0.8, capsize=0, zorder=3,
        )  # fmt: skip
        ax.plot(means, estimates, color=color, lw=0.6, zorder=1)
    ax.axhline(0, color=REFERENCE_COLOR, lw=0.6, ls="--", zorder=1)
    ax.text(
        0.99, 0.0, "\u2191 shared content", ha="right", va="bottom",
        transform=blended_transform_factory(ax.transAxes, ax.transData),
        fontsize=SMALL_SIZE, color=REFERENCE_COLOR,
    )  # fmt: skip
    for k, (name, color) in enumerate(
        zip(("No SWR", "SWR"), GROUP_COLORS.values(), strict=True)
    ):
        ax.text(
            0.99, 0.97 - 0.1 * k, name, ha="right", va="top",
            transform=ax.transAxes, fontsize=SMALL_SIZE,
            color=readable_text_color(color),
        )  # fmt: skip
    ax.set_xlim(-0.6, bins[-1] + 0.6)
    ax.set_xticks(bins, SPREAD_BIN_LABELS)
    ax.set_xlabel(SHARED_XLABEL)
    ax.set_ylabel(SHARED_YLABEL)
    ax.set_title("Shared content", fontsize=SMALL_SIZE, pad=3)


def draw_outside_swr_series(
    ax: Axes,
    outside: pd.DataFrame,
    parts: pd.DataFrame,
    subset: str,
    thresholds: Sequence[float],
    names: Sequence[str],
    xlabel: str,
) -> None:
    """Panel d: the full decoder minus the SWR-only decoder per spike
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
    top = 1.05 * max(max(float(r["animals"].max()), r["ci_high"]) for r in rows)
    ax.set_ylim(-GAIN_FLOOR * top, top)
    ax.set_ylabel(DIFFERENCE_YLABEL)
    full_better_cue(ax)
    # the parts' label, centered under their tick labels
    ax.annotate(
        xlabel, (float(np.mean(xs[1:])), 0), xycoords=("data", "axes fraction"),
        xytext=(0, -PARTS_LABEL_DROP), textcoords="offset points", ha="center",
        va="top",
        fontsize=SMALL_SIZE, annotation_clip=False,
    )  # fmt: skip


def draw_no_spike_mua(ax: Axes, trace: dict) -> None:
    """Panel a: MUA around No-Spike starts (``trace``: :func:`read_no_spike_trace`),
    the median with its interquartile band as Figure 2g draws its rows, a dotted
    line at the start and a dashed line at the level outside the periods."""
    t_ms = 1000 * np.asarray(trace["time"])
    ax.fill_between(
        t_ms, trace["q25"], trace["q75"], color=NO_SPIKE_COLOR, alpha=0.4, lw=0
    )
    ax.plot(t_ms, trace["median"], color=readable_text_color(NO_SPIKE_COLOR), lw=1.0)
    ax.axhline(trace["level"], color=REFERENCE_COLOR, lw=0.6, ls="--", zorder=3)
    ax.text(
        t_ms[-1], trace["level"], "outside\nperiods", ha="right", va="bottom",
        fontsize=SMALL_SIZE, color=REFERENCE_COLOR,
    )  # fmt: skip
    ax.axvline(0, color="k", lw=0.5, ls=":")
    ax.set_xlim(t_ms[0], t_ms[-1])
    ax.set_xlabel("Time from No-Spike\nstart (ms)")
    ax.set_ylabel("MUA (z)")
    ax.set_title("No-Spike state", fontsize=SMALL_SIZE, pad=3)


def generate_figure(statistics: dict) -> Figure:
    """Build the figure from :func:`read_inputs`, the panels in :data:`PANELS`'
    order and laid out by :data:`ROWS` and :data:`FRAMES`."""
    assert sorted(p for _, row in ROWS for p in row) == sorted(PANELS)
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    axes = {
        panel: fig.add_axes((FRAMES[panel][0], bottom, FRAMES[panel][1], top - bottom))
        for (bottom, top), row in ROWS for panel in row
    }  # fmt: skip
    draw_no_spike_mua(axes["no_spike_mua"], statistics["no_spike"])
    draw_heldout_tetrode(axes["heldout_gain"], statistics["tetrode"])
    draw_shared_content(axes["shared_content"], statistics["shared"])
    draw_outside_swr_series(
        axes["outside_distance"], statistics["outside"], statistics["parts"],
        "beyond_swr_margin", OUTSIDE_SWR_MARGINS_S,
        [f"{t * 1000:g}" for t in OUTSIDE_SWR_MARGINS_S], DISTANCE_LABEL,
    )  # fmt: skip
    draw_outside_swr_shares(
        axes["outside_shares"], statistics["outside"], SHARES_YLABEL
    )
    fig.draw_without_rendering()
    # convention 10: a row's letters at one height, just above its highest
    # panel; the rows' first panels share the left column and its letter edge
    extent = {panel: drawn_extent(fig, [ax]) for panel, ax in axes.items()}
    left = {panel: max(e.x0 - 0.004, 0.0) for panel, e in extent.items()}
    column = min(left[row[0]] for _, row in ROWS)
    for _, row in ROWS:
        left[row[0]] = column
        top = max(extent[panel].y1 for panel in row) + LETTER_HEIGHT
        for panel in row:
            letter_panel(fig, LETTERS[panel], left[panel], top)
    check_placement(fig, {LETTERS[p]: [ax] for p, ax in axes.items()})
    return fig


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("paper/figures/supp_figure02")
    )
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig = generate_figure(read_inputs())
    check_journal_limits(fig)
    logging.info(f"Saving to {args.output_dir}...")
    save_figure(fig, "supp_figure02", base_dir=args.output_dir)
    plt.close("all")


if __name__ == "__main__":
    main()
