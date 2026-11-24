#!/usr/bin/env python
"""Generate Supplementary Figure S2, the supplement to Figure 2, its panels in
the order the text cites them, in five rows: the decoder on these recordings
(a to d), the count is robust (e, f), the held-out tests (g to i), the animals
agree and the posterior's spread (j, k), and a fourth decoder (l). Each panel's
question, and the analysis function it draws:

    a — do the decoder's states mean what they claim, by held-out spikes?
        (``state_evidence.calibration_statistics``)
    b — MUA around the start of No-Spike periods
        (``triggered_analysis.stored_no_spike_trace``).
    c — how the immobile time is spread over non-local probability
        (``nonlocal_probability.animal_probability_distribution``).
    d — how far the decoded place is from the animal by Local probability
        (``quality_metrics.animal_distance_by_local_probability``).
    e — the share of events with no SWR at other settings of the event rule and
        the SWR detector (``rule_sweep.sweep_statistics``).
    f — each event's start and end against its SWR's
        (``event_statistics.boundary_timing_statistics``).
    g — where non-local time sits in MUA (``nonlocal_by_signal.animal_nonlocal_counts``).
    h — Figure 2j's first comparison by bins with and without a spike
        (``tetrode_holdout.holdout_statistics``).
    i — Figure 2k's gain outside SWRs, away from the SWRs
        (``restriction_totals.outside_swr_subset_statistics``).
    j — Figure 2g's traces, animal by animal
        (``triggered_analysis.stored_animal_window_medians``).
    k — how concentrated the decoded position is, no-SWR against SWR events
        (``event_statistics.hpd_distribution``).
    l — Figure 2k against a fourth decoder, SWR-allowed
        (``restriction_totals.swr_allowed_statistics``).

Every event panel is over the events that start inside a well visit. What the
figure reads, and its tests, are in ``docs/figure-pipeline.md``; each panel's
values and questions in ``reported/questions.py``.

Usage
-----
    python scripts/figures/generate_supp_figure02.py
"""

import argparse
import logging
import string
import warnings
from collections.abc import Iterable, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, NullLocator

import continuum_swr_replay.visualization.figure_utilities as figure_utilities
from continuum_swr_replay.analysis.decode_quality.nonlocal_probability import (
    HIGH_CUT,
    animal_probability_distribution,
)
from continuum_swr_replay.analysis.decode_quality.quality_metrics import (
    animal_distance_by_local_probability,
)
from continuum_swr_replay.analysis.event_definition.analyzed_time import (
    read_swr_table,
)
from continuum_swr_replay.analysis.event_definition.cohort_selection import (
    select_visit_context,
)
from continuum_swr_replay.analysis.event_definition.swr_source import (
    event_table_epochs,
)
from continuum_swr_replay.analysis.event_evidence.heldout_runs import (
    read_tetrode_holdout,
)
from continuum_swr_replay.analysis.event_evidence.restriction_totals import (
    OUTSIDE_SWR_MARGINS_S,
    RESTRICTION_FIGURE_EVENT_GROUP,
    RESTRICTION_FIGURE_FAMILY,
    RESTRICTION_FIGURE_OUTSIDE_SWR,
    SWR_ALLOWED_FIGURE_DECODERS,
    SWR_ALLOWED_FIGURE_DIFFERENCES,
    SwrAllowedStatistics,
    outside_swr_statistics,
    outside_swr_subset_statistics,
    swr_allowed_statistics,
)
from continuum_swr_replay.analysis.event_evidence.state_evidence.summary import (
    probability_bin_center,
)
from continuum_swr_replay.analysis.event_evidence.tetrode_holdout import (
    holdout_statistics,
    window_scores,
)
from continuum_swr_replay.analysis.event_properties.event_statistics import (
    HPD_BIN_EDGES_CM,
    HPD_TIGHT_CM,
    OFFSET_TAILS_S,
    TIMING_TOLERANCE,
    boundary_offsets,
    boundary_timing_statistics,
    hpd_distribution,
    nonlocal_below_threshold_statistics,
)
from continuum_swr_replay.analysis.event_properties.nonlocal_by_signal import (
    DETECTION_LEVELS,
    MUA_LEVEL_Z,
    animal_nonlocal_counts,
)
from continuum_swr_replay.analysis.event_properties.rule_sweep import (
    RULE,
    sweep_statistics,
)
from continuum_swr_replay.analysis.event_properties.state_by_signal import (
    state_distribution,
)
from continuum_swr_replay.analysis.event_properties.triggered_analysis import (
    TRIGGERED_SIGNALS,
    WINDOW_KINDS,
    stored_animal_window_medians,
    stored_no_spike_trace,
    stored_windows_of,
)
from continuum_swr_replay.analysis.shared_statistics import animal_mean
from continuum_swr_replay.data_loaders.epoch_register import (
    analyzed_epoch_ids,
    require_analyzed_epochs,
)
from continuum_swr_replay.parameters import DECODE_SAMPLE_S
from continuum_swr_replay.reported.formatting import format_number
from continuum_swr_replay.reported.inputs import (
    default_inputs,
    output_staleness,
    panel_refusal,
    read_checked,
    read_figure_restriction,
    read_trigger_windows,
)
from continuum_swr_replay.reported.values import rel
from continuum_swr_replay.runs.staleness import read_output_manifest
from continuum_swr_replay.visualization.evidence_drawings import draw_boundary_cases
from continuum_swr_replay.visualization.figure_utilities import save_figure
from continuum_swr_replay.visualization.heldout_gain_panels import (
    DIFFERENCE_LABELS,
    EQUAL_FIT,
    EVENTS_LABELS,
    EVENTS_TITLE,
    GROUP_COLORS,
    MEAN_OFFSET,
    NEUTRAL_DOTS,
    REFERENCE_COLOR,
    SPLIT_LABELS,
    animal_values,
    draw_decoder_gain_axis,
    draw_full_minus_swr_axis,
    draw_split_gain,
    gain_range,
    label_gain_axis,
    name_groups,
    name_zero_line,
)
from continuum_swr_replay.visualization.palettes import (
    ANIMAL_LINE_COLOR,
    COLORS,
    EVENT_KINDS,
    SIGNAL_LABELS,
    readable_text_color,
)
from continuum_swr_replay.visualization.panel_layout import (
    fit_panel,
    letter_layout,
    panel_spans,
    place_letters,
)
from continuum_swr_replay.visualization.publication_style import (
    SMALL_SIZE,
    check_journal_limits,
    check_placement,
    set_publication_style,
)
from continuum_swr_replay.visualization.signal_share_panels import draw_share_by_signal
from continuum_swr_replay.visualization.simulated_run_panels import STATE_COLORS

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%d-%b-%y %H:%M:%S"
)
figure_utilities.set_figure_defaults(context="paper")
set_publication_style()
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Data choices
# ---------------------------------------------------------------------------

FIGURE = "Figure 2's supplement"
"""The figure's name in its refusals (``reported.inputs.panel_refusal``)."""
TIMING_CASES = (
    ("Same start and end", "together", "together"),
    ("Starts later", "later", "together"),
    ("Starts earlier", "earlier", "together"),
    ("Ends earlier", "together", "earlier"),
    ("Ends later", "together", "later"),
    ("Shorter at both ends", "later", "earlier"),
    ("Longer at both ends", "earlier", "later"),
    ("Starts and ends later", "later", "later"),
    ("Starts and ends earlier", "earlier", "earlier"),
)
"""Panel f's rows, top to bottom: each case's name and the start and end class
its drawing shows (``event_statistics.TIMING_CLASSES``): the same start and
end, the four cases with one bound apart, then the four with both apart; each
case beside its opposite (a fixed order, not sorted by share)."""
SWEEP_SCALES = {
    "probability_threshold": 1.0,
    "minimum_samples": 1000 * DECODE_SAMPLE_S,
    "swr_threshold_z": 1.0,
}
"""What each setting's value is multiplied by on the sweep panel's x axis: the
minimum run length in samples is drawn as the shortest event it keeps, in ms
(n samples of 2 ms; the rule's 7 samples, its 15 ms, is a shortest event of
14 ms)."""

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

# Inches: the full page width, five rows (the module's docstring).
WIDTH = 6.5
HEIGHT = 6.8
COLUMNS = {"a": -0.0036, "b": 0.322, "c": 0.570, "d": 0.774}
"""Where the drawings of the panels of each letter column start (figure width;
convention 10), named for the first row's panel that starts there: the rows'
first panels (at the figure's edge: it is saved cropped to its ink), then
three columns set by the first row's panels b, c and d."""
LAYOUT = (
    (
        (0.861, 0.950),
        (
            ("state_calibration", "a", 0.2746),
            ("no_spike_mua", "b", 0.545),
            ("nonlocal_probability", "c", 0.7585),
            ("decoded_distance", "d", 0.9935),
        ),
    ),
    (
        (0.634, 0.759),
        (
            ("rule_sweep", "a", 0.53),
            ("boundary_timing", "c", 0.9871),
        ),
    ),
    (
        (0.399, 0.509),
        (
            ("nonlocal_mua", "a", 0.2971),
            ("left_out_split", "b", 0.545),
            ("outside_distance", "c", 0.98),
        ),
    ),
    (
        (0.230, 0.297),
        (
            ("event_traces", "a", 0.7226),
            ("hpd", "d", 0.9878),
        ),
    ),
    (
        (0.054, 0.127),
        (("four_decoders", "a", 0.98),),
    ),
)
"""The figure's rows, top to bottom: each row's frames' bottom and top (figure
height), then its panels left to right, each with the column where its drawing
starts (:data:`COLUMNS`) and where it ends (figure width), as
``panel_layout.panel_spans`` reads them. Read in this order the panels are in
the order the text cites them, so their letters follow from it."""
PANELS = tuple(panel for _, row in LAYOUT for panel, _, _ in row)
"""The panels in the order the text cites them."""
LETTERS = dict(zip(PANELS, string.ascii_lowercase, strict=False))
LETTER_HEIGHT = 0.0268
"""A panel letter's height (figure height): the letters' tops sit this far
above the highest drawn top of their row."""
PANEL_GRIDS = {
    "boundary_timing": (1, 2),
    "nonlocal_mua": (2, 1),
    "rule_sweep": (2, 3),
    "event_traces": (1, len(TRIGGERED_SIGNALS)),
    "four_decoders": (1, 3),
}
"""The panels drawn on several axes in their frame: rows and columns."""
PANEL_ROW_SHARES = {"rule_sweep": (2.2, 1), "nonlocal_mua": (1.2, 1)}
"""The heights of a stacked panel's rows, as shares (equal when not named)."""
ROW_GAP = 0.1
"""The gap between a stacked panel's rows, as a share of its frame's height."""
PANEL_COLUMN_SHARES = {
    "boundary_timing": (1.15, 1.0),
    "four_decoders": (0.7, 1.6, 0.35),
}
"""The widths of a panel's columns, as shares (equal when not named): the
timing panel's drawing and bars, and the fourth-decoder panel's three axes."""
PANEL_COLUMN_GAPS = {
    "rule_sweep": 0.375,
    "boundary_timing": 0.0619,
    "event_traces": 0.75,
    "four_decoders": 0.7059,
}
"""The gap between a panel's columns of axes, as a share of their mean width
(``panel_layout.fit_panel``)."""
CALIBRATION_DODGE = 0.025
"""How far apart the states' points sit at each probability bin (x units)."""
SWEEP_TICKS = {
    "probability_threshold": (0.5, 0.8, 0.95),
    "minimum_samples": (14, 50, 100),
    "swr_threshold_z": (0, 2, 4, 6),
}
"""The sweep panel's x ticks, in its drawn units: the value used and the ends."""
SERIES_START = 1.3
"""The x position of a series' first part, after the "Outside SWRs" column at 0;
the parts are 1 apart."""
PARTS_LABEL_DROP = 13
"""How far below the axis a series' parts label starts (points): just under
the parts' one-line tick labels, beside the two lines of "Outside SWRs" (it
is centered under the parts, clear of that column), so that it sits well above
the next row's letters."""
HPD_TICKS_CM = (10, 25, 50, 100, 400)
FOUR_DECODER_STEP = 0.75
"""Panel l, second axis: the x distance between the two differences at one time."""
FOUR_DECODER_GROUP = 2.0
"""Panel l, second axis: the x distance between one time and the next."""

# ---------------------------------------------------------------------------
# Labels and colors
# ---------------------------------------------------------------------------

CALIBRATION_COLORS = {
    "Local": STATE_COLORS["Local"],
    "Non-Local Continuous": STATE_COLORS["Continuous"],
    "Non-Local Fragmented": STATE_COLORS["Fragmented"],
}
"""The calibration panel's states (``state_evidence.IN_VISIT_CALIBRATION_METRICS``),
in the decoder figures' state colors."""
CALIBRATION_YLABEL = "State's place over a\nmismatched place\n(nats/spike)"
"""The calibration panel's y label."""
NO_SPIKE_COLOR = COLORS["No-Spike"]
SWEEP_XLABELS = {
    "probability_threshold": "Non-local\nprobability",
    "minimum_samples": "Minimum\nduration (ms)",
    "swr_threshold_z": "SWR\nthreshold (z)",
}
"""The sweep panel's x label of each swept setting."""
TIMING_NOTE = f"same: within {1000 * OFFSET_TAILS_S[TIMING_TOLERANCE]:g} ms"
"""Panel f's note of what "same" means (the tolerance of
``event_statistics.TIMING_TOLERANCE``)."""
TIMING_BAR_COLOR = "#bbbbbb"
"""Panel f's bars of the mean over animals: a neutral gray, under the animals'
dots."""
LEFT_OUT_YLABEL = "Left-out tetrode:\nbetter fit (nats/spike)"
"""The y label of the left-out tetrode's split."""
SPLIT_TITLE = "Decoded place over\nanimal's place"
DISTANCE_LABEL = "at least this far from\nany SWR (ms)"
HPD_XLABEL = "95 % HPD length (cm)"
"""The posterior-spread panel's x label: each event's mean over its bins of the
length of track that holds 95 % of the decoded probability (on one line, clear
of the next row's titles; the mean over the bins is in the caption)."""
HPD_YLABEL = "% of events"
FOUR_DECODER_TIMES = {
    "immobile": "all",
    "inside_swr": "inside_swr",
    "outside_swr": "outside_swr",
}
"""Panel l, second axis: each time of ``restriction_totals.CONTRAST_TIMES``, in
drawing order, by its label's key in ``DIFFERENCE_LABELS``."""
FOUR_DECODER_DIFFERENCES = {
    "full_minus_swr_allowed_per_spike": ("Full \u2212 SWR-allowed", False),
    "swr_allowed_minus_swr_per_spike": ("SWR-allowed \u2212 SWR-only", True),
}
"""Panel l, second axis: each difference of
``restriction_totals.SWR_ALLOWED_FIGURE_DIFFERENCES``, its name in the legend
and whether its dots are open (the one against the SWR-only decoder, of which
the SWR-allowed decoder is a variant)."""
FOUR_DECODER_YLABEL = "Held-out log score\ndifference\n(nats/spike)"
FOUR_DECODER_EVENTS_YLABEL = (
    "Held-out log score,\nfull \u2212 SWR-allowed\n(nats/spike)"
)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def read_inputs() -> dict:
    """The figure's statistics, as the events summary reads them.

    Returns
    -------
    dict
        By panel, in :data:`PANELS`' order: ``state_calibration``,
        :func:`read_state_calibration`; ``no_spike_mua`` and ``event_traces``,
        :func:`read_traces`; ``nonlocal_probability``,
        ``nonlocal_probability.animal_probability_distribution`` of the stored
        counts; ``decoded_distance``,
        ``quality_metrics.animal_distance_by_local_probability`` of the
        decoder-quality run's table (each animal's values and the statistics);
        ``rule_sweep``, :func:`read_rule_sweep`; ``boundary_timing``, the
        selected events' ``boundary_timing_statistics``; ``nonlocal_mua``,
        :func:`read_mua_counts` and :func:`read_mua_states`; ``left_out_split``,
        by column of ``heldout_gain_panels.SPLIT_LABELS``, ``holdout_statistics``
        of the events of the held-out tetrode run that start inside a well
        visit, indexed by group (a run not held out in folds of whole trials is
        refused, ``heldout_runs.read_tetrode_holdout``); ``outside_distance``,
        ``outside_swr_statistics`` over the held-out event run, indexed by
        family and measure, and ``outside_swr_subset_statistics`` of the
        margins over its time sums, indexed by subset, threshold, family and
        measure; ``hpd``, the selected events' ``hpd_distribution``;
        ``four_decoders``, ``swr_allowed_statistics`` over the same held-out
        event run.

    Raises
    ------
    ValueError
        If a held-out event run, or what it was made from, is not current
        (``reported.inputs.HELDOUT_RESTRICTION_RERUN`` names what to make again).
    """
    inputs = default_inputs()
    table = read_swr_table(inputs.event_table)
    epochs = event_table_epochs(table, analyzed_epoch_ids())
    events = select_visit_context(table)
    # the left-out split: the held-out tetrode run over the selected events, as
    # the events summary reads it (refused when not held out in folds of whole trials)
    holdout = window_scores(
        read_tetrode_holdout(
            inputs.tetrode_holdout,
            epochs,
            events["event_id"],
            rel(inputs.tetrode_holdout),
        )
    )
    heldout, refusal = read_figure_restriction(inputs, epochs)
    if refusal is not None:
        raise ValueError(panel_refusal(FIGURE, LETTERS["outside_distance"], *refusal))
    read = {
        "state_calibration": read_state_calibration(inputs.state_calibration_in_visits),
        "nonlocal_probability": read_nonlocal_probability(inputs.nonlocal_probability),
        "decoded_distance": read_decoded_distance(inputs.decoder_quality),
        "rule_sweep": read_rule_sweep(inputs.event_rule_sweep, inputs.event_table),
        "boundary_timing": boundary_timing_statistics(boundary_offsets(events), events),
        "nonlocal_mua": (
            read_mua_counts(inputs.nonlocal_by_signal),
            read_mua_states(inputs.state_by_signal),
        ),
        "left_out_split": {
            column: holdout_statistics(holdout, "event", column)
            for column in SPLIT_LABELS
        },
        "outside_distance": (
            outside_swr_statistics(heldout.time_sums, heldout.bin_s, heldout.events),
            outside_swr_subset_statistics(
                heldout.time_sums, heldout.bin_s, subsets=("beyond_swr_margin",)
            ).set_index(["subset", "threshold_s", "family", "measure"]),
        ),
        "hpd": hpd_distribution(events),
        "four_decoders": swr_allowed_statistics(
            heldout.events, heldout.time_sums, heldout.bin_s
        ),
    }
    read.update(read_traces(inputs.trigger_windows, events))
    return {panel: read[panel] for panel in PANELS}


def _refuse_stale(
    folder: Path,
    panel: str,
    rerun: str,
    *,
    event_table: Path | None = None,
    epoch_ids: Iterable[str] | None = None,
) -> None:
    """Refuse an output the events summary marks pending
    (``reported.inputs.output_staleness``): not made from the current event
    table (when ``event_table`` is given), not over the analyses' subset, or,
    for a time-masked output (``epoch_ids`` given), not from the current well
    visits."""
    reason = output_staleness(folder, event_table=event_table, epoch_ids=epoch_ids)
    if reason is not None:
        raise ValueError(panel_refusal(FIGURE, LETTERS[panel], reason, rerun))


def read_state_calibration(folder: Path) -> pd.DataFrame:
    """The calibration panel's bins: the held-out state calibration over the
    immobile windows inside the well visits, as the events summary reads it
    (``state_evidence.calibration_statistics``' bins, written by
    ``scripts/event_evidence/summarize_state_calibration_in_visits.py``).

    Raises
    ------
    ValueError
        If the summary was not made over the analyses' subset from the current
        well visits, or not over the analyzed epochs.
    """
    epochs = list(read_output_manifest(folder)["epochs"])
    _refuse_stale(
        folder, "state_calibration",
        "scripts/event_evidence/summarize_state_calibration_in_visits.py",
        epoch_ids=epochs,
    )  # fmt: skip
    require_analyzed_epochs(epochs, f"Figure 2's supplement: {rel(folder)}")
    return pd.read_csv(folder / "calibration_bins.csv")


def read_traces(folder: Path, events: pd.DataFrame) -> dict:
    """The No-Spike panel's trace and the traces panel's, from the stored
    windows around event starts and No-Spike starts
    (``reported.inputs.read_trigger_windows``).

    Returns
    -------
    dict
        ``no_spike_mua``: ``triggered_analysis.stored_no_spike_trace``,
        ``time`` (s from the start), ``q25``, ``median`` and ``q75``
        animal-balanced over the periods, and ``level``, the mean over animals
        of each animal's median over its epochs of each epoch's MUA level
        outside its periods; ``event_traces``:
        ``triggered_analysis.stored_animal_window_medians`` of the selected
        events' windows, by SWR class, each signal's per-animal median windows,
        with ``time``.

    Raises
    ------
    ValueError
        If the stored windows are not current, as the events summary refuses a
        per-epoch measurement.
    """
    try:
        windows, epochs = read_trigger_windows(folder)
    except ValueError as error:
        raise ValueError(
            panel_refusal(
                FIGURE,
                LETTERS["no_spike_mua"],
                str(error),
                "scripts/event_properties/measure_trigger_windows.py",
            )
        ) from error
    return {
        "no_spike_mua": stored_no_spike_trace(windows, epochs),
        "event_traces": stored_animal_window_medians(
            stored_windows_of(windows, events)
        ),
    }


def read_nonlocal_probability(folder: Path) -> pd.DataFrame:
    """The non-local probability panel's distribution: ``nonlocal_probability.animal_probability_distribution``
    of the stored counts (``scripts/decoding/measure_nonlocal_probability.py``),
    refused when they were not made over the analyses' subset from the current
    well visits, as the events summary marks their values pending."""
    counts = pd.read_parquet(folder / "counts.parquet")
    _refuse_stale(
        folder,
        "nonlocal_probability",
        "scripts/decoding/measure_nonlocal_probability.py",
        epoch_ids=counts["epoch_id"].unique(),
    )
    return animal_probability_distribution(counts)


def read_decoded_distance(folder: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The decoded-place panel's lines: ``quality_metrics.animal_distance_by_local_probability``
    of the decoder-quality run's table (``calibration.parquet``), as the events
    summary reads it."""
    return animal_distance_by_local_probability(
        pd.read_parquet(folder / "calibration.parquet")
    )


def read_rule_sweep(folder: Path, event_table: Path) -> pd.DataFrame:
    """The sweep panel's statistics (``rule_sweep.sweep_statistics`` of the
    stored per-epoch counts), as the events summary reads them.

    Raises
    ------
    ValueError
        If the counts were not made from the current event table over the
        analyses' subset and the current well visits, or do not hold the
        analyzed epochs.
    """
    counts = pd.read_parquet(folder / "counts.parquet")
    _refuse_stale(
        folder, "rule_sweep", "scripts/event_properties/sweep_event_rule.py",
        event_table=event_table, epoch_ids=counts["epoch_id"].unique(),
    )  # fmt: skip
    require_analyzed_epochs(
        counts["epoch_id"], f"Figure 2's supplement: {rel(folder)}/counts.parquet"
    )
    return sweep_statistics(counts)


def read_mua_counts(folder: Path) -> pd.DataFrame:
    """The MUA panel's counts: each animal's immobile time bins inside the well
    visits, and its non-local ones, by MUA (``nonlocal_by_signal.animal_nonlocal_counts``
    of the stored counts), as the events summary reads them.

    Raises
    ------
    ValueError
        If the counts were not made over the analyses' subset from the current
        well visits and the decodes the event features run records, for the
        analyzed epochs (``reported.inputs.checked_reference``, as Figure 2i).
    """
    try:
        counts = read_checked(folder, "counts by signal")
    except ValueError as error:
        raise ValueError(
            panel_refusal(
                FIGURE,
                LETTERS["nonlocal_mua"],
                str(error),
                "scripts/event_properties/measure_nonlocal_by_signal.py",
            )
        ) from error
    return animal_nonlocal_counts(counts, "mua")


def read_mua_states(folder: Path) -> pd.DataFrame:
    """The MUA panel's top strip: ``state_by_signal.state_distribution`` of the
    stored counts by MUA and decoder state, refused as :func:`read_mua_counts`
    refuses its counts."""
    try:
        counts = read_checked(folder, "counts by signal and state")
    except ValueError as error:
        raise ValueError(
            panel_refusal(
                FIGURE,
                LETTERS["nonlocal_mua"],
                str(error),
                "scripts/event_properties/measure_state_by_signal.py",
            )
        ) from error
    return state_distribution(counts, "mua")


# ---------------------------------------------------------------------------
# Drawing, in panel order
# ---------------------------------------------------------------------------


def draw_state_calibration(ax: Axes, bins: pd.DataFrame) -> None:
    """The calibration panel: for Local, Continuous and Fragmented, on one
    axis, the held-out evidence per spike for the state's place over a
    mismatched place against the state's probability, the animals' mean with
    its 95 % interval at each probability bin's center (the states a little
    apart so that the intervals show), each state named at its line's end in
    its color, and a dashed line at zero (``bins``: :func:`read_inputs`'
    ``state_calibration``). Random 10-bin immobile windows that start inside a
    well visit, grouped by the state's probability: for Local, the held-out
    tetrodes' spikes at the animal's place over mismatched places; for
    Continuous and Fragmented, the place decoded from one half of the tetrodes
    scored on the other half over mismatched places (a window in which the
    scored half never spiked has no value). No-Spike is not drawn: a confident
    No-Spike label forces both halves silent, so it is no test. The probability
    bins come from the decode with every tetrode, only the remote place is held
    out by the tetrode split, and the encodings are fit on running time in other
    time blocks."""
    ends = []
    for k, (state, color) in enumerate(CALIBRATION_COLORS.items()):
        mine = bins[bins["state"] == state]
        estimate = mine["animal_estimate"].to_numpy()
        centers = mine["pbin"].map(probability_bin_center).to_numpy()
        x = centers + (k - 1) * CALIBRATION_DODGE
        ax.errorbar(
            x, estimate,
            yerr=[estimate - mine["animal_ci_lo"], mine["animal_ci_hi"] - estimate],
            color=color, marker="o", ms=2.5, lw=1.0, capsize=0,
        )  # fmt: skip
        ends.append((estimate[-1], state, color))
    # each state named at its line's end, the names at least a line apart
    # (points), from the highest down
    ax.autoscale_view()
    to_points = 72 / ax.figure.dpi
    offsets, previous = {}, None
    for y, state, _ in sorted(ends, reverse=True):
        y_pt = ax.transData.transform((0, y))[1] * to_points
        placed = y_pt if previous is None else min(y_pt, previous - SMALL_SIZE * 1.15)
        offsets[state], previous = placed - y_pt, placed
    for y, state, color in ends:
        ax.annotate(
            state.removeprefix("Non-Local "), (1.0, y),
            xytext=(2, offsets[state]), textcoords="offset points", ha="left",
            va="center", fontsize=SMALL_SIZE, color=readable_text_color(color),
            annotation_clip=False,
        )  # fmt: skip
    ax.plot([0, 1], [0, 0], color=REFERENCE_COLOR, lw=0.6, ls="--", zorder=1)
    ax.set_xlim(0, 1.3)  # room for the states' names inside the axis
    ax.spines["bottom"].set_bounds(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xlabel("State probability")
    ax.set_ylabel(CALIBRATION_YLABEL)
    ax.set_title("Decoder states, held out", fontsize=SMALL_SIZE, pad=3)


def draw_no_spike_mua(ax: Axes, trace: dict) -> None:
    """The No-Spike panel: MUA around No-Spike starts (``trace``:
    :func:`read_traces`' ``no_spike_mua``), the animal-balanced median
    with its band as Figure 2g draws its rows, a dotted line at the start and a
    dashed line at the MUA level of immobile time outside the periods. A period
    is the non-local events' rule applied to the No-Spike state
    (``no_spike_periods.no_spike_periods``). The decoder enters the No-Spike
    state when the decoded tetrodes fall silent, so the fall in their MUA
    describes the state and is not a test of it."""
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


def draw_nonlocal_probability(ax: Axes, distribution: pd.DataFrame) -> None:
    """The non-local probability panel: how the immobile time inside the well
    visits is spread over the decoder's non-local probability, the percent of
    the time per bin of 0.02 on a log axis, a thin line per animal and their
    mean in bold (``shared_statistics.animal_mean``), with the event rule's cut (0.8) dotted and named "used"
    (``distribution``: :func:`read_inputs`' ``nonlocal_probability``,
    ``nonlocal_probability.animal_probability_distribution``)."""
    by_animal = distribution.pivot(index="bin_low", columns="animal", values="percent")
    high = distribution.groupby("bin_low")["bin_high"].first()
    center = (by_animal.index.to_numpy() + high.to_numpy()) / 2
    ax.plot(center, by_animal.to_numpy(), color=ANIMAL_LINE_COLOR, lw=0.5)
    ax.plot(
        center, animal_mean(distribution, "bin_low", "percent").to_numpy(),
        color="black", lw=1.2,
    )  # fmt: skip
    ax.set_yscale("log")
    ax.set_yticks([0.1, 1, 10, 100], ["0.1", "1", "10", "100"])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.axvline(HIGH_CUT, color=REFERENCE_COLOR, lw=0.6, ls=":", zorder=1)
    ax.text(
        HIGH_CUT, 0.98, "used ", transform=ax.get_xaxis_transform(), ha="right",
        va="top", fontsize=SMALL_SIZE, color=REFERENCE_COLOR,
    )  # fmt: skip
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
    ax.set_xlabel("Non-local probability")
    ax.set_ylabel("% of time")
    ax.set_title("Non-local probability,\nimmobile time", fontsize=SMALL_SIZE, pad=3)


def draw_decoded_distance(
    ax: Axes, by_animal: pd.DataFrame, statistics: pd.DataFrame
) -> None:
    """The decoded-place panel: how far the decoded place (the maximum of the
    position estimate conditional on the non-local states) is from the animal
    against the Local probability, over all analyzed time, a thin line per
    animal (the median over its sessions of each session's median) and the mean
    over animals in bold (``by_animal``, ``statistics``: :func:`read_inputs`'
    ``decoded_distance``, ``quality_metrics.animal_distance_by_local_probability``)."""
    centers = by_animal.index.to_numpy()
    ax.plot(centers, by_animal.to_numpy(), color=ANIMAL_LINE_COLOR, lw=0.5)
    ax.plot(centers, statistics["animal_mean"].to_numpy(), color="black", lw=1.2)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
    # a log axis: the lowest bin of one animal is far above every other value
    ax.set_yscale("log")
    ax.set_yticks([3, 10, 30, 100], ["3", "10", "30", "100"])
    ax.minorticks_off()
    ax.set_xlabel("Local probability")
    ax.set_ylabel("Distance from\nthe animal (cm)")
    ax.set_title(
        "Decoded place against\nthe animal's place", fontsize=SMALL_SIZE, pad=3
    )


def draw_rule_sweep(axes: Sequence[Axes], sweep: pd.DataFrame) -> None:
    """The sweep panel: on the top row, the share of events with no SWR at each
    value of each swept setting (one axis each, sharing the y axis), a thin
    line per animal and their mean over animals in bold with its dots, and on every
    axis the value the paper uses dotted and marked "used"; on the bottom row,
    under each, the events detected at each value (thousands, summed over the
    epochs; flat on the SWR threshold's axis, which relabels the rule's
    events). The first two columns are settings of the event rule, under one
    heading; the third is the SWR detector's threshold (``sweep``:
    :func:`read_inputs`' ``rule_sweep``)."""
    n = len(SWEEP_XLABELS)
    shares, counts = axes[:n], axes[n:]
    for ax, ax_count, (setting, xlabel) in zip(
        shares, counts, SWEEP_XLABELS.items(), strict=True
    ):
        mine = sweep[sweep["setting"] == setting]
        x = SWEEP_SCALES[setting] * mine["value"].to_numpy()
        used = SWEEP_SCALES[setting] * RULE[setting]
        animals = pd.DataFrame(
            [animal_values(row, "animals") for _, row in mine.iterrows()]
        )
        ax.plot(x, 100 * animals.to_numpy(), color="k", lw=0.5, alpha=0.5)
        ax.plot(x, 100 * mine["no_swr_fraction_animal_mean"].to_numpy(), color="k", lw=1.0,
                marker="o", ms=2)  # fmt: skip
        ax_count.plot(x, mine["n_events"].to_numpy() / 1000, color="k", lw=1.0,
                      marker="o", ms=2)  # fmt: skip
        for a in (ax, ax_count):
            a.axvline(used, color=REFERENCE_COLOR, lw=0.6, ls=":", zorder=1)
        ax_count.sharex(ax)
        ax.tick_params(labelbottom=False)
        ax_count.set_xticks(SWEEP_TICKS[setting])
        ax_count.set_xlabel(xlabel)
        if ax is not shares[0]:
            ax.sharey(shares[0])
            ax.tick_params(labelleft=False)
            ax_count.sharey(counts[0])
            ax_count.tick_params(labelleft=False)
    shares[0].set_ylim(0, 100)
    shares[0].set_yticks([0, 50, 100])
    shares[0].set_ylabel("No SWR (%)")
    counts[0].set_ylim(0, 1.15 * sweep["n_events"].max() / 1000)
    counts[0].yaxis.set_major_locator(MaxNLocator(2))
    counts[0].set_ylabel("Events\n(1000s)")
    # one left edge for the two rows' labels
    for ax in (shares[0], counts[0]):
        ax.yaxis.set_label_coords(-0.32, 0.5)
    for ax, setting in zip(shares, SWEEP_XLABELS, strict=True):
        ax.text(
            SWEEP_SCALES[setting] * RULE[setting], 3, " used", fontsize=SMALL_SIZE,
            color=REFERENCE_COLOR, ha="left", va="bottom",
        )  # fmt: skip
    # one heading over the event rule's two axes, another over the detector's
    first, second = shares[0].get_position(), shares[1].get_position()
    for ax, heading, x in (
        (
            shares[0],
            "Event rule",
            ((first.x0 + second.x1) / 2 - first.x0) / first.width,
        ),
        (shares[2], "SWR detector", 0.5),
    ):
        ax.annotate(
            heading, (x, 1.0), xycoords="axes fraction", xytext=(0, 3),
            textcoords="offset points", ha="center", va="bottom", fontsize=SMALL_SIZE,
        )  # fmt: skip


def draw_boundary_timing(ax_cases: Axes, ax: Axes, timing: pd.DataFrame) -> None:
    """The timing panel: each SWR-overlapping event against its matched SWR, one row per
    case of :data:`TIMING_CASES`, in its fixed order. At the left the
    case's name and a small drawing of it
    (``evidence_drawings.draw_boundary_cases``); at the right a bar of the mean
    over animals of each animal's share with a dot per animal, and the percent
    at the bar's end (``timing``: ``boundary_timing_statistics``, a row per
    start and end class). A note in gray says what "same" means."""
    draw_boundary_cases(ax_cases, list(TIMING_CASES))
    rows = [timing.loc[(start, end)] for _, start, end in TIMING_CASES]
    for k, row in enumerate(rows):
        mean = 100 * row["share_animal_mean"]
        ax.barh(k, mean, height=0.55, color=TIMING_BAR_COLOR, lw=0, zorder=1)
        animals = 100 * np.asarray(row["animals"], dtype=float)
        ax.scatter(
            animals, np.full(animals.size, k), s=5, color="black", lw=0, zorder=2
        )
        ax.text(
            max(mean, animals.max()) + 1.5, k,
            f"{format_number(mean / 100, 'pctnum')}%", ha="left",
            va="center", fontsize=SMALL_SIZE,
        )  # fmt: skip
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    # room for the largest animal's dot and the percent written after it
    largest = max(100 * float(np.max(row["animals"])) for row in rows)
    upper = 10 * np.ceil((largest + 8) / 10)
    ax.set_xlim(0, upper)
    ax.set_xticks(np.arange(0, upper + 1, 10 if upper <= 50 else 25))
    ax.set_xlabel("% of overlapping pairs")
    ax.text(
        1.0, 1.0, TIMING_NOTE, transform=ax.transAxes, ha="right", va="bottom",
        fontsize=SMALL_SIZE, color=REFERENCE_COLOR,
    )  # fmt: skip
    ax_cases.set_title("Event against its SWR", fontsize=SMALL_SIZE, pad=3, loc="left")


def draw_nonlocal_mua(
    ax_spread: Axes, ax_share: Axes, counts: pd.DataFrame, states: pd.DataFrame
) -> None:
    """The MUA panel, drawn as Figure 2i draws ripple power
    (``signal_share_panels.draw_share_by_signal``): where confident Local and
    confident non-local time sit in MUA (top; ``states``: :func:`read_inputs`'
    ``nonlocal_mua``) and the share of immobile bins decoded non-local at each
    MUA level (bottom), with the MUA level of the reported values and the
    share of non-local time below it (``nonlocal_below_threshold_statistics``,
    the events summary's value; ``counts``: :func:`read_inputs`' ``nonlocal_mua``)."""
    level = DETECTION_LEVELS["mua"]
    below = nonlocal_below_threshold_statistics(counts, level)
    draw_share_by_signal(
        ax_spread, ax_share, counts, threshold=level,
        threshold_label=f"{MUA_LEVEL_Z:g} z", threshold_color=REFERENCE_COLOR,
        xlabel="MUA (z)",
        below_note=f"{format_number(below['fraction_animal_mean'], 'pctnum')}% "
        f"below\n{MUA_LEVEL_Z:g} z", states=states,
    )  # fmt: skip


def draw_left_out_split(ax: Axes, statistics: dict) -> None:
    """The left-out split: Figure 2j's first comparison, the left-out tetrode's better
    fit to the decoded place than to the animal's place (the gain over local
    coding per spike), for no-SWR and SWR events, over all bins, the bins with
    a spike and the silent bins (``statistics``: :func:`read_inputs`'
    ``left_out_split``), drawn by ``heldout_gain_panels.draw_split_gain``; the dashed
    zero line named :data:`EQUAL_FIT`."""
    draw_split_gain(ax, statistics, SPLIT_TITLE)
    low, high = gain_range(
        table.loc[group] for table in statistics.values() for group in table.index
    )
    label_gain_axis(ax, LEFT_OUT_YLABEL, min(-0.12, low - 0.08 * (high - low)))
    name_zero_line(ax, EQUAL_FIT, 0.02, offset_pt=-2, va="top")
    ax.set_ylim(None, high + 0.15 * (high - low))  # room for the groups' names


def draw_outside_swr_series(
    ax: Axes, outside: pd.DataFrame, parts: pd.DataFrame
) -> None:
    """The gain outside SWRs: the full decoder minus the SWR-only decoder per spike
    over every immobile bin outside SWRs, then over the bins farther than each
    of :data:`OUTSIDE_SWR_MARGINS_S` from every SWR (``parts``' ``beyond_swr_margin``
    rows), a dot per animal and the mean with its interval, a thin line through
    the means, the margins' label under their tick labels
    (:data:`DISTANCE_LABEL`), a dashed zero line and which way the full decoder
    is better."""
    rows = [outside.loc[(RESTRICTION_FIGURE_FAMILY, RESTRICTION_FIGURE_OUTSIDE_SWR)]]
    rows += [
        parts.loc[
            ("beyond_swr_margin", margin, RESTRICTION_FIGURE_FAMILY, "gain_per_spike")
        ]
        for margin in OUTSIDE_SWR_MARGINS_S
    ]
    names = [f"{margin * 1000:g}" for margin in OUTSIDE_SWR_MARGINS_S]
    xs = [0.0, *(SERIES_START + k for k in range(len(OUTSIDE_SWR_MARGINS_S)))]
    draw_full_minus_swr_axis(ax, xs, rows, 1.05, through_means=True)
    ax.set_xlim(-0.45, xs[-1] + 0.6)
    ax.set_xticks(xs, [DIFFERENCE_LABELS["outside_swr"], *names])
    # the parts' label, centered under their tick labels
    ax.annotate(
        DISTANCE_LABEL, (float(np.mean(xs[1:])), 0), xycoords=("data", "axes fraction"),
        xytext=(0, -PARTS_LABEL_DROP), textcoords="offset points", ha="center",
        va="top",
        fontsize=SMALL_SIZE, annotation_clip=False,
    )  # fmt: skip


def draw_event_traces(axes: Sequence[Axes], traces: dict) -> None:
    """The traces panel: Figure 2g's signals around event starts, one axis per
    signal, a thin line per animal and class (each animal's median over its
    events, ``traces``: :func:`read_traces`' ``event_traces``), the
    classes in Figure 2g's colors and named past the last axis, a dotted line
    at the start."""
    t_ms = 1000 * np.asarray(traces["time"])
    for ax, signal in zip(axes, TRIGGERED_SIGNALS, strict=True):
        for kind, is_swr in WINDOW_KINDS.items():
            color = EVENT_KINDS[is_swr][1]
            ax.plot(t_ms, traces[kind][signal].to_numpy(), color=color, lw=0.5)
        ax.axvline(0, color="k", lw=0.5, ls=":")
        ax.set_xlim(t_ms[0], t_ms[-1])
        ax.set_xticks([-100, 0, 100])
        ax.set_title(SIGNAL_LABELS[signal], fontsize=SMALL_SIZE, pad=3)
        ax.yaxis.set_major_locator(MaxNLocator(3))
    # the classes named past the last axis, clear of the lines
    for k, is_swr in enumerate(WINDOW_KINDS.values()):
        name, color = EVENT_KINDS[is_swr]
        axes[-1].text(
            1.08, 0.75 - 0.2 * k, name, transform=axes[-1].transAxes, ha="left",
            va="top", fontsize=SMALL_SIZE, color=readable_text_color(color),
        )  # fmt: skip
    axes[0].set_ylabel("Each animal's\nmedian")
    for ax in axes:
        ax.set_xlabel("")
    axes[len(axes) // 2].set_xlabel("Time from event start (ms)", loc="left")


def draw_hpd(ax: Axes, distribution: pd.DataFrame) -> None:
    """The posterior-spread panel: for no-SWR and SWR events, the percent of
    the class's events in each HPD coverage bin on a log axis, a thin line per
    animal and their mean in bold (``shared_statistics.animal_mean``) in the class's color, the classes named in
    their colors, and a dotted line at the tight cut
    (``distribution``: :func:`read_inputs`' ``hpd``)."""
    for group, color in GROUP_COLORS.items():
        mine = distribution[distribution["group"] == group]
        by_animal = mine.pivot(index="bin_low", columns="animal", values="percent")
        high = mine.groupby("bin_low")["bin_high"].first()
        center = np.sqrt(by_animal.index.to_numpy() * high.to_numpy())
        ax.plot(center, by_animal.to_numpy(), color=color, lw=0.5, alpha=0.5)
        ax.plot(
            center, animal_mean(mine, "bin_low", "percent").to_numpy(), color=color,
            lw=1.2,
        )  # fmt: skip
    ax.axvline(HPD_TIGHT_CM, color=REFERENCE_COLOR, lw=0.6, ls=":", zorder=1)
    ax.text(
        HPD_TIGHT_CM / 1.08, 0.98, f"{HPD_TIGHT_CM:g} cm",
        transform=ax.get_xaxis_transform(), ha="right", va="top",
        fontsize=SMALL_SIZE, color=REFERENCE_COLOR,
    )  # fmt: skip
    # the cut's label ends left of its line and the names start right of it,
    # so that they cannot run into each other in a wider font
    low, high = HPD_BIN_EDGES_CM[0], HPD_BIN_EDGES_CM[-1]
    right_of_cut = np.log(1.08 * HPD_TIGHT_CM / low) / np.log(high / low)
    name_groups(ax, right_of_cut, 0.98, 0.17, "left")
    ax.set_xscale("log")
    ax.set_xlim(low, high)
    ax.set_xticks(HPD_TICKS_CM, [f"{t:g}" for t in HPD_TICKS_CM])
    ax.xaxis.set_minor_locator(NullLocator())
    top = distribution.groupby(["group", "bin_low"])["percent"].max().max()
    ax.set_ylim(0, 1.35 * top)  # room for the names above the lines
    ax.set_xlabel(HPD_XLABEL)
    ax.set_ylabel(HPD_YLABEL)
    ax.set_title("Posterior spread", fontsize=SMALL_SIZE, pad=3)


def draw_four_decoders(axes: Sequence[Axes], statistics: SwrAllowedStatistics) -> None:
    """The fourth-decoder panel: three axes with the fourth decoder (``statistics``:
    :func:`read_inputs`' ``four_decoders``), in
    ``restriction_totals.RESTRICTION_FIGURE_FAMILY`` per spike. First axis: the
    SWR-only, SWR-allowed and full decoders over the local-only decoder over
    all the immobile time (``SWR_ALLOWED_FIGURE_DECODERS``). Second axis: at
    each time (:data:`FOUR_DECODER_TIMES`) the two differences of
    :data:`FOUR_DECODER_DIFFERENCES`, named in a legend above the axis. Third
    axis: the full minus the SWR-allowed decoder outside SWRs in the no-SWR
    held-out events."""
    ax_decoders, ax_difference, ax_events = axes
    family = RESTRICTION_FIGURE_FAMILY
    times = statistics.times
    draw_decoder_gain_axis(
        ax_decoders,
        {m: times.loc[("immobile", family, m)] for m in SWR_ALLOWED_FIGURE_DECODERS},
    )
    xs, rows, hollow = [], [], []
    for k, time in enumerate(FOUR_DECODER_TIMES):
        for j, measure in enumerate(SWR_ALLOWED_FIGURE_DIFFERENCES):
            xs.append(k * FOUR_DECODER_GROUP + j * FOUR_DECODER_STEP)
            rows.append(times.loc[(time, family, measure)])
            hollow.append(FOUR_DECODER_DIFFERENCES[measure][1])
    draw_full_minus_swr_axis(
        ax_difference, xs, rows, 1.08, hollow=hollow, ylabel=FOUR_DECODER_YLABEL,
        cue=False,
    )  # fmt: skip
    centers = [
        k * FOUR_DECODER_GROUP + (FOUR_DECODER_STEP + MEAN_OFFSET) / 2
        for k in range(len(FOUR_DECODER_TIMES))
    ]
    ax_difference.set_xlim(-0.45, xs[-1] + MEAN_OFFSET + 0.45)
    ax_difference.set_xticks(
        centers, [DIFFERENCE_LABELS[name] for name in FOUR_DECODER_TIMES.values()]
    )
    handles = [
        Line2D(
            [], [], ls="", marker="o", ms=2.5, color=NEUTRAL_DOTS, mew=0.5,
            mfc="none" if open_dots else NEUTRAL_DOTS,
        )
        for _, open_dots in FOUR_DECODER_DIFFERENCES.values()
    ]  # fmt: skip
    ax_difference.legend(
        handles, [name for name, _ in FOUR_DECODER_DIFFERENCES.values()],
        loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
        fontsize=SMALL_SIZE, handletextpad=0.1, columnspacing=1.0,
        borderaxespad=0.2,
    )  # fmt: skip
    draw_full_minus_swr_axis(
        ax_events, [0.0], [statistics.events.loc[RESTRICTION_FIGURE_EVENT_GROUP]],
        1.08, ylabel=FOUR_DECODER_EVENTS_YLABEL,
    )  # fmt: skip
    ax_events.set_xlim(-0.45, MEAN_OFFSET + 0.45)
    ax_events.set_xticks([0.0], [EVENTS_LABELS["in_events"]])
    ax_events.set_title(EVENTS_TITLE, fontsize=SMALL_SIZE, pad=3)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def _n_axes(panel: str) -> int:
    rows, columns = PANEL_GRIDS.get(panel, (1, 1))
    return rows * columns


def panel_axes(fig: Figure, panel: str) -> list[Axes]:
    """A panel's axes of a figure :func:`generate_figure` made, row by row
    (several for a panel of :data:`PANEL_GRIDS`)."""
    start = sum(_n_axes(p) for p in PANELS[: PANELS.index(panel)])
    return fig.axes[start : start + _n_axes(panel)]


def generate_figure(inputs: dict) -> Figure:
    """Build the figure from :func:`read_inputs`, the panels in :data:`PANELS`'
    order and laid out by :data:`LAYOUT`."""
    heights = {panel: span for span, row in LAYOUT for panel, _, _ in row}
    spans = panel_spans([row for _, row in LAYOUT], COLUMNS)
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    axes = {}
    columns = {}
    for panel in PANELS:  # the axes in the panels' order (panel_axes)
        (bottom, top), (left, right) = heights[panel], spans[panel]
        rows, n_columns = PANEL_GRIDS.get(panel, (1, 1))
        shares = np.asarray(PANEL_ROW_SHARES.get(panel, (1,) * rows), dtype=float)
        frame_heights = (
            (top - bottom) * (1 - ROW_GAP * (rows - 1)) * shares / shares.sum()
        )
        tops = (
            top
            - np.concatenate([[0.0], np.cumsum(frame_heights[:-1])])
            - np.arange(rows) * ROW_GAP * (top - bottom)
        )
        axes[panel] = [
            fig.add_axes(
                (left, tops[r] - frame_heights[r], right - left, frame_heights[r])
            )
            for r in range(rows)
            for _ in range(n_columns)
        ]
        columns[panel] = [axes[panel][c::n_columns] for c in range(n_columns)]

    def fit(panel: str) -> None:
        fit_panel(
            fig,
            columns[panel],
            *spans[panel],
            weights=PANEL_COLUMN_SHARES.get(panel),
            gap=PANEL_COLUMN_GAPS.get(panel, 0.0),
        )

    # placed once before they are drawn, so that a panel that places a text
    # by its axes' positions sees their proportions, then again on what they draw
    for panel in PANELS:
        fit(panel)
    draw_state_calibration(axes["state_calibration"][0], inputs["state_calibration"])
    draw_no_spike_mua(axes["no_spike_mua"][0], inputs["no_spike_mua"])
    draw_nonlocal_probability(
        axes["nonlocal_probability"][0], inputs["nonlocal_probability"]
    )
    draw_decoded_distance(axes["decoded_distance"][0], *inputs["decoded_distance"])
    draw_rule_sweep(axes["rule_sweep"], inputs["rule_sweep"])
    draw_boundary_timing(*axes["boundary_timing"], inputs["boundary_timing"])
    draw_nonlocal_mua(*axes["nonlocal_mua"], *inputs["nonlocal_mua"])
    draw_left_out_split(axes["left_out_split"][0], inputs["left_out_split"])
    draw_outside_swr_series(axes["outside_distance"][0], *inputs["outside_distance"])
    draw_event_traces(axes["event_traces"], inputs["event_traces"])
    draw_hpd(axes["hpd"][0], inputs["hpd"])
    draw_four_decoders(axes["four_decoders"], inputs["four_decoders"])
    for panel in PANELS:
        fit(panel)
    fig.draw_without_rendering()
    panels = {LETTERS[panel]: panel_list for panel, panel_list in axes.items()}
    letter_rows, letter_columns = letter_layout([row for _, row in LAYOUT], LETTERS)
    place_letters(
        fig,
        panels,
        rows=letter_rows,
        columns=list(letter_columns.values()),
        above=LETTER_HEIGHT,
        left_of=0.004,
    )
    check_placement(fig, panels)
    return fig


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
