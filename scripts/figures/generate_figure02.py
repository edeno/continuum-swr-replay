#!/usr/bin/env python
"""Generate Figure 2: non-local events and SWRs are partly overlapping inventories.

The decoder of Figure 1 applied to the recordings, and how its non-local events
relate to sharp-wave ripples (SWRs, Kay's detector, the SWR source):

    a — The decode over a run and the rest after it, with the 2D track. Lanes
        mark the immobile (analyzed) time, the Local / non-local / No-Spike
        stretches and the SWRs; the counted non-local events are shaded.
    b — Six examples: a non-local event inside an SWR, one that starts before
        its SWR, one that ends after it, two with no SWR (a trajectory and a
        stationary event), and an SWR whose content is Local; the ripple
        traces carry panel a's dashed SWR threshold.
    c — The SWRs by content: with a non-local event, mostly Local, or other.
        A dot per animal (its share, the mean over its epochs of each epoch's
        share), the bar and its percent the mean of the dots; the count is
        every SWR. The decoder finds the replay SWRs are known for.
    d — The non-local events by SWR overlap (bars and animal dots as in c): most
        events have no SWR.
    e — For each SWR definition, MUA and the Widloski detector, c's and d's
        questions: the share of its own events that overlap a non-local event,
        and the share of non-local events its events overlap (bars and animal
        dots as in c).
    f — Each SWR-overlapping event's start and end minus its SWR's: the percent
        of event-SWR pairs per 5 ms on a log axis, bars the animals' mean and
        outlines the animals.
    g — Around non-local event starts, SWR vs no-SWR: ripple power, MUA, theta
        power and head speed, each animal-balanced: the line the mean over
        animals of each animal's median over its events (the lines of Figure
        2's supplement, panel j), the band the mean over animals of each
        animal's quartiles (a spread over events, every animal weighted the
        same), with the epochs' level between events, theta's running level and
        the immobility threshold. Both kinds of event come with a rise in MUA
        that is not due to movement or theta: the animal is still and theta
        stays at its level between events, far below running.
    h — Tetrode participation: the events are population events. The share of
        tetrodes with a spike in each event against a Local-rest window of its
        length lying wholly inside one well visit, one line per animal from its
        controls to its events, for no-SWR and SWR events (Local rest in the
        Local state's color, the events in their class's). Each group's
        controls have its events' lengths, so the two groups' levels are not
        compared: a longer window holds a spike on more tetrodes
        (``participation.participation_statistics``;
        ``Processed-Data/event_participation``, refused when it was not made
        over the analyses' subset from the current well visits; the events
        summary's participation values).
    i — Where the non-local time sits in ripple power. Top, confident Local
        time (Local probability above 0.8) and confident non-local time
        (non-local probability above 0.8) spread over ripple power, each the
        mean over animals of each animal's mean over its epochs of its percent
        per 0.25 z, in its state's color and named
        (``state_by_signal.state_distribution``;
        ``Processed-Data/state_by_signal``), with the share of non-local time
        below the SWR threshold written: no threshold separates the two.
        Bottom, the share of immobile time bins decoded non-local at each
        power, one line per animal and their mean, which has no step at the
        threshold (``nonlocal_by_signal.rest_signal_counts``;
        ``Processed-Data/nonlocal_by_signal``). Drawn by
        ``visualization.signal_share_panels.draw_share_by_signal``, which
        Figure 2's supplement draws MUA with).
    j — Do the spikes of a left-out tetrode point to the decoded place? At
        the left, a drawing of the test
        (``visualization.evidence_drawings.draw_left_out_test``): each tetrode
        is left out of the decoder's smoother in turn and its spikes scored at
        every bin of each event under the other tetrodes' position distribution
        (the decoded place), summed over tetrodes and bins and divided by the
        event's spikes, against two places it must fit better than. At the
        right, one axis with the two comparisons, no-SWR and SWR events (the
        immobile non-local events that start inside a well visit) each: a dot
        per animal in the event's color, the animals' mean with its 95 %
        interval (``heldout_gain_panels.draw_animal_gain``), and a dashed line
        at zero, equal fit. 1, the animal's place: the gain over local coding
        around the animal, encodings held out in folds of whole trials
        (``tetrode_holdout.holdout_statistics`` of
        ``Processed-Data/event_tetrode_holdout``, column ``gain_per_spike``;
        the events summary's ``*_heldout_gain_per_spike`` values; Figure 2's
        supplement splits it into the bins with and without a spike). 2,
        another event's place: the fit to the decoded place minus the fit to
        another event's decoded place of the same group, fold and width bin
        (the same run, column ``agreement_minus_null_per_spike``; the events summary's
        ``*_heldout_agreement_minus_reference_per_spike`` values; by width it
        is for the text, the ``*_shared_content_*`` values). The run is refused when it was not
        held out in folds of whole trials or has not every column of the run.
    k — Does the decoder need non-local states outside SWRs? At the left, a
        drawing of the two decoders compared
        (``visualization.evidence_drawings.draw_restriction_test``): the SWR-only
        decoder (the non-local states allowed only inside SWRs, where only they
        are allowed) and the full decoder (every state at any time), each
        scored by every tetrode's one-step-ahead predictive score of each bin,
        the whole decoder (place fields, state transitions, initial conditions)
        fit on folds of whole trials and each fold scored under the fit that
        did not see it. Every value is the full minus the SWR-only decoder over
        immobile bins inside a well visit (the first nose-poke at a reward well
        to the last poke-out there), per epoch, summed over its bins and
        divided by the spikes in them, a dot per animal the median over its
        epochs, the animals' mean with its 95 % interval, a dashed line at
        zero, equal fit. First axis, by time: inside SWRs
        (``restriction_totals.inside_swr_gain_statistics``: below zero, forcing
        the non-local states there predicts the spikes better) and outside SWRs
        (the test; ``restriction_totals.outside_swr_gain_statistics``); the bins
        outside SWRs are chosen by speed, ripple detection and the well visits
        alone, not by the decoder. Second axis, on its own scale: where the
        gain outside SWRs sits, over the bins of the no-SWR events detected
        from the held-out decode (each event's value per spike, the epoch's
        median over its events, then the animal's median over its epochs;
        ``RestrictionFigureStatistics.events``) and over the other bins outside
        SWRs (summed per epoch as on the first axis). The two reduce
        differently, so they show where the gain is, not one quantity twice.
        The gains over the local-only decoder are in the events summary, not
        drawn; Figure 2's supplement draws the gain outside SWRs away from the
        SWRs. All from
        ``Processed-Data/event_nonlocal_restriction_heldout_events`` (its time
        sums and events) by ``restriction_totals.restriction_figure_statistics``,
        which the events summary's values share.

Panels d and f are drawn from the canonical event table by the functions the
events summary uses (``event_statistics.overlap_class_statistics``,
``boundary_offsets``); c and e from ``nonlocal_comparison`` on the event
features run the summary reads; g's between-events and running lines from
``event_statistics.reference_medians`` of the brain-state reference; h from
the participation run over the selected events, as the events summary reads
it; i from the stored counts by the functions the events summary uses
(``nonlocal_by_signal.animal_nonlocal_counts``,
``event_statistics.nonlocal_below_threshold_statistics``); j from the
held-out tetrode runs over the selected events; k from the held-out event run's
events and time sums as the events summary reads them
(``reported.inputs.read_heldout_restriction``: the events detected from the
held-out decode that start inside a well visit, and the time inside a well
visit) and by ``restriction_totals.restriction_figure_statistics``, refused when
the restriction run is not the fold fit on folds of whole trials or was not
made from the current decode frames and SWR source, when the held-out event
run was not made from the current restriction run or per-bin
facts run, or when the per-bin facts, which give the time and the events' start
their well-visit flag, were not made from the current session cache. Panel g is a
per-time-bin signal no stored output holds, computed from the aggregated decode
frames (cached), whose event counts by SWR overlap must equal the canonical
table's.

Every event panel is over the events that start inside a well visit (the first
poke-in at a well to the final poke-out), the canonical events. The canonical table is checked
against the frames whole and then selected once
(``cohort_selection.select_visit_context``), the stored per-event tables are read
over the selected events (``cohort_selection.select_events_in``), and g's
averages are over the frames' rows of those events
(``triggered_analysis.triggering_events_in``). The SWRs and the other detectors'
events of c and e are those whose first sample is inside a visit, against the
selected non-local events (``nonlocal_comparison.read_comparison``); g's
quiet-rest line and i count only the time inside a visit (their producers' in-visit
mask).

Usage
-----
    python scripts/figures/generate_figure02.py
    python scripts/figures/generate_figure02.py --force-recompute
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import string
import warnings
from collections.abc import Sequence
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec, SubplotSpec
from matplotlib.text import Annotation
from matplotlib.ticker import NullLocator
from non_local_detector.environment import Environment

import continuum_swr_replay.visualization.figure_utilities as figure_utilities
from continuum_swr_replay.analysis.detection.nonlocal_events import (
    get_event_types,
    load_data_over_time,
)
from continuum_swr_replay.analysis.detector_comparison import nonlocal_comparison
from continuum_swr_replay.analysis.detector_comparison.feature_run import (
    EVENT_FEATURES_RUN,
)
from continuum_swr_replay.analysis.detector_comparison.inventory_run import (
    SWR_DEFINITIONS,
)
from continuum_swr_replay.analysis.event_definition.analyzed_time import (
    read_swr_table,
    swr_intervals,
)
from continuum_swr_replay.analysis.event_definition.cohort_selection import (
    check_visit_context_starts,
    select_events_in,
    select_visit_context,
)
from continuum_swr_replay.analysis.event_definition.nonlocal_event_rule import (
    nonlocal_state_probability,
)
from continuum_swr_replay.analysis.event_definition.swr_source import (
    event_table_epochs,
    ripple_power_inputs,
    swr_epoch_id,
    swr_source_name,
)
from continuum_swr_replay.analysis.event_evidence.heldout_runs import (
    read_tetrode_holdout,
)
from continuum_swr_replay.analysis.event_evidence.participation import (
    participation_statistics,
)
from continuum_swr_replay.analysis.event_evidence.restriction_totals import (
    RESTRICTION_FIGURE_EVENT_GROUP,
    RESTRICTION_FIGURE_FAMILY,
    RESTRICTION_FIGURE_INSIDE_SWR,
    RESTRICTION_FIGURE_OUTSIDE_EVENTS,
    RESTRICTION_FIGURE_OUTSIDE_SWR,
    RestrictionFigureStatistics,
    restriction_figure_statistics,
)
from continuum_swr_replay.analysis.event_evidence.tetrode_holdout import (
    holdout_statistics,
    window_scores,
)
from continuum_swr_replay.analysis.event_properties.class_or_continuum import (
    KAY_ZSCORE_THRESHOLD,
)
from continuum_swr_replay.analysis.event_properties.event_statistics import (
    boundary_offsets,
    nonlocal_below_threshold_statistics,
    overlap_class_animal_shares,
    overlap_class_statistics,
    reference_medians,
)
from continuum_swr_replay.analysis.event_properties.nonlocal_by_signal import (
    DETECTION_LEVELS,
    animal_nonlocal_counts,
)
from continuum_swr_replay.analysis.event_properties.state_by_signal import (
    state_distribution,
)
from continuum_swr_replay.analysis.event_properties.triggered_analysis import (
    TRIGGERED_SIGNALS,
    TRIGGERED_WINDOW_S,
    WINDOW_KINDS,
    compute_triggered_average_curves,
    triggering_events_in,
)
from continuum_swr_replay.analysis.event_tables.prepare_data import (
    behavioral_context_cache_path,
    well_visit_times,
)
from continuum_swr_replay.data_loaders import load_data
from continuum_swr_replay.data_loaders.epoch_register import (
    analyzed_decode_files,
    analyzed_epoch_ids,
)
from continuum_swr_replay.decoding.files import data_over_time_suffix, decode_paths
from continuum_swr_replay.decoding.identifiers import (
    format_epoch_identifier,
    parse_nwb_filename,
    session_name,
)
from continuum_swr_replay.decoding.results import load_results
from continuum_swr_replay.intervals import contiguous_true_runs
from continuum_swr_replay.parameters import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    DEFAULT_NON_LOCAL_THRESHOLD,
)
from continuum_swr_replay.paths import (
    PARTICIPATION_NAME,
    event_features_root,
    processed_data_dir,
)
from continuum_swr_replay.reported.formatting import format_number
from continuum_swr_replay.reported.inputs import (
    EventsInputs,
    default_inputs,
    event_table_staleness,
    panel_refusal,
    participation_staleness,
    read_checked,
    read_figure_restriction,
)
from continuum_swr_replay.reported.values import rel
from continuum_swr_replay.runs.io import (
    ok_epochs,
    read_complete_run,
    read_manifest,
    sha256_file,
)
from continuum_swr_replay.runs.staleness import read_output_manifest
from continuum_swr_replay.visualization.evidence_drawings import (
    draw_left_out_test,
    draw_restriction_test,
)
from continuum_swr_replay.visualization.figure_utilities import save_figure
from continuum_swr_replay.visualization.heldout_gain_panels import (
    DIFFERENCE_LABELS,
    EQUAL_FIT,
    EVENTS_LABELS,
    GROUP_COLORS,
    REFERENCE_COLOR,
    animal_values,
    draw_animal_gain,
    draw_full_minus_swr_axis,
    draw_zero_line,
    name_groups,
    name_zero_line,
)
from continuum_swr_replay.visualization.inventory_examples import detector_color
from continuum_swr_replay.visualization.palettes import (
    ANIMAL_LINE_COLOR,
    COLORS,
    EVENT_KINDS,
    OVERLAP_CLASSES,
    SIGNAL_LABELS,
    readable_text_color,
)
from continuum_swr_replay.visualization.panel_layout import (
    fit_panel,
    letter_layout,
    panel_spans,
    place_letters,
)
from continuum_swr_replay.visualization.plots import (
    plot_linear_track_graph_strip,
    plot_ripple_consensus_trace_zscore,
    plot_spike_times,
    plot_track_graph_2d,
)
from continuum_swr_replay.visualization.publication_style import (
    LABEL_SIZE,
    LINE_WIDTH_LIMITS_PT,
    SMALL_SIZE,
    check_journal_limits,
    check_placement,
    drawn_extent,
    set_publication_style,
)
from continuum_swr_replay.visualization.signal_share_panels import (
    draw_share_by_signal,
)
from continuum_swr_replay.visualization.simulated_run_panels import ANIMAL_POS_COLOR

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%d-%b-%y %H:%M:%S"
)
figure_utilities.set_figure_defaults(context="paper")
set_publication_style()
warnings.filterwarnings("ignore")

FIGURE = "Figure 2"
"""The figure's name in its refusals (``reported.inputs.panel_refusal``)."""

# ---------------------------------------------------------------------------
# Data choices
# ---------------------------------------------------------------------------

EXAMPLE_NWB_FILE = "wilbur20210329_.nwb"
EXAMPLE_EPOCH = "02_r1"
"""The example session of panels a and b."""
SESSION_INPUTS = ("data", "results", "detector", "env")
"""The example session's inputs (panels a and b): ``load_data``'s data and the
decode's results, detector and environment (``load_results``)."""
OVERVIEW_START_S = 1617036555.9
OVERVIEW_DURATION_S = 12.6
"""Panel a: a run and the rest after it in the example session (start and
duration, s)."""
EXAMPLES = {
    "Inside SWR": (2185516843327500879, "swr_contains_event"),
    "Starts before SWR": (63427667006281341, "single_swr_partial"),
    "Ends after SWR": (3628740333881355820, "single_swr_partial"),
    "No SWR, trajectory": (5216998177801911428, "no_overlap"),
    "No SWR, stationary": (8826983562067101773, "no_overlap"),
}
"""Panel b's examples, chosen by eye from contact sheets of the example
session's events with peak non-local probability >= 0.95, no other event and no
other SWR within the window, and a sharp posterior (small HPD coverage): each
title's event_id and the overlap subclass it is drawn as, which
:func:`example_windows` checks."""
PARTIAL_SIDE = {"Starts before SWR": "start", "Ends after SWR": "end"}
"""Panel b's partial examples: the one side on which the event runs past its
SWR."""
LOCAL_SWR_START_S = 1617036373.982971
"""Panel b's SWR with Local content: its start in the SWR source (s). No
non-local event overlaps it and the decode is Local through it."""
EXAMPLE_HALF_WINDOW_S = 0.2
"""Panel b: half the window around each example's center (s)."""
OFFSET_BINS_MS = np.arange(-100, 101, 5)
"""Panel f's bins, ms: event bound minus SWR bound."""

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

# Inches: saved with 0.1 in margins, about 17.3 cm (Cell Press full width).
WIDTH = 6.5
HEIGHT = 7.86
BOTTOM_RIGHT = 0.995
"""Where the last panel of each bottom row ends (figure width): the figure is
saved cropped to its ink, so the edge is the page's."""
TOP_LAYOUT = (
    (("overview", "left", None),),
    (("examples", "left", None),),
    (
        ("swr_content", "left", None),
        ("event_overlap", "event_overlap", None),
        ("detector_overlap", "detector_overlap", None),
        ("boundary_offsets", "boundary_offsets", None),
    ),
)
"""The first three rows, top to bottom, their panels left to right, each with
the column its drawing starts at (``left`` is the figure's left column); their
widths are set by their grids (:func:`generate_figure`), not by edges."""
BOTTOM_LAYOUT = (
    (
        ("event_traces", "left", "detector_overlap"),
        ("participation", "detector_overlap", "boundary_offsets"),
        ("nonlocal_ripple_power", "boundary_offsets", BOTTOM_RIGHT),
    ),
    (
        ("left_out", "left", "detector_overlap"),
        ("decoder_comparison", "detector_overlap", BOTTOM_RIGHT),
    ),
)
"""The bottom rows, left to right, each panel with the left and right edges it
spans (``panel_layout.panel_spans``): a panel starts at a column (the left
edge of the third row's panel of that name, or the figure's left column) and
ends :data:`COLUMN_GAP` before the next panel's column, the last of a row at
:data:`BOTTOM_RIGHT`."""
LAYOUT = (*TOP_LAYOUT, *BOTTOM_LAYOUT)
"""The figure's rows, top to bottom. Read in this order the panels are in the
order the text cites them, so their letters follow from it: the one place the
panels' order is set."""
PANELS = tuple(panel for row in LAYOUT for panel, _, _ in row)
"""The panels in the order the text cites them."""
LETTERS = dict(zip(PANELS, string.ascii_lowercase, strict=False))
"""Each panel's letter, by its position."""
_letter_rows, _letter_columns = letter_layout(LAYOUT, LETTERS)
LETTER_ROWS = tuple(_letter_rows)
"""The letters that share a height (convention 10): each row's."""
LETTER_COLUMNS = tuple(_letter_columns.values())
"""The letters that share a left edge (convention 10): the panels that start
at one column."""
OUTER_HEIGHTS = (1.85, 1.88, 1.02, 1.12, 1.23)
"""The rows' heights, in proportion."""
OUTER_HSPACE = 0.72
OVERVIEW_GAP = 0.02
"""Panel a: the space between the track drawing and the labels of the example
beside it (figure width)."""
COLUMN_GAP = 0.03
"""The bottom rows: the space between a panel's drawing and the column where
the next panel of its row starts (figure width)."""
TRACES_WSPACE = 0.32
"""Panel g: the gap between its two columns of strips, as a share of a strip's
width."""
TRACES_CENTER = 1 + TRACES_WSPACE / 2
"""The middle of panel g, in the left column's axes coordinates: where the
legend and the x label are centered."""
REST_EVENT_X = {"no SWR": (0.0, 1.0), "SWR": (2.4, 3.4)}
"""Panel h: each group's Local-rest and event positions on the x axis."""
LEFT_OUT_WIDTHS = (1.25, 1.0)
"""Panel j's drawing and axis frames' widths, in proportion (``fit_panel``)."""
FIT_X = {
    ("animal", "no SWR"): 0.0,
    ("animal", "SWR"): 0.7,
    ("other", "no SWR"): 2.2,
    ("other", "SWR"): 2.9,
}
"""Panel j: each comparison and group's x position."""
DECODER_AXES_WIDTHS = (1.1, 0.8, 0.85)
"""Panel k's frame widths, in proportion: its drawing, the full minus the
SWR-only decoder by time (inside and outside SWRs), and that difference outside
SWRs in and outside the held-out events (``fit_panel``)."""
DECODER_AXES_GAP = 0.012
"""Panels j and k: the space between one axes' drawing and the next (figure
width)."""
HEADING_OFFSET_PT = 10
"""How far above its axes' top a panel's heading sits (points), clear of the
axes' titles."""
TRACK_LINEWIDTH = LINE_WIDTH_LIMITS_PT[0]
TRACK_EDGE_LINEWIDTH = LINE_WIDTH_LIMITS_PT[1]
"""The track plotters' line widths, inside the journal's limits."""

# ---------------------------------------------------------------------------
# Labels and colors
# ---------------------------------------------------------------------------

C_LOCAL = COLORS["Local"]
C_NOSPIKE = COLORS["No-Spike"]
C_OTHER = "white"
"""Panel c's 'Other' SWRs: an outlined bar, no state's color (they are mostly
mixed-state, not No-Spike)."""
C_REFERENCE = REFERENCE_COLOR
"""A reference level (panels j and k's zeros, panel g's between-events level): a neutral
gray, not the No-Spike state's."""
C_NL = COLORS["Non-Local"]
C_SWR = COLORS["SWR"]
C_QUIET = COLORS["All Events"]
DETECTOR_LABELS = dict(
    zip(
        nonlocal_comparison.FIGURE_DETECTORS,
        ("Kay (SWR)", "Karlsson", "Yu", "Zugaro", "Carey", "MUA", "Widloski"),
        strict=True,
    )
)
"""Panel e's name of each detector it draws
(``nonlocal_comparison.FIGURE_DETECTORS``, in its order)."""
LEFT_OUT_HEADING = "Do the spikes of a left-out tetrode point to the decoded place?"
DECODER_HEADING = "Does the decoder need non-local states outside SWRs?"
"""Panels j and k's headings, centered over each panel."""
FIT_YLABEL = "Better fit to the\ndecoded place\n(nats/spike)"
FIT_GROUPS = {
    "animal": "1  animal's\nplace",
    "other": "2  another\nevent's place",
}
"""Panel j's x groups: what the decoded place's fit is compared with."""
DECODER_YLABEL = "Better fit with non-local\nat any time (nats/spike)"


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def read_inputs(
    nwb_file_name: str = EXAMPLE_NWB_FILE,
    epoch_name: str = EXAMPLE_EPOCH,
    use_cache: bool = True,
) -> dict:
    """The figure's inputs: the example session and each panel's statistics,
    read and checked as the events summary reads them.

    Parameters
    ----------
    nwb_file_name, epoch_name : str
        The example session (panels a and b).
    use_cache : bool, default True
        Load the cached session and frame inputs (:func:`_compute_figure_inputs`)
        when they exist; when False (``--force-recompute``), recompute and
        overwrite them.

    Returns
    -------
    dict
        ``session``: the example session's ``data``, ``results``, ``detector``
        and ``env`` (a, b); then by panel (:data:`PANELS`): ``overview``, the
        session's selected events; ``examples``, :func:`example_windows`;
        ``swr_content``, Kay's row of ``other_detector_statistics`` and each
        animal's shares (:func:`detector_comparison`); ``event_overlap``,
        ``overlap_class_statistics`` and ``overlap_class_animal_shares`` of the
        selected events; ``detector_overlap``, the share of non-local events each
        detector overlaps and the share of its events that overlap one;
        ``boundary_offsets``, ``boundary_offsets`` and the events' animals;
        ``event_traces``, the triggered curves and ``reference_medians``;
        ``participation``, :func:`read_participation`;
        ``nonlocal_ripple_power``, the counts by ripple power and the states by
        ripple power; ``left_out``, ``holdout_statistics`` of the two
        comparisons; ``decoder_comparison``, ``restriction_figure_statistics``.

    Raises
    ------
    ValueError
        If the canonical event table is not the decode frames' or a stored
        output is not current (each refusal names its panel and what to rerun).
    """
    inputs = default_inputs()
    table = read_swr_table(inputs.event_table)
    # the one selection: the events that start inside a well visit
    events = select_visit_context(table)
    cache_path = _figure02_cache_path(nwb_file_name, epoch_name)
    if use_cache and cache_path.exists():
        logging.info(f"Loading cached render inputs from {cache_path}...")
        cached = joblib.load(cache_path)
    else:
        cached = _compute_figure_inputs(nwb_file_name, epoch_name, events)
        logging.info(f"Caching render inputs to {cache_path}...")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(cached, cache_path)
    # the frames hold every event: the whole table is checked against them
    check_table_against_frames(
        overlap_class_statistics(table), cached["frame_event_counts"]
    )
    animal, day = parse_nwb_filename(nwb_file_name)
    session_events = events[
        (events["animal"] == animal)
        & (events["session_date"] == day)
        & (events["epoch"] == epoch_name)
    ]
    # every analyzed epoch with an event: an epoch with none has no table
    epochs = event_table_epochs(table, analyzed_epoch_ids())
    # the time masks read the visits themselves: they and the labels must agree
    check_visit_context_starts(
        table, {identifier: well_visit_times(identifier) for identifier in epochs}
    )
    # the held-out tetrode run, as the events summary reads it: held out in
    # folds of whole trials, over the selected events
    holdout = window_scores(
        read_tetrode_holdout(
            inputs.tetrode_holdout,
            epochs,
            events["event_id"],
            rel(inputs.tetrode_holdout),
        )
    )
    # the held-out event run as the events summary reads it, refused when it or
    # what it was made from is not current
    heldout, refusal = read_figure_restriction(inputs, epochs)
    if refusal is not None:
        raise ValueError(panel_refusal(FIGURE, LETTERS["decoder_comparison"], *refusal))
    return {
        "session": {name: cached[name] for name in SESSION_INPUTS},
        "overview": session_events,
        "examples": example_windows(
            session_events, cached["data"]["ripple_times"], cached["results"]
        ),
        "swr_content": (cached["kay"], cached["kay_content_animals"]),
        "event_overlap": (
            overlap_class_statistics(events),
            overlap_class_animal_shares(events),
        ),
        "detector_overlap": (cached["overlap"], cached["with_nonlocal"]),
        "boundary_offsets": (boundary_offsets(events), events["animal"]),
        "event_traces": (
            cached["triggered_curves"],
            reference_medians(
                read_checked(
                    inputs.brain_state_reference.parent,
                    "brain-state reference",
                    inputs.brain_state_reference.name,
                ),
                ["ripple", "mua", "theta"],
            ),
        ),
        "participation": read_participation(inputs, epochs, events["event_id"]),
        "nonlocal_ripple_power": (
            animal_nonlocal_counts(
                read_checked(inputs.nonlocal_by_signal, "counts by signal"),
                "ripple_power",
            ),
            # the Local and non-local time by ripple power, for the top strip
            state_distribution(
                read_checked(inputs.state_by_signal, "counts by signal and state"),
                "ripple_power",
            ),
        ),
        "left_out": {
            "animal": holdout_statistics(holdout, "event", "gain_per_spike"),
            "other": holdout_statistics(
                holdout, "event", "agreement_minus_null_per_spike"
            ),
        },
        "decoder_comparison": restriction_figure_statistics(
            heldout.events, heldout.time_sums, heldout.bin_s
        ),
    }


def read_participation(
    inputs: EventsInputs, epochs: Sequence[str], event_ids: pd.Series
) -> pd.DataFrame:
    """The participation panel's statistics: ``participation_statistics`` of the
    participation run over ``epochs``, read over the selected events
    (``event_ids``), as the events summary reads it.

    Raises
    ------
    ValueError
        If the run was not made from the current event table, its settings do
        not record the analyses' subset, or it did not read the current session
        cache (``reported.inputs.participation_staleness``): the events summary
        marks its values pending then.
    """
    folder = inputs.event_participation
    manifest = read_output_manifest(folder)
    reason = event_table_staleness(
        manifest, sha256_file(inputs.event_table), rel(folder)
    ) or participation_staleness(manifest, epochs, rel(folder))
    if reason is not None:
        raise ValueError(
            panel_refusal(
                FIGURE,
                LETTERS["participation"],
                reason,
                "scripts/event_evidence/measure_event_participation.py",
            )
        )
    return participation_statistics(
        select_events_in(
            read_complete_run(folder, PARTICIPATION_NAME, epochs), event_ids
        )
    )


def check_table_against_frames(classes: dict, frame_counts: dict[str, int]) -> None:
    """Refuse a canonical event table whose counts by SWR overlap are not the frames'.

    Equal counts do not prove the two hold the same events.
    """
    table_counts = {
        "swr": classes["n_events"] - classes["n_non_swr"],
        "non_swr": classes["n_non_swr"],
        "total": classes["n_events"],
    }
    if table_counts != frame_counts:
        raise ValueError(
            "the canonical event table's event counts by SWR overlap are not "
            f"the decode frames' (table {table_counts}, frames {frame_counts}); "
            "rebuild the canonical tables from these decode frames"
        )


def example_windows(
    events: pd.DataFrame, ripple_times: pd.DataFrame, results: xr.Dataset
) -> dict[str, tuple[float, float]]:
    """Panel b's examples as (start, end) seconds, each checked.

    Raises
    ------
    AssertionError
        If an example's ``event_id`` is not one row of the table, its overlap
        subclass is not the one it is drawn as, or a partial example does not
        run past its SWR on its title's side alone; or if the Local SWR is not
        one of the source's SWRs, a non-local event overlaps it, or the decode is
        not mostly Local through it.
    """
    windows = {}
    for title, (event_id, subclass) in EXAMPLES.items():
        rows = events[events["event_id"] == event_id]
        if len(rows) != 1:
            raise AssertionError(f"event_id {event_id} matches {len(rows)} rows")
        row = rows.iloc[0]
        if row["overlap_subclass"] != subclass:
            raise AssertionError(
                f"{title}: event {event_id} is {row['overlap_subclass']}"
            )
        side = PARTIAL_SIDE.get(title)
        sides = (
            row["event_start_s"] < row["matched_swr_start"],
            row["event_end_s"] > row["matched_swr_end"],
        )
        if side is not None and sides != (side == "start", side == "end"):
            raise AssertionError(
                f"{title}: event {event_id} does not run past its SWR at the "
                f"{side} alone"
            )
        windows[title] = (float(row["event_start_s"]), float(row["event_end_s"]))
    starts = ripple_times["start_time"].to_numpy()
    k = int(np.argmin(np.abs(starts - LOCAL_SWR_START_S)))
    if abs(starts[k] - LOCAL_SWR_START_S) > 1e-3:
        raise AssertionError("the Local SWR is not one of the source's SWRs")
    start, end = float(starts[k]), float(ripple_times["end_time"].iloc[k])
    if ((events["event_start_s"] < end) & (events["event_end_s"] > start)).any():
        raise AssertionError("a non-local event overlaps the Local SWR")
    local = float(
        results.acausal_state_probabilities.sel(
            time=slice(start, end), states="Local"
        ).mean()
    )
    if not local > 0.8:  # NaN (no decode bin in the SWR) fails too
        raise AssertionError(
            f"the decode is not mostly Local in the Local SWR ({local})"
        )
    windows["SWR, mostly Local"] = (start, end)
    return windows


# ---------------------------------------------------------------------------
# The render cache: the example session and the decode frames
# ---------------------------------------------------------------------------


# The render cache's version, part of its file name. The name also carries the
# SWR source's name and fingerprints of the epoch register's analyzed epochs, of
# the decode files and the detector comparison's files read, and of the panel
# settings (``_settings_fingerprint``), so a change to any of them makes a new
# cache by itself. Bump this whenever the cached values could change for any
# other reason; git history holds what each earlier bump was for.
# 14: j's share by ripple power left the bundle; it is drawn from the stored
#     counts (then ``Processed-Data/nonlocal_by_ripple_power``).
# 15: i's averages and c and e's comparison are over the events that start
#     inside a well visit.
# 16: g's quartiles are animal-balanced (the mean over animals of each
#     animal's quartiles).
FIGURE02_CACHE_VERSION = 16


def _compute_figure_inputs(
    nwb_file_name: str, epoch_name: str, events: pd.DataFrame
) -> dict:
    """Run the expensive loads and derivations; return the cached render inputs.

    The example session's ``data``/``results``/``detector``/``env`` (panels a
    and b), the triggered-average curves (g) over the frames' rows of
    ``events`` (the selected events), the frames' immobile non-local events by
    SWR overlap, which the canonical event table is checked against, and the
    detector comparison (c and e). The multi-GB ``data_over_time`` is consumed
    here and not returned.
    """
    logging.info(f"Loading session {nwb_file_name} {epoch_name}...")
    # the ripple trace is the source's (the Kay consensus over R-T's tetrodes)
    data = dict(
        load_data(
            nwb_file_name,
            epoch_name,
            ripple_power=ripple_power_inputs(swr_epoch_id(nwb_file_name, epoch_name)),
        )
    )
    results, detector, env = load_results(
        nwb_file_name, epoch_name, decoding_brain_area="HPC", root=processed_data_dir()
    )
    logging.info("Loading the aggregated decode frames (panel g)...")
    data_over_time = load_data_over_time(processed_data_dir())
    # The SWRs drawn: the session's SWR source intervals, the ones the event
    # table's classes are computed against.
    data["ripple_times"] = swr_intervals(
        data_over_time, nwb_file_name=nwb_file_name, epoch_name=epoch_name
    )
    event_types = get_event_types(data_over_time)
    event_counts = frame_event_counts(event_types)
    logging.info("Computing the triggered averages (panel g)...")
    triggered_curves = compute_triggered_average_curves(
        data_over_time,
        triggering_events_in(event_types, events),
        list(TRIGGERED_SIGNALS),
        time_col="start_time",
        window_sec=TRIGGERED_WINDOW_S,
    )
    del data_over_time
    logging.info("Reading the detector comparison (panels c and e)...")
    comparison = detector_comparison(events)
    return {
        "nwb_file_name": nwb_file_name,
        "epoch_name": epoch_name,
        "data": data,
        "results": results,
        "detector": detector,
        "env": env,
        "frame_event_counts": event_counts,
        "triggered_curves": triggered_curves,
        **comparison,
    }


def frame_event_counts(event_types: dict) -> dict[str, int]:
    """The decode frames' immobile non-local events, by SWR overlap.

    Parameters
    ----------
    event_types : dict
        :func:`get_event_types` of the aggregated frames.

    Returns
    -------
    dict
        ``swr``, ``non_swr`` and ``total``: the events (one per session and
        label) that overlap an SWR, that do not, and all of them.
    """

    def n_events(frame: pd.DataFrame) -> int:
        return len(
            frame[["animal", "day", "epoch", "non_local_event"]].drop_duplicates()
        )

    return {
        "swr": n_events(event_types["non_local_event_immobile_overlap_SWR_event"]),
        "non_swr": n_events(
            event_types["non_local_event_immobile_no_overlap_SWR_event"]
        ),
        "total": n_events(event_types["non_local_event_immobile"]),
    }


def detector_comparison(events: pd.DataFrame) -> dict:
    """Panels c and e: the event features run's comparison, as the summary reads it.

    Over the other detectors' events whose first sample is inside a well visit
    (each ok epoch's visits from the session cache) and the non-local events of
    ``events``, the selected events.

    Returns
    -------
    dict
        ``kay``: :func:`nonlocal_comparison.other_detector_statistics`'s row of
        Kay's SWRs (c); ``overlap``: the share of non-local events each of
        ``nonlocal_comparison.FIGURE_DETECTORS`` overlaps (the events
        summary's ``nonlocal_events_found_by_*`` values) and ``with_nonlocal``, the share of
        each one's events that overlap a non-local event (e); and
        ``kay_content_animals``, each animal's share of Kay's SWRs in each
        content class (c).
    """
    run = event_features_root() / EVENT_FEATURES_RUN
    visits = {i: well_visit_times(i) for i in ok_epochs(read_manifest(run))}
    comparison = nonlocal_comparison.read_comparison(run, visits)
    others, directional = comparison.others, comparison.directional
    return {
        "kay": nonlocal_comparison.other_detector_statistics(others).loc["kay"],
        "kay_content_animals": nonlocal_comparison.content_class_animal_shares(
            others, "kay"
        ),
        "overlap": nonlocal_comparison.nonlocal_events_overlapped(
            nonlocal_comparison.nonlocal_event_memberships(
                run, directional, events["event_id"]
            ),
            nonlocal_comparison.FIGURE_DETECTORS,
        ),
        "with_nonlocal": nonlocal_comparison.events_overlapping_nonlocal(
            others, nonlocal_comparison.FIGURE_DETECTORS
        ),
    }


def _figure02_cache_path(nwb_file_name: str, epoch_name: str) -> Path:
    """Path for the cached Figure-2 render inputs (joblib bundle).

    A single joblib bundle is used rather than netCDF because the bundle mixes
    a session ``data`` dict, an xarray ``results`` Dataset whose ``state_bins``
    MultiIndex netCDF cannot serialize, the fitted detector and environment
    objects, and small DataFrames; joblib (pickle) preserves them all exactly.
    """
    cohort = hashlib.sha256("\n".join(analyzed_epoch_ids()).encode()).hexdigest()
    decodes = _decode_files_fingerprint(nwb_file_name, epoch_name)
    comparison = _comparison_fingerprint()
    return (
        processed_data_dir()
        / "intermediates"
        / f"{session_name(nwb_file_name, epoch_name)}_figure02_v{FIGURE02_CACHE_VERSION}_"
        f"{swr_source_name()}_register{cohort[:12]}_decodes{decodes[:12]}"
        f"_features{EVENT_FEATURES_RUN}_comparison{comparison[:12]}"
        f"_settings{_settings_fingerprint()[:12]}_cache.joblib"
    )


def _decode_files_fingerprint(nwb_file_name: str, epoch_name: str) -> str:
    """SHA-256 over the decode files ``_compute_figure_inputs`` reads.

    The files are the analyzed epochs' ``data_over_time`` (as
    ``load_data_over_time`` finds them) and the example session's HPC
    ``results.nc`` and ``detector.pkl`` (as ``load_results`` names them), all
    in the data root. Each contributes its path, size and ``st_mtime_ns``, or
    ``missing``, so the fingerprint is a stat per file, not a read.
    """
    example = decode_paths(
        format_epoch_identifier(nwb_file_name, epoch_name, "HPC"),
        processed_data_dir(),
    )
    files = [
        *analyzed_decode_files(processed_data_dir(), data_over_time_suffix()),
        example.results,
        example.detector,
    ]
    return _stat_fingerprint(files)


def _comparison_fingerprint() -> str:
    """A stat fingerprint of the detector comparison's files, of the canonical
    event table (the selected events) and of the session
    cache's entries (the well visits); a run without a manifest is
    fingerprinted as missing (``detector_comparison`` then raises)."""
    run = event_features_root() / EVENT_FEATURES_RUN
    if not (run / "manifest.json").exists():
        return _stat_fingerprint([run / "manifest.json", default_inputs().event_table])
    return _stat_fingerprint(
        [
            *nonlocal_comparison.comparison_files(run),
            default_inputs().event_table,
            *(
                behavioral_context_cache_path({"session_id": identifier})
                for identifier in ok_epochs(read_manifest(run))
            ),
        ]
    )


def _settings_fingerprint() -> str:
    """A hash of the module settings the cached inputs are computed with."""
    settings = (
        TRIGGERED_WINDOW_S,
        list(TRIGGERED_SIGNALS),
        list(DETECTOR_LABELS),
    )
    return hashlib.sha256(repr(settings).encode()).hexdigest()


def _stat_fingerprint(files: Sequence[Path]) -> str:
    """A hash of each file's path, size and ``st_mtime_ns`` (or ``missing``)."""
    lines = []
    for path in files:
        if path.exists():
            stat = path.stat()
            lines.append(f"{path}\t{stat.st_size}\t{stat.st_mtime_ns}")
        else:
            lines.append(f"{path}\tmissing")
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Drawing, in panel order
# ---------------------------------------------------------------------------


def summed_posterior(window: xr.Dataset, env: Environment) -> np.ndarray:
    """Posterior over position summed over the states with one, (n_time, n_bins)."""
    post = window.acausal_posterior
    states = post.state.values
    total = sum(
        post.isel(state_bins=np.flatnonzero(states == s)).to_numpy()
        for s in ("Local", "Non-Local Continuous", "Non-Local Fragmented")
    )
    total[:, ~env.is_track_interior_] = np.nan
    return total


def draw_posterior(
    ax: Axes,
    data: dict,
    results: xr.Dataset,
    env: Environment,
    sl: slice,
    t_offset: float,
    max_pos: float,
) -> None:
    """The decoded position (bone_r) with the animal's position (magenta dots)."""
    window = results.sel(time=sl)
    cmap = plt.get_cmap("bone_r").copy()
    cmap.set_bad("#e6e6e6")
    centers = env.place_bin_centers_
    t = window.time.values - t_offset
    ax.imshow(
        summed_posterior(window, env).T,
        aspect="auto",
        origin="lower",
        extent=[t[0], t[-1], centers[0], centers[-1]],
        cmap=cmap,
        vmin=0,
        vmax=0.1,
        interpolation="none",
        rasterized=True,
    )
    position = data["position_info"].loc[sl]
    ax.scatter(
        position.index - t_offset,
        position.linear_position,
        color=ANIMAL_POS_COLOR,
        s=0.4,
        lw=0,
        rasterized=True,
    )
    ax.set_ylim(0, max_pos)


def draw_state_stack(ax: Axes, results: xr.Dataset, sl: slice, t_offset: float) -> None:
    """Stacked state probabilities: Non-local (both states), Local, No-Spike."""
    sp = results.acausal_state_probabilities.sel(time=sl)
    ax.stackplot(
        sp.time.values - t_offset,
        nonlocal_state_probability(sp).to_numpy(),
        sp.sel(states="Local").to_numpy(),
        sp.sel(states="No-Spike").to_numpy(),
        colors=[C_NL, C_LOCAL, C_NOSPIKE],
        lw=0,
        rasterized=True,
    )
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 1])


def add_track_strip(ax: Axes, data: dict, max_pos: float) -> Axes:
    """The linearized track as the posterior's y key, flush to its left."""
    strip = ax.inset_axes([-0.075, 0.0, 0.07, 1.0])
    plot_linear_track_graph_strip(
        data["track_graph"],
        data["linear_edge_order"],
        data["linear_edge_spacing"],
        ax=strip,
        max_position=max_pos,
        edge_linewidth=TRACK_EDGE_LINEWIDTH,
    )
    strip.set_ylabel("")
    ax.tick_params(labelleft=False, left=False)
    return strip


def spans(mask: np.ndarray, t: np.ndarray) -> list[tuple[float, float]]:
    """(first, last) time of each run of True in ``mask``."""
    return [(t[a], t[b - 1]) for a, b in contiguous_true_runs(mask)]


def label_reward_well(ax: Axes, track_graph) -> None:
    """Name the wells once: label the top-left well (a degree-1 node) as a
    reward well; the other well dots, here and on the decode's axis, match it."""
    wells = [
        track_graph.nodes[n]["pos"]
        for n in track_graph.nodes
        if track_graph.degree(n) == 1
    ]
    x, y = min(wells, key=lambda pos: (-round(pos[1]), pos[0]))
    ax.annotate(
        "Reward well", (x, y), xytext=(-5, 0), textcoords="offset points",
        ha="right", va="center", fontsize=SMALL_SIZE,
    )  # fmt: skip


def fit_track_to_height(fig: Figure, ax_track: Axes, left: float) -> None:
    """Size the track's equal-aspect axes to fill its row's height, at ``left``."""
    box = ax_track.get_position(original=True)
    x0, x1 = ax_track.get_xlim()
    y0, y1 = ax_track.get_ylim()
    fig_w, fig_h = fig.get_size_inches()
    width = box.height * fig_h / fig_w * abs(x1 - x0) / abs(y1 - y0)
    ax_track.set_position([left, box.y0, width, box.height])


def draw_overview(
    fig: Figure,
    spec: SubplotSpec,
    session: dict,
    session_events: pd.DataFrame,
    max_pos: float,
) -> tuple[Axes, list[Axes]]:
    """Panel a: the 2D track, then the decode over a run and the rest after it
    (returned: the track's axes and the example's five axes, top to bottom)."""
    data, results, env = session["data"], session["results"], session["env"]
    start, dur = OVERVIEW_START_S, OVERVIEW_DURATION_S
    sl = slice(start, start + dur)
    gs = spec.subgridspec(1, 2, width_ratios=[1.6, 5.0], wspace=0.3)
    ax_track = fig.add_subplot(gs[0])
    plot_track_graph_2d(
        data["track_graph"],
        data["position_info"],
        ax=ax_track,
        edge_order=data["linear_edge_order"],
        occupancy_linewidth=TRACK_LINEWIDTH,
    )
    label_reward_well(ax_track, data["track_graph"])
    rows = gs[1].subgridspec(
        5, 1, height_ratios=[1.2, 1.45, 0.42, 0.42, 0.42], hspace=0.14
    )
    ax_pos, ax_lane, ax_rip, ax_st, ax_sp = (fig.add_subplot(rows[i]) for i in range(5))
    draw_posterior(ax_pos, data, results, env, sl, start, max_pos)
    strip = add_track_strip(ax_pos, data, max_pos)
    # the top only: a 0 at the strip's foot would sit on the lanes' first name
    strip.set_yticks([0, max_pos], ["", f"{max_pos:g}"])
    ax_pos.text(
        0.48, 0.45, "animal's position", transform=ax_pos.transAxes,
        color=ANIMAL_POS_COLOR, fontsize=SMALL_SIZE, ha="left", va="top",
    )  # fmt: skip

    position = data["position_info"].loc[sl]
    t_pos = position.index.to_numpy() - start
    probabilities = results.acausal_state_probabilities.sel(time=sl)
    t_dec = probabilities.time.values - start
    lanes = {"Immobile": 4, "Local": 3, "Non-local": 2, "No-Spike": 1, "SWR": 0}
    fills = {
        "Immobile": C_QUIET,
        "Local": C_LOCAL,
        "Non-local": C_NL,
        "No-Spike": C_NOSPIKE,
        "SWR": C_SWR,
    }

    def lane(name: str, s0: float, e0: float) -> None:
        ax_lane.barh(
            lanes[name],
            max(e0 - s0, 0.03),
            left=s0,
            height=0.72,
            color=fills[name],
            lw=0,
        )

    speed = position["head_speed"].to_numpy()
    for s0, e0 in spans(speed < DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC, t_pos):
        lane("Immobile", s0, e0)
    for state in ("Local", "No-Spike"):
        state_probability = probabilities.sel(states=state).to_numpy()
        for s0, e0 in spans(state_probability > DEFAULT_NON_LOCAL_THRESHOLD, t_dec):
            lane(state, s0, e0)
    in_window = session_events[
        (session_events["event_end_s"] > start)
        & (session_events["event_start_s"] < start + dur)
    ]
    for row in in_window.itertuples():
        s0, e0 = row.event_start_s - start, row.event_end_s - start
        lane("Non-local", s0, e0)
        ax_pos.axvspan(s0, e0, color=C_NL, alpha=0.25, lw=0)
    ripples = data["ripple_times"]
    for s0, e0 in zip(ripples["start_time"], ripples["end_time"], strict=True):
        if e0 > start and s0 < start + dur:
            lane("SWR", s0 - start, e0 - start)
    ax_lane.set_ylim(-0.5, 4.5)
    ax_lane.set_yticks(list(lanes.values()), list(lanes))
    for label, name in zip(ax_lane.get_yticklabels(), lanes, strict=True):
        label.set_color("k" if name == "Immobile" else readable_text_color(fills[name]))
    ax_lane.tick_params(axis="both", length=0)
    ax_lane.spines[["left", "bottom"]].set_visible(False)

    ripple = data["zscored_ripple_consensus_trace"].loc[sl]
    ripple = ripple.iloc[:, 0] if ripple.ndim > 1 else ripple
    ax_rip.plot(ripple.index - start, ripple.to_numpy(), color="k", lw=0.5)
    ax_rip.axhline(KAY_ZSCORE_THRESHOLD, color=C_SWR, lw=0.5, ls="--")
    ax_rip.text(
        1.0, KAY_ZSCORE_THRESHOLD, "SWR threshold", transform=ax_rip.get_yaxis_transform(),
        ha="right", va="bottom", fontsize=SMALL_SIZE, color=C_SWR,
    )  # fmt: skip
    ax_rip.set_ylim(-2, 8)
    draw_state_stack(ax_st, results, sl, start)
    # the thin strips name one tick each: two labels on a strip touch the
    # neighboring strips' labels
    ax_rip.set_yticks([0, 5], ["", "5"])
    ax_st.set_yticks([0, 1], ["", "1"])
    ax_sp.fill_between(t_pos, speed, color=C_QUIET, lw=0)
    ax_sp.axhline(DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC, color="k", ls=":", lw=0.5)
    ax_sp.text(
        dur, DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC + 2,
        f"{DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC:g} cm/s",
        ha="right", va="bottom", fontsize=SMALL_SIZE,
    )  # fmt: skip
    ax_sp.set_xlabel("Time (s)")
    for ax, label, x in (
        (ax_pos, "Decoded\nposition (cm)", -0.1),
        (ax_rip, "Ripple (z)", -0.05),
        (ax_st, "State prob.", -0.05),
        (ax_sp, "Speed (cm/s)", -0.05),
    ):
        ax.set_ylabel(label, rotation=0, ha="right", va="center")
        ax.yaxis.set_label_coords(x, 0.5)
    for ax in (ax_pos, ax_lane, ax_rip, ax_st, ax_sp):
        ax.set_xlim(0, dur)
    for ax in (ax_pos, ax_lane, ax_rip, ax_st):
        ax.tick_params(labelbottom=False)
    return ax_track, [ax_pos, ax_lane, ax_rip, ax_st, ax_sp]


def draw_examples(
    fig: Figure, spec: SubplotSpec, session: dict, windows: dict, max_pos: float
) -> Axes:
    """Panel b: one column per example (ripple, raster, decode, states)."""
    data, results, detector, env = (session[name] for name in SESSION_INPUTS)
    n = len(windows)
    gs = spec.subgridspec(
        4, n, hspace=0.08, wspace=0.14, height_ratios=[0.5, 0.85, 1.1, 0.5]
    )
    first = None
    for j, (title, (start, end)) in enumerate(windows.items()):
        ax_rip, ax_raster, ax_post, ax_states = (
            fig.add_subplot(gs[r, j]) for r in range(4)
        )
        center = (start + end) / 2
        sl = slice(center - EXAMPLE_HALF_WINDOW_S, center + EXAMPLE_HALF_WINDOW_S)
        plot_ripple_consensus_trace_zscore(data, sl, ax_rip)
        # the SWR threshold as panel a draws it, under the trace
        ax_rip.axhline(KAY_ZSCORE_THRESHOLD, color=C_SWR, lw=0.5, ls="--", zorder=-1)
        plot_spike_times(data, sl, detector, env, results, ax_raster)
        draw_posterior(ax_post, data, results, env, sl, center, max_pos)
        draw_state_stack(ax_states, results, sl, center)
        for ax in (ax_rip, ax_raster):
            ax.set_xlim(sl.start, sl.stop)
            ax.tick_params(labelbottom=False)
        for ax in (ax_post, ax_states):
            ax.set_xlim(-EXAMPLE_HALF_WINDOW_S, EXAMPLE_HALF_WINDOW_S)
        ax_rip.set_title(title, fontsize=LABEL_SIZE, pad=3)
        ax_rip.set_ylim(-2, 8)
        ax_rip.set_yticks([0, 5])
        ax_post.tick_params(labelbottom=False, labelleft=False)
        ax_post.set_yticks([0, max_pos])
        ax_states.set_xticks([-0.1, 0, 0.1], ["-0.1", "0", "0.1"])
        ax_states.set_xlabel("Time from event center (s)" if j == n // 2 else "")
        if j == 0:
            first = (ax_rip, ax_raster, ax_post, ax_states)
            for ax, label in zip(
                first,
                ("Ripple\n(z)", "Tetrode", "Decoded\npos. (cm)", "State\nprob."),
                strict=True,
            ):
                ax.set_ylabel(label)
        else:
            for ax in (ax_rip, ax_raster, ax_states):
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)
    add_track_strip(first[2], data, max_pos)
    for ax in first:
        ax.yaxis.set_label_coords(-0.42, 0.5)
    return first[0]


def draw_class_bars(
    ax: Axes,
    rows: Sequence[tuple[str, float, pd.Series, str, bool]],
    title: str,
    xlabel: str,
) -> None:
    """Classes as horizontal bars, top to bottom: the mean over animals as a bar
    with its percent in a column past the axis's 100, each animal's share as a
    dot.

    ``rows`` are (label, the mean over animals, the animals' shares, color,
    bold label).
    """
    n_rows = len(rows)
    for i, (_, share, animals, color, _) in enumerate(rows):
        y = n_rows - 1 - i
        ax.barh(
            y, 100 * share, color=color, height=0.65, linewidth=0.5,
            edgecolor="#777777" if color == C_OTHER else "none",
        )  # fmt: skip
        ax.scatter(
            100 * animals.to_numpy(), y + np.linspace(-0.18, 0.18, len(animals)),
            s=3, color="k", lw=0, zorder=3,
        )  # fmt: skip
        ax.text(
            1.0, y, f"{format_number(share, 'pctnum')}%",
            transform=ax.get_yaxis_transform(), ha="right", va="center",
            fontsize=SMALL_SIZE, fontweight="bold" if rows[i][4] else "normal",
        )  # fmt: skip
    ax.set_yticks(range(n_rows), [label for label, *_ in rows][::-1])
    for tick, (*_, bold) in zip(ax.get_yticklabels()[::-1], rows, strict=True):
        tick.set_fontweight("bold" if bold else "normal")
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, n_rows - 0.4)
    ax.set_xlim(0, 125)  # 100 to 125 holds the percents
    ax.spines["bottom"].set_bounds(0, 100)
    ax.set_xticks([0, 50, 100])
    ax.set_xlabel(xlabel)
    # right-aligned, so a title wider than the bars runs over the class labels
    # rather than under the next panel's letter
    ax.set_title(title, fontsize=SMALL_SIZE, pad=3, loc="right")


def draw_swr_bars(ax: Axes, kay: pd.Series, animals: pd.DataFrame) -> None:
    """Panel c: the SWRs by content.

    Dots are the animals (``content_class_animal_shares``), bars their mean
    (``other_detector_statistics``' ``*_animal_mean``); the count is every SWR.
    """
    no_nonlocal = kay["no_nonlocal_fraction_animal_mean"]
    mostly_local = kay["mostly_local_fraction_animal_mean"]
    rows = [
        ("With non-local", 1 - no_nonlocal, animals["with_nonlocal"], C_NL, True),
        ("Mostly Local", mostly_local, animals["mostly_local"], C_LOCAL, False),
        ("Other", no_nonlocal - mostly_local, animals["other"], C_OTHER, False),
    ]
    draw_class_bars(ax, rows, f"SWRs (n = {int(kay['n_events']):,})", "% SWRs")


def draw_events_bars(ax: Axes, classes: dict, animals: pd.DataFrame) -> None:
    """Panel d: the non-local events by SWR overlap.

    Dots are the animals (``overlap_class_animal_shares``), bars their mean
    (``overlap_class_statistics``' ``*_fraction_animal_mean``); the count is
    every event.
    """
    rows = [
        (name, classes[f"{c}_fraction_animal_mean"], animals[c], color, c == "non_swr")
        for c, (name, color) in OVERLAP_CLASSES.items()
    ]
    draw_class_bars(
        ax, rows, f"Non-local events (n = {classes['n_events']:,})", "% events"
    )


def draw_detector_bars(
    ax: Axes, shares: pd.DataFrame, n_rows: int, xlabel: str, show_labels: bool
) -> None:
    """One bar per detector (the mean over animals) with a dot per animal, top
    to bottom.

    The SWR definitions other than Kay's are drawn faded.
    """
    for i, (detector, row) in enumerate(shares.iterrows()):
        y = n_rows - 1 - i
        faded = detector in SWR_DEFINITIONS and detector != "kay"
        ax.barh(
            y, 100 * row["animal_mean"], color=detector_color(detector), height=0.65,
            alpha=0.55 if faded else 1.0,
        )  # fmt: skip
        animals = 100 * row["animals"]
        ax.scatter(
            animals, y + np.linspace(-0.18, 0.18, len(animals)), s=3, color="k", lw=0,
            zorder=3,
        )  # fmt: skip
    ax.set_yticks(range(n_rows), [DETECTOR_LABELS[d] for d in shares.index][::-1])
    ax.tick_params(axis="y", labelleft=show_labels)
    ax.set_ylim(-0.6, n_rows - 0.4)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 104)  # a dot at 100 % is drawn whole
    ax.spines["bottom"].set_bounds(0, 100)
    ax.set_xticks([0, 50, 100])
    ax.set_xlabel(xlabel)


def draw_detector_overlap(
    ax_content: Axes,
    ax_caught: Axes,
    overlap: pd.DataFrame,
    with_nonlocal: pd.DataFrame,
) -> None:
    """Panel e: what each detector's events contain, and what it catches.

    Dots are animals; bars are their mean.

    Left, panel c's question for every detector: the share of its own events
    that overlap a non-local event; right, panel d's: the share of non-local
    events its events overlap. Both tables list the detectors in one order.
    """
    n_rows = len(overlap)
    draw_detector_bars(ax_content, with_nonlocal, n_rows, "% of\ndetections", True)
    ax_content.set_title(
        "Detections with\nnon-local activity", fontsize=SMALL_SIZE, pad=3
    )
    draw_detector_bars(ax_caught, overlap, n_rows, "% of non-local\nevents", False)
    ax_caught.set_title("Non-local\nevents found", fontsize=SMALL_SIZE, pad=3)
    ax_caught.spines["left"].set_visible(False)


def draw_offsets(
    ax_start: Axes, ax_end: Axes, offsets: pd.DataFrame, animal: pd.Series
) -> None:
    """Panel f: event minus SWR start (top) and end (bottom), as the percent of
    event-SWR pairs in each 5 ms bin on a log axis.

    Bars are the animals' mean, outlines the animals. ``offsets`` are
    ``boundary_offsets``; ``animal`` is the event table's column, matched on the
    index. An animal's percent counts all its pairs, those beyond the bins too.
    """
    width = np.diff(OFFSET_BINS_MS)[0]
    centers = OFFSET_BINS_MS[:-1] + width / 2
    for ax, bound, label in ((ax_start, "start", "Start"), (ax_end, "end", "End")):
        offsets_ms = 1000 * offsets[f"{bound}_offset_s"]
        shares = np.column_stack(
            [
                100 * np.histogram(values, OFFSET_BINS_MS)[0] / len(values)
                for _, values in offsets_ms.groupby(animal)
            ]
        )
        ax.bar(centers, shares.mean(axis=1), width=width, color="#bbbbbb", lw=0)
        ax.step(centers, shares, where="mid", color="k", lw=0.5, alpha=0.35)
        ax.axvline(0, color="k", lw=0.5, ls=":")
        ax.set_xlim(OFFSET_BINS_MS[0], OFFSET_BINS_MS[-1])
        ax.set_yscale("log")
        ax.set_ylim(0.03, 60)
        ax.set_yticks([0.1, 1, 10], ["0.1", "1", "10"])
        ax.yaxis.set_minor_locator(NullLocator())
        ax.text(
            0.03, 0.95, label, transform=ax.transAxes, va="top", fontsize=SMALL_SIZE
        )  # fmt: skip
    ax_start.tick_params(labelbottom=False)
    ax_end.set_xlabel("Event − SWR bound (ms)")
    ax_start.set_ylabel("% of pairs per 5 ms")
    ax_start.yaxis.set_label_coords(-0.3, 0.0)


def draw_triggered(axes: Sequence[Axes], curves: dict, rest: pd.DataFrame) -> None:
    """Panel g: ripple power, MUA, theta and head speed around event starts, in
    two rows of two strips (ripple and MUA above theta and speed).

    Every row is animal-balanced: the line is the mean over animals of each
    animal's median over its events, the band the mean over animals of each
    animal's quartiles (``triggered_analysis.animal_balanced_quartiles``), so
    the spread over events is seen, not only the center: ripple and MUA rise
    at the start, theta stays at
    its level between events, far below running, and the animal is still. The
    dashed "between events" line is the epochs' median in immobile bins outside
    every non-local, SWR and MUA event (``reference_medians``).
    """
    ax_rip, ax_mua, ax_theta, ax_speed = axes
    groups = [(key, *EVENT_KINDS[is_swr][::-1]) for key, is_swr in WINDOW_KINDS.items()]
    for ax, signal in (
        (ax_rip, "SWR_zscore"),
        (ax_mua, "MUA_zscore"),
        (ax_theta, "theta_power_zscore"),
        (ax_speed, "head_speed"),
    ):
        curve = curves[signal]
        t_ms = 1000 * np.asarray(curve["time"])
        for key, color, name in groups:
            ax.fill_between(
                t_ms, curve[f"{key}_q25"], curve[f"{key}_q75"], color=color,
                alpha=0.25, lw=0,
            )  # fmt: skip
            ax.plot(t_ms, curve[f"{key}_median"], color=color, lw=1.0, label=name)
    for ax, signal, word in (
        (ax_rip, "SWR_zscore", "ripple"),
        (ax_mua, "MUA_zscore", "mua"),
    ):
        curve = curves[signal]
        ax.axhline(rest.loc[word, "rest"], color=C_REFERENCE, lw=0.6, ls="--", zorder=3)
        low = min(np.nanmin(curve[f"{key}_q25"]) for key, _, _ in groups)
        top = max(np.nanmax(curve[f"{key}_q75"]) for key, _, _ in groups)
        ax.set_ylim(min(low, rest.loc[word, "rest"]) - 0.3, top + 1.2)
    ax_theta.axhline(
        rest.loc["theta", "rest"], color=C_REFERENCE, lw=0.6, ls="--", zorder=3
    )
    ax_theta.axhline(rest.loc["theta", "running"], color="k", lw=0.6, ls="--")
    ax_theta.text(
        t_ms[-1] - 5, rest.loc["theta", "running"], "running", ha="right",
        va="bottom", fontsize=SMALL_SIZE,
    )  # fmt: skip
    ax_theta.set_ylim(None, rest.loc["theta", "running"] + 1.3)
    ax_speed.axhline(DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC, color="k", lw=0.6, ls=":")
    ax_speed.set_ylim(0, 1.6 * DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC)
    ax_speed.set_yticks([0, DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC])
    for ax, label in zip(
        axes, (SIGNAL_LABELS[signal] for signal in TRIGGERED_SIGNALS), strict=True
    ):
        ax.axvline(0, color="k", lw=0.5, ls=":")
        ax.set_xlim(t_ms[0], t_ms[-1])
        # on white, so that the event-start line does not run through the name
        ax.text(
            0.03, 1.0, label, transform=ax.transAxes, ha="left", va="top",
            fontsize=SMALL_SIZE, zorder=5,
            bbox={"fc": "white", "ec": "none", "pad": 0.4},
        )  # fmt: skip
    for ax in (ax_rip, ax_mua):
        ax.tick_params(labelbottom=False)
    ax_speed.set_xticks([-100, 0, 100])
    # two centered rows, each narrower than the axes, so the legend stays clear
    # of the row's panel letters: the dashed line just above the axes, the two
    # event classes above it
    style = {
        "frameon": False, "loc": "lower center", "handlelength": 0.8,
        "handletextpad": 0.4, "columnspacing": 0.8, "borderaxespad": 0.5,
        "borderpad": 0.0,
    }  # fmt: skip
    reference = ax_rip.legend(
        [plt.Line2D([], [], color=C_REFERENCE, ls="--", lw=0.6)],
        ["between events"], bbox_to_anchor=(TRACES_CENTER, 1.0), **style,
    )  # fmt: skip
    row = 1.3 * reference.get_texts()[0].get_fontsize() / 72  # one row, inches
    axes_height = ax_rip.get_position().height * ax_rip.figure.get_figheight()
    handles, labels = ax_rip.get_legend_handles_labels()
    ax_rip.legend(
        handles, labels, ncol=2,
        bbox_to_anchor=(TRACES_CENTER, 1.0 + row / axes_height),
        **style,
    )  # fmt: skip
    ax_rip.add_artist(reference)
    # one label under the two columns, centered between them
    ax_theta.set_xlabel("Time from event start (ms)", x=TRACES_CENTER)


def draw_participation(ax: Axes, statistics: pd.DataFrame) -> None:
    """Panel h, the participation panel: tetrode participation, Local rest against
    events, one line per animal from its length-matched Local-rest controls to
    its events, for no-SWR and SWR events (``statistics``:
    :func:`read_participation`)."""
    for group, color in GROUP_COLORS.items():
        x = REST_EVENT_X[group]
        rest, event = (
            animal_values(statistics.loc[(group, kind)], "participation_animals", 100)
            for kind in ("control", "event")
        )
        for animal in event.index.intersection(rest.index):  # paired by animal
            ax.plot(x, (rest[animal], event[animal]), color=ANIMAL_LINE_COLOR, lw=0.5,
                    zorder=1)  # fmt: skip
        ax.scatter(np.full(rest.size, x[0]), rest, s=6, color=COLORS["Local"], lw=0,
                   zorder=2)  # fmt: skip
        ax.scatter(np.full(event.size, x[1]), event, s=6, color=color, lw=0,
                   zorder=2)  # fmt: skip
    ax.set_ylabel("Tetrodes active (%)")
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 50, 100])
    ax.set_xlim(-0.5, 3.9)
    ticks = [x for pair in REST_EVENT_X.values() for x in pair]
    # horizontal, on short lines, so that they do not hang into the next row;
    # "matched": each rest window has its event's duration, so the two rest
    # columns differ as the two kinds of event differ in duration
    rest = "Matched\nlocal\nrest"
    ax.set_xticks(ticks, [rest, "No\nSWR", rest, "SWR"])
    ax.set_title("Participation", fontsize=SMALL_SIZE, pad=3)


def draw_share_by_power(
    ax_spread: Axes, ax_share: Axes, counts: pd.DataFrame, states: pd.DataFrame
) -> None:
    """Panel i: where confident Local and confident non-local time sit in
    ripple power (top) and the share of immobile bins decoded non-local at each
    power (bottom), drawn by ``signal_share_panels.draw_share_by_signal`` with
    the share of non-local time below the SWR threshold written
    (``nonlocal_below_threshold_statistics``, the events summary's value).

    Parameters
    ----------
    ax_spread, ax_share : Axes
    counts : pd.DataFrame
        ``animal_signal_counts`` of the stored counts' ripple power.
    states : pd.DataFrame
        ``state_by_signal.state_distribution`` of ripple power.
    """
    level = DETECTION_LEVELS["ripple_power"]
    below = nonlocal_below_threshold_statistics(counts, level)
    draw_share_by_signal(
        ax_spread, ax_share, counts, threshold=level,
        threshold_label="SWR threshold", threshold_color=C_SWR,
        xlabel="Ripple power (z)",
        below_note=f"{format_number(below['fraction_animal_mean'], 'pctnum')}% "
        "below\nthreshold", states=states,
    )  # fmt: skip


def draw_heading(ax: Axes, text: str) -> Annotation:
    """A panel's heading, :data:`HEADING_OFFSET_PT` above ``ax``'s top; it is
    centered over the panel once the panel's axes are placed
    (:func:`center_heading`)."""
    return ax.annotate(
        text, (0.0, 1.0), xycoords="axes fraction",
        xytext=(0, HEADING_OFFSET_PT), textcoords="offset points",
        ha="left", va="bottom", fontsize=SMALL_SIZE + 1,
    )  # fmt: skip


def center_heading(heading: Annotation, left: float, right: float) -> None:
    """Center a panel's heading (:func:`draw_heading`) between ``left`` and
    ``right`` (figure coordinates). The heading stays in its axes' coordinates:
    a text in figure coordinates moves when the figure is saved cropped to its
    ink."""
    box = heading.axes.get_position()
    heading.xy = (((left + right) / 2 - box.x0) / box.width, 1.0)
    heading.set_ha("center")


def draw_left_out_fit(
    ax_drawing: Axes, ax: Axes, statistics: dict[str, pd.DataFrame]
) -> Annotation:
    """Panel j: do the spikes of a left-out tetrode point to the decoded place?

    Left, the test as a drawing (``evidence_drawings.draw_left_out_test``).
    Right, one axis: how much better the left-out tetrode's spikes fit the
    place decoded from the other tetrodes than (1) the animal's place (the
    gain over local coding over all bins, ``statistics["animal"]``) and (2) the
    place decoded in another event of the same group and fold (the agreement
    minus the shuffled-event reference, all widths, ``statistics["other"]``),
    each ``tetrode_holdout.holdout_statistics`` indexed by group: for no-SWR
    and SWR events a dot per animal and the animals' mean with its 95 %
    interval, the groups named in their colors, a dashed zero line named
    :data:`EQUAL_FIT`. The heading (:data:`LEFT_OUT_HEADING`) sits over the
    drawing."""
    draw_left_out_test(ax_drawing)
    heading = draw_heading(ax_drawing, LEFT_OUT_HEADING)
    for (comparison, group), x in FIT_X.items():
        draw_animal_gain(
            ax, x, statistics[comparison].loc[group], GROUP_COLORS[group], spread=True
        )
    draw_zero_line(ax)
    name_zero_line(ax, EQUAL_FIT, 0.02, offset_pt=1)
    centers = [
        np.mean([x for (c, _), x in FIT_X.items() if c == comparison]) + 0.12
        for comparison in FIT_GROUPS
    ]
    ax.set_xticks(centers, list(FIT_GROUPS.values()))
    ax.set_xlim(-0.4, max(FIT_X.values()) + 0.6)
    low, high = ax.get_ylim()
    ax.set_ylim(min(low, -0.12 * high), high * 1.15)
    name_groups(ax, 0.99, 0.99, 0.1, "right")
    ax.set_ylabel(FIT_YLABEL)
    return heading


def draw_decoder_need(
    ax_drawing: Axes,
    ax_time: Axes,
    ax_outside: Axes,
    statistics: RestrictionFigureStatistics,
) -> Annotation:
    """Panel k: does the decoder need non-local states outside SWRs?

    Left, the two decoders as a drawing
    (``evidence_drawings.draw_restriction_test``). Then two axes of how much
    better the decoder with the non-local states at any time predicts the
    spikes than the one with them only in SWRs (the full minus the SWR-only
    decoder per spike, held-out score): "All immobility time", over the immobile bins
    inside SWRs and outside SWRs inside a well visit
    (``restriction_totals.inside_swr_gain_statistics``,
    ``outside_swr_gain_statistics``); "Outside SWRs", over the bins outside
    SWRs of the no-SWR events detected from the held-out decode and over the
    rest of that time (``RestrictionFigureStatistics.events``,
    :data:`~continuum_swr_replay.analysis.event_evidence.restriction_totals.RESTRICTION_FIGURE_OUTSIDE_EVENTS`).
    Each a dot per animal (the median over its epochs) and the animals' mean
    with its 95 % interval, a dashed zero line named :data:`EQUAL_FIT`; the
    second axis on its own scale (about ten times the first's). The heading
    (:data:`DECODER_HEADING`) sits over the drawing."""
    draw_restriction_test(ax_drawing)
    heading = draw_heading(ax_drawing, DECODER_HEADING)
    family = RESTRICTION_FIGURE_FAMILY
    for ax, rows, title in (
        (
            ax_time,
            {
                "inside_swr": statistics.inside.loc[
                    (family, RESTRICTION_FIGURE_INSIDE_SWR)
                ],
                "outside_swr": statistics.outside.loc[
                    (family, RESTRICTION_FIGURE_OUTSIDE_SWR)
                ],
            },
            "All immobility time",
        ),
        (
            ax_outside,
            {
                "in_events": statistics.events.loc[RESTRICTION_FIGURE_EVENT_GROUP],
                "outside_events": statistics.outside.loc[
                    (family, RESTRICTION_FIGURE_OUTSIDE_EVENTS)
                ],
            },
            "Outside SWRs",
        ),
    ):
        labels = DIFFERENCE_LABELS if ax is ax_time else EVENTS_LABELS
        xs = [float(k) for k in range(len(rows))]
        draw_full_minus_swr_axis(
            ax, xs, list(rows.values()), 1.08, ylabel="", cue=False
        )
        name_zero_line(ax, EQUAL_FIT, 0.02, offset_pt=1)
        ax.set_xlim(-0.45, xs[-1] + 0.6)
        ax.set_xticks(xs, [labels[name] for name in rows])
        ax.set_title(title, fontsize=SMALL_SIZE, pad=3)
    ax_time.set_ylabel(DECODER_YLABEL)
    return heading


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def generate_figure(inputs: dict) -> Figure:
    """Build the figure from :func:`read_inputs`: the panels drawn in their
    grids, the bottom rows placed by :data:`LAYOUT`, and the letters placed by
    its rows and columns."""
    session = inputs["session"]
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    outer = GridSpec(
        len(LAYOUT),
        1,
        figure=fig,
        height_ratios=OUTER_HEIGHTS,
        hspace=OUTER_HSPACE,
        left=0.1,
        right=0.99,
        top=0.985,
        bottom=0.085,
    )
    max_pos = float(np.ceil(session["data"]["position_info"].linear_position.max()))
    ax_a, decode_axes = draw_overview(
        fig, outer[0], session, inputs["overview"], max_pos
    )
    ax_b = draw_examples(fig, outer[1], session, inputs["examples"], max_pos)
    row3 = outer[2].subgridspec(1, 4, width_ratios=[1.1, 1.1, 1.4, 1.15], wspace=0.7)
    ax_c = fig.add_subplot(row3[0])
    draw_swr_bars(ax_c, *inputs["swr_content"])
    ax_d = fig.add_subplot(row3[1])
    draw_events_bars(ax_d, *inputs["event_overlap"])
    detectors = row3[2].subgridspec(1, 2, wspace=0.3)
    ax_e = fig.add_subplot(detectors[0])
    ax_e2 = fig.add_subplot(detectors[1])
    draw_detector_overlap(ax_e, ax_e2, *inputs["detector_overlap"])
    offsets_grid = row3[3].subgridspec(2, 1, hspace=0.35)
    ax_f1 = fig.add_subplot(offsets_grid[0])
    ax_f2 = fig.add_subplot(offsets_grid[1], sharex=ax_f1)
    draw_offsets(ax_f1, ax_f2, *inputs["boundary_offsets"])
    cells = {}
    # the grids set the bottom panels' heights; fit_panel then places them across
    for k, row in enumerate(BOTTOM_LAYOUT, start=len(TOP_LAYOUT)):
        grid = outer[k].subgridspec(1, len(row))
        cells.update(zip((panel for panel, _, _ in row), grid, strict=True))
    triggered = cells["event_traces"].subgridspec(2, 2, hspace=0.3)
    ax_ripple = fig.add_subplot(triggered[0, 0])
    triggered_axes = [
        ax_ripple,
        *(
            fig.add_subplot(triggered[r, c], sharex=ax_ripple)
            for r, c in ((0, 1), (1, 0), (1, 1))
        ),
    ]
    draw_triggered(triggered_axes, *inputs["event_traces"])
    left_out_grid = cells["left_out"].subgridspec(1, 2)
    ax_left_out_drawing = fig.add_subplot(left_out_grid[0])
    ax_left_out = fig.add_subplot(left_out_grid[1])
    headings = {
        "left_out": draw_left_out_fit(
            ax_left_out_drawing, ax_left_out, inputs["left_out"]
        )
    }
    ax_participation = fig.add_subplot(cells["participation"])
    draw_participation(ax_participation, inputs["participation"])
    power_grid = cells["nonlocal_ripple_power"].subgridspec(
        2, 1, height_ratios=[1.5, 1.6], hspace=0.3
    )
    ax_spread = fig.add_subplot(power_grid[0])
    ax_share = fig.add_subplot(power_grid[1], sharex=ax_spread)
    draw_share_by_power(ax_spread, ax_share, *inputs["nonlocal_ripple_power"])
    decoder_grid = cells["decoder_comparison"].subgridspec(1, 3)
    ax_decoder_drawing, ax_time, ax_outside = (
        fig.add_subplot(decoder_grid[k]) for k in range(3)
    )
    headings["decoder_comparison"] = draw_decoder_need(
        ax_decoder_drawing, ax_time, ax_outside, inputs["decoder_comparison"]
    )
    panels = {
        "overview": [ax_a],
        "examples": [ax_b],
        "swr_content": [ax_c],
        "event_overlap": [ax_d],
        "detector_overlap": [ax_e, ax_e2],
        "boundary_offsets": [ax_f1, ax_f2],
        "event_traces": triggered_axes,
        "participation": [ax_participation],
        "nonlocal_ripple_power": [ax_spread, ax_share],
        "left_out": [ax_left_out_drawing, ax_left_out],
        "decoder_comparison": [ax_decoder_drawing, ax_time, ax_outside],
    }

    fig.draw_without_rendering()
    renderer = fig.canvas.get_renderer()
    # panel a's track starts where the drawing of the figure's left column
    # (panels b and c) does
    x_left = (
        min(
            ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted()).x0
            for ax in (ax_b, ax_c)
        )
        - 0.004
    )
    # the columns of the bottom rows (BOTTOM_LAYOUT): where the drawings of the
    # third row's panels start
    starts = {
        "left": x_left + 0.004,
        "detector_overlap": drawn_extent(fig, panels["detector_overlap"]).x0,
        "boundary_offsets": drawn_extent(fig, panels["boundary_offsets"]).x0,
    }
    columns = {
        "event_traces": [triggered_axes[::2], triggered_axes[1::2]],
        "participation": [[ax_participation]],
        "nonlocal_ripple_power": [[ax_spread, ax_share]],
        "left_out": [[ax] for ax in panels["left_out"]],
        "decoder_comparison": [[ax] for ax in panels["decoder_comparison"]],
    }
    fits = {
        "event_traces": {"gap": TRACES_WSPACE},
        "left_out": {
            "weights": LEFT_OUT_WIDTHS, "gap": DECODER_AXES_GAP, "between": "drawings",
        },
        "decoder_comparison": {
            "weights": DECODER_AXES_WIDTHS, "gap": DECODER_AXES_GAP,
            "between": "drawings",
        },
    }  # fmt: skip
    for panel, (left, right) in panel_spans(BOTTOM_LAYOUT, starts, COLUMN_GAP).items():
        fit_panel(fig, columns[panel], left, right, **fits.get(panel, {}))
        if panel in headings:
            center_heading(headings[panel], left, right)
    fit_track_to_height(fig, ax_a, x_left)
    # panel a's example fills the space the track drawing leaves: its labels
    # start a gap after the drawing, and its right edge stays
    fig.draw_without_rendering()
    to_figure = fig.transFigure.inverted()
    labels = max(
        ax.get_position().x0 - ax.get_tightbbox(renderer).transformed(to_figure).x0
        for ax in decode_axes
    )
    decode_left = drawn_extent(fig, [ax_a]).x1 + OVERVIEW_GAP + labels
    for ax in decode_axes:
        box = ax.get_position()
        ax.set_position((decode_left, box.y0, box.x1 - decode_left, box.height))
    fig.draw_without_rendering()
    lettered = {LETTERS[panel]: panels[panel] for panel in PANELS}
    place_letters(
        fig,
        lettered,
        rows=LETTER_ROWS,
        columns=LETTER_COLUMNS,
        above=0.002,
        left_of=0.004,
        ceiling=0.985,
        va="bottom",
        heights={
            # panel a's letter sits at the top of its decode, not of the track
            # drawing
            LETTERS["overview"]: decode_axes[0].get_position().y1 + 0.004,
            # the legend drawn above the traces' axes would lift the row's
            # letters away from the row's other panels
            LETTERS["event_traces"]: [],
        },
    )
    check_placement(fig, lettered)
    return fig


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("paper/figures/figure02")
    )
    parser.add_argument("--nwb-file-name", type=str, default=EXAMPLE_NWB_FILE)
    parser.add_argument("--epoch-name", type=str, default=EXAMPLE_EPOCH)
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help=(
            "Recompute the cached render inputs (the session load and the "
            "aggregated decode frames) instead of loading them."
        ),
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig = generate_figure(
        read_inputs(
            args.nwb_file_name, args.epoch_name, use_cache=not args.force_recompute
        )
    )
    check_journal_limits(fig)
    logging.info(f"Saving to {args.output_dir}...")
    save_figure(fig, "figure02", base_dir=args.output_dir)
    if args.show:
        plt.show()
    else:
        plt.close("all")
    logging.info("Done.")


if __name__ == "__main__":
    main()
