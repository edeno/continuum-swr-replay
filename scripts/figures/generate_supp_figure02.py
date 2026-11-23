#!/usr/bin/env python
"""Generate Supplementary Figure S2, the supplement to Figure 2, its panels in
the order the text cites them, in five rows:

- the decoder on these recordings (a, b): its states calibrated by held-out
  spikes, and MUA in the No-Spike state;
- the count is robust (c, d): the share of events with no SWR at other
  settings of the event rule and of the SWR detector, and the timing of events
  against their SWRs;
- the events are coordinated and specific (e to h): non-local time against
  MUA, the held-out tetrode gain split by bins with and without spikes, the
  posterior's spread, and tetrode participation;
- ripple power and the held-out test (i, j): no step in the events'
  properties at the SWR threshold, and the gain outside SWRs away from the
  SWRs;
- the animals agree (k): Figure 2g's traces animal by animal.

    a — Supports Figure 2a: on these recordings the decoder's states mean
        what they claim. Random 10-bin immobile windows that start inside a
        well visit, grouped by a state's probability (five bins), and the
        held-out evidence for what the state claims, per spike of the scored
        tetrodes, on one axis: for Local, the held-out tetrodes' spikes at the
        animal's place over at mismatched places; for Continuous and
        Fragmented, the place decoded from one half of the tetrodes scored on
        the other half over mismatched places (nats per spike, the spike-mark
        log likelihood ratio over the scored half's spikes, averaged over the
        runs in which that half spiked; a window in which it never spiked has
        no value and is left out). The animals' mean with its 95 % interval, in
        the state's color, each state named at its line's end, and a dashed
        line at zero. No-Spike is not drawn: a confident No-Spike label forces
        both halves silent, so it is no test (panel b covers that state).
        Caveats: the probability bins come from the decode with every tetrode;
        only the remote place is held out by the tetrode split; the encodings
        are fit on running time in other time blocks
        (``state_evidence.calibration_statistics`` over
        ``IN_VISIT_CALIBRATION_METRICS``, re-summarized from the stored
        windows of the held-out tetrode run by
        ``scripts/event_evidence/summarize_state_calibration_in_visits.py``;
        ``Processed-Data/state_calibration_in_visits``; the events summary's
        ``state_calibration_*`` values).

    b — Supports Figure 2a, where the No-Spike state first appears. Multiunit
        activity (``MUA_zscore``, as Figure 2g draws it) around
        the start of No-Spike periods, animal-balanced as Figure 2g: the mean
        over animals of each animal's median over its periods, with the mean
        over animals of each animal's quartiles as the band
        (``triggered_analysis.animal_balanced_quartiles``), a dotted line at
        the start and a dashed line at the MUA level of immobile time outside
        the periods (``no_spike_periods.level_outside_periods``, each epoch's
        median, then each animal's median over its epochs, then the mean over
        animals; Figure 2g's "between events" level is outside
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

    c — Supports Figure 2d, the share of events with no SWR. The text cites
        this panel twice: its event-rule axes with the count of events (the
        share changes little with the rule), and its SWR-threshold axis with
        the ripple-power panels (Figure 2i, panel i), because the share follows
        the threshold, as a threshold on one distribution must. The share of
        the immobile non-local events that start inside a well visit with no
        SWR, at each value of three settings swept one at a time (the others
        at the values the paper uses). Two are settings of the event rule
        (heading "Event rule"): the non-local probability an event stays
        strictly above (0.5 to 0.95) and the minimum run length, drawn as the
        shortest event it keeps (n samples of 2 ms: 14 ms, the rule's 7
        samples or 15 ms, to 100 ms, 50 samples). The third is the SWR detector's
        threshold (heading "SWR detector", 0 to 6 z). A thin line per animal
        (the mean over its epochs of each epoch's share) and their mean over
        animals in bold; the value the paper uses is dotted and marked
        "used" on every axis. Under each axis, the events detected at each
        value (thousands; flat under the SWR threshold, which relabels the
        rule's events). About a third of the events have no SWR even at
        1 z. The events are detected again from the stored decode by the rule
        itself and labeled as the canonical table labels them; at the rule's
        settings they are the canonical events. The SWRs are not detected
        again: an event counts as overlapping an SWR at a threshold when its
        Kay criterion on the decode grid reaches it, a label that agrees with
        the canonical one for 99 % of the events at 2 z
        (``rule_sweep.sweep_statistics``; ``Processed-Data/event_rule_sweep``,
        ``scripts/event_properties/sweep_event_rule.py``; the events summary's
        ``rule_sweep_*`` values).

    d — Supports Figure 2f, which shows that a good share of the events start
        or end more than 25 ms from their SWR without saying which way. Each
        SWR-overlapping event of the canonical event table that starts inside
        a well visit, against its matched SWR (the pairs and offsets of Figure
        2f, ``event_statistics.boundary_offsets``: event minus SWR, so a
        positive start offset is an event that starts later than its SWR, a
        positive end offset one that ends later), classed by its start
        (earlier, within 25 ms, later) and its end (the same), 25 ms being the
        tolerance of the events summary's ``*_beyond_twenty_five_ms_fraction``
        values (``event_statistics.TIMING_TOLERANCE``). Each cell is the mean
        over animals of each animal's share of its pairs in it (the mean over
        its epochs of each epoch's share), as Figure 2c and d take their bars,
        with the lowest and highest animal's share under it in small type
        (``event_statistics.boundary_timing_statistics``; the events summary's
        ``timing_*_share_animal_*`` values). The four corner cells are named, as the event
        and its SWR overlap there with neither bound together: start and end
        earlier, the event leads; start earlier and end later, it contains the
        SWR; start later and end earlier, it is inside the SWR; start and end
        later, it lags.

    e — Supports Figure 2g and h, on population activity and the events:
        where the non-local time sits in MUA (the z-scored multiunit rate of
        the decoded tetrodes, ``MUA_zscore``), in Figure 2i's format. Top, each
        animal's immobile time bins decoded non-local spread over MUA (percent
        per 0.25 z), the share below 2 z written; bottom, the percent of
        immobile bins decoded non-local at each MUA level; a line per animal,
        their mean in bold, a dashed line at 2 z, the MUA event detector's
        threshold (``nonlocal_by_signal.MUA_LEVEL_Z``). Over the immobile time
        inside the well visits (``nonlocal_by_signal.animal_nonlocal_counts`` of
        the stored counts, the same table Figure 2i reads, by
        ``signal_share_panels.draw_share_by_signal``;
        ``Processed-Data/nonlocal_by_signal``; the events summary's
        ``nonlocal_share_mua_above_level*`` and
        ``nonlocal_time_below_mua_level_fraction*`` values). Most non-local
        time is below the MUA level, and nearly all time above the MUA level is
        decoded non-local. The panel does not show that non-local detection is
        independent of population bursts: the decoder needs spikes, so the two
        go together. It poses the question that the held-out tetrode panels
        (f, h) answer: whether the spikes point to the decoded place.

    f and h — Support Figure 2h, shared content: the events are driven by
    several tetrodes. Each tetrode is left out of the decoder in turn: the
    other tetrodes' estimate predicts its spikes better than local coding does
    (f); and the events have a spike on more tetrodes than Local rest of their
    length (h).

    f — The held-out tetrode gain over local coding. The immobile non-local
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
        so they are in the events summary and not drawn. The gain is drawn
        whole ("All bins") and split by the scored tetrode's own bins: those
        where it spiked ("Bins with a spike") and those where it was silent
        ("Silent bins"). The split answers whether the events' advantage over
        local coding comes from the spikes or from their absence (the ground
        process). A silent bin's term is the expected-firing term alone; a
        spiking bin's holds the spike's waveform term and the bin's
        expected-firing term, so the split is by bins, not an exact split of
        the likelihood into its mark and ground terms. For each event the two
        parts add to the whole; each drawn value is a median over events, so
        over the cohort they need not add up exactly. The events' spike rate
        against matched Local rest, which speaks to the same question, is a
        value for the text (``*_event_spike_rate_hz``, ``*_control_spike_rate_hz``)
        (``tetrode_holdout.holdout_statistics`` of the held-out tetrode run,
        ``scripts/event_evidence/score_tetrode_holdout.py``;
        ``Processed-Data/event_tetrode_holdout``; the events summary's
        ``*_heldout_gain_*``, ``*_heldout_spiking_gain_*`` and
        ``*_heldout_silent_gain_*`` values).

    g — Supports Figure 2's case that the no-SWR events are specific: how
        concentrated the decoded position is, no-SWR against SWR events. Each
        event's HPD coverage is the length of track (cm) that holds 95 % of
        the decoded probability, averaged over the event's bins
        (``hpd_coverage_cm`` of the canonical event table). Its distribution
        on a log axis, the percent of a class's events per bin
        (``event_statistics.HPD_BIN_EDGES_CM``), a thin line per animal and
        their mean in bold (``event_statistics.hpd_distribution``). Each class
        holds a tight group of events (near 20-25 cm) and a diffuse one spread
        over most of the track, so a median falls between them and describes
        neither; the events summary reports the share of each class below the
        dotted cut at 50 cm (``event_statistics.hpd_tight_share_statistics``;
        the ``hpd_tight_*`` values). The track's length is not drawn: it
        differs by animal and is known only from each decode's environment.
        The pooled class medians (``hpd_median_*``) are for the text.

    h — Tetrode participation (the share of tetrodes with a spike) of each
        event against a Local-rest window of its length lying wholly inside
        one well visit, one line per animal from its controls to its events,
        for no-SWR and SWR events (Local rest in the Local state's color, the
        events in their class's). Each group's controls have its events'
        lengths, so the two groups' levels are not compared: a longer window
        holds a spike on more tetrodes (``participation.participation_statistics``;
        ``Processed-Data/event_participation``, refused when it was not made
        over the analyses' subset from the current well visits; the events
        summary's participation values). Was Figure 2h.

    i — Supports Figure 2i, which shows no step in non-local time at the SWR
        threshold: at the SWR detector's threshold, does an event property
        jump (two kinds of events) or change smoothly with ripple power (one
        continuum)? The step test of the one-class-or-two analysis
        (``class_or_continuum``, ``scripts/event_properties/measure_ripple_class_structure.py``):
        per animal and property, ``y ~ s(criterion)`` against
        ``y ~ s(criterion) + (criterion >= 2 z)`` on the events within 1.5 z
        of the threshold (cubic B-spline, 8 knots; the Kay criterion, no
        covariate: ``event_statistics.main_step_fits``). Left, spike evidence
        per spike: each animal's jump as a share of its SWR minus no-SWR
        difference over all its events, with its 95 % interval, a filled dot
        when the animal has a step (the term's interval excludes 0 and it
        lowers the held-out error), against two dashed lines, 0 (no jump: one
        continuum) and 1 (the whole difference is a jump: two kinds). Right,
        the animals with a step in each property
        (``event_statistics.threshold_step_statistics``). The other properties
        are drawn as counts, not shares: where a property's SWR minus no-SWR
        difference is near zero (posterior peak, HPD coverage) the share is
        unstable. ``Processed-Data/ripple_class_structure/step_fits.csv``,
        refused when it was not made from the current event table and
        spike-evidence run; the events summary's ``step_*`` values.
        The dip test of the events' peak ripple power is a reported value for
        the text (``ripple_peak_*``), not drawn: on a stored simulation it
        rejects in none of the two-class samples planted at the observed
        separation, so it has no power here.

    j — Supports Figure 2j: its main result, the full decoder's gain over the
        SWR-only decoder outside SWRs, looked at more closely. Every value is
        over the immobile bins inside a well visit, which runs from the
        animal's first nose-poke at a reward well to its last poke-out there.
        Each is per epoch, summed over its bins and divided by the spikes in
        them (held-out log score, the whole decoder fit on folds of whole
        trials and each fold scored under the fit that did not see it); a dot
        per animal is the median over its epochs, beside the animals' mean
        with its 95 % interval. That gain over every immobile bin outside
        SWRs ("Outside SWRs",
        ``restriction_totals.outside_swr_gain_statistics``), then over only
        the bins farther than 100, 250 and 500 ms from every SWR
        (``restriction_totals.outside_swr_subset_statistics``, the
        ``beyond_swr_margin`` parts), a thin line through the means: the gain
        is not SWR edges. The same gain over only the bins whose place has at
        least 10 and at least 60 s of training data (the ``place_sampling``
        parts) is a reported value for the text, not drawn
        (``restriction_outside_swr_sampled_*``). Where that gain sits, its
        shares in the events detected from the held-out decode and outside
        them, is also for the text, not drawn
        (``restriction_outside_swr_full_minus_swr_share_*``): the shares follow
        from how the events are defined, so they describe where the gain sits
        and do not test the events.

    k — Supports Figure 2g, whose lines are the mean over animals of these
        lines: the same signals around the same events' starts (ripple
        power, MUA, theta power, head speed; ``triggered_analysis.TRIGGERED_SIGNALS``
        over the window ``TRIGGERED_WINDOW_S``), a thin line per animal and
        class (SWR, No SWR, in Figure 2g's colors), each the animal's median over
        its events (``triggered_analysis.animal_window_medians`` of
        ``triggered_analysis.triggered_windows``, the windows Figure 2g reduces;
        read from every analyzed epoch's decode frame in panel b's pass). The
        animals agree.

Panel f reads the held-out tetrode run over the epochs the events summary
reads and refuses a run not held out in folds of whole trials, as the summary
does; panel h reads the participation run over the same epochs. Panel j reads the held-out event run
(``Processed-Data/event_nonlocal_restriction_heldout_events``, its events and
time sums) by the events summary's reader
(``reported.events.read_heldout_restriction``) and refuses it, as Figure 2j
does, when the restriction run is not the fold fit on folds of whole trials or
was not made from the current decode frames and SWR source, when the held-out
event run was not made from the current restriction run or per-bin facts run,
or when the per-bin facts were not made from the current session cache. Its
values are the events summary's ``restriction_outside_swr_*`` values. Panels d,
g and i read the canonical event table (i with the stored output it names); a,
c and e read their stored outputs. Each output is refused, as the summary marks
its values pending, when it was not made from the current event table (c and
i), over the analyses' subset, or from the current well visits (a, c and e; e
also from the decodes the event features run records, as Figure 2i). Panel b reads every analyzed epoch's decode frame
(``Processed-Data``), which takes a few minutes; it has no reported value.

Usage
-----
    python scripts/figures/generate_supp_figure02.py
"""

import argparse
import logging
import string
import warnings
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, NullLocator

import continuum_swr_replay.visualization.figure_utilities as figure_utilities
from continuum_swr_replay.analysis.detection.nonlocal_events import get_event_types
from continuum_swr_replay.analysis.event_definition.analyzed_time import (
    read_dot_with_swr_source,
    read_swr_table,
)
from continuum_swr_replay.analysis.event_definition.cohort_selection import (
    select_events_in,
    select_visit_context,
    subset_staleness,
)
from continuum_swr_replay.analysis.event_definition.no_spike_periods import (
    level_outside_periods,
    no_spike_periods,
)
from continuum_swr_replay.analysis.event_definition.swr_source import (
    event_table_epochs,
)
from continuum_swr_replay.analysis.event_evidence.participation import (
    participation_statistics,
)
from continuum_swr_replay.analysis.event_evidence.restriction_totals import (
    OUTSIDE_SWR_MARGINS_S,
    RESTRICTION_FIGURE_FAMILY,
    RESTRICTION_FIGURE_OUTSIDE_SWR,
    outside_swr_statistics,
    outside_swr_subset_statistics,
)
from continuum_swr_replay.analysis.event_evidence.spike_evidence import (
    SPIKE_EVIDENCE_METRICS,
)
from continuum_swr_replay.analysis.event_evidence.tetrode_holdout import (
    holdout_statistics,
    window_scores,
)
from continuum_swr_replay.analysis.event_properties.event_statistics import (
    HPD_BIN_EDGES_CM,
    HPD_TIGHT_CM,
    TIMING_CLASSES,
    boundary_offsets,
    boundary_timing_statistics,
    hpd_distribution,
    main_step_fits,
    nonlocal_below_threshold_statistics,
    threshold_step_statistics,
)
from continuum_swr_replay.analysis.event_properties.nonlocal_by_signal import (
    MUA_LEVEL_Z,
    animal_nonlocal_counts,
)
from continuum_swr_replay.analysis.event_properties.rule_sweep import (
    RULE,
    sweep_statistics,
)
from continuum_swr_replay.analysis.event_properties.triggered_analysis import (
    TRIGGERED_SIGNALS,
    animal_balanced_quartiles,
    animal_window_medians,
    get_signal_windows,
    triggered_windows,
    triggering_events_in,
)
from continuum_swr_replay.analysis.event_tables.prepare_data import (
    time_mask_staleness,
    well_visit_times,
)
from continuum_swr_replay.analysis.shared_statistics import animal_medians
from continuum_swr_replay.data_loaders.epoch_register import (
    analyzed_animals,
    analyzed_decode_files,
    analyzed_epoch_ids,
    require_analyzed_epochs,
)
from continuum_swr_replay.decoding.files import data_over_time_suffix
from continuum_swr_replay.parameters import DECODE_SAMPLE_S
from continuum_swr_replay.paths import (
    PARTICIPATION_NAME,
    TETRODE_HOLDOUT_NAME,
    event_spike_evidence_dir,
    processed_data_dir,
)
from continuum_swr_replay.reported.events import (
    OTHER_STEP_PROPERTIES,
    STEP_KNOTS,
    STEP_WINDOW_Z,
    checked_reference,
    default_inputs,
    event_table_staleness,
    heldout_restriction_refusal,
    read_heldout_restriction,
    require_tetrode_holdout_folds,
    restriction_inputs_note,
    spike_evidence_staleness,
)
from continuum_swr_replay.reported.formatting import format_number
from continuum_swr_replay.reported.values import rel
from continuum_swr_replay.runs.io import files_sha256, read_complete_run, sha256_file
from continuum_swr_replay.runs.staleness import read_output_manifest
from continuum_swr_replay.visualization.figure_utilities import save_figure
from continuum_swr_replay.visualization.heldout_gain_panels import (
    GROUP_COLORS,
    REFERENCE_COLOR,
    SPLIT_LABELS,
    animal_values,
    draw_full_minus_swr_axis,
    draw_split_gain,
    label_gain_axis,
)
from continuum_swr_replay.visualization.palettes import COLORS, readable_text_color
from continuum_swr_replay.visualization.signal_share_panels import draw_share_by_signal
from continuum_swr_replay.visualization.simulated_run_panels import (
    SMALL_SIZE,
    STATE_COLORS,
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

# Inches: the full page width, five rows (the module's docstring).
WIDTH = 6.5
HEIGHT = 7.65

TITLE = "Held-out tetrode"
YLABEL = "Held-out gain\nover local (nats/spike)"
TIMING_LABELS = ("Earlier", "Within\n25 ms", "Later")
"""The timing table's tick labels for :data:`event_statistics.TIMING_CLASSES`."""
TIMING_CORNERS = {
    ("earlier", "earlier"): "event leads",
    ("earlier", "later"): "contains SWR",
    ("later", "earlier"): "inside SWR",
    ("later", "later"): "event lags",
}
"""The timing table's names of its corner cells, by (start, end) class: the event and its
SWR overlap there, and neither bound is within 25 ms of the SWR's."""
CORNER_DROP = -0.3
SHARE_DROP = 0.0
RANGE_DROP = 0.28
"""Where a cell's corner name, share and animals' range sit, from its center
(cell heights, down)."""
TIMING_CMAP = "bone_r"
"""The timing table's colormap: sequential, the posteriors' (prints in gray)."""
STEP_PROPERTIES = {
    "evidence": "Spike evidence",
    **{key: name[0].upper() + name[1:] for key, name in OTHER_STEP_PROPERTIES.items()},
}
"""The step panel's properties, top to bottom, by their word in
``step_fits.csv``, with their names."""
STEP_REFERENCES = {0.0: "one continuum", 1.0: "two kinds"}
"""The step panel's reference shares and what each would mean."""
CALIBRATION_COLORS = {
    "Local": STATE_COLORS["Local"],
    "Non-Local Continuous": STATE_COLORS["Continuous"],
    "Non-Local Fragmented": STATE_COLORS["Fragmented"],
}
"""The calibration panel's states (``state_evidence.IN_VISIT_CALIBRATION_METRICS``),
in the decoder figures' state colors."""
CALIBRATION_YLABEL = (
    "Evidence for the state's place\nover a mismatched place\n(nats/spike)"
)
"""The calibration panel's y label."""
CALIBRATION_DODGE = 0.025
"""How far apart the states' points sit at each probability bin (x units)."""
SWEEP_XLABELS = {
    "probability_threshold": "Non-local\nprobability",
    "minimum_samples": "Minimum\nduration (ms)",
    "swr_threshold_z": "SWR\nthreshold (z)",
}
"""The sweep panel's x label of each swept setting."""
SWEEP_SCALES = {
    "probability_threshold": 1.0,
    "minimum_samples": 1000 * DECODE_SAMPLE_S,
    "swr_threshold_z": 1.0,
}
"""What each setting's value is multiplied by on the sweep panel's x axis: the
minimum run length in samples is drawn as the shortest event it keeps, in ms
(n samples of 2 ms; the rule's 7 samples, its 15 ms, is a shortest event of
14 ms)."""
SWEEP_TICKS = {
    "probability_threshold": (0.5, 0.8, 0.95),
    "minimum_samples": (14, 50, 100),
    "swr_threshold_z": (0, 2, 4, 6),
}
"""The sweep panel's x ticks, in its drawn units: the value used and the ends."""
HPD_XLABEL = "95 % HPD length (cm),\nmean over the event's bins"
"""The posterior-spread panel's x label: each event's mean over its bins of the
length of track that holds 95 % of the decoded probability."""
HPD_YLABEL = "% of events"
HPD_TICKS_CM = (10, 25, 50, 100, 200, 400)
REST_EVENT_X = {"no SWR": (0.0, 1.0), "SWR": (2.4, 3.4)}
"""The participation panel: each group's Local-rest and event positions on the
x axis."""
PAIR_COLOR = "#999999"
"""A line joining one animal's two values (the participation panel)."""

DISTANCE_LABEL = "at least this far from\nany SWR (ms)"
OUTSIDE_LABEL = "Outside\nSWRs"
PARTS_LABEL_DROP = 21
"""How far below the axis a series' parts label starts (points): under the two
lines of "Outside SWRs", so that it can be wider than its columns."""
SERIES_START = 1.3
"""The x position of a series' first part, after the "Outside SWRs" column at 0;
the parts are 1 apart."""
LAYOUT = (
    ((0.855, 0.955), (("state_calibration", 0.12, 0.4), ("no_spike_mua", 0.71, 0.27))),
    ((0.635, 0.75), (("rule_sweep", 0.085, 0.47), ("boundary_timing", 0.71, 0.27))),
    (
        (0.415, 0.53),
        (
            ("nonlocal_mua", 0.075, 0.13),
            ("heldout_gain", 0.305, 0.19),
            ("hpd", 0.56, 0.16),
            ("participation", 0.835, 0.155),
        ),
    ),
    ((0.2, 0.31), (("step_test", 0.085, 0.42), ("outside_distance", 0.66, 0.32))),
    ((0.04, 0.105), (("event_traces", 0.085, 0.6),)),
)
"""The figure's rows, top to bottom: each row's frames' bottom and top (figure
height), then its panels left to right, each with its frame's left edge and
width (figure width). Read in this order the panels are in the order the text
cites them, so their letters follow from it."""
PANELS = tuple(panel for _, row in LAYOUT for panel, _, _ in row)
"""The panels in the order the text cites them."""
LETTERS = dict(zip(PANELS, string.ascii_lowercase, strict=False))
ROWS = tuple((span, tuple(panel for panel, _, _ in row)) for span, row in LAYOUT)
"""Each row's frames' bottom and top and its panels, left to right."""
FRAMES = {panel: (left, width) for _, row in LAYOUT for panel, left, width in row}
"""Each panel's frame's left edge and width (figure width)."""
LETTER_HEIGHT = 0.035
"""A panel letter's height (figure height): the letters' tops sit this far
above the highest drawn top of their row."""
NO_SPIKE_SIGNAL = "MUA_zscore"
TRIGGER_WINDOW_S = 0.3
"""The No-Spike panel's window around each start (s, in all), Figure 2g's: the periods
last a few tens of ms."""
NO_SPIKE_COLOR = COLORS["No-Spike"]
NO_SPIKE_WORKERS = 4
"""The processes that read the decode frames for the No-Spike panel by
default: each holds one frame at a time."""
GROUP_NAMES = ("No SWR", "SWR")
"""The groups' names on the panels, in ``GROUP_COLORS``' order."""
REFUSED = f"Figure 2's supplement: panel {LETTERS['outside_distance']} is not drawn"


def read_inputs(workers: int = NO_SPIKE_WORKERS) -> dict:
    """The figure's statistics, as the events summary reads them (the No-Spike
    trace's decode frames read by ``workers`` processes).

    Returns
    -------
    dict
        In panel order: ``calibration`` (a): :func:`read_state_calibration`;
        ``no_spike`` (b) and ``event_traces`` (k): :func:`read_frame_traces`; ``sweep`` (c):
        :func:`read_rule_sweep`; ``timing`` (d): the selected events'
        ``boundary_timing_statistics``; ``mua`` (e): :func:`read_mua_counts`;
        ``tetrode`` (f): the held-out tetrode statistics of the events that
        start inside a well visit, by column of ``SPLIT_LABELS``, each indexed
        by group (a held-out tetrode run not held out in folds of whole trials
        is refused, ``reported.events.require_tetrode_holdout_folds``); ``hpd``
        (g): the selected events' ``hpd_distribution``; ``participation`` (h):
        :func:`read_participation`;
        ``step_test`` (i): :func:`read_step_test`; ``outside`` (j):
        ``outside_swr_statistics`` over the held-out event run, indexed by
        family and measure; ``parts`` (j): ``outside_swr_subset_statistics`` of
        the margins over its time sums, indexed by subset, threshold, family
        and measure.

    Raises
    ------
    ValueError
        If the held-out event run, or what it was made from, is not current
        (``reported.events.HELDOUT_RESTRICTION_RERUN`` names what to make again).
    """
    inputs = default_inputs()
    require_tetrode_holdout_folds(
        read_output_manifest(inputs.tetrode_holdout), rel(inputs.tetrode_holdout)
    )
    table = read_swr_table(inputs.event_table)
    epochs = event_table_epochs(table, analyzed_epoch_ids())
    scores = read_complete_run(inputs.tetrode_holdout, TETRODE_HOLDOUT_NAME, epochs)
    events = select_visit_context(table)
    selected = select_events_in(scores, events["event_id"])
    heldout = read_heldout_restriction(
        inputs.nonlocal_restriction_heldout_events,
        inputs.nonlocal_restriction,
        inputs.bin_facts,
        epochs,
        sha256_file(inputs.event_table),
    )
    refusal = heldout_restriction_refusal(
        heldout, restriction_inputs_note(inputs.nonlocal_restriction, epochs)
    )
    if refusal is not None:
        cause, rerun = refusal
        raise ValueError(f"{REFUSED}: {cause}. Rerun {rerun}.")
    windows = window_scores(selected)
    return {
        "calibration": read_state_calibration(inputs.state_calibration_in_visits),
        "sweep": read_rule_sweep(inputs.event_rule_sweep, inputs.event_table),
        "mua": read_mua_counts(inputs.nonlocal_by_signal),
        "step_test": read_step_test(inputs.class_structure, inputs.event_table),
        "hpd": hpd_distribution(events),
        "timing": boundary_timing_statistics(boundary_offsets(events), events),
        "tetrode": {
            column: holdout_statistics(windows, "event", column)
            for column in SPLIT_LABELS
        },
        "participation": read_participation(
            inputs.event_participation, epochs, events["event_id"]
        ),
        "outside": outside_swr_statistics(
            heldout.time_sums, heldout.bin_s, heldout.events
        ),
        "parts": outside_swr_subset_statistics(
            heldout.time_sums, heldout.bin_s, subsets=("beyond_swr_margin",)
        ).set_index(["subset", "threshold_s", "family", "measure"]),
        **read_frame_traces(events, workers),
    }


def _refuse_stale(
    folder: Path,
    panel: str,
    rerun: str,
    *,
    event_table: Path | None = None,
    epoch_ids: Iterable[str] | None = None,
) -> None:
    """Refuse an output the events summary marks pending: one not made from the
    current event table (when ``event_table`` is given), not over the analyses'
    subset, or, for a time-masked output (``epoch_ids`` given), not from the
    current well visits (``prepare_data.time_mask_staleness``)."""
    manifest = read_output_manifest(folder)
    what = rel(folder)
    for reason in (
        None
        if event_table is None
        else event_table_staleness(manifest, sha256_file(event_table), what),
        subset_staleness(manifest, what)
        if epoch_ids is None
        else time_mask_staleness(manifest, epoch_ids, what),
    ):
        if reason is not None:
            raise ValueError(
                f"Figure 2's supplement: panel {LETTERS[panel]} is not drawn: "
                f"{reason}. Rerun {rerun}."
            )


def read_participation(
    folder: Path, epochs: Sequence[str], event_ids: pd.Series
) -> pd.DataFrame:
    """The participation panel's statistics: ``participation_statistics`` of the
    participation run over ``epochs``, read over the selected events
    (``event_ids``), as the events summary reads it.

    Raises
    ------
    ValueError
        If the run's settings do not record the analyses' subset or it did not
        read the current session cache (its controls are Local rest inside the
        current visits only then).
    """
    manifest = read_output_manifest(folder)
    note = time_mask_staleness(
        {**manifest, "subset": manifest.get("settings", {}).get("subset")},
        epochs,
        rel(folder),
    )
    if note is not None:
        raise ValueError(
            f"Figure 2's supplement: panel {LETTERS['participation']} is not drawn: "
            f"{note}. Rerun scripts/event_evidence/measure_event_participation.py."
        )
    return participation_statistics(
        select_events_in(
            read_complete_run(folder, PARTICIPATION_NAME, epochs), event_ids
        )
    )


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
        analyzed epochs (``reported.events.checked_reference``, as Figure 2i).
    """
    counts = pd.read_parquet(folder / "counts.parquet")
    try:
        checked_reference(counts, folder, "counts by signal")
    except ValueError as error:
        raise ValueError(
            f"Figure 2's supplement: panel {LETTERS['nonlocal_mua']} is not drawn: "
            f"{error}. Rerun scripts/event_properties/measure_nonlocal_by_signal.py."
        ) from error
    return animal_nonlocal_counts(counts, "mua")


def read_step_test(folder: Path, event_table: Path) -> dict:
    """The step panel's statistics from the one-class-or-two output, as the
    events summary reads them: the main setting's fits
    (``event_statistics.main_step_fits``) and the animals with a step per
    property (``event_statistics.threshold_step_statistics``).

    Returns
    -------
    dict
        ``fits`` (one row per property and animal) and ``counts`` (indexed by
        property).

    Raises
    ------
    ValueError
        If the output was not made from the current event table over the
        analyses' subset, or from the current spike-evidence run.
    """
    _refuse_stale(
        folder, "step_test",
        "scripts/event_properties/measure_ripple_class_structure.py analyze",
        event_table=event_table,
    )  # fmt: skip
    evidence = sorted(event_spike_evidence_dir().glob(f"*/{SPIKE_EVIDENCE_METRICS}"))
    stale = spike_evidence_staleness(
        read_output_manifest(folder), files_sha256(evidence), rel(folder)
    )
    if stale is not None:
        raise ValueError(
            f"Figure 2's supplement: panel {LETTERS['step_test']} is not drawn: "
            f"{stale}. Rerun scripts/event_properties/measure_ripple_class_structure.py."
        )
    steps = pd.read_csv(folder / "step_fits.csv")
    n_animals = len(analyzed_animals())
    return {
        "fits": main_step_fits(steps, STEP_KNOTS, STEP_WINDOW_Z, n_animals),
        "counts": threshold_step_statistics(
            steps, STEP_KNOTS, STEP_WINDOW_Z, n_animals
        ),
        "n_animals": n_animals,
    }


def _frame_epoch(path: Path, keys: pd.DataFrame | None = None) -> dict:
    """One epoch's part of :func:`read_frame_traces`: its MUA level outside
    its No-Spike periods, and, when it has periods, their windows
    ``(n_periods, n_lags)``, their lags, its animal and the periods'
    durations; and, when ``keys`` (the selected events' ``animal``,
    ``session_date``, ``epoch`` and ``non_local_event``) are given, the
    windows of Figure 2g's signals around its selected events' starts
    (``triggered_analysis.triggered_windows``, as Figure 2g reads them)."""
    frame = read_dot_with_swr_source(path)
    arrival, departure = well_visit_times(
        path.name.removesuffix(data_over_time_suffix())
    )
    periods = no_spike_periods(frame, arrival, departure)
    session = frame[["animal", "day", "epoch"]].iloc[0]
    epoch = {
        "level": level_outside_periods(
            frame, periods, arrival, departure, NO_SPIKE_SIGNAL
        ),
        "animal": session["animal"],
    }
    if keys is not None:
        epoch["event_windows"] = triggered_windows(
            frame,
            triggering_events_in(get_event_types(frame), keys),
            list(TRIGGERED_SIGNALS),
            window_sec=TRIGGER_WINDOW_S,
        )
    if periods.empty:
        return epoch
    windows = get_signal_windows(
        frame, periods.assign(**session.to_dict()), [NO_SPIKE_SIGNAL],
        window_sec=TRIGGER_WINDOW_S,
    )  # fmt: skip
    return {
        **epoch,
        "windows": windows.sel(signal=NO_SPIKE_SIGNAL).values,
        "time": windows["time"].values,
        "durations": periods["duration"].to_numpy(),
    }


def read_frame_traces(
    events: pd.DataFrame | None = None, workers: int = NO_SPIKE_WORKERS
) -> dict:
    """The traces read from every analyzed epoch's decode frame, in one pass
    (the SWR source's reader), by ``workers`` processes (1 reads them in this
    one), in the epochs' order.

    Parameters
    ----------
    events : pd.DataFrame, optional
        The selected events (``animal``, ``session_date``, ``epoch``,
        ``non_local_event``); when None, only the No-Spike trace is read.
    workers : int, optional

    Returns
    -------
    dict
        ``no_spike``, the No-Spike panel's trace: ``time`` (s from the
        start), ``q25``, ``median`` and ``q75``, animal-balanced over the
        periods with a window (``triggered_analysis.animal_balanced_quartiles``);
        ``level``: the mean over animals of each animal's median over its
        epochs of each epoch's MUA level outside its periods; ``periods``: the
        periods with a window, by animal; ``median_duration``: their median
        duration (s). ``event_traces`` (None without ``events``): by SWR class
        (``"overlap"``, ``"no_overlap"``), each signal's per-animal median
        windows (``triggered_analysis.animal_window_medians``), with ``time``.
    """
    paths = analyzed_decode_files(
        processed_data_dir(), data_over_time_suffix(), require_all=True
    )
    keys = (
        None
        if events is None
        else events[["animal", "session_date", "epoch", "non_local_event"]]
    )
    read = partial(_frame_epoch, keys=keys)
    if workers == 1:
        epochs = list(map(read, paths))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            epochs = list(pool.map(read, paths))
    with_periods = [epoch for epoch in epochs if "windows" in epoch]
    counts: dict = {}
    for epoch in with_periods:
        counts[epoch["animal"]] = counts.get(epoch["animal"], 0) + len(epoch["windows"])
    levels = pd.Series([epoch["level"] for epoch in epochs])
    no_spike = {
        "time": with_periods[-1]["time"],
        **animal_balanced_quartiles(
            np.concatenate([e["windows"] for e in with_periods]),
            np.concatenate([[e["animal"]] * len(e["windows"]) for e in with_periods]),
        ),
        "level": float(
            animal_medians(levels, pd.Series([e["animal"] for e in epochs])).mean()
        ),
        "periods": pd.Series(counts, name="periods"),
        "median_duration": float(
            np.median(np.concatenate([e["durations"] for e in with_periods]))
        ),
    }
    if events is None:
        return {"no_spike": no_spike, "event_traces": None}
    traces: dict = {}
    for kind in ("overlap", "no_overlap"):
        windows = xr.concat(
            [epoch["event_windows"][kind] for epoch in epochs], dim="event"
        )
        traces[kind] = {
            signal: animal_window_medians(windows, signal)
            for signal in TRIGGERED_SIGNALS
        }
        traces["time"] = windows["time"].values
    return {"no_spike": no_spike, "event_traces": traces}


def draw_boundary_timing(ax: Axes, timing: pd.DataFrame) -> None:
    """The timing table: the share of SWR-overlapping events in each combination of
    start (rows) and end (columns) against the matched SWR's, as a heat map
    with the mean over animals' percent in each cell, the animals' range (in
    percent) under it
    and the corner cells' names (:data:`TIMING_CORNERS`) above it, in black
    or white, whichever reads against the cell (``timing``: :func:`read_inputs`'
    ``timing``)."""
    n = len(TIMING_CLASSES)
    shares = timing["share_animal_mean"].to_numpy().reshape(n, n)
    cmap = plt.get_cmap(TIMING_CMAP)
    ax.imshow(100 * shares, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for (row, column), cell in zip(np.ndindex(n, n), timing.itertuples(), strict=True):
        red, green, blue, _ = cmap(cell.share_animal_mean)
        color = "white" if 0.299 * red + 0.587 * green + 0.114 * blue < 0.5 else "black"
        if cell.Index in TIMING_CORNERS:
            ax.text(
                column, row + CORNER_DROP, TIMING_CORNERS[cell.Index],
                ha="center", va="center", fontsize=SMALL_SIZE, color=color,
                style="italic",
            )  # fmt: skip
        ax.text(
            column, row + SHARE_DROP, f"{100 * cell.share_animal_mean:.0f}%",
            ha="center",
            va="center", fontsize=SMALL_SIZE, color=color,
        )  # fmt: skip
        ax.text(
            column, row + RANGE_DROP,
            f"{100 * cell.animal_min:.0f}\u2013{100 * cell.animal_max:.0f}",
            ha="center", va="center", fontsize=SMALL_SIZE, color=color,
        )  # fmt: skip
    ax.set_xticks(range(n), TIMING_LABELS)
    ax.set_yticks(range(n), TIMING_LABELS)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xlabel("Event ends, vs SWR end")
    ax.set_ylabel("Event starts,\nvs SWR start")
    ax.set_title("Timing against the SWR", fontsize=SMALL_SIZE, pad=3)


def draw_heldout_tetrode(ax: Axes, statistics: dict) -> None:
    """The held-out tetrode gain over local coding per spike, by group, over
    all bins, the bins with a spike and the silent bins (``statistics``:
    :func:`read_inputs`' ``tetrode``)."""
    draw_split_gain(ax, statistics, TITLE)
    rows = [table.loc[group] for table in statistics.values() for group in table.index]
    low = min(min(float(np.min(r["animals"])), r["ci_low"]) for r in rows)
    high = max(max(float(np.max(r["animals"])), r["ci_high"]) for r in rows)
    label_gain_axis(ax, YLABEL, floor=min(-0.12, low - 0.08 * (high - low)))
    ax.set_ylim(None, high + 0.15 * (high - low))  # room for the groups' names


def draw_participation(ax: Axes, statistics: pd.DataFrame) -> None:
    """The participation panel: tetrode participation, Local rest against
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
            ax.plot(x, (rest[animal], event[animal]), color=PAIR_COLOR, lw=0.5,
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
    ax.set_xticks(
        ticks, ["Local rest", "No SWR", "Local rest", "SWR"], rotation=45, ha="right"
    )
    ax.set_title("Participation", fontsize=SMALL_SIZE, pad=3)


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
    ax.set_xticks(xs, [OUTSIDE_LABEL, *names])
    # the parts' label, centered under their tick labels
    ax.annotate(
        DISTANCE_LABEL, (float(np.mean(xs[1:])), 0), xycoords=("data", "axes fraction"),
        xytext=(0, -PARTS_LABEL_DROP), textcoords="offset points", ha="center",
        va="top",
        fontsize=SMALL_SIZE, annotation_clip=False,
    )  # fmt: skip


def draw_step_test(ax_share: Axes, ax_count: Axes, statistics: dict) -> None:
    """The step panel (``statistics``: :func:`read_step_test`). Left, spike
    evidence: each animal's jump at the threshold as a share of its SWR minus
    no-SWR difference with its 95 % interval, filled when the animal has a
    step, open when not, against dashed lines at 0 (one continuum) and 1 (two
    kinds). Right, the animals with a step in each property, as bars."""
    fits = statistics["fits"]
    evidence = fits[fits["property"] == "evidence"].reset_index(drop=True)
    x = np.arange(len(evidence))
    for k, row in evidence.iterrows():
        ax_share.errorbar(
            x[k], row["step_share_overall"],
            yerr=[[row["step_share_overall"] - row["share_overall_ci_low"]],
                  [row["share_overall_ci_high"] - row["step_share_overall"]]],
            fmt="o", ms=3.5, lw=0.8, capsize=0, color="black",
            mfc="black" if row["has_step"] else "white",
        )  # fmt: skip
    for level, meaning in STEP_REFERENCES.items():
        ax_share.axhline(level, color=REFERENCE_COLOR, lw=0.6, ls="--", zorder=1)
        # past the frame's right edge, clear of the animals' intervals
        ax_share.text(
            1.03, level, meaning, transform=ax_share.get_yaxis_transform(),
            ha="left", va="center", fontsize=SMALL_SIZE, color=REFERENCE_COLOR,
        )  # fmt: skip
    ax_share.set_xlim(-0.6, x[-1] + 0.6)
    ax_share.set_xticks([])
    ax_share.set_xlabel("Animals")
    ax_share.set_ylim(-0.3, 1.2)
    ax_share.set_yticks([0, 0.5, 1])
    ax_share.set_ylabel("Jump at threshold\n(share of SWR −\nno-SWR difference)")
    ax_share.set_title("Spike evidence", fontsize=SMALL_SIZE, pad=3)
    counts = statistics["counts"]["n_animals_with_step"]
    y = np.arange(len(STEP_PROPERTIES))[::-1]
    ax_count.barh(
        y, [counts[p] for p in STEP_PROPERTIES], height=0.65, color="#999999", lw=0
    )
    ax_count.set_yticks(y, list(STEP_PROPERTIES.values()))
    ax_count.tick_params(axis="y", length=0)
    ax_count.set_xlim(0, statistics["n_animals"])
    ax_count.set_xticks(range(statistics["n_animals"] + 1))
    ax_count.set_ylim(-0.6, len(STEP_PROPERTIES) - 0.4)
    ax_count.set_xlabel("Animals with a step")
    ax_count.set_title("Step at the SWR threshold", fontsize=SMALL_SIZE, pad=3)


def draw_nonlocal_mua(ax_spread: Axes, ax_share: Axes, counts: pd.DataFrame) -> None:
    """The MUA panel, drawn as Figure 2i draws ripple power
    (``signal_share_panels.draw_share_by_signal``): where the non-local time
    sits in MUA (top) and the share of immobile bins decoded non-local at each
    MUA level (bottom), with the MUA level of the reported values and the
    share of non-local time below it (``nonlocal_below_threshold_statistics``,
    the events summary's value; ``counts``: :func:`read_inputs`' ``mua``)."""
    below = nonlocal_below_threshold_statistics(counts, MUA_LEVEL_Z)
    draw_share_by_signal(
        ax_spread, ax_share, counts, threshold=MUA_LEVEL_Z,
        threshold_label=f"{MUA_LEVEL_Z:g} z", threshold_color=REFERENCE_COLOR,
        shade_color=COLORS["All Events"], xlabel="MUA (z)",
        below_note=f"{format_number(below['fraction_animal_mean'], 'pctnum')}% "
        f"below\n{MUA_LEVEL_Z:g} z",
    )  # fmt: skip


def draw_state_calibration(ax: Axes, bins: pd.DataFrame) -> None:
    """The calibration panel: for Local, Continuous and Fragmented, on one
    axis, the held-out evidence per spike for the state's place over a
    mismatched place against the state's probability, the animals' mean with
    its 95 % interval at each probability bin's center (the states a little
    apart so that the intervals show), each state named at its line's end in
    its color, and a dashed line at zero (``bins``: :func:`read_inputs`'
    ``calibration``)."""
    centers = np.arange(0.1, 1.0, 0.2)
    for k, (state, color) in enumerate(CALIBRATION_COLORS.items()):
        mine = bins[bins["state"] == state]
        estimate = mine["animal_estimate"].to_numpy()
        x = centers + (k - 1) * CALIBRATION_DODGE
        ax.errorbar(
            x, estimate,
            yerr=[estimate - mine["animal_ci_lo"], mine["animal_ci_hi"] - estimate],
            color=color, marker="o", ms=2.5, lw=1.0, capsize=0,
        )  # fmt: skip
        ax.annotate(
            state.removeprefix("Non-Local "), (1.0, estimate[-1]),
            xytext=(2, 0), textcoords="offset points", ha="left", va="center",
            fontsize=SMALL_SIZE, color=readable_text_color(color),
            annotation_clip=False,
        )  # fmt: skip
    ax.plot([0, 1], [0, 0], color=REFERENCE_COLOR, lw=0.6, ls="--", zorder=1)
    ax.set_xlim(0, 1.3)  # room for the states' names inside the axis
    ax.spines["bottom"].set_bounds(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xlabel("State probability")
    ax.set_ylabel(CALIBRATION_YLABEL)
    ax.set_title("Decoder states, held out", fontsize=SMALL_SIZE, pad=3)


def draw_rule_sweep(axes: Sequence[Axes], sweep: pd.DataFrame) -> None:
    """The sweep panel: on the top row, the share of events with no SWR at each
    value of each swept setting (one axis each, sharing the y axis), a thin
    line per animal and their mean over animals in bold with its dots, and on every
    axis the value the paper uses dotted and marked "used"; on the bottom row,
    under each, the events detected at each value (thousands, summed over the
    epochs; flat on the SWR threshold's axis, which relabels the rule's
    events). The first two columns are settings of the event rule, under one
    heading; the third is the SWR detector's threshold (``sweep``:
    :func:`read_inputs`' ``sweep``)."""
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


TRACE_LABELS = dict(
    zip(
        TRIGGERED_SIGNALS,
        ("Ripple (z)", "MUA (z)", "Theta (z)", "Speed (cm/s)"),
        strict=True,
    )
)
"""The traces panel's name of each of Figure 2g's signals, as Figure 2g names
them."""
TRACE_CLASSES = {
    "overlap": ("SWR", COLORS["SWR"]),
    "no_overlap": ("No SWR", COLORS["No SWR"]),
}
"""The traces panel's classes, as Figure 2g draws them."""


def draw_event_traces(axes: Sequence[Axes], traces: dict) -> None:
    """The traces panel: Figure 2g's signals around event starts, one axis per
    signal, a thin line per animal and class (each animal's median over its
    events, ``traces``: :func:`read_frame_traces`' ``event_traces``), the
    classes in Figure 2g's colors and named past the last axis, a dotted line
    at the start."""
    t_ms = 1000 * np.asarray(traces["time"])
    for ax, signal in zip(axes, TRIGGERED_SIGNALS, strict=True):
        for kind, (_, color) in TRACE_CLASSES.items():
            ax.plot(t_ms, traces[kind][signal].to_numpy(), color=color, lw=0.5)
        ax.axvline(0, color="k", lw=0.5, ls=":")
        ax.set_xlim(t_ms[0], t_ms[-1])
        ax.set_xticks([-100, 0, 100])
        ax.set_title(TRACE_LABELS[signal], fontsize=SMALL_SIZE, pad=3)
        ax.yaxis.set_major_locator(MaxNLocator(3))
    # the classes named past the last axis, clear of the lines
    for k, (name, color) in enumerate(TRACE_CLASSES.values()):
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
    animal and their mean in bold in the class's color, the classes named in
    their colors, and a dotted line at the tight cut
    (``distribution``: :func:`read_inputs`' ``hpd``)."""
    for group, color in GROUP_COLORS.items():
        mine = distribution[distribution["group"] == group]
        by_animal = mine.pivot(index="bin_low", columns="animal", values="percent")
        high = mine.groupby("bin_low")["bin_high"].first()
        center = np.sqrt(by_animal.index.to_numpy() * high.to_numpy())
        ax.plot(center, by_animal.to_numpy(), color=color, lw=0.5, alpha=0.5)
        ax.plot(center, by_animal.mean(axis=1).to_numpy(), color=color, lw=1.2)
    ax.axvline(HPD_TIGHT_CM, color=REFERENCE_COLOR, lw=0.6, ls=":", zorder=1)
    ax.text(
        HPD_TIGHT_CM / 1.08, 0.98, f"{HPD_TIGHT_CM:g} cm",
        transform=ax.get_xaxis_transform(), ha="right", va="top",
        fontsize=SMALL_SIZE, color=REFERENCE_COLOR,
    )  # fmt: skip
    for k, (name, color) in enumerate(
        zip(GROUP_NAMES, GROUP_COLORS.values(), strict=True)
    ):
        ax.text(
            0.98, 0.98 - 0.1 * k, name, transform=ax.transAxes, ha="right",
            va="top", fontsize=SMALL_SIZE, color=readable_text_color(color),
        )  # fmt: skip
    ax.set_xscale("log")
    ax.set_xlim(HPD_BIN_EDGES_CM[0], HPD_BIN_EDGES_CM[-1])
    ax.set_xticks(HPD_TICKS_CM, [f"{t:g}" for t in HPD_TICKS_CM])
    ax.xaxis.set_minor_locator(NullLocator())
    top = distribution.groupby(["group", "bin_low"])["percent"].max().max()
    ax.set_ylim(0, 1.35 * top)  # room for the names above the lines
    ax.set_xlabel(HPD_XLABEL)
    ax.set_ylabel(HPD_YLABEL)
    ax.set_title("Posterior spread", fontsize=SMALL_SIZE, pad=3)


def draw_no_spike_mua(ax: Axes, trace: dict) -> None:
    """The No-Spike panel: MUA around No-Spike starts (``trace``: :func:`read_frame_traces`' ``no_spike``),
    the animal-balanced median with its band as Figure 2g draws its rows, a dotted
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


PANEL_GRIDS = {
    "nonlocal_mua": (2, 1),
    "rule_sweep": (2, 3),
    "step_test": (1, 2),
    "event_traces": (1, len(TRIGGERED_SIGNALS)),
}
"""The panels drawn on several axes in their frame: rows and columns."""
PANEL_ROW_SHARES = {"rule_sweep": (2.2, 1)}
"""The heights of a stacked panel's rows, as shares (equal when not named)."""
GRID_GAP = (0.1, 0.1)
"""The gap between a panel's axes, as a share of its frame's height and width."""
PANEL_COLUMN_GAPS = {"step_test": 0.6, "event_traces": 0.12}
"""A panel's gap between its axes' columns, as a share of its frame's width,
where :data:`GRID_GAP`'s is too small (the step panel's property names)."""


def _n_axes(panel: str) -> int:
    rows, columns = PANEL_GRIDS.get(panel, (1, 1))
    return rows * columns


def panel_axes(fig: Figure, panel: str) -> list[Axes]:
    """A panel's axes of a figure :func:`generate_figure` made, row by row
    (several for a panel of :data:`PANEL_GRIDS`)."""
    start = sum(_n_axes(p) for p in PANELS[: PANELS.index(panel)])
    return fig.axes[start : start + _n_axes(panel)]


def generate_figure(statistics: dict) -> Figure:
    """Build the figure from :func:`read_inputs`, the panels in :data:`PANELS`'
    order and laid out by :data:`LAYOUT`."""
    spans = {panel: span for span, row in ROWS for panel in row}
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    axes = {}
    for panel in PANELS:  # the axes in the panels' order (panel_axes)
        (bottom, top), (left, width) = spans[panel], FRAMES[panel]
        rows, columns = PANEL_GRIDS.get(panel, (1, 1))
        shares = np.asarray(PANEL_ROW_SHARES.get(panel, (1,) * rows), dtype=float)
        heights = (
            (top - bottom) * (1 - GRID_GAP[0] * (rows - 1)) * shares / shares.sum()
        )
        tops = (
            top
            - np.concatenate([[0.0], np.cumsum(heights[:-1])])
            - np.arange(rows) * GRID_GAP[0] * (top - bottom)
        )
        gap = PANEL_COLUMN_GAPS.get(panel, GRID_GAP[1])
        across = width * (1 - gap * (columns - 1)) / columns
        axes[panel] = [
            fig.add_axes(
                (
                    left + c * (across + gap * width),
                    tops[r] - heights[r],
                    across,
                    heights[r],
                )
            )
            for r in range(rows)
            for c in range(columns)
        ]
    draw_no_spike_mua(axes["no_spike_mua"][0], statistics["no_spike"])
    draw_event_traces(axes["event_traces"], statistics["event_traces"])
    draw_boundary_timing(axes["boundary_timing"][0], statistics["timing"])
    draw_heldout_tetrode(axes["heldout_gain"][0], statistics["tetrode"])
    draw_participation(axes["participation"][0], statistics["participation"])
    draw_outside_swr_series(
        axes["outside_distance"][0], statistics["outside"], statistics["parts"]
    )
    draw_hpd(axes["hpd"][0], statistics["hpd"])
    draw_step_test(*axes["step_test"], statistics["step_test"])
    draw_nonlocal_mua(*axes["nonlocal_mua"], statistics["mua"])
    draw_rule_sweep(axes["rule_sweep"], statistics["sweep"])
    draw_state_calibration(axes["state_calibration"][0], statistics["calibration"])
    fig.draw_without_rendering()
    # convention 10: a row's letters at one height, just above its highest
    # panel; the rows' first panels share the left column and its letter edge
    extent = {
        panel: drawn_extent(fig, panel_list) for panel, panel_list in axes.items()
    }
    left = {panel: max(e.x0 - 0.004, 0.0) for panel, e in extent.items()}
    column = min(left[row[0]] for _, row in ROWS)
    for _, row in ROWS:
        left[row[0]] = column
        top = max(extent[panel].y1 for panel in row) + LETTER_HEIGHT
        for panel in row:
            letter_panel(fig, LETTERS[panel], left[panel], top)
    check_placement(fig, {LETTERS[p]: panel_list for p, panel_list in axes.items()})
    return fig


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("paper/figures/supp_figure02")
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=NO_SPIKE_WORKERS,
        help="processes that read the decode frames for the No-Spike panel "
        f"(default {NO_SPIKE_WORKERS}; 1 reads them in this process)",
    )
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig = generate_figure(read_inputs(args.workers))
    check_journal_limits(fig)
    logging.info(f"Saving to {args.output_dir}...")
    save_figure(fig, "supp_figure02", base_dir=args.output_dir)
    plt.close("all")


if __name__ == "__main__":
    main()
