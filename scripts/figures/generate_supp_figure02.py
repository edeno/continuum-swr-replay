#!/usr/bin/env python
"""Generate Supplementary Figure S2, the supplement to Figure 2, its panels in
the order the text cites them, in four rows:

- the decoder on these recordings (a, b): its states calibrated by held-out
  spikes, and MUA in the No-Spike state;
- the count is robust (c, d): the share of events with no SWR at other
  settings of the event rule, and the timing of events against their SWRs;
- the events are coordinated and specific (e to h): non-local time against
  MUA, the held-out tetrode gain split by bins with and without spikes, the
  posterior's spread, and shared content;
- ripple power and the held-out test (i, j): the events' peak ripple power has
  one mode and the share with no SWR follows the SWR threshold, and the gain
  outside SWRs away from the SWRs.

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

    c — Supports Figure 2d, the share of events with no SWR: it changes little
        with the event rule's settings. The share of the immobile non-local
        events that start inside a well visit with no SWR, at each value of
        the event rule's two settings swept one at a time (the other at the
        value the paper uses): the non-local probability an event stays
        strictly above (0.5 to 0.95) and the minimum run length (7, the rule's
        15 ms, to 25 samples of 2 ms). A thin line per animal (the mean over
        its epochs of each epoch's share) and the share pooled over events in
        bold; the value the paper uses is dotted and marked "used". The events
        are detected again from the stored decode by the rule itself and
        labeled as the canonical table labels them; at the rule's settings
        they are the canonical events (``rule_sweep.sweep_statistics``;
        ``Processed-Data/event_rule_sweep``,
        ``scripts/event_properties/sweep_event_rule.py``; the events summary's
        ``rule_sweep_probability_threshold_*`` and
        ``rule_sweep_minimum_samples_*`` values). The SWR detector's threshold,
        swept by the same runner, is drawn in panel i.

    d — Supports Figure 2f, which shows that a good share of the events start
        or end more than 25 ms from their SWR without saying which way. Each
        SWR-overlapping event of the canonical event table that starts inside
        a well visit, against its matched SWR (the pairs and offsets of Figure
        2f, ``event_statistics.boundary_offsets``: event minus SWR, so a
        positive start offset is an event that starts later than its SWR, a
        positive end offset one that ends later), classed by its start
        (earlier, within 25 ms, later) and its end (the same), 25 ms being the
        tolerance of the events summary's ``*_beyond_twenty_five_ms_fraction``
        values (``event_statistics.TIMING_TOLERANCE``). Each cell is the share
        of the pairs in it, pooled over every pair as Figure 2c and d pool,
        with the lowest and highest animal's share under it in small type
        (``event_statistics.boundary_timing_statistics``; the events summary's
        ``timing_*`` values). The four corner cells are named, as the event
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
        inside the well visits (``nonlocal_by_signal.nonlocal_counts`` of the
        stored counts, the same table Figure 2i reads, by
        ``signal_share_panels.draw_share_by_signal``;
        ``Processed-Data/nonlocal_by_signal``; the events summary's
        ``nonlocal_share_mua_above_level*`` and
        ``nonlocal_time_below_mua_level_fraction*`` values). Most non-local
        time is below the MUA level, and nearly all time above the MUA level is
        decoded non-local. The panel does not show that non-local detection is
        independent of population bursts: the decoder needs spikes, so the two
        go together. It poses the question that the held-out tetrode panels
        (f, h) answer: whether the spikes point to the decoded place.

    f and h — Support Figure 2h, the events driven by several tetrodes. Each
    tetrode is left out of the decoder in turn: the other tetrodes' estimate
    predicts its spikes better than local coding does (f), and in the events
    where that estimate is specific, its activity is where the others point (h).

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

    h — Shared content: during an event, do the tetrodes point to the same
        remote place? The same events and run as f, each left-out tetrode
        scored under the other tetrodes' position distribution against no
        place preference (the agreement; above zero only when the left-out
        tetrode's activity is where the others point), summed over tetrodes and
        bins and divided by the event's spikes, as the gain is. The events are
        grouped, in bins fixed in advance (``shared_content.SPREAD_BINS_CM``),
        by the width of the other tetrodes' position estimate. For each
        left-out tetrode and each 2 ms bin, the other tetrodes give a
        probability distribution over track position (Local, Continuous and
        Fragmented together, No-Spike left out); its effective width is
        exp(entropy) place bins times the bin size, the length of track the
        distribution effectively covers (a flat distribution over 100 cm gives
        100 cm; two separate narrow peaks give the sum of their widths, not the
        distance between them); the event's value is the mean over its scored
        bins and its left-out tetrodes. It is not the same measure as panel
        g's (the 95 % HPD length of the all-tetrode non-local posterior). A dot
        per animal for no-SWR and SWR events in their colors, beside them the
        animals' mean with its 95 % interval, a thin line through each group's
        means and a dashed line at zero, no place preferred
        (``shared_content.agreement_by_spread_statistics``; the events
        summary's ``*_shared_content_*`` values). Above, a thin strip sharing
        the x axis: the percent of each group's events in each width bin, two
        narrow bars side by side in the groups' colors (``share_of_events``;
        the ``*_shared_content_*_share_of_events`` values).

    i — Supports Figure 2i, which shows no step in non-local time at the SWR
        threshold. Two axes share the ripple-power x axis (-1 to 7 z). Top
        ("Event peak"): the events' peak ripple power (the maximum of the
        z-scored Kay consensus over the event's rows) has one mode. Each
        animal's density of it, a thin line per animal and their mean in bold
        (as Figure 2i draws its animals), is the Gaussian KDE (Silverman's
        bandwidth) whose main mode the one-class-or-two analysis reports
        (``class_or_continuum.peak_power_densities``), with the dip test's
        result in the title (``event_statistics.peak_ripple_mode_statistics``;
        ``Processed-Data/ripple_class_structure``; the events summary's
        ``ripple_peak_*`` values). The distribution is not split by SWR label:
        the label is set from this power, so a split would show the
        threshold, not two kinds. Bottom ("No SWR at this threshold"): the
        percent of events with no SWR if the SWR detector's threshold were at
        each value from 0 to 6 z, a thin line per animal and the cohort's line
        with its points (the SWR-threshold rows of
        ``rule_sweep.sweep_statistics``; the events summary's
        ``rule_sweep_swr_threshold_*`` values). The SWRs are not detected
        again: an event counts as overlapping an SWR at a threshold when its
        Kay criterion on the decode grid reaches it, a label that agrees with
        the canonical one for 99 % of the events at 2 z
        (``rule_sweep_swr_criterion_agreement``), so the bottom's x is that
        threshold, close to but not the same quantity as the top's peak. The
        share follows the threshold, as a threshold on one distribution must,
        and about a third of the events have no SWR even at 1 z. One dashed
        line at the threshold the paper uses (2 z) runs through both axes.

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

Panels g and h read the held-out tetrode run over the epochs the events
summary reads and refuse a run not held out in folds of whole trials, as the summary does.
Panel j reads the held-out event run
(``Processed-Data/event_nonlocal_restriction_heldout_events``, its events and
time sums) by the events summary's reader
(``reported.events.read_heldout_restriction``), and refuse it, as Figure 2j
does, when the restriction run is not the fold fit on folds of whole trials or
was not made from the current decode frames and SWR source, when the held-out
event run was not made from the current restriction run or per-bin facts run,
or when the per-bin facts were not made from the current session cache. Their
values are the events summary's ``restriction_outside_swr_*`` values. Panels d,
f and i read the canonical event table (i with the stored output it names), refused, as their values are marked pending, when an output was not
made from the current event table over the analyses' subset; c, e and a their
stored outputs likewise. Panel b reads every analyzed epoch's decode frame
(``Processed-Data``), which takes a few minutes; it has no reported value.

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
from matplotlib.ticker import MaxNLocator, NullLocator
from matplotlib.transforms import blended_transform_factory

import continuum_swr_replay.visualization.figure_utilities as figure_utilities
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
from continuum_swr_replay.analysis.event_properties.class_or_continuum import (
    KAY_ZSCORE_THRESHOLD,
    peak_power_densities,
)
from continuum_swr_replay.analysis.event_properties.event_statistics import (
    HPD_BIN_EDGES_CM,
    HPD_TIGHT_CM,
    TIMING_CLASSES,
    animal_signal_counts,
    boundary_offsets,
    boundary_timing_statistics,
    hpd_distribution,
    nonlocal_below_threshold_statistics,
    peak_ripple_mode_statistics,
)
from continuum_swr_replay.analysis.event_properties.nonlocal_by_signal import (
    MUA_LEVEL_Z,
    nonlocal_counts,
)
from continuum_swr_replay.analysis.event_properties.rule_sweep import (
    RULE,
    sweep_statistics,
)
from continuum_swr_replay.analysis.event_properties.triggered_analysis import (
    get_signal_windows,
    window_quartiles,
)
from continuum_swr_replay.analysis.event_tables.prepare_data import well_visit_times
from continuum_swr_replay.data_loaders.epoch_register import (
    analyzed_animals,
    analyzed_decode_files,
    analyzed_epoch_ids,
)
from continuum_swr_replay.decoding.files import data_over_time_suffix
from continuum_swr_replay.paths import TETRODE_HOLDOUT_NAME, processed_data_dir
from continuum_swr_replay.reported.events import (
    default_inputs,
    event_table_staleness,
    read_heldout_restriction,
    require_tetrode_holdout_folds,
    restriction_inputs_note,
)
from continuum_swr_replay.reported.formatting import format_number
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
    SPLIT_LABELS,
    draw_animal_gain,
    draw_split_gain,
    full_better_cue,
    label_gain_axis,
    spread_offsets,
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

# Inches: the full page width, four rows (the module's docstring).
WIDTH = 6.5
HEIGHT = 7.6

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
SHARED_YLABEL = "Held-out agreement\n(nats/spike)"
SHARED_XLABEL = "Width of the other tetrodes'\nposition estimate (cm)"
RIPPLE_GRID_Z = np.linspace(-2.0, 12.0, 281)
"""Where the ripple panel reads each animal's density of the events' peak power
(z): past 12 z lies a thin tail."""
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
    "minimum_samples": "Minimum\nsamples",
}
"""The sweep panel's x label of each of the event rule's two settings (the SWR
detector's threshold is drawn in the ripple panel)."""
SWEEP_TICKS = {
    "probability_threshold": (0.5, 0.8, 0.95),
    "minimum_samples": (7, 15, 25),
}
"""The sweep panel's x ticks: the rule's value and the ends."""
RIPPLE_X_LIMITS_Z = (-1.0, 7.0)
"""The ripple panel's shared x range (z): where the events' peak ripple power
and the share with no SWR at each SWR threshold both have something to show."""
HPD_XLABEL = "95 % HPD length (cm),\nmean over the event's bins"
"""The posterior-spread panel's x label: each event's mean over its bins of the
length of track that holds 95 % of the decoded probability."""
HPD_YLABEL = "% of events"
HPD_TICKS_CM = (10, 25, 50, 100, 200, 400)
SHARED_OFFSET = 0.2
"""The shared-content panel: each group's dots sit this far either side of its spread bin; the
mean with its interval sits a further :data:`SHARED_MEAN_OFFSET` out."""
SHARED_MEAN_OFFSET = 0.11
SHARE_BAR_WIDTH = 0.3
"""The width of each group's bar in the shared-content panel's strip (x units)."""

DISTANCE_LABEL = "at least this far from\nany SWR (ms)"
OUTSIDE_LABEL = "Outside\nSWRs"
PARTS_LABEL_DROP = 21
"""How far below the axis a series' parts label starts (points): under the two
lines of "Outside SWRs", so that it can be wider than its columns."""
SERIES_START = 1.3
"""The x position of a series' first part, after the "Outside SWRs" column at 0;
the parts are 1 apart."""
PANELS = (
    "state_calibration",
    "no_spike_mua",
    "rule_sweep",
    "boundary_timing",
    "nonlocal_mua",
    "heldout_gain",
    "hpd",
    "shared_content",
    "ripple_modes",
    "outside_distance",
)
"""The panels in the order the text cites them; their letters follow from it."""
LETTERS = dict(zip(PANELS, string.ascii_lowercase, strict=False))
ROWS = (
    ((0.81, 0.95), ("state_calibration", "no_spike_mua")),
    ((0.56, 0.7), ("rule_sweep", "boundary_timing")),
    ((0.31, 0.45), ("nonlocal_mua", "heldout_gain", "hpd", "shared_content")),
    ((0.06, 0.2), ("ripple_modes", "outside_distance")),
)
"""Each row's frames' bottom and top (figure height) and its panels, left to
right."""
FRAMES = {
    "state_calibration": (0.12, 0.4),
    "no_spike_mua": (0.71, 0.27),
    "rule_sweep": (0.085, 0.38),
    "boundary_timing": (0.66, 0.32),
    "nonlocal_mua": (0.075, 0.13),
    "heldout_gain": (0.305, 0.19),
    "hpd": (0.57, 0.155),
    "shared_content": (0.8, 0.185),
    "ripple_modes": (0.085, 0.3),
    "outside_distance": (0.55, 0.36),
}
"""Each panel's frame's left edge and width (figure width)."""
LETTER_HEIGHT = 0.035
"""A panel letter's height (figure height): the letters' tops sit this far
above the highest drawn top of their row."""
NO_SPIKE_SIGNAL = "MUA_zscore"
TRIGGER_WINDOW_S = 0.3
"""The No-Spike panel's window around each start (s, in all), Figure 2g's: the periods
last a few tens of ms."""
NO_SPIKE_COLOR = COLORS["No-Spike"]
REFUSED = f"Figure 2's supplement: panel {LETTERS['outside_distance']} is not drawn"
RERUN = {
    "restriction_inputs": "scripts/event_evidence/score_nonlocal_restriction.py "
    "and the runs made from it",
    "restriction_run": "scripts/event_evidence/detect_heldout_events.py",
    "bin_facts_run": "scripts/event_evidence/detect_heldout_events.py",
    "session_cache": "scripts/event_evidence/measure_bin_facts.py, then "
    "scripts/event_evidence/detect_heldout_events.py",
}
"""The runners to make again when the panel of the gain outside SWRs refuses
its inputs."""


def read_inputs() -> dict:
    """The figure's statistics, as the events summary reads them.

    Returns
    -------
    dict
        ``tetrode``: the held-out tetrode statistics of the events that start
        inside a well visit, indexed by group (a held-out tetrode run not held
        out in folds of whole trials is refused,
        ``reported.events.require_tetrode_holdout_folds``); ``timing``: the
        selected events' ``boundary_timing_statistics``; ``shared``: their
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
    events = select_visit_context(table)
    selected = select_events_in(scores, events["event_id"])
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
        "calibration": read_state_calibration(inputs.state_calibration_in_visits),
        "sweep": read_rule_sweep(inputs.event_rule_sweep, inputs.event_table),
        "mua": read_mua_counts(inputs.nonlocal_by_signal),
        "ripple_modes": read_ripple_modes(
            events, inputs.class_structure, inputs.event_table
        ),
        "hpd": read_hpd(events),
        "timing": boundary_timing_statistics(
            boundary_offsets(events), events["animal"]
        ),
        "tetrode": {
            column: holdout_statistics(windows, "event", column)
            for column in SPLIT_LABELS
        },
        "shared": agreement_by_spread_statistics(windows[windows["kind"] == "event"]),
        "outside": outside_swr_gain_statistics(
            epoch_outside_swr_gains(heldout.time_sums, heldout.bin_s, heldout.events)
        ).set_index(["family", "measure"]),
        "parts": outside_swr_subset_statistics(
            heldout.time_sums, heldout.bin_s
        ).set_index(["subset", "threshold_s", "family", "measure"]),
        "no_spike": read_no_spike_trace(),
    }


def read_hpd(events: pd.DataFrame) -> pd.DataFrame:
    """The posterior-spread panel's distribution: each animal's percent of its
    events of each class per HPD coverage bin
    (``event_statistics.hpd_distribution``); the share below the tight cut is
    the events summary's (``hpd_tight_share_*``)."""
    return hpd_distribution(events)


def _refuse_stale(folder: Path, event_table: Path, panel: str, rerun: str) -> None:
    """Refuse an output not made from the current event table over the
    analyses' subset, as the events summary marks its values pending."""
    manifest = read_output_manifest(folder)
    for reason in (
        event_table_staleness(manifest, sha256_file(event_table), rel(folder)),
        subset_staleness(manifest, rel(folder)),
    ):
        if reason is not None:
            raise ValueError(
                f"Figure 2's supplement: panel {LETTERS[panel]} is not drawn: "
                f"{reason}. Rerun {rerun}."
            )


def read_state_calibration(folder: Path) -> pd.DataFrame:
    """The calibration panel's bins: the held-out state calibration over the
    immobile windows inside the well visits, as the events summary reads it
    (``state_evidence.calibration_statistics``' bins, written by
    ``scripts/event_evidence/summarize_state_calibration_in_visits.py``).

    Raises
    ------
    ValueError
        If the summary was not made over the analyses' subset, or not over the
        analyzed epochs.
    """
    manifest = read_output_manifest(folder)
    if (reason := subset_staleness(manifest, rel(folder))) is not None:
        raise ValueError(
            f"Figure 2's supplement: panel {LETTERS['state_calibration']} is not "
            f"drawn: {reason}. Rerun "
            "scripts/event_evidence/summarize_state_calibration_in_visits.py."
        )
    if set(manifest["epochs"]) != set(analyzed_epoch_ids()):
        raise ValueError(
            f"Figure 2's supplement: {rel(folder)} holds {len(manifest['epochs'])} "
            "epochs, not the analyzed ones"
        )
    return pd.read_csv(folder / "calibration_bins.csv")


def read_rule_sweep(folder: Path, event_table: Path) -> pd.DataFrame:
    """The sweep panel's statistics (``rule_sweep.sweep_statistics`` of the
    stored per-epoch counts), as the events summary reads them.

    Raises
    ------
    ValueError
        If the counts were not made from the current event table over the
        analyses' subset, or do not hold the analyzed epochs.
    """
    _refuse_stale(
        folder, event_table, "rule_sweep",
        "scripts/event_properties/sweep_event_rule.py",
    )  # fmt: skip
    counts = pd.read_parquet(folder / "counts.parquet")
    if set(counts["epoch_id"]) != set(analyzed_epoch_ids()):
        raise ValueError(
            f"Figure 2's supplement: {rel(folder)}/counts.parquet holds "
            f"{counts['epoch_id'].nunique()} epochs, not the analyzed ones"
        )
    return sweep_statistics(counts)


def read_mua_counts(folder: Path) -> pd.DataFrame:
    """The MUA panel's counts: each animal's immobile time bins inside the well
    visits, and its non-local ones, by MUA (``nonlocal_by_signal.nonlocal_counts``
    of the stored counts, ``event_statistics.animal_signal_counts``), as the
    events summary reads them.

    Raises
    ------
    ValueError
        If the counts were not made over the analyses' subset, or do not hold
        the analyzed epochs.
    """
    if (
        reason := subset_staleness(read_output_manifest(folder), rel(folder))
    ) is not None:
        raise ValueError(
            f"Figure 2's supplement: panel {LETTERS['nonlocal_mua']} is not drawn: "
            f"{reason}. Rerun scripts/event_properties/measure_nonlocal_by_signal.py."
        )
    counts = pd.read_parquet(folder / "counts.parquet")
    if set(counts["epoch_id"]) != set(analyzed_epoch_ids()):
        raise ValueError(
            f"Figure 2's supplement: {rel(folder)}/counts.parquet holds "
            f"{counts['epoch_id'].nunique()} epochs, not the analyzed ones"
        )
    return animal_signal_counts(nonlocal_counts(counts, "mua"))


def read_ripple_modes(events: pd.DataFrame, folder: Path, event_table: Path) -> dict:
    """The ripple panel's statistics from the one-class-or-two output: each
    animal's density of the selected events' peak ripple power
    (``class_or_continuum.peak_power_densities``, the KDE whose modes the
    output's ``distribution.csv`` reports) and the one-mode statistic of the
    events summary (``event_statistics.peak_ripple_mode_statistics``).

    Returns
    -------
    dict
        ``densities`` (indexed by :data:`RIPPLE_GRID_Z`, a column per animal),
        ``modes`` and ``n_animals``.

    Raises
    ------
    ValueError
        If the output is not current, or a selected event has no peak power.
    """
    _refuse_stale(
        folder, event_table, "ripple_modes",
        "scripts/event_properties/measure_ripple_class_structure.py analyze",
    )  # fmt: skip
    peaks = pd.concat(
        pd.read_parquet(path, columns=["event_id", "ripple_z_max"])
        for path in sorted((folder / "criterion").glob("*.parquet"))
    )
    events = events[["event_id", "animal"]].merge(
        peaks, on="event_id", how="left", validate="1:1"
    )
    if events["ripple_z_max"].isna().any():
        raise ValueError(
            f"Figure 2's supplement: {int(events['ripple_z_max'].isna().sum())} "
            f"selected events have no peak ripple power in {rel(folder)}/criterion"
        )
    n_animals = len(analyzed_animals())
    return {
        "densities": peak_power_densities(events, RIPPLE_GRID_Z),
        "modes": peak_ripple_mode_statistics(
            pd.read_csv(folder / "distribution.csv"), n_animals
        ),
        "n_animals": n_animals,
    }


def read_no_spike_trace() -> dict:
    """The No-Spike panel's trace: MUA around the start of every analyzed epoch's No-Spike
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


def draw_boundary_timing(ax: Axes, timing: pd.DataFrame) -> None:
    """The timing table: the share of SWR-overlapping events in each combination of
    start (rows) and end (columns) against the matched SWR's, as a heat map
    with the percent in each cell, the animals' range (in percent) under it
    and the corner cells' names (:data:`TIMING_CORNERS`) above it, in black
    or white, whichever reads against the cell (``timing``: :func:`read_inputs`'
    ``timing``)."""
    n = len(TIMING_CLASSES)
    shares = timing["share"].to_numpy().reshape(n, n)
    cmap = plt.get_cmap(TIMING_CMAP)
    ax.imshow(100 * shares, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for (row, column), cell in zip(np.ndindex(n, n), timing.itertuples(), strict=True):
        red, green, blue, _ = cmap(cell.share)
        color = "white" if 0.299 * red + 0.587 * green + 0.114 * blue < 0.5 else "black"
        if cell.Index in TIMING_CORNERS:
            ax.text(
                column, row + CORNER_DROP, TIMING_CORNERS[cell.Index],
                ha="center", va="center", fontsize=SMALL_SIZE, color=color,
                style="italic",
            )  # fmt: skip
        ax.text(
            column, row + SHARE_DROP, f"{100 * cell.share:.0f}%", ha="center",
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


def draw_shared_content(ax_share: Axes, ax: Axes, statistics: pd.DataFrame) -> None:
    """The shared-content panel: the held-out agreement per spike by spread bin, for no-SWR and
    SWR events (``statistics``: :func:`read_inputs`' ``shared``). Each group's
    dots (one per animal) in its color on its side of the bin, the animals'
    mean with its 95 % interval in black farther out, a thin line through the
    group's means, the group's name beside its first bin, a dashed line at zero
    and, above it at the right, which way means shared content. Above, a thin
    strip sharing the x axis: each group's percent of its events in each bin
    (``share_of_events``), two narrow bars side by side in the groups' colors."""
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
    # above the zero line at the right, over the widest bin, whose values are
    # all below zero
    ax.annotate(
        "\u2191 shared\ncontent", (0.99, 0.0),
        xycoords=blended_transform_factory(ax.transAxes, ax.transData),
        xytext=(0, 4), textcoords="offset points", ha="right", va="bottom",
        fontsize=SMALL_SIZE, color=REFERENCE_COLOR, linespacing=0.95,
    )  # fmt: skip
    for k, (name, color) in enumerate(
        zip(("No SWR", "SWR"), GROUP_COLORS.values(), strict=True)
    ):
        ax.text(
            0.99, 0.97 - 0.1 * k, name, ha="right", va="top",
            transform=ax.transAxes, fontsize=SMALL_SIZE,
            color=readable_text_color(color),
        )  # fmt: skip
    # the strip above: each group's percent of its events in each bin
    for side, (group, color) in zip((-1, 1), GROUP_COLORS.items(), strict=True):
        shares = [
            100 * statistics.loc[(label, group), "share_of_events"]
            for label in SPREAD_BIN_LABELS
        ]
        ax_share.bar(
            bins + side * SHARE_BAR_WIDTH / 2, shares, width=SHARE_BAR_WIDTH,
            color=color, lw=0,
        )  # fmt: skip
    ax_share.sharex(ax)
    ax_share.tick_params(labelbottom=False)
    ax_share.set_ylim(0, None)
    ax_share.yaxis.set_major_locator(MaxNLocator(2))
    ax_share.set_ylabel("% of\nevents")
    ax_share.set_title("Shared content", fontsize=SMALL_SIZE, pad=3)
    ax.set_xlim(-0.6, bins[-1] + 0.6)
    ax.set_xticks(bins, SPREAD_BIN_LABELS)
    ax.set_xlabel(SHARED_XLABEL)
    ax.set_ylabel(SHARED_YLABEL)


def draw_outside_swr_series(
    ax: Axes,
    outside: pd.DataFrame,
    parts: pd.DataFrame,
    subset: str,
    thresholds: Sequence[float],
    names: Sequence[str],
    xlabel: str,
) -> None:
    """The gain outside SWRs: the full decoder minus the SWR-only decoder per spike
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


def draw_ripple_modes(
    ax_peak: Axes, ax_share: Axes, statistics: dict, sweep: pd.DataFrame
) -> None:
    """The ripple panel, two axes sharing the ripple-power x axis. Top, each
    animal's density of the events' peak ripple power as a thin line, their
    mean in bold (as Figure 2i draws its animals), the one-mode result in its
    title; not
    split by SWR label, which is set from this power (``statistics``:
    :func:`read_ripple_modes`). Bottom, the percent of events with no SWR if
    the SWR threshold were at each value, applied to each event's Kay
    criterion on the decode grid: a thin line per animal and the cohort's line
    with its points (``sweep``: the SWR-threshold rows of :func:`read_inputs`'
    ``sweep``). One dashed line at the threshold the paper uses through both,
    labeled once; each axis names what its x is."""
    densities = statistics["densities"]
    grid = densities.index.to_numpy()
    ax_peak.plot(grid, densities.to_numpy(), color="k", lw=0.5, alpha=0.5)
    ax_peak.plot(grid, densities.mean(axis=1).to_numpy(), color="k", lw=1.0)
    modes = statistics["modes"]
    ax_peak.set_ylim(0, 1.45 * float(densities.to_numpy().max()))  # room for labels
    ax_peak.set_ylabel("Density\n(per z)")
    ax_peak.set_title(
        f"One mode in {modes['n_animals_unimodal']} of {statistics['n_animals']} "
        "animals (dip test)",
        fontsize=SMALL_SIZE, pad=3,
    )  # fmt: skip
    ax_peak.tick_params(labelbottom=False)

    mine = sweep[sweep["setting"] == "swr_threshold_z"]
    x = mine["value"].to_numpy()
    animals = pd.DataFrame(
        [pd.Series(r.animals, index=r.animal_names) for r in mine.itertuples()]
    )
    ax_share.plot(x, 100 * animals.to_numpy(), color="k", lw=0.5, alpha=0.5)
    ax_share.plot(x, 100 * mine["no_swr_fraction"].to_numpy(), color="k", lw=1.0,
                  marker="o", ms=2)  # fmt: skip
    ax_share.sharex(ax_peak)
    ax_share.set_ylim(0, 100)
    ax_share.set_yticks([0, 50, 100])
    ax_share.set_ylabel("Events with\nno SWR (%)")
    ax_share.set_xlabel("Ripple power (z)")
    # each axis names its x: the top's at its upper left, the bottom's at its
    # lower right, where the curves leave room
    for ax, heading, (x, y, ha, va) in (
        (ax_peak, "Event peak", (0.02, 0.95, "left", "top")),
        (ax_share, "No SWR at this threshold", (0.98, 0.05, "right", "bottom")),
    ):
        ax.axvline(KAY_ZSCORE_THRESHOLD, color=COLORS["SWR"], lw=0.8, ls="--")
        ax.text(
            x, y, heading, transform=ax.transAxes, ha=ha, va=va,
            fontsize=SMALL_SIZE, color=REFERENCE_COLOR,
        )  # fmt: skip
    ax_peak.text(
        KAY_ZSCORE_THRESHOLD + 0.15, 0.95, "SWR threshold used",
        transform=ax_peak.get_xaxis_transform(), ha="left", va="top",
        fontsize=SMALL_SIZE, color=readable_text_color(COLORS["SWR"]),
    )  # fmt: skip
    ax_peak.set_xlim(*RIPPLE_X_LIMITS_Z)
    ax_peak.set_xticks([0, 2, 4, 6])


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
    """The sweep panel: the share of events with no SWR at each value of each
    of the event rule's two settings (one axis each, sharing the y axis), a
    thin line per animal and the pooled share in bold with its dots, the value
    the paper uses dotted and marked "used", and the title "Event rule" over
    both (``sweep``: :func:`read_inputs`' ``sweep``; its SWR-threshold rows
    are drawn in the ripple panel)."""
    for ax, (setting, xlabel) in zip(axes, SWEEP_XLABELS.items(), strict=True):
        mine = sweep[sweep["setting"] == setting]
        x = mine["value"].to_numpy()
        animals = pd.DataFrame(
            [pd.Series(r.animals, index=r.animal_names) for r in mine.itertuples()]
        )
        ax.plot(x, 100 * animals.to_numpy(), color="k", lw=0.5, alpha=0.5)
        ax.plot(x, 100 * mine["no_swr_fraction"].to_numpy(), color="k", lw=1.0,
                marker="o", ms=2)  # fmt: skip
        ax.axvline(RULE[setting], color=REFERENCE_COLOR, lw=0.6, ls=":", zorder=1)
        ax.set_xticks(SWEEP_TICKS[setting])
        ax.set_xlabel(xlabel)
        if ax is not axes[0]:
            ax.sharey(axes[0])
            ax.tick_params(labelleft=False)
    axes[0].set_ylim(0, 100)
    axes[0].set_yticks([0, 50, 100])
    axes[0].set_ylabel("Events with\nno SWR (%)")
    for ax, setting in zip(axes, SWEEP_XLABELS, strict=True):
        ax.text(
            RULE[setting], 3, " used", fontsize=SMALL_SIZE, color=REFERENCE_COLOR,
            ha="left", va="bottom",
        )  # fmt: skip
    # the panel's title, centered over its two axes
    first, second = axes[0].get_position(), axes[1].get_position()
    axes[0].annotate(
        "Event rule", (((first.x0 + second.x1) / 2 - first.x0) / first.width, 1.0),
        xycoords="axes fraction", xytext=(0, 3), textcoords="offset points",
        ha="center", va="bottom", fontsize=SMALL_SIZE,
    )  # fmt: skip


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
        zip(("No SWR", "SWR"), GROUP_COLORS.values(), strict=True)
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
    """The No-Spike panel: MUA around No-Spike starts (``trace``: :func:`read_no_spike_trace`),
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


PANEL_GRIDS = {
    "nonlocal_mua": (2, 1),
    "rule_sweep": (1, 2),
    "ripple_modes": (2, 1),
    "shared_content": (2, 1),
}
"""The panels drawn on several axes in their frame: rows and columns."""
PANEL_ROW_SHARES = {"shared_content": (1, 3)}
"""The heights of a stacked panel's rows, as shares (equal when not named)."""
GRID_GAP = (0.1, 0.1)
"""The gap between a panel's axes, as a share of its frame's height and width."""


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
    order and laid out by :data:`ROWS` and :data:`FRAMES`."""
    spans = {panel: span for span, row in ROWS for panel in row}
    assert sorted(spans) == sorted(PANELS)
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
        across = width * (1 - GRID_GAP[1] * (columns - 1)) / columns
        axes[panel] = [
            fig.add_axes(
                (
                    left + c * (across + GRID_GAP[1] * width),
                    tops[r] - heights[r],
                    across,
                    heights[r],
                )
            )
            for r in range(rows)
            for c in range(columns)
        ]
    draw_no_spike_mua(axes["no_spike_mua"][0], statistics["no_spike"])
    draw_boundary_timing(axes["boundary_timing"][0], statistics["timing"])
    draw_heldout_tetrode(axes["heldout_gain"][0], statistics["tetrode"])
    draw_shared_content(*axes["shared_content"], statistics["shared"])
    draw_outside_swr_series(
        axes["outside_distance"][0], statistics["outside"], statistics["parts"],
        "beyond_swr_margin", OUTSIDE_SWR_MARGINS_S,
        [f"{t * 1000:g}" for t in OUTSIDE_SWR_MARGINS_S], DISTANCE_LABEL,
    )  # fmt: skip
    draw_hpd(axes["hpd"][0], statistics["hpd"])
    draw_ripple_modes(
        *axes["ripple_modes"], statistics["ripple_modes"], statistics["sweep"]
    )
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
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig = generate_figure(read_inputs())
    check_journal_limits(fig)
    logging.info(f"Saving to {args.output_dir}...")
    save_figure(fig, "supp_figure02", base_dir=args.output_dir)
    plt.close("all")


if __name__ == "__main__":
    main()
