#!/usr/bin/env python
"""Generate Supplementary Figure S2, the supplement to Figure 2, its panels in
the order the text cites them, in five rows:

- the decoder on these recordings (a to d): its states calibrated by held-out
  spikes, MUA in the No-Spike state, how the immobile time is spread over
  non-local probability, and how far the decoded place is from the animal by
  Local probability;
- the count is robust (e, f): the share of events with no SWR at other
  settings of the event rule and of the SWR detector, and the timing of events
  against their SWRs;
- the held-out tests (g to i): non-local time against MUA, Figure 2j's
  first comparison looked at more closely (over the animal's place, by bins
  with and without a spike), and Figure 2k's gain outside SWRs away from the
  SWRs;
- the animals agree, and the posterior's spread (j, k): Figure 2g's traces
  animal by animal, and the distribution of the width of the decoded place;
- a fourth decoder (l): Figure 2k's comparison with the SWR-allowed decoder.

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

    c — Supports Figure 2a and the event rule: how the immobile time inside
        the well visits is spread over the decoder's non-local probability
        (Continuous plus Fragmented). Each epoch's immobile, analyzed time bins
        inside a well visit counted in 50 equal probability bins
        (``nonlocal_probability.probability_counts``,
        ``scripts/decoding/measure_nonlocal_probability.py``,
        ``Processed-Data/nonlocal_probability``, refused when it was not made
        over the analyses' subset from the current well visits); the percent
        of the time per bin on a log axis, a thin line per animal (the mean
        over its epochs of each epoch's percent) and the mean over animals in
        bold (``nonlocal_probability.animal_probability_distribution``), the
        event rule's cut (0.8) dotted and named "used". Almost all the time is
        near 0 or near 1, so the cut is not a fine choice (the events summary's
        ``nonlocal_probability_*_share`` values: the time at most 0.2, between
        0.2 and 0.8, and above 0.8,
        ``nonlocal_probability.probability_share_statistics``).

    d — Supports Figure 2a: how far the decoded place is from the animal
        against the decoder's Local probability, over all analyzed time,
        moving and immobile (the place-field fit, not the events at the
        wells). The distance along the track between the animal and the
        maximum of the position estimate conditional on the non-local states
        (``ahead_behind_distance`` of the decode frame), each session's median
        in ten equal Local-probability bins
        (``quality_metrics.compute_calibration_curve``, the decoder-quality
        run's table); a thin line per animal (the median over its sessions) and
        the mean over animals in bold
        (``quality_metrics.animal_distance_by_local_probability``; the events
        summary's ``calibration_error_bin_*`` values, whose keys keep their old
        name). The distance falls as the Local probability rises.

    e — Supports Figure 2d, the share of events with no SWR. The text cites
        this panel twice: its event-rule axes with the count of events (the
        share changes little with the rule), and its SWR-threshold axis with
        the ripple-power panel (Figure 2i), because the share follows
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

    f — Supports Figure 2f, which shows that a good share of the events start
        or end more than 10 ms from their SWR without saying which way. Each
        SWR-overlapping event of the canonical event table that starts inside
        a well visit, against its matched SWR (the pairs and offsets of Figure
        2f, ``event_statistics.boundary_offsets``: event minus SWR, so a
        positive start offset is an event that starts later than its SWR, a
        positive end offset one that ends later), classed by its start
        (earlier, within 10 ms, later) and its end (the same), 10 ms being the
        tolerance of the events summary's ``*_beyond_ten_ms_fraction``
        values (``event_statistics.TIMING_TOLERANCE``; "same" in the panel).
        Nine rows (:data:`TIMING_CASES`): the same start and end, the four
        combinations with one bound apart, then the four with both apart, each
        with its own drawing and each beside its opposite. Each row: its name and a small drawing of the case (the
        SWR as a bar in the SWR color, the event under it in the non-local
        color, longer or shorter at the bound that differs;
        ``evidence_drawings.draw_boundary_cases``), a bar of the mean over
        animals of each animal's share of its pairs (the mean over its epochs
        of each epoch's share), a dot per animal, and the percent at the bar's
        end (``event_statistics.boundary_timing_statistics``; the events
        summary's ``timing_start_*_end_*_share_animal_*`` values).

    g — Supports Figure 2g and i, on population activity and the events:
        where the non-local time sits in MUA (the z-scored multiunit rate of
        the decoded tetrodes, ``MUA_zscore``), in Figure 2i's format. Top,
        confident Local time (Local probability above 0.8) and confident
        non-local time (non-local probability above 0.8) spread over MUA
        (percent of each class's time per 0.25 z; the mean over animals of
        each animal's mean over its epochs,
        ``state_by_signal.state_distribution`` of
        ``Processed-Data/state_by_signal``), each in its state's color and
        named, the share of non-local time below 2 z written; bottom, the percent of
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
        (Figure 2j, and h here) answer: whether the spikes point
        to the decoded place.

    h — Supports Figure 2j, its first comparison: the left-out tetrode's
        better fit to the decoded place than to the animal's place (the gain
        over local coding per spike; each tetrode left out of the decoder's
        smoother in turn and scored under the other tetrodes' position
        distribution, encodings held out in folds of whole trials), for no-SWR
        and SWR events, over all bins and split by the scored tetrode's own
        bins: those where it spiked and those where it was silent, which
        answers whether the advantage comes from the spikes or from their
        absence (the ground process). For each event the two parts add to the
        whole; each drawn value is a median over events, so over the cohort
        they need not add up exactly. A dot per animal and the animals' mean
        with its 95 % interval, a dashed line at zero, equal fit
        (``tetrode_holdout.holdout_statistics`` of
        ``Processed-Data/event_tetrode_holdout``, drawn by
        ``heldout_gain_panels.draw_split_gain``; the events summary's
        ``*_heldout_gain_per_spike``, ``*_heldout_spiking_gain_*`` and
        ``*_heldout_silent_gain_*`` values).

    i — Supports Figure 2k: its main result, the full decoder's gain over the
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

    j — Supports Figure 2g, whose lines are the mean over animals of these
        lines: the same signals around the same events' starts (ripple
        power, MUA, theta power, head speed; ``triggered_analysis.TRIGGERED_SIGNALS``
        over the window ``TRIGGERED_WINDOW_S``), a thin line per animal and
        class (SWR, No SWR, in Figure 2g's colors), each the animal's median over
        its events (``triggered_analysis.animal_window_medians`` of
        ``triggered_analysis.triggered_windows``, the windows Figure 2g reduces;
        read from every analyzed epoch's decode frame in panel b's pass). The
        animals agree.

    k — Supports Figure 2's case that the no-SWR events are specific: how
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

    l — Supports Figure 2k against a permissive comparator: a fourth decoder,
        SWR-allowed (every state inside SWRs, only the local states outside
        them; open dots, a variant of SWR-only), scored with the other three
        in one pass under the stored fold fits
        (``Processed-Data/event_nonlocal_restriction_swr_allowed``, from
        ``scripts/event_evidence/score_nonlocal_restriction.py
        --from-stored-fit``), drawn as Figure 2k's decoder comparisons are. First axis: the SWR-only,
        SWR-allowed and full decoders' gains over the local-only decoder per
        spike of all the immobile time inside a well visit. Second axis: the
        full minus the SWR-allowed decoder (filled) and the SWR-allowed minus
        the SWR-only decoder (open) over all the immobile time, inside SWRs and
        outside SWRs; outside SWRs the SWR-allowed and SWR-only decoders allow
        the same states (their scores there differ by about 2e-5 nats per
        spike, through what each carries out of the SWRs), inside SWRs they
        differ only in whether the local states are allowed. Third axis: the full minus the SWR-allowed decoder outside
        SWRs in the no-SWR events detected from that run's held-out decode.
        Each a dot per animal and the animals' mean with its 95 % interval
        (``restriction_totals.swr_allowed_statistics``; the events summary's
        values tagged S2l).

Panel h reads the held-out tetrode run over the epochs the events summary
reads, and refuses a run not held out in folds of whole trials, as the summary
does. Panel i reads the held-out event run
(``Processed-Data/event_nonlocal_restriction_heldout_events``, its events and
time sums) by the events summary's reader
(``reported.events.read_heldout_restriction``) and refuses it, as Figure 2k
does, when the restriction run is not the fold fit on folds of whole trials or
was not made from the current decode frames and SWR source, when the held-out
event run was not made from the current restriction run or per-bin facts run,
or when the per-bin facts were not made from the current session cache. Its
values are the events summary's ``restriction_outside_swr_*`` values. Panel l
reads the held-out event run made from the run with the fourth decoder
(``Processed-Data/event_nonlocal_restriction_swr_allowed_heldout_events``) by
the same reader and refuses it on the same grounds. Panels f and k read the
canonical event table; a, c, e and g read
their stored outputs, d the decoder-quality run's table
(``Processed-Data/decoder_quality/calibration.parquet``) as the events summary
reads it. Each output is refused, as the summary marks its values
pending, when it was not made from the current event table (e), over the
analyses' subset, or from the current well visits (a, c, e and g; g also from the
decodes the event features run records, as Figure 2i). Panel b reads every
analyzed epoch's decode frame (``Processed-Data``), which takes a few minutes;
it has no reported value.

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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
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
from continuum_swr_replay.analysis.detection.nonlocal_events import get_event_types
from continuum_swr_replay.analysis.event_definition.analyzed_time import (
    read_dot_with_swr_source,
    read_swr_table,
)
from continuum_swr_replay.analysis.event_definition.cohort_selection import (
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
    analyzed_decode_files,
    analyzed_epoch_ids,
    require_analyzed_epochs,
)
from continuum_swr_replay.decoding.files import data_over_time_suffix
from continuum_swr_replay.decoding.identifiers import parse_epoch_identifier
from continuum_swr_replay.parameters import DECODE_SAMPLE_S
from continuum_swr_replay.paths import (
    processed_data_dir,
)
from continuum_swr_replay.reported.events import (
    default_inputs,
    event_table_staleness,
    heldout_restriction_refusal,
    read_checked,
    read_heldout_restriction,
    restriction_inputs_note,
)
from continuum_swr_replay.reported.formatting import format_number
from continuum_swr_replay.reported.values import rel
from continuum_swr_replay.runs.io import sha256_file
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
    readable_text_color,
)
from continuum_swr_replay.visualization.panel_layout import (
    fit_panel,
    letter_layout,
    panel_spans,
    place_letters,
)
from continuum_swr_replay.visualization.signal_share_panels import draw_share_by_signal
from continuum_swr_replay.visualization.simulated_run_panels import (
    SMALL_SIZE,
    STATE_COLORS,
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

# Inches: the full page width, five rows (the module's docstring).
WIDTH = 6.5
HEIGHT = 6.8
LEFT_OUT_YLABEL = "Left-out tetrode:\nbetter fit (nats/spike)"
"""The y label of the left-out tetrode's split (panel h)."""
SPLIT_TITLE = "Decoded place over\nanimal's place"
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
TIMING_NOTE = f"same: within {1000 * OFFSET_TAILS_S[TIMING_TOLERANCE]:g} ms"
"""Panel f's note of what "same" means (the tolerance of
``event_statistics.TIMING_TOLERANCE``)."""
TIMING_BAR_COLOR = "#bbbbbb"
"""Panel f's bars of the mean over animals: a neutral gray, under the animals'
dots."""
CALIBRATION_COLORS = {
    "Local": STATE_COLORS["Local"],
    "Non-Local Continuous": STATE_COLORS["Continuous"],
    "Non-Local Fragmented": STATE_COLORS["Fragmented"],
}
"""The calibration panel's states (``state_evidence.IN_VISIT_CALIBRATION_METRICS``),
in the decoder figures' state colors."""
CALIBRATION_YLABEL = "State's place over a\nmismatched place\n(nats/spike)"
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
HPD_XLABEL = "95 % HPD length (cm)"
"""The posterior-spread panel's x label: each event's mean over its bins of the
length of track that holds 95 % of the decoded probability (on one line, clear
of the next row's titles; the mean over the bins is in the caption)."""
HPD_YLABEL = "% of events"
HPD_TICKS_CM = (10, 25, 50, 100, 400)
DISTANCE_LABEL = "at least this far from\nany SWR (ms)"
OUTSIDE_LABEL = "Outside\nSWRs"
PARTS_LABEL_DROP = 13
"""How far below the axis a series' parts label starts (points): just under
the parts' one-line tick labels, beside the two lines of "Outside SWRs" (it
is centered under the parts, clear of that column), so that it sits well above
the next row's letters."""
SERIES_START = 1.3
"""The x position of a series' first part, after the "Outside SWRs" column at 0;
the parts are 1 apart."""
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
NO_SPIKE_SIGNAL = "MUA_zscore"
TRIGGER_WINDOW_S = 0.3
"""The No-Spike panel's window around each start (s, in all), Figure 2g's: the periods
last a few tens of ms."""
NO_SPIKE_COLOR = COLORS["No-Spike"]
NO_SPIKE_WORKERS = 4
"""The processes that read the decode frames for the No-Spike panel by
default: each holds one frame at a time."""
REFUSED = f"Figure 2's supplement: panel {LETTERS['outside_distance']} is not drawn"
REFUSED_FOUR = f"Figure 2's supplement: panel {LETTERS['four_decoders']} is not drawn"
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
FOUR_DECODER_STEP = 0.75
"""Panel l, second axis: the x distance between the two differences at one time."""
FOUR_DECODER_GROUP = 2.0
"""Panel l, second axis: the x distance between one time and the next."""
FOUR_DECODER_YLABEL = "Held-out log score\ndifference\n(nats/spike)"
FOUR_DECODER_EVENTS_YLABEL = (
    "Held-out log score,\nfull \u2212 SWR-allowed\n(nats/spike)"
)


def read_inputs(workers: int = NO_SPIKE_WORKERS) -> dict:
    """The figure's statistics, as the events summary reads them (the No-Spike
    trace's decode frames read by ``workers`` processes).

    Returns
    -------
    dict
        In panel order: ``calibration`` (a): :func:`read_state_calibration`;
        ``no_spike`` (b) and ``event_traces`` (j): :func:`read_frame_traces`;
        ``probability`` (c): ``nonlocal_probability.animal_probability_distribution``
        of the stored counts; ``distance`` (d):
        ``quality_metrics.animal_distance_by_local_probability`` of the
        decoder-quality run's table (each animal's values and the statistics);
        ``sweep`` (e): :func:`read_rule_sweep`; ``timing`` (f): the selected
        events' ``boundary_timing_statistics``; ``mua`` (g):
        :func:`read_mua_counts`; ``split`` (h): by column of
        ``heldout_gain_panels.SPLIT_LABELS``, ``holdout_statistics`` of the
        events of the held-out tetrode run that start inside a well
        visit, indexed by group (a run not held out in folds of whole trials is
        refused, ``heldout_runs.read_tetrode_holdout``);
        ``outside`` (i): ``outside_swr_statistics``
        over the held-out event run, indexed by family and measure; ``parts``
        (i): ``outside_swr_subset_statistics`` of the margins over its time
        sums, indexed by subset, threshold, family and measure; ``hpd``
        (k): the selected events' ``hpd_distribution``;
        ``four_decoders`` (l): ``swr_allowed_statistics`` over the held-out
        event run made from the run with the fourth decoder.

    Raises
    ------
    ValueError
        If a held-out event run, or what it was made from, is not current
        (``reported.events.HELDOUT_RESTRICTION_RERUN`` names what to make again).
    """
    inputs = default_inputs()
    table = read_swr_table(inputs.event_table)
    epochs = event_table_epochs(table, analyzed_epoch_ids())
    events = select_visit_context(table)
    # panel h: the held-out tetrode run over the selected events, as the events
    # summary reads it (refused when not held out in folds of whole trials)
    holdout = window_scores(
        read_tetrode_holdout(
            inputs.tetrode_holdout,
            epochs,
            events["event_id"],
            rel(inputs.tetrode_holdout),
        )
    )
    event_table_sha256 = sha256_file(inputs.event_table)
    heldout = read_heldout_restriction(
        inputs.nonlocal_restriction_heldout_events,
        inputs.nonlocal_restriction,
        inputs.bin_facts,
        epochs,
        event_table_sha256,
    )
    refusal = heldout_restriction_refusal(
        heldout, restriction_inputs_note(inputs.nonlocal_restriction, epochs)
    )
    if refusal is not None:
        cause, rerun = refusal
        raise ValueError(f"{REFUSED}: {cause}. Rerun {rerun}.")
    # panel l: the run with the fourth decoder, read and refused as above
    four = read_heldout_restriction(
        inputs.nonlocal_restriction_swr_allowed_heldout_events,
        inputs.nonlocal_restriction_swr_allowed,
        inputs.bin_facts,
        epochs,
        event_table_sha256,
    )
    refusal = heldout_restriction_refusal(
        four, restriction_inputs_note(inputs.nonlocal_restriction_swr_allowed, epochs)
    )
    if refusal is not None:
        cause, rerun = refusal
        raise ValueError(f"{REFUSED_FOUR}: {cause}. Rerun {rerun}.")
    return {
        "calibration": read_state_calibration(inputs.state_calibration_in_visits),
        "probability": read_nonlocal_probability(inputs.nonlocal_probability),
        "distance": read_decoded_distance(inputs.decoder_quality),
        "sweep": read_rule_sweep(inputs.event_rule_sweep, inputs.event_table),
        "mua": read_mua_counts(inputs.nonlocal_by_signal),
        "mua_states": read_mua_states(inputs.state_by_signal),
        "hpd": hpd_distribution(events),
        "timing": boundary_timing_statistics(boundary_offsets(events), events),
        "split": {
            column: holdout_statistics(holdout, "event", column)
            for column in SPLIT_LABELS
        },
        "outside": outside_swr_statistics(
            heldout.time_sums, heldout.bin_s, heldout.events
        ),
        "parts": outside_swr_subset_statistics(
            heldout.time_sums, heldout.bin_s, subsets=("beyond_swr_margin",)
        ).set_index(["subset", "threshold_s", "family", "measure"]),
        "four_decoders": swr_allowed_statistics(
            four.events, four.time_sums, four.bin_s
        ),
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


def read_nonlocal_probability(folder: Path) -> pd.DataFrame:
    """Panel c's distribution: ``nonlocal_probability.animal_probability_distribution``
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
    """Panel d's lines: ``quality_metrics.animal_distance_by_local_probability``
    of the decoder-quality run's table (``calibration.parquet``), as the events
    summary reads it."""
    return animal_distance_by_local_probability(
        pd.read_parquet(folder / "calibration.parquet")
    )


def read_mua_states(folder: Path) -> pd.DataFrame:
    """The MUA panel's top strip: ``state_by_signal.state_distribution`` of the
    stored counts by MUA and decoder state, refused as :func:`read_mua_counts`
    refuses its counts."""
    try:
        counts = read_checked(folder, "counts by signal and state")
    except ValueError as error:
        raise ValueError(
            f"Figure 2's supplement: panel {LETTERS['nonlocal_mua']} is not drawn: "
            f"{error}. Rerun scripts/event_properties/measure_state_by_signal.py."
        ) from error
    return state_distribution(counts, "mua")


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
    try:
        counts = read_checked(folder, "counts by signal")
    except ValueError as error:
        raise ValueError(
            f"Figure 2's supplement: panel {LETTERS['nonlocal_mua']} is not drawn: "
            f"{error}. Rerun scripts/event_properties/measure_nonlocal_by_signal.py."
        ) from error
    return animal_nonlocal_counts(counts, "mua")


def _epoch_keys(path: Path, events: pd.DataFrame | None) -> pd.DataFrame | None:
    """The selected events' keys (``animal``, ``session_date``, ``epoch`` and
    ``non_local_event``) of the epoch of a decode frame, so that each epoch's
    task carries its own rows only (None without ``events``)."""
    if events is None:
        return None
    key = parse_epoch_identifier(path.name.removesuffix(data_over_time_suffix()))
    mine = (
        (events["animal"].astype(str) == key.animal)
        & (events["session_date"].astype(np.int64) == key.date)
        & (events["epoch"].astype(str) == key.epoch_name)
    )
    return events.loc[mine, ["animal", "session_date", "epoch", "non_local_event"]]


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
    keys = [_epoch_keys(path, events) for path in paths]
    if workers == 1:
        epochs = list(map(_frame_epoch, paths, keys))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            epochs = list(pool.map(_frame_epoch, paths, keys))
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


def draw_boundary_timing(ax_cases: Axes, ax: Axes, timing: pd.DataFrame) -> None:
    """Panel f: each SWR-overlapping event against its matched SWR, one row per
    case of :data:`TIMING_CASES` (the largest share first). At the left the
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


def draw_left_out_split(ax: Axes, statistics: dict) -> None:
    """Panel h: Figure 2j's first comparison, the left-out tetrode's better
    fit to the decoded place than to the animal's place (the gain over local
    coding per spike), for no-SWR and SWR events, over all bins, the bins with
    a spike and the silent bins (``statistics``: :func:`read_inputs`'
    ``split``), drawn by ``heldout_gain_panels.draw_split_gain``; the dashed
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
    ax.set_xticks(xs, [OUTSIDE_LABEL, *names])
    # the parts' label, centered under their tick labels
    ax.annotate(
        DISTANCE_LABEL, (float(np.mean(xs[1:])), 0), xycoords=("data", "axes fraction"),
        xytext=(0, -PARTS_LABEL_DROP), textcoords="offset points", ha="center",
        va="top",
        fontsize=SMALL_SIZE, annotation_clip=False,
    )  # fmt: skip


def draw_nonlocal_mua(
    ax_spread: Axes, ax_share: Axes, counts: pd.DataFrame, states: pd.DataFrame
) -> None:
    """The MUA panel, drawn as Figure 2i draws ripple power
    (``signal_share_panels.draw_share_by_signal``): where confident Local and
    confident non-local time sit in MUA (top; ``states``: :func:`read_inputs`'
    ``mua_states``) and the share of immobile bins decoded non-local at each
    MUA level (bottom), with the MUA level of the reported values and the
    share of non-local time below it (``nonlocal_below_threshold_statistics``,
    the events summary's value; ``counts``: :func:`read_inputs`' ``mua``)."""
    level = DETECTION_LEVELS["mua"]
    below = nonlocal_below_threshold_statistics(counts, level)
    draw_share_by_signal(
        ax_spread, ax_share, counts, threshold=level,
        threshold_label=f"{MUA_LEVEL_Z:g} z", threshold_color=REFERENCE_COLOR,
        xlabel="MUA (z)",
        below_note=f"{format_number(below['fraction_animal_mean'], 'pctnum')}% "
        f"below\n{MUA_LEVEL_Z:g} z", states=states,
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
    ends = []
    for k, (state, color) in enumerate(CALIBRATION_COLORS.items()):
        mine = bins[bins["state"] == state]
        estimate = mine["animal_estimate"].to_numpy()
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


def draw_nonlocal_probability(ax: Axes, distribution: pd.DataFrame) -> None:
    """The non-local probability panel: how the immobile time inside the well
    visits is spread over the decoder's non-local probability, the percent of
    the time per bin of 0.02 on a log axis, a thin line per animal and their
    mean in bold, with the event rule's cut (0.8) dotted and named "used"
    (``distribution``: :func:`read_inputs`' ``probability``,
    ``nonlocal_probability.animal_probability_distribution``)."""
    by_animal = distribution.pivot(index="bin_low", columns="animal", values="percent")
    high = distribution.groupby("bin_low")["bin_high"].first()
    center = (by_animal.index.to_numpy() + high.to_numpy()) / 2
    ax.plot(center, by_animal.to_numpy(), color=ANIMAL_LINE_COLOR, lw=0.5)
    ax.plot(center, by_animal.mean(axis=1).to_numpy(), color="black", lw=1.2)
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
    ``distance``, ``quality_metrics.animal_distance_by_local_probability``)."""
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
    name_groups(ax, 0.84, 0.98, 0.17, "right")
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


def draw_four_decoders(axes: Sequence[Axes], statistics: SwrAllowedStatistics) -> None:
    """Panel l: three axes with the fourth decoder (``statistics``:
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
    draw_no_spike_mua(axes["no_spike_mua"][0], statistics["no_spike"])
    draw_event_traces(axes["event_traces"], statistics["event_traces"])
    draw_boundary_timing(*axes["boundary_timing"], statistics["timing"])
    draw_left_out_split(axes["left_out_split"][0], statistics["split"])
    draw_hpd(axes["hpd"][0], statistics["hpd"])
    draw_outside_swr_series(
        axes["outside_distance"][0], statistics["outside"], statistics["parts"]
    )
    draw_nonlocal_mua(
        *axes["nonlocal_mua"], statistics["mua"], statistics["mua_states"]
    )
    draw_rule_sweep(axes["rule_sweep"], statistics["sweep"])
    draw_state_calibration(axes["state_calibration"][0], statistics["calibration"])
    draw_nonlocal_probability(
        axes["nonlocal_probability"][0], statistics["probability"]
    )
    draw_decoded_distance(axes["decoded_distance"][0], *statistics["distance"])
    draw_four_decoders(axes["four_decoders"], statistics["four_decoders"])
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
