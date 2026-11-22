#!/usr/bin/env python
"""Generate Figure 2: non-local events and SWRs are partly overlapping inventories.

The decoder of Figure 1 applied to the recordings, and how its non-local events
relate to sharp-wave ripples (SWRs, Kay's detector, the SWR source):

    a — The decode over a run and the rest after it, with the 2D track. Lanes
        mark the immobile (analyzed) time, the Local / non-local / No-Spike
        stretches and the SWRs; the counted non-local events are shaded.
    b — Six examples: a non-local event inside an SWR, one that starts before
        its SWR, one that ends after it, two with no SWR (a trajectory and a
        stationary event), and an SWR whose content is Local.
    c — The SWRs by content (bars pooled over SWRs, dots one per animal): with a
        non-local event, mostly Local, or other. The decoder finds the replay
        SWRs are known for.
    d — The non-local events by SWR overlap (bars and animal dots as in c): most
        events have no SWR.
    e — For each SWR definition, MUA and the Widloski detector, c's and d's
        questions: the share of its own events that overlap a non-local event,
        and the share of non-local events its events overlap.
    f — Each SWR-overlapping event's start and end minus its SWR's: the percent
        of event-SWR pairs per 5 ms on a log axis, bars the animals' mean and
        outlines the animals.
    g — Around non-local event starts, SWR vs no-SWR: ripple power, MUA, theta
        power and head speed, each the median and interquartile range over
        events, with the epochs' level between events, theta's running level and
        the immobility threshold. Both kinds of event come with a rise in MUA
        that is not due to movement or theta: the animal is still and theta
        stays at its level between events, far below running.
    h — Tetrode participation (the share of tetrodes with a spike) of each
        non-local event against a Local-rest window of its length, one line
        per animal, for no-SWR and SWR events: the events are driven by
        several tetrodes, not one (``event_evidence.participation``;
        ``Processed-Data/event_participation``).
    i — Where the non-local time sits in ripple power, one line per animal:
        the immobile time bins decoded non-local spread over ripple power, with
        the share of them below the SWR threshold (top), and the share of
        immobile time bins decoded non-local at each power, which has no step
        at the threshold (bottom)
        (``nonlocal_by_signal.rest_signal_counts``;
        ``Processed-Data/nonlocal_by_signal``; drawn by
        ``visualization.signal_share_panels.draw_share_by_signal``, which
        Figure 2's supplement draws MUA with).
    j — Three decoders that differ only in when the non-local states may be
        occupied, scored by every tetrode's one-step-ahead predictive score of
        each bin, the whole decoder (place fields, state transitions, initial
        conditions) fit on folds of whole trials and each fold scored under the
        fit that did not see it: the local-only decoder (Local and No-Spike
        states only), the SWR-only decoder (non-local states only inside SWRs)
        and the full decoder. Every value is over the immobile bins inside a
        well visit, which runs from the animal's first nose-poke at a reward
        well to its last poke-out there. Each value is per epoch, summed over
        its bins and divided by the spikes in them; a dot per animal is the
        median over its epochs.
        First axis: the SWR-only and the full decoder's gain over the
        local-only decoder over every immobile bin, a dot per animal (dark
        gray, the Non-Local state's green), the animals' mean with its 95 %
        interval and a dashed line at zero, the local-only decoder
        (``restriction_totals.immobile_gain_statistics``). Second axis, the
        full minus the SWR-only decoder: over every immobile bin ("All
        immobility", each epoch's paired difference, the same statistic) and
        over the immobile bins outside SWRs ("Outside SWRs", the main result,
        its label in bold; ``restriction_totals.outside_swr_gain_statistics``),
        a dot per animal in dark gray, the mean with its interval, a dashed
        line at zero with which way the full decoder is better. The bins
        outside SWRs are chosen by speed, ripple detection and the well visits
        alone, not by the decoder: the test. An animal's dots are spread
        sideways so that each shows. Figure 2's supplement
        (``generate_supp_figure02.py``) looks at the gain outside SWRs more
        closely: away from the SWRs, and where it sits in the events detected
        from the held-out decode (the gain at well-sampled places is a
        reported value for the text). All from
        ``Processed-Data/event_nonlocal_restriction_heldout_events`` (its time
        sums and events) by ``restriction_totals.restriction_figure_statistics``,
        which the events summary's values share.

Panels d and f are drawn from the canonical event table by the functions the
events summary uses (``event_statistics.overlap_class_statistics``,
``boundary_offsets``); c and e from ``nonlocal_comparison`` on the event
features run the summary reads; g's between-events and running lines from
``event_statistics.reference_medians`` of the brain-state reference; i from
the stored counts by the functions the events summary uses
(``nonlocal_by_signal.nonlocal_counts``, ``event_statistics.animal_signal_counts``,
``nonlocal_below_threshold_statistics``); j from the held-out event run's
events and time sums as the events summary reads them
(``reported.events.read_heldout_restriction``: the events detected from the
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
import warnings
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.backend_bases import RendererBase
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec, SubplotSpec
from matplotlib.ticker import NullLocator
from matplotlib.transforms import Bbox
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
from continuum_swr_replay.analysis.event_evidence.participation import (
    participation_statistics,
)
from continuum_swr_replay.analysis.event_evidence.restriction_totals import (
    RESTRICTION_FIGURE_DECODERS,
    RESTRICTION_FIGURE_DIFFERENCE,
    RESTRICTION_FIGURE_FAMILY,
    RESTRICTION_FIGURE_OUTSIDE_SWR,
    restriction_figure_statistics,
)
from continuum_swr_replay.analysis.event_properties.class_or_continuum import (
    KAY_ZSCORE_THRESHOLD,
)
from continuum_swr_replay.analysis.event_properties.event_statistics import (
    animal_signal_counts,
    boundary_offsets,
    nonlocal_below_threshold_statistics,
    overlap_class_animal_shares,
    overlap_class_statistics,
    reference_medians,
)
from continuum_swr_replay.analysis.event_properties.nonlocal_by_signal import (
    nonlocal_counts,
)
from continuum_swr_replay.analysis.event_properties.triggered_analysis import (
    compute_triggered_average_curves,
    triggering_events_in,
)
from continuum_swr_replay.analysis.event_tables.prepare_data import (
    behavioral_context_cache_path,
    time_mask_staleness,
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
    BRAIN_STATE_REFERENCE_DIR,
    EVENT_BIN_FACTS_DIR,
    EVENT_NONLOCAL_RESTRICTION_DIR,
    EVENT_NONLOCAL_RESTRICTION_HELDOUT_EVENTS_DIR,
    EVENT_PARTICIPATION_DIR,
    NONLOCAL_BY_SIGNAL_DIR,
    PARTICIPATION_NAME,
    event_features_root,
    processed_data_dir,
    replay_content_dir,
    table_path,
)
from continuum_swr_replay.reported.events import (
    checked_reference,
    heldout_restriction_refusal,
    read_heldout_restriction,
    restriction_inputs_note,
)
from continuum_swr_replay.reported.formatting import format_number
from continuum_swr_replay.runs.io import (
    ok_epochs,
    read_complete_run,
    read_manifest,
    sha256_file,
)
from continuum_swr_replay.runs.staleness import read_output_manifest
from continuum_swr_replay.visualization.figure_utilities import save_figure
from continuum_swr_replay.visualization.heldout_gain_panels import (
    DIFFERENCE_YLABEL,
    GAIN_FLOOR,
    NEUTRAL_DOTS,
    REFERENCE_COLOR,
    animal_values,
    draw_animal_gain,
    full_better_cue,
    label_gain_axis,
)
from continuum_swr_replay.visualization.inventory_examples import detector_color
from continuum_swr_replay.visualization.palettes import COLORS, readable_text_color
from continuum_swr_replay.visualization.plots import (
    plot_linear_track_graph_strip,
    plot_ripple_consensus_trace_zscore,
    plot_spike_times,
    plot_track_graph_2d,
)
from continuum_swr_replay.visualization.signal_share_panels import (
    draw_share_by_signal,
)
from continuum_swr_replay.visualization.simulated_run_panels import (
    ANIMAL_POS_COLOR,
    LABEL_SIZE,
    LINE_WIDTH_LIMITS_PT,
    PANEL_LETTER_SIZE,
    SMALL_SIZE,
    check_journal_limits,
)

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
FIGURE02_CACHE_VERSION = 15

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%d-%b-%y %H:%M:%S"
)
figure_utilities.set_figure_defaults(context="paper")
plt.rcParams.update(
    {
        "font.size": SMALL_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "axes.titlesize": LABEL_SIZE,
        "xtick.labelsize": SMALL_SIZE,
        "ytick.labelsize": SMALL_SIZE,
        "legend.fontsize": SMALL_SIZE,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.constrained_layout.use": False,
    }
)
warnings.filterwarnings("ignore")

# Inches: saved with 0.1 in margins, about 17.3 cm (Cell Press full width).
WIDTH = 6.5
HEIGHT = 7.72

C_LOCAL = COLORS["Local"]
C_NOSPIKE = COLORS["No-Spike"]
C_OTHER = "white"
"""Panel c's 'Other' SWRs: an outlined bar, no state's color (they are mostly
mixed-state, not No-Spike)."""
C_REFERENCE = REFERENCE_COLOR
"""A reference level (panel j's zeros, panel g's between-events level): a neutral
gray, not the No-Spike state's."""
C_NL = COLORS["Non-Local"]
C_NO_SWR = COLORS["No SWR"]
"""The no-SWR events as a class (d, g, h): not the Non-Local state's green,
which marks the state and every non-local event in a and b and the SWRs with
non-local content in c."""
C_SWR = COLORS["SWR"]
C_SWR_PARTIAL = COLORS["SWR partial"]
C_PAIR = "#999999"
"""A line joining one animal's two values (panel h)."""
C_QUIET = COLORS["All Events"]
CLASS_ORDER = ("swr_contained", "swr_partial", "non_swr")
CLASS_COLORS = {
    "swr_contained": C_SWR,
    "swr_partial": C_SWR_PARTIAL,
    "non_swr": C_NO_SWR,
}
CLASS_LABELS = {
    "swr_contained": "Inside SWR",
    "swr_partial": "Partly in SWR",
    "non_swr": "No SWR",
}
# The track plotters' line widths, inside the journal's limits.
TRACK_LINEWIDTH = LINE_WIDTH_LIMITS_PT[0]
TRACK_EDGE_LINEWIDTH = LINE_WIDTH_LIMITS_PT[1]

# Panel a: a run and the rest after it in the example session (start, seconds).
OVERVIEW_START_S = 1617036555.9
OVERVIEW_DURATION_S = 12.6
# Panel b: half the window around each example's center, seconds.
EXAMPLE_HALF_WINDOW_S = 0.2
# Panel b's examples, chosen by eye from contact sheets of the example session's
# events with peak non-local probability >= 0.95, no other event and no other
# SWR within the window, and a sharp posterior (small HPD coverage): each
# title's event_id and the overlap subclass it is drawn as, which
# ``example_windows`` checks.
EXAMPLES = {
    "Inside SWR": (2185516843327500879, "swr_contains_event"),
    "Starts before SWR": (63427667006281341, "single_swr_partial"),
    "Ends after SWR": (3628740333881355820, "single_swr_partial"),
    "No SWR, trajectory": (5216998177801911428, "no_overlap"),
    "No SWR, stationary": (8826983562067101773, "no_overlap"),
}
# Panel b's partial examples: the one side on which the event runs past its SWR.
PARTIAL_SIDE = {"Starts before SWR": "start", "Ends after SWR": "end"}
# The SWR with Local content: its start in the SWR source, seconds. No non-local
# event overlaps it and the decode is Local through it.
LOCAL_SWR_START_S = 1617036373.982971
# Panel g: the triggered averages' window (``compute_triggered_average_curves``'s
# window_sec; the curves span half of it either side of the start), seconds.
TRIGGERED_WINDOW_S = 0.3
TRIGGERED_SIGNALS = ["SWR_zscore", "MUA_zscore", "theta_power_zscore", "head_speed"]
# Panel e: the detectors, in the order drawn, and their labels.
DETECTOR_LABELS = dict(
    zip(
        nonlocal_comparison.FIGURE_DETECTORS,
        ("Kay (SWR)", "Karlsson", "Yu", "Zugaro", "Carey", "MUA", "Widloski"),
        strict=True,
    )
)
"""Panel e's name of each detector it draws
(``nonlocal_comparison.FIGURE_DETECTORS``, in its order)."""


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


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


def _decode_files_fingerprint(
    nwb_file_name: str, epoch_name: str, data_dir: Path
) -> str:
    """SHA-256 over the decode files ``_compute_figure_inputs`` reads.

    The files are the analyzed epochs' ``data_over_time`` (as
    ``load_data_over_time`` finds them) and the example session's HPC
    ``results.nc`` and ``detector.pkl`` (as ``load_results`` names them), all
    in `data_dir`. Each contributes its path, size and ``st_mtime_ns``, or
    ``missing``, so the fingerprint is a stat per file, not a read.
    """
    example = decode_paths(
        format_epoch_identifier(nwb_file_name, epoch_name, "HPC"), data_dir
    )
    files = [
        *analyzed_decode_files(data_dir, data_over_time_suffix()),
        example.results,
        example.detector,
    ]
    return _stat_fingerprint(files)


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


def _event_table_path(data_dir: Path) -> Path:
    """The canonical event table the figure reads, in ``data_dir``."""
    return table_path(
        "event_table", data_dir / replay_content_dir().relative_to(processed_data_dir())
    )


def _comparison_fingerprint(data_dir: Path) -> str:
    """A stat fingerprint of the detector comparison's files, of the canonical
    event table read from ``data_dir`` (the selected events) and of the session
    cache's entries (the well visits); a run without a manifest is
    fingerprinted as missing (``detector_comparison`` then raises)."""
    run = event_features_root() / EVENT_FEATURES_RUN
    if not (run / "manifest.json").exists():
        return _stat_fingerprint([run / "manifest.json", _event_table_path(data_dir)])
    return _stat_fingerprint(
        [
            *nonlocal_comparison.comparison_files(run),
            _event_table_path(data_dir),
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
        TRIGGERED_SIGNALS,
        list(DETECTOR_LABELS),
    )
    return hashlib.sha256(repr(settings).encode()).hexdigest()


def _figure02_cache_path(nwb_file_name: str, epoch_name: str, data_dir: Path) -> Path:
    """Path for the cached Figure-2 render inputs (joblib bundle).

    A single joblib bundle is used rather than netCDF because the bundle mixes
    a session ``data`` dict, an xarray ``results`` Dataset whose ``state_bins``
    MultiIndex netCDF cannot serialize, the fitted detector and environment
    objects, and small DataFrames; joblib (pickle) preserves them all exactly.
    """
    cohort = hashlib.sha256("\n".join(analyzed_epoch_ids()).encode()).hexdigest()
    decodes = _decode_files_fingerprint(nwb_file_name, epoch_name, data_dir)
    comparison = _comparison_fingerprint(data_dir)
    return (
        data_dir
        / "intermediates"
        / f"{session_name(nwb_file_name, epoch_name)}_figure02_v{FIGURE02_CACHE_VERSION}_"
        f"{swr_source_name()}_register{cohort[:12]}_decodes{decodes[:12]}"
        f"_features{EVENT_FEATURES_RUN}_comparison{comparison[:12]}"
        f"_settings{_settings_fingerprint()[:12]}_cache.joblib"
    )


def _compute_figure_inputs(
    nwb_file_name: str, epoch_name: str, data_dir: Path, events: pd.DataFrame
) -> dict:
    """Run the expensive loads and derivations; return the cached render inputs.

    The example session's ``data``/``results``/``detector``/``env`` (panels a
    and b), the triggered-average curves (i) over the frames' rows of
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
        nwb_file_name, epoch_name, decoding_brain_area="HPC", root=data_dir
    )
    logging.info("Loading the aggregated decode frames (panel g)...")
    data_over_time = load_data_over_time(data_dir)
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
        TRIGGERED_SIGNALS,
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


# ---------------------------------------------------------------------------
# Drawing
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


def draw_overview(
    fig: Figure,
    spec: SubplotSpec,
    inputs: dict,
    session_events: pd.DataFrame,
    max_pos: float,
) -> tuple[Axes, Axes]:
    """Panel a: the 2D track, then the decode over a run and the rest after it."""
    data, results, env = inputs["data"], inputs["results"], inputs["env"]
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
        5, 1, height_ratios=[1.5, 1.05, 0.42, 0.42, 0.42], hspace=0.14
    )
    ax_pos, ax_lane, ax_rip, ax_st, ax_sp = (fig.add_subplot(rows[i]) for i in range(5))
    draw_posterior(ax_pos, data, results, env, sl, start, max_pos)
    add_track_strip(ax_pos, data, max_pos)
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
    ax_rip.set_yticks([0, 5])
    draw_state_stack(ax_st, results, sl, start)
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
    return ax_track, ax_pos


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


def fit_axes_in_row(
    fig: Figure,
    renderer: Any,
    axes: Sequence[Axes],
    left: float,
    right: float,
    weights: Sequence[float],
) -> None:
    """Place a panel's axes side by side so that what they draw runs from
    ``left`` to ``right`` (figure coordinates), with :data:`DECODER_AXES_GAP`
    between one axes' drawing and the next, and their frames' widths in
    proportion to ``weights``.

    Each axes' y axis (its label and tick labels) and the texts it writes
    beside its frame, past the right edge (a reference's name, direct labels),
    take what they need,
    measured on the drawn figure, and the frames share the rest, so no label
    runs into the next axes. The x tick labels, rotated to hang down and left
    below the frames, do not count: parallel labels do not cross, and the
    first axes' hang into the figure's margin as panel c's labels do. Nor do
    the titles and texts above the frames, which may reach over the gaps;
    whether they meet is checked on the image. Measured again after each move
    (:data:`FIT_PASSES`), as the extents change with the frames.
    """
    for _ in range(FIT_PASSES):
        fig.canvas.draw()
        to_figure = fig.transFigure.inverted()
        pads = []
        for ax in axes:
            y_axis = ax.yaxis.get_tightbbox(renderer).transformed(to_figure)
            box = ax.get_position()
            beside = [
                text.get_window_extent(renderer).transformed(to_figure)
                for text in ax.texts
            ]
            ink_right = max([box.x1] + [b.x1 for b in beside if b.y0 < box.y1])
            pads.append((box.x0 - y_axis.x0, ink_right - box.x1))
        free = (
            right
            - left
            - DECODER_AXES_GAP * (len(axes) - 1)
            - sum(before + after for before, after in pads)
        )
        x = left
        for ax, (before, after), weight in zip(axes, pads, weights, strict=True):
            width = free * weight / sum(weights)
            box = ax.get_position()
            ax.set_position((x + before, box.y0, width, box.height))
            x += before + width + after + DECODER_AXES_GAP


def fit_track_to_height(fig: Figure, ax_track: Axes, left: float) -> None:
    """Size the track's equal-aspect axes to fill its row's height, at ``left``."""
    box = ax_track.get_position(original=True)
    x0, x1 = ax_track.get_xlim()
    y0, y1 = ax_track.get_ylim()
    fig_w, fig_h = fig.get_size_inches()
    width = box.height * fig_h / fig_w * abs(x1 - x0) / abs(y1 - y0)
    ax_track.set_position([left, box.y0, width, box.height])


def draw_examples(
    fig: Figure, spec: SubplotSpec, inputs: dict, windows: dict, max_pos: float
) -> Axes:
    """Panel b: one column per example (ripple, raster, decode, states)."""
    data, results, env, detector = (
        inputs["data"],
        inputs["results"],
        inputs["env"],
        inputs["detector"],
    )
    n = len(windows)
    gs = spec.subgridspec(
        4, n, hspace=0.08, wspace=0.14, height_ratios=[0.5, 0.65, 1.2, 0.5]
    )
    first = None
    for j, (title, (start, end)) in enumerate(windows.items()):
        ax_rip, ax_raster, ax_post, ax_states = (
            fig.add_subplot(gs[r, j]) for r in range(4)
        )
        center = (start + end) / 2
        sl = slice(center - EXAMPLE_HALF_WINDOW_S, center + EXAMPLE_HALF_WINDOW_S)
        plot_ripple_consensus_trace_zscore(data, sl, ax_rip)
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
                ("Ripple\n(z)", "Tetrode", "Decoded\nposition (cm)", "State\nprob."),
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
    """Classes as horizontal bars, top to bottom: the pooled share as a bar with
    its percent in a column past the axis's 100, each animal's share as a dot.

    ``rows`` are (label, pooled share, the animals' shares, color, bold label).
    """
    n_rows = len(rows)
    for i, (_, pooled, animals, color, _) in enumerate(rows):
        y = n_rows - 1 - i
        ax.barh(
            y, 100 * pooled, color=color, height=0.65, linewidth=0.5,
            edgecolor="#777777" if color == C_OTHER else "none",
        )  # fmt: skip
        ax.scatter(
            100 * animals.to_numpy(), y + np.linspace(-0.18, 0.18, len(animals)),
            s=3, color="k", lw=0, zorder=3,
        )  # fmt: skip
        ax.text(
            1.0, y, f"{format_number(pooled, 'pctnum')}%",
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


def draw_events_bars(ax: Axes, classes: dict, animals: pd.DataFrame) -> None:
    """Panel d: the non-local events by SWR overlap.

    Bars pool the events (``overlap_class_statistics``); dots are the animals
    (``overlap_class_animal_shares``).
    """
    rows = [
        (
            CLASS_LABELS[c],
            classes[f"{c}_fraction"],
            animals[c],
            CLASS_COLORS[c],
            c == "non_swr",
        )
        for c in CLASS_ORDER
    ]
    draw_class_bars(
        ax, rows, f"Non-local events (n = {classes['n_events']:,})", "% events"
    )


def draw_swr_bars(ax: Axes, kay: pd.Series, animals: pd.DataFrame) -> None:
    """Panel c: the SWRs by content.

    Bars pool Kay's SWRs (``other_detector_statistics``); dots are the animals
    (``content_class_animal_shares``).
    """
    no_nonlocal = kay["no_nonlocal_fraction"]
    mostly_local = kay["mostly_local_fraction"]
    rows = [
        ("With non-local", 1 - no_nonlocal, animals["with_nonlocal"], C_NL, True),
        ("Mostly Local", mostly_local, animals["mostly_local"], C_LOCAL, False),
        ("Other", no_nonlocal - mostly_local, animals["other"], C_OTHER, False),
    ]
    draw_class_bars(ax, rows, f"SWRs (n = {int(kay['n_events']):,})", "% SWRs")


REST_EVENT_X = {"no SWR": (0.0, 1.0), "SWR": (2.4, 3.4)}
"""Panel h: each group's Local-rest and event positions on the x axis."""

DECODER_X = dict(zip(RESTRICTION_FIGURE_DECODERS, (0.0, 1.0), strict=True))
"""Panel j, first axis: each decoder's measure and its position on the x axis."""
DECODER_LABELS = dict(
    zip(RESTRICTION_FIGURE_DECODERS, ("SWR-\nonly", "Full"), strict=True)
)
DECODER_COLORS = dict(
    zip(RESTRICTION_FIGURE_DECODERS, (NEUTRAL_DOTS, C_NL), strict=True)
)
"""The SWR-only decoder in a neutral dark gray, the full decoder in the
Non-Local state's green."""
DECODERS_YLABEL = "Held-out log score\nover local-only\ndecoder (nats/spike)"
DECODERS_REFERENCE = "local-only"
DIFFERENCE_X = {"all": 0.0, "outside_swr": 1.0}
"""Panel j, second axis: the full minus the SWR-only decoder over all the
immobile time, then over the immobile time outside SWRs."""
DIFFERENCE_LABELS = {"all": "All\nimmobility", "outside_swr": "Outside\nSWRs"}
"""Horizontal, on two lines; the main result, outside SWRs, in bold (as panels
c and d mark their key class)."""


def restriction_refusal(cause: str, rerun: str) -> str:
    """Why the decoder comparison is not drawn: what is not current (``cause``)
    and the runners to make again (``rerun``), from
    ``reported.events.heldout_restriction_refusal``."""
    letter = ROW4_LETTERS["decoder_comparison"]
    return (
        f"the decoder comparison (panel {letter}) is not drawn: {cause}. Rerun {rerun}."
    )


ROW4_PANELS = ("traces", "participation", "ripple_power", "decoder_comparison")
"""Row 4's panels by what they draw, left to right: the traces around event
starts, tetrode participation, where the non-local time sits in ripple power,
and the decoder comparison. The one place the row's order is set: the grid,
the letters (:data:`ROW4_LETTERS`), the decoder comparison's bounds and the
letter columns follow from it (the docstrings that name a panel by letter are
found by grepping for ``ROW4_LETTERS`` and "panel g" to "panel j")."""
ROW4_LETTERS = dict(zip(ROW4_PANELS, "ghij", strict=True))
"""Each row-4 panel's letter, by its position."""
ROW4_WIDTHS = {
    "traces": 1.0,
    "participation": 0.7,
    "ripple_power": 0.9,
    "decoder_comparison": 2.8,
}
"""Each row-4 panel's cell width, in proportion; the decoder comparison's two
axes are then placed in its space by :func:`fit_axes_in_row`."""
ROW4_WSPACE = 0.6
DECODER_AXES_WIDTHS = (0.7, 1.1)
"""The decoder comparison's two axes' frame widths, in proportion: the decoders
over the local-only decoder, the full minus the SWR-only decoder
(:func:`fit_axes_in_row`)."""
DECODER_AXES_WSPACE = 1.0
"""The grid's first guess; :func:`fit_axes_in_row` then places the axes."""
FIT_PASSES = 10
"""The passes of :func:`fit_axes_in_row`: what overhangs a frame shrinks as the
frame grows, so the widths settle over a few passes."""
DECODER_AXES_GAP = 0.012
"""The decoder comparison: the space between one axes' drawing and the next
(figure width)."""
PANEL_GAP = 0.012
"""The space between the decoder comparison's drawing and its neighbors'
(figure width)."""
ROW4_RIGHT = 0.995
"""Where the decoder comparison's drawing may end when it is the row's last
panel (figure width): the figure is saved cropped to its ink, so the edge is
the page's."""
LETTER_COLUMNS = (
    "abc" + ROW4_LETTERS[ROW4_PANELS[0]],
    "d",
    "e",
    "f",
    *(ROW4_LETTERS[panel] for panel in ROW4_PANELS[1:]),
)
"""The panel letters that share a left edge (convention 10): a, b, c and row
4's first panel at the figure's left. Row 4's other panels sit under no letter
of row 3, so each is placed by itself."""


def draw_rest_to_event(
    ax: Axes, rest: pd.Series, event: pd.Series, group: str, color: str
) -> None:
    """One line per animal from its Local-rest value to its event value."""
    x = REST_EVENT_X[group]
    for animal in event.index.intersection(rest.index):  # paired by animal
        ax.plot(x, (rest[animal], event[animal]), color=C_PAIR, lw=0.5, zorder=1)
    ax.scatter(np.full(rest.size, x[0]), rest, s=6, color=C_LOCAL, lw=0, zorder=2)
    ax.scatter(np.full(event.size, x[1]), event, s=6, color=color, lw=0, zorder=2)


def draw_participation(ax: Axes, statistics: pd.DataFrame) -> None:
    """Panel h: tetrode participation, Local rest vs events.

    One line per animal from its length-matched Local-rest controls to its
    events, for no-SWR events and for SWR events (``participation_statistics``).
    Each group's controls have its events' lengths, so the two groups' levels
    are not compared: a longer window holds a spike on more tetrodes.
    """
    for group, color in (("no SWR", C_NO_SWR), ("SWR", C_SWR)):
        rest, event = (
            animal_values(statistics.loc[(group, kind)], "participation_animals", 100)
            for kind in ("control", "event")
        )
        draw_rest_to_event(ax, rest, event, group, color)
    ax.set_ylabel("Tetrodes active (%)")
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 50, 100])
    ax.set_xlim(-0.5, 3.9)
    ticks = [x for pair in REST_EVENT_X.values() for x in pair]
    ax.set_xticks(
        ticks, ["Local rest", "No SWR", "Local rest", "SWR"], rotation=45, ha="right"
    )


def draw_decoder_gains(ax: Axes, statistics: pd.DataFrame) -> None:
    """Panel j, first axis: the SWR-only and full decoders' gains over the
    local-only decoder per spike of the immobile time inside a well visit.

    For each decoder, a dot per animal (the median over its epochs, spread
    sideways; dark gray for the SWR-only decoder, the Non-Local state's green
    for the full one) and the animals' mean with its 95 % interval, with a
    dashed line at zero, the local-only decoder
    (``restriction_totals.immobile_gain_statistics``,
    :data:`~continuum_swr_replay.analysis.event_evidence.restriction_totals.RESTRICTION_FIGURE_DECODERS`).

    Parameters
    ----------
    ax : Axes
    statistics : pd.DataFrame
        ``immobile_gain_statistics``' rows, indexed by ``family`` and
        ``measure``.
    """
    rows = {
        measure: statistics.loc[(RESTRICTION_FIGURE_FAMILY, measure)]
        for measure in RESTRICTION_FIGURE_DECODERS
    }
    for measure, row in rows.items():
        draw_animal_gain(
            ax, DECODER_X[measure], row, DECODER_COLORS[measure], spread=True
        )
    ax.axhline(0, color=C_REFERENCE, lw=0.6, ls="--", zorder=1)
    ax.set_xlim(-0.45, 1.5)
    ax.set_xticks(list(DECODER_X.values()), list(DECODER_LABELS.values()))
    top = (
        max(max(float(r["animals"].max()), r["ci_high"]) for r in rows.values()) * 1.08
    )
    label_gain_axis(ax, DECODERS_YLABEL, DECODERS_REFERENCE, floor=-GAIN_FLOOR * top)
    ax.set_ylim(top=top)


def draw_full_minus_swr(
    ax: Axes, decoders: pd.DataFrame, outside: pd.DataFrame
) -> None:
    """Panel j, second axis: the full decoder minus the SWR-only decoder per
    spike, over all the immobile time inside a well visit (the paired
    difference, ``restriction_totals.immobile_gain_statistics``'
    :data:`~continuum_swr_replay.analysis.event_evidence.restriction_totals.RESTRICTION_FIGURE_DIFFERENCE`)
    and over its bins outside SWRs
    (``restriction_totals.outside_swr_gain_statistics``'
    :data:`~continuum_swr_replay.analysis.event_evidence.restriction_totals.RESTRICTION_FIGURE_OUTSIDE_SWR`).

    For each, a dot per animal (the median over its epochs, spread sideways)
    in a neutral dark gray and the animals' mean with its 95 % interval, with
    a dashed line at zero, where the two decoders predict the spikes equally
    well, and which way the full decoder is better. The main result, outside
    SWRs, has its label in bold.

    Parameters
    ----------
    ax : Axes
    decoders : pd.DataFrame
        ``immobile_gain_statistics``' rows, indexed by ``family`` and
        ``measure``.
    outside : pd.DataFrame
        ``outside_swr_gain_statistics``' rows, indexed by ``family`` and
        ``measure``.
    """
    rows = {
        "all": decoders.loc[(RESTRICTION_FIGURE_FAMILY, RESTRICTION_FIGURE_DIFFERENCE)],
        "outside_swr": outside.loc[
            (RESTRICTION_FIGURE_FAMILY, RESTRICTION_FIGURE_OUTSIDE_SWR)
        ],
    }
    for name, row in rows.items():
        draw_animal_gain(ax, DIFFERENCE_X[name], row, NEUTRAL_DOTS, spread=True)
    ax.axhline(0, color=C_REFERENCE, lw=0.6, ls="--", zorder=1)
    ax.set_xlim(-0.45, 1.5)
    ax.set_xticks(list(DIFFERENCE_X.values()), list(DIFFERENCE_LABELS.values()))
    ax.get_xticklabels()[list(DIFFERENCE_X).index("outside_swr")].set_fontweight("bold")
    top = (
        max(max(float(r["animals"].max()), r["ci_high"]) for r in rows.values()) * 1.08
    )
    ax.set_ylim(-GAIN_FLOOR * top, top)
    ax.set_ylabel(DIFFERENCE_YLABEL)
    full_better_cue(ax)


def draw_triggered(axes: Sequence[Axes], curves: dict, rest: pd.DataFrame) -> None:
    """Panel g: ripple power, MUA, theta and head speed around event starts.

    Every row is the median over events with the interquartile band, so the
    spread over events is seen, not only the center, and the line is the
    statistic the text quotes: ripple and MUA rise at the start, theta stays at
    its level between events, far below running, and the animal is still. The
    dashed "between events" line is the epochs' median in immobile bins outside
    every non-local, SWR and MUA event (``reference_medians``).
    """
    ax_rip, ax_mua, ax_theta, ax_speed = axes
    groups = (("overlap", C_SWR, "SWR"), ("no_overlap", C_NO_SWR, "No SWR"))
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
        axes, ("Ripple (z)", "MUA (z)", "Theta (z)", "Speed (cm/s)"), strict=True
    ):
        ax.axvline(0, color="k", lw=0.5, ls=":")
        ax.set_xlim(t_ms[0], t_ms[-1])
        ax.text(
            0.03, 1.0, label, transform=ax.transAxes, ha="left", va="top",
            fontsize=SMALL_SIZE,
        )  # fmt: skip
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    # two centered rows, each narrower than the axes, so the legend stays clear
    # of the row's panel letters: the dashed line just above the axes, the two
    # event classes above it
    style = {
        "frameon": False, "loc": "lower center", "handlelength": 0.8,
        "handletextpad": 0.4, "columnspacing": 0.8, "borderaxespad": 0.1,
        "borderpad": 0.0,
    }  # fmt: skip
    reference = ax_rip.legend(
        [plt.Line2D([], [], color=C_REFERENCE, ls="--", lw=0.6)],
        ["between events"], bbox_to_anchor=(0.5, 1.0), **style,
    )  # fmt: skip
    row = 1.3 * reference.get_texts()[0].get_fontsize() / 72  # one row, inches
    axes_height = ax_rip.get_position().height * ax_rip.figure.get_figheight()
    handles, labels = ax_rip.get_legend_handles_labels()
    ax_rip.legend(
        handles, labels, ncol=2, bbox_to_anchor=(0.5, 1.0 + row / axes_height),
        **style,
    )  # fmt: skip
    ax_rip.add_artist(reference)
    ax_speed.set_xlabel("Time from event start (ms)")


def draw_detector_bars(
    ax: Axes, shares: pd.DataFrame, n_rows: int, xlabel: str, show_labels: bool
) -> None:
    """One bar per detector (pooled share) with a dot per animal, top to bottom.

    The SWR definitions other than Kay's are drawn faded.
    """
    for i, (detector, row) in enumerate(shares.iterrows()):
        y = n_rows - 1 - i
        faded = detector in SWR_DEFINITIONS and detector != "kay"
        ax.barh(
            y, 100 * row["pooled"], color=detector_color(detector), height=0.65,
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

    Bars pool all events; dots are animals (for the caption).

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


def draw_share_by_power(ax_spread: Axes, ax_share: Axes, counts: pd.DataFrame) -> None:
    """Panel i: where the non-local time sits in ripple power, drawn by
    ``signal_share_panels.draw_share_by_signal`` with the share of non-local
    time below the SWR threshold written (``nonlocal_below_threshold_statistics``,
    the events summary's value).

    Parameters
    ----------
    ax_spread, ax_share : Axes
    counts : pd.DataFrame
        ``animal_signal_counts`` of the stored counts' ripple power.
    """
    below = nonlocal_below_threshold_statistics(counts, KAY_ZSCORE_THRESHOLD)
    draw_share_by_signal(
        ax_spread, ax_share, counts, threshold=KAY_ZSCORE_THRESHOLD,
        threshold_label="SWR\nthreshold", threshold_color=C_SWR, shade_color=C_QUIET,
        xlabel="Ripple power (z)",
        below_note=f"{format_number(below['fraction_animal_mean'], 'pctnum')}% "
        "below\nthreshold",
    )  # fmt: skip


OFFSET_BINS_MS = np.arange(-100, 101, 5)
"""Panel f's bins, ms: event bound minus SWR bound."""


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


def panel_extent(fig: Figure, renderer: RendererBase, axes: Sequence[Axes]) -> Bbox:
    """The drawn extent of a panel's axes (labels, titles and legends included),
    in figure coordinates."""
    return Bbox.union(
        [
            ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
            for ax in axes
        ]
    )


def place_letters(
    fig: Figure,
    renderer: RendererBase,
    panels: Mapping[str, Sequence[Axes]],
    rows: Sequence[str],
    columns: Sequence[str],
    heights: Mapping[str, Sequence[Axes] | float] | None = None,
) -> dict[str, tuple[float, float]]:
    """Each panel letter's figure position, aligned by row and by column.

    The letters of a row (each string of ``rows``, such as ``"cdef"``) share the
    height just above the highest of their panels' drawn tops; the letters of a
    column (each string of ``columns``, such as ``"dh"``) share the left edge of
    the leftmost of their panels' drawn extents.

    ``heights`` names, by letter, which part of its panel sets its row's
    height in place of the panel's whole drawn extent: the axes whose drawn
    tops count (none: the panel does not count, as when a legend above it
    would lift the row's letters away from the others), or the letter's own
    height in figure coordinates.
    """
    heights = heights or {}
    y = {}
    for row in rows:
        tops = [
            panel_extent(fig, renderer, axes).y1
            for axes in (heights.get(name, panels[name]) for name in row)
            if not isinstance(axes, Real) and len(axes) > 0
        ]
        for name in row:
            given = heights.get(name)
            y[name] = (
                given if isinstance(given, Real) else min(max(tops) + 0.002, 0.985)
            )
    x = {
        name: max(
            min(panel_extent(fig, renderer, panels[n]).x0 for n in column) - 0.004,
            0.0,
        )
        for column in columns
        for name in column
    }
    return {name: (x[name], y[name]) for name in panels}


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def generate_figure(
    nwb_file_name: str = "wilbur20210329_.nwb",
    epoch_name: str = "02_r1",
    data_dir: Path | None = None,
    use_cache: bool = True,
):
    """Build Figure 2, loading cached render inputs when available.

    Parameters
    ----------
    use_cache : bool, default True
        When True and a cached bundle exists under ``data_dir / intermediates``,
        load it and skip the session load and the aggregated frames. When False
        (``--force-recompute``), recompute and overwrite the cache.
    """
    if data_dir is None:
        data_dir = processed_data_dir()
    table = read_swr_table(_event_table_path(data_dir))
    # the one selection: the events that start inside a well visit
    events = select_visit_context(table)
    cache_path = _figure02_cache_path(nwb_file_name, epoch_name, data_dir)
    if use_cache and cache_path.exists():
        logging.info(f"Loading cached render inputs from {cache_path}...")
        inputs = joblib.load(cache_path)
    else:
        inputs = _compute_figure_inputs(nwb_file_name, epoch_name, data_dir, events)
        logging.info(f"Caching render inputs to {cache_path}...")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(inputs, cache_path)

    # the frames hold every event: the whole table is checked against them
    check_table_against_frames(
        overlap_class_statistics(table), inputs["frame_event_counts"]
    )
    classes = overlap_class_statistics(events)
    animal, day = parse_nwb_filename(nwb_file_name)
    session_events = events[
        (events["animal"] == animal)
        & (events["session_date"] == day)
        & (events["epoch"] == epoch_name)
    ]
    windows = example_windows(
        session_events, inputs["data"]["ripple_times"], inputs["results"]
    )

    reference = reference_medians(
        checked_reference(
            pd.read_parquet(data_dir / BRAIN_STATE_REFERENCE_DIR / "reference.parquet"),
            data_dir / BRAIN_STATE_REFERENCE_DIR,
            "brain-state reference",
        ),
        ["ripple", "mua", "theta"],
    )
    power_counts = animal_signal_counts(
        nonlocal_counts(
            checked_reference(
                pd.read_parquet(data_dir / NONLOCAL_BY_SIGNAL_DIR / "counts.parquet"),
                data_dir / NONLOCAL_BY_SIGNAL_DIR,
                "counts by signal",
            ),
            "ripple_power",
        )
    )
    # every analyzed epoch with an event: an epoch with none has no table
    event_epochs = event_table_epochs(table, analyzed_epoch_ids())
    # the time masks read the visits themselves: they and the labels must agree
    check_visit_context_starts(
        table, {identifier: well_visit_times(identifier) for identifier in event_epochs}
    )
    # its controls are Local rest inside the current visits only when its
    # settings record the subset and it read the current session cache
    participation_dir = data_dir / EVENT_PARTICIPATION_DIR
    manifest = read_output_manifest(participation_dir)
    note = time_mask_staleness(
        {**manifest, "subset": manifest.get("settings", {}).get("subset")},
        event_epochs,
        f"the participation run ({participation_dir})",
    )
    if note is not None:
        raise ValueError(note)
    participation = participation_statistics(
        select_events_in(
            read_complete_run(participation_dir, PARTICIPATION_NAME, event_epochs),
            events["event_id"],
        )
    )
    # the events summary's restriction inputs: the same runs, cohorts, time
    # steps and subset; panel j draws the fold fit on whole trials, and only
    # from a held-out event run made from the restriction run and the per-bin
    # facts run as they are now, with facts made from the current session
    # cache, and from a restriction run made from the current decode frames
    # and SWR source: content of another design, decoder run, decode or visits
    # is refused, not drawn (the event-table reason is the summary's alone)
    heldout = read_heldout_restriction(
        data_dir / EVENT_NONLOCAL_RESTRICTION_HELDOUT_EVENTS_DIR,
        data_dir / EVENT_NONLOCAL_RESTRICTION_DIR,
        data_dir / EVENT_BIN_FACTS_DIR,
        event_epochs,
        sha256_file(_event_table_path(data_dir)),
    )
    refusal = heldout_restriction_refusal(
        heldout,
        restriction_inputs_note(
            data_dir / EVENT_NONLOCAL_RESTRICTION_DIR, event_epochs
        ),
    )
    if refusal is not None:
        raise ValueError(restriction_refusal(*refusal))
    restriction_decoders, restriction_outside = restriction_figure_statistics(
        heldout.events, heldout.time_sums, heldout.bin_s
    )

    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    outer = GridSpec(
        4,
        1,
        figure=fig,
        height_ratios=[1.9, 2.0, 1.3, 1.45],
        hspace=0.62,
        left=0.1,
        right=0.99,
        top=0.985,
        bottom=0.07,
    )
    max_pos = float(np.ceil(inputs["data"]["position_info"].linear_position.max()))
    ax_a, ax_a_top = draw_overview(fig, outer[0], inputs, session_events, max_pos)
    ax_b = draw_examples(fig, outer[1], inputs, windows, max_pos)
    row3 = outer[2].subgridspec(1, 4, width_ratios=[1.1, 1.1, 1.4, 1.15], wspace=0.7)
    ax_c = fig.add_subplot(row3[0])
    draw_swr_bars(ax_c, inputs["kay"], inputs["kay_content_animals"])
    ax_d = fig.add_subplot(row3[1])
    draw_events_bars(ax_d, classes, overlap_class_animal_shares(events))
    detectors = row3[2].subgridspec(1, 2, wspace=0.3)
    ax_e = fig.add_subplot(detectors[0])
    ax_e2 = fig.add_subplot(detectors[1])
    draw_detector_overlap(ax_e, ax_e2, inputs["overlap"], inputs["with_nonlocal"])
    offsets_grid = row3[3].subgridspec(2, 1, hspace=0.35)
    ax_f1 = fig.add_subplot(offsets_grid[0])
    ax_f2 = fig.add_subplot(offsets_grid[1], sharex=ax_f1)
    draw_offsets(ax_f1, ax_f2, boundary_offsets(events), events["animal"])
    row4 = outer[3].subgridspec(
        1, 4, width_ratios=[ROW4_WIDTHS[p] for p in ROW4_PANELS], wspace=ROW4_WSPACE
    )
    cells = dict(zip(ROW4_PANELS, row4, strict=True))
    triggered = cells["traces"].subgridspec(4, 1, hspace=0.2)
    ax_ripple = fig.add_subplot(triggered[0])
    triggered_axes = [
        ax_ripple,
        *(fig.add_subplot(triggered[r], sharex=ax_ripple) for r in (1, 2, 3)),
    ]
    draw_triggered(triggered_axes, inputs["triggered_curves"], reference)
    ax_participation = fig.add_subplot(cells["participation"])
    draw_participation(ax_participation, participation)
    power_grid = cells["ripple_power"].subgridspec(
        2, 1, height_ratios=[1.0, 1.6], hspace=0.2
    )
    ax_spread = fig.add_subplot(power_grid[0])
    ax_share = fig.add_subplot(power_grid[1], sharex=ax_spread)
    draw_share_by_power(ax_spread, ax_share, power_counts)
    decoder_grid = cells["decoder_comparison"].subgridspec(
        1, 2, width_ratios=DECODER_AXES_WIDTHS, wspace=DECODER_AXES_WSPACE
    )
    ax_decoders, ax_difference = (fig.add_subplot(decoder_grid[k]) for k in range(2))
    draw_decoder_gains(ax_decoders, restriction_decoders)
    draw_full_minus_swr(ax_difference, restriction_decoders, restriction_outside)
    row4_axes = {
        "traces": triggered_axes,
        "participation": [ax_participation],
        "ripple_power": [ax_spread, ax_share],
        "decoder_comparison": [ax_decoders, ax_difference],
    }

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # panel a's track starts where the drawing of the a, b, c, g column does
    x_left = (
        min(
            ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted()).x0
            for ax in (ax_b, ax_c)
        )
        - 0.004
    )
    # the decoder comparison fills the space between its neighbors' drawings, a
    # gap from each (panel c's drawing's left edge, or the figure's right, where
    # it has no neighbor on that side)
    position = ROW4_PANELS.index("decoder_comparison")
    if position > 0:
        neighbor = row4_axes[ROW4_PANELS[position - 1]]
        left = panel_extent(fig, renderer, neighbor).x1 + PANEL_GAP
    else:
        left = panel_extent(fig, renderer, [ax_c]).x0
    if position < len(ROW4_PANELS) - 1:
        neighbor = row4_axes[ROW4_PANELS[position + 1]]
        right = panel_extent(fig, renderer, neighbor).x0 - PANEL_GAP
    else:
        right = ROW4_RIGHT
    fit_axes_in_row(
        fig, renderer, row4_axes["decoder_comparison"], left, right,
        DECODER_AXES_WIDTHS,
    )  # fmt: skip
    fit_track_to_height(fig, ax_a, x_left)
    fig.canvas.draw()
    positions = place_letters(
        fig,
        renderer,
        {
            "a": [ax_a],
            "b": [ax_b],
            "c": [ax_c],
            "d": [ax_d],
            "e": [ax_e, ax_e2],
            "f": [ax_f1, ax_f2],
            **{ROW4_LETTERS[p]: row4_axes[p] for p in ROW4_PANELS},
        },
        rows=["a", "b", "cdef", "ghij"],
        columns=LETTER_COLUMNS,
        heights={
            # panel a's letter sits at the top of its decode, not of the track
            # drawing
            "a": ax_a_top.get_position().y1 + 0.004,
            # the legend drawn above the traces' axes would lift the last row's
            # letters away from the row's other panels
            ROW4_LETTERS["traces"]: [],
        },
    )
    for name, (x, y) in positions.items():
        fig.text(
            x, y, name, fontsize=PANEL_LETTER_SIZE, fontweight="bold", ha="left",
            va="bottom",
        )  # fmt: skip
    return fig


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("paper/figures/figure02")
    )
    parser.add_argument("--nwb-file-name", type=str, default="wilbur20210329_.nwb")
    parser.add_argument("--epoch-name", type=str, default="02_r1")
    parser.add_argument("--data-dir", type=Path, default=processed_data_dir())
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
        nwb_file_name=args.nwb_file_name,
        epoch_name=args.epoch_name,
        data_dir=args.data_dir,
        use_cache=not args.force_recompute,
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
