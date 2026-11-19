#!/usr/bin/env python
"""Generate Figure 2: non-local events and SWRs are partly overlapping inventories.

The decoder of Figure 1 applied to the recordings, and how its non-local events
relate to sharp-wave ripples (SWRs, Kay's detector, the SWR source):

    a — The decode over a run and the rest after it, with the 2D track. Lanes
        mark the immobile (analysed) time, the Local / non-local / No-Spike
        stretches and the SWRs; the counted non-local events are shaded.
    b — Six examples: a non-local event inside an SWR, one that starts before
        its SWR, one that ends after it, two with no SWR (a trajectory and a
        stationary event), and an SWR whose content is Local.
    c — The non-local events by SWR overlap (donut), with the animals' range of
        the no-SWR share and the share of non-local time outside SWRs.
    d — The SWRs by content (donut): with a non-local event, mostly Local, or
        other.
    e — For each SWR definition, MUA and the Widloski detector: the share of
        non-local events its events overlap, and the share of its own events
        that overlap a non-local event.
    f — Each SWR-overlapping event's start and end minus its SWR's, on a log
        count axis, with the share of pairs beyond 25 and 50 ms.
    g — Tetrode participation (share of tetrodes with a spike) and spike rate of
        each non-local event against a Local-rest window of its length, one line
        per animal, for no-SWR and SWR events
        (``event_evidence.participation``; ``Processed-Data/event_participation``).
    h — The share of event time in Continuous and in Fragmented (each event's
        mean state probability), no-SWR and SWR events; bars the animals' mean,
        dots the animals (``event_statistics.state_composition_statistics``).
    i — Held-out tetrode score: each tetrode left out of the decoder's smoother in
        turn and scored at every bin of each event under the other tetrodes'
        position distribution against local coding around the animal (time-fold
        encodings); the gain summed over tetrodes and bins, per spike. One line
        per animal from its Local-rest controls to its events, for no-SWR and
        SWR events, and the animals' mean with its interval
        (``event_evidence.tetrode_holdout``;
        ``Processed-Data/event_tetrode_holdout``).
    j — Ripple power and MUA around non-local event starts, SWR vs no-SWR, with
        the epochs' rest level (rest bins in no event).
    k — The share of rest time bins decoded non-local by ripple power, one line
        per animal: no step at the SWR threshold.

Panels c and f are drawn from the canonical event table by the functions the
events summary uses (``event_statistics.overlap_class_statistics``,
``boundary_offsets``); d from ``nonlocal_comparison.other_detector_statistics``
on the event features run the summary reads, and c's time share from
``left_out_statistics``; j's rest lines from ``event_statistics.reference_medians``
of the brain-state reference. Panels j and k are per-time-bin signals no stored
output holds, computed from the aggregated decode frames (cached), whose event
counts by SWR overlap must equal the canonical table's.

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
from collections.abc import Sequence
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

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
from continuum_swr_replay.analysis.event_definition.analysed_time import (
    immobile_analysed_bins,
    read_swr_table,
    swr_intervals,
)
from continuum_swr_replay.analysis.event_definition.nonlocal_event_rule import (
    nonlocal_state_probability,
)
from continuum_swr_replay.analysis.event_definition.swr_source import (
    ripple_power_inputs,
    swr_epoch_id,
    swr_source_name,
)
from continuum_swr_replay.analysis.event_evidence.likelihood_advantage import (
    likelihood_advantage_statistics,
)
from continuum_swr_replay.analysis.event_evidence.participation import (
    participation_statistics,
)
from continuum_swr_replay.analysis.event_evidence.tetrode_holdout import window_scores
from continuum_swr_replay.analysis.event_properties.class_or_continuum import (
    KAY_ZSCORE_THRESHOLD,
    nonlocal_share_by_power,
)
from continuum_swr_replay.analysis.event_properties.event_statistics import (
    OFFSET_TAILS_MS,
    boundary_offset_statistics,
    boundary_offsets,
    overlap_class_statistics,
    reference_medians,
    state_composition_statistics,
)
from continuum_swr_replay.analysis.event_properties.triggered_analysis import (
    compute_triggered_average_curves,
)
from continuum_swr_replay.data_loaders import load_data
from continuum_swr_replay.data_loaders.epoch_register import (
    analysed_decode_files,
    analysed_epoch_ids,
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
    EVENT_PARTICIPATION_DIR,
    EVENT_TETRODE_HOLDOUT_DIR,
    event_features_root,
    inventory_comparison_root,
    processed_data_dir,
    replay_content_dir,
    table_path,
)
from continuum_swr_replay.reported.formatting import format_number
from continuum_swr_replay.runs.io import read_complete_run, read_manifest
from continuum_swr_replay.visualization.figure_utilities import save_figure
from continuum_swr_replay.visualization.inventory_examples import detector_color
from continuum_swr_replay.visualization.palettes import COLORS, readable_text_color
from continuum_swr_replay.visualization.plots import (
    plot_linear_track_graph_strip,
    plot_ripple_consensus_trace_zscore,
    plot_spike_times,
    plot_track_graph_2d,
)
from continuum_swr_replay.visualization.simulated_run_panels import (
    LABEL_SIZE,
    LINE_WIDTH_LIMITS_PT,
    PANEL_LETTER_SIZE,
    SMALL_SIZE,
    check_journal_limits,
)

# The render cache's version, part of its file name. The name also carries the
# SWR source's name, a fingerprint of the epoch register's analysed epochs and
# a fingerprint of the decode files read, so a new SWR inventory, artifact run,
# register change or rewritten decode makes a new cache by itself. Bump this
# whenever the cached values could change for any other reason.
# 4: panel a's examples read from the event table when drawn; frame event counts.
# 5: Figure 2 redrawn: triggered ripple and MUA (f) and the non-local share by
#    ripple power (h) are cached; the decoding-error panel left the figure.
# 6: the detector comparison's statistics (d, g) are cached; the event features
#    run is part of the name.
# 7: the comparison's share of non-local time outside Kay's SWRs is cached (c).
# 8: each detector's share of its events overlapping a non-local event (h).
FIGURE02_CACHE_VERSION = 8

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
C_NL = COLORS["Non-Local"]
C_SWR = COLORS["SWR"]
C_QUIET = COLORS["All Events"]
CLASS_ORDER = ("swr_contained", "swr_partial", "non_swr")
CLASS_COLORS = {
    "swr_contained": C_SWR,
    "swr_partial": COLORS["SWR partial"],
    "non_swr": C_NL,
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
# Panel b: half the window around each example's centre, seconds.
EXAMPLE_HALF_WINDOW_S = 0.2
# Panel b's examples, chosen by eye from contact sheets of the example session's
# events with peak non-local probability >= 0.95, no other event and no other
# SWR within the window, and a sharp posterior (small HPD coverage): each
# title's event_id and the overlap class it is drawn as.
EXAMPLES = {
    "Inside SWR": (2185516843327500879, "swr_contained"),
    "Starts before SWR": (63427667006281341, "swr_partial"),
    "Ends after SWR": (3628740333881355820, "swr_partial"),
    "No SWR, trajectory": (5216998177801911428, "non_swr"),
    "No SWR, stationary": (8826983562067101773, "non_swr"),
}
# The SWR with Local content: its start in the SWR source, seconds. No non-local
# event overlaps it and the decode is Local through it.
LOCAL_SWR_START_S = 1617036373.982971
# Panel j: the triggered averages' window (``compute_triggered_average_curves``'s
# window_sec; the curves span half of it either side of the start), seconds.
TRIGGERED_WINDOW_S = 0.3
TRIGGERED_SIGNALS = ["SWR_zscore", "MUA_zscore"]
# Panel k: ripple-power bins, z.
POWER_EDGES = np.arange(-2.0, 10.01, 0.25)
# Panel e: the detectors, in the order drawn, and their labels.
DETECTOR_LABELS = {
    "kay": "Kay (SWR)",
    "karlsson": "Karlsson",
    "yu": "Yu",
    "zugaro": "Zugaro",
    "carey": "Carey",
    "mua": "MUA",
    "memoryless": "Widloski",
}


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


def rest_nonlocal_share(data_over_time: pd.DataFrame) -> pd.DataFrame:
    """Panel k: the share of rest bins decoded non-local, by ripple power.

    Rest bins are the immobile analysed ones (:func:`immobile_analysed_bins`);
    each animal is one group (:func:`nonlocal_share_by_power`).
    """
    rest = data_over_time.loc[
        immobile_analysed_bins(data_over_time),
        ["SWR_zscore", "Non-Local_prob", "animal"],
    ]
    return nonlocal_share_by_power(
        rest["SWR_zscore"].to_numpy(),
        rest["Non-Local_prob"].to_numpy(),
        rest["animal"].to_numpy(),
        edges=POWER_EDGES,
        threshold=DEFAULT_NON_LOCAL_THRESHOLD,
    )


def _decode_files_fingerprint(
    nwb_file_name: str, epoch_name: str, data_dir: Path
) -> str:
    """SHA-256 over the decode files ``_compute_figure_inputs`` reads.

    The files are the analysed epochs' ``data_over_time`` (as
    ``load_data_over_time`` finds them) and the example session's HPC
    ``results.nc`` and ``detector.pkl`` (as ``load_results`` names them), all
    in `data_dir`. Each contributes its path, size and ``st_mtime_ns``, or
    ``missing``, so the fingerprint is a stat per file, not a read.
    """
    example = decode_paths(
        format_epoch_identifier(nwb_file_name, epoch_name, "HPC"), data_dir
    )
    files = [
        *analysed_decode_files(data_dir, data_over_time_suffix()),
        example.results,
        example.detector,
    ]
    lines = []
    for path in files:
        if path.exists():
            stat = path.stat()
            lines.append(f"{path}\t{stat.st_size}\t{stat.st_mtime_ns}")
        else:
            lines.append(f"{path}\tmissing")
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def _figure02_cache_path(nwb_file_name: str, epoch_name: str, data_dir: Path) -> Path:
    """Path for the cached Figure-2 render inputs (joblib bundle).

    A single joblib bundle is used rather than netCDF because the bundle mixes
    a session ``data`` dict, an xarray ``results`` Dataset whose ``state_bins``
    MultiIndex netCDF cannot serialize, the fitted detector and environment
    objects, and small DataFrames; joblib (pickle) preserves them all exactly.
    """
    cohort = hashlib.sha256("\n".join(analysed_epoch_ids()).encode()).hexdigest()
    decodes = _decode_files_fingerprint(nwb_file_name, epoch_name, data_dir)
    return (
        data_dir
        / "intermediates"
        / f"{session_name(nwb_file_name, epoch_name)}_figure02_v{FIGURE02_CACHE_VERSION}_"
        f"{swr_source_name()}_register{cohort[:12]}_decodes{decodes[:12]}"
        f"_features{EVENT_FEATURES_RUN}"
        "_cache.joblib"
    )


def _compute_figure_inputs(nwb_file_name: str, epoch_name: str, data_dir: Path) -> dict:
    """Run the expensive loads and derivations; return the cached render inputs.

    The example session's ``data``/``results``/``detector``/``env`` (panels a
    and b), the triggered-average curves (f), the non-local share by ripple
    power (h) and the frames' immobile non-local events by SWR overlap, which
    the canonical event table is checked against. The multi-GB
    ``data_over_time`` is consumed here and not returned.
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
    logging.info("Loading the aggregated decode frames (panels f and h)...")
    data_over_time = load_data_over_time(data_dir)
    # The SWRs drawn: the session's SWR source intervals, the ones the event
    # table's classes are computed against.
    data["ripple_times"] = swr_intervals(
        data_over_time, nwb_file_name=nwb_file_name, epoch_name=epoch_name
    )
    event_types = get_event_types(data_over_time)
    event_counts = frame_event_counts(event_types)
    logging.info("Computing the triggered averages (panel j)...")
    triggered_curves = compute_triggered_average_curves(
        data_over_time,
        event_types,
        TRIGGERED_SIGNALS,
        time_col="start_time",
        window_sec=TRIGGERED_WINDOW_S,
    )
    logging.info("Computing the non-local share by ripple power (panel k)...")
    share_by_power = rest_nonlocal_share(data_over_time)
    del data_over_time
    logging.info("Reading the detector comparison (panels d and g)...")
    comparison = detector_comparison()
    return {
        "nwb_file_name": nwb_file_name,
        "epoch_name": epoch_name,
        "data": data,
        "results": results,
        "detector": detector,
        "env": env,
        "frame_event_counts": event_counts,
        "triggered_curves": triggered_curves,
        "share_by_power": share_by_power,
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
    events: pd.DataFrame, ripple_times: pd.DataFrame, results
) -> dict[str, tuple[float, float]]:
    """Panel b's examples as (start, end) seconds, each checked.

    Raises
    ------
    AssertionError
        If an example's ``event_id`` is not one row of the table or its class is
        not the one it is drawn as; or if the Local SWR is not one of the
        source's SWRs, a non-local event overlaps it, or the decode is not
        mostly Local through it.
    """
    windows = {}
    for title, (event_id, overlap_class) in EXAMPLES.items():
        rows = events[events["event_id"] == event_id]
        if len(rows) != 1:
            raise AssertionError(f"event_id {event_id} matches {len(rows)} rows")
        row = rows.iloc[0]
        if row["overlap_class"] != overlap_class:
            raise AssertionError(f"{title}: event {event_id} is {row['overlap_class']}")
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
    if local <= 0.8:
        raise AssertionError(
            f"the decode is not mostly Local in the Local SWR ({local})"
        )
    windows["SWR, mostly Local"] = (start, end)
    return windows


def detector_comparison() -> dict:
    """Panels d and g: the event features run's comparison, as the summary reads it.

    Returns
    -------
    dict
        ``kay``: :func:`nonlocal_comparison.other_detector_statistics`'s row of
        Kay's SWRs (d); ``overlap``: the share of non-local events each of
        :data:`DETECTOR_LABELS` overlaps and ``with_nonlocal``, the share of
        each one's events that overlap a non-local event (h); and, from
        ``left_out_statistics``, ``time_outside_swr``, the share of non-local
        time outside Kay's SWRs (c).
    """
    run = event_features_root() / EVENT_FEATURES_RUN
    others = nonlocal_comparison.other_detector_events(run)
    inventory_dir = (
        inventory_comparison_root()
        / read_manifest(run)["settings"]["input_runs"]["inventory"]
    )
    directional = pd.read_csv(inventory_dir / "report" / "directional_by_animal.csv")
    nonlocal_comparison.check_against_directional(others, directional)
    left_out = nonlocal_comparison.left_out_statistics(run, inventory_dir)
    return {
        "kay": nonlocal_comparison.other_detector_statistics(others).loc["kay"],
        "overlap": nonlocal_comparison.nonlocal_events_overlapped(
            directional, tuple(DETECTOR_LABELS)
        ),
        "with_nonlocal": nonlocal_comparison.events_overlapping_nonlocal(
            directional, tuple(DETECTOR_LABELS)
        ),
        "time_outside_swr": left_out["nonlocal_time_outside_kay_fraction"],
    }


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def summed_posterior(window, env) -> np.ndarray:
    """Posterior over position summed over the states with one, (n_time, n_bins)."""
    post = window.acausal_posterior
    states = post.state.values
    total = sum(
        post.isel(state_bins=np.flatnonzero(states == s)).to_numpy()
        for s in ("Local", "Non-Local Continuous", "Non-Local Fragmented")
    )
    total[:, ~env.is_track_interior_] = np.nan
    return total


def draw_posterior(ax, data, results, env, sl: slice, t_offset: float, max_pos: float):
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
        color="magenta",
        s=0.4,
        lw=0,
        rasterized=True,
    )
    ax.set_ylim(0, max_pos)


def draw_state_stack(ax, results, sl: slice, t_offset: float) -> None:
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


def add_track_strip(ax, data, max_pos: float):
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


def draw_overview(fig, spec, inputs, session_events, max_pos: float):
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
    rows = gs[1].subgridspec(
        5, 1, height_ratios=[1.5, 1.05, 0.42, 0.42, 0.42], hspace=0.14
    )
    ax_pos, ax_lane, ax_rip, ax_st, ax_sp = (fig.add_subplot(rows[i]) for i in range(5))
    draw_posterior(ax_pos, data, results, env, sl, start, max_pos)
    add_track_strip(ax_pos, data, max_pos)
    ax_pos.text(
        0.48, 0.45, "animal's position", transform=ax_pos.transAxes,
        color="magenta", fontsize=SMALL_SIZE, ha="left", va="top",
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


def fit_track_to_height(fig, ax_track, left: float) -> None:
    """Size the track's equal-aspect axes to fill its row's height, at ``left``."""
    box = ax_track.get_position(original=True)
    x0, x1 = ax_track.get_xlim()
    y0, y1 = ax_track.get_ylim()
    fig_w, fig_h = fig.get_size_inches()
    width = box.height * fig_h / fig_w * abs(x1 - x0) / abs(y1 - y0)
    ax_track.set_position([left, box.y0, width, box.height])


def draw_examples(fig, spec, inputs, windows: dict, max_pos: float):
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


def draw_donut(
    ax, wedges, centre: tuple[str, str], title, label_space: float = 2.2
) -> None:
    """A donut of (label, fraction, colour, highlighted) wedges.

    The wedges with a label are listed, one line each, to the right of the
    donut in their colours, the highlighted ones in bold; the others are faded. ``centre`` is
    a number and its caption, written in the hole. ``label_space`` is the room
    kept right of the donut for the labels, in donut radii: the labels' width
    over the donut's sets the donut's size in its axes.
    """
    patches, _ = ax.pie(
        [w[1] for w in wedges],
        colors=[w[2] for w in wedges],
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.28, "edgecolor": "white", "linewidth": 0.6},
    )
    k = 0
    for patch, (label, _, color, highlighted) in zip(patches, wedges, strict=True):
        if not highlighted:
            patch.set_alpha(0.5)
        if label:
            ax.text(
                1.2, 0.4 - 0.4 * k, label, ha="left", va="center",
                fontsize=SMALL_SIZE, fontweight="bold" if highlighted else "normal",
                color=readable_text_color(color),
            )  # fmt: skip
            k += 1
    number, caption = centre
    ax.text(
        0,
        0.17,
        number,
        ha="center",
        va="center",
        fontsize=SMALL_SIZE,
        fontweight="bold",
    )
    ax.text(0, -0.19, caption, ha="center", va="center", fontsize=SMALL_SIZE)
    ax.set_xlim(-1.05, 1.2 + label_space)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=SMALL_SIZE, pad=2, loc="left")


def draw_events_donut(ax, classes: dict, time_outside_swr: float) -> None:
    """Panel c: the non-local events by SWR overlap, and the time outside SWRs."""
    wedges = [
        (
            f"{CLASS_LABELS[c]} {format_number(classes[f'{c}_fraction'], 'pctnum')}%",
            classes[f"{c}_fraction"],
            CLASS_COLORS[c],
            c == "non_swr",
        )
        for c in CLASS_ORDER
    ]
    draw_donut(
        ax,
        wedges,
        (f"{classes['n_non_swr']:,}", "no SWR"),
        f"Non-local events (n = {classes['n_events']:,})",
        label_space=1.6,
    )
    low = format_number(classes["non_swr_fraction_animal_min"], "pctnum")
    high = format_number(classes["non_swr_fraction_animal_max"], "pctnum")
    ax.text(
        -1.05, -1.2,
        f"No SWR: {low}–{high}% across animals\n"
        f"{format_number(time_outside_swr, 'pctnum')}% of non-local time outside SWRs",
        ha="left", va="top", fontsize=SMALL_SIZE,
    )  # fmt: skip


def draw_swr_donut(ax, kay: pd.Series) -> None:
    """Panel d: the SWRs by content."""
    no_nonlocal = kay["no_nonlocal_fraction"]
    mostly_local = kay["mostly_local_fraction"]
    wedges = [
        (
            f"With non-local {format_number(1 - no_nonlocal, 'pctnum')}%",
            1 - no_nonlocal,
            C_NL,
            False,
        ),
        (
            f"Mostly Local {format_number(mostly_local, 'pctnum')}%",
            mostly_local,
            C_LOCAL,
            True,
        ),
        (
            f"Other {format_number(no_nonlocal - mostly_local, 'pctnum')}%",
            no_nonlocal - mostly_local,
            C_NOSPIKE,
            False,
        ),
    ]
    draw_donut(
        ax,
        wedges,
        (f"{format_number(no_nonlocal, 'pctnum')}%", "no non-local"),
        f"SWRs (n = {int(kay['n_events']):,})",
    )


def draw_participation(ax_share, ax_rate, statistics: pd.DataFrame) -> None:
    """Panel g: tetrode participation and spike rate, Local rest vs events.

    One line per animal from its length-matched Local-rest controls to its
    events, for no-SWR events and for SWR events (``participation_statistics``).
    """
    for ax, measure, label in (
        (ax_share, "participation", "Tetrodes\nactive (%)"),
        (ax_rate, "spike_rate_hz", "Spike rate\n(spikes/s)"),
    ):
        scale = 100 if measure == "participation" else 1
        for g, (group, color) in enumerate((("no SWR", C_NL), ("SWR", C_SWR))):
            x = (2.4 * g, 2.4 * g + 1)
            rest = scale * statistics.loc[(group, "control"), f"{measure}_animals"]
            event = scale * statistics.loc[(group, "event"), f"{measure}_animals"]
            for a, b in zip(rest, event, strict=True):
                ax.plot(x, (a, b), color="#999999", lw=0.5, zorder=1)
            ax.scatter(
                np.full(rest.size, x[0]), rest, s=6, color=C_LOCAL, lw=0, zorder=2
            )
            ax.scatter(
                np.full(event.size, x[1]), event, s=6, color=color, lw=0, zorder=2
            )
        ax.set_ylabel(label)
        ax.set_xlim(-0.5, 3.9)
        ax.set_ylim(0, None)
    ax_share.tick_params(labelbottom=False)
    ax_rate.set_xticks(
        [0, 1, 2.4, 3.4], ["rest", "no SWR", "rest", "SWR"], rotation=45, ha="right"
    )


def draw_state_composition(ax, statistics: pd.DataFrame) -> None:
    """Panel h: the share of event time in Continuous and in Fragmented.

    Bars are the animals' mean, dots the animals
    (``event_statistics.state_composition_statistics``).
    """
    for g, (group, color) in enumerate((("no SWR", C_NL), ("SWR", C_SWR))):
        row = statistics.loc[group]
        for x, state in enumerate(("continuous", "fragmented")):
            position = x + (g - 0.5) * 0.36
            ax.bar(position, 100 * row[f"{state}_animal_mean"], width=0.34, color=color)
            animals = 100 * row[f"{state}_animals"]
            ax.scatter(
                position + np.linspace(-0.08, 0.08, len(animals)), animals, s=3,
                color="k", lw=0, zorder=3,
            )  # fmt: skip
    ax.set_xticks([0, 1], ["Continuous", "Fragmented"], rotation=30, ha="right")
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 50, 100])
    ax.set_ylabel("% event time")
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in (C_NL, C_SWR)]
    ax.legend(
        handles,
        ["no SWR", "SWR"],
        frameon=False,
        fontsize=SMALL_SIZE,
        loc="upper right",
        handlelength=0.8,
        borderaxespad=0,
    )


def draw_heldout_gain(ax, events: pd.DataFrame, controls: pd.DataFrame) -> None:
    """Panel i: the held-out tetrode gain over local coding, per spike.

    One line per animal from its Local-rest controls to its events; the bar is
    the events' animal mean with its 95 % interval
    (``likelihood_advantage_statistics`` of ``gain_per_spike``).
    """
    for g, (group, color) in enumerate((("no SWR", C_NL), ("SWR", C_SWR))):
        x = (2.4 * g, 2.4 * g + 1)
        rest, event = controls.loc[group], events.loc[group]
        rest_values = pd.Series(rest["animals"], index=rest["animal_names"])
        event_values = pd.Series(event["animals"], index=event["animal_names"])
        for animal in event_values.index:
            ax.plot(x, (rest_values[animal], event_values[animal]), color="#999999",
                    lw=0.5, zorder=1)  # fmt: skip
        ax.scatter(np.full(rest_values.size, x[0]), rest_values, s=6, color=C_LOCAL,
                   lw=0, zorder=2)  # fmt: skip
        ax.scatter(np.full(event_values.size, x[1]), event_values, s=6, color=color,
                   lw=0, zorder=2)  # fmt: skip
        ax.errorbar(
            x[1] + 0.3, event["estimate"],
            yerr=[[event["estimate"] - event["ci_low"]],
                  [event["ci_high"] - event["estimate"]]],
            fmt="_", color="black", ms=5, lw=0.8, capsize=0,
        )  # fmt: skip
    ax.axhline(0, color="#999999", lw=0.5, ls=":", zorder=0)
    ax.set_xlim(-0.5, 3.9)
    ax.set_xticks(
        [0, 1, 2.4, 3.4], ["rest", "no SWR", "rest", "SWR"], rotation=45, ha="right"
    )
    ax.set_ylabel("Held-out gain\nover local (nats/spike)")
    ax.set_title("Held-out tetrode", fontsize=SMALL_SIZE, pad=3)


def draw_triggered(ax_rip, ax_mua, curves: dict, rest: pd.DataFrame) -> None:
    """Panel j: ripple power and MUA around event starts, SWR vs no-SWR."""
    for ax, signal, label, word in (
        (ax_rip, "SWR_zscore", "Ripple (z)", "ripple"),
        (ax_mua, "MUA_zscore", "MUA (z)", "mua"),
    ):
        curve = curves[signal]
        t_ms = 1000 * np.asarray(curve["time"])
        for key, color, name in (
            ("overlap", C_SWR, "SWR"),
            ("no_overlap", C_NL, "No SWR"),
        ):
            mean, sem = (
                np.asarray(curve[f"{key}_mean"]),
                np.asarray(curve[f"{key}_sem"]),
            )
            ax.fill_between(t_ms, mean - sem, mean + sem, color=color, alpha=0.3, lw=0)
            ax.plot(t_ms, mean, color=color, lw=1.0, label=name)
        ax.axvline(0, color="k", lw=0.5, ls=":")
        ax.axhline(rest.loc[word, "rest"], color=C_NOSPIKE, lw=0.6, ls="--")
        ax.set_ylabel(label)
        ax.set_xlim(t_ms[0], t_ms[-1])
        ax.set_ylim(rest.loc[word, "rest"] - 0.3, None)
    handles, labels = ax_rip.get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=C_NOSPIKE, ls="--", lw=0.6))
    labels.append("rest")
    ax_rip.legend(
        handles,
        labels,
        frameon=False,
        loc="upper left",
        handlelength=0.8,
        borderaxespad=0.1,
        labelspacing=0.2,
    )
    ax_rip.tick_params(labelbottom=False)
    ax_mua.set_xlabel("Time from event start (ms)")


def draw_detector_bars(
    ax, shares: pd.DataFrame, n_rows: int, xlabel: str, show_labels: bool
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
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 50, 100])
    ax.set_xlabel(xlabel)


def draw_detector_overlap(
    ax_caught, ax_content, overlap: pd.DataFrame, with_nonlocal: pd.DataFrame
) -> None:
    """Panel e: what each detector catches, and what its events contain.

    Bars pool all events; dots are animals (for the caption).

    Left, the share of non-local events each detector's events overlap; right,
    the share of its own events that overlap a non-local event.
    """
    n_rows = len(overlap)
    draw_detector_bars(ax_caught, overlap, n_rows, "% events", True)
    ax_caught.set_title("Non-local\nevents found", fontsize=SMALL_SIZE, pad=3)
    draw_detector_bars(ax_content, with_nonlocal, n_rows, "% events", False)
    ax_content.set_title(
        "Detections with\nnon-local activity", fontsize=SMALL_SIZE, pad=3
    )
    ax_content.spines["left"].set_visible(False)


def draw_share_by_power(ax, shares: pd.DataFrame) -> None:
    """Panel k: the share of rest bins decoded non-local, by ripple power.

    One thin line per animal, their mean in bold.
    """
    by_animal = shares.pivot(index="power_low", columns="group", values="share")
    centre = by_animal.index + np.diff(POWER_EDGES)[0] / 2
    ax.plot(centre, by_animal.to_numpy(), color="k", lw=0.5, alpha=0.5)
    ax.plot(centre, by_animal.mean(axis=1), color="k", lw=1.0)
    ax.axvline(KAY_ZSCORE_THRESHOLD, color=C_SWR, lw=0.8, ls="--")
    ax.text(
        KAY_ZSCORE_THRESHOLD + 0.3, 0.05, "SWR\nthreshold", fontsize=SMALL_SIZE,
        color=C_SWR, va="bottom",
    )  # fmt: skip
    ax.set_xlim(POWER_EDGES[0], POWER_EDGES[-1])
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Ripple power (z)")
    ax.set_ylabel("Share of rest bins\nnon-local")


def draw_offsets(ax_start, ax_end, offsets: pd.DataFrame) -> None:
    """Panel f: event minus SWR start (top) and end (bottom), log counts.

    Each row prints the share of event-SWR pairs whose bound differs by more
    than 25 and 50 ms (``boundary_offset_statistics``).
    """
    tails = boundary_offset_statistics(offsets)
    bins = np.arange(-150, 151, 4)
    for ax, bound, label in ((ax_start, "start", "Start"), (ax_end, "end", "End")):
        ax.hist(1000 * offsets[f"{bound}_offset_s"], bins=bins, color="#555555", lw=0)
        ax.set_yscale("log")
        ax.axvline(0, color="k", lw=0.5, ls=":")
        ax.set_xlim(-150, 150)
        ax.set_ylim(1, 2e6)
        ax.set_yticks([1, 100, 10000], ["1", "$10^2$", "$10^4$"])
        beyond = ", ".join(
            f">{ms} ms {format_number(tails[f'{bound}_offset_beyond_{ms}ms_fraction'], 'pctnum')}%"
            for ms in OFFSET_TAILS_MS
        )
        ax.text(
            0.02, 0.97, f"{label}\n{beyond}", transform=ax.transAxes, va="top",
            fontsize=SMALL_SIZE,
        )  # fmt: skip
    ax_start.tick_params(labelbottom=False)
    ax_end.set_xlabel("Event − SWR bound (ms)\n< 0: event first")
    ax_start.set_ylabel("Event–SWR pairs")
    ax_start.yaxis.set_label_coords(-0.3, 0.0)


def letter(
    fig, renderer, ax, text: str, x: float | None = None, y: float | None = None
):
    """A panel letter at the axes' drawn top-left, or at the given figure x / y.

    ``renderer`` is the canvas's after one draw; adding the letters moves no
    axes (the figure has no layout engine), so one draw serves them all.
    """
    box = ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
    fig.text(
        max(box.x0 - 0.004, 0.0) if x is None else x,
        min(box.y1 + 0.002, 0.985) if y is None else y,
        text,
        fontsize=PANEL_LETTER_SIZE,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


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
    cache_path = _figure02_cache_path(nwb_file_name, epoch_name, data_dir)
    if use_cache and cache_path.exists():
        logging.info(f"Loading cached render inputs from {cache_path}...")
        inputs = joblib.load(cache_path)
    else:
        inputs = _compute_figure_inputs(nwb_file_name, epoch_name, data_dir)
        logging.info(f"Caching render inputs to {cache_path}...")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(inputs, cache_path)

    events = read_swr_table(
        table_path(
            "event_table",
            data_dir / replay_content_dir().relative_to(processed_data_dir()),
        )
    )
    classes = overlap_class_statistics(events)
    check_table_against_frames(classes, inputs["frame_event_counts"])
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
        pd.read_parquet(data_dir / BRAIN_STATE_REFERENCE_DIR / "reference.parquet"),
        ["ripple", "mua"],
    )
    participation = participation_statistics(
        read_complete_run(data_dir / EVENT_PARTICIPATION_DIR, "participation")
    )
    heldout = window_scores(
        read_complete_run(data_dir / EVENT_TETRODE_HOLDOUT_DIR, "tetrode_holdout")
    )
    heldout_events, heldout_controls = (
        likelihood_advantage_statistics(
            heldout[heldout["kind"] == kind], column="gain_per_spike"
        )
        for kind in ("event", "control")
    )

    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    outer = GridSpec(
        4,
        1,
        figure=fig,
        height_ratios=[2.2, 1.95, 1.1, 1.2],
        hspace=0.55,
        left=0.1,
        right=0.99,
        top=0.985,
        bottom=0.07,
    )
    max_pos = float(np.ceil(inputs["data"]["position_info"].linear_position.max()))
    ax_a, ax_a_top = draw_overview(fig, outer[0], inputs, session_events, max_pos)
    ax_b = draw_examples(fig, outer[1], inputs, windows, max_pos)
    row3 = outer[2].subgridspec(1, 4, width_ratios=[1.1, 1.3, 1.3, 1.05], wspace=0.5)
    ax_c = fig.add_subplot(row3[0])
    draw_events_donut(ax_c, classes, inputs["time_outside_swr"])
    ax_d = fig.add_subplot(row3[1])
    draw_swr_donut(ax_d, inputs["kay"])
    detectors = row3[2].subgridspec(1, 2, wspace=0.3)
    ax_e = fig.add_subplot(detectors[0])
    ax_e2 = fig.add_subplot(detectors[1])
    draw_detector_overlap(ax_e, ax_e2, inputs["overlap"], inputs["with_nonlocal"])
    offsets_grid = row3[3].subgridspec(2, 1, hspace=0.15)
    ax_f1 = fig.add_subplot(offsets_grid[0])
    ax_f2 = fig.add_subplot(offsets_grid[1], sharex=ax_f1)
    draw_offsets(ax_f1, ax_f2, boundary_offsets(events))
    row4 = outer[3].subgridspec(
        1, 5, width_ratios=[1.0, 0.95, 0.8, 0.95, 0.95], wspace=0.9
    )
    participation_grid = row4[0].subgridspec(2, 1, hspace=0.15)
    ax_g = fig.add_subplot(participation_grid[0])
    ax_g2 = fig.add_subplot(participation_grid[1])
    draw_participation(ax_g, ax_g2, participation)
    ax_h = fig.add_subplot(row4[1])
    draw_state_composition(ax_h, state_composition_statistics(events))
    ax_i = fig.add_subplot(row4[2])
    draw_heldout_gain(ax_i, heldout_events, heldout_controls)
    triggered = row4[3].subgridspec(2, 1, hspace=0.15)
    ax_j1 = fig.add_subplot(triggered[0])
    ax_j2 = fig.add_subplot(triggered[1], sharex=ax_j1)
    draw_triggered(ax_j1, ax_j2, inputs["triggered_curves"], reference)
    ax_k = fig.add_subplot(row4[4])
    draw_share_by_power(ax_k, inputs["share_by_power"])

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    x_left = (
        min(
            ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted()).x0
            for ax in (ax_b, ax_g)
        )
        - 0.004
    )
    fit_track_to_height(fig, ax_a, x_left + 0.02)
    letter(fig, renderer, ax_a, "a", x=x_left, y=ax_a_top.get_position().y1 + 0.004)
    for ax, text, x in (
        (ax_b, "b", x_left),
        (ax_c, "c", x_left),
        (ax_d, "d", None),
        (ax_e, "e", None),
        (ax_f1, "f", None),
        (ax_g, "g", x_left),
        (ax_h, "h", None),
        (ax_i, "i", None),
        (ax_j1, "j", None),
        (ax_k, "k", None),
    ):
        letter(fig, renderer, ax, text, x=x)
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
