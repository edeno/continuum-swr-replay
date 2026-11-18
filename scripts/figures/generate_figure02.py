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
    c — The non-local events by SWR overlap (donut), the share an SWR-based
        analysis would leave out in the middle.
    d — The SWRs by content (donut): overlapping a non-local event, or not
        (mostly Local, or other).
    e — Held-out spike evidence, SWR vs no-SWR, each against matched quiet
        windows. PLACEHOLDER: today the Continuous-state bouts of the
        2026-10-16 run; to be replaced by the run on the canonical events.
    f — Ripple power and MUA around non-local event starts, SWR vs no-SWR, with
        the epochs' rest level (rest bins in no event).
    g — The share of non-local events each SWR definition, MUA and the
        Widloski detector overlaps; any SWR or MUA together as a reference.
    h — The share of rest time bins decoded non-local by ripple power, one line
        per animal: no step at the SWR threshold.
    i — Each SWR-overlapping event's start and end minus its SWR's, on a log
        count axis (the peak and the tails).

Panels c and i are drawn from the canonical event table by the functions the
events summary uses (``event_statistics.overlap_class_statistics``,
``boundary_offsets``); d from ``nonlocal_comparison.other_detector_statistics``
on the event features run the summary reads, and g's "any SWR or MUA" from
``left_out_statistics``; f's rest lines from ``event_statistics.reference_medians``
of the brain-state reference. Panels f and h are per-time-bin signals no stored
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
from continuum_swr_replay.analysis.event_definition.analysed_time import (
    read_swr_table,
    swr_intervals,
)
from continuum_swr_replay.analysis.event_definition.swr_source import (
    ripple_power_inputs,
    swr_epoch_id,
    swr_source_name,
)
from continuum_swr_replay.analysis.event_properties.class_or_continuum import (
    KAY_ZSCORE_THRESHOLD,
    nonlocal_share_by_power,
)
from continuum_swr_replay.analysis.event_properties.event_statistics import (
    boundary_offsets,
    overlap_class_statistics,
    reference_medians,
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
from continuum_swr_replay.parameters import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    DEFAULT_NON_LOCAL_THRESHOLD,
)
from continuum_swr_replay.paths import (
    BRAIN_STATE_REFERENCE_DIR,
    NL_BOUTS_BY_SWR_STEM,
    STATE_EVIDENCE_DIR,
    STATE_EVIDENCE_RUN,
    event_features_root,
    inventory_comparison_root,
    processed_data_dir,
    replay_content_dir,
    table_path,
)
from continuum_swr_replay.reported.formatting import format_number
from continuum_swr_replay.runs.io import read_manifest
from continuum_swr_replay.visualization.figure_panels import plot_event_example_stack
from continuum_swr_replay.visualization.figure_utilities import save_figure
from continuum_swr_replay.visualization.inventory_examples import MEMORYLESS_COLOR
from continuum_swr_replay.visualization.palettes import COLORS
from continuum_swr_replay.visualization.plots import (
    plot_linear_track_graph_strip,
    plot_track_graph_2d,
)
from continuum_swr_replay.visualization.simulated_run_panels import (
    LABEL_SIZE,
    SMALL_SIZE,
    check_journal_limits,
    readable_text_color,
)

# The render cache's version, part of its file name. The name also carries the
# SWR source's name, a fingerprint of the epoch register's analysed epochs and
# a fingerprint of the decode files read, so a new SWR inventory, artifact run,
# register change or rewritten decode makes a new cache by itself. Bump this
# whenever the cached values could change for any other reason.
# 4: panel a's examples read from the event table when drawn; frame event counts.
# 5: Figure 2 redrawn: triggered ripple and MUA (f) and the non-local share by
#    ripple power (h) are cached; the decoding-error panel left the figure.
FIGURE02_CACHE_VERSION = 5

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
C_MUA = COLORS["MUA"]
C_QUIET = COLORS["All Events"]
# The partial-overlap colour of figure_panels.plot_swr_overlap_class_bar;
# COLORS has none.
C_PART = "#b5825a"
CLASS_ORDER = ("swr_contained", "swr_partial", "non_swr")
CLASS_COLORS = {"swr_contained": C_SWR, "swr_partial": C_PART, "non_swr": C_NL}
CLASS_LABELS = {
    "swr_contained": "Inside SWR",
    "swr_partial": "Partly in SWR",
    "non_swr": "No SWR",
}

# Panel a: a run and the rest after it in the example session (start, seconds).
OVERVIEW_START_S = 1617036555.9
OVERVIEW_DURATION_S = 12.6
# Panel b: half the window around each example's centre, seconds.
EXAMPLE_HALF_WINDOW_S = 0.2
# Panel b's examples, chosen by eye from contact sheets of the example session's
# events with peak non-local probability >= 0.95, no other event and no other
# SWR within the window, and a sharp posterior (small HPD coverage).
EXAMPLE_EVENT_IDS = {
    "Inside SWR": 2185516843327500879,
    "Starts before SWR": 63427667006281341,
    "Ends after SWR": 3628740333881355820,
    "No SWR, trajectory": 5216998177801911428,
    "No SWR, stationary": 8826983562067101773,
}
EXAMPLE_CLASSES = {
    "Inside SWR": "swr_contained",
    "Starts before SWR": "swr_partial",
    "Ends after SWR": "swr_partial",
    "No SWR, trajectory": "non_swr",
    "No SWR, stationary": "non_swr",
}
# The SWR with Local content: its start in the SWR source, seconds. No non-local
# event overlaps it and the decode is Local through it.
LOCAL_SWR_START_S = 1617036373.982971
# Panel f: half the window around each event start, seconds.
TRIGGERED_HALF_WINDOW_S = 0.3
TRIGGERED_SIGNALS = ["SWR_zscore", "MUA_zscore"]
# Panel h: ripple-power bins, z.
POWER_EDGES = np.arange(-2.0, 10.01, 0.25)
# Panel g: the detectors, in the order drawn, with their labels and colours.
DETECTORS = (
    ("kay", "SWR", C_SWR),
    ("karlsson", "Karlsson", C_SWR),
    ("yu", "Yu", C_SWR),
    ("zugaro", "Zugaro", C_SWR),
    ("carey", "Carey", C_SWR),
    ("mua", "MUA", C_MUA),
    ("memoryless", "Widloski", MEMORYLESS_COLOR),
)
OTHER_SWR_DEFINITIONS = ("karlsson", "yu", "zugaro", "carey")


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
    """Panel h: the share of rest bins decoded non-local, by ripple power.

    Rest bins are those below the immobility speed and outside excluded time;
    each animal is one group (:func:`nonlocal_share_by_power`).
    """
    rest = (
        data_over_time["head_speed"].to_numpy()
        < DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC
    ) & ~data_over_time["in_excluded_time"].to_numpy(dtype=bool)
    rest_bins = data_over_time.loc[rest]
    return nonlocal_share_by_power(
        rest_bins["SWR_zscore"].to_numpy(),
        rest_bins["Non-Local_prob"].to_numpy(),
        rest_bins["animal"].to_numpy(),
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
    logging.info("Computing the triggered averages (panel f)...")
    triggered_curves = compute_triggered_average_curves(
        data_over_time,
        event_types,
        TRIGGERED_SIGNALS,
        time_col="start_time",
        window_sec=TRIGGERED_HALF_WINDOW_S,
    )
    logging.info("Computing the non-local share by ripple power (panel h)...")
    share_by_power = rest_nonlocal_share(data_over_time)
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
    for title, event_id in EXAMPLE_EVENT_IDS.items():
        rows = events[events["event_id"] == event_id]
        if len(rows) != 1:
            raise AssertionError(f"event_id {event_id} matches {len(rows)} rows")
        row = rows.iloc[0]
        if row["overlap_class"] != EXAMPLE_CLASSES[title]:
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
    windows["SWR, Local only"] = (start, end)
    return windows


def detector_overlap(directional: pd.DataFrame) -> pd.DataFrame:
    """Panel g: the share of non-local events each detector's events overlap.

    Pooled over animals (summed matched over summed eligible), and each
    animal's own share.
    """
    mine = directional[directional["source_detector"] == "state_space"]
    rows = []
    for detector, _, _ in DETECTORS:
        d = mine[mine["target_detector"] == detector]
        rows.append(
            {
                "detector": detector,
                "pooled": d["n_matched"].sum() / d["n_eligible"].sum(),
                "animals": (d["n_matched"] / d["n_eligible"]).to_numpy(),
            }
        )
    return pd.DataFrame(rows)


def heldout_placeholder(run_dir: Path) -> dict:
    """Panel e PLACEHOLDER: the Continuous-bout held-out evidence by SWR."""
    bouts = pd.read_parquet(
        run_dir / f"{NL_BOUTS_BY_SWR_STEM}.parquet",
        columns=[
            "epoch_id",
            "animal",
            "kind",
            "state",
            "regime",
            "runcv_nl_anim_mark_ps",
        ],
    )
    stats = pd.read_csv(run_dir / f"{NL_BOUTS_BY_SWR_STEM}_stats.csv")
    immobile = bouts[
        (bouts["regime"] == "immobile")
        & bouts["state"].str.startswith("Non-Local Continuous")
    ]
    epoch = immobile.groupby(["animal", "epoch_id", "kind", "state"])[
        "runcv_nl_anim_mark_ps"
    ].median()
    animal = (
        epoch.groupby(["animal", "kind", "state"]).mean().unstack(["kind", "state"])
    )
    stats = stats[
        (stats["metric"] == "runcv_nl_anim_mark_ps") & (stats["regime"] == "immobile")
    ].set_index("state")
    return {"animal": animal, "stats": stats}


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def clamp_lines(ax) -> None:
    """Keep every drawn line within Cell Press's 0.5-1.5 pt."""
    for line in ax.get_lines():
        line.set_linewidth(float(np.clip(line.get_linewidth(), 0.5, 1.5)))
    for collection in ax.collections:
        widths = np.atleast_1d(collection.get_linewidths())
        if len(widths) and np.any(widths > 0):
            collection.set_linewidths(
                np.where(widths > 0, np.clip(widths, 0.5, 1.5), 0)
            )
    for patch in ax.patches:
        if patch.get_linewidth() > 0:
            patch.set_linewidth(float(np.clip(patch.get_linewidth(), 0.5, 1.5)))


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
    non_local = (
        sp.sel(states="Non-Local Continuous") + sp.sel(states="Non-Local Fragmented")
    ).to_numpy()
    ax.stackplot(
        sp.time.values - t_offset,
        non_local,
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
    )
    strip.set_ylabel("")
    ax.tick_params(labelleft=False, left=False)
    clamp_lines(strip)
    return strip


def runs_above(values: np.ndarray, t: np.ndarray, threshold: float) -> list:
    """(start, end) times of each run of samples with ``values > threshold``."""
    above = np.r_[False, values > threshold, False]
    edges = np.flatnonzero(np.diff(above.astype(int)))
    return [(t[a], t[b - 1]) for a, b in zip(edges[::2], edges[1::2], strict=True)]


def draw_overview(fig, spec, inputs, session_events):
    """Panel a: the 2D track, then the decode over a run and the rest after it."""
    data, results, env = inputs["data"], inputs["results"], inputs["env"]
    start, dur = OVERVIEW_START_S, OVERVIEW_DURATION_S
    sl = slice(start, start + dur)
    max_pos = float(np.ceil(data["position_info"].linear_position.max()))
    gs = spec.subgridspec(1, 2, width_ratios=[1.6, 5.0], wspace=0.3)
    ax_track = fig.add_subplot(gs[0])
    plot_track_graph_2d(
        data["track_graph"],
        data["position_info"],
        ax=ax_track,
        edge_order=data["linear_edge_order"],
    )
    clamp_lines(ax_track)
    rows = gs[1].subgridspec(
        5, 1, height_ratios=[1.5, 1.05, 0.42, 0.42, 0.42], hspace=0.14
    )
    ax_pos, ax_lane, ax_rip, ax_st, ax_sp = (fig.add_subplot(rows[i]) for i in range(5))
    draw_posterior(ax_pos, data, results, env, sl, start, max_pos)
    add_track_strip(ax_pos, data, max_pos)
    ax_pos.text(
        6.0,
        270,
        "animal's position",
        color="magenta",
        fontsize=SMALL_SIZE,
        ha="left",
        va="top",
    )

    position = data["position_info"].loc[sl]
    t_pos = position.index.to_numpy() - start
    probabilities = results.acausal_state_probabilities.sel(time=sl)
    t_dec = probabilities.time.values - start
    lanes = {"Immobile": 4, "Local": 3, "Non-local": 2, "No-Spike": 1, "SWR": 0}
    lane_colors = {
        "Immobile": "k",
        "Local": C_LOCAL,
        "Non-local": C_NL,
        "No-Spike": C_NOSPIKE,
        "SWR": C_SWR,
    }

    def lane(name: str, s0: float, e0: float, color: str) -> None:
        ax_lane.barh(
            lanes[name], max(e0 - s0, 0.03), left=s0, height=0.72, color=color, lw=0
        )

    still = (
        position["head_speed"] < DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC
    ).to_numpy()
    for s0, e0 in runs_above(still.astype(float), t_pos, 0.5):
        lane("Immobile", s0, e0, C_QUIET)
    for name, state in (("Local", "Local"), ("No-Spike", "No-Spike")):
        for s0, e0 in runs_above(
            probabilities.sel(states=state).to_numpy(),
            t_dec,
            DEFAULT_NON_LOCAL_THRESHOLD,
        ):
            lane(name, s0, e0, lane_colors[name])
    in_window = session_events[
        (session_events["event_end_s"] > start)
        & (session_events["event_start_s"] < start + dur)
    ]
    for row in in_window.itertuples():
        lane("Non-local", row.event_start_s - start, row.event_end_s - start, C_NL)
        ax_pos.axvspan(
            row.event_start_s - start,
            row.event_end_s - start,
            color=C_NL,
            alpha=0.25,
            lw=0,
        )
    ripples = data["ripple_times"]
    for s0, e0 in zip(ripples["start_time"], ripples["end_time"], strict=True):
        if e0 > start and s0 < start + dur:
            lane("SWR", s0 - start, e0 - start, C_SWR)
    ax_lane.set_ylim(-0.5, 4.5)
    ax_lane.set_yticks(list(lanes.values()), list(lanes))
    for label, name in zip(ax_lane.get_yticklabels(), lanes, strict=True):
        label.set_color(readable_text_color(lane_colors[name]))
    ax_lane.tick_params(axis="y", length=0)
    ax_lane.tick_params(axis="x", length=0)
    ax_lane.spines["left"].set_visible(False)
    ax_lane.spines["bottom"].set_visible(False)

    ripple = data["zscored_ripple_consensus_trace"].loc[sl]
    ripple = ripple.iloc[:, 0] if ripple.ndim > 1 else ripple
    ax_rip.plot(ripple.index - start, ripple.to_numpy(), color="k", lw=0.5)
    ax_rip.set_ylim(-2, 8)
    ax_rip.set_yticks([0, 5])
    draw_state_stack(ax_st, results, sl, start)
    ax_sp.fill_between(t_pos, position["head_speed"], color=C_QUIET, lw=0)
    ax_sp.axhline(DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC, color="k", ls=":", lw=0.5)
    ax_sp.text(dur, 6, "4 cm/s", ha="right", va="bottom", fontsize=SMALL_SIZE)
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
    """Size the track's axes to fill its row's height and sit at ``left``."""
    fig.canvas.draw()
    box = ax_track.get_position()
    x0, x1 = ax_track.get_xlim()
    y0, y1 = ax_track.get_ylim()
    fig_w, fig_h = fig.get_size_inches()
    width = box.height * fig_h / fig_w * abs(x1 - x0) / abs(y1 - y0)
    ax_track.set_position([left, box.y0, width, box.height])
    ax_track.set_anchor("W")


def draw_examples(fig, spec, inputs, windows: dict):
    """Panel b: one column per example (ripple, raster, decode, states)."""
    data, results, env, detector = (
        inputs["data"],
        inputs["results"],
        inputs["env"],
        inputs["detector"],
    )
    max_pos = float(np.ceil(data["position_info"].linear_position.max()))
    n = len(windows)
    gs = spec.subgridspec(
        4, n, hspace=0.08, wspace=0.14, height_ratios=[0.5, 0.65, 1.2, 0.5]
    )
    first = None
    for j, (title, (start, end)) in enumerate(windows.items()):
        axes = [fig.add_subplot(gs[r, j]) for r in range(4)]
        center = (start + end) / 2
        sl = slice(center - EXAMPLE_HALF_WINDOW_S, center + EXAMPLE_HALF_WINDOW_S)
        plot_event_example_stack(
            data=data,
            results=results,
            detector=detector,
            env=env,
            time_slice=sl,
            axes=axes,
            column_title=title,
            show_ylabels=(j == 0),
            relative_time_center=center,
        )
        for ax in (axes[2], axes[3]):
            ax.clear()
        draw_posterior(axes[2], data, results, env, sl, center, max_pos)
        draw_state_stack(axes[3], results, sl, center)
        for ax in axes[:2]:
            ax.set_xlim(sl.start, sl.stop)
        for ax in axes[2:]:
            ax.set_xlim(-EXAMPLE_HALF_WINDOW_S, EXAMPLE_HALF_WINDOW_S)
        for ax in axes:
            ax.tick_params(labelsize=SMALL_SIZE)
            clamp_lines(ax)
        axes[0].set_title(title, fontsize=LABEL_SIZE, pad=3)
        axes[0].set_ylim(-2, 8)
        axes[0].set_yticks([0, 5])
        axes[2].tick_params(labelbottom=False, labelleft=False)
        axes[2].set_yticks([0, max_pos])
        axes[3].set_xticks([-0.1, 0, 0.1], ["-0.1", "0", "0.1"])
        axes[3].set_xlabel("Time from event center (s)" if j == n // 2 else "")
        axes[0].set_ylabel("Ripple\n(z)" if j == 0 else "")
        axes[1].set_ylabel("Tetrode" if j == 0 else "")
        axes[3].set_ylabel("State\nprob." if j == 0 else "")
        if j > 0:
            for ax in (axes[0], axes[1], axes[3]):
                ax.tick_params(labelleft=False)
        if j == 0:
            first = axes
    add_track_strip(first[2], data, max_pos)
    first[2].set_ylabel("Decoded\nposition (cm)")
    for ax in first:
        ax.yaxis.set_label_coords(-0.42, 0.5)
    return first[0]


def draw_donut(ax, wedges, centre_lines, title) -> None:
    """A donut of (label, fraction, colour, highlighted) wedges.

    The wedges' labels are listed to the right of the donut in their colours,
    the highlighted ones in bold; ``centre_lines`` are written in the hole.
    """
    patches, _ = ax.pie(
        [w[1] for w in wedges],
        colors=[w[2] for w in wedges],
        startangle=90,
        counterclock=False,
        radius=1.0,
        center=(0, 0),
        wedgeprops={"width": 0.36, "edgecolor": "white", "linewidth": 0.6},
    )
    labelled = [(p, w) for p, w in zip(patches, wedges, strict=True) if w[0]]
    for k, (patch, (label, _, color, highlighted)) in enumerate(labelled):
        if not highlighted:
            patch.set_alpha(0.5)
        ax.text(
            1.25,
            0.55 - 0.55 * k,
            label,
            ha="left",
            va="center",
            fontsize=SMALL_SIZE,
            fontweight="bold" if highlighted else "normal",
            color=readable_text_color(color),
        )
    for patch, w in zip(patches, wedges, strict=True):
        if not w[0] and not w[3]:
            patch.set_alpha(0.5)
    for k, (text, size, weight) in enumerate(centre_lines):
        ax.text(
            0,
            0.12 - 0.28 * k,
            text,
            ha="center",
            va="center",
            fontsize=size,
            fontweight=weight,
        )
    ax.set_xlim(-1.05, 2.9)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=SMALL_SIZE, pad=2, loc="left")


def draw_events_donut(ax, classes: dict) -> None:
    """Panel c: the non-local events by SWR overlap."""
    wedges = [
        (
            f"{CLASS_LABELS[c]}\n{format_number(classes[f'{c}_fraction'], 'pctnum')}%",
            classes[f"{c}_fraction"],
            CLASS_COLORS[c],
            c == "non_swr",
        )
        for c in CLASS_ORDER
    ]
    draw_donut(
        ax,
        wedges,
        [
            (f"{classes['n_non_swr']:,}", SMALL_SIZE, "bold"),
            ("no SWR", SMALL_SIZE, "normal"),
        ],
        f"Non-local events (n = {classes['n_events']:,})",
    )


def draw_swr_donut(ax, kay: pd.Series) -> None:
    """Panel d: the SWRs by content."""
    no_nonlocal = kay["no_nonlocal_fraction"]
    local = no_nonlocal * kay["no_nonlocal_mostly_local_fraction"]
    wedges = [
        (
            f"With non-local {format_number(1 - no_nonlocal, 'pctnum')}%",
            1 - no_nonlocal,
            C_NL,
            False,
        ),
        ("Mostly Local", local, C_LOCAL, True),
        ("", no_nonlocal - local, C_NOSPIKE, True),
    ]
    draw_donut(
        ax,
        wedges,
        [
            (f"{format_number(no_nonlocal, 'pctnum')}%", SMALL_SIZE, "bold"),
            ("no non-local", SMALL_SIZE, "normal"),
        ],
        f"SWRs (n = {int(kay['n_events']):,})",
    )
    ax.text(
        1.25,
        -0.55,
        f"({format_number(kay['no_nonlocal_mostly_local_fraction'], 'pctnum')}% of\nthose without)",
        ha="left",
        va="center",
        fontsize=SMALL_SIZE,
        color=readable_text_color(C_LOCAL),
    )


def draw_heldout(ax, heldout: dict) -> None:
    """Panel e PLACEHOLDER: evidence per held-out spike, each group vs quiet."""
    animal, stats = heldout["animal"], heldout["stats"]
    groups = (
        ("Non-Local Continuous | SWR", C_SWR),
        ("Non-Local Continuous | no SWR", C_NL),
    )
    for g, (state, color) in enumerate(groups):
        for k, (kind, bar_color) in enumerate((("bout", color), ("baseline", C_QUIET))):
            x = g * 2.6 + k
            values = animal[(kind, state)]
            if kind == "bout":
                est, lo, hi = stats.loc[
                    state, ["animal_estimate", "animal_ci_lo", "animal_ci_hi"]
                ]
                ax.bar(x, est, color=bar_color, width=0.8)
                ax.errorbar(
                    x,
                    est,
                    yerr=[[est - lo], [hi - est]],
                    color="k",
                    lw=0.7,
                    capsize=1.5,
                )
            else:
                ax.bar(x, values.mean(), color=bar_color, width=0.8)
            ax.scatter(
                x + np.linspace(-0.18, 0.18, len(values)),
                values,
                s=4,
                color="k",
                lw=0,
                zorder=3,
            )
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks([0, 1, 2.6, 3.6], ["SWR", "quiet", "No\nSWR", "quiet"])
    ax.set_ylabel("Evidence for decoded\nposition (nats/spike)")
    ax.set_title(
        "PLACEHOLDER: held-out spikes,\nContinuous bouts (not events)",
        fontsize=SMALL_SIZE,
        color="#b00000",
    )


def draw_triggered(ax_rip, ax_mua, curves: dict, rest: pd.DataFrame) -> None:
    """Panel f: ripple power and MUA around event starts, SWR vs no-SWR."""
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
    handles, labels = ax_rip.get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=C_NOSPIKE, ls="--", lw=0.6))
    labels.append("rest, no event")
    ax_rip.legend(
        handles,
        labels,
        frameon=False,
        loc="upper left",
        handlelength=1.0,
        bbox_to_anchor=(-0.02, 1.15),
    )
    ax_rip.tick_params(labelbottom=False)
    ax_mua.set_xlabel("Time from event start (ms)")


def draw_detector_overlap(ax, overlap: pd.DataFrame, any_swr_or_mua: float) -> None:
    """Panel g: the share of non-local events each detector overlaps."""
    colors = {d: c for d, _, c in DETECTORS}
    labels = {d: lab for d, lab, _ in DETECTORS}
    n = len(overlap) + 1
    for i, row in enumerate(overlap.itertuples()):
        y = n - 1 - i
        ax.barh(
            y,
            100 * row.pooled,
            color=colors[row.detector],
            height=0.65,
            alpha=0.55 if row.detector in OTHER_SWR_DEFINITIONS else 1.0,
        )
        ax.scatter(
            100 * row.animals,
            y + np.linspace(-0.18, 0.18, len(row.animals)),
            s=3,
            color="k",
            lw=0,
            zorder=3,
        )
    ax.barh(0, 100 * any_swr_or_mua, color="#666666", height=0.65)
    ax.set_yticks(
        range(n), ["Any SWR/MUA", *[labels[d] for d in overlap["detector"]][::-1]]
    )
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Non-local events it\noverlaps (%)")


def draw_share_by_power(ax, shares: pd.DataFrame) -> None:
    """Panel h: the share of rest bins decoded non-local, by ripple power."""
    curves = []
    for _, mine in shares.groupby("group"):
        centre = (mine["power_low"] + mine["power_high"]) / 2
        ax.plot(centre, mine["share"], color="k", lw=0.5, alpha=0.5)
        curves.append(mine["share"].to_numpy())
    centre = (shares["power_low"] + shares["power_high"]).unique() / 2
    ax.plot(centre, np.nanmean(curves, axis=0), color="k", lw=1.0)
    ax.axvline(KAY_ZSCORE_THRESHOLD, color=C_SWR, lw=0.8, ls="--")
    ax.text(
        KAY_ZSCORE_THRESHOLD + 0.3,
        0.05,
        "SWR\nthreshold",
        fontsize=SMALL_SIZE,
        color=C_SWR,
        va="bottom",
    )
    ax.set_xlim(POWER_EDGES[0], POWER_EDGES[-1])
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Ripple power (z)")
    ax.set_ylabel("Share of rest bins\nnon-local")


def draw_offsets(ax_start, ax_end, offsets: pd.DataFrame) -> None:
    """Panel i: event minus SWR start (top) and end (bottom), log counts."""
    bins = np.arange(-150, 151, 4)
    for ax, column, label in (
        (ax_start, "start_offset_s", "Start"),
        (ax_end, "end_offset_s", "End"),
    ):
        ax.hist(1000 * offsets[column], bins=bins, color="#555555", lw=0)
        ax.set_yscale("log")
        ax.axvline(0, color="k", lw=0.5, ls=":")
        ax.set_xlim(-150, 150)
        ax.set_ylim(1, 2e4)
        ax.set_yticks([1, 100, 10000], ["1", "$10^2$", "$10^4$"])
        ax.text(
            0.02, 0.95, label, transform=ax.transAxes, va="top", fontsize=SMALL_SIZE
        )
    ax_start.tick_params(labelbottom=False)
    ax_end.set_xlabel("Event − SWR\nboundary (ms)")
    ax_start.set_ylabel("SWR events")
    ax_start.yaxis.set_label_coords(-0.3, 0.0)


def letter(fig, ax, text: str, x: float | None = None, y: float | None = None) -> None:
    """A panel letter at the axes' drawn top-left, or at the given figure x / y."""
    fig.canvas.draw()
    box = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(
        fig.transFigure.inverted()
    )
    fig.text(
        max(box.x0 - 0.004, 0.0) if x is None else x,
        min(box.y1 + 0.002, 0.985) if y is None else y,
        text,
        fontsize=8,
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

    run = event_features_root() / EVENT_FEATURES_RUN
    others = nonlocal_comparison.other_detector_events(run)
    inventory_dir = (
        inventory_comparison_root()
        / read_manifest(run)["settings"]["input_runs"]["inventory"]
    )
    directional = pd.read_csv(inventory_dir / "report" / "directional_by_animal.csv")
    nonlocal_comparison.check_against_directional(others, directional)
    kay = nonlocal_comparison.other_detector_statistics(others).loc["kay"]
    left_out = nonlocal_comparison.left_out_statistics(run, inventory_dir)
    reference = reference_medians(
        pd.read_parquet(data_dir / BRAIN_STATE_REFERENCE_DIR / "reference.parquet"),
        ["ripple", "mua"],
    )
    heldout = heldout_placeholder(data_dir / STATE_EVIDENCE_RUN / STATE_EVIDENCE_DIR)

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
    ax_a, ax_a_top = draw_overview(fig, outer[0], inputs, session_events)
    ax_b = draw_examples(fig, outer[1], inputs, windows)
    row3 = outer[2].subgridspec(1, 3, width_ratios=[1.0, 1.0, 1.1], wspace=0.5)
    ax_c = fig.add_subplot(row3[0])
    draw_events_donut(ax_c, classes)
    ax_d = fig.add_subplot(row3[1])
    draw_swr_donut(ax_d, kay)
    ax_e = fig.add_subplot(row3[2])
    draw_heldout(ax_e, heldout)
    row4 = outer[3].subgridspec(1, 4, width_ratios=[1.15, 1.0, 1.0, 1.0], wspace=0.7)
    triggered = row4[0].subgridspec(2, 1, hspace=0.15)
    ax_f1 = fig.add_subplot(triggered[0])
    ax_f2 = fig.add_subplot(triggered[1], sharex=ax_f1)
    draw_triggered(ax_f1, ax_f2, inputs["triggered_curves"], reference)
    ax_g = fig.add_subplot(row4[1])
    draw_detector_overlap(
        ax_g,
        detector_overlap(directional),
        1 - left_out["nonlocal_events_outside_trigger_fraction"],
    )
    ax_h = fig.add_subplot(row4[2])
    draw_share_by_power(ax_h, inputs["share_by_power"])
    offsets_grid = row4[3].subgridspec(2, 1, hspace=0.15)
    ax_i1 = fig.add_subplot(offsets_grid[0])
    ax_i2 = fig.add_subplot(offsets_grid[1], sharex=ax_i1)
    draw_offsets(ax_i1, ax_i2, boundary_offsets(events))
    for ax in (ax_e, ax_f1, ax_f2, ax_g, ax_h, ax_i1, ax_i2):
        clamp_lines(ax)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    x_left = (
        min(
            ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted()).x0
            for ax in (ax_b, ax_f1)
        )
        - 0.004
    )
    fit_track_to_height(fig, ax_a, x_left + 0.02)
    y_a = ax_a_top.get_position().y1 + 0.004
    letter(fig, ax_a, "a", x=x_left, y=y_a)
    for ax, text, x in (
        (ax_b, "b", x_left),
        (ax_c, "c", x_left),
        (ax_d, "d", None),
        (ax_e, "e", None),
        (ax_f1, "f", x_left),
        (ax_g, "g", None),
        (ax_h, "h", None),
        (ax_i1, "i", None),
    ):
        letter(fig, ax, text, x=x)
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
