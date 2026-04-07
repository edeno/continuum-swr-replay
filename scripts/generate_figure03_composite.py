#!/usr/bin/env python
"""Generate Figure 3: Non-local events exist outside detected SWR/MUA.

Composite figure with 5 panels:

    Row 1 (panel a — full width):
        Real data example showing ~2s around a non-local event WITHOUT SWR.
        Stacked sub-rows: SWR z-score (flat), spike raster, non-local
        posterior (shows clear replay), state probabilities.

    Row 2 (panels b, c, d, e):
        b: SWR z-score density for Local vs Non-Local timepoints — shows
           the model finds non-local activity at low SWR power
        c: Venn diagram of non-local, SWR, MUA overlap
        d: Replay speed by SWR status — low-SWR replay moves more slowly
        e: Replay position by SWR status — non-SWR replay represents
           reward zones

Usage:
    python scripts/generate_figure03_composite.py
    python scripts/generate_figure03_composite.py --output-dir paper/figures/figure03
"""

import argparse
import logging
import warnings
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import refactored panel functions
from generate_figure03 import (
    load_data_over_time,
    plot_replay_position_by_swr,
    plot_replay_speed_by_swr,
    plot_swr_zscore_by_state,
)
from matplotlib.gridspec import GridSpec

import continuum_swr_replay.figure_utilities as figure_utilities
from continuum_swr_replay.analysis.constants import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    DEFAULT_NON_LOCAL_THRESHOLD,
)
from continuum_swr_replay.analysis.hippocampus import (
    get_confident_times,
    get_overlap_times,
    load_results,
)
from continuum_swr_replay.data_loaders import load_data
from continuum_swr_replay.figure_utilities import (
    PAGE_HEIGHT,
    TWO_COLUMN,
    label_panels,
    save_figure,
)
from continuum_swr_replay.parameters import (
    COLORS,
    PROCESSED_DATA_DIR,
)
from continuum_swr_replay.visualization.plots import (
    plot_acausal_probabilities,
    plot_conditional_non_local_posterior,
    plot_ripple_consensus_trace_zscore,
    plot_spike_times,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%d-%b-%y %H:%M:%S"
)
figure_utilities.set_figure_defaults(context="paper")
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Panel a — Real data example (non-SWR non-local event)
# ---------------------------------------------------------------------------

def find_non_swr_example_event(data, results, window_sec=1.0):
    """Find a representative non-local event NOT co-occurring with SWR.

    Selects the event with highest peak non-local probability that does
    not overlap with any detected SWR.

    Returns
    -------
    time_slice : slice
        Time window centered on the event.
    """
    non_local_prob = (
        results.acausal_state_probabilities.sel(states="Non-Local Continuous")
        + results.acausal_state_probabilities.sel(states="Non-Local Fragmented")
    )
    position_info = data["position_info"]
    speed = position_info.head_speed.to_numpy()
    time = position_info.index.to_numpy()

    non_local_times = get_confident_times(
        non_local_prob, speed, time, DEFAULT_NON_LOCAL_THRESHOLD,
    )
    non_local_times = non_local_times[
        non_local_times.max_speed < DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC
    ]

    # Find events NOT overlapping with SWR
    ripple_times = data["ripple_times"]
    _, non_swr_events = get_overlap_times(non_local_times, ripple_times, results)

    if len(non_swr_events) == 0:
        raise ValueError("No non-local events without SWR overlap found.")

    # Pick longest event (likely the clearest example)
    best_idx = non_swr_events["duration"].idxmax()
    event = non_swr_events.loc[best_idx]

    center = (event.start_time + event.end_time) / 2
    return slice(center - window_sec, center + window_sec)


# ---------------------------------------------------------------------------
# Panel c — Venn diagram
# ---------------------------------------------------------------------------

def plot_venn_diagram(data_over_time, ax):
    """Plot Venn diagram showing timepoint percentages for non-local, SWR, MUA.

    Uses manual circle drawing for precise control over label placement.
    """
    is_immobile = data_over_time.head_speed < DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC
    df = data_over_time.loc[is_immobile]

    # Boolean masks
    swr = df["SWR_event"] > 0
    mua = df["MUA_event"] > 0
    non_local = df["non_local_event"] > 0

    # Venn regions
    only_swr = swr & ~mua & ~non_local
    only_mua = mua & ~swr & ~non_local
    only_non_local = non_local & ~swr & ~mua
    swr_and_mua = swr & mua & ~non_local
    swr_and_non_local = swr & non_local & ~mua
    mua_and_non_local = mua & non_local & ~swr
    all_three = swr & mua & non_local

    total = len(df)

    counts = {
        "SWR only": 100 * only_swr.sum() / total,
        "MUA only": 100 * only_mua.sum() / total,
        "NL only": 100 * only_non_local.sum() / total,
        "SWR+MUA": 100 * swr_and_mua.sum() / total,
        "SWR+NL": 100 * swr_and_non_local.sum() / total,
        "MUA+NL": 100 * mua_and_non_local.sum() / total,
        "All": 100 * all_three.sum() / total,
    }

    # Draw circles
    r = 0.22
    circle_swr = mpatches.Circle((0.38, 0.45), r, fc=COLORS["SWR"], alpha=0.3, ec="none")
    circle_mua = mpatches.Circle((0.62, 0.45), r, fc=COLORS["MUA"], alpha=0.3, ec="none")
    circle_nl = mpatches.Circle((0.50, 0.68), r, fc=COLORS["Non-Local"], alpha=0.3, ec="none")

    ax.add_patch(circle_swr)
    ax.add_patch(circle_mua)
    ax.add_patch(circle_nl)

    # Region labels
    fs = 5
    ax.text(0.25, 0.42, f'{counts["SWR only"]:.1f}%', ha="center", fontsize=fs)
    ax.text(0.75, 0.42, f'{counts["MUA only"]:.1f}%', ha="center", fontsize=fs)
    ax.text(0.50, 0.82, f'{counts["NL only"]:.1f}%', ha="center", fontsize=fs)
    ax.text(0.50, 0.52, f'{counts["All"]:.1f}%', ha="center", fontsize=fs)
    ax.text(0.50, 0.36, f'{counts["SWR+MUA"]:.1f}%', ha="center", fontsize=fs)
    ax.text(0.36, 0.62, f'{counts["SWR+NL"]:.1f}%', ha="center", fontsize=fs)
    ax.text(0.64, 0.62, f'{counts["MUA+NL"]:.1f}%', ha="center", fontsize=fs)

    # Circle labels
    ax.text(0.20, 0.30, "SWR", color=COLORS["SWR"], fontweight="bold", fontsize=6)
    ax.text(0.72, 0.30, "MUA", color=COLORS["MUA"], fontweight="bold", fontsize=6)
    ax.text(0.50, 0.93, "Non-Local", color=COLORS["Non-Local"],
            fontweight="bold", fontsize=6, ha="center")

    ax.set_xlim(0, 1)
    ax.set_ylim(0.15, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("% Immobile Time", fontsize=7, pad=2)


# ---------------------------------------------------------------------------
# Main Figure Assembly
# ---------------------------------------------------------------------------

def generate_figure(
    nwb_file_name="wilbur20210329_.nwb",
    epoch_name="02_r1",
    data_dir=PROCESSED_DATA_DIR,
):
    """Generate composite Figure 3.

    Parameters
    ----------
    nwb_file_name : str
        NWB file for the real data example panel.
    epoch_name : str
        Epoch name for the real data example panel.
    data_dir : Path
        Directory containing processed data files.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # --- Load data ---
    logging.info("Loading session data for example panel...")
    data = load_data(nwb_file_name, epoch_name)
    results, detector, env = load_results(
        nwb_file_name, epoch_name, decoding_brain_area="HPC"
    )

    logging.info("Loading aggregated data for statistics panels...")
    data_over_time = load_data_over_time(data_dir)

    # Find example non-SWR event
    logging.info("Selecting non-SWR example event...")
    time_slice = find_non_swr_example_event(data, results)

    # --- Create figure ---
    fig_height = min(7.5, PAGE_HEIGHT)
    fig = plt.figure(figsize=(TWO_COLUMN, fig_height), dpi=450,
                     constrained_layout=True)

    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.6, 1.0], hspace=0.35)

    # ── Row 1: Real data example (panel a) — 4 stacked sub-rows ──
    gs_a = gs[0].subgridspec(4, 1, hspace=0.05,
                              height_ratios=[0.6, 0.8, 1.2, 0.8])
    ax_swr = fig.add_subplot(gs_a[0])
    ax_raster = fig.add_subplot(gs_a[1], sharex=ax_swr)
    ax_posterior = fig.add_subplot(gs_a[2], sharex=ax_swr)
    ax_probs = fig.add_subplot(gs_a[3], sharex=ax_swr)

    plot_ripple_consensus_trace_zscore(data, time_slice, ax_swr)
    plot_spike_times(data, time_slice, detector, env, results, ax_raster)
    plot_conditional_non_local_posterior(
        data, time_slice, results, env, detector, ax_posterior
    )
    plot_acausal_probabilities(results, time_slice, ax_probs, plot_local=True)

    # Convert absolute timestamps to relative time (seconds from event start)
    t0 = time_slice.start
    for ax in [ax_swr, ax_raster, ax_posterior, ax_probs]:
        ticks = ax.get_xticks()
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t - t0:.1f}" for t in ticks])

    # Remove x-labels from non-bottom sub-rows
    for ax in [ax_swr, ax_raster, ax_posterior]:
        ax.tick_params(labelbottom=False)
    ax_probs.set_xlabel("Time (s)")

    # ── Row 2: Statistics panels (b, c, d, e) ──
    gs_bot = gs[1].subgridspec(1, 4, width_ratios=[1, 1.1, 1, 1], wspace=0.55)

    # Panel b: SWR z-score by state
    ax_b = fig.add_subplot(gs_bot[0])
    plot_swr_zscore_by_state(data_over_time, ax=ax_b)

    # Panel c: Venn diagram
    ax_c = fig.add_subplot(gs_bot[1])
    plot_venn_diagram(data_over_time, ax_c)

    # Panel d: Replay speed by SWR
    ax_d = fig.add_subplot(gs_bot[2])
    plot_replay_speed_by_swr(data_over_time, ax=ax_d)
    ax_d.set_xlim(0, 1500)  # Focus on meaningful range

    # Panel e: Replay position by SWR
    ax_e = fig.add_subplot(gs_bot[3])
    plot_replay_position_by_swr(data_over_time, ax=ax_e)

    # ── Panel Labels ──
    label_panels(
        [ax_swr, ax_b, ax_c, ax_d, ax_e],
        labels=["a", "b", "c", "d", "e"],
    )

    return fig


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("paper/figures/figure03"),
    )
    parser.add_argument("--nwb-file-name", type=str, default="chimi20200216_.nwb")
    parser.add_argument("--epoch-name", type=str, default="02_r1")
    parser.add_argument(
        "--data-dir", type=Path, default=PROCESSED_DATA_DIR,
    )
    parser.add_argument("--show", action="store_true")

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig = generate_figure(
        nwb_file_name=args.nwb_file_name,
        epoch_name=args.epoch_name,
        data_dir=args.data_dir,
    )

    logging.info(f"Saving to {args.output_dir}...")
    save_figure("figure03", base_dir=args.output_dir)

    if args.show:
        plt.show()
    else:
        plt.close("all")

    logging.info("Done.")


if __name__ == "__main__":
    main()
