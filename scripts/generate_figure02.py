#!/usr/bin/env python
"""Generate Figure 2: The model works on real data.

Composite figure with 7 panels:

    Row 1 (panel a — full width):
        Real data example showing ~2s around a non-local event during SWR.
        Stacked sub-rows: SWR z-score, spike raster, non-local posterior,
        state probabilities.

    Row 2 (panels b, c):
        b: Non-local probability distribution during immobility (bimodal,
           justifies 0.8 threshold)
        c: Event-triggered averages at event start (4 signals stacked:
           non-local prob, theta, MUA, SWR z-score)

    Row 3 (panels d, e, f, g):
        d: Calibration curve — decoding error vs. local probability
        e: Decoding error distribution during local state across sessions
        f: State probability by theta phase (doubled axis, movement only)
        g: State fractions during movement vs. immobility

Usage:
    python scripts/generate_figure02.py
    python scripts/generate_figure02.py --output-dir paper/figures/figure02
"""

import argparse
import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

import continuum_swr_replay.figure_utilities as figure_utilities
from continuum_swr_replay.analysis.constants import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    DEFAULT_NON_LOCAL_THRESHOLD,
)
from continuum_swr_replay.analysis.hippocampus import (
    get_confident_times,
    get_event_start_end_times,
    get_event_types,
    get_immobile_events,
    get_overlap_times,
    load_data_over_time,
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
    SAMPLING_FREQUENCY,
)
from continuum_swr_replay.visualization.figure_panels import (
    plot_calibration_curve,
    plot_decoding_error_distribution,
    plot_state_fractions_by_movement,
    plot_state_probability_by_theta_phase,
    plot_triggered_averages,
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
# Panel a — Real data example
# ---------------------------------------------------------------------------

def get_example_event_slice(data, results, event_type="non_local_SWR",
                            event_ind=None, window_sec=0.5):
    """Get a time slice around a specific non-local event.

    Parameters
    ----------
    data : dict
        Session data from load_data().
    results : xr.Dataset
        Decoding results.
    event_type : str
        One of "non_local_SWR", "non_local_no_SWR", "SWR".
    event_ind : int or None
        Index into the event DataFrame. If None, picks the longest event.
    window_sec : float
        Padding on each side of the event.

    Returns
    -------
    time_slice : slice
        Time window around the event.
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

    ripple_times = data["ripple_times"]
    swr_overlap, no_swr_events = get_overlap_times(
        non_local_times, ripple_times, results
    )

    event_map = {
        "non_local_SWR": swr_overlap,
        "non_local_no_SWR": no_swr_events,
        "SWR": ripple_times,
    }
    events = event_map[event_type]

    if len(events) == 0:
        raise ValueError(f"No {event_type} events found.")

    if event_ind is not None:
        event = events.iloc[event_ind]
    else:
        event = events.loc[events["duration"].idxmax()]

    return slice(event.start_time - window_sec, event.end_time + window_sec)


# ---------------------------------------------------------------------------
# Panel b — Non-local probability distribution
# ---------------------------------------------------------------------------

def plot_non_local_prob_distribution(data_over_time, ax):
    """Plot bimodal non-local probability distribution during immobility."""
    is_immobile = data_over_time.head_speed < DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC

    ax.hist(
        data_over_time.loc[is_immobile, "Non-Local_prob"],
        bins=50,
        color=COLORS["Non-Local"],
        alpha=0.7,
        density=True,
        edgecolor="none",
        range=(0, 1),
    )

    ax.axvline(
        DEFAULT_NON_LOCAL_THRESHOLD,
        color="black", linewidth=0.8, linestyle="--",
        label=f"{DEFAULT_NON_LOCAL_THRESHOLD} threshold",
    )

    ax.set_xlabel("Non-Local Probability")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 1)
    ax.legend()


# ---------------------------------------------------------------------------
# Main Figure Assembly
# ---------------------------------------------------------------------------

def generate_figure(
    nwb_file_name="wilbur20210329_.nwb",
    epoch_name="02_r1",
    data_dir=PROCESSED_DATA_DIR,
):
    """Generate composite Figure 2.

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
    event_types = get_event_types(data_over_time)

    # Select example event (longest SWR-coincident non-local event)
    logging.info("Selecting example event...")
    time_slice = get_example_event_slice(
        data, results, event_type="non_local_SWR",
    )

    # --- Create figure ---
    fig_height = PAGE_HEIGHT
    fig = plt.figure(figsize=(TWO_COLUMN, fig_height), dpi=450,
                     constrained_layout=True)

    gs = GridSpec(3, 1, figure=fig, height_ratios=[1.6, 1.2, 1.0], hspace=0.35)

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

    # ── Row 2: Statistics panels (b, c) ──
    gs_mid = gs[1].subgridspec(1, 2, width_ratios=[1.0, 1.4], wspace=0.35)

    # Panel b: probability distribution
    ax_b = fig.add_subplot(gs_mid[0])
    plot_non_local_prob_distribution(data_over_time, ax_b)

    # Panel c: triggered averages (4 stacked signals)
    gs_c = gs_mid[1].subgridspec(4, 1, hspace=0.08)
    ax_c = [fig.add_subplot(gs_c[i]) for i in range(4)]
    # Share x-axis across triggered average sub-panels
    for i in range(1, 4):
        ax_c[i].sharex(ax_c[0])
    for i in range(3):
        ax_c[i].tick_params(labelbottom=False)

    plot_triggered_averages(
        data_over_time, event_types,
        time_col="start_time", axes=ax_c,
    )

    # ── Row 3: Validation panels (d, e, f, g) ──
    gs_bot = gs[2].subgridspec(1, 4, wspace=0.5, width_ratios=[1.0, 1.0, 1.0, 1.3])

    ax_d = fig.add_subplot(gs_bot[0])
    plot_calibration_curve(data_over_time, ax=ax_d)

    ax_e = fig.add_subplot(gs_bot[1])
    plot_decoding_error_distribution(data_over_time, ax=ax_e)

    ax_f = fig.add_subplot(gs_bot[2])
    plot_state_probability_by_theta_phase(data_over_time, ax=ax_f)

    ax_g = fig.add_subplot(gs_bot[3])
    plot_state_fractions_by_movement(data_over_time, ax=ax_g)

    # ── Panel Labels ──
    label_panels(
        [ax_swr, ax_b, ax_c[0], ax_d, ax_e, ax_f, ax_g],
        labels=["a", "b", "c", "d", "e", "f", "g"],
    )

    return fig


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("paper/figures/figure02"),
    )
    parser.add_argument("--nwb-file-name", type=str, default="wilbur20210329_.nwb")
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
    save_figure("figure02", base_dir=args.output_dir)

    if args.show:
        plt.show()
    else:
        plt.close("all")

    logging.info("Done.")


if __name__ == "__main__":
    main()
