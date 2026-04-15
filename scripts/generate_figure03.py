#!/usr/bin/env python
"""Generate Figure 3: Non-local events exist outside detected SWRs.

Composite figure with 5 panels:

    Row 1 (panel a — full width):
        Real data example showing ~2s around a non-local event WITHOUT
        SWR. Stacked sub-rows: SWR z-score (flat), spike raster,
        non-local posterior (shows clear replay), state probabilities.

    Row 2 (panels b, c, d, e):
        b: 2-circle Venn of SWR events and immobile non-local events.
           Answers "how often do these populations overlap?" and
           shows the SWR→NL vs NL→SWR asymmetry that motivates the
           rest of the row.
        c: SWR z-score density for Local vs Non-Local timepoints —
           shows non-local activity lives at low SWR power, not just
           barely below the SWR threshold.
        d: Replay speed by SWR status — low-SWR replay moves more
           slowly.
        e: Replay position by SWR status — non-SWR replay represents
           reward zones.

Panel order was reshuffled in April 2026: the Venn is now panel b
(first statistic after the example) rather than panel c, because
"how often does this happen?" is the reader's first question.

Usage:
    python scripts/generate_figure03.py
    python scripts/generate_figure03.py --output-dir paper/figures/figure03
"""

import argparse
import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import continuum_swr_replay.figure_utilities as figure_utilities
from continuum_swr_replay.analysis.constants import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    DEFAULT_NON_LOCAL_THRESHOLD,
)
from continuum_swr_replay.analysis.hippocampus import (
    compute_swr_nl_event_counts,
    get_confident_times,
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
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR
from continuum_swr_replay.visualization.figure_panels import (
    plot_replay_position_by_swr,
    plot_replay_speed_by_swr,
    plot_swr_nl_venn_2circle,
    plot_swr_zscore_by_state,
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
# Main Figure Assembly
# ---------------------------------------------------------------------------

def generate_figure(
    # TODO: swap to a non-wilbur session so Fig 3a comes from a different
    # animal than Fig 2a (cross-cutting narrative requirement). Candidates
    # from other animals will be evaluated separately.
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
    fig_height = min(8.5, PAGE_HEIGHT)
    fig = plt.figure(figsize=(TWO_COLUMN, fig_height), dpi=450,
                     constrained_layout=True)

    # 3 rows: (a) example stack, (b) Venn in its own row, (c/d/e) stats
    gs = GridSpec(3, 1, figure=fig, height_ratios=[1.8, 0.9, 1.1], hspace=0.35)

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

    # ── Row 2: panel b — Venn in its own row so it has space to breathe ──
    # Panel order was reshuffled so the reader's first question after
    # panel a ("how often does this happen?") is answered immediately
    # by the Venn in panel b. Old panel b (SWR z-score by state)
    # becomes the new panel c.
    logging.info("Computing SWR / non-local event Venn counts...")
    venn_counts = compute_swr_nl_event_counts(data_over_time)
    logging.info(
        f"  SWR: {venn_counts['n_swr']:,} total, "
        f"{venn_counts['n_swr_with_nl']:,} with NL"
    )
    logging.info(
        f"  NL (immobile): {venn_counts['n_nl_immobile']:,} total, "
        f"{venn_counts['n_nl_immobile_with_swr']:,} with SWR"
    )
    # Center the Venn within its row, leaving left/right margins so
    # the circles don't stretch full width.
    gs_b = gs[1].subgridspec(1, 3, width_ratios=[1.0, 2.0, 1.0])
    ax_b = fig.add_subplot(gs_b[0, 1])
    plot_swr_nl_venn_2circle(venn_counts, ax=ax_b)

    # ── Row 3: panels c, d, e (stats panels) ──
    gs_cde = gs[2].subgridspec(1, 3, wspace=0.5)

    # Panel c: SWR z-score by state (was old panel b)
    ax_c = fig.add_subplot(gs_cde[0])
    plot_swr_zscore_by_state(data_over_time, ax=ax_c)

    # Panel d: Replay speed by SWR
    ax_d = fig.add_subplot(gs_cde[1])
    plot_replay_speed_by_swr(data_over_time, ax=ax_d)
    ax_d.set_xlim(0, 1500)  # Focus on meaningful range

    # Panel e: Replay position by SWR
    ax_e = fig.add_subplot(gs_cde[2])
    plot_replay_position_by_swr(data_over_time, ax=ax_e)

    # ── Panel Labels ──
    label_panels(
        [ax_swr, ax_b, ax_c, ax_d, ax_e],
        labels=["a", "b", "c", "d", "e"],
        xpos=-0.08,
        ypos=1.15,
        fontsize=11,
    )

    return fig


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("paper/figures/figure03"),
    )
    parser.add_argument("--nwb-file-name", type=str, default="wilbur20210329_.nwb")
    parser.add_argument("--epoch-name", type=str, default="02_r1")
    parser.add_argument(
        "--data-dir", type=Path, default=PROCESSED_DATA_DIR,
    )
    parser.add_argument("--show", action="store_true")

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    generate_figure(
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
