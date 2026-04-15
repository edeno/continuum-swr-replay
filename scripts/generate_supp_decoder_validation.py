#!/usr/bin/env python
"""Generate Supplementary Figure: Decoder validation.

Collects the "is the decoder well-behaved?" panels that were pushed out
of the main Figure 2 when it was refocused on the SWR/non-local
relationship. None of these are first-order scientific claims — they
are positive controls that support the rest of the paper.

Panels:

    a — Non-local probability distribution during immobility.
        Bimodality justifies the 0.8 threshold used throughout the paper.

    b — State fractions moving vs. immobile.
        Local and No-Spike dominate during movement; Non-Local fragments
        appear during immobility — the sanity check for the state labels.

    c — Calibration curve: decoding error vs. local probability.
        Error falls as the decoder's confidence in Local state rises.

    d — Event duration distributions (non-local, SWR, MUA).
        Non-local events have a different duration profile from SWR/MUA
        events, which the main text alludes to but does not visualise.

    e — Full four-signal event-triggered averages.
        The reduced two-signal version sits in main Figure 2c; here we
        show theta and MUA alongside non-local probability and SWR
        z-score for completeness.

Usage
-----
    python scripts/generate_supp_decoder_validation.py
"""

from __future__ import annotations

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
    get_event_types,
    load_data_over_time,
)
from continuum_swr_replay.figure_utilities import (
    PAGE_HEIGHT,
    TWO_COLUMN,
    label_panels,
    save_figure,
)
from continuum_swr_replay.parameters import COLORS, PROCESSED_DATA_DIR
from continuum_swr_replay.visualization.figure_panels import (
    plot_calibration_curve,
    plot_event_duration_distribution,
    plot_state_fractions_by_movement,
    plot_triggered_averages,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
figure_utilities.set_figure_defaults(context="paper")
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Panel a — Threshold distribution (inline; only used here)
# ---------------------------------------------------------------------------

def plot_non_local_prob_distribution(data_over_time, ax) -> None:
    """Bimodal non-local probability distribution during immobility.

    Used to justify the 0.8 threshold applied throughout the paper.
    """
    is_immobile = (
        data_over_time.head_speed < DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC
    )
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
    ax.set_xlabel("Non-local probability\n(immobile timepoints)")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=6, frameon=False)


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def generate_figure(data_dir: Path = PROCESSED_DATA_DIR):
    logging.info("Loading aggregated data...")
    data_over_time = load_data_over_time(data_dir)
    event_types = get_event_types(data_over_time)

    fig_height = min(9.0, PAGE_HEIGHT)
    fig = plt.figure(
        figsize=(TWO_COLUMN, fig_height), dpi=450, constrained_layout=True,
    )
    # Row 1: a (threshold) | b (state fractions)
    # Row 2: c (calibration) | d (durations)
    # Row 3: e (full 4-signal triggered averages, stacked sub-rows)
    outer = GridSpec(
        3, 1, figure=fig,
        height_ratios=[1.0, 1.0, 1.6],
        hspace=0.4,
    )

    # ── Row 1 ──
    gs_r1 = outer[0].subgridspec(1, 2, wspace=0.35)
    ax_a = fig.add_subplot(gs_r1[0])
    plot_non_local_prob_distribution(data_over_time, ax_a)

    ax_b = fig.add_subplot(gs_r1[1])
    plot_state_fractions_by_movement(data_over_time, ax=ax_b)

    # ── Row 2 ──
    gs_r2 = outer[1].subgridspec(1, 2, wspace=0.35)
    ax_c = fig.add_subplot(gs_r2[0])
    plot_calibration_curve(data_over_time, ax=ax_c)

    ax_d = fig.add_subplot(gs_r2[1])
    plot_event_duration_distribution(event_types, ax=ax_d)

    # ── Row 3: full 4-signal triggered averages ──
    gs_r3 = outer[2].subgridspec(4, 1, hspace=0.08)
    ax_e = [fig.add_subplot(gs_r3[i]) for i in range(4)]
    for ax in ax_e[1:]:
        ax.sharex(ax_e[0])
    for ax in ax_e[:-1]:
        ax.tick_params(labelbottom=False)
    plot_triggered_averages(
        data_over_time,
        event_types,
        time_col="start_time",
        axes=ax_e,
        signals=[
            "Non-Local_prob",
            "theta_power_zscore",
            "MUA_zscore",
            "SWR_zscore",
        ],
    )

    # ── Panel labels ──
    label_panels(
        [ax_a, ax_b, ax_c, ax_d, ax_e[0]],
        labels=["a", "b", "c", "d", "e"],
        xpos=-0.08,
        ypos=1.15,
        fontsize=11,
    )

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("paper/figures/supp_decoder_validation"),
    )
    parser.add_argument("--data-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--show", action="store_true")

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    generate_figure(data_dir=args.data_dir)

    logging.info(f"Saving to {args.output_dir}...")
    save_figure("supp_decoder_validation", base_dir=args.output_dir)

    if args.show:
        plt.show()
    else:
        plt.close("all")

    logging.info("Done.")


if __name__ == "__main__":
    main()
