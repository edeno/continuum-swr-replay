#!/usr/bin/env python
"""Generate Figure 2b: Event-Triggered Averages.

This script creates peri-event time histograms (PETH) showing physiological signals
around non-local events. Generates:

1. Mean±SEM triggered averages (aligned to non-local event start)
2. Mean±SEM triggered averages (aligned to non-local event end)
3. Percentile-based triggered averages (aligned to event start)
4. Percentile-based triggered averages (aligned to event end)

Signals analyzed:
- Theta power (z-scored)
- Multi-unit activity (MUA, z-scored)
- Sharp-wave ripple power (SWR, z-scored)
- Head speed (cm/s)

Usage:
    python scripts/generate_figure02b_triggered_averages.py --output-dir paper/figures/figure02
    python scripts/generate_figure02b_triggered_averages.py --format png  # PNG only
"""

import argparse
import logging
import warnings
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import continuum_swr_replay.figure_utilities as figure_utilities
from continuum_swr_replay.analysis.constants import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    DEFAULT_SAMPLING_FREQUENCY_HZ,
    DEFAULT_WINDOW_EVENT_TRIGGERED_SEC,
)
from continuum_swr_replay.analysis.hippocampus import (
    get_event_start_end_times,
    get_immobile_events,
    get_signal_windows,
)
from continuum_swr_replay.figure_utilities import (
    get_figure_size,
    save_figure,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%d-%b-%y %H:%M:%S"
)

figure_utilities.set_figure_defaults(context="paper")
warnings.filterwarnings("ignore")


def load_data(data_dir: Path) -> pd.DataFrame:
    """Load and prepare processed data.

    Parameters
    ----------
    data_dir : Path
        Directory containing processed data files

    Returns
    -------
    pd.DataFrame
        Concatenated data with derived columns
    """
    data_path = data_dir.glob("*_HPC_80_data_over_time.pkl")
    data_over_time = pd.concat([pd.read_pickle(p) for p in data_path])

    # Add derived columns
    data_over_time["Non-Local_prob"] = (
        data_over_time["Non-Local_Continuous_prob"]
        + data_over_time["Non-Local_Fragmented_prob"]
    )
    data_over_time["distance"] = np.abs(data_over_time["ahead_behind_distance"])

    return data_over_time


def plot_triggered_average_mean(
    signal_windows: np.ndarray,
    time_axis_ms: np.ndarray,
    signals: list[str],
    titles: list[str],
    output_dir: Path,
    figure_name: str,
    xlabel: str = "Time from non-local event start (ms)",
    formats: list[str] | None = None,
    dpi: int = 300,
) -> None:
    """Plot event-triggered averages with mean ± SEM.

    Parameters
    ----------
    signal_windows : np.ndarray
        Shape (n_events, n_timepoints, n_signals)
    time_axis_ms : np.ndarray
        Time axis in milliseconds
    signals : list[str]
        Signal names (for indexing)
    titles : list[str]
        Signal titles (for y-axis labels)
    output_dir : Path
        Output directory
    figure_name : str
        Base name for the figure (without extension)
    xlabel : str
        X-axis label
    formats : list of str, optional
        Output formats (default: ["pdf", "png"])
    dpi : int
        DPI for raster formats
    """
    fig, axes = plt.subplots(
        4, 1, figsize=get_figure_size(4, 1), sharex=True, constrained_layout=True
    )

    for i, (signal, title) in enumerate(zip(signals, titles, strict=True)):
        ax = axes[i]

        # Extract signal column
        signal_idx = signals.index(signal)
        signal_data = signal_windows[:, :, signal_idx]

        # Compute mean and SEM
        mean_y = np.nanmean(signal_data, axis=0)
        sem_y = np.nanstd(signal_data, ddof=1, axis=0) / np.sqrt(
            np.sum(~np.isnan(signal_data), axis=0)
        )

        # Plot
        ax.plot(time_axis_ms, mean_y, color="black", lw=1)
        ax.fill_between(
            time_axis_ms,
            mean_y - sem_y,
            mean_y + sem_y,
            color="gray",
            alpha=0.3,
            linewidth=0,
        )

        # Style
        sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
        ax.grid(False)
        ax.set_axisbelow(True)

        ax.set_ylabel(title, labelpad=2)
        ax.set_xlim(time_axis_ms[0], time_axis_ms[-1])
        ax.axvline(0, color="k", linestyle="--", lw=0.5)

        if signal != "head_speed":
            ax.set_ylim((-2, 2))
            ax.axhline(0, color="k", linestyle="--", lw=0.5)
        else:
            ax.set_ylim((0, 5))
            ax.axhline(
                DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
                color="k",
                linestyle="--",
                lw=0.5,
            )

        ax.tick_params(axis="both", which="both", length=2, width=0.5, labelsize=7)
        ax.set_title("", pad=2)

    axes[-1].set_xlabel(xlabel, labelpad=2)
    plt.subplots_adjust(hspace=0.15)

    save_figure(plt.gcf(), figure_name, base_dir=output_dir, dpi=dpi, formats=formats)
    plt.close(fig)
    logging.info(f"Saved: {figure_name}")


def plot_triggered_average_percentiles(
    signal_windows: np.ndarray,
    time_axis_ms: np.ndarray,
    signals: list[str],
    titles: list[str],
    output_dir: Path,
    figure_name: str,
    xlabel: str = "Time from non-local event start (ms)",
    formats: list[str] | None = None,
    dpi: int = 300,
) -> None:
    """Plot event-triggered averages with percentiles.

    Parameters
    ----------
    signal_windows : np.ndarray
        Shape (n_events, n_timepoints, n_signals)
    time_axis_ms : np.ndarray
        Time axis in milliseconds
    signals : list[str]
        Signal names (for indexing)
    titles : list[str]
        Signal titles (for y-axis labels)
    output_dir : Path
        Output directory
    figure_name : str
        Base name for the figure (without extension)
    xlabel : str
        X-axis label
    formats : list of str, optional
        Output formats (default: ["pdf", "png"])
    dpi : int
        DPI for raster formats
    """
    fig, axes = plt.subplots(
        4, 1, figsize=get_figure_size(4, 1), sharex=True, constrained_layout=True
    )

    for i, (signal, title) in enumerate(zip(signals, titles, strict=True)):
        ax = axes[i]

        # Extract signal column
        signal_idx = signals.index(signal)
        signal_data = signal_windows[:, :, signal_idx]

        # Compute percentiles
        pct_y = np.nanpercentile(signal_data, axis=0, q=[5, 25, 50, 75, 95])

        # Plot
        ax.fill_between(
            time_axis_ms,
            pct_y[0],
            pct_y[4],
            color="lightgray",
            alpha=0.3,
            linewidth=0,
            label="5th-95th Percentile",
        )
        ax.fill_between(
            time_axis_ms,
            pct_y[1],
            pct_y[3],
            color="gray",
            alpha=0.3,
            linewidth=0,
            label="25th-75th Percentile",
        )
        ax.plot(time_axis_ms, pct_y[2], color="black", lw=1)

        # Style
        sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
        ax.grid(False)
        ax.set_axisbelow(True)

        ax.set_ylabel(title, labelpad=2)
        ax.set_xlim(time_axis_ms[0], time_axis_ms[-1])
        ax.axvline(0, color="k", linestyle="--", lw=0.5)

        if signal != "head_speed":
            ax.set_ylim((-5, 10))
            ax.axhline(0, color="k", linestyle="--", lw=0.5)
        else:
            ax.set_ylim((0, 5))
            ax.axhline(
                DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
                color="k",
                linestyle="--",
                lw=0.5,
            )

        ax.tick_params(axis="both", which="both", length=2, width=0.5, labelsize=7)
        ax.set_title("", pad=2)

    axes[-1].set_xlabel(xlabel, labelpad=2)
    axes[0].legend(
        loc="upper right",
        bbox_to_anchor=(1.15, 1),
        frameon=False,
        title="Percentiles",
        handlelength=1.5,
        handletextpad=0.5,
    )
    plt.subplots_adjust(hspace=0.15)

    save_figure(plt.gcf(), figure_name, base_dir=output_dir, dpi=dpi, formats=formats)
    plt.close(fig)
    logging.info(f"Saved: {figure_name}")


def main(argv: Sequence[str] | None = None) -> None:
    """Main entry point for figure generation."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/figures/figure02"),
        help="Output directory for figures (default: paper/figures/figure02)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["pdf", "png", "svg"],
        default=None,
        help="Output format (default: both pdf and png)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for raster formats (default: 300)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/cumulus/edeno/non_local_paper/Processed-Data"),
        help="Directory containing processed data (default: /cumulus/edeno/non_local_paper/Processed-Data)",
    )

    args = parser.parse_args(argv)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logging.info("Loading data...")
    data_over_time = load_data(args.data_dir)

    # Get immobile non-local events
    logging.info("Filtering immobile non-local events...")
    immobile_non_local = get_immobile_events(
        data_over_time,
        event_col="non_local_event",
        speed_col="head_speed",
        speed_threshold=DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
        method="endpoints",
    )

    # Get event start and end times
    logging.info("Extracting event times...")
    event_times = get_event_start_end_times(
        immobile_non_local, event_col="non_local_event"
    )

    # Define signals to analyze
    signals = ["theta_power_zscore", "MUA_zscore", "SWR_zscore", "head_speed"]
    titles = ["Theta Power", "MUA", "SWR", "Head Speed"]

    # Extract signal windows around event starts
    logging.info("Extracting signal windows around event starts...")
    start_windows = get_signal_windows(
        data_over_time,
        event_times,
        signal_cols=signals,
        time_col="start_time",
        window_sec=2 * DEFAULT_WINDOW_EVENT_TRIGGERED_SEC,  # 0.5 s total
        sampling_frequency=DEFAULT_SAMPLING_FREQUENCY_HZ,
    )

    # Extract signal windows around event ends
    logging.info("Extracting signal windows around event ends...")
    end_windows = get_signal_windows(
        data_over_time,
        event_times,
        signal_cols=signals,
        time_col="end_time",
        window_sec=2 * DEFAULT_WINDOW_EVENT_TRIGGERED_SEC,  # 0.5 s total
        sampling_frequency=DEFAULT_SAMPLING_FREQUENCY_HZ,
    )

    # Get time axis in milliseconds
    time_axis_ms = start_windows.coords["time"].values * 1000

    # Generate figures
    logging.info("Generating triggered average plots...")
    formats = [args.format] if args.format else None

    # 1. Mean±SEM aligned to event start
    plot_triggered_average_mean(
        start_windows.values,
        time_axis_ms,
        signals,
        titles,
        args.output_dir,
        "fig02b_triggered_mean_start",
        xlabel="Time from non-local event start (ms)",
        formats=formats,
        dpi=args.dpi,
    )

    # 2. Mean±SEM aligned to event end
    plot_triggered_average_mean(
        end_windows.values,
        time_axis_ms,
        signals,
        titles,
        args.output_dir,
        "fig02b_triggered_mean_end",
        xlabel="Time from non-local event end (ms)",
        formats=formats,
        dpi=args.dpi,
    )

    # 3. Percentiles aligned to event start
    plot_triggered_average_percentiles(
        start_windows.values,
        time_axis_ms,
        signals,
        titles,
        args.output_dir,
        "fig02b_triggered_percentiles_start",
        xlabel="Time from non-local event start (ms)",
        formats=formats,
        dpi=args.dpi,
    )

    # 4. Percentiles aligned to event end
    plot_triggered_average_percentiles(
        end_windows.values,
        time_axis_ms,
        signals,
        titles,
        args.output_dir,
        "fig02b_triggered_percentiles_end",
        xlabel="Time from non-local event end (ms)",
        formats=formats,
        dpi=args.dpi,
    )

    logging.info("Done!")


if __name__ == "__main__":
    main()
