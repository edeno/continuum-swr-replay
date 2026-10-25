#!/usr/bin/env python
"""Generate Figure 3: Non-local event characteristics and physiological correlates.

This script generates panels showing:
- Event duration distributions
- Spatial properties (distance, HPD)
- Decoded position metrics
- Event-triggered averages of physiological signals

Usage:
    python scripts/figures/generate_figure03.py
    python scripts/figures/generate_figure03.py --output-dir paper/figures/figure03
    python scripts/figures/generate_figure03.py --format png  # PNG only
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
from scipy.stats import sem

import continuum_swr_replay.figure_utilities as figure_utilities
from continuum_swr_replay.analysis.constants import (
    BINS_DISTANCE_CM,
    BINS_DURATION_MS,
    BINS_HPD_CM,
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
)
from continuum_swr_replay.analysis.hippocampus import (
    get_event_durations_ms,
    get_event_start_end_times,
    get_immobile_events,
    get_signal_windows,
)
from continuum_swr_replay.figure_utilities import (
    FIGURE_TWO_PANEL_HORIZONTAL,
    PANEL_SINGLE,
    get_figure_size,
    save_figure,
)
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR, SAMPLING_FREQUENCY
from continuum_swr_replay.visualization.palettes import COLORS

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%d-%b-%y %H:%M:%S"
)

figure_utilities.set_figure_defaults(context="paper")
warnings.filterwarnings("ignore")


def load_data_over_time(data_dir: Path) -> pd.DataFrame:
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
    data_over_time["Non-Local_prob"] = (
        data_over_time["Non-Local_Continuous_prob"]
        + data_over_time["Non-Local_Fragmented_prob"]
    )
    data_over_time["distance"] = np.abs(data_over_time["ahead_behind_distance"])

    return data_over_time


def get_event_types(data_over_time: pd.DataFrame) -> dict:
    """Extract different event type subsets from the data.

    Parameters
    ----------
    data_over_time : pd.DataFrame
        Full dataset with event labels

    Returns
    -------
    dict
        Dictionary mapping event type names to filtered DataFrames
    """
    non_local_immobile = get_immobile_events(
        data_over_time,
        event_col="non_local_event",
        speed_col="head_speed",
        speed_threshold=DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
        method="endpoints",
    )

    swr_events = data_over_time.loc[(data_over_time["SWR_event"] > 0)]
    mua_events = data_over_time.loc[(data_over_time["MUA_event"] > 0)]

    non_local_immobile_overlap_swr = non_local_immobile.groupby(
        ["animal", "day", "epoch", "non_local_event"]
    ).filter(lambda g: (g["SWR_event"] > 0).any())
    non_local_immobile_overlap_mua = non_local_immobile.groupby(
        ["animal", "day", "epoch", "non_local_event"]
    ).filter(lambda g: (g["MUA_event"] > 0).any())

    non_local_immobile_no_overlap = non_local_immobile.groupby(
        ["animal", "day", "epoch", "non_local_event"]
    ).filter(lambda g: (g["SWR_event"] == 0).all() & (g["MUA_event"] == 0).all())
    non_local_immobile_no_overlap_SWR = non_local_immobile.groupby(
        ["animal", "day", "epoch", "non_local_event"]
    ).filter(lambda g: (g["SWR_event"] == 0).all())
    non_local_immobile_no_overlap_MUA = non_local_immobile.groupby(
        ["animal", "day", "epoch", "non_local_event"]
    ).filter(lambda g: (g["MUA_event"] == 0).all())
    swr_events_no_non_local = swr_events.groupby(
        ["animal", "day", "epoch", "SWR_event"]
    ).filter(lambda g: (g["non_local_event"] == 0).all())
    mua_events_no_non_local = mua_events.groupby(
        ["animal", "day", "epoch", "MUA_event"]
    ).filter(lambda g: (g["non_local_event"] == 0).all())

    return {
        "non_local_immobile": non_local_immobile,
        "swr_events": swr_events,
        "mua_events": mua_events,
        "non_local_immobile_overlap_swr": non_local_immobile_overlap_swr,
        "non_local_immobile_overlap_mua": non_local_immobile_overlap_mua,
        "non_local_immobile_no_overlap": non_local_immobile_no_overlap,
        "non_local_immobile_no_overlap_SWR": non_local_immobile_no_overlap_SWR,
        "non_local_immobile_no_overlap_MUA": non_local_immobile_no_overlap_MUA,
        "swr_events_no_non_local": swr_events_no_non_local,
        "mua_events_no_non_local": mua_events_no_non_local,
    }


def plot_event_duration_distribution(
    event_types: dict,
    output_dir: Path | None = None,
    figure_name: str | None = None,
    formats: list[str] | None = None,
    dpi: int = 300,
    ax: plt.Axes | None = None,
) -> None:
    """Plot event duration distribution comparing all vs no-SWR overlap.

    Parameters
    ----------
    event_types : dict
        Dictionary of event type DataFrames
    output_dir : Path, optional
        Output directory (standalone mode)
    figure_name : str, optional
        Base name for the figure (standalone mode)
    formats : list of str, optional
        Output formats
    dpi : int
        DPI for raster formats
    ax : matplotlib Axes, optional
        Axes to plot on. If None, creates standalone figure.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=PANEL_SINGLE)

    for event_type in ("non_local_immobile", "non_local_immobile_no_overlap_SWR"):
        event_times = get_event_start_end_times(
            event_types[event_type], event_col="non_local_event"
        )
        duration = get_event_durations_ms(event_times)

        color = (
            COLORS["Non-Local"]
            if event_type == "non_local_immobile_no_overlap_SWR"
            else COLORS["All Events"]
        )
        label = "No SWR Overlap" if "no_overlap" in event_type else "All Events"
        ax.hist(
            duration,
            bins=BINS_DURATION_MS,
            color=color,
            alpha=0.7,
            edgecolor="none",
            density=True,
            label=label,
        )

    ax.set_xlabel("Event Duration (ms)")
    ax.set_ylabel("Density")
    ax.legend()

    if standalone:
        sns.despine(ax=ax)
        plt.tight_layout()
        save_figure(figure_name, base_dir=output_dir, dpi=dpi, formats=formats)
        plt.close(fig)
        logging.info(f"Saved: {figure_name}")


def plot_hpd_distribution(
    event_types: dict,
    output_dir: Path,
    figure_name: str,
    formats: list[str] | None = None,
    dpi: int = 300,
) -> None:
    """Plot HPD (highest posterior density) distribution.

    Parameters
    ----------
    event_types : dict
        Dictionary of event type DataFrames
    output_dir : Path
        Output directory
    figure_name : str
        Base name for the figure
    formats : list of str, optional
        Output formats
    dpi : int
        DPI for raster formats
    """
    fig, ax = plt.subplots(figsize=PANEL_SINGLE)
    for event_type in ("non_local_immobile", "non_local_immobile_no_overlap_SWR"):
        color = (
            COLORS["Non-Local"]
            if event_type == "non_local_immobile_no_overlap_SWR"
            else "lightgrey"
        )
        label = "No SWR Overlap" if "no_overlap" in event_type else "All Events"
        ax.hist(
            event_types[event_type]["hpd"],
            bins=BINS_HPD_CM,
            color=color,
            alpha=0.7,
            edgecolor="none",
            density=True,
            label=label,
        )

    ax.set_xlabel("HPD (cm)")
    ax.set_ylabel("Density")
    ax.legend()
    sns.despine(ax=ax)
    plt.tight_layout()

    save_figure(figure_name, base_dir=output_dir, dpi=dpi, formats=formats)
    plt.close(fig)
    logging.info(f"Saved: {figure_name}")


def plot_distance_distribution(
    event_types: dict,
    output_dir: Path,
    figure_name: str,
    formats: list[str] | None = None,
    dpi: int = 300,
) -> None:
    """Plot distance from animal distribution.

    Parameters
    ----------
    event_types : dict
        Dictionary of event type DataFrames
    output_dir : Path
        Output directory
    figure_name : str
        Base name for the figure
    formats : list of str, optional
        Output formats
    dpi : int
        DPI for raster formats
    """
    fig, ax = plt.subplots(figsize=PANEL_SINGLE)
    for event_type in ("non_local_immobile", "non_local_immobile_no_overlap_SWR"):
        color = (
            COLORS["Non-Local"]
            if event_type == "non_local_immobile_no_overlap_SWR"
            else "lightgrey"
        )
        label = "No SWR Overlap" if "no_overlap" in event_type else "All Events"
        ax.hist(
            np.abs(event_types[event_type]["ahead_behind_distance"]),
            bins=BINS_DISTANCE_CM,
            color=color,
            alpha=0.7,
            edgecolor="none",
            density=True,
            label=label,
        )

    ax.set_xlabel("Distance from Animal (cm)")
    ax.set_ylabel("Density")
    ax.legend()
    sns.despine(ax=ax)
    plt.tight_layout()

    save_figure(figure_name, base_dir=output_dir, dpi=dpi, formats=formats)
    plt.close(fig)
    logging.info(f"Saved: {figure_name}")


def plot_mean_hpd_per_event(
    event_types: dict,
    output_dir: Path,
    figure_name: str,
    formats: list[str] | None = None,
    dpi: int = 300,
) -> None:
    """Plot mean HPD per event distribution.

    Parameters
    ----------
    event_types : dict
        Dictionary of event type DataFrames
    output_dir : Path
        Output directory
    figure_name : str
        Base name for the figure
    formats : list of str, optional
        Output formats
    dpi : int
        DPI for raster formats
    """
    fig, ax = plt.subplots(figsize=PANEL_SINGLE)
    for event_type in ("non_local_immobile", "non_local_immobile_no_overlap_SWR"):
        mean_event = (
            event_types[event_type]
            .groupby(["animal", "day", "epoch", "non_local_event"])
            .mean(numeric_only=True)
        )

        color = (
            COLORS["Non-Local"]
            if event_type == "non_local_immobile_no_overlap_SWR"
            else "lightgrey"
        )
        label = "No SWR Overlap" if "no_overlap" in event_type else "All Events"
        ax.hist(
            mean_event["hpd"],
            bins=BINS_HPD_CM,
            color=color,
            alpha=0.7,
            edgecolor="none",
            density=True,
            label=label,
        )

    ax.set_xlabel("Mean HPD per Event (cm)")
    ax.set_ylabel("Density")
    ax.legend()
    sns.despine(ax=ax)
    plt.tight_layout()

    save_figure(figure_name, base_dir=output_dir, dpi=dpi, formats=formats)
    plt.close(fig)
    logging.info(f"Saved: {figure_name}")


def plot_position_animal_vs_replay(
    event_types: dict,
    output_dir: Path,
    figure_name: str,
    formats: list[str] | None = None,
    dpi: int = 300,
) -> None:
    """Plot 2D histogram of animal position vs replay position.

    Parameters
    ----------
    event_types : dict
        Dictionary of event type DataFrames
    output_dir : Path
        Output directory
    figure_name : str
        Base name for the figure
    formats : list of str, optional
        Output formats
    dpi : int
        DPI for raster formats
    """
    fig, axes = plt.subplots(1, 2, figsize=FIGURE_TWO_PANEL_HORIZONTAL, sharex=True, sharey=True)

    for event_type, ax in zip(
        ("non_local_immobile", "non_local_immobile_no_overlap_SWR"), axes
    ):
        title = "No SWR Overlap" if "no_overlap" in event_type else "All Events"
        data = event_types[event_type]
        ax.hist2d(
            data["linear_position"] / data["max_linear_position"],
            data["map_estimate"] / data["max_linear_position"],
            bins=(np.arange(0, 1.01, 0.02), np.arange(0, 1.01, 0.02)),
            density=True,
            vmin=0.0,
            vmax=40.0,
            cmap="viridis",
        )
        ax.set_xlabel("Normalized Animal Position")
        ax.set_ylabel("Normalized Replay Position")
        ax.set_title(title)
        ax.plot([0, 1], [0, 1], "w--", alpha=0.5, linewidth=1)

    plt.tight_layout()

    save_figure(figure_name, base_dir=output_dir, dpi=dpi, formats=formats)
    plt.close(fig)
    logging.info(f"Saved: {figure_name}")


def plot_triggered_averages(
    data_over_time: pd.DataFrame,
    event_types: dict,
    output_dir: Path | None = None,
    figure_name: str | None = None,
    time_col: str = "start_time",
    formats: list[str] | None = None,
    dpi: int = 300,
    axes: list[plt.Axes] | None = None,
) -> None:
    """Plot event-triggered averages of physiological signals.

    Parameters
    ----------
    data_over_time : pd.DataFrame
        Full dataset
    event_types : dict
        Dictionary of event type DataFrames
    output_dir : Path, optional
        Output directory (standalone mode)
    figure_name : str, optional
        Base name for the figure (standalone mode)
    time_col : str
        Column to use for alignment ('start_time' or 'end_time')
    formats : list of str, optional
        Output formats
    dpi : int
        DPI for raster formats
    axes : list of matplotlib Axes, optional
        List of 4 axes for the 4 signals. If None, creates standalone figure.
    """
    signal_types = [
        "Non-Local_prob",
        "theta_power_zscore",
        "MUA_zscore",
        "SWR_zscore",
    ]

    # Get signal windows for no-overlap events
    non_local_start_end_time = get_event_start_end_times(
        event_types["non_local_immobile_no_overlap_SWR"], event_col="non_local_event"
    )
    signal_windows = get_signal_windows(
        data=data_over_time,
        event_times=non_local_start_end_time,
        signal_cols=signal_types,
        time_col=time_col,
        window_sec=2.0,
        sampling_frequency=SAMPLING_FREQUENCY,
        event_id_col="non_local_event",
    )

    # Get signal windows for overlap events
    non_local_start_end_time_overlap = get_event_start_end_times(
        event_types["non_local_immobile_overlap_swr"], event_col="non_local_event"
    )
    signal_windows_overlap = get_signal_windows(
        data=data_over_time,
        event_times=non_local_start_end_time_overlap,
        signal_cols=signal_types,
        time_col=time_col,
        window_sec=2.0,
        sampling_frequency=SAMPLING_FREQUENCY,
        event_id_col="non_local_event",
    )

    standalone = axes is None
    n_signals = len(signal_types)
    if standalone:
        fig, axes = plt.subplots(
            nrows=n_signals,
            ncols=1,
            figsize=get_figure_size(n_signals, 1),
            sharex=True,
            constrained_layout=True,
        )
        axes = list(axes.flatten())

    no_overlap_mean_all = signal_windows.mean(dim="event")
    overlap_mean_all = signal_windows_overlap.mean(dim="event")

    for i, (ax, signal_type) in enumerate(zip(axes, signal_types)):
        no_overlap_mean = no_overlap_mean_all.sel(signal=signal_type)
        overlap_mean = overlap_mean_all.sel(signal=signal_type)

        # Compute SEM for this specific signal
        no_overlap_data = signal_windows.sel(signal=signal_type)
        overlap_data = signal_windows_overlap.sel(signal=signal_type)
        no_overlap_sem = sem(no_overlap_data.values, axis=0, nan_policy="omit")
        overlap_sem = sem(overlap_data.values, axis=0, nan_policy="omit")

        time = no_overlap_mean["time"].values

        # Add shaded error bars
        ax.fill_between(
            time,
            no_overlap_mean.values - no_overlap_sem,
            no_overlap_mean.values + no_overlap_sem,
            color=COLORS["Non-Local"],
            alpha=0.3,
        )
        ax.fill_between(
            time,
            overlap_mean.values - overlap_sem,
            overlap_mean.values + overlap_sem,
            color=COLORS["SWR"],
            alpha=0.3,
        )

        ax.plot(
            time, no_overlap_mean.values,
            label="No SWR Overlap", color=COLORS["Non-Local"],
        )
        ax.plot(
            time, overlap_mean.values,
            label="SWR Overlap", color=COLORS["SWR"],
        )

        # Set ylabel based on signal type
        if signal_type.endswith("_zscore"):
            name = signal_type.replace("_zscore", "").replace("_", " ").title()
            if name == "Theta Power":
                name = "Theta"
            ax.set_ylabel(f"{name}\nZ-score")
            ax.axhline(2.0, color="lightgrey", linestyle="--", alpha=0.5, linewidth=0.75)
        elif signal_type == "Non-Local_prob":
            ax.set_ylabel("Non-Local\nProb.")
            ax.axhline(
                0.8, color="lightgrey", linestyle="--", alpha=0.5, linewidth=0.75
            )
            ax.set_ylim(0.0, 1.0)

        ax.axvline(0.0, color="lightgrey", linestyle="--", alpha=0.5, linewidth=0.75)
        ax.axhline(0.0, color="lightgrey", linestyle="--", alpha=0.5, linewidth=0.75)

        if i == 0:
            ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time from Event (s)")
    axes[-1].set_xlim(-1.0, 1.0)

    if standalone:
        save_figure(figure_name, base_dir=output_dir, dpi=dpi, formats=formats)
        plt.close(fig)
        logging.info(f"Saved: {figure_name}")


def plot_replay_position_by_swr(
    data_over_time: pd.DataFrame,
    output_dir: Path | None = None,
    figure_name: str | None = None,
    formats: list[str] | None = None,
    dpi: int = 300,
    ax: plt.Axes | None = None,
) -> None:
    """Plot replay position distribution by SWR overlap status.

    Parameters
    ----------
    data_over_time : pd.DataFrame
        Full dataset
    output_dir : Path, optional
        Output directory (standalone mode)
    figure_name : str, optional
        Base name for the figure (standalone mode)
    formats : list of str, optional
        Output formats
    dpi : int
        DPI for raster formats
    ax : matplotlib Axes, optional
        Axes to plot on. If None, creates standalone figure.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=PANEL_SINGLE)

    # Use nanmedian to handle potential NaN values
    unique_positions = data_over_time["max_linear_position"].unique()
    bins = int(np.nanmedian(unique_positions[~np.isnan(unique_positions)]) / 2)

    # SWR overlap
    swr_mask = (data_over_time["non_local_event"] > 0) & (
        data_over_time["SWR_event"] > 0
    )
    ax.hist(
        data_over_time.loc[swr_mask, "map_estimate"]
        / data_over_time.loc[swr_mask, "max_linear_position"],
        bins=bins,
        density=True,
        alpha=0.7,
        color=COLORS["SWR"],
        label="With SWR",
    )

    # No SWR overlap
    no_swr_mask = (data_over_time["non_local_event"] > 0) & (
        data_over_time["SWR_event"] == 0
    )
    ax.hist(
        data_over_time.loc[no_swr_mask, "map_estimate"]
        / data_over_time.loc[no_swr_mask, "max_linear_position"],
        bins=bins,
        density=True,
        alpha=0.7,
        color=COLORS["Non-Local"],
        label="Without SWR",
    )

    ax.set_xlabel("Normalized Replay Position")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 1)
    ax.legend()

    if standalone:
        sns.despine(ax=ax)
        plt.tight_layout()
        save_figure(figure_name, base_dir=output_dir, dpi=dpi, formats=formats)
        plt.close(fig)
        logging.info(f"Saved: {figure_name}")


def plot_swr_zscore_by_state(
    data_over_time: pd.DataFrame,
    output_dir: Path | None = None,
    figure_name: str | None = None,
    formats: list[str] | None = None,
    dpi: int = 300,
    ax: plt.Axes | None = None,
) -> None:
    """Plot SWR z-score distribution by local/non-local state.

    Parameters
    ----------
    data_over_time : pd.DataFrame
        Full dataset
    output_dir : Path, optional
        Output directory (standalone mode)
    figure_name : str, optional
        Base name for the figure (standalone mode)
    formats : list of str, optional
        Output formats
    dpi : int
        DPI for raster formats
    ax : matplotlib Axes, optional
        Axes to plot on. If None, creates standalone figure.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=PANEL_SINGLE)

    is_immobile = data_over_time.head_speed < 4.0

    # Local state
    local_mask = is_immobile & (data_over_time["Local_prob"] > 0.8)
    ax.hist(
        data_over_time.loc[local_mask, "SWR_zscore"],
        bins=np.arange(-5, 11),
        alpha=0.7,
        density=True,
        label="Local",
        color=COLORS["Local"],
    )

    # Non-local state
    nonlocal_mask = is_immobile & (data_over_time["Non-Local_prob"] > 0.8)
    ax.hist(
        data_over_time.loc[nonlocal_mask, "SWR_zscore"],
        bins=np.arange(-5, 11),
        alpha=0.7,
        density=True,
        label="Non-Local",
        color=COLORS["Non-Local"],
    )

    ax.set_xlabel("SWR Z-score")
    ax.set_ylabel("Density")
    ax.legend()

    if standalone:
        sns.despine(ax=ax)
        plt.tight_layout()
        save_figure(figure_name, base_dir=output_dir, dpi=dpi, formats=formats)
        plt.close(fig)
        logging.info(f"Saved: {figure_name}")


def plot_replay_speed_by_swr(
    data_over_time: pd.DataFrame,
    output_dir: Path | None = None,
    figure_name: str | None = None,
    formats: list[str] | None = None,
    dpi: int = 300,
    ax: plt.Axes | None = None,
) -> None:
    """Plot replay speed distribution by SWR overlap status.

    Parameters
    ----------
    data_over_time : pd.DataFrame
        Full dataset
    output_dir : Path, optional
        Output directory (standalone mode)
    figure_name : str, optional
        Base name for the figure (standalone mode)
    formats : list of str, optional
        Output formats
    dpi : int
        DPI for raster formats
    ax : matplotlib Axes, optional
        Axes to plot on. If None, creates standalone figure.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=PANEL_SINGLE)

    bins = np.arange(0, 3000, 100)

    # SWR overlap
    swr_mask = (data_over_time["non_local_event"] > 0) & (
        data_over_time["SWR_event"] > 0
    )
    ax.hist(
        data_over_time.loc[swr_mask, "map_speed"],
        bins=bins,
        density=True,
        alpha=0.7,
        color=COLORS["SWR"],
        label="With SWR",
    )

    # No SWR overlap
    no_swr_mask = (data_over_time["non_local_event"] > 0) & (
        data_over_time["SWR_event"] == 0
    )
    ax.hist(
        data_over_time.loc[no_swr_mask, "map_speed"],
        bins=bins,
        density=True,
        alpha=0.7,
        color=COLORS["Non-Local"],
        label="Without SWR",
    )

    ax.set_xlabel("Replay Speed (cm/s)")
    ax.set_ylabel("Density")
    ax.legend()

    if standalone:
        sns.despine(ax=ax)
        plt.tight_layout()
        save_figure(figure_name, base_dir=output_dir, dpi=dpi, formats=formats)
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
        default=Path("paper/figures/figure03"),
        help="Output directory for figures (default: paper/figures/figure03)",
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
        default=PROCESSED_DATA_DIR,
        help=f"Directory containing processed data (default: {PROCESSED_DATA_DIR})",
    )

    args = parser.parse_args(argv)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logging.info("Loading data...")
    data_over_time = load_data_over_time(args.data_dir)
    event_types = get_event_types(data_over_time)

    # Generate figures
    logging.info("Generating figures...")
    formats = [args.format] if args.format else None

    plot_event_duration_distribution(
        event_types,
        args.output_dir,
        "fig03_event_duration",
        formats,
        args.dpi,
    )

    plot_hpd_distribution(
        event_types,
        args.output_dir,
        "fig03_hpd_distribution",
        formats,
        args.dpi,
    )

    plot_distance_distribution(
        event_types,
        args.output_dir,
        "fig03_distance_from_animal",
        formats,
        args.dpi,
    )

    plot_mean_hpd_per_event(
        event_types,
        args.output_dir,
        "fig03_mean_hpd_per_event",
        formats,
        args.dpi,
    )

    plot_position_animal_vs_replay(
        event_types,
        args.output_dir,
        "fig03_position_animal_vs_replay",
        formats,
        args.dpi,
    )

    plot_triggered_averages(
        data_over_time,
        event_types,
        args.output_dir,
        "fig03_triggered_averages_start",
        time_col="start_time",
        formats=formats,
        dpi=args.dpi,
    )

    plot_triggered_averages(
        data_over_time,
        event_types,
        args.output_dir,
        "fig03_triggered_averages_end",
        time_col="end_time",
        formats=formats,
        dpi=args.dpi,
    )

    plot_replay_position_by_swr(
        data_over_time,
        args.output_dir,
        "fig03_replay_position_by_swr",
        formats,
        args.dpi,
    )

    plot_swr_zscore_by_state(
        data_over_time,
        args.output_dir,
        "fig03_swr_zscore_by_state",
        formats,
        args.dpi,
    )

    plot_replay_speed_by_swr(
        data_over_time,
        args.output_dir,
        "fig03_replay_speed_by_swr",
        formats,
        args.dpi,
    )

    logging.info("Done!")


if __name__ == "__main__":
    main()
