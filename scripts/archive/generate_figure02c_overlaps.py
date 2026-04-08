#!/usr/bin/env python
"""Generate Figure 2c: Event Overlaps and Durations.

This script analyzes overlaps between different event types (non-local, SWR, MUA)
and generates duration distribution comparisons. Creates:

1. Event duration histograms (separate panels for each event type)
2. Overlaid duration histograms (all event types on one plot)
3. Overlap percentage calculations
4. Bar charts showing overlap percentages
5. Venn diagrams (timepoint percentages and event counts)
6. Non-local probability distributions during SWR/MUA events

Usage:
    python scripts/generate_figure02c_overlaps.py --output-dir paper/figures/figure02
    python scripts/generate_figure02c_overlaps.py --format png --dpi 300
"""

import argparse
import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

import continuum_swr_replay.figure_utilities as figure_utilities
from continuum_swr_replay.analysis.constants import (
    BINS_DURATION_MS,
    BINS_PROBABILITY,
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    DEFAULT_SAMPLING_FREQUENCY_HZ,
)
from continuum_swr_replay.analysis.hippocampus import (
    get_event_durations_ms,
    get_event_start_end_times,
    get_immobile_events,
)
from continuum_swr_replay.figure_utilities import (
    FIGURE_THREE_PANEL_HORIZONTAL,
    FIGURE_TWO_PANEL_HORIZONTAL,
    ONE_COLUMN,
    PANEL_SINGLE,
    PANEL_SINGLE_SQUARE,
    TWO_COLUMN,
    get_figure_size,
    save_figure,
)
from continuum_swr_replay.parameters import COLORS

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


def plot_duration_histograms_separate(
    non_local_dur: pd.Series,
    swr_dur: pd.Series,
    mua_dur: pd.Series,
    output_dir: Path,
    figure_name: str,
    formats: list[str] | None = None,
    dpi: int = 300,
) -> None:
    """Plot duration histograms in separate panels.

    Parameters
    ----------
    non_local_dur : pd.Series
        Non-local event durations (ms)
    swr_dur : pd.Series
        SWR event durations (ms)
    mua_dur : pd.Series
        MUA event durations (ms)
    output_dir : Path
        Output directory
    figure_name : str
        Base name for the figure (without extension)
    formats : list of str, optional
        Output formats (default: ["pdf", "png"])
    dpi : int
        DPI for raster formats
    """
    fig, axes = plt.subplots(
        3, 1, figsize=get_figure_size(3, 1), sharex=True, sharey=True, constrained_layout=True
    )

    # Non-local
    axes[0].hist(
        non_local_dur,
        bins=BINS_DURATION_MS.tolist(),
        color=COLORS["Non-Local"],
        alpha=0.7,
        edgecolor="none",
        density=True,
    )
    axes[0].set_title("Non-local (Immobile)", pad=10)

    # SWR
    axes[1].hist(
        swr_dur,
        bins=BINS_DURATION_MS.tolist(),
        color=COLORS["SWR"],
        alpha=0.7,
        edgecolor="none",
        density=True,
    )
    axes[1].set_title("SWR", pad=10)

    # MUA
    axes[2].hist(
        mua_dur,
        bins=BINS_DURATION_MS.tolist(),
        color=COLORS["MUA"],
        alpha=0.7,
        edgecolor="none",
        density=True,
    )
    axes[2].set_xlabel("Event Duration (ms)")
    axes[2].set_title("MUA", pad=10)

    for ax in axes:
        sns.despine(ax=ax, top=True, right=True, left=True, bottom=False)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)
        ax.set_axisbelow(True)
        ax.set_ylabel("Density")
        ax.tick_params(axis="both", which="both", length=3, direction="out")
        ax.xaxis.set_major_locator(mticker.MultipleLocator(50))
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(10))
        ax.set_xlim(0, 250)

    save_figure(figure_name, base_dir=output_dir, dpi=dpi, formats=formats)
    plt.close(fig)
    logging.info(f"Saved: {figure_name}")


def plot_duration_histograms_overlaid(
    non_local_dur: pd.Series,
    swr_dur: pd.Series,
    mua_dur: pd.Series,
    output_dir: Path,
    figure_name: str,
    formats: list[str] | None = None,
    dpi: int = 300,
) -> None:
    """Plot all duration histograms overlaid on one axis.

    Parameters
    ----------
    non_local_dur : pd.Series
        Non-local event durations (ms)
    swr_dur : pd.Series
        SWR event durations (ms)
    mua_dur : pd.Series
        MUA event durations (ms)
    output_dir : Path
        Output directory
    figure_name : str
        Base name for the figure (without extension)
    formats : list of str, optional
        Output formats (default: ["pdf", "png"])
    dpi : int
        DPI for raster formats
    """
    fig, ax = plt.subplots(figsize=PANEL_SINGLE)

    # Plot all three histograms
    ax.hist(
        non_local_dur,
        bins=BINS_DURATION_MS.tolist(),
        color=COLORS["Non-Local"],
        alpha=0.6,
        edgecolor="none",
        density=True,
        label="Non-local (Immobile)",
    )
    ax.hist(
        swr_dur,
        bins=BINS_DURATION_MS.tolist(),
        color=COLORS["SWR"],
        alpha=0.6,
        edgecolor="none",
        density=True,
        label="SWR",
    )
    ax.hist(
        mua_dur,
        bins=BINS_DURATION_MS.tolist(),
        color=COLORS["MUA"],
        alpha=0.6,
        edgecolor="none",
        density=True,
        label="MUA",
    )

    # Clean style
    sns.despine(ax=ax, top=True, right=True, left=True, bottom=False)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)
    ax.set_axisbelow(True)

    ax.set_xlabel("Event Duration (ms)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Event Durations", pad=10)
    ax.legend(frameon=False)
    ax.tick_params(axis="both", which="both", length=3, direction="out")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(50))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(10))

    save_figure(figure_name, base_dir=output_dir, dpi=dpi, formats=formats)
    plt.close(fig)
    logging.info(f"Saved: {figure_name}")


def calculate_overlap_percentages(
    data: pd.DataFrame, immobile_non_local: pd.DataFrame
) -> dict[str, float]:
    """Calculate overlap percentages between event types.

    Parameters
    ----------
    data : pd.DataFrame
        Full dataset
    immobile_non_local : pd.DataFrame
        Immobile non-local events

    Returns
    -------
    dict
        Overlap percentages
    """
    # Non-local events that overlap with SWR
    non_local_swr_overlap = (
        immobile_non_local.groupby(["animal", "day", "epoch", "non_local_event"])
        .apply(lambda g: (g["SWR_event"] > 0).any())
        .mean()
        * 100.0
    )

    # Non-local events that overlap with MUA
    non_local_mua_overlap = (
        immobile_non_local.groupby(["animal", "day", "epoch", "non_local_event"])
        .apply(lambda g: (g["MUA_event"] > 0).any())
        .mean()
        * 100.0
    )

    # SWR events that overlap with non-local
    swr_non_local_overlap_pct = (
        data.loc[data["SWR_event"] > 0]
        .groupby(["animal", "day", "epoch", "SWR_event"])
        .apply(lambda g: (g["non_local_event"] > 0).any())
        .mean()
        * 100.0
    )

    # MUA events that overlap with non-local
    mua_non_local_overlap_pct = (
        data.loc[data["MUA_event"] > 0]
        .groupby(["animal", "day", "epoch", "MUA_event"])
        .apply(lambda g: (g["non_local_event"] > 0).any())
        .mean()
        * 100.0
    )

    return {
        "non_local_swr": non_local_swr_overlap,
        "non_local_mua": non_local_mua_overlap,
        "swr_non_local": swr_non_local_overlap_pct,
        "mua_non_local": mua_non_local_overlap_pct,
    }


def plot_overlap_bar_chart(
    overlaps: dict[str, float],
    output_dir: Path,
    figure_name: str,
    formats: list[str] | None = None,
    dpi: int = 300,
) -> None:
    """Plot bar chart of overlap percentages.

    Parameters
    ----------
    overlaps : dict
        Overlap percentages
    output_dir : Path
        Output directory
    figure_name : str
        Base name for the figure (without extension)
    formats : list of str, optional
        Output formats (default: ["pdf", "png"])
    dpi : int
        DPI for raster formats
    """
    fig, ax = plt.subplots(figsize=PANEL_SINGLE)

    labels = [
        "Non-local\nwith SWR",
        "Non-local\nwith MUA",
        "SWR\nwith\nNon-local",
        "MUA\nwith\nNon-local",
    ]
    values = [
        overlaps["non_local_swr"],
        overlaps["non_local_mua"],
        overlaps["swr_non_local"],
        overlaps["mua_non_local"],
    ]
    colors_list = [
        COLORS["SWR"],
        COLORS["MUA"],
        COLORS["Non-Local"],
        COLORS["Non-Local"],
    ]

    bars = ax.bar(labels, values, color=colors_list, alpha=0.7, edgecolor="none")

    # Add value labels above bars
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Minimalist style
    sns.despine(ax=ax, top=True, right=True, left=True, bottom=False)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)
    ax.set_axisbelow(True)

    ax.set_ylabel("Overlap (%)")
    ax.set_ylim(0, 100)
    ax.set_xticklabels(labels)
    ax.set_title("Event Overlap: Model vs. SWR/MUA", pad=12)

    save_figure(figure_name, base_dir=output_dir, dpi=dpi, formats=formats)
    plt.close(fig)
    logging.info(f"Saved: {figure_name}")


def plot_venn_diagram_timepoints(
    data: pd.DataFrame,
    is_immobile: pd.Series,
    output_dir: Path,
    figure_name: str,
    formats: list[str] | None = None,
    dpi: int = 300,
) -> None:
    """Plot Venn diagram showing timepoint percentages.

    Parameters
    ----------
    data : pd.DataFrame
        Full dataset
    is_immobile : pd.Series
        Boolean mask for immobile timepoints
    output_dir : Path
        Output directory
    figure_name : str
        Base name for the figure (without extension)
    formats : list of str, optional
        Output formats (default: ["pdf", "png"])
    dpi : int
        DPI for raster formats
    """
    df = data.loc[is_immobile]

    # Boolean masks
    swr = df["SWR_event"] > 0
    mua = df["MUA_event"] > 0
    non_local = df["non_local_event"] > 0

    # Venn regions (as timepoint counts)
    only_swr = swr & ~mua & ~non_local
    only_mua = mua & ~swr & ~non_local
    only_non_local = non_local & ~swr & ~mua

    swr_and_mua = swr & mua & ~non_local
    swr_and_non_local = swr & non_local & ~mua
    mua_and_non_local = mua & non_local & ~swr

    all_three = swr & mua & non_local

    total_immobile_time = len(df)

    # Count timepoints in each category
    counts = {
        "SWR only": 100 * only_swr.sum() / total_immobile_time,
        "MUA only": 100 * only_mua.sum() / total_immobile_time,
        "Non-local only": 100 * only_non_local.sum() / total_immobile_time,
        "SWR ∩ MUA": 100 * swr_and_mua.sum() / total_immobile_time,
        "SWR ∩ Non-local": 100 * swr_and_non_local.sum() / total_immobile_time,
        "MUA ∩ Non-local": 100 * mua_and_non_local.sum() / total_immobile_time,
        "All three": 100 * all_three.sum() / total_immobile_time,
    }

    # Plot
    fig, ax = plt.subplots(figsize=PANEL_SINGLE_SQUARE)

    # Draw the three circles
    circle_swr = plt.Circle((0.4, 0.5), 0.25, color=COLORS["SWR"], alpha=0.3)
    circle_mua = plt.Circle((0.6, 0.5), 0.25, color=COLORS["MUA"], alpha=0.3)
    circle_nonlocal = plt.Circle((0.5, 0.7), 0.25, color=COLORS["Non-Local"], alpha=0.3)

    ax.add_patch(circle_swr)
    ax.add_patch(circle_mua)
    ax.add_patch(circle_nonlocal)

    # Label the regions
    ax.text(0.27, 0.5, f"{counts['SWR only']:.1f}%", ha="center")
    ax.text(0.73, 0.5, f"{counts['MUA only']:.1f}%", ha="center")
    ax.text(0.5, 0.8, f"{counts['Non-local only']:.1f}%", ha="center")

    ax.text(0.5, 0.55, f"{counts['All three']:.1f}%", ha="center")

    ax.text(0.5, 0.4, f"{counts['SWR ∩ MUA']:.1f}%", ha="center")
    ax.text(0.35, 0.65, f"{counts['SWR ∩ Non-local']:.1f}%", ha="center")
    ax.text(0.65, 0.65, f"{counts['MUA ∩ Non-local']:.1f}%", ha="center")

    # Circle labels
    ax.text(0.25, 0.35, "SWR", color=COLORS["SWR"], fontweight="bold")
    ax.text(0.70, 0.35, "MUA", color=COLORS["MUA"], fontweight="bold")
    ax.text(
        0.5,
        0.90,
        "Non-local",
        color=COLORS["Non-Local"],
        fontweight="bold",
        ha="center",
    )

    # Final plot setup
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Percent of Immobile Time")

    save_figure(figure_name, base_dir=output_dir, dpi=dpi, formats=formats)
    plt.close(fig)
    logging.info(f"Saved: {figure_name}")


def plot_venn_diagram_events(
    data: pd.DataFrame,
    is_immobile: pd.Series,
    output_dir: Path,
    figure_name: str,
    formats: list[str] | None = None,
    dpi: int = 300,
) -> None:
    """Plot Venn diagram showing event counts.

    Parameters
    ----------
    data : pd.DataFrame
        Full dataset
    is_immobile : pd.Series
        Boolean mask for immobile timepoints
    output_dir : Path
        Output directory
    figure_name : str
        Base name for the figure (without extension)
    formats : list of str, optional
        Output formats (default: ["pdf", "png"])
    dpi : int
        DPI for raster formats
    """
    df = data.loc[is_immobile]

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

    def count_events(mask: pd.Series, label_col: str) -> int:
        """Count unique events in masked data."""
        return len(
            df.loc[mask & (df[label_col] > 0)].groupby(
                ["animal", "day", "epoch", label_col]
            )
        )

    # Count events in each category
    event_counts = {
        "SWR only": count_events(only_swr, "SWR_event"),
        "MUA only": count_events(only_mua, "MUA_event"),
        "Non-local only": count_events(only_non_local, "non_local_event"),
        "SWR ∩ MUA": count_events(swr_and_mua, "SWR_event"),
        "SWR ∩ Non-local": count_events(swr_and_non_local, "SWR_event"),
        "MUA ∩ Non-local": count_events(mua_and_non_local, "MUA_event"),
        "All three": count_events(all_three, "SWR_event"),
    }

    # Plot
    fig, ax = plt.subplots(figsize=PANEL_SINGLE_SQUARE)

    # Draw the three circles
    circle_swr = plt.Circle((0.4, 0.5), 0.25, color=COLORS["SWR"], alpha=0.3)
    circle_mua = plt.Circle((0.6, 0.5), 0.25, color=COLORS["MUA"], alpha=0.3)
    circle_nonlocal = plt.Circle((0.5, 0.7), 0.25, color=COLORS["Non-Local"], alpha=0.3)

    ax.add_patch(circle_swr)
    ax.add_patch(circle_mua)
    ax.add_patch(circle_nonlocal)

    # Label the regions
    ax.text(0.27, 0.5, f"{event_counts['SWR only']}", ha="center")
    ax.text(0.73, 0.5, f"{event_counts['MUA only']}", ha="center")
    ax.text(0.5, 0.8, f"{event_counts['Non-local only']}", ha="center")

    ax.text(0.5, 0.55, f"{event_counts['All three']}", ha="center")

    ax.text(0.5, 0.4, f"{event_counts['SWR ∩ MUA']}", ha="center")
    ax.text(0.35, 0.65, f"{event_counts['SWR ∩ Non-local']}", ha="center")
    ax.text(0.65, 0.65, f"{event_counts['MUA ∩ Non-local']}", ha="center")

    # Circle labels
    ax.text(0.25, 0.35, "SWR", color=COLORS["SWR"], fontweight="bold")
    ax.text(0.70, 0.35, "MUA", color=COLORS["MUA"], fontweight="bold")
    ax.text(
        0.5,
        0.90,
        "Non-local",
        color=COLORS["Non-Local"],
        fontweight="bold",
        ha="center",
    )

    # Final plot setup
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Number of Events")

    save_figure(figure_name, base_dir=output_dir, dpi=dpi, formats=formats)
    plt.close(fig)
    logging.info(f"Saved: {figure_name}")


def plot_nonlocal_prob_during_events(
    data: pd.DataFrame,
    output_dir: Path,
    figure_name: str,
    formats: list[str] | None = None,
    dpi: int = 300,
) -> None:
    """Plot non-local probability during SWR and MUA events.

    Parameters
    ----------
    data : pd.DataFrame
        Full dataset
    output_dir : Path
        Output directory
    figure_name : str
        Base name for the figure (without extension)
    formats : list of str, optional
        Output formats (default: ["pdf", "png"])
    dpi : int
        DPI for raster formats
    """
    fig, axes = plt.subplots(
        1, 2, figsize=FIGURE_TWO_PANEL_HORIZONTAL, sharey=True, sharex=True, constrained_layout=True
    )

    data[data["SWR_event"] > 0]["Non-Local_prob"].hist(
        bins=BINS_PROBABILITY,
        color=COLORS["Non-Local"],
        alpha=0.7,
        edgecolor="none",
        density=True,
        label="SWR Events",
        ax=axes[0],
    )

    axes[0].set_title("SWR Event\nNon-Local Probability")
    axes[0].set_xlabel("Non-Local Probability")
    axes[0].set_ylabel("Density")

    data[data["MUA_event"] > 0]["Non-Local_prob"].hist(
        bins=BINS_PROBABILITY,
        color=COLORS["Non-Local"],
        alpha=0.7,
        edgecolor="none",
        density=True,
        label="MUA Events",
        ax=axes[1],
    )

    axes[1].set_title("MUA Event\nNon-Local Probability")
    axes[1].set_xlabel("Non-Local Probability")

    for ax in axes:
        sns.despine(ax=ax, top=True, right=True, left=True, bottom=False)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", which="both", length=3, direction="out")

    save_figure(figure_name, base_dir=output_dir, dpi=dpi, formats=formats)
    plt.close(fig)
    logging.info(f"Saved: {figure_name}")


def main() -> None:
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

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logging.info("Loading data...")
    data_over_time = load_data(args.data_dir)

    # Define immobility mask
    is_immobile = data_over_time.head_speed < DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC

    # Get immobile non-local events
    logging.info("Filtering immobile non-local events...")
    immobile_non_local = get_immobile_events(
        data_over_time,
        event_col="non_local_event",
        speed_col="head_speed",
        speed_threshold=DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
        method="endpoints",
    )

    # Calculate event durations using library function
    logging.info("Calculating event durations...")

    # Non-local (immobile) - use get_event_durations_ms
    non_local_times = get_event_start_end_times(
        immobile_non_local, event_col="non_local_event"
    )
    non_local_event_durations = get_event_durations_ms(non_local_times)

    # SWR - manual calculation (all events, not just immobile)
    SWR_event_durations = (
        data_over_time.groupby(["animal", "day", "epoch", "SWR_event"]).size()
        / DEFAULT_SAMPLING_FREQUENCY_HZ
    ) * 1000.0

    # MUA - manual calculation (all events, not just immobile)
    MUA_event_durations = (
        data_over_time.groupby(["animal", "day", "epoch", "MUA_event"]).size()
        / DEFAULT_SAMPLING_FREQUENCY_HZ
    ) * 1000.0

    # Remove events with ID 0 (non-events)
    SWR_event_durations = SWR_event_durations[
        SWR_event_durations.index.get_level_values("SWR_event") > 0
    ]
    MUA_event_durations = MUA_event_durations[
        MUA_event_durations.index.get_level_values("MUA_event") > 0
    ]

    # Calculate overlap percentages
    logging.info("Calculating overlap percentages...")
    overlaps = calculate_overlap_percentages(data_over_time, immobile_non_local)

    # Generate figures
    logging.info("Generating overlap and duration plots...")
    formats = [args.format] if args.format else None

    # 1. Duration histograms (separate panels)
    plot_duration_histograms_separate(
        non_local_event_durations,
        SWR_event_durations,
        MUA_event_durations,
        args.output_dir,
        "fig02c_durations_separate",
        formats,
        args.dpi,
    )

    # 2. Duration histograms (overlaid)
    plot_duration_histograms_overlaid(
        non_local_event_durations,
        SWR_event_durations,
        MUA_event_durations,
        args.output_dir,
        "fig02c_durations_overlaid",
        formats,
        args.dpi,
    )

    # 3. Overlap bar chart
    plot_overlap_bar_chart(
        overlaps,
        args.output_dir,
        "fig02c_overlap_bar",
        formats,
        args.dpi,
    )

    # 4. Venn diagram (timepoint percentages)
    plot_venn_diagram_timepoints(
        data_over_time,
        is_immobile,
        args.output_dir,
        "fig02c_venn_timepoints",
        formats,
        args.dpi,
    )

    # 5. Venn diagram (event counts)
    plot_venn_diagram_events(
        data_over_time,
        is_immobile,
        args.output_dir,
        "fig02c_venn_events",
        formats,
        args.dpi,
    )

    # 6. Non-local probability during SWR/MUA
    plot_nonlocal_prob_during_events(
        data_over_time,
        args.output_dir,
        "fig02c_nonlocal_prob_during_events",
        formats,
        args.dpi,
    )

    logging.info("Done!")


if __name__ == "__main__":
    main()
