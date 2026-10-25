#!/usr/bin/env python
"""Generate Figure 2a: Non-Local Probability Distributions.

This script creates probability distribution plots showing the bimodal nature of
non-local probabilities during immobile and running states. Generates:

1. Overall non-local probability histogram (immobile)
2. Per-animal non-local probability histograms (immobile)
3. Per-animal non-local probability histograms (running)
4. Local probability histogram (immobile)
5. No-spike probability histogram (immobile)

Usage:
    python scripts/generate_figure02a_distributions.py --output-dir paper/figures/figure02
    python scripts/generate_figure02a_distributions.py --format png  # PNG only
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
    DEFAULT_NON_LOCAL_THRESHOLD,
)
from continuum_swr_replay.figure_utilities import (
    PANEL_SINGLE,
    TWO_COLUMN,
    get_figure_size,
    save_figure,
)
from continuum_swr_replay.visualization.palettes import COLORS

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


def plot_overall_distribution(
    data: pd.DataFrame,
    is_immobile: pd.Series,
    output_dir: Path,
    figure_name: str,
    formats: list[str] | None = None,
    dpi: int = 300,
) -> None:
    """Plot overall non-local probability distribution during immobility.

    Parameters
    ----------
    data : pd.DataFrame
        Data with Non-Local_prob column
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
    fig, ax = plt.subplots(figsize=PANEL_SINGLE)

    # Plot histogram
    data.loc[is_immobile, "Non-Local_prob"].plot.hist(
        bins=50,
        color=COLORS["Non-Local"],
        alpha=0.7,
        density=True,
        edgecolor="none",
        histtype="stepfilled",
        range=(0, 1),
        ax=ax,
    )

    # Remove chartjunk: no top/right spines, minimal grid, no box
    sns.despine(ax=ax, left=True, top=True, right=True, bottom=False)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)
    ax.set_axisbelow(True)

    # Clean, informative labels
    ax.set_xlabel("Non-Local Probability")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Non-Local Probability (Immobile)", pad=10)

    # Remove legend, ticks on top/right, and reduce tick length
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    ax.tick_params(axis="both", which="both", length=3, direction="out")
    ax.set_xlim(0, 1)
    ax.axvline(
        DEFAULT_NON_LOCAL_THRESHOLD,
        color="black",
        linewidth=0.5,
        linestyle="--",
        label=f"{DEFAULT_NON_LOCAL_THRESHOLD} Threshold",
    )
    plt.tight_layout()

    save_figure(figure_name, base_dir=output_dir, dpi=dpi, formats=formats)
    plt.close(fig)
    logging.info(f"Saved: {figure_name}")


def plot_per_animal_distributions(
    data: pd.DataFrame,
    is_immobile: pd.Series,
    output_dir: Path,
    figure_name: str,
    formats: list[str] | None = None,
    dpi: int = 300,
) -> None:
    """Plot per-animal non-local probability distributions during immobility.

    Parameters
    ----------
    data : pd.DataFrame
        Data with animal and Non-Local_prob columns
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
    # Get unique animals
    animals = data["animal"].unique()

    # Set up the grid for small multiples (5 animals = full page width)
    n_animals = len(animals)
    fig, axes = plt.subplots(
        1, n_animals, figsize=get_figure_size(1, n_animals, base_width=TWO_COLUMN), sharey=True
    )

    if n_animals == 1:
        axes = [axes]

    for ax, animal in zip(axes, animals, strict=True):
        group = data.loc[(data["animal"] == animal) & is_immobile]
        ax.hist(
            group["Non-Local_prob"],
            bins=10,
            alpha=0.7,
            color=COLORS.get("Non-Local", "#1b9e77"),
            density=True,
            edgecolor="black",
            linewidth=0.7,
            histtype="stepfilled",
        )
        sns.despine(ax=ax)
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        ax.set_xlabel("Non-Local Probability")
        ax.set_title(f"Animal: {animal}")

    axes[0].set_ylabel("Density")
    fig.suptitle("Distribution of Non-Local Probability (Immobile Periods)")
    plt.tight_layout(rect=(0, 0, 1, 0.96))

    save_figure(figure_name, base_dir=output_dir, dpi=dpi, formats=formats)
    plt.close(fig)
    logging.info(f"Saved: {figure_name}")


def plot_per_animal_running_distributions(
    data: pd.DataFrame,
    is_immobile: pd.Series,
    output_dir: Path,
    figure_name: str,
    formats: list[str] | None = None,
    dpi: int = 300,
) -> None:
    """Plot per-animal non-local probability distributions during running.

    Parameters
    ----------
    data : pd.DataFrame
        Data with animal and Non-Local_prob columns
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
    # Get unique animals
    animals = data["animal"].unique()

    # Set up the grid for small multiples (5 animals = full page width)
    n_animals = len(animals)
    fig, axes = plt.subplots(
        1, n_animals, figsize=get_figure_size(1, n_animals, base_width=TWO_COLUMN), sharey=True
    )

    if n_animals == 1:
        axes = [axes]

    for ax, animal in zip(axes, animals, strict=True):
        group = data.loc[(data["animal"] == animal) & (~is_immobile)]
        ax.hist(
            group["Non-Local_prob"],
            bins=10,
            alpha=0.7,
            color=COLORS.get("Non-Local", "#1b9e77"),
            density=True,
            edgecolor="black",
            linewidth=0.7,
            histtype="stepfilled",
        )
        sns.despine(ax=ax)
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        ax.set_xlabel("Non-Local Probability")
        ax.set_title(f"Animal: {animal}")

    axes[0].set_ylabel("Density")
    fig.suptitle("Distribution of Non-Local Probability (Run Periods)")
    plt.tight_layout(rect=(0, 0, 1, 0.96))

    save_figure(figure_name, base_dir=output_dir, dpi=dpi, formats=formats)
    plt.close(fig)
    logging.info(f"Saved: {figure_name}")


def plot_local_distribution(
    data: pd.DataFrame,
    is_immobile: pd.Series,
    output_dir: Path,
    figure_name: str,
    formats: list[str] | None = None,
    dpi: int = 300,
) -> None:
    """Plot local probability distribution during immobility.

    Parameters
    ----------
    data : pd.DataFrame
        Data with Local_prob column
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
    fig, ax = plt.subplots(figsize=PANEL_SINGLE)

    # Plot histogram
    data.loc[is_immobile, "Local_prob"].plot.hist(
        bins=50,
        color=COLORS["Local"],
        alpha=0.7,
        density=True,
        edgecolor="none",
        histtype="stepfilled",
        range=(0, 1),
        ax=ax,
    )

    # Remove chartjunk: no top/right spines, minimal grid, no box
    sns.despine(ax=ax, left=True, top=True, right=True, bottom=False)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)
    ax.set_axisbelow(True)

    # Clean, informative labels
    ax.set_xlabel("Local Probability")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Local Probability (Immobile)", pad=10)

    # Remove legend, ticks on top/right, and reduce tick length
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    ax.tick_params(axis="both", which="both", length=3, direction="out")
    ax.set_xlim(0, 1)
    ax.axvline(
        DEFAULT_NON_LOCAL_THRESHOLD,
        color="black",
        linewidth=0.5,
        linestyle="--",
        label=f"{DEFAULT_NON_LOCAL_THRESHOLD} Threshold",
    )
    plt.tight_layout()

    save_figure(figure_name, base_dir=output_dir, dpi=dpi, formats=formats)
    plt.close(fig)
    logging.info(f"Saved: {figure_name}")


def plot_nospike_distribution(
    data: pd.DataFrame,
    is_immobile: pd.Series,
    output_dir: Path,
    figure_name: str,
    formats: list[str] | None = None,
    dpi: int = 300,
) -> None:
    """Plot no-spike probability distribution during immobility.

    Parameters
    ----------
    data : pd.DataFrame
        Data with No-Spike_prob column
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
    fig, ax = plt.subplots(figsize=PANEL_SINGLE)

    # Plot histogram
    data.loc[is_immobile, "No-Spike_prob"].plot.hist(
        bins=50,
        color=COLORS["No-Spike"],
        alpha=0.7,
        density=True,
        edgecolor="none",
        histtype="stepfilled",
        range=(0, 1),
        ax=ax,
    )

    # Remove chartjunk: no top/right spines, minimal grid, no box
    sns.despine(ax=ax, left=True, top=True, right=True, bottom=False)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)
    ax.set_axisbelow(True)

    # Clean, informative labels
    ax.set_xlabel("No-Spike Probability")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of No-Spike Probability (Immobile)", pad=10)

    # Remove legend, ticks on top/right, and reduce tick length
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    ax.tick_params(axis="both", which="both", length=3, direction="out")
    ax.set_xlim(0, 1)
    ax.axvline(
        DEFAULT_NON_LOCAL_THRESHOLD,
        color="black",
        linewidth=0.5,
        linestyle="--",
        label=f"{DEFAULT_NON_LOCAL_THRESHOLD} Threshold",
    )
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

    # Define immobility mask using library constant
    is_immobile = data_over_time.head_speed < DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC

    # Generate figures
    logging.info("Generating figures...")
    formats = [args.format] if args.format else None

    plot_overall_distribution(
        data_over_time,
        is_immobile,
        args.output_dir,
        "fig02a_nonlocal_prob_overall",
        formats,
        args.dpi,
    )

    plot_per_animal_distributions(
        data_over_time,
        is_immobile,
        args.output_dir,
        "fig02a_nonlocal_prob_per_animal_immobile",
        formats,
        args.dpi,
    )

    plot_per_animal_running_distributions(
        data_over_time,
        is_immobile,
        args.output_dir,
        "fig02a_nonlocal_prob_per_animal_running",
        formats,
        args.dpi,
    )

    plot_local_distribution(
        data_over_time,
        is_immobile,
        args.output_dir,
        "fig02a_local_prob",
        formats,
        args.dpi,
    )

    plot_nospike_distribution(
        data_over_time,
        is_immobile,
        args.output_dir,
        "fig02a_nospike_prob",
        formats,
        args.dpi,
    )

    logging.info("Done!")


if __name__ == "__main__":
    main()
