#!/usr/bin/env python
"""Generate Figure 2d: Event Timing Relationships.

This script analyzes temporal relationships between non-local events and
SWR/MUA events by computing time differences between event starts/ends.
Generates histograms with KDE smoothing showing:

1. Non-local event start relative to SWR start
2. Non-local event end relative to SWR end
3. Non-local event start relative to MUA start
4. Non-local event end relative to MUA end

Each plot includes:
- Histogram of time differences
- KDE (kernel density estimate) smoothing
- Mean and median markers
- Annotations for interpretation

Usage:
    python scripts/generate_figure02d_timing.py --output-dir paper/figures/figure02
    python scripts/generate_figure02d_timing.py --format png --dpi 300
"""

import argparse
import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import continuum_swr_replay.figure_utilities as figure_utilities
from continuum_swr_replay.figure_utilities import PANEL_SINGLE, save_figure
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


def compute_time_differences(
    data: pd.DataFrame,
    reference_event_col: str,
    target_event_col: str,
    mode: str = "start",
) -> pd.DataFrame:
    """Compute time differences between two event types.

    Parameters
    ----------
    data : pd.DataFrame
        Full dataset
    reference_event_col : str
        Reference event column (e.g., "SWR_event")
    target_event_col : str
        Target event column (e.g., "non_local_event")
    mode : str
        "start" or "end" - which timepoint to use

    Returns
    -------
    pd.DataFrame
        DataFrame with time differences (in seconds)
    """
    # Filter data
    ref_data = data[data[reference_event_col] > 0]
    target_data = data[data[target_event_col] > 0]

    # Group keys
    group_keys = ["animal", "day", "epoch", reference_event_col]

    # Identify reference events with target overlap
    ref_groups = ref_data.groupby(group_keys)
    overlap_mask = ref_groups[target_event_col].transform(lambda x: (x > 0).any())
    ref_data = ref_data[overlap_mask]

    # Get reference event times
    if mode == "start":
        ref_times = (
            ref_data.groupby(group_keys)
            .apply(lambda g: g.index[0])
            .rename(f"{reference_event_col}_time")
        )
    else:  # end
        ref_times = (
            ref_data.groupby(group_keys)
            .apply(lambda g: g.index[-1])
            .rename(f"{reference_event_col}_time")
        )

    # Get target events per reference event
    target_events = (
        ref_data[ref_data[target_event_col] > 0]
        .groupby(group_keys)[target_event_col]
        .unique()
        .explode()
        .rename(target_event_col)
    )

    # Get target event times
    if mode == "start":
        target_times = (
            target_data.groupby(["animal", "day", "epoch", target_event_col])
            .apply(lambda g: g.index[0])
            .rename(f"{target_event_col}_time")
        )
    else:  # end
        target_times = (
            target_data.groupby(["animal", "day", "epoch", target_event_col])
            .apply(lambda g: g.index[-1])
            .rename(f"{target_event_col}_time")
        )

    # Join everything
    result = target_events.reset_index().set_index(
        ["animal", "day", "epoch", target_event_col]
    )
    result = result.join(target_times)
    result = result.reset_index().set_index(
        ["animal", "day", "epoch", reference_event_col]
    )
    result = result.join(ref_times)
    result = result.reset_index()

    # Compute time difference
    result["time_diff"] = (
        result[f"{target_event_col}_time"] - result[f"{reference_event_col}_time"]
    )

    return result


def plot_timing_histogram(
    time_diffs_df: pd.DataFrame,
    event_type: str,
    mode: str,
    reference_label: str,
    target_label: str,
    output_dir: Path,
    figure_name: str,
    formats: list[str] | None = None,
    dpi: int = 300,
) -> None:
    """Plot timing relationship histogram with KDE.

    Parameters
    ----------
    time_diffs_df : pd.DataFrame
        DataFrame with time_diff column (in seconds)
    event_type : str
        Event type for coloring ("SWR" or "MUA")
    mode : str
        "start" or "end"
    reference_label : str
        Label for reference event (e.g., "SWR Start")
    target_label : str
        Label for target event (e.g., "Non-local Start")
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

    # Convert to milliseconds for plotting
    time_diff_ms = time_diffs_df["time_diff"] * 1000

    # Histogram
    ax.hist(
        time_diff_ms,
        bins=np.arange(-100, 100.1, 5).tolist(),
        color=COLORS.get(event_type, "gray"),
        alpha=0.7,
        edgecolor="none",
        density=True,
        histtype="stepfilled",
    )

    # Add KDE for smoother distribution
    sns.kdeplot(
        time_diff_ms,
        ax=ax,
        color="black",
        linewidth=2,
        label="KDE",
        clip=(-100, 100),
    )

    # Add mean and median lines
    mean_ms = time_diffs_df["time_diff"].mean() * 1000
    median_ms = time_diffs_df["time_diff"].median() * 1000
    ax.axvline(
        mean_ms, color="red", linestyle="--", lw=1.5, label=f"Mean: {mean_ms:.1f} ms"
    )
    ax.axvline(
        median_ms,
        color="blue",
        linestyle=":",
        lw=1.5,
        label=f"Median: {median_ms:.1f} ms",
    )
    ax.axvline(0, color="k", linestyle="--", lw=1, label=reference_label)

    # Style
    sns.despine(ax=ax, top=True, right=True, left=True, bottom=False)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)
    ax.set_axisbelow(True)

    # Labels
    time_label = mode.capitalize()
    ax.set_xlabel(
        f"Time Difference (ms)\n({target_label} {time_label} minus {reference_label} {time_label})"
    )
    ax.set_ylabel("Density")
    ax.set_title(
        f"Timing of {target_label} Events Relative to {reference_label} {time_label}",
        pad=10,
    )
    ax.set_xlim(-100, 100)
    ax.tick_params(axis="both", which="both", length=3, direction="out")

    # Annotate regions
    ax.text(
        40,
        ax.get_ylim()[1] * 0.9,
        f"{target_label} {mode}s later",
        ha="right",
    )
    ax.text(
        -40,
        ax.get_ylim()[1] * 0.9,
        f"{target_label} {mode}s earlier",
        ha="left",
    )

    # Legend
    ax.legend(frameon=False, loc="upper left")

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

    # Compute time differences for all four comparisons
    logging.info("Computing time differences...")

    # 1. Non-local start vs SWR start
    nl_swr_start = compute_time_differences(
        data_over_time,
        reference_event_col="SWR_event",
        target_event_col="non_local_event",
        mode="start",
    )

    # 2. Non-local end vs SWR end
    nl_swr_end = compute_time_differences(
        data_over_time,
        reference_event_col="SWR_event",
        target_event_col="non_local_event",
        mode="end",
    )

    # 3. Non-local start vs MUA start
    nl_mua_start = compute_time_differences(
        data_over_time,
        reference_event_col="MUA_event",
        target_event_col="non_local_event",
        mode="start",
    )

    # 4. Non-local end vs MUA end
    nl_mua_end = compute_time_differences(
        data_over_time,
        reference_event_col="MUA_event",
        target_event_col="non_local_event",
        mode="end",
    )

    # Generate figures
    logging.info("Generating timing relationship plots...")
    formats = [args.format] if args.format else None

    # 1. Non-local start vs SWR start
    plot_timing_histogram(
        nl_swr_start,
        event_type="SWR",
        mode="start",
        reference_label="SWR",
        target_label="Non-local",
        output_dir=args.output_dir,
        figure_name="fig02d_timing_nl_start_swr_start",
        formats=formats,
        dpi=args.dpi,
    )

    # 2. Non-local end vs SWR end
    plot_timing_histogram(
        nl_swr_end,
        event_type="SWR",
        mode="end",
        reference_label="SWR",
        target_label="Non-local",
        output_dir=args.output_dir,
        figure_name="fig02d_timing_nl_end_swr_end",
        formats=formats,
        dpi=args.dpi,
    )

    # 3. Non-local start vs MUA start
    plot_timing_histogram(
        nl_mua_start,
        event_type="MUA",
        mode="start",
        reference_label="MUA",
        target_label="Non-local",
        output_dir=args.output_dir,
        figure_name="fig02d_timing_nl_start_mua_start",
        formats=formats,
        dpi=args.dpi,
    )

    # 4. Non-local end vs MUA end
    plot_timing_histogram(
        nl_mua_end,
        event_type="MUA",
        mode="end",
        reference_label="MUA",
        target_label="Non-local",
        output_dir=args.output_dir,
        figure_name="fig02d_timing_nl_end_mua_end",
        formats=formats,
        dpi=args.dpi,
    )

    logging.info("Done!")


if __name__ == "__main__":
    main()
