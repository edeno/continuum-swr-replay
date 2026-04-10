"""Reusable panel-level plotting functions for composite figures.

These functions plot individual figure panels and can accept a matplotlib Axes
object for embedding in composite layouts. When called without an ``ax``
argument, they create a standalone figure.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import sem

from continuum_swr_replay.analysis.constants import (
    BINS_DURATION_MS,
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
)
from continuum_swr_replay.analysis.hippocampus import (
    compute_calibration_curve,
    compute_local_decoding_error,
    compute_state_fraction_during_movement,
    get_event_durations_ms,
    get_event_start_end_times,
    get_signal_windows,
)
from continuum_swr_replay.figure_utilities import (
    PANEL_SINGLE,
    get_figure_size,
    save_figure,
)
from continuum_swr_replay.parameters import (
    COLORS,
    SAMPLING_FREQUENCY,
)


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
        Dictionary of event type DataFrames.
    output_dir, figure_name, formats, dpi :
        Standalone-mode save parameters (ignored when ``ax`` is provided).
    ax : matplotlib Axes, optional
        Axes to plot on. If None, creates standalone figure.

    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=PANEL_SINGLE)

    for event_type in ("non_local_immobile", "non_local_immobile_no_overlap_swr"):
        event_times = get_event_start_end_times(
            event_types[event_type], event_col="non_local_event"
        )
        duration = get_event_durations_ms(event_times)

        color = (
            COLORS["Non-Local"]
            if event_type == "non_local_immobile_no_overlap_swr"
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
        Full dataset.
    event_types : dict
        Dictionary of event type DataFrames.
    output_dir, figure_name, formats, dpi :
        Standalone-mode save parameters (ignored when ``axes`` is provided).
    time_col : str
        Column to use for alignment ('start_time' or 'end_time').
    axes : list of matplotlib Axes, optional
        List of 4 axes for the 4 signals. If None, creates standalone figure.

    """
    signal_types = [
        "Non-Local_prob",
        "theta_power_zscore",
        "MUA_zscore",
        "SWR_zscore",
    ]

    non_local_start_end_time = get_event_start_end_times(
        event_types["non_local_immobile_no_overlap_swr"], event_col="non_local_event"
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

        no_overlap_data = signal_windows.sel(signal=signal_type)
        overlap_data = signal_windows_overlap.sel(signal=signal_type)
        no_overlap_sem = sem(no_overlap_data.values, axis=0, nan_policy="omit")
        overlap_sem = sem(overlap_data.values, axis=0, nan_policy="omit")

        time = no_overlap_mean["time"].values

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
            time,
            no_overlap_mean.values,
            label="No SWR Overlap",
            color=COLORS["Non-Local"],
        )
        ax.plot(
            time,
            overlap_mean.values,
            label="SWR Overlap",
            color=COLORS["SWR"],
        )

        if signal_type.endswith("_zscore"):
            name = signal_type.replace("_zscore", "").replace("_", " ").title()
            if name == "Theta Power":
                name = "Theta"
            ax.set_ylabel(f"{name}\nZ-score")
            ax.axhline(
                2.0, color="lightgrey", linestyle="--", alpha=0.5, linewidth=0.75
            )
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
        Aggregated data with ``SWR_zscore``, ``Local_prob``,
        ``Non-Local_prob``, and ``head_speed`` columns.
    output_dir, figure_name, formats, dpi :
        Standalone-mode save parameters (ignored when ``ax`` is provided).
    ax : matplotlib Axes, optional
        Axes to plot on. If None, creates standalone figure.

    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=PANEL_SINGLE)

    is_immobile = data_over_time.head_speed < DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC

    local_mask = is_immobile & (data_over_time["Local_prob"] > 0.8)
    ax.hist(
        data_over_time.loc[local_mask, "SWR_zscore"],
        bins=np.arange(-5, 11),
        alpha=0.7,
        density=True,
        label="Local",
        color=COLORS["Local"],
    )

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
        Aggregated data with ``map_speed``, ``non_local_event``,
        and ``SWR_event`` columns.
    output_dir, figure_name, formats, dpi :
        Standalone-mode save parameters (ignored when ``ax`` is provided).
    ax : matplotlib Axes, optional
        Axes to plot on. If None, creates standalone figure.

    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=PANEL_SINGLE)

    bins = np.arange(0, 3000, 100)

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
        Aggregated data with ``map_estimate``, ``max_linear_position``,
        ``non_local_event``, and ``SWR_event`` columns.
    output_dir, figure_name, formats, dpi :
        Standalone-mode save parameters (ignored when ``ax`` is provided).
    ax : matplotlib Axes, optional
        Axes to plot on. If None, creates standalone figure.

    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=PANEL_SINGLE)

    unique_positions = data_over_time["max_linear_position"].unique()
    bins = int(np.nanmedian(unique_positions[~np.isnan(unique_positions)]) / 2)

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


def plot_calibration_curve(
    data_over_time: pd.DataFrame,
    ax: plt.Axes | None = None,
) -> None:
    """Plot decoding error vs. local probability (calibration curve).

    Shows that higher decoder confidence corresponds to lower decoding error,
    pooled across all sessions with per-session variability.

    Parameters
    ----------
    data_over_time : pd.DataFrame
        Aggregated ``data_over_time`` DataFrame.
    ax : matplotlib Axes, optional
        Axes to plot on. If None, creates standalone figure.

    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=PANEL_SINGLE)

    calib = compute_calibration_curve(data_over_time)

    # Aggregate across sessions: median and IQR per probability bin
    grouped = calib.groupby("prob_bin_center")["median_error_cm"]
    agg_median = grouped.median()
    agg_q25 = grouped.quantile(0.25)
    agg_q75 = grouped.quantile(0.75)

    ax.fill_between(
        agg_median.index,
        agg_q25.values,
        agg_q75.values,
        color=COLORS["Local"],
        alpha=0.3,
    )
    ax.plot(
        agg_median.index,
        agg_median.values,
        color=COLORS["Local"],
        linewidth=1.5,
        marker="o",
        markersize=3,
    )

    ax.set_xlabel("Local Probability")
    ax.set_ylabel("Median Error (cm)")
    ax.set_xlim(0, 1)
    sns.despine(ax=ax)

    if standalone:
        plt.tight_layout()


def plot_decoding_error_distribution(
    data_over_time: pd.DataFrame,
    ax: plt.Axes | None = None,
) -> None:
    """Plot distribution of decoding error during local state across sessions.

    Parameters
    ----------
    data_over_time : pd.DataFrame
        Aggregated ``data_over_time`` DataFrame.
    ax : matplotlib Axes, optional
        Axes to plot on. If None, creates standalone figure.

    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=PANEL_SINGLE)

    error_df = compute_local_decoding_error(data_over_time)
    errors = error_df["median_error_cm"].dropna()

    ax.hist(
        errors,
        bins=np.histogram_bin_edges(errors, bins="auto"),
        color=COLORS["Local"],
        alpha=0.7,
        edgecolor="none",
    )

    grand_median = errors.median()
    ax.axvline(
        grand_median,
        color="black",
        linewidth=0.8,
        linestyle="--",
        label=f"Median: {grand_median:.1f} cm",
    )

    ax.set_xlabel("Median Decoding Error (cm)")
    ax.set_ylabel("Number of Sessions")
    ax.legend()
    sns.despine(ax=ax)

    if standalone:
        plt.tight_layout()


def plot_state_fractions_by_movement(
    data_over_time: pd.DataFrame,
    ax: plt.Axes | None = None,
) -> None:
    """Plot state fractions during movement vs. immobility.

    Shows stacked bars of decoder state fractions for moving and immobile
    periods, demonstrating that the model assigns mostly local during movement.
    Both bars use per-session medians for consistent aggregation.

    Parameters
    ----------
    data_over_time : pd.DataFrame
        Aggregated ``data_over_time`` DataFrame.
    ax : matplotlib Axes, optional
        Axes to plot on. If None, creates standalone figure.

    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=PANEL_SINGLE)

    group_cols = ["animal", "day", "epoch"]
    prob_cols = [
        "Local_prob",
        "No-Spike_prob",
        "Non-Local_Continuous_prob",
        "Non-Local_Fragmented_prob",
    ]

    # Compute per-session state fractions during movement
    moving_df = compute_state_fraction_during_movement(data_over_time)
    moving_frac_cols = [
        "local_fraction",
        "nospike_fraction",
        "nonlocal_continuous_fraction",
        "nonlocal_fragmented_fraction",
    ]
    moving_fracs = [moving_df[col].median() for col in moving_frac_cols]

    # Compute per-session state fractions during immobility (same aggregation)
    is_immobile = (
        data_over_time["head_speed"] <= DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC
    )
    immobile_session = (
        data_over_time.loc[is_immobile]
        .groupby(group_cols, sort=False)[prob_cols]
        .mean()
    )
    immobile_fracs = [immobile_session[col].median() for col in prob_cols]

    state_labels = [
        "Local",
        "No-Spike",
        "Non-Local\nContinuous",
        "Non-Local\nFragmented",
    ]
    # Colors consistent with Figure 1: WONG[5]=Blue, Gray, WONG[1]=Orange, WONG[6]=Vermillion
    state_colors = [
        COLORS["Local"],
        COLORS["No-Spike"],
        "#E69F00",  # WONG[1] Orange — Non-Local Continuous
        "#D55E00",  # WONG[6] Vermillion — Non-Local Fragmented
    ]

    x = np.arange(2)
    width = 0.6

    bottom_moving = 0.0
    bottom_immobile = 0.0
    for i, (label, color) in enumerate(zip(state_labels, state_colors, strict=True)):
        ax.bar(
            x[0],
            moving_fracs[i],
            width,
            bottom=bottom_moving,
            color=color,
            alpha=0.7,
            label=label,
        )
        ax.bar(
            x[1],
            immobile_fracs[i],
            width,
            bottom=bottom_immobile,
            color=color,
            alpha=0.7,
        )
        bottom_moving += moving_fracs[i]
        bottom_immobile += immobile_fracs[i]

    ax.set_xticks(x)
    ax.set_xticklabels(["Moving", "Immobile"])
    ax.set_ylabel("State Fraction")
    ax.set_ylim(0, 1)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=5,
        frameon=False,
    )
    sns.despine(ax=ax)

    if standalone:
        plt.tight_layout()


def plot_state_probability_by_theta_phase(
    data_over_time: pd.DataFrame,
    n_bins: int = 36,
    ax: plt.Axes | None = None,
) -> None:
    """Plot Local and Non-Local probability as a function of theta phase.

    During movement, non-local activity (theta sequences) should cluster at
    specific theta phases while local coding dominates others. The phase axis
    is doubled (0 to 4pi) to show oscillatory structure without edge wrapping.

    Uses per-session binning then median across sessions for consistent
    aggregation.

    Parameters
    ----------
    data_over_time : pd.DataFrame
        Aggregated ``data_over_time`` DataFrame with ``theta_phase``,
        ``Local_prob``, ``Non-Local_Continuous_prob``,
        ``Non-Local_Fragmented_prob``, ``head_speed``, and session grouping
        columns (``animal``, ``day``, ``epoch``).
    n_bins : int, optional
        Number of phase bins over [0, 2pi]. Default 36 (10-degree bins).
    ax : matplotlib Axes, optional
        Axes to plot on. If None, creates standalone figure.

    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=PANEL_SINGLE)

    group_cols = ["animal", "day", "epoch"]

    # Movement periods only — theta phase is unreliable during immobility
    is_moving = data_over_time["head_speed"] > DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC
    moving = data_over_time.loc[is_moving].copy()

    # Combined non-local probability
    moving["Non-Local_prob"] = (
        moving["Non-Local_Continuous_prob"] + moving["Non-Local_Fragmented_prob"]
    )

    # Bin theta phase into [0, 2pi) bins
    bin_edges = np.linspace(0, 2 * np.pi, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    moving["phase_bin"] = np.digitize(moving["theta_phase"], bin_edges[1:-1])
    moving["phase_bin"] = moving["phase_bin"].clip(0, n_bins - 1)

    # Per-session mean probability in each phase bin, then median across sessions
    prob_cols = ["Local_prob", "Non-Local_prob"]
    session_binned = (
        moving.groupby(group_cols + ["phase_bin"], sort=False)[prob_cols]
        .mean()
        .reset_index()
    )

    # Double the phase axis: tile bins for [0, 4pi]
    centers_doubled = np.concatenate([bin_centers, bin_centers + 2 * np.pi])

    for prob_col, color, label in [
        ("Local_prob", COLORS["Local"], "Local"),
        ("Non-Local_prob", COLORS["Non-Local"], "Non-Local"),
    ]:
        grouped = session_binned.groupby("phase_bin")[prob_col]
        median_vals = grouped.median().values
        q25_vals = grouped.quantile(0.25).values
        q75_vals = grouped.quantile(0.75).values

        # Tile for doubled axis
        median_doubled = np.tile(median_vals, 2)
        q25_doubled = np.tile(q25_vals, 2)
        q75_doubled = np.tile(q75_vals, 2)

        ax.fill_between(
            centers_doubled,
            q25_doubled,
            q75_doubled,
            color=color,
            alpha=0.3,
        )
        ax.plot(
            centers_doubled,
            median_doubled,
            color=color,
            linewidth=1.5,
            label=label,
        )

    # Format x-axis as multiples of pi
    ax.set_xticks([0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi])
    ax.set_xticklabels(["0", "\u03c0", "2\u03c0", "3\u03c0", "4\u03c0"])
    ax.set_xlim(0, 4 * np.pi)
    ax.set_xlabel("Theta Phase")
    ax.set_ylabel("State Probability")
    ax.legend(fontsize=5, frameon=False)
    sns.despine(ax=ax)

    if standalone:
        plt.tight_layout()
