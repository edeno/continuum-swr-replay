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
            time, no_overlap_mean.values,
            label="No SWR Overlap", color=COLORS["Non-Local"],
        )
        ax.plot(
            time, overlap_mean.values,
            label="SWR Overlap", color=COLORS["SWR"],
        )

        if signal_type.endswith("_zscore"):
            name = signal_type.replace("_zscore", "").replace("_", " ").title()
            if name == "Theta Power":
                name = "Theta"
            ax.set_ylabel(f"{name}\nZ-score")
            ax.axhline(2.0, color="lightgrey", linestyle="--", alpha=0.5, linewidth=0.75)
        elif signal_type == "Non-Local_prob":
            ax.set_ylabel("Non-Local\nProb.")
            ax.axhline(0.8, color="lightgrey", linestyle="--", alpha=0.5, linewidth=0.75)
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
    """Plot SWR z-score distribution by local/non-local state."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=PANEL_SINGLE)

    is_immobile = data_over_time.head_speed < DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC

    local_mask = is_immobile & (data_over_time["Local_prob"] > 0.8)
    ax.hist(
        data_over_time.loc[local_mask, "SWR_zscore"],
        bins=np.arange(-5, 11),
        alpha=0.7, density=True, label="Local", color=COLORS["Local"],
    )

    nonlocal_mask = is_immobile & (data_over_time["Non-Local_prob"] > 0.8)
    ax.hist(
        data_over_time.loc[nonlocal_mask, "SWR_zscore"],
        bins=np.arange(-5, 11),
        alpha=0.7, density=True, label="Non-Local", color=COLORS["Non-Local"],
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
    """Plot replay speed distribution by SWR overlap status."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=PANEL_SINGLE)

    bins = np.arange(0, 3000, 100)

    swr_mask = (data_over_time["non_local_event"] > 0) & (
        data_over_time["SWR_event"] > 0
    )
    ax.hist(
        data_over_time.loc[swr_mask, "map_speed"],
        bins=bins, density=True, alpha=0.7,
        color=COLORS["SWR"], label="With SWR",
    )

    no_swr_mask = (data_over_time["non_local_event"] > 0) & (
        data_over_time["SWR_event"] == 0
    )
    ax.hist(
        data_over_time.loc[no_swr_mask, "map_speed"],
        bins=bins, density=True, alpha=0.7,
        color=COLORS["Non-Local"], label="Without SWR",
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
    """Plot replay position distribution by SWR overlap status."""
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
        bins=bins, density=True, alpha=0.7,
        color=COLORS["SWR"], label="With SWR",
    )

    no_swr_mask = (data_over_time["non_local_event"] > 0) & (
        data_over_time["SWR_event"] == 0
    )
    ax.hist(
        data_over_time.loc[no_swr_mask, "map_estimate"]
        / data_over_time.loc[no_swr_mask, "max_linear_position"],
        bins=bins, density=True, alpha=0.7,
        color=COLORS["Non-Local"], label="Without SWR",
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
