"""Reusable panel-level plotting functions for composite figures.

These functions plot individual figure panels and can accept a matplotlib Axes
object for embedding in composite layouts. When called without an ``ax``
argument, they create a standalone figure.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.patches as mpatches
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
from continuum_swr_replay.analysis.hippocampus.event_detection import (
    event_strict_overlap_mask,
)
from continuum_swr_replay.analysis.hippocampus.event_immobility import (
    filter_event_labels_by_speed,
    validate_nonlocal_event_probability_contract,
)
from continuum_swr_replay.figure_utilities import (
    PANEL_SINGLE,
    PANEL_SINGLE_SQUARE,
    get_figure_size,
    save_figure,
)
from continuum_swr_replay.parameters import (
    COLORS,
    SAMPLING_FREQUENCY,
)
from continuum_swr_replay.visualization.plots import (
    plot_acausal_probabilities,
    plot_conditional_non_local_posterior,
    plot_ripple_consensus_trace_zscore,
    plot_spike_times,
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

    for event_type in ("non_local_event_immobile", "non_local_event_immobile_no_overlap_SWR_event"):
        event_times = get_event_start_end_times(
            event_types[event_type], event_col="non_local_event"
        )
        duration = get_event_durations_ms(event_times)

        color = (
            COLORS["Non-Local"]
            if event_type == "non_local_event_immobile_no_overlap_SWR_event"
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
        if figure_name is not None:
            save_figure(figure_name, base_dir=output_dir, dpi=dpi, formats=formats)
            logging.info(f"Saved: {figure_name}")
        plt.close(fig)


def compute_triggered_average_curves(
    data_over_time: pd.DataFrame,
    event_types: dict,
    signals: list[str],
    time_col: str = "start_time",
    window_sec: float = 2.0,
) -> dict:
    """Reduce ``data_over_time`` to event-triggered mean/SEM curves.

    Splits the work that :func:`plot_triggered_averages` used to do inline
    into a small, cacheable product: for each signal, the time axis and the
    across-event mean and SEM for non-SWR (no-overlap) vs SWR-coincident
    (overlap) immobile non-local events. The returned dict is a few KB,
    letting callers cache it instead of the multi-GB ``data_over_time``.

    Parameters
    ----------
    data_over_time : pd.DataFrame
        Aggregated per-timebin dataset.
    event_types : dict
        Output of :func:`get_event_types`.
    signals : list of str
        Signal columns to reduce, in order.
    time_col : str
        Alignment column ('start_time' or 'end_time').
    window_sec : float
        Half-window around each event, in seconds.

    Returns
    -------
    dict
        ``{signal: {"time", "no_overlap_mean", "no_overlap_sem",
        "overlap_mean", "overlap_sem"}}`` with numpy-array values.

    """
    signal_types = list(signals)
    no_overlap_times = get_event_start_end_times(
        event_types["non_local_event_immobile_no_overlap_SWR_event"],
        event_col="non_local_event",
    )
    overlap_times = get_event_start_end_times(
        event_types["non_local_event_immobile_overlap_SWR_event"],
        event_col="non_local_event",
    )
    windows_no = get_signal_windows(
        data=data_over_time,
        event_times=no_overlap_times,
        signal_cols=signal_types,
        time_col=time_col,
        window_sec=window_sec,
        sampling_frequency=SAMPLING_FREQUENCY,
        event_id_col="non_local_event",
    )
    windows_ov = get_signal_windows(
        data=data_over_time,
        event_times=overlap_times,
        signal_cols=signal_types,
        time_col=time_col,
        window_sec=window_sec,
        sampling_frequency=SAMPLING_FREQUENCY,
        event_id_col="non_local_event",
    )
    no_mean_all = windows_no.mean(dim="event")
    ov_mean_all = windows_ov.mean(dim="event")

    curves: dict = {}
    for signal_type in signal_types:
        no_mean = no_mean_all.sel(signal=signal_type)
        curves[signal_type] = {
            "time": no_mean["time"].values,
            "no_overlap_mean": no_mean.values,
            "no_overlap_sem": sem(
                windows_no.sel(signal=signal_type).values,
                axis=0,
                nan_policy="omit",
            ),
            "overlap_mean": ov_mean_all.sel(signal=signal_type).values,
            "overlap_sem": sem(
                windows_ov.sel(signal=signal_type).values,
                axis=0,
                nan_policy="omit",
            ),
        }
    return curves


def plot_triggered_averages(
    data_over_time: pd.DataFrame,
    event_types: dict,
    output_dir: Path | None = None,
    figure_name: str | None = None,
    time_col: str = "start_time",
    formats: list[str] | None = None,
    dpi: int = 300,
    axes: list[plt.Axes] | None = None,
    signals: list[str] | None = None,
    precomputed: dict | None = None,
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
        Axes to draw on, one per signal in ``signals`` (in order). If
        None, a standalone figure is created with one row per signal.
    signals : list of str, optional
        Which signal columns to plot, in order top-to-bottom. Defaults
        to the full four-signal stack used in the original Figure 2
        design. For the current (reduced) Figure 2, pass
        ``["Non-Local_prob", "SWR_zscore"]``.
    precomputed : dict, optional
        Output of :func:`compute_triggered_average_curves`. When given,
        the per-event windowing is skipped and the curves are plotted
        directly from this dict (so ``data_over_time`` is unused). Lets a
        caller cache the small reduced product instead of the full
        dataset. Must cover every signal in ``signals``.

    """
    if signals is None:
        signal_types = [
            "Non-Local_prob",
            "theta_power_zscore",
            "MUA_zscore",
            "SWR_zscore",
        ]
    else:
        signal_types = list(signals)

    if precomputed is None:
        curves = compute_triggered_average_curves(
            data_over_time, event_types, signal_types, time_col=time_col,
        )
    else:
        curves = precomputed

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

    for i, (ax, signal_type) in enumerate(zip(axes, signal_types, strict=True)):
        curve = curves[signal_type]
        time = curve["time"]
        no_overlap_mean = curve["no_overlap_mean"]
        overlap_mean = curve["overlap_mean"]
        no_overlap_sem = curve["no_overlap_sem"]
        overlap_sem = curve["overlap_sem"]

        ax.fill_between(
            time,
            no_overlap_mean - no_overlap_sem,
            no_overlap_mean + no_overlap_sem,
            color=COLORS["Non-Local"],
            alpha=0.3,
        )
        ax.fill_between(
            time,
            overlap_mean - overlap_sem,
            overlap_mean + overlap_sem,
            color=COLORS["SWR"],
            alpha=0.3,
        )

        ax.plot(
            time,
            no_overlap_mean,
            label="non-SWR non-local",
            color=COLORS["Non-Local"],
        )
        ax.plot(
            time,
            overlap_mean,
            label="SWR-coincident non-local",
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
        if figure_name is not None:
            save_figure(figure_name, base_dir=output_dir, dpi=dpi, formats=formats)
            logging.info(f"Saved: {figure_name}")
        plt.close(fig)


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
        if figure_name is not None:
            save_figure(figure_name, base_dir=output_dir, dpi=dpi, formats=formats)
            logging.info(f"Saved: {figure_name}")
        plt.close(fig)


def _nonlocal_swr_event_bin_masks(
    data_over_time: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Return whole-event SWR/non-SWR masks for eligible Non-Local bins."""
    validate_nonlocal_event_probability_contract(data_over_time)
    eligible = filter_event_labels_by_speed(
        data_over_time,
        ["non_local_event", "SWR_event"],
    )
    active = eligible["non_local_event"] > 0
    is_swr_event = event_strict_overlap_mask(
        eligible, "non_local_event", "SWR_event"
    )
    return active & is_swr_event, active & ~is_swr_event


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

    swr_mask, no_swr_mask = _nonlocal_swr_event_bin_masks(data_over_time)
    ax.hist(
        data_over_time.loc[swr_mask, "map_speed"],
        bins=bins,
        density=True,
        alpha=0.7,
        color=COLORS["SWR"],
        label="SWR-coincident non-local",
    )

    ax.hist(
        data_over_time.loc[no_swr_mask, "map_speed"],
        bins=bins,
        density=True,
        alpha=0.7,
        color=COLORS["Non-Local"],
        label="non-SWR non-local",
    )

    ax.set_xlabel("Replay Speed (cm/s)")
    ax.set_ylabel("Density")
    ax.legend()

    if standalone:
        sns.despine(ax=ax)
        plt.tight_layout()
        if figure_name is not None:
            save_figure(figure_name, base_dir=output_dir, dpi=dpi, formats=formats)
            logging.info(f"Saved: {figure_name}")
        plt.close(fig)


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

    swr_mask, no_swr_mask = _nonlocal_swr_event_bin_masks(data_over_time)
    ax.hist(
        data_over_time.loc[swr_mask, "map_estimate"]
        / data_over_time.loc[swr_mask, "max_linear_position"],
        bins=bins,
        density=True,
        alpha=0.7,
        color=COLORS["SWR"],
        label="SWR-coincident non-local",
    )

    ax.hist(
        data_over_time.loc[no_swr_mask, "map_estimate"]
        / data_over_time.loc[no_swr_mask, "max_linear_position"],
        bins=bins,
        density=True,
        alpha=0.7,
        color=COLORS["Non-Local"],
        label="non-SWR non-local",
    )

    ax.set_xlabel("Normalized Replay Position")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 1)
    ax.legend()

    if standalone:
        sns.despine(ax=ax)
        plt.tight_layout()
        if figure_name is not None:
            save_figure(figure_name, base_dir=output_dir, dpi=dpi, formats=formats)
            logging.info(f"Saved: {figure_name}")
        plt.close(fig)


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
    data_over_time: pd.DataFrame | None = None,
    ax: plt.Axes | None = None,
    error_df: pd.DataFrame | None = None,
) -> None:
    """Plot distribution of decoding error during local state across sessions.

    Parameters
    ----------
    data_over_time : pd.DataFrame, optional
        Aggregated ``data_over_time`` DataFrame. Required unless
        ``error_df`` is supplied.
    ax : matplotlib Axes, optional
        Axes to plot on. If None, creates standalone figure.
    error_df : pd.DataFrame, optional
        Precomputed output of :func:`compute_local_decoding_error` (one
        row per session with ``median_error_cm``). When given,
        ``data_over_time`` is not used, so a caller can cache this small
        per-session table instead of the full dataset.

    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=PANEL_SINGLE)

    if error_df is None:
        if data_over_time is None:
            raise ValueError(
                "plot_decoding_error_distribution requires either "
                "data_over_time or a precomputed error_df."
            )
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


def plot_event_example_stack(
    data: dict,
    results,
    detector,
    env,
    time_slice: slice,
    axes: list,
    *,
    column_title: str | None = None,
    show_ylabels: bool = True,
    relative_time_center: float | None = None,
) -> None:
    """Plot one stacked example event (SWR / raster / posterior / state probs).

    Produces the 4-row visual used as a building block for panel a in
    Figures 2 and 3. The caller is responsible for creating and arranging
    the four axes (typically from a GridSpec).

    Parameters
    ----------
    data : dict
        Session data dictionary from :func:`load_data`.
    results : xr.Dataset
        Decoding results for the session.
    detector : Detector
        Fitted non-local detector, used by the raster and posterior plots.
    env : Environment
        Environment object matching the detector.
    time_slice : slice
        Time window to plot (absolute seconds).
    axes : list of matplotlib.axes.Axes
        Exactly four axes, ordered top-to-bottom: SWR trace, raster,
        conditional non-local posterior, acausal state probabilities.
    column_title : str, optional
        Title to place above the top sub-row (SWR trace axis).
    show_ylabels : bool, default True
        If False, the y-axis labels are blanked — useful when this column
        is to the right of another column that already carries the labels.
    relative_time_center : float, optional
        If provided, the x-axis tick labels on the bottom sub-row are
        rewritten as offsets from this absolute time (e.g. the event
        center). If None, absolute timestamps are kept.

    """
    if len(axes) != 4:
        raise ValueError(
            f"plot_event_example_stack needs exactly 4 axes, got {len(axes)}"
        )
    ax_swr, ax_raster, ax_post, ax_probs = axes

    plot_ripple_consensus_trace_zscore(data, time_slice, ax_swr)
    plot_spike_times(
        data, time_slice, detector, env, results, ax_raster,
    )
    plot_conditional_non_local_posterior(
        data, time_slice, results, env, detector, ax_post,
    )
    plot_acausal_probabilities(
        results, time_slice, ax_probs, plot_local=True,
    )

    # Tighten the x-limits to the requested window. The four sub-row
    # plotters each autoscale to their own data with default margins,
    # which leaves the window loose and asymmetric about the event
    # center; pin it to the exact slice so the displayed time is tight.
    for ax in (ax_swr, ax_raster, ax_post, ax_probs):
        ax.set_xlim(time_slice.start, time_slice.stop)

    # Hide x tick labels on every sub-row except the bottom one
    for ax in (ax_swr, ax_raster, ax_post):
        ax.tick_params(labelbottom=False)

    if relative_time_center is not None:
        # Place ticks at round offsets from the event center so the
        # bottom row reads as clean ±s values within the tight window.
        half = (time_slice.stop - time_slice.start) / 2.0
        offsets = np.arange(-0.4, 0.4 + 1e-9, 0.2)
        offsets = offsets[np.abs(offsets) <= half + 1e-9]
        ax_probs.set_xticks(relative_time_center + offsets)
        ax_probs.set_xticklabels([f"{o:.1f}" for o in offsets])
        ax_probs.set_xlabel("Time from event center (s)")

    if not show_ylabels:
        for ax in (ax_swr, ax_raster, ax_post, ax_probs):
            ax.set_ylabel("")

    if column_title is not None:
        ax_swr.set_title(column_title, fontsize=8, pad=6)


def plot_swr_overlap_class_bar(
    counts: dict[str, int],
    ax: plt.Axes | None = None,
) -> None:
    """Horizontal stacked bar of non-local events by SWR-overlap class.

    Shows the three-way breakdown of immobile non-local events into
    SWR-contained, SWR-partial, and non-SWR. Designed as Figure 2
    panel b — the same three categories that panel a's three example
    columns represent.

    Parameters
    ----------
    counts : dict
        Output of :func:`compute_swr_overlap_class_counts`. Must
        contain the keys ``swr_contained``, ``swr_partial``,
        ``non_swr``, and ``total``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If None, a standalone figure is created.

    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=PANEL_SINGLE)

    total = counts["total"]
    if total == 0:
        raise ValueError("No events in counts dict (total=0)")

    # Order: SWR-contained (leftmost) -> SWR-partial -> non-SWR
    # (rightmost). Matches the left-to-right ordering of the three
    # example columns in Figure 2 panel a.
    order = ("swr_contained", "swr_partial", "non_swr")
    segment_colors = {
        "swr_contained": COLORS["SWR"],
        "swr_partial": "#b5825a",        # neutral sand: overlap but not contained
        "non_swr": COLORS["Non-Local"],
    }
    segment_labels = {
        "swr_contained": "SWR-\ncontained",
        "swr_partial": "SWR-\npartial",
        "non_swr": "non-SWR",
    }

    left = 0.0
    for cls in order:
        frac = counts[cls] / total
        ax.barh(
            0,
            frac,
            left=left,
            height=0.55,
            color=segment_colors[cls],
            edgecolor="white",
            linewidth=0.8,
        )
        if frac > 0.05:
            ax.text(
                left + frac / 2,
                0,
                f"{100 * frac:.0f}%\n{segment_labels[cls]}",
                ha="center",
                va="center",
                color="white",
                fontsize=6,
                fontweight="bold",
            )
        left += frac

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([0])
    ax.set_yticklabels(
        [f"non-local events\n(immobile, n={total:,})"],
        fontsize=7,
    )
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Fraction of events")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    if standalone:
        plt.tight_layout()


def plot_event_timing_offsets(
    offsets: pd.DataFrame,
    axes: list[plt.Axes] | None = None,
    *,
    xlim_ms: tuple[float, float] = (-100.0, 100.0),
    bin_width_ms: float = 5.0,
) -> None:
    """Two side-by-side histograms of non-local/SWR timing offsets.

    Takes a per-event ``start_diff`` / ``end_diff`` DataFrame and renders:

        - Left axis:  non-local-start minus SWR-start (ms)
        - Right axis: non-local-end   minus SWR-end   (ms)

    The offsets are computed inline in ``generate_figure02.generate_figure``
    by subtracting the matched SWR's ``matched_swr_start`` /
    ``matched_swr_end`` from each event's ``event_start`` / ``event_end``
    (columns of :func:`build_classified_nonlocal_event_table`), restricted
    to immobile, SWR-overlapping events — one row per event.

    Positive left-axis / negative right-axis means non-local events
    are typically contained inside the SWR window. A long tail on
    either side means a subset of events extend beyond the SWR
    envelope.

    Parameters
    ----------
    offsets : pd.DataFrame
        Per-overlap offset DataFrame with ``start_diff`` and
        ``end_diff`` columns (seconds).
    axes : list of matplotlib.axes.Axes, optional
        Two axes [left, right]. If None, a standalone 1x2 figure
        is created.
    xlim_ms : tuple of float
        X-axis limits in milliseconds. Default ±100 ms.
    bin_width_ms : float
        Histogram bin width in milliseconds.

    """
    standalone = axes is None
    if standalone:
        fig, axes = plt.subplots(
            1, 2, figsize=(5.0, 2.2),
            sharey=True, constrained_layout=True,
        )
        axes = list(axes)

    if len(axes) != 2:
        raise ValueError(
            f"plot_event_timing_offsets needs exactly 2 axes, got {len(axes)}"
        )
    ax_start, ax_end = axes

    bins = np.arange(xlim_ms[0], xlim_ms[1] + bin_width_ms, bin_width_ms)

    for ax, col, xlabel, title in (
        (
            ax_start,
            "start_diff",
            "NL start − SWR start (ms)",
            "Start alignment",
        ),
        (
            ax_end,
            "end_diff",
            "NL end − SWR end (ms)",
            "End alignment",
        ),
    ):
        values_ms = offsets[col].to_numpy() * 1000.0
        ax.hist(
            values_ms,
            bins=bins,
            color=COLORS["Non-Local"],
            alpha=0.7,
            edgecolor="none",
            density=True,
        )

        median_ms = float(np.median(values_ms))
        ax.axvline(
            0.0, color="grey", linestyle="-", linewidth=0.75, alpha=0.6,
        )
        ax.axvline(
            median_ms, color="black", linestyle="--", linewidth=1.0,
        )
        # Annotate the median as plain text in a top corner rather than a
        # legend. The median line sits near zero (panel centre), so a
        # corner placement keeps the text clear of the dashed line.
        ax.text(
            0.97, 0.97, f"median {median_ms:+.0f} ms",
            transform=ax.transAxes, ha="right", va="top", fontsize=6,
        )

        ax.set_xlabel(xlabel)
        ax.set_xlim(xlim_ms)
        ax.set_title(title, fontsize=8, pad=4)
        sns.despine(ax=ax)

    ax_start.set_ylabel("Density")

    if standalone:
        plt.tight_layout()



def plot_swr_nl_venn_2circle(
    counts: dict[str, int],
    ax: plt.Axes | None = None,
) -> None:
    """Two-circle Venn of SWR events and immobile non-local events.

    Shows the three-region breakdown:

    - "SWR only"         = SWR events not overlapping any non-local event
    - "both"             = the shared overlap region
    - "non-local only"   = immobile non-local events not overlapping any SWR

    Each region's count is annotated along with the percent of its
    parent population, which makes the ~68% SWR → NL vs ~60% NL → no-SWR
    asymmetry readable at a glance.

    Parameters
    ----------
    counts : dict
        Output of :func:`compute_swr_nl_event_counts`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If None, a standalone figure is created.

    Notes
    -----
    The two "shared" counts (``n_swr_with_nl`` vs ``n_nl_immobile_with_swr``)
    typically differ by <1% because a single SWR can overlap multiple
    non-local events or vice versa. Their mean is shown as a single
    number in the overlap region; the caption can clarify.

    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=PANEL_SINGLE_SQUARE)

    n_swr = counts.get("n_b", counts.get("n_swr", 0))
    n_nl = counts.get("n_a", counts.get("n_nl_immobile", 0))
    n_swr_with_nl = counts.get("n_b_with_a", counts.get("n_swr_with_nl", 0))
    n_nl_with_swr = counts.get("n_a_with_b", counts.get("n_nl_immobile_with_swr", 0))

    n_swr_only = n_swr - n_swr_with_nl
    n_nl_only = n_nl - n_nl_with_swr
    n_shared = (n_swr_with_nl + n_nl_with_swr) // 2

    # Circle geometry in data coordinates; set_aspect("equal") keeps circles
    # round regardless of panel aspect.
    r = 1.0
    c_swr = (-0.55, 0.0)
    c_nl = (0.55, 0.0)

    circle_swr = mpatches.Circle(
        c_swr, r, fc=COLORS["SWR"], alpha=0.45, ec="black", lw=0.8,
    )
    circle_nl = mpatches.Circle(
        c_nl, r, fc=COLORS["Non-Local"], alpha=0.45, ec="black", lw=0.8,
    )
    ax.add_patch(circle_swr)
    ax.add_patch(circle_nl)

    # Region counts inside each visible wedge
    ax.text(
        -1.1, 0.0, f"{n_swr_only:,}",
        ha="center", va="center", fontsize=8, fontweight="bold",
    )
    ax.text(
        1.1, 0.0, f"{n_nl_only:,}",
        ha="center", va="center", fontsize=8, fontweight="bold",
    )
    ax.text(
        0.0, 0.0, f"{n_shared:,}",
        ha="center", va="center", fontsize=8, fontweight="bold", color="white",
    )

    # Circle labels above the circles
    ax.text(
        c_swr[0], 1.15, f"SWR\n(n={n_swr:,})",
        ha="center", va="bottom", fontsize=7, fontweight="bold",
        color=COLORS["SWR"],
    )
    ax.text(
        c_nl[0], 1.15, f"non-local\n(n={n_nl:,})",
        ha="center", va="bottom", fontsize=7, fontweight="bold",
        color=COLORS["Non-Local"],
    )

    # Leave plenty of vertical headroom for the two labels above
    ax.set_xlim(-2.1, 2.1)
    ax.set_ylim(-1.3, 1.7)
    ax.set_aspect("equal")
    ax.axis("off")

    if standalone:
        plt.tight_layout()
