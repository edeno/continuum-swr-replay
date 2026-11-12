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

from continuum_swr_replay.analysis.constants import (
    BINS_DURATION_MS,
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    DEFAULT_NON_LOCAL_THRESHOLD,
)
from continuum_swr_replay.analysis.events.analysed_time import (
    analysed_bins,
    immobile_analysed_bins,
)
from continuum_swr_replay.analysis.events.nonlocal_detection import (
    nonlocal_event_bins_by_swr,
)
from continuum_swr_replay.analysis.hippocampus import (
    compute_calibration_curve,
    compute_local_decoding_error,
    get_event_durations_ms,
    get_event_start_end_times,
)
from continuum_swr_replay.analysis.hippocampus.posterior_analysis import (
    normalized_replay_position_by_swr,
)
from continuum_swr_replay.analysis.hippocampus.quality_metrics import (
    calibration_curve_statistics,
    local_decoding_error_statistics,
    session_state_fractions,
    state_fraction_statistics,
)
from continuum_swr_replay.analysis.hippocampus.triggered_analysis import (
    compute_triggered_average_curves,
)
from continuum_swr_replay.visualization.figure_utilities import (
    PANEL_SINGLE,
    get_figure_size,
    save_figure,
)
from continuum_swr_replay.visualization.palettes import COLORS
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

    for event_type in (
        "non_local_event_immobile",
        "non_local_event_immobile_no_overlap_SWR_event",
    ):
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
            save_figure(fig, figure_name, base_dir=output_dir, dpi=dpi, formats=formats)
            logging.info(f"Saved: {figure_name}")
        plt.close(fig)


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
            data_over_time,
            event_types,
            signal_types,
            time_col=time_col,
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
                DEFAULT_NON_LOCAL_THRESHOLD,
                color="lightgrey",
                linestyle="--",
                alpha=0.5,
                linewidth=0.75,
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
            save_figure(fig, figure_name, base_dir=output_dir, dpi=dpi, formats=formats)
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
        ``Non-Local_prob``, ``head_speed`` and ``in_excluded_time`` columns.
        Bins in excluded time are left out.
    output_dir, figure_name, formats, dpi :
        Standalone-mode save parameters (ignored when ``ax`` is provided).
    ax : matplotlib Axes, optional
        Axes to plot on. If None, creates standalone figure.

    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=PANEL_SINGLE)

    # Immobile bins outside excluded time. The mask bins have no SWR z-score,
    # but the bins in its 0.3-s reach do, and they are not analysed either.
    is_immobile = immobile_analysed_bins(data_over_time)

    local_mask = is_immobile & (
        data_over_time["Local_prob"] > DEFAULT_NON_LOCAL_THRESHOLD
    )
    ax.hist(
        data_over_time.loc[local_mask, "SWR_zscore"],
        bins=np.arange(-5, 11),
        alpha=0.7,
        density=True,
        label="Local",
        color=COLORS["Local"],
    )

    nonlocal_mask = is_immobile & (
        data_over_time["Non-Local_prob"] > DEFAULT_NON_LOCAL_THRESHOLD
    )
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
            save_figure(fig, figure_name, base_dir=output_dir, dpi=dpi, formats=formats)
            logging.info(f"Saved: {figure_name}")
        plt.close(fig)


#: The two classes of immobile non-local events, as the event table's ``is_swr``,
#: with the colour and legend label each per-event panel gives them.
_EVENT_CLASSES = (
    (False, "non_swr", COLORS["Non-Local"], "non-SWR"),
    (True, "swr", COLORS["SWR"], "SWR"),
)


def _event_class_histograms(
    events: pd.DataFrame,
    column: str,
    medians: dict[str, float],
    bins: np.ndarray,
    ax: plt.Axes,
) -> None:
    """Each class's per-event ``column`` as a density histogram, with a dashed
    line at the class median it is given (gid ``median_<class>``)."""
    for is_swr, word, color, label in _EVENT_CLASSES:
        ax.hist(
            events.loc[events["is_swr"] == is_swr, column].dropna(),
            bins=bins,
            density=True,
            alpha=0.6,
            color=color,
            edgecolor="none",
            label=label,
        )
        ax.axvline(
            medians[word],
            color=color,
            linestyle="--",
            linewidth=1.0,
            gid=f"median_{word}",
        )


def plot_event_ripple_by_swr(
    events: pd.DataFrame,
    medians: dict[str, float],
    references: dict[str, float],
    ax: plt.Axes | None = None,
) -> None:
    """Ripple power of immobile non-local events, SWR-overlapping against non-SWR.

    Each event's value is its mean over its bins of the epoch's z-scored ripple
    power (the event table's ``ripple_power_z``). The lines are drawn where they
    are given, not computed here: each class's median (dashed, gid
    ``median_non_swr`` / ``median_swr``) and the running and quiet-rest
    references (gid ``reference_running`` / ``reference_rest``).

    Parameters
    ----------
    events : pd.DataFrame, shape (n_events, n_columns)
        The canonical event table, with ``is_swr`` and ``ripple_power_z``.
    medians : dict
        ``non_swr`` and ``swr``: each class's median event value, z.
    references : dict
        ``running`` and ``rest``: the typical running and quiet-rest level, z.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If None, a standalone figure is created.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=PANEL_SINGLE)

    _event_class_histograms(
        events, "ripple_power_z", medians, np.arange(-3.0, 10.5, 0.5), ax
    )
    ax.axvline(
        references["rest"],
        color="grey",
        linestyle=":",
        linewidth=1.0,
        label="quiet rest",
        gid="reference_rest",
    )
    ax.axvline(
        references["running"],
        color="black",
        linestyle="-.",
        linewidth=1.0,
        label="running",
        gid="reference_running",
    )

    ax.set_xlabel("Ripple power z (event mean)")
    ax.set_ylabel("Density")
    ax.legend()

    if standalone:
        sns.despine(ax=ax)
        plt.tight_layout()


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

    swr_mask, no_swr_mask = nonlocal_event_bins_by_swr(data_over_time)
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
            save_figure(fig, figure_name, base_dir=output_dir, dpi=dpi, formats=formats)
            logging.info(f"Saved: {figure_name}")
        plt.close(fig)


def plot_event_speed_by_swr(
    events: pd.DataFrame,
    medians: dict[str, float],
    contrasts: pd.Series,
    axes: list[plt.Axes] | None = None,
) -> None:
    """Replay speed of immobile non-local events, SWR-overlapping against non-SWR.

    Each event's value is the median over its bins of the decoded position's
    speed (the event table's ``replay_speed_median_cm_s``). The first axes
    draws each class's events as a density histogram and its median as a
    dashed line (gid ``median_non_swr`` / ``median_swr``); the second draws the
    SWR minus non-SWR difference of medians, plain and at matched duration,
    each with its 95 % nested-bootstrap interval (gid ``difference`` /
    ``matched_difference``). Every number is drawn as given; none is computed
    here.

    Parameters
    ----------
    events : pd.DataFrame, shape (n_events, n_columns)
        The canonical event table, with ``is_swr`` and
        ``replay_speed_median_cm_s``.
    medians : dict
        ``non_swr`` and ``swr``: each class's median event value, cm/s.
    contrasts : pd.Series
        ``difference``, ``difference_low``, ``difference_high``,
        ``matched_difference``, ``matched_difference_low`` and
        ``matched_difference_high``, cm/s: the bootstrap's estimates and
        interval bounds.
    axes : list of matplotlib.axes.Axes, optional
        Two axes, [histograms, differences]. If None, a standalone figure is
        created.
    """
    standalone = axes is None
    if standalone:
        fig, axes = plt.subplots(
            1, 2, figsize=PANEL_SINGLE, gridspec_kw={"width_ratios": [3, 1]}
        )
    if len(axes) != 2:
        raise ValueError(f"plot_event_speed_by_swr needs 2 axes, got {len(axes)}")
    ax_hist, ax_difference = axes

    _event_class_histograms(
        events, "replay_speed_median_cm_s", medians, np.arange(0, 3000, 100), ax_hist
    )
    ax_hist.set_xlabel("Replay speed (cm/s, event median)")
    ax_hist.set_ylabel("Density")
    ax_hist.legend(loc="upper right")

    for x, word in enumerate(("difference", "matched_difference")):
        estimate = contrasts[word]
        ax_difference.errorbar(
            x,
            estimate,
            yerr=[
                [estimate - contrasts[f"{word}_low"]],
                [contrasts[f"{word}_high"] - estimate],
            ],
            fmt="o",
            color="black",
            markersize=3,
            capsize=2,
            linewidth=1.0,
            gid=word,
        )
    ax_difference.axhline(0.0, color="grey", linewidth=0.75)
    ax_difference.set_xticks([0, 1])
    ax_difference.set_xticklabels(
        ["all", "duration-\nmatched"], rotation=45, ha="right", rotation_mode="anchor"
    )
    ax_difference.set_xlim(-0.6, 1.6)
    ax_difference.set_ylabel("SWR − non-SWR (cm/s)")

    if standalone:
        sns.despine(fig=fig)
        plt.tight_layout()


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

    positions, n_position_bins = normalized_replay_position_by_swr(data_over_time)
    ax.hist(
        positions["swr"],
        bins=n_position_bins,
        density=True,
        alpha=0.7,
        color=COLORS["SWR"],
        label="SWR-coincident non-local",
    )

    ax.hist(
        positions["non_swr"],
        bins=n_position_bins,
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
            save_figure(fig, figure_name, base_dir=output_dir, dpi=dpi, formats=formats)
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

    curve = calibration_curve_statistics(compute_calibration_curve(data_over_time))

    ax.fill_between(
        curve.index,
        curve["q25"].values,
        curve["q75"].values,
        color=COLORS["Local"],
        alpha=0.3,
    )
    ax.plot(
        curve.index,
        curve["median"].values,
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

    grand_median = local_decoding_error_statistics(error_df)["median_error_cm"]
    ax.axvline(
        grand_median,
        color="black",
        linewidth=0.8,
        linestyle="--",
        label=f"Median: {grand_median:.1f} cm",
    )

    # Short: Figure 2 draws this in its narrow right-edge cell, where the longer
    # "Median Decoding Error (cm)" ran off the saved figure.
    ax.set_xlabel("Median error (cm)")
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
    The fractions are
    :func:`~continuum_swr_replay.analysis.hippocampus.quality_metrics.state_fraction_statistics`'s
    over :func:`~continuum_swr_replay.analysis.hippocampus.quality_metrics.session_state_fractions`:
    the median over sessions of each session's mean state probability, as the
    events summary reports them.

    Parameters
    ----------
    data_over_time : pd.DataFrame
        Aggregated ``data_over_time`` DataFrame, with ``in_excluded_time``.
        Bins in excluded time are left out of both bars.
    ax : matplotlib Axes, optional
        Axes to plot on. If None, creates standalone figure.

    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=PANEL_SINGLE)

    fractions = state_fraction_statistics(session_state_fractions(data_over_time))
    moving_fracs = list(fractions.loc["moving"])
    immobile_fracs = list(fractions.loc["immobile"])

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
        ``Non-Local_Fragmented_prob``, ``head_speed``, ``in_excluded_time``
        and session grouping columns (``animal``, ``day``, ``epoch``). Bins in
        excluded time are left out.
    n_bins : int, optional
        Number of phase bins over [0, 2pi]. Default 36 (10-degree bins).
    ax : matplotlib Axes, optional
        Axes to plot on. If None, creates standalone figure.

    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=PANEL_SINGLE)

    group_cols = ["animal", "day", "epoch"]

    # Movement periods only — theta phase is unreliable during immobility —
    # outside excluded time
    is_moving = (
        data_over_time["head_speed"] > DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC
    ) & analysed_bins(data_over_time)
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
        data,
        time_slice,
        detector,
        env,
        results,
        ax_raster,
    )
    plot_conditional_non_local_posterior(
        data,
        time_slice,
        results,
        env,
        ax_post,
    )
    plot_acausal_probabilities(
        results,
        time_slice,
        ax_probs,
        plot_local=True,
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
    percent_labels: dict[str, str],
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
    percent_labels : dict
        For each of ``swr_contained``, ``swr_partial`` and ``non_swr``, the
        class's share of the events as a percent, already rounded and printed
        (``"57"``). Written on the class's segment, followed by ``%``; no
        percent is computed here.
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
        "swr_partial": "#b5825a",  # neutral sand: overlap but not contained
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
                f"{percent_labels[cls]}%\n{segment_labels[cls]}",
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
    medians: dict[str, float],
    median_labels: dict[str, str],
    axes: list[plt.Axes] | None = None,
    *,
    xlim_ms: tuple[float, float] = (-100.0, 100.0),
    bin_width_ms: float = 5.0,
) -> None:
    """Two side-by-side histograms of non-local/SWR timing offsets.

    Takes a per-event ``start_offset_s`` / ``end_offset_s`` DataFrame and
    renders:

        - Left axis:  non-local-start minus SWR-start (ms)
        - Right axis: non-local-end   minus SWR-end   (ms)

    The offsets are those of
    :func:`~continuum_swr_replay.analysis.hippocampus.event_statistics.boundary_offsets`
    on the canonical event table: each immobile, SWR-overlapping event's
    ``event_start_s`` / ``event_end_s`` minus its matched SWR's
    ``matched_swr_start`` / ``matched_swr_end`` — one row per event.

    Positive left-axis / negative right-axis means non-local events
    are typically contained inside the SWR window. A long tail on
    either side means a subset of events extend beyond the SWR
    envelope.

    Parameters
    ----------
    offsets : pd.DataFrame, shape (n_overlapping_events, 2)
        ``start_offset_s`` and ``end_offset_s`` (seconds).
    medians : dict
        For each of those columns, its median in seconds
        (:func:`~continuum_swr_replay.analysis.hippocampus.event_statistics.boundary_offset_statistics`'s
        ``start_offset_median_s`` / ``end_offset_median_s``). The dashed line
        is drawn there; it is not computed here.
    median_labels : dict
        For each of those columns, its median in milliseconds, already rounded
        and printed (``"+4"``). Written in the panel's corner as
        ``median <label> ms``.
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
            1,
            2,
            figsize=(5.0, 2.2),
            sharey=True,
            constrained_layout=True,
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
            "start_offset_s",
            "NL start − SWR start (ms)",
            "Start alignment",
        ),
        (
            ax_end,
            "end_offset_s",
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

        median_ms = 1000.0 * medians[col]
        ax.axvline(
            0.0,
            color="grey",
            linestyle="-",
            linewidth=0.75,
            alpha=0.6,
        )
        ax.axvline(
            median_ms,
            color="black",
            linestyle="--",
            linewidth=1.0,
        )
        # Annotate the median as plain text in a top corner rather than a
        # legend. The median line sits near zero (panel centre), so a
        # corner placement keeps the text clear of the dashed line.
        ax.text(
            0.97,
            0.97,
            f"median {median_labels[col]} ms",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6,
        )

        ax.set_xlabel(xlabel)
        ax.set_xlim(xlim_ms)
        ax.set_title(title, fontsize=8, pad=4)
        sns.despine(ax=ax)

    ax_start.set_ylabel("Density")

    if standalone:
        plt.tight_layout()
