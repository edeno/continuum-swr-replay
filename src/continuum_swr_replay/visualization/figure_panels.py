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

from continuum_swr_replay.analysis.constants import BINS_DURATION_MS
from continuum_swr_replay.analysis.decode_quality.posterior_analysis import (
    normalized_replay_position_by_swr,
)
from continuum_swr_replay.analysis.decode_quality.quality_metrics import (
    calibration_curve_statistics,
    compute_calibration_curve,
    compute_local_decoding_error,
    local_decoding_error_statistics,
    session_state_fractions,
    state_fraction_statistics,
)
from continuum_swr_replay.analysis.event_definition.analyzed_time import (
    immobile_analyzed_bins,
)
from continuum_swr_replay.analysis.event_definition.nonlocal_event_rule import (
    nonlocal_event_bins_by_swr,
)
from continuum_swr_replay.analysis.event_properties.event_analysis import (
    get_event_durations_ms,
    get_event_start_end_times,
)
from continuum_swr_replay.analysis.event_properties.triggered_analysis import (
    compute_triggered_average_curves,
)
from continuum_swr_replay.parameters import DEFAULT_NON_LOCAL_THRESHOLD
from continuum_swr_replay.visualization.figure_utilities import (
    PANEL_SINGLE,
    get_figure_size,
    save_figure,
)
from continuum_swr_replay.visualization.palettes import COLORS, EVENT_KINDS


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
            COLORS["No SWR"]
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
            color=COLORS["No SWR"],
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
            color=COLORS["No SWR"],
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
                2.0, color="lightgray", linestyle="--", alpha=0.5, linewidth=0.75
            )
        elif signal_type == "Non-Local_prob":
            ax.set_ylabel("Non-Local\nProb.")
            ax.axhline(
                DEFAULT_NON_LOCAL_THRESHOLD,
                color="lightgray",
                linestyle="--",
                alpha=0.5,
                linewidth=0.75,
            )
            ax.set_ylim(0.0, 1.0)

        ax.axvline(0.0, color="lightgray", linestyle="--", alpha=0.5, linewidth=0.75)
        ax.axhline(0.0, color="lightgray", linestyle="--", alpha=0.5, linewidth=0.75)

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
    # but the bins in its 0.3-s reach do, and they are not analyzed either.
    is_immobile = immobile_analyzed_bins(data_over_time)

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
#: with the color and legend label each per-event panel gives them.
def _event_class_histograms(
    events: pd.DataFrame,
    column: str,
    medians: dict[str, float],
    bins: np.ndarray,
    ax: plt.Axes,
) -> None:
    """Each class's per-event ``column`` as a density histogram, with a dashed
    line at the class median it is given (gid ``median_<class>``)."""
    for (is_swr, (label, color)), word in zip(
        EVENT_KINDS.items(), ("non_swr", "swr"), strict=True
    ):
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
        color="gray",
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
        color=COLORS["No SWR"],
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
    ax_difference.axhline(0.0, color="gray", linewidth=0.75)
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
        color=COLORS["No SWR"],
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
    :func:`~continuum_swr_replay.analysis.decode_quality.quality_metrics.state_fraction_statistics`'s
    over :func:`~continuum_swr_replay.analysis.decode_quality.quality_metrics.session_state_fractions`:
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
