"""Static matplotlib plotting functions for neural decoding visualization.

This module provides matplotlib-based plotting functions for visualizing
neural data, decoding results, and behavioral variables. These functions
are designed to be composable and can be combined into multi-panel figures
for publications and presentations.

The module contains two types of functions:

1. **Component plots**: Individual plotting functions that can be combined
   (e.g., plot_speed, plot_spike_times, plot_theta_lfp)
2. **Composite plots**: Functions that create multi-panel figures
   (e.g., plot_detector, plot_peri_event_probability)

All plotting functions follow a consistent interface:
- Take data dictionaries or DataFrames as input
- Accept matplotlib Axes objects for flexible layout
- Use the repository's color scheme from COLORS
- Support rasterization for large datasets
- Include proper axis labels and formatting

Functions
---------
Component Plots
~~~~~~~~~~~~~~~
plot_theta_lfp
    Plot theta-band filtered LFP trace
plot_theta_power_zscore
    Plot z-scored theta power with optional thresholds
plot_theta_phase
    Plot instantaneous theta phase (-π to π)
plot_multiunit_firing
    Plot multiunit activity rate with high-synchrony events
plot_ripple_consensus_trace_zscore
    Plot z-scored ripple consensus trace with detected events
plot_spike_times
    Plot spike raster sorted by place field peak
plot_acausal_probabilities
    Plot decoder state probabilities over time
plot_conditional_non_local_posterior
    Plot decoded position posterior conditioned on non-local state
plot_speed
    Plot animal running speed
plot_head_direction
    Plot head direction angle
plot_graph_as_1D
    Plot track graph as 1D linearized representation

Composite Plots
~~~~~~~~~~~~~~~
plot_detector
    Create comprehensive 6-8 panel decoding figure
plot_peri_event_probability
    Create peri-event time histogram with multiple signals

See Also
--------
continuum_swr_replay.visualization.figurl : Interactive web visualizations
continuum_swr_replay.visualization.animation : Movie generation
matplotlib.pyplot : Core plotting library
seaborn : Statistical plotting enhancements

Notes
-----
**Design Principles:**

- All functions accept existing Axes objects for integration into complex layouts
- Functions modify axes in-place and do not create or return figures
- Time is always in seconds, position in cm, speed in cm/s
- Rasterization is used for dense data (spike rasters, posteriors)
- Color scheme follows the repository standard (COLORS dictionary)

**Common Parameters:**

- `data` : dict containing time-indexed DataFrames and arrays
- `time_slice` : slice object defining the time window to plot
- `ax` : matplotlib Axes object to plot on
- `plot_thresholds` : bool to show detection thresholds

Examples
--------
Create a simple decoding figure:

>>> import matplotlib.pyplot as plt
>>> from continuum_swr_replay import load_data
>>> from continuum_swr_replay.visualization.plots import (
...     plot_speed, plot_spike_times, plot_acausal_probabilities
... )
>>>
>>> data = load_data("chimi20200212_.nwb", "02_r1")
>>> # After running decoding...
>>>
>>> fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
>>> time_slice = slice(1234.0, 1240.0)
>>>
>>> plot_speed(data, time_slice, axes[0])
>>> plot_spike_times(data, time_slice, detector, env, results, axes[1])
>>> plot_acausal_probabilities(results, time_slice, axes[2])
>>>
>>> axes[-1].set_xlabel('Time (s)')
>>> plt.tight_layout()
>>> plt.savefig('decoding_example.pdf')

Create a comprehensive decoding figure:

>>> from continuum_swr_replay.visualization.plots import plot_detector
>>>
>>> plot_detector(
...     data, detector, results,
...     figsize=(11, 8),
...     plot_theta=True,
...     plot_multiunit_zscore=True,
...     time_slice=slice(1234.0, 1240.0)
... )
>>> plt.savefig('detector_output.pdf', dpi=300)

"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr

if TYPE_CHECKING:
    import networkx as nx
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

from continuum_swr_replay.parameters import SAMPLING_FREQUENCY
from continuum_swr_replay.visualization.palettes import COLORS


def plot_peri_event_probability(
    results: xr.Dataset,
    ripple_zscore: pd.Series,
    mua_zscore: pd.Series,
    start_times: NDArray[np.floating],
    end_times: NDArray[np.floating],
    relative_time_range: tuple[float, float] = (-0.4, 0.4),
    sort_by: str = "TIME",
    show_end_times: bool = False,
    sampling_frequency: int = SAMPLING_FREQUENCY,
) -> None:
    """Plot peri-event time histogram of non-local probability.

    Parameters
    ----------
    results : xr.Dataset
        Decoding results containing state probabilities
    ripple_zscore : pd.Series
        Ripple z-score time series
    mua_zscore : pd.Series
        Multiunit activity z-score time series
    start_times : array-like
        Event start times
    end_times : array-like
        Event end times
    relative_time_range : tuple, optional
        Time window relative to event, by default (-0.4, 0.4)
    sort_by : str, optional
        How to sort events ("TIME", "PROB", "MUA", "SWR"), by default "TIME"
    show_end_times : bool, optional
        Whether to show event end times, by default False
    sampling_frequency : int, optional
        Sampling frequency in Hz, by default 500

    """
    non_local_prob = results.acausal_state_probabilities.sel(
        states=["Non-Local Continuous", "Non-Local Fragmented"]
    ).sum("states")

    n_time_bins = int(np.ceil(np.diff(relative_time_range) * sampling_frequency))
    relative_time = relative_time_range[0] + np.arange(n_time_bins) / sampling_frequency

    event_probs: list[NDArray[np.floating]] = []
    event_swr_zscores: list[NDArray[np.floating]] = []
    event_mua_zscores: list[NDArray[np.floating]] = []

    mean_event_prob: list[float] = []
    max_swr_zscores: list[float] = []
    max_mua_zscores: list[float] = []

    for start_time, end_time in zip(start_times, end_times, strict=True):
        time_slice = slice(
            start_time + relative_time_range[0], start_time + relative_time_range[1]
        )

        prob_event_time_slice = slice(start_time, end_time)

        # Non-local probability
        event_prob = non_local_prob.sel(time=time_slice)
        event_prob["time"] = event_prob.time - start_time
        event_prob = event_prob.reindex(
            {"time": relative_time},
            tolerance=1 / (2 * sampling_frequency),
            method="nearest",
            fill_value=np.nan,
        )
        event_probs.append(event_prob.values)
        mean_event_prob.append(
            non_local_prob.sel(time=prob_event_time_slice).mean().values
        )

        # SWR z-score
        event_swr_zscore = ripple_zscore.loc[time_slice]
        event_swr_zscore = event_swr_zscore.set_index(
            event_swr_zscore.index - start_time
        ).reindex(
            relative_time, tolerance=1 / (2 * sampling_frequency), method="nearest"
        )
        event_swr_zscores.append(event_swr_zscore.to_numpy().squeeze())
        max_swr_zscores.append(np.nanmax(ripple_zscore.loc[prob_event_time_slice]))

        # MUA z-score
        event_mua_zscore = mua_zscore.loc[time_slice]
        event_mua_zscore = event_mua_zscore.set_index(
            event_mua_zscore.index - start_time
        ).reindex(
            relative_time, tolerance=1 / (2 * sampling_frequency), method="nearest"
        )
        event_mua_zscores.append(event_mua_zscore.to_numpy().squeeze())
        max_mua_zscores.append(np.nanmax(mua_zscore.loc[prob_event_time_slice]))

    event_probs_arr: NDArray[np.floating] = np.stack(event_probs, axis=1)
    event_swr_zscores_arr: NDArray[np.floating] = np.stack(event_swr_zscores, axis=1)
    event_mua_zscores_arr: NDArray[np.floating] = np.stack(event_mua_zscores, axis=1)

    if sort_by.upper() == "PROB":
        sort_ind = np.argsort(mean_event_prob)[::-1]
    elif sort_by.upper() == "MUA":
        sort_ind = np.argsort(max_mua_zscores)[::-1]
    elif sort_by.upper() == "SWR":
        sort_ind = np.argsort(max_swr_zscores)[::-1]
    elif sort_by.upper() == "TIME":
        sort_ind = np.argsort(start_times)

    event_probs_sorted: NDArray[np.floating] = event_probs_arr[:, sort_ind]
    event_swr_zscores_sorted: NDArray[np.floating] = event_swr_zscores_arr[:, sort_ind]
    event_mua_zscores_sorted: NDArray[np.floating] = event_mua_zscores_arr[:, sort_ind]

    duration = end_times - start_times
    duration = duration[sort_ind]

    fig, axes = plt.subplots(
        6,
        1,
        sharex=True,
        figsize=(10, 20),
        gridspec_kw={"height_ratios": [3, 3, 3, 1, 1, 1]},
        constrained_layout=True,
    )
    n_events = len(start_times)
    t, n = np.meshgrid(relative_time, np.arange(n_events))

    # Non-local probability
    h = axes[0].pcolormesh(
        t, n, event_probs_sorted.T, vmin=0.0, vmax=1.0, cmap="cividis"
    )
    axes[0].axvline(0, color=COLORS["Non-Local"], linestyle="--")
    axes[0].set_ylabel("Event number")
    axes[0].set_title("Non-Local Event Probability")
    plt.colorbar(h, ax=axes[0])

    # SWR z-score
    h = axes[1].pcolormesh(
        t, n, event_swr_zscores_sorted.T, vmin=-3, vmax=3, cmap="RdBu_r"
    )
    axes[1].axvline(0, color=COLORS["Non-Local"], linestyle="--")
    axes[1].set_ylabel("Event number")
    axes[1].set_title("SWR z-score")
    plt.colorbar(h, ax=axes[1])

    # MUA z-score
    h = axes[2].pcolormesh(
        t, n, event_mua_zscores_sorted.T, vmin=-3, vmax=3, cmap="RdBu_r"
    )
    axes[2].axvline(0, color=COLORS["Non-Local"], linestyle="--")
    axes[2].set_ylabel("Event number")
    axes[2].set_title("MUA z-score")
    plt.colorbar(h, ax=axes[2])

    if show_end_times:
        for ax in axes[:3]:
            ax.scatter(duration, np.arange(n_events), color=COLORS["Non-Local"], s=1)

    # Non-Local IQR
    prob_iqr = np.nanpercentile(event_probs_sorted, [25, 50, 75], axis=1)
    axes[3].axvline(0, color=COLORS["Non-Local"], linestyle="--")
    axes[3].plot(
        relative_time, prob_iqr[1], color=COLORS["Non-Local"], label="Non-Local"
    )
    axes[3].fill_between(
        relative_time,
        prob_iqr[0],
        prob_iqr[2],
        alpha=0.5,
        color=COLORS["Non-Local"],
    )
    axes[3].set_ylabel("Prob.")
    axes[3].legend()

    # SWR IQR
    swr_iqr = np.nanpercentile(event_swr_zscores_sorted, [25, 50, 75], axis=1)
    axes[4].axvline(0, color=COLORS["Non-Local"], linestyle="--")
    axes[4].plot(relative_time, swr_iqr[1], color=COLORS["SWR"], label="SWR")
    axes[4].fill_between(
        relative_time,
        swr_iqr[0],
        swr_iqr[2],
        alpha=0.5,
        color=COLORS["SWR"],
    )
    axes[4].set_ylabel("z-score")
    axes[4].set_ylim(-3, 3)
    axes[4].axhline(0, color="k", linestyle="--")
    axes[4].legend()

    # MUA IQR
    mua_iqr = np.nanpercentile(event_mua_zscores_sorted, [25, 50, 75], axis=1)
    axes[5].axvline(0, color=COLORS["Non-Local"], linestyle="--")
    axes[5].plot(relative_time, mua_iqr[1], color=COLORS["MUA"], label="MUA")
    axes[5].fill_between(
        relative_time,
        mua_iqr[0],
        mua_iqr[2],
        alpha=0.5,
        color=COLORS["MUA"],
    )
    axes[5].set_ylabel("z-score")
    axes[5].set_ylim(-3, 3)
    axes[5].axhline(0, color=COLORS["Non-Local"], linestyle="--")
    axes[5].legend()

    axes[-1].set_xlabel("Time relative to event (s)")


def plot_theta_lfp(data: dict[str, Any], time_slice: slice, ax: Axes) -> None:
    """Plot theta-band filtered LFP.

    Parameters
    ----------
    data : dict
        Data dictionary containing 'theta_lfp' key
    time_slice : slice
        Time slice to plot
    ax : matplotlib.axes.Axes
        Axes to plot on

    """
    theta_lfp = data["theta_lfp"].loc[time_slice]
    ax.plot(theta_lfp.index, theta_lfp, color="black")
    ax.set_ylabel("Theta\nTrace")


def plot_theta_power_zscore(
    data: dict[str, Any], time_slice: slice, ax: Axes, plot_thresholds: bool = False
) -> None:
    """Plot theta power z-score.

    Parameters
    ----------
    data : dict
        Data dictionary containing 'theta_power_zscore' key
    time_slice : slice
        Time slice to plot
    ax : matplotlib.axes.Axes
        Axes to plot on
    plot_thresholds : bool, optional
        Whether to plot threshold lines, by default False

    """
    theta_power_zscore = data["theta_power_zscore"].loc[time_slice]

    max_consensus = max(int(np.ceil(theta_power_zscore.max())), 5)
    min_consensus = min(int(np.ceil(theta_power_zscore.min())), -1)

    ax.fill_between(
        theta_power_zscore.index,
        theta_power_zscore.values.squeeze(),
        color="black",
    )
    ax.set_ylim((min_consensus, max_consensus))
    ax.set_yticks((min_consensus, max_consensus))
    ax.set_ylabel("Theta\nStd")

    if plot_thresholds:
        ax.axhline(2, color="grey", linestyle="--", linewidth=1, zorder=10)


def plot_theta_phase(data: dict[str, Any], time_slice: slice, ax: Axes) -> None:
    """Plot theta phase.

    Parameters
    ----------
    data : dict
        Data dictionary containing 'theta_phase' key
    time_slice : slice
        Time slice to plot
    ax : matplotlib.axes.Axes
        Axes to plot on

    """
    theta_phase = data["theta_phase"].loc[time_slice]

    ax.plot(
        theta_phase.index,
        theta_phase.values.squeeze(),
        color="black",
    )
    ax.set_ylim((-np.pi, np.pi))
    ax.set_yticks((-np.pi, 0, np.pi))
    ax.set_ylabel("Theta\nPhase")
    ax.set_yticklabels([r"$-\pi$", r"$0$", r"$\pi$"])


def intervals_to_timebins(
    intervals: pd.DataFrame,
    time_idx: NDArray[np.floating],
    *,
    start_col: str = "start_time",
    end_col: str = "end_time",
    value_col: str | None = None,
    fill_value: Any = np.nan,
) -> NDArray:
    """Project per-interval values onto a per-timebin axis.

    For every timepoint in ``time_idx``, find the interval (row of
    ``intervals``) that contains it and return the corresponding
    value. Timebins outside every interval return ``fill_value`` (or
    ``False`` when ``value_col`` is ``None``). Intervals are treated
    as closed: ``start <= t <= end``.

    Parameters
    ----------
    intervals : pd.DataFrame
        Interval table with ``start_col`` / ``end_col`` columns. When
        ``value_col`` is given it must be present too. Non-overlapping
        intervals are expected (events / stops / trials are all
        non-overlapping by construction); behavior with overlapping
        intervals is "closest-start wins".
    time_idx : np.ndarray, shape (n_time,)
        Time values to label. Does not need to be sorted.
    start_col, end_col : str
        Column names for interval bounds.
    value_col : str, optional
        Column to carry onto the timebin axis. ``None`` returns a
        boolean membership mask (the legacy ``_intervals_to_mask``
        behavior).
    fill_value : any
        Value for timebins outside every interval (only used when
        ``value_col`` is given).

    Returns
    -------
    np.ndarray, shape (n_time,)
        Boolean mask if ``value_col is None``; else a value array
        whose dtype matches ``intervals[value_col]`` (promoted as
        needed to hold ``fill_value``).

    Notes
    -----
    Uses ``searchsorted`` on a sorted copy of the intervals — O(N log
    N) up-front plus O(T log N) per query — rather than the naive
    broadcast ``any(axis=0)`` pattern that materializes an
    ``(N, T)`` boolean matrix. On a full session (T ≈ 1.8M, N ≈ 1000)
    the broadcast form would be ~2 GB; this stays linear.
    """
    if intervals.empty:
        if value_col is None:
            return np.zeros(len(time_idx), dtype=bool)
        return np.full(len(time_idx), fill_value)

    order = np.argsort(intervals[start_col].to_numpy())
    starts = intervals[start_col].to_numpy()[order]
    ends = intervals[end_col].to_numpy()[order]
    time_arr = np.asarray(time_idx)

    idx = np.searchsorted(starts, time_arr, side="right") - 1
    clipped = np.clip(idx, 0, starts.size - 1)
    in_interval = (idx >= 0) & (time_arr <= ends[clipped])

    if value_col is None:
        return in_interval

    values = intervals[value_col].to_numpy()[order]
    # ``np.where`` promotes the output dtype to accommodate
    # ``fill_value`` (e.g., int values + NaN fill → float64).
    return np.where(in_interval, values[clipped], fill_value)


def _intervals_to_mask(
    intervals: pd.DataFrame,
    time_idx: NDArray[np.floating],
) -> NDArray[np.bool_]:
    """Boolean shortcut for :func:`intervals_to_timebins`.

    Backward-compatible wrapper around the generalized helper. Kept
    so existing call sites (e.g., ``plot_spike_times``) don't change.
    """
    return intervals_to_timebins(intervals, time_idx, value_col=None).astype(bool)


def plot_multiunit_firing(
    data: dict[str, Any],
    time_slice: slice,
    ax: Axes,
    plot_multiunit_zscore: bool = False,
    plot_thresholds: bool = False,
    brain_area: str = "HPC",
) -> None:
    """Plot multiunit activity.

    Parameters
    ----------
    data : dict
        Data dictionary from load_data(). Uses keys ``multiunit_firing_rate``
        or ``multiunit_rate_zscore`` (dict[str, pd.DataFrame] keyed by brain
        area) and ``hse_times`` (dict[str, pd.DataFrame]).
    time_slice : slice
        Time slice to plot
    ax : matplotlib.axes.Axes
        Axes to plot on
    plot_multiunit_zscore : bool, optional
        Whether to plot z-scored MUA, by default False
    plot_thresholds : bool, optional
        Whether to plot threshold lines, by default False
    brain_area : str, optional
        Brain area to plot, by default "HPC"

    """
    key = "multiunit_rate_zscore" if plot_multiunit_zscore else "multiunit_firing_rate"
    multiunit_data = data[key]
    # Resolve dict-by-brain-area to a single DataFrame
    if isinstance(multiunit_data, dict):
        multiunit_firing = multiunit_data[brain_area]
    else:
        multiunit_firing = multiunit_data

    if plot_multiunit_zscore:
        ax.set_ylabel("MUA\n[Std].")
        if plot_thresholds:
            ax.axhline(2, color="grey", linestyle="--", linewidth=1, zorder=10)
    else:
        ax.set_ylabel("MUA\n[spikes/s]")

    max_multiunit = np.ceil(multiunit_firing.values.max()).astype(int)
    min_multiunit = (
        np.floor(multiunit_firing.values.min()).astype(int)
        if plot_multiunit_zscore
        else 0
    )

    ax.fill_between(
        multiunit_firing.loc[time_slice].index.to_numpy(),
        multiunit_firing.loc[time_slice].to_numpy().squeeze(),
        color="black",
    )

    # Compute HSE active periods from hse_times
    time_idx = multiunit_firing.loc[time_slice].index.to_numpy()
    hse_times = data.get("hse_times", {})
    if isinstance(hse_times, dict):
        hse_times = hse_times.get(brain_area, pd.DataFrame())
    hse_is_active = _intervals_to_mask(hse_times, time_idx)

    ax.fill_between(
        time_idx,
        np.ones_like(time_idx) * max_multiunit,
        where=hse_is_active,
        color=COLORS["MUA"],
        zorder=-2,
        alpha=0.6,
        step="pre",
        rasterized=True,
    )
    ax.set_ylim((min_multiunit, max_multiunit))
    ax.set_yticks((min_multiunit, max_multiunit))

    if plot_thresholds:
        ax.axhline(2, color="grey", linestyle="--", linewidth=1, zorder=10)


def plot_ripple_consensus_trace_zscore(
    data: dict[str, Any], time_slice: slice, ax: Axes, plot_thresholds: bool = False
) -> None:
    """Plot ripple consensus trace z-score.

    Parameters
    ----------
    data : dict
        Data dictionary containing ripple data
    time_slice : slice
        Time slice to plot
    ax : matplotlib.axes.Axes
        Axes to plot on
    plot_thresholds : bool, optional
        Whether to plot threshold lines, by default False

    """
    ripple_consensus_trace = data["zscored_ripple_consensus_trace"].loc[time_slice]

    max_consensus = max(int(np.ceil(ripple_consensus_trace.max())), 5)
    min_consensus = min(int(np.ceil(ripple_consensus_trace.min())), -1)

    # Compute ripple active periods from ripple_times
    time_idx = ripple_consensus_trace.index.to_numpy()
    ripple_times = data.get("ripple_times", pd.DataFrame())
    ripple_is_active = _intervals_to_mask(ripple_times, time_idx)

    ax.fill_between(
        ripple_consensus_trace.index,
        np.ones_like(ripple_consensus_trace.index) * max_consensus,
        where=ripple_is_active,
        color=COLORS["SWR"],
        zorder=-2,
        alpha=0.6,
        step="pre",
        rasterized=True,
    )
    ax.fill_between(
        ripple_consensus_trace.index,
        ripple_consensus_trace.values.squeeze(),
        color="black",
        rasterized=True,
    )
    ax.set_ylim((min_consensus, max_consensus))
    ax.set_yticks((min_consensus, max_consensus))
    ax.set_ylabel("SWR\n[Std]")

    if plot_thresholds:
        ax.axhline(2, color="grey", linestyle="--", linewidth=1, zorder=10)


def plot_spike_times(
    data: dict[str, Any],
    time_slice: slice,
    detector: Any,
    env: Any,
    results: xr.Dataset,
    ax: Axes,
    data_type: str = "clusterless",
    brain_area: str = "HPC",
) -> None:
    """Plot spike raster with non-local events highlighted.

    Parameters
    ----------
    data : dict
        Data dictionary from load_data(). Uses ``spike_times`` key which is
        a dict[str, list[np.ndarray]] keyed by brain area.
    time_slice : slice
        Time slice to plot
    detector : Detector
        Fitted detector object
    env : Environment
        Environment object with place field information
    results : xr.Dataset
        Decoding results
    ax : matplotlib.axes.Axes
        Axes to plot on
    data_type : str, optional
        Type of spike data ("clusterless" or "sorted"), by default "clusterless"
    brain_area : str, optional
        Brain area to plot spikes from, by default "HPC"

    """
    spike_times_dict = data.get("spike_times", {})
    if isinstance(spike_times_dict, dict):
        spike_times = spike_times_dict.get(brain_area, [])
    else:
        spike_times = spike_times_dict
    n_neurons = len(spike_times)
    acausal_state_probabilities = results.sel(
        time=time_slice
    ).acausal_state_probabilities
    non_local_probabilities = acausal_state_probabilities.sel(
        states="Non-Local Continuous"
    ) + acausal_state_probabilities.sel(states="Non-Local Fragmented")
    is_non_local = (non_local_probabilities > 0.8).values.squeeze()
    try:
        place_fields = detector.encoding_model_[("", 0)]["place_fields"]
        neuron_sort_ind = np.argsort(
            env.place_bin_centers_[np.nanargmax(place_fields, axis=1)].squeeze()
        )
        cell_label = "Neuron"
    except KeyError:
        neuron_sort_ind = np.arange(len(spike_times))
        cell_label = "Elec."
    new_spike_times = [
        spike_times[neuron_id][
            np.logical_and(
                spike_times[neuron_id] >= time_slice.start,
                spike_times[neuron_id] <= time_slice.stop,
            )
        ]
        for neuron_id in neuron_sort_ind
    ]
    ax.eventplot(
        new_spike_times,
        color="black",
        rasterized=True,
        linelengths=0.9,
        linewidths=1,
    )
    ax.set_yticks((1, n_neurons))
    ax.fill_between(
        non_local_probabilities.time,
        (np.ones_like(non_local_probabilities.values) * n_neurons) + 0.75,
        where=np.asarray(is_non_local).squeeze().tolist(),
        color=COLORS["Non-Local"],
        zorder=-1,
        alpha=0.6,
        step="pre",
    )
    ax.set_ylabel(cell_label)


def plot_acausal_probabilities(
    results: xr.Dataset,
    time_slice: slice,
    ax: Axes,
    plot_thresholds: bool = False,
    plot_local: bool = False,
) -> None:
    """Plot state probabilities over time.

    Parameters
    ----------
    results : xr.Dataset
        Decoding results containing state probabilities
    time_slice : slice
        Time slice to plot
    ax : matplotlib.axes.Axes
        Axes to plot on
    plot_thresholds : bool, optional
        Whether to plot threshold lines, by default False
    plot_local : bool, optional
        Whether to plot local and no-spike states, by default False

    """
    time = results.sel(time=time_slice).time
    non_local_prob = results.acausal_state_probabilities.sel(
        time=time_slice, states=["Non-Local Continuous", "Non-Local Fragmented"]
    ).sum("states")

    if plot_local:
        no_spike_prob = results.acausal_state_probabilities.sel(
            time=time_slice, states="No-Spike"
        )
        local_prob = results.acausal_state_probabilities.sel(
            time=time_slice, states="Local"
        )

        ax.plot(
            time,
            no_spike_prob,
            color=COLORS["No-Spike"],
            label="No-Spike",
            linewidth=1,
            clip_on=False,
        )
        ax.plot(
            time,
            local_prob,
            color=COLORS["Local"],
            label="Local",
            linewidth=1,
            clip_on=False,
        )
    ax.plot(
        time,
        non_local_prob,
        color=COLORS["Non-Local"],
        label="Non-Local",
        linewidth=1,
        clip_on=False,
    )
    if plot_local:
        ax.set_ylabel("State\nProb.")
    else:
        ax.set_ylabel("Non-\nLocal\nProb.")
    ax.set_xlabel("")
    ax.set_yticks((0, 1))
    ax.set_yticklabels(["0", "1"])
    ax.set_ylim((0, 1))

    if plot_thresholds:
        ax.axhline(0.8, color="grey", linestyle="--", linewidth=1, zorder=10)


def plot_conditional_non_local_posterior(
    data: dict[str, Any],
    time_slice: slice,
    results: xr.Dataset,
    env: Any,
    detector: Any,
    ax: Axes,
    posterior_max: float = 0.25,
) -> None:
    """Plot conditional non-local posterior with animal position.

    Parameters
    ----------
    data : dict
        Data dictionary containing position information
    time_slice : slice
        Time slice to plot
    results : xr.Dataset
        Decoding results
    env : Environment
        Environment object
    detector : Detector
        Fitted detector object
    ax : matplotlib.axes.Axes
        Axes to plot on
    posterior_max : float, optional
        Maximum value for colormap, by default 0.25

    """
    cmap = copy.copy(plt.get_cmap("bone_r"))
    cmap.set_bad(color="lightgrey", alpha=1.0)

    state_ind = detector.state_ind_
    acausal_posterior = results.sel(time=time_slice).acausal_posterior.values

    t, x = np.meshgrid(results.sel(time=time_slice).time.values, env.place_bin_centers_)
    non_local_inds = np.nonzero(
        ["Non-Local" in state for state in detector.state_names]
    )[0]
    conditional_non_local_acausal_posterior = np.zeros(
        (len(results.sel(time=time_slice).time.values), len(env.place_bin_centers_))
    )
    for non_local_ind in non_local_inds:
        conditional_non_local_acausal_posterior += acausal_posterior[
            :, state_ind == non_local_ind
        ]
    conditional_non_local_acausal_posterior /= np.nansum(
        conditional_non_local_acausal_posterior, axis=1
    )[:, np.newaxis]
    conditional_non_local_acausal_posterior[:, ~env.is_track_interior_] = np.nan

    ax.pcolormesh(
        t,
        x,
        conditional_non_local_acausal_posterior.T,
        vmin=0.0,
        vmax=posterior_max,
        cmap=cmap,
        rasterized=True,
    )
    position_info = data["position_info"].loc[time_slice]
    ax.scatter(
        position_info.index,
        position_info.linear_position,
        s=1,
        color="magenta",
        clip_on=False,
        rasterized=True,
        zorder=2,
    )
    ax.set_xlabel("")
    max_position = np.ceil(data["position_info"].linear_position.max()).astype(int)
    ax.set_yticks((0.0, max_position))
    ax.set_ylim((0.0, max_position))
    ax.set_ylabel("Position [cm]")


def plot_linear_track_graph_strip(
    track_graph: nx.Graph,
    edge_order: list,
    edge_spacing: Any,
    ax: Axes,
    max_position: float | None = None,
    node_size: int = 14,
) -> None:
    """Draw the linearized track graph as a vertical key beside a posterior.

    Wraps :func:`track_linearization.plot_graph_as_1D` with ``axis="y"``,
    using the same ``edge_order`` / ``edge_spacing`` that produced the
    decoder's linear position axis so the arm segments line up exactly with
    the conditional-posterior y-axis (linear position in cm). Acts as a
    spatial key: each line segment is one track arm, gaps are the junction
    spacing, and node markers sit at well/junction boundaries.

    Parameters
    ----------
    track_graph : nx.Graph
        Track graph (``data["track_graph"]``).
    edge_order : list of 2-tuples
        Linearization edge order (``data["linear_edge_order"]``).
    edge_spacing : float or sequence of float
        Linearization edge spacing (``data["linear_edge_spacing"]``).
    ax : matplotlib.axes.Axes
        Axes to draw the vertical strip on.
    max_position : float, optional
        If given, the y-limit (cm) — match the adjacent posterior axis.
    node_size : int, default 14
        Marker size for reward-well node boundaries.

    """
    # ``plot_graph_as_1D`` (this module) already lays the linearized track
    # along y at ``other_axis_start`` with per-arm tab10 colors, using the
    # same cumulative coordinate as the decoder — so it aligns with the
    # posterior's linear-position axis.
    reward_well_nodes = [n for n in track_graph.nodes if track_graph.degree(n) == 1]
    plot_graph_as_1D(
        track_graph,
        ax=ax,
        edge_order=edge_order,
        edge_spacing=edge_spacing,
        reward_well_nodes=reward_well_nodes,
        other_axis_start=0.0,
        reward_well_size=node_size,
    )
    # Put the colored track line (drawn at x=0) at the right edge of the
    # axes so it sits flush against the posterior it keys.
    ax.set_xlim(-1.6, 0.0)
    if max_position is not None:
        ax.set_ylim(0.0, max_position)
        ax.set_yticks((0.0, max_position))
    ax.set_xticks([])
    for spine in ("top", "right", "bottom"):
        ax.spines[spine].set_visible(False)
    ax.set_ylabel("Position [cm]")


def _track_edge_colors() -> NDArray[np.floating]:
    """tab10 per-edge colors, matching :func:`plot_graph_as_1D`'s default.

    Both the 1D strip and the 2D key index this by position in
    ``edge_order`` so an arm has the same color in both views.
    """
    return np.array(matplotlib.colormaps["tab10"].colors)


def plot_track_graph_2d(
    track_graph: nx.Graph,
    position_info: pd.DataFrame,
    ax: Axes,
    *,
    edge_order: list | None = None,
    occupancy_color: str = "lightgrey",
    mark_wells: bool = True,
    reward_well_size: int = 22,
) -> None:
    """Draw the 2D track graph over the animal's 2D occupancy.

    Spatial key that connects the linearized posterior (panels in
    Figure 2a) back to the physical track. Plots the faint 2D head-position
    trajectory and the track-graph edges, coloring each arm with the same
    per-edge tab10 color that :func:`plot_linear_track_graph_strip` uses, so
    the 2D and 1D views share an arm color map. Degree-1 well nodes are
    optionally marked in their arm's color.

    Parameters
    ----------
    track_graph : nx.Graph
        Track graph with ``pos`` node attributes (``data["track_graph"]``).
    position_info : pd.DataFrame
        Position table with ``head_position_x`` / ``head_position_y``
        columns (``data["position_info"]``).
    ax : matplotlib.axes.Axes
        Axes to draw on.
    edge_order : list of 2-tuples, optional
        Linearization edge order (``data["linear_edge_order"]``). Edge
        colors are assigned by position in this list to match the 1D
        strip. Defaults to ``list(track_graph.edges)``.
    occupancy_color : str, default "lightgrey"
        Color of the faint 2D occupancy trajectory.
    mark_wells : bool, default True
        If True, mark degree-1 (well) nodes with filled circles in their
        arm's color.
    reward_well_size : int, default 22
        Marker size for well nodes.

    """
    ax.plot(
        position_info["head_position_x"].to_numpy(),
        position_info["head_position_y"].to_numpy(),
        color=occupancy_color,
        linewidth=0.3,
        alpha=0.6,
        rasterized=True,
        zorder=1,
    )
    if edge_order is None:
        edge_order = list(track_graph.edges)
    edge_colors = _track_edge_colors()
    reward_well_nodes = [n for n in track_graph.nodes if track_graph.degree(n) == 1]
    node_pos = {n: track_graph.nodes[n]["pos"] for n in track_graph.nodes}
    for edge_ind, (n1, n2) in enumerate(edge_order):
        xy = np.stack((node_pos[n1], node_pos[n2]))
        ax.plot(
            xy[:, 0], xy[:, 1], color=edge_colors[edge_ind], linewidth=1.5, zorder=2
        )
        if mark_wells:
            for node in (n1, n2):
                if node in reward_well_nodes:
                    pos = node_pos[node]
                    ax.scatter(
                        pos[0],
                        pos[1],
                        s=reward_well_size,
                        color=edge_colors[edge_ind],
                        edgecolor="white",
                        linewidth=0.5,
                        zorder=3,
                    )
    ax.set_aspect("equal")
    ax.axis("off")


def plot_speed(data: dict[str, Any], time_slice: slice, ax: Axes) -> None:
    """Plot animal speed.

    Parameters
    ----------
    data : dict
        Data dictionary containing position information
    time_slice : slice
        Time slice to plot
    ax : matplotlib.axes.Axes
        Axes to plot on

    """
    position_info = data["position_info"].loc[time_slice]
    # ``data_loaders.position`` canonically produces ``head_speed``;
    # older datasets used ``nose_vel``. Prefer the canonical column
    # and fall back so this panel works on legacy data without a
    # rerun.
    if "head_speed" in position_info.columns:
        speed = np.asarray(position_info.head_speed).squeeze()
    elif "nose_vel" in position_info.columns:
        speed = np.asarray(position_info.nose_vel).squeeze()
    else:
        raise KeyError(
            "position_info has neither 'head_speed' nor 'nose_vel'; "
            f"available columns: {list(position_info.columns)}"
        )
    max_speed = np.ceil(speed.max()).astype(int)
    max_speed = max(5, max_speed)
    ax.fill_between(position_info.index, speed, color="lightgrey")
    ax.set_ylabel("Speed\n[cm/s]")
    ax.set_yticks((0.0, max_speed))
    ax.set_ylim((0.0, max_speed))


def plot_head_direction(data: dict[str, Any], time_slice: slice, ax: Axes) -> None:
    """Plot head direction.

    Parameters
    ----------
    data : dict
        Data dictionary containing position information
    time_slice : slice
        Time slice to plot
    ax : matplotlib.axes.Axes
        Axes to plot on

    """
    position_info = data["position_info"].loc[time_slice]
    direction = np.asarray(position_info.body_dir).squeeze()
    min_dir, max_dir = -np.pi, np.pi
    ax.scatter(position_info.index, direction, color="black", s=1, clip_on=False)
    ax.set_ylim(min_dir, max_dir)
    ax.set_yticks([min_dir, 0, max_dir])
    ax.set_yticklabels([r"$-\pi$", r"$0$", r"$\pi$"])


def plot_detector(
    data: dict[str, Any],
    detector: Any,
    results: xr.Dataset,
    figsize: tuple[float, float] = (11, 6.0),
    plot_multiunit_zscore: bool = True,
    plot_thresholds: bool = False,
    plot_direction: bool = False,
    plot_theta: bool = False,
    time_slice: slice | None = None,
    data_type: str = "clusterless",
    posterior_max: float = 0.25,
) -> None:
    """Create comprehensive multi-panel decoding figure.

    Parameters
    ----------
    data : dict
        Data dictionary
    detector : Detector
        Fitted detector object
    results : xr.Dataset
        Decoding results
    figsize : tuple, optional
        Figure size, by default (11, 6.0)
    plot_multiunit_zscore : bool, optional
        Whether to plot z-scored MUA, by default True
    plot_thresholds : bool, optional
        Whether to plot threshold lines, by default False
    plot_direction : bool, optional
        Whether to plot head direction, by default False
    plot_theta : bool, optional
        Whether to plot theta oscillation panels, by default False
    time_slice : slice, optional
        Time slice to plot, by default None (plot all)
    data_type : str, optional
        Type of spike data, by default "clusterless"
    posterior_max : float, optional
        Maximum value for posterior colormap, by default 0.25

    """
    if time_slice is None:
        time_slice = slice(results.time.values[0], results.time.values[-1])

    env = detector.environments[0]

    n_subplots = 8 if plot_theta else 6
    height_ratios = np.ones(n_subplots)
    height_ratios[-2] = 3
    height_ratios[-4] = 3
    subplot_offset = 2 if plot_theta else 0

    _, axes = plt.subplots(
        n_subplots,
        1,
        figsize=figsize,
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": height_ratios},
        dpi=100,
    )

    if plot_theta:
        plot_theta_lfp(data, time_slice, axes[0])
        plot_theta_power_zscore(data, time_slice, axes[1], plot_thresholds)
        plot_theta_phase(data, time_slice, axes[2])

    else:
        plot_ripple_consensus_trace_zscore(data, time_slice, axes[0], plot_thresholds)

    plot_multiunit_firing(
        data,
        time_slice,
        axes[subplot_offset + 1],
        plot_multiunit_zscore,
        plot_thresholds,
    )

    plot_spike_times(
        data,
        time_slice,
        detector,
        env,
        results,
        axes[subplot_offset + 2],
        data_type,
    )

    plot_acausal_probabilities(
        results, time_slice, axes[subplot_offset + 3], plot_thresholds=False
    )

    plot_conditional_non_local_posterior(
        data,
        time_slice,
        results,
        env,
        detector,
        axes[subplot_offset + 4],
        posterior_max,
    )

    plot_speed(data, time_slice, axes[subplot_offset + 5])

    if plot_direction:
        plot_head_direction(data, time_slice, axes[subplot_offset + 5].twinx())

    axes[-1].set_xticks((time_slice.start, time_slice.stop))
    axes[-1].set_xticklabels((time_slice.start, time_slice.stop))
    axes[-1].set_xlabel("Time [s]")
    axes[-1].xaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
    sns.despine(offset=5)


def plot_graph_as_1D(
    track_graph: nx.Graph,
    ax: Axes | None = None,
    edge_order: NDArray[np.int_] | None = None,
    edge_spacing: float | list[float] = 0.0,
    reward_well_nodes: list[int] | None = None,
    other_axis_start: float = 0,
    edge_colors: NDArray[np.floating] | None = None,
    reward_well_size: int = 10,
    edge_linewidth: int = 2,
) -> None:
    """Plot track graph as 1D linearized representation.

    Parameters
    ----------
    track_graph : networkx.Graph
        Track graph with distance attributes on edges
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, by default None (use current axes)
    edge_order : array-like, optional
        Order of edges to plot, by default None
    edge_spacing : float or list, optional
        Spacing between edges, by default 0.0
    reward_well_nodes : list, optional
        Nodes representing reward wells, by default None
    other_axis_start : float, optional
        Starting position on other axis, by default 0
    edge_colors : array-like, optional
        Colors for each edge, by default None
    reward_well_size : int, optional
        Size of reward well markers, by default 10
    edge_linewidth : int, optional
        Line width for edges, by default 2

    """
    if ax is None:
        ax = plt.gca()
    # If no edge_order is given, then arange edges in the order passed to
    # construct the track graph
    if edge_order is None:
        edge_order = np.asarray(track_graph.edges)
    if reward_well_nodes is None:
        reward_well_nodes = []
    if edge_colors is None:
        # Get tab10 colormap colors - use getattr since .colors not in type stubs
        cmap = matplotlib.colormaps.get_cmap("tab10")
        colors = getattr(cmap, "colors", None)
        if colors is not None:
            edge_colors = np.array(colors)
        else:
            # Fallback: sample colormap at normalized indices
            edge_colors = np.array([cmap(i / 10) for i in range(10)])

    n_edges = len(edge_order)
    edge_spacing_list: list[float]
    if isinstance(edge_spacing, (int, float)):
        edge_spacing_list = [float(edge_spacing)] * (n_edges - 1)
    else:
        edge_spacing_list = edge_spacing

    start_node_linear_position = 0.0

    for edge_ind, edge in enumerate(edge_order):
        end_node_linear_position = (
            start_node_linear_position + track_graph.edges[edge]["distance"]
        )
        ax.plot(
            (other_axis_start, other_axis_start),
            (start_node_linear_position, end_node_linear_position),
            color=edge_colors[edge_ind],
            clip_on=False,
            zorder=7,
            linewidth=edge_linewidth,
        )
        if edge[0] in reward_well_nodes:
            ax.scatter(
                other_axis_start,
                start_node_linear_position,
                color=edge_colors[edge_ind],
                s=reward_well_size,
                zorder=10,
                clip_on=False,
            )
        if edge[1] in reward_well_nodes:
            ax.scatter(
                other_axis_start,
                end_node_linear_position,
                color=edge_colors[edge_ind],
                s=reward_well_size,
                zorder=10,
                clip_on=False,
            )

        try:
            start_node_linear_position += (
                track_graph.edges[edge]["distance"] + edge_spacing_list[edge_ind]
            )
        except IndexError:
            pass
