"""Plotting utilities for Generalized Linear Model (GLM) analysis results.

This module provides visualization functions for GLM-based analysis of neural
responses to events (e.g., sharp-wave ripples, replay events). Functions support
event-aligned spike rasters, peri-stimulus time histograms (PSTH), and model
comparison via AIC.

Main Functions
--------------
plot_raster_relative_to_event
    Create event-aligned spike raster plot showing spike times relative to event onsets.
plot_hist_relative_to_event
    Create peri-stimulus time histogram (PSTH) showing firing rate relative to events.
plot_AICs
    Compare GLM models using Akaike Information Criterion (AIC) values.

Examples
--------
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> from continuum_swr_replay.visualization.glm_plots import plot_raster_relative_to_event
>>>
>>> # Create example data
>>> spike_times = np.array([0.1, 0.5, 0.8, 1.2, 1.5, 2.1])
>>> event_times = np.array([0.5, 1.0, 2.0])
>>>
>>> # Plot raster
>>> fig, ax = plt.subplots()
>>> plot_raster_relative_to_event(
...     spike_times=spike_times,
...     event_times=event_times,
...     window=(-0.5, 0.5),
...     ax=ax,
... )
>>> plt.show()

See Also
--------
continuum_swr_replay.analysis.pfc : PFC analysis and GLM fitting
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from non_local_detector.likelihoods.common import get_spikecount_per_time_bin

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray


def plot_raster_relative_to_event(
    spike_times: NDArray[np.floating],
    event_times: NDArray[np.floating],
    window: tuple[float, float],
    ax: Axes | None = None,
    color: str = "black",
) -> Axes:
    """Plot spike raster aligned to event times.

    Creates an event-aligned raster plot where each row represents one event
    and shows spike times relative to that event's onset. Spikes are plotted
    as vertical lines centered at time 0 (event onset).

    Parameters
    ----------
    spike_times : NDArray[np.floating]
        Array of spike times in seconds for a single neuron.
    event_times : NDArray[np.floating]
        Array of event times in seconds to align spikes to.
    window : tuple[float, float]
        Time window (start, end) in seconds relative to event onset.
        Example: (-0.5, 0.5) shows spikes from 500ms before to 500ms after event.
    ax : Axes | None, optional
        Matplotlib axes to plot on. If None, uses current axes (plt.gca()).
    color : str, default="black"
        Color for spike markers.

    Returns
    -------
    Axes
        The axes object with the raster plot.

    Notes
    -----
    - Time 0 corresponds to event onset (marked with vertical dashed line)
    - Each row in the raster represents one event occurrence
    - Only spikes within the specified window are shown for each event
    - Empty spike trains or event lists are handled gracefully

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> spike_times = np.array([0.1, 0.5, 0.8, 1.2, 1.5])
    >>> event_times = np.array([0.5, 1.0])
    >>> fig, ax = plt.subplots()
    >>> plot_raster_relative_to_event(
    ...     spike_times, event_times, window=(-0.3, 0.3), ax=ax
    ... )
    >>> plt.xlabel("Time from event (s)")
    >>> plt.ylabel("Event #")
    >>> plt.show()
    """
    if ax is None:
        ax = plt.gca()

    # Validate window parameter
    if window[0] >= window[1]:
        raise ValueError(f"window[0] must be < window[1], got window={window}")

    event_times = np.asarray(event_times)
    relative_spike_times = []

    for event_time in event_times:
        # Compute spike times relative to this event
        event_relative_spike_times = spike_times - event_time

        # Filter to window (inclusive boundaries)
        in_window = np.logical_and(
            event_relative_spike_times >= window[0],
            event_relative_spike_times <= window[1],
        )
        relative_spike_times.append(event_relative_spike_times[in_window])

    # Plot raster
    ax.eventplot(relative_spike_times, color=color)
    ax.set_xlim(window)
    ax.axvline(0, color="black", linestyle="--")

    return ax


def plot_hist_relative_to_event(
    spike_times: NDArray[np.floating],
    event_times: NDArray[np.floating],
    window: tuple[float, float],
    ax: Axes | None = None,
    color: str = "grey",
    bin_size: float = 0.050,
) -> Axes:
    """Plot peri-stimulus time histogram (PSTH) aligned to event times.

    Creates an event-aligned histogram showing average firing rate across all
    events. The firing rate is computed by binning spikes relative to event
    onset and normalizing by the number of events and bin size.

    Parameters
    ----------
    spike_times : NDArray[np.floating]
        Array of spike times in seconds for a single neuron.
    event_times : NDArray[np.floating]
        Array of event times in seconds to align spikes to.
    window : tuple[float, float]
        Time window (start, end) in seconds relative to event onset.
        Example: (-1.0, 1.0) shows activity from 1s before to 1s after event.
    ax : Axes | None, optional
        Matplotlib axes to plot on. If None, uses current axes (plt.gca()).
    color : str, default="grey"
        Color for histogram bars.
    bin_size : float, default=0.050
        Bin width in seconds. Default is 50ms bins.

    Returns
    -------
    Axes
        The axes object with the histogram plot.

    Notes
    -----
    - Y-axis shows firing rate in spikes per second (sp/s)
    - Firing rate is averaged across all events
    - Time 0 corresponds to event onset (marked with vertical dashed line)
    - Normalization: spike count / (n_events * bin_size)

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> spike_times = np.array([0.1, 0.2, 0.5, 0.8, 1.2, 1.5])
    >>> event_times = np.array([0.5, 1.0])
    >>> fig, ax = plt.subplots()
    >>> plot_hist_relative_to_event(
    ...     spike_times, event_times, window=(-0.5, 0.5), ax=ax, bin_size=0.1
    ... )
    >>> plt.xlabel("Time from event (s)")
    >>> plt.ylabel("Firing rate (sp/s)")
    >>> plt.show()
    """
    if ax is None:
        ax = plt.gca()

    # Validate window parameter
    if window[0] >= window[1]:
        raise ValueError(f"window[0] must be < window[1], got window={window}")

    # Validate bin_size
    if bin_size <= 0:
        raise ValueError(f"bin_size must be positive, got bin_size={bin_size}")

    # Create bins
    bins = np.arange(window[0], window[1] + bin_size, bin_size)
    event_times = np.asarray(event_times)

    # Accumulate spike counts across events
    relative_spike_counts = np.zeros_like(bins)
    for event_time in event_times:
        # Compute spike times relative to this event
        event_relative_spike_times = spike_times - event_time

        # Filter to window (inclusive boundaries)
        in_window = np.logical_and(
            event_relative_spike_times >= window[0],
            event_relative_spike_times <= window[1],
        )

        # Bin spikes
        relative_spike_counts += get_spikecount_per_time_bin(
            event_relative_spike_times[in_window],
            bins,
        )

    # Normalize by number of events and bin size to get firing rate
    n_events = len(event_times)
    if n_events == 0:
        # No events - return zeros with correct shape
        firing_rate = np.zeros_like(bins)
    else:
        firing_rate = relative_spike_counts / (n_events * bin_size)

    # Plot histogram
    ax.bar(
        bins,
        firing_rate,
        width=bin_size,
        color=color,
        align="edge",
    )
    ax.set_xlim(window)
    ax.axvline(0, color="black", linestyle="--")

    return ax


def plot_AICs(
    glm_results: dict,
    model_list: list[str] | None = None,
    figsize: tuple[float, float] = (4, 3),
) -> tuple[plt.Figure, Axes]:
    """Compare GLM models using Akaike Information Criterion (AIC).

    Creates a scatter plot showing AIC values for different GLM models. Lower
    AIC values indicate better model fit penalized for complexity. Points are
    plotted vertically with model names as y-axis labels.

    Parameters
    ----------
    glm_results : dict
        Dictionary mapping model names to fitted GLM result objects.
        Each result must have an `aic` attribute.
    model_list : list[str] | None, optional
        List of model names to include in the plot (subset of glm_results keys).
        If None, uses default list:
        ["constant", "SWR", "non_local_SWR", "non_local_no_SWR",
         "non_local", "SWR + non_local_no_SWR"]
    figsize : tuple[float, float], default=(4, 3)
        Figure size in inches (width, height).

    Returns
    -------
    fig : Figure
        Matplotlib Figure object.
    ax : Axes
        Matplotlib Axes object with the AIC comparison plot.

    Notes
    -----
    - Lower AIC values indicate better models (better fit vs. complexity tradeoff)
    - Model names are automatically formatted for readability:
        - "non_local_SWR" → "Non-Local,\\nSWR"
        - "constant" → "Constant"
    - X-axis ticks show only min and max AIC values for clarity

    Examples
    --------
    >>> from unittest.mock import MagicMock
    >>> # Create mock GLM results
    >>> glm_results = {}
    >>> for name, aic in [("constant", 1000), ("SWR", 950), ("non_local", 920)]:
    ...     result = MagicMock()
    ...     result.aic = aic
    ...     glm_results[name] = result
    >>> fig, ax = plot_AICs(glm_results, model_list=["constant", "SWR", "non_local"])
    >>> plt.show()

    See Also
    --------
    continuum_swr_replay.analysis.pfc._fit_immobility_models : GLM fitting
    """
    if model_list is None:
        model_list = [
            "constant",
            "SWR",
            "non_local_SWR",
            "non_local_no_SWR",
            "non_local",
            "SWR + non_local_no_SWR",
        ]

    fig, ax = plt.subplots(figsize=figsize)

    # Validate all model names exist in glm_results
    missing_models = [m for m in model_list if m not in glm_results]
    if missing_models:
        raise ValueError(
            f"Model(s) not found in glm_results: {missing_models}. "
            f"Available models: {list(glm_results.keys())}"
        )

    # Model name formatting mapping
    model_list_to_mode_names = {
        "constant": "Constant",
        "SWR": "SWR",
        "non_local_SWR": "Non-Local,\nSWR",
        "non_local_no_SWR": "Non-Local,\nNo SWR",
        "non_local": "Non-Local",
        "SWR + non_local_no_SWR": "SWR + Non-Local",
    }

    # Validate known model names
    unknown_models = [m for m in model_list if m not in model_list_to_mode_names]
    if unknown_models:
        raise ValueError(
            f"Unknown model name(s): {unknown_models}. "
            f"Valid names: {list(model_list_to_mode_names.keys())}"
        )

    model_names = [model_list_to_mode_names[model_name] for model_name in model_list]

    # Extract AICs
    aics = [glm_results[model_name].aic for model_name in model_list]

    # Plot AICs as scatter points
    ax.scatter(
        aics,
        np.arange(len(aics)),
    )

    # Format y-axis with model names
    ax.set_yticks(np.arange(len(aics)))
    ax.set_yticklabels(model_names)

    # X-axis shows AIC range
    ax.set_xticks([np.min(aics), np.max(aics)])
    ax.set_xlabel("AIC (lower is better)")

    return fig, ax
