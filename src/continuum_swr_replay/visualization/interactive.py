"""Interactive visualization components using Panel and Holoviews.

This module provides interactive components for exploring neural decoding
results using Panel widgets and Holoviews dynamic visualizations. These
components are designed for use in Jupyter notebooks and support real-time
playback and temporal navigation of decoding results.

The module implements a widget-based approach where multiple synchronized
visualizations can be controlled by shared player widgets. This allows
for coordinated exploration of:
- Decoded posterior probabilities over 2D space
- Time series of neural activity (MUA, theta, ripples)
- Behavioral variables (speed, head direction)
- Decoding quality metrics (HPD, ahead-behind distance)

All visualizations use HoloViews DynamicMaps for efficient rendering and
Panel widgets for interactivity.

Functions
---------
create_player
    Create player widget for time-based playback
create_playback_speed_dropdown
    Create dropdown widget to control playback speed
create_scrolling_line_plot
    Create scrolling time series plot centered on current time
create_visualization
    Create complete interactive visualization layout with 2D decoding

See Also
--------
continuum_swr_replay.visualization.figurl : Web-based sharing
continuum_swr_replay.visualization.plots : Static matplotlib plots
holoviews : Declarative visualization library
panel : Widget and dashboard library

Notes
-----
**Requirements:**

To use this module in Jupyter notebooks:

>>> import holoviews as hv
>>> import panel as pn
>>> hv.extension('bokeh')  # or 'matplotlib'
>>> pn.extension()

**Widget Synchronization:**

The player widget returned by visualization functions can be used to
synchronize multiple visualizations:

>>> decode_plot, player = create_2D_decoding_plot(...)
>>> speed_plot = create_scrolling_line_plot(..., player=player)
>>> layout = pn.Column(decode_plot, speed_plot)

**Performance:**

- DynamicMaps only render the current frame, enabling smooth playback
- Rasterization is applied automatically for large datasets
- Playback speed can be adjusted from 1/8x to 8x real-time

Examples
--------
Create an interactive 2D decoding visualization:

>>> import holoviews as hv
>>> import panel as pn
>>> from continuum_swr_replay.visualization.interactive import create_visualization
>>>
>>> # Enable extensions
>>> hv.extension('bokeh')
>>> pn.extension()
>>>
>>> # Load data and run decoding
>>> data = load_data("chimi20200212_.nwb", "02_r1")
>>> # ... decode to get posterior, map_estimate, etc.
>>>
>>> # Create interactive visualization
>>> layout = create_visualization(
...     position_info=data['position_info'],
...     posterior=posterior_2d,
...     multiunit_firing_rate=data['multiunit_firing_rate']['HPC'],
...     non_local_prob=non_local_probability,
...     sampling_frequency=250,
...     width=1200,
...     height=800
... )
>>>
>>> # Display in notebook
>>> layout

Create a custom synchronized layout:

>>> from continuum_swr_replay.visualization.figurl import create_2D_decoding_plot
>>> from continuum_swr_replay.visualization.interactive import (
...     create_scrolling_line_plot, create_playback_speed_dropdown
... )
>>>
>>> # Create base plot with player
>>> decode_plot, player = create_2D_decoding_plot(
...     position_info, posterior, sampling_frequency=250
... )
>>>
>>> # Add synchronized time series
>>> speed_plot = create_scrolling_line_plot(
...     data['position_info']['head_speed'],
...     ylabel='Speed (cm/s)',
...     ylim=(0, 100),
...     player=player,
...     sampling_frequency=250
... )
>>>
>>> # Add speed control
>>> speed_dropdown = create_playback_speed_dropdown(player)
>>>
>>> # Combine into layout
>>> pn.Column(
...     pn.Row(decode_plot, speed_dropdown),
...     speed_plot
... )

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import xarray as xr

if TYPE_CHECKING:
    from numpy.typing import NDArray


def create_scrolling_line_plot(
    data: pd.Series | NDArray[np.floating],
    ylabel: str,
    ylim: tuple[float, float],
    player: pn.widgets.Player,
    window_lim: tuple[float, float] = (-0.4, 0.4),
    height_factor: int = 3,
    sampling_frequency: int = 250,
    width: int = 800,
    height: int = 600,
) -> hv.DynamicMap:
    """Create scrolling line plot centered on current time.

    Parameters
    ----------
    data : array-like
        Time series data to plot
    ylabel : str
        Y-axis label
    ylim : tuple
        Y-axis limits
    player : pn.widgets.Player
        Player widget to control time
    window_lim : tuple, optional
        Time window relative to current time, by default (-0.4, 0.4)
    height_factor : int, optional
        Factor to scale height, by default 3
    sampling_frequency : int, optional
        Sampling frequency in Hz, by default 250
    width : int, optional
        Plot width, by default 800
    height : int, optional
        Plot height, by default 600

    Returns
    -------
    hv.DynamicMap
        Holoviews dynamic map with scrolling plot

    """
    n_time = len(data)
    n_samples = int((window_lim[1] - window_lim[0]) * sampling_frequency)
    half_samples = n_samples // 2

    def plot_func(n: int) -> hv.Curve:
        start_idx = max(0, n - half_samples)
        end_idx = min(n_time, n + half_samples)
        window = np.asarray(data, dtype=float)[start_idx:end_idx].squeeze()
        line = hv.Curve(
            (np.arange(start_idx - n, end_idx - n) / sampling_frequency, window),
            kdims="time relative to current (s)",
            vdims=ylabel,
        ).opts(
            color="black",
            width=width,
            height=int(height / height_factor),
            ylim=ylim,
            xlim=window_lim,
        )
        return line

    return hv.DynamicMap(pn.bind(plot_func, player))


def create_player(
    n_time: int, sampling_frequency: int = 250, width: int = 800
) -> pn.widgets.Player:
    """Create player widget for time-based playback.

    Parameters
    ----------
    n_time : int
        Number of time points
    sampling_frequency : int, optional
        Sampling frequency in Hz, by default 250
    width : int, optional
        Widget width, by default 800

    Returns
    -------
    pn.widgets.Player
        Panel player widget

    """
    base_interval = int(1000 / sampling_frequency)  # ms/frame
    player = pn.widgets.Player(
        start=0,
        end=n_time - 1,
        value=0,
        step=1,
        interval=base_interval,
        name="Time",
        width=width,
        show_loop_controls=False,
    )
    return player


def create_playback_speed_dropdown(
    player: pn.widgets.Player, sampling_frequency: int = 250, width: int = 800
) -> pn.widgets.Select:
    """Create dropdown to control playback speed.

    Parameters
    ----------
    player : pn.widgets.Player
        Player widget to control
    sampling_frequency : int, optional
        Sampling frequency in Hz, by default 250
    width : int, optional
        Widget width, by default 800

    Returns
    -------
    pn.widgets.Select
        Panel select dropdown widget

    """
    # Dropdown to control the speed of the player (interval)
    speed_options = {
        "1/8x": 8,
        "1/4x": 4,
        "1/2x": 2,
        "1x": 1,
        "2x": 0.5,
        "4x": 0.25,
        "8x": 0.125,
    }
    speed_dropdown = pn.widgets.Select(
        name="Playback Speed", options=speed_options, value=1, width=width
    )
    base_interval = 1000 / sampling_frequency  # ms/frame

    # Update the player's interval property based on the dropdown's value
    def update_interval(event: object) -> None:
        player.interval = int(event.new * base_interval)  # type: ignore[attr-defined]

    speed_dropdown.param.watch(update_interval, "value")

    return speed_dropdown


def create_visualization(
    position_info: pd.DataFrame,
    posterior: xr.DataArray,
    multiunit_firing_rate: pd.Series,
    non_local_prob: pd.Series,
    map_estimate: pd.DataFrame | None = None,
    sampling_frequency: int = 250,
    width: int = 800,
    height: int = 600,
    window_lim: tuple[float, float] = (-0.4, 0.4),
    position_names: tuple[str, str] = ("x", "y"),
    speed_name: str = "speed",
    head_dir_name: str = "head_direction",
) -> pn.Row:
    """Create complete interactive visualization layout.

    Combines 2D decoding plot with scrolling line plots for multiunit activity,
    speed, and non-local probability.

    Parameters
    ----------
    position_info : pd.DataFrame
        Position information indexed by time
    posterior : xr.DataArray
        Posterior probability distribution over 2D space
    multiunit_firing_rate : pd.Series
        Multiunit firing rate time series
    non_local_prob : pd.Series
        Non-local probability time series
    map_estimate : pd.DataFrame, optional
        Maximum a posteriori position estimate, by default None
    sampling_frequency : int, optional
        Sampling frequency in Hz, by default 250
    width : int, optional
        Total width of visualization, by default 800
    height : int, optional
        Total height of visualization, by default 600
    window_lim : tuple, optional
        Time window for scrolling plots, by default (-0.4, 0.4)
    position_names : tuple, optional
        Names of x, y position columns, by default ("x", "y")
    speed_name : str, optional
        Name of speed column, by default "speed"
    head_dir_name : str, optional
        Name of head direction column, by default "head_direction"

    Returns
    -------
    pn.Row
        Panel row layout containing all visualizations

    """
    # Import here to avoid circular dependency
    from .figurl import create_2D_decoding_plot

    decode_plot, player = create_2D_decoding_plot(
        position_info,
        posterior,
        map_estimate=map_estimate,
        sampling_frequency=sampling_frequency,
        width=width // 2,
        height=height,
        position_names=position_names,
        head_dir_name=head_dir_name,
    )

    multiunit_plot = create_scrolling_line_plot(
        multiunit_firing_rate,
        "Rate (spikes/s)",
        (0, multiunit_firing_rate.to_numpy().max()),
        player,
        window_lim=window_lim,
        height_factor=3,
        sampling_frequency=sampling_frequency,
        width=width // 2,
        height=height,
    )
    speed_plot = create_scrolling_line_plot(
        position_info[speed_name],
        "Speed (cm/s)",
        (0, 100),
        player,
        window_lim=window_lim,
        height_factor=3,
        sampling_frequency=sampling_frequency,
        width=width // 2,
        height=height,
    )
    non_local_plot = create_scrolling_line_plot(
        non_local_prob,
        "Non-Local prob",
        (0.0, 1.01),
        player,
        window_lim=window_lim,
        height_factor=3,
        sampling_frequency=sampling_frequency,
        width=width // 2,
        height=height,
    )

    # Layout the components using Panel
    layout = pn.Row(
        decode_plot,
        pn.Column(multiunit_plot, speed_plot, non_local_plot),
    )

    return layout
