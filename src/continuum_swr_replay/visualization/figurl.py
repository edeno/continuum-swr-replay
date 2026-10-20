"""Interactive figurl visualizations for neural decoding.

This module provides functions to create interactive web-based visualizations
using the sortingview/figurl framework. These visualizations allow exploring
decoding results, neural activity, and behavioral data in an interactive interface
that can be shared via URLs.

Figurl (Figure URL) is a web-based framework for creating shareable, interactive
scientific visualizations. The functions in this module create complex multi-panel
views that synchronize behavioral data, neural activity, and decoded spatial
representations for exploring replay events during sharp-wave ripples.

Functions
---------
create_interactive_1D_decoding_figurl
    Create interactive 1D decoding visualization with multiple data streams
create_multiregion_figurl
    Create multi-region comparison visualization
create_2D_decoding_plot
    Create 2D spatial decoding visualization with holoviews

See Also
--------
continuum_swr_replay.visualization.plots : Static matplotlib plots
continuum_swr_replay.visualization.interactive : Panel/Holoviews components
sortingview : Framework for creating figurl visualizations

Notes
-----
All figurl functions return URLs that can be:
- Opened in a web browser for interactive exploration
- Shared with collaborators (data is embedded in the URL)
- Embedded in Jupyter notebooks using IFrame

The visualizations support:
- Synchronized time axis across all panels
- Zoom and pan interactions
- Time range selection
- Export of selected time windows

Examples
--------
Create a 1D decoding figurl:

>>> from continuum_swr_replay import load_data
>>> from continuum_swr_replay.visualization import create_interactive_1D_decoding_figurl
>>>
>>> data = load_data("chimi20200212_.nwb", "02_r1")
>>> # After running decoding...
>>> url = create_interactive_1D_decoding_figurl(
...     linear_position=data['position_info']['linear_position'],
...     speed=data['position_info']['head_speed'],
...     head_direction=data['position_info']['head_orientation'],
...     results=results,
...     MUA_times=mua_times,
...     MUA_zscore=data['multiunit_rate_zscore']['HPC'],
...     SWR_times=data['ripple_times'],
...     SWR_zscore=data['zscored_ripple_consensus_trace'],
...     ahead_behind_distance=ahead_behind_dist,
...     hpd=hpd,
...     theta_filtered_lfp=data['theta_filtered_lfp'],
...     label="chimi_20200212_02_r1_HPC"
... )
>>> print(f"View at: {url}")

Display in Jupyter notebook:

>>> from IPython.display import IFrame
>>> IFrame(url, width=1200, height=800)

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import sortingview.views as vv
import xarray as xr
from non_local_detector import NonLocalSortedSpikesDetector
from non_local_detector.analysis.posterior import maximum_a_posteriori_estimate
from non_local_detector.visualization.figurl_1D import create_1D_decode_view

if TYPE_CHECKING:
    from numpy.typing import NDArray

from continuum_swr_replay.analysis.events.detection import get_confident_times
from continuum_swr_replay.analysis.hippocampus import (
    get_conditional_non_local_posterior,
)
from continuum_swr_replay.visualization.palettes import COLORS

from .interactive import create_playback_speed_dropdown, create_player


def _create_speed_view(speed: pd.Series) -> vv.TimeseriesGraph:
    """Create speed visualization timeseries graph.

    Parameters
    ----------
    speed : pd.Series, shape (n_time,)
        Speed of the animal in cm/s. Time-indexed.

    Returns
    -------
    vv.TimeseriesGraph
        Speed timeseries graph view.

    """
    return vv.TimeseriesGraph().add_line_series(
        name="Speed [cm/s]",
        t=np.asarray(speed.index),
        y=np.asarray(speed, dtype=np.float32).squeeze(),
        color="black",
        width=1,
    )


def _create_head_direction_view(head_direction: pd.Series) -> vv.TimeseriesGraph:
    """Create head direction visualization timeseries graph.

    Parameters
    ----------
    head_direction : pd.Series, shape (n_time,)
        Head direction of the animal in radians (-π to π). Time-indexed.

    Returns
    -------
    vv.TimeseriesGraph
        Head direction timeseries graph view with y-range [-π, π].

    """
    return vv.TimeseriesGraph(
        y_range=[-np.pi, np.pi],
    ).add_line_series(
        name="Dir. [rad]",
        t=np.asarray(head_direction.index),
        y=np.asarray(head_direction, dtype=np.float32).squeeze(),
        color="black",
        width=1,
    )


def _create_mua_view(
    MUA_times: pd.DataFrame,
    MUA_zscore: pd.Series,
) -> vv.TimeseriesGraph:
    """Create multiunit activity (MUA) visualization timeseries graph.

    Parameters
    ----------
    MUA_times : pd.DataFrame, shape (n_events, 2)
        Multiunit activity event times with columns ['start_time', 'end_time']
        in seconds.
    MUA_zscore : pd.Series, shape (n_time,)
        Z-scored multiunit activity firing rate. Time-indexed.

    Returns
    -------
    vv.TimeseriesGraph
        MUA timeseries graph view with event intervals and z-score line.

    """
    multiunit_firing_rate_view = vv.TimeseriesGraph()
    multiunit_firing_rate_view.add_interval_series(
        name="MUA Events",
        t_start=MUA_times.start_time.to_numpy(),
        t_end=MUA_times.end_time.to_numpy(),
        color=COLORS["MUA"],
    )
    multiunit_firing_rate_view.add_line_series(
        name="MUA [std]",
        t=np.asarray(MUA_zscore.index),
        y=np.asarray(MUA_zscore, dtype=np.float32).squeeze(),
        color="black",
        width=1,
    )
    return multiunit_firing_rate_view


def _create_swr_view(
    SWR_times: pd.DataFrame,
    SWR_zscore: pd.Series,
) -> vv.TimeseriesGraph:
    """Create sharp-wave ripple (SWR) visualization timeseries graph.

    Parameters
    ----------
    SWR_times : pd.DataFrame, shape (n_events, 2)
        Sharp-wave ripple event times with columns ['start_time', 'end_time']
        in seconds.
    SWR_zscore : pd.Series, shape (n_time,)
        Z-scored ripple consensus trace. Time-indexed.

    Returns
    -------
    vv.TimeseriesGraph
        SWR timeseries graph view with event intervals and z-score line.

    """
    ripple_zscore_view = vv.TimeseriesGraph()
    ripple_zscore_view.add_interval_series(
        name="SWR Events",
        t_start=SWR_times.start_time.to_numpy(),
        t_end=SWR_times.end_time.to_numpy(),
        color=COLORS["SWR"],
    )
    ripple_zscore_view.add_line_series(
        name="SWR [std]",
        t=np.asarray(SWR_zscore.index),
        y=np.asarray(SWR_zscore, dtype=np.float32).squeeze(),
        color="black",
        width=1,
    )
    return ripple_zscore_view


def _create_theta_view(theta_filtered_lfp: pd.Series) -> vv.TimeseriesGraph:
    """Create theta-filtered LFP visualization timeseries graph.

    Parameters
    ----------
    theta_filtered_lfp : pd.Series, shape (n_time,)
        LFP signal filtered in theta band (6-10 Hz). Time-indexed.

    Returns
    -------
    vv.TimeseriesGraph
        Theta LFP timeseries graph view.

    """
    theta_view = vv.TimeseriesGraph()
    theta_view.add_line_series(
        name="Theta Filtered LFP",
        t=np.asarray(theta_filtered_lfp.index),
        y=np.asarray(
            theta_filtered_lfp,
            dtype=np.float32,
        ).squeeze(),
        color="black",
        width=1,
    )
    return theta_view


def _create_non_local_probability_view(results: xr.Dataset) -> vv.TimeseriesGraph:
    """Create non-local state probability visualization timeseries graph.

    Parameters
    ----------
    results : xr.Dataset
        Decoding results containing acausal_state_probabilities with
        states including "Non-Local Continuous" and "Non-Local Fragmented".

    Returns
    -------
    vv.TimeseriesGraph
        Non-local probability timeseries graph view with event intervals
        and probability lines for continuous and fragmented states.

    """
    non_local_times = get_confident_times(
        results.sel(states="Non-Local Continuous").acausal_state_probabilities
        + results.sel(states="Non-Local Fragmented").acausal_state_probabilities,
        minimum_duration=0.015,
    )

    non_local_probability_view = vv.TimeseriesGraph()
    non_local_probability_view.add_interval_series(
        name="Non-Local Events",
        t_start=non_local_times.start_time.to_numpy(),
        t_end=non_local_times.end_time.to_numpy(),
        color=COLORS["Non-Local"],
    )
    non_local_probability_view.add_line_series(
        name="Non-Local Continuous",
        t=np.asarray(results.time),
        y=np.asarray(
            results.sel(states="Non-Local Continuous").acausal_state_probabilities,
            dtype=np.float32,
        ).squeeze(),
        color="black",
        width=1,
    )
    non_local_probability_view.add_line_series(
        name="Non-Local Fragmented",
        t=np.asarray(results.time),
        y=np.asarray(
            results.sel(states="Non-Local Fragmented").acausal_state_probabilities,
            dtype=np.float32,
        ).squeeze(),
        color="#d62728",
        width=1,
    )
    return non_local_probability_view


def _create_local_probability_view(results: xr.Dataset) -> vv.TimeseriesGraph:
    """Create local state probability visualization timeseries graph.

    Parameters
    ----------
    results : xr.Dataset
        Decoding results containing acausal_state_probabilities with
        states including "Local" and "No-Spike".

    Returns
    -------
    vv.TimeseriesGraph
        Local probability timeseries graph view with event intervals
        and probability lines for local and no-spike states.

    """
    local_times = get_confident_times(
        results.sel(states="Local").acausal_state_probabilities,
        minimum_duration=0.0,
    )

    local_probability_view = vv.TimeseriesGraph()
    local_probability_view.add_interval_series(
        name="Local Events",
        t_start=local_times.start_time.to_numpy(),
        t_end=local_times.end_time.to_numpy(),
        color="#d3d3d3",
    )
    local_probability_view.add_line_series(
        name="No-Spike",
        t=np.asarray(results.time),
        y=np.asarray(
            results.sel(states="No-Spike").acausal_state_probabilities,
            dtype=np.float32,
        ).squeeze(),
        color="#ff7f0e",
        width=1,
    )
    local_probability_view.add_line_series(
        name="Local",
        t=np.asarray(results.time),
        y=np.asarray(
            results.sel(states="Local").acausal_state_probabilities,
            dtype=np.float32,
        ).squeeze(),
        color="black",
        width=1,
    )
    return local_probability_view


def _create_ahead_behind_distance_view(
    ahead_behind_distance: NDArray[np.floating],
    results: xr.Dataset,
) -> vv.TimeseriesGraph:
    """Create ahead-behind distance metric visualization timeseries graph.

    Parameters
    ----------
    ahead_behind_distance : np.ndarray, shape (n_time,)
        Distance in cm between decoded position and actual position.
        Positive indicates ahead, negative indicates behind.
    results : xr.Dataset
        Decoding results containing time coordinate.

    Returns
    -------
    vv.TimeseriesGraph
        Ahead-behind distance timeseries graph view.

    """
    return vv.TimeseriesGraph().add_line_series(
        name="Ahead-Behind Distance",
        t=np.asarray(results.time),
        y=np.asarray(
            ahead_behind_distance,
            dtype=np.float32,
        ).squeeze(),
        color="black",
        width=1,
    )


def _create_hpd_view(
    hpd: NDArray[np.floating],
    results: xr.Dataset,
) -> vv.TimeseriesGraph:
    """Create HPD (Highest Posterior Density) metric visualization timeseries graph.

    Parameters
    ----------
    hpd : np.ndarray, shape (n_time,)
        Highest Posterior Density interval width in cm.
    results : xr.Dataset
        Decoding results containing time coordinate.

    Returns
    -------
    vv.TimeseriesGraph
        HPD timeseries graph view.

    """
    return vv.TimeseriesGraph().add_line_series(
        name="HPD",
        t=np.asarray(results.time),
        y=np.asarray(
            hpd,
            dtype=np.float32,
        ).squeeze(),
        color="black",
        width=1,
    )


def create_interactive_1D_decoding_figurl(
    linear_position: pd.Series,
    speed: pd.Series,
    head_direction: pd.Series,
    results: xr.Dataset,
    MUA_times: pd.DataFrame,
    MUA_zscore: pd.Series,
    SWR_times: pd.DataFrame,
    SWR_zscore: pd.Series,
    ahead_behind_distance: NDArray[np.floating],
    hpd: NDArray[np.floating],
    theta_filtered_lfp: pd.Series,
    posterior: NDArray[np.floating] | xr.DataArray | None = None,
    view_height: int = 800,
    label: str = "",
) -> str:
    """Create interactive 1D decoding visualization with multiple data streams.

    Creates a comprehensive figurl visualization showing decoded position,
    behavioral variables, neural activity (MUA, SWR, theta), and state
    probabilities in a synchronized, interactive web interface.

    Parameters
    ----------
    linear_position : pd.Series, shape (n_time,)
        Position of the animal in 1D linearized coordinates (cm).
        Index should be time in seconds.
    speed : pd.Series, shape (n_time,)
        Speed of the animal in cm/s. Time-indexed.
    head_direction : pd.Series, shape (n_time,)
        Head direction of the animal in radians (-π to π). Time-indexed.
    results : xr.Dataset
        Decoding results containing coordinates (time, position) and variables:
        - acausal_state_probabilities : state probabilities over time
        - acausal_posterior : posterior probability distribution
    MUA_times : pd.DataFrame, shape (n_events, 2)
        Multiunit activity event times with columns ['start_time', 'end_time']
        in seconds.
    MUA_zscore : pd.Series, shape (n_time,)
        Z-scored multiunit activity firing rate. Time-indexed.
    SWR_times : pd.DataFrame, shape (n_events, 2)
        Sharp-wave ripple event times with columns ['start_time', 'end_time']
        in seconds.
    SWR_zscore : pd.Series, shape (n_time,)
        Z-scored ripple consensus trace. Time-indexed.
    ahead_behind_distance : np.ndarray, shape (n_time,)
        Distance in cm between decoded position and actual position.
        Positive indicates ahead, negative indicates behind.
    hpd : np.ndarray, shape (n_time,)
        Highest Posterior Density interval width in cm.
    theta_filtered_lfp : pd.Series, shape (n_time,)
        LFP signal filtered in theta band (6-10 Hz). Time-indexed.
    posterior : np.ndarray, optional, shape (n_time, n_position)
        Conditional non-local posterior probability. If None, will be
        computed from results. Default is None.
    view_height : int, optional
        Height of the figurl view in pixels. Default is 800.
    label : str, optional
        Label for the figurl URL for organization. Default is empty string.

    Returns
    -------
    str
        URL to the interactive figurl visualization that can be opened
        in a web browser.

    Notes
    -----
    The visualization includes synchronized panels showing:
    - Decoded position posterior heatmap
    - State probabilities (Non-Local Continuous, Non-Local Fragmented, Local, No-Spike)
    - Neural activity (SWR z-score, MUA z-score, Theta LFP)
    - Behavioral variables (speed, head direction)
    - Decoding quality metrics (ahead-behind distance, HPD)

    All panels share a common time axis and can be synchronized for exploration.

    Examples
    --------
    >>> from continuum_swr_replay import load_data
    >>> from continuum_swr_replay.visualization import create_interactive_1D_decoding_figurl
    >>>
    >>> # Load data
    >>> data = load_data("chimi20200212_.nwb", "02_r1")
    >>>
    >>> # Assuming you have run decoding and have results
    >>> url = create_interactive_1D_decoding_figurl(
    ...     linear_position=data['position_info']['linear_position'],
    ...     speed=data['position_info']['head_speed'],
    ...     head_direction=data['position_info']['head_orientation'],
    ...     results=results,
    ...     MUA_times=mua_times,
    ...     MUA_zscore=data['multiunit_rate_zscore']['HPC'],
    ...     SWR_times=data['ripple_times'],
    ...     SWR_zscore=data['zscored_ripple_consensus_trace'],
    ...     ahead_behind_distance=ahead_behind_dist,
    ...     hpd=hpd,
    ...     theta_filtered_lfp=data['theta_filtered_lfp'],
    ...     label="chimi_20200212_02_r1_HPC"
    ... )
    >>> print(f"View results at: {url}")

    """
    if posterior is None:
        posterior = get_conditional_non_local_posterior(results)

    decode_view = create_1D_decode_view(
        posterior=posterior,
        linear_position=linear_position,
    )

    # Create all visualization views using helper functions
    speed_view = _create_speed_view(speed)
    head_direction_view = _create_head_direction_view(head_direction)
    multiunit_firing_rate_view = _create_mua_view(MUA_times, MUA_zscore)
    ripple_zscore_view = _create_swr_view(SWR_times, SWR_zscore)
    non_local_probability_view = _create_non_local_probability_view(results)
    local_probability_view = _create_local_probability_view(results)
    theta_view = _create_theta_view(theta_filtered_lfp)
    ahead_behind_distance_view = _create_ahead_behind_distance_view(
        ahead_behind_distance, results
    )
    hpd_view = _create_hpd_view(hpd, results)

    vertical_panel_content = [
        vv.LayoutItem(theta_view, stretch=1, title="Theta", collapsible=True),
        vv.LayoutItem(ripple_zscore_view, stretch=1, title="SWR", collapsible=True),
        vv.LayoutItem(
            multiunit_firing_rate_view, stretch=1, title="MUA", collapsible=True
        ),
        vv.LayoutItem(
            non_local_probability_view, stretch=1, title="Non-Local", collapsible=True
        ),
        vv.LayoutItem(
            local_probability_view, stretch=1, title="Local", collapsible=True
        ),
        vv.LayoutItem(decode_view, stretch=3, title="Decode", collapsible=True),
        vv.LayoutItem(
            ahead_behind_distance_view, stretch=1, title="Dist.", collapsible=True
        ),
        vv.LayoutItem(hpd_view, stretch=1, title="HPD", collapsible=True),
        vv.LayoutItem(speed_view, stretch=1, title="Speed", collapsible=True),
        vv.LayoutItem(
            head_direction_view, stretch=1, title="Head Dir.", collapsible=True
        ),
    ]

    view = vv.Box(
        direction="horizontal",
        show_titles=True,
        height=view_height,
        items=[
            vv.LayoutItem(
                vv.Box(
                    direction="vertical",
                    show_titles=True,
                    items=vertical_panel_content,
                )
            ),
        ],
    )

    url = view.url(label=label)
    if not isinstance(url, str):
        raise TypeError(
            f"Expected string URL from view.url(), got {type(url).__name__}"
        )
    return url


def create_multiregion_figurl(
    decoding_results: dict[str, xr.Dataset],
    position_info: pd.DataFrame,
    multiunit_firing_rate: dict[str, pd.Series],
    view_height: int = 800,
    label: str = "",
) -> str:
    """Create interactive multi-region decoding comparison visualization.

    Creates a figurl visualization comparing decoding results across multiple
    brain regions (e.g., HPC, mPFC, OFC) with synchronized timeseries.

    Parameters
    ----------
    decoding_results : dict of str to xr.Dataset
        Dictionary mapping brain area names (e.g., 'HPC', 'mPFC') to their
        corresponding decoding results. Each Dataset should contain
        acausal_state_probabilities and related variables.
    position_info : pd.DataFrame
        Position information indexed by time (seconds) with columns including:
        - linear_position : float, position in cm
        - head_speed : float, speed in cm/s
    multiunit_firing_rate : dict of str to pd.Series
        Dictionary mapping brain area names to their multiunit firing rates
        (spikes/s). Each Series should be time-indexed.
    view_height : int, optional
        Height of the figurl view in pixels. Default is 800.
    label : str, optional
        Label for organizing the figurl URL. Default is empty string.

    Returns
    -------
    str
        URL to the interactive multi-region figurl visualization.

    Notes
    -----
    The visualization displays:
    - Multiunit firing rates from all brain areas (overlaid)
    - Non-local probabilities from all brain areas (overlaid)
    - Individual decode view for each brain area
    - Animal speed

    This allows direct comparison of replay events across brain regions.

    Examples
    --------
    >>> results_hpc = ... # HPC decoding results
    >>> results_mpfc = ... # mPFC decoding results
    >>>
    >>> url = create_multiregion_figurl(
    ...     decoding_results={'HPC': results_hpc, 'mPFC': results_mpfc},
    ...     position_info=data['position_info'],
    ...     multiunit_firing_rate={
    ...         'HPC': data['multiunit_firing_rate']['HPC'],
    ...         'mPFC': data['multiunit_firing_rate']['mPFC']
    ...     },
    ...     label="chimi_20200212_multiregion"
    ... )

    """
    # Multiunit firing rate view
    multiunit_firing_rate_view = vv.TimeseriesGraph()
    for brain_area, rate in multiunit_firing_rate.items():
        multiunit_firing_rate_view.add_line_series(
            name=f"{brain_area} Rate [spikes/s]",
            t=np.asarray(position_info.index),
            y=np.asarray(rate, dtype=np.float32).squeeze(),
            color="red",
            width=1,
        )

    # Non-local probability view
    non_local_probability_view = vv.TimeseriesGraph()
    for brain_area, results in decoding_results.items():
        non_local_prob = results.acausal_state_probabilities.sel(
            states=["Non-Local Continuous", "Non-Local Fragmented"]
        ).sum("states")
        non_local_probability_view.add_line_series(
            name=f"{brain_area} Non-Local",
            t=np.asarray(non_local_prob.time),
            y=np.asarray(
                non_local_prob,
                dtype=np.float32,
            ).squeeze(),
            color="green",
            width=1,
        )

    # Decode views
    decode_views = {}
    for brain_area, results in decoding_results.items():
        decode_views[brain_area] = create_1D_decode_view(
            posterior=NonLocalSortedSpikesDetector.get_conditional_non_local_posterior(
                results
            ),
            linear_position=position_info.linear_position,
        )

    # Speed view
    speed_view = vv.TimeseriesGraph().add_line_series(
        name="Speed [cm/s]",
        t=np.asarray(position_info.index),
        y=np.asarray(position_info.head_speed, dtype=np.float32).squeeze(),
        color="black",
        width=1,
    )

    vertical_panel_content = [
        vv.LayoutItem(multiunit_firing_rate_view, stretch=1, title="MUA"),
        vv.LayoutItem(non_local_probability_view, stretch=1, title="Non-Local"),
    ]
    for brain_area, decode_view in decode_views.items():
        vertical_panel_content.append(
            vv.LayoutItem(decode_view, stretch=3, title=brain_area)
        )

    vertical_panel_content.append(
        vv.LayoutItem(speed_view, stretch=1, title="Speed"),
    )

    view = vv.Box(
        direction="horizontal",
        show_titles=True,
        height=view_height,
        items=[
            vv.LayoutItem(
                vv.Box(
                    direction="vertical",
                    show_titles=True,
                    items=vertical_panel_content,
                )
            ),
        ],
    )

    url = view.url(label=label)
    if not isinstance(url, str):
        raise TypeError(
            f"Expected string URL from view.url(), got {type(url).__name__}"
        )
    return url


def create_2D_decoding_plot(
    position_info: pd.DataFrame,
    posterior: xr.DataArray,
    map_estimate: pd.DataFrame | None = None,
    sampling_frequency: int = 250,
    width: int = 800,
    height: int = 600,
    head_dir_radius: float = 4.0,
    position_names: tuple[str, str] = ("x", "y"),
    head_dir_name: str = "head_direction",
    cmap: str = "viridis",
    animal_color: str = "magenta",
    decode_opacity: float = 1.0,
) -> tuple[pn.Column, pn.widgets.Player]:
    """Create interactive 2D spatial decoding visualization with playback controls.

    Creates a holoviews-based interactive plot showing the decoded posterior
    probability distribution over 2D space, overlaid with the animal's actual
    position and head direction. Includes playback controls for temporal navigation.

    Parameters
    ----------
    position_info : pd.DataFrame, shape (n_time, n_features)
        Position information indexed by time (seconds). Must contain columns
        matching `position_names` and `head_dir_name`.
    posterior : xr.DataArray, shape (n_time, n_x_bins, n_y_bins)
        Posterior probability distribution over 2D space. Should have
        dimensions (time, x_position, y_position) and a 'probability' variable.
    map_estimate : pd.DataFrame, optional, shape (n_time, 2)
        Maximum a posteriori position estimate with columns matching
        x_position and y_position. If None, computed automatically from
        posterior. Default is None.
    sampling_frequency : int, optional
        Sampling frequency of the data in Hz. Used to set playback speed.
        Default is 250.
    width : int, optional
        Width of plot in pixels. Default is 800.
    height : int, optional
        Height of plot in pixels. Default is 600.
    head_dir_radius : float, optional
        Length of head direction arrow in spatial units (cm). Default is 4.
    position_names : tuple of str, optional
        Names of (x, y) position columns in position_info. Default is ("x", "y").
    head_dir_name : str, optional
        Name of head direction column in position_info (radians).
        Default is "head_direction".
    cmap : str, optional
        Matplotlib colormap name for posterior probability. Default is "viridis".
    animal_color : str, optional
        Color for actual animal position marker and direction arrow.
        Default is "magenta".
    decode_opacity : float, optional
        Opacity of decoded posterior heatmap (0-1). Default is 1.0.

    Returns
    -------
    decode_plot : pn.Column
        Panel column layout containing the interactive visualization and
        playback controls.
    player : pn.widgets.Player
        Player widget for controlling time index. Can be used to synchronize
        with other visualizations.

    Notes
    -----
    The visualization shows:
    - Decoded posterior probability as a 2D heatmap
    - Most likely decoded position as a green dot
    - Actual animal position as a magenta dot
    - Head direction as a magenta arrow

    The playback controls allow temporal navigation with adjustable speed.
    This visualization is designed for use in Jupyter notebooks with
    holoviews and panel installed.

    Examples
    --------
    >>> import holoviews as hv
    >>> import panel as pn
    >>> hv.extension('bokeh')
    >>> pn.extension()
    >>>
    >>> # Assuming you have 2D posterior from decoding
    >>> decode_plot, player = create_2D_decoding_plot(
    ...     position_info=data['position_info'],
    ...     posterior=posterior_2d,
    ...     width=600,
    ...     height=600
    ... )
    >>> decode_plot  # Display in Jupyter notebook

    """
    n_time = len(posterior.time)

    if map_estimate is None:
        map_estimate = maximum_a_posteriori_estimate(posterior)

    def get_animal_position_dot(time_ind: int) -> hv.Overlay:
        head_point = hv.Scatter(
            position_info.iloc[[time_ind]],
            kdims=position_names[0],
            vdims=position_names[1],
        ).opts(size=10, color=animal_color, width=width, height=height)

        # Calculate the end point of the head direction line
        x_start = position_info.iloc[time_ind][position_names[0]]
        y_start = position_info.iloc[time_ind][position_names[1]]

        direction = position_info.iloc[time_ind][head_dir_name]

        x_end = x_start + head_dir_radius * np.cos(direction)
        y_end = y_start + head_dir_radius * np.sin(direction)

        head_direction_line = hv.Segments(
            [{"x0": x_start, "y0": y_start, "x1": x_end, "y1": y_end}]
        ).opts(color=animal_color, line_width=3)

        return head_point * head_direction_line

    def get_posterior_quadmesh(time_ind: int) -> hv.QuadMesh:
        posterior_slice = posterior.isel(time=time_ind)
        return hv.QuadMesh(
            posterior_slice,
            kdims=["x_position", "y_position"],
            vdims=["probability"],
        ).opts(cmap=cmap, alpha=decode_opacity, width=width, height=height)

    def get_most_likely_decoded_position(time_ind: int) -> hv.Scatter:
        return hv.Scatter(
            map_estimate[[time_ind]],
            kdims="x_position",
            vdims="y_position",
        ).opts(color="green", size=10, width=width, height=height)

    # Create player controls
    player = create_player(
        n_time, sampling_frequency=sampling_frequency, width=int(width * 0.7)
    )
    playback_speed_dropdown = create_playback_speed_dropdown(
        player, sampling_frequency=sampling_frequency, width=int(width * 0.2)
    )

    animal_position_dot = hv.DynamicMap(pn.bind(get_animal_position_dot, player))
    posterior_image = hv.DynamicMap(pn.bind(get_posterior_quadmesh, player))
    most_likely_decoded_position_dot = hv.DynamicMap(
        pn.bind(get_most_likely_decoded_position, player)
    )

    # Overlay the animal position on the posterior
    decode_plot = pn.Column(
        posterior_image * most_likely_decoded_position_dot * animal_position_dot,
        pn.Row(player, playback_speed_dropdown),
    )

    return decode_plot, player
