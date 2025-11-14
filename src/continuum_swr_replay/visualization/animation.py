"""Animation and movie generation for neural decoding visualization.

This module provides functions for creating animated videos of decoding
results overlaid on behavioral trajectories. Videos are generated using
matplotlib animation with a dark theme suitable for presentations.

The primary function `make_non_local_movie` creates a synchronized animation
showing:
- 2D spatial posterior probability as a heatmap
- Actual animal position and head direction
- Decoded (MAP) position trajectory
- Scrolling time series of non-local probability, SWR, and MUA

The resulting videos are saved in MP4 format using FFmpeg and can be used
in presentations or for detailed examination of specific replay events.

Functions
---------
make_non_local_movie
    Create animated movie of 2D decoding with neural activity traces

Notes
-----
**Requirements:**

This module requires FFmpeg to be installed and available in the system PATH.
Install FFmpeg:
- Ubuntu/Debian: `sudo apt-get install ffmpeg`
- macOS: `brew install ffmpeg`
- Windows: Download from https://ffmpeg.org/

**Video Format:**

Videos are created with:
- MP4 container format (H.264 codec)
- Variable bitrate for quality
- Dark background theme (black with white text)
- Configurable frame rate (default: 62.5 fps for 8x slowdown at 500 Hz)
- High DPI output (200 dpi) for presentation quality

**Performance:**

Movie generation can be slow for long time segments:
- ~1 minute to render 5 seconds of data (2500 frames at 500 Hz)
- Progress bar (tqdm) shows rendering progress
- Consider shorter time slices for faster iteration

**Memory Usage:**

- Entire posterior must fit in memory
- Time series are windowed to reduce memory footprint
- Peak memory usage ~2-4 GB for typical sessions

See Also
--------
continuum_swr_replay.visualization.interactive : Real-time playback in notebooks
matplotlib.animation : Animation framework
ffmpeg : Video encoding (external dependency)

Examples
--------
Create a movie of a replay event:

>>> from continuum_swr_replay import load_data
>>> from continuum_swr_replay.visualization.animation import make_non_local_movie
>>>
>>> data = load_data("chimi20200212_.nwb", "02_r1")
>>> # After running 2D decoding...
>>>
>>> # Select a replay event (e.g., during a ripple)
>>> ripple_time = data['ripple_times'].iloc[10]
>>> start_idx = int(ripple_time.start_time * 500)
>>> end_idx = int(ripple_time.end_time * 500)
>>> event_slice = slice(start_idx, end_idx)
>>>
>>> # Generate movie
>>> make_non_local_movie(
...     time_slice=event_slice,
...     position_info=data['position_info'],
...     posterior=posterior_2d,
...     non_local_prob=results['non_local_probability'],
...     MUA_firing_rate=data['multiunit_firing_rate']['HPC'],
...     SWR_zscore=data['zscored_ripple_consensus_trace'],
...     movie_name='replay_event_ripple10.mp4',
...     sampling_frequency=500,
...     video_slowdown=8  # 8x slower than real-time
... )

Generate higher quality movie:

>>> make_non_local_movie(
...     time_slice=slice(0, 1000),  # First 2 seconds at 500 Hz
...     position_info=data['position_info'],
...     posterior=posterior_2d,
...     non_local_prob=non_local_prob,
...     MUA_firing_rate=mua_rate,
...     SWR_zscore=swr_zscore,
...     movie_name='high_quality.mp4',
...     sampling_frequency=500,
...     video_slowdown=4,  # Faster playback
...     vmax=0.1  # Adjust posterior color scale
... )
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.animation as animation
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.lines import Line2D
    from numpy.typing import NDArray

from continuum_swr_replay.parameters import COLORS


def _setup_line(
    ax: Axes,
    color: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    ylabel: str,
    title: str | None = None,
) -> Line2D:
    """Set up animated line plot with dark theme for movie generation.

    Creates an empty animated line on the given axes and configures
    the visual appearance for dark-themed video output.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib axes object to configure for animation.
    color : str
        Color for the line. Can be any matplotlib color specification
        (e.g., 'red', '#FF0000', (1, 0, 0)).
    xlim : tuple of float, length 2
        X-axis limits as (xmin, xmax) in seconds.
    ylim : tuple of float, length 2
        Y-axis limits as (ymin, ymax) in data units.
    ylabel : str
        Label for the y-axis.
    title : str, optional
        Title for the axes. If None, no title is set. Default is None.

    Returns
    -------
    matplotlib.lines.Line2D
        Animated line object that can be updated with set_data().

    Notes
    -----
    The axes are styled with:
    - Black background for dark theme
    - Black spines (top, right) to blend with background
    - White text (set by dark_background style context)
    - Animated line with clip_on=False for visibility
    """
    (line,) = ax.plot([], color=color, linewidth=2, animated=True, clip_on=False)
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.set_ylabel(ylabel)
    ax.set_facecolor("black")
    ax.spines["top"].set_color("black")
    ax.spines["right"].set_color("black")

    if title is not None:
        ax.set_title(title)

    return line


def _update_line(
    time_ind: int,
    line: Line2D,
    window_time: NDArray[np.floating],
    df: pd.DataFrame | pd.Series,
    window_factor: NDArray[np.int_],
) -> None:
    """Update animated line plot data for current frame.

    Extracts data for a sliding window centered on the current time
    index and updates the line object for animation.

    Parameters
    ----------
    time_ind : int
        Current time index (frame number) in the animation.
    line : matplotlib.lines.Line2D
        Line object to update with new data.
    window_time : np.ndarray, shape (window_size,)
        Time values relative to current frame (e.g., [-1.0, ..., 0, ..., 1.0] s).
    df : pd.DataFrame or pd.Series
        Time series data to plot. Should have enough samples to cover
        the windowed indices.
    window_factor : np.ndarray, shape (window_size,)
        Index offsets relative to time_ind for extracting windowed data.
        E.g., for a 501-point window: array([-250, ..., 0, ..., 250]).

    Notes
    -----
    This function gracefully handles IndexError when the window extends
    beyond the available data (e.g., at the beginning or end of the recording).
    In such cases, the line data is not updated and remains unchanged.

    The windowing approach creates a scrolling effect where the current
    time stays centered while historical and future data scroll past.
    """
    try:
        line.set_data(
            window_time,
            np.asarray(df.iloc[time_ind + window_factor]),
        )
    except IndexError:
        pass


def make_non_local_movie(
    time_slice: slice,
    position_info: pd.DataFrame,
    posterior: xr.DataArray,
    non_local_prob: pd.Series,
    MUA_firing_rate: pd.Series,
    SWR_zscore: pd.Series,
    movie_name: str = "video_name.mp4",
    sampling_frequency: int = 500,
    video_slowdown: int = 8,
    position_name: list[str] | None = None,
    direction_name: str = "head_orientation",
    vmax: float = 0.07,
) -> None:
    """Create animated movie of 2D decoding with neural activity.

    Creates an animated video showing the decoded posterior probability
    distribution in 2D space overlaid with actual and decoded position,
    along with synchronized time series of non-local probability, SWR,
    and MUA activity.

    Parameters
    ----------
    time_slice : slice
        Time slice to animate
    position_info : pd.DataFrame
        Position information with columns for x, y, and direction
    posterior : xr.DataArray
        Posterior probability over 2D space with dimensions (time, x_position, y_position)
    non_local_prob : pd.Series
        Non-local probability time series (should extend beyond time_slice for window)
    MUA_firing_rate : pd.Series
        Multiunit activity firing rate (should extend beyond time_slice for window)
    SWR_zscore : pd.Series
        Sharp-wave ripple z-score (should extend beyond time_slice for window)
    movie_name : str, optional
        Output filename, by default "video_name.mp4"
    sampling_frequency : int, optional
        Data sampling frequency in Hz, by default 500
    video_slowdown : int, optional
        Factor to slow down video playback, by default 8
    position_name : list, optional
        Names of x, y position columns, by default ["head_position_x", "head_position_y"]
    direction_name : str, optional
        Name of head direction column, by default "head_orientation"
    vmax : float, optional
        Maximum value for posterior colormap, by default 0.07

    Notes
    -----
    This function creates a dark-themed video suitable for presentations.
    The video shows:
    - Left: 2D posterior heatmap with actual (magenta) and decoded (green) position
    - Right: Three scrolling time series (Non-Local prob, SWR z-score, MUA rate)

    The function uses matplotlib animation and requires ffmpeg to be installed.
    """
    if position_name is None:
        position_name = ["head_position_x", "head_position_y"]

    posterior = posterior.isel(time=time_slice)

    # Set up formatting for the movie files
    Writer = animation.writers["ffmpeg"]
    fps = sampling_frequency // video_slowdown
    writer = Writer(fps=fps, bitrate=-1)

    # Set up data
    map_position_ind = posterior.argmax(["x_position", "y_position"])
    map_position = np.stack(
        (
            posterior.x_position[map_position_ind["x_position"]],
            posterior.y_position[map_position_ind["y_position"]],
        ),
        axis=1,
    )

    position = np.asarray(position_info.iloc[time_slice][position_name])
    direction = np.asarray(position_info.iloc[time_slice][direction_name])

    window_size = 501

    window_ind = np.arange(window_size) - window_size // 2
    window_slice = slice(
        time_slice.start + window_ind[0], time_slice.stop + window_ind[-1]
    )
    non_local_prob = non_local_prob.iloc[window_slice]
    MUA_firing_rate = MUA_firing_rate.iloc[window_slice]
    SWR_zscore = SWR_zscore.iloc[window_slice]

    with plt.style.context("dark_background"):
        # Set up plots
        fig = plt.figure(layout="constrained", figsize=(6, 4))
        # Use string-based mosaic to avoid matplotlib HashableList type issues
        # A = decode (spans left column), B/C/D = Non-Local, SWR, MUA (right column)
        mosaic = """
            AB
            AC
            AD
            """
        axes = fig.subplot_mosaic(mosaic, width_ratios=[2, 1])

        # Map single-letter keys to descriptive names
        ax_dict = {
            "decode": axes["A"],
            "Non-Local": axes["B"],
            "SWR": axes["C"],
            "MUA": axes["D"],
        }

        ax_dict["decode"].tick_params(colors="white", which="both")
        ax_dict["decode"].spines["bottom"].set_color("white")
        ax_dict["decode"].spines["left"].set_color("white")

        position_dot = ax_dict["decode"].scatter(
            [],
            [],
            s=80,
            zorder=102,
            color="magenta",
            label="actual position",
            animated=True,
        )
        (position_line,) = ax_dict["decode"].plot(
            [], [], color="magenta", linewidth=5, animated=True
        )
        map_dot = ax_dict["decode"].scatter(
            [],
            [],
            s=80,
            zorder=102,
            color="green",
            label="decoded position",
            animated=True,
        )
        (map_line,) = ax_dict["decode"].plot(
            [], [], "green", linewidth=3, animated=True
        )

        mesh = posterior.isel(time=0).plot(
            x="x_position",
            y="y_position",
            vmin=0.0,
            vmax=vmax,
            ax=ax_dict["decode"],
            animated=True,
            add_colorbar=False,
        )
        ax_dict["decode"].set_xlabel("")
        ax_dict["decode"].set_ylabel("")
        ax_dict["decode"].spines["top"].set_color("black")
        ax_dict["decode"].spines["right"].set_color("black")
        xlim = (
            position_info[position_name[0]].min() - 10,
            position_info[position_name[0]].max() + 10,
        )
        ylim = (
            position_info[position_name[1]].min() - 10,
            position_info[position_name[1]].max() + 10,
        )

        ax_dict["decode"].set_xlim(xlim)
        ax_dict["decode"].set_ylim(ylim)

        title = ax_dict["decode"].set_title(
            f"time = {posterior.isel(time=0).time.values:0.2f}",
        )
        fontprops = fm.FontProperties(size=16)
        scalebar = AnchoredSizeBar(
            ax_dict["decode"].transData,
            20,
            "20 cm",
            "lower right",
            pad=0.1,
            color="white",
            frameon=False,
            size_vertical=1,
            fontproperties=fontprops,
        )

        ax_dict["decode"].add_artist(scalebar)
        ax_dict["decode"].axis("off")

        # Non-Local
        line_xlim = (
            window_ind[0] / sampling_frequency,
            window_ind[-1] / sampling_frequency,
        )
        non_local_line = _setup_line(
            ax=ax_dict["Non-Local"],
            color=COLORS["Non-Local"],
            xlim=line_xlim,
            ylim=(0.0, 1.05),
            ylabel="Non-Local\nProb.",
            title="Non-Local",
        )

        # SWR
        ax_dict["SWR"].axhline(0, color="white", linestyle="--")
        swr_line = _setup_line(
            ax=ax_dict["SWR"],
            color=COLORS["SWR"],
            xlim=line_xlim,
            ylim=(-3.0, 10.0),
            ylabel="SWR [std]",
            title="SWR",
        )

        # MUA
        mua_line = _setup_line(
            ax=ax_dict["MUA"],
            color=COLORS["MUA"],
            xlim=line_xlim,
            ylim=(0.0, 3000.0),
            ylabel="MUA [sp/s]",
            title="MUA",
        )
        ax_dict["MUA"].set_xlabel("Time [s]")

        n_frames = posterior.shape[0]
        progress_bar = tqdm()
        progress_bar.reset(total=n_frames)
        window_time = window_ind / sampling_frequency
        window_factor = (window_size // 2) + window_ind

        def _update_plot(time_ind: int) -> tuple:
            start_ind = max(0, time_ind - 5)
            time_slice = slice(start_ind, time_ind)

            position_dot.set_offsets(position[time_ind])

            r = 4
            position_line.set_data(
                [
                    position[time_ind, 0],
                    position[time_ind, 0] + r * np.cos(direction[time_ind]),
                ],
                [
                    position[time_ind, 1],
                    position[time_ind, 1] + r * np.sin(direction[time_ind]),
                ],
            )

            map_dot.set_offsets(map_position[time_ind])
            map_line.set_data(map_position[time_slice, 0], map_position[time_slice, 1])

            mesh.set_array(posterior.isel(time=time_ind).values.ravel(order="F"))

            title.set_text(f"time = {posterior.isel(time=time_ind).time.values:0.2f}")

            _update_line(
                time_ind=time_ind,
                line=non_local_line,
                window_time=window_time,
                df=non_local_prob,
                window_factor=window_factor,
            )
            _update_line(
                time_ind=time_ind,
                line=swr_line,
                window_time=window_time,
                df=SWR_zscore,
                window_factor=window_factor,
            )
            _update_line(
                time_ind=time_ind,
                line=mua_line,
                window_time=window_time,
                df=MUA_firing_rate,
                window_factor=window_factor,
            )

            progress_bar.update()

            return (
                position_dot,
                position_line,
                map_dot,
                map_line,
                mesh,
                title,
                non_local_line,
                swr_line,
                mua_line,
            )

        movie = animation.FuncAnimation(
            fig, _update_plot, frames=n_frames, interval=1000 / fps, blit=True
        )
        if movie_name is not None:
            movie.save(movie_name, writer=writer, dpi=200)

        plt.close(fig)
