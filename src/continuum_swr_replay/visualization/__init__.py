"""Visualization utilities for neural decoding results.

This package provides a comprehensive suite of visualization functions for
exploring neural decoding results from the continuum replay analysis. The
tools support multiple visualization paradigms to serve different needs:

**Interactive Web Visualizations (figurl)**
- Shareable URLs with embedded data
- Multi-panel synchronized views
- Time-based navigation and zooming
- Suitable for collaboration and exploration

**Static Publication Plots (plots)**
- High-quality matplotlib figures
- Composable plotting functions
- Consistent styling for publications
- Support for multi-panel layouts

**Interactive Notebook Visualizations (interactive)**
- Real-time playback with widgets
- Synchronized multi-view layouts
- HoloViews/Panel based
- Ideal for exploratory analysis in Jupyter

**Animated Movies (animation)**
- MP4 videos for presentations
- Dark theme with high DPI
- Synchronized neural and behavioral data
- FFmpeg-based rendering

Submodules
----------
figurl
    Interactive web visualizations using sortingview framework
plots
    Static matplotlib plotting functions for publications
interactive
    Interactive Panel/Holoviews components for Jupyter notebooks
animation
    Movie generation with matplotlib animation

See Also
--------
continuum_swr_replay.data_loaders : Load and process neural data
continuum_swr_replay.analysis : Decoding algorithms
sortingview : Figurl visualization framework
matplotlib : Static plotting library
holoviews : Declarative visualization library
panel : Dashboard and widget library

Notes
-----
**Choosing a Visualization Type:**

Use **figurl** when:
- Sharing results with collaborators
- Exploring long recording sessions
- Need synchronized multi-panel views
- Want persistent shareable links

Use **plots** when:
- Creating publication figures
- Need precise control over appearance
- Generating batch figures programmatically
- Working with vector formats (PDF, SVG)

Use **interactive** when:
- Exploring data in Jupyter notebooks
- Need real-time playback
- Want to iterate quickly on visualizations
- Prefer widget-based controls

Use **animation** when:
- Creating presentation videos
- Showing temporal dynamics clearly
- Need offline playback
- Demonstrating specific events

**Installation Requirements:**

Core functionality:
- matplotlib
- numpy
- pandas
- xarray

For figurl:
- sortingview
- kachery-cloud

For interactive:
- holoviews
- panel
- bokeh (recommended backend)

For animation:
- ffmpeg (system package)
- tqdm

Examples
--------
Create a comprehensive figurl visualization:

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
...     theta_filtered_lfp=data['theta_filtered_lfp']
... )
>>> print(f"View at: {url}")

Create a publication-quality figure:

>>> from continuum_swr_replay.visualization import plot_detector
>>> import matplotlib.pyplot as plt
>>>
>>> plot_detector(
...     data, detector, results,
...     figsize=(11, 8),
...     plot_theta=True,
...     time_slice=slice(1234.0, 1240.0)
... )
>>> plt.savefig('figure2_decoding.pdf', dpi=300)

Create an interactive notebook visualization:

>>> import holoviews as hv
>>> import panel as pn
>>> from continuum_swr_replay.visualization import create_visualization
>>>
>>> hv.extension('bokeh')
>>> pn.extension()
>>>
>>> layout = create_visualization(
...     position_info=data['position_info'],
...     posterior=posterior_2d,
...     multiunit_firing_rate=data['multiunit_firing_rate']['HPC'],
...     non_local_prob=non_local_prob,
...     sampling_frequency=250
... )
>>> layout  # Display in notebook

Generate a movie of a replay event:

>>> from continuum_swr_replay.visualization import make_non_local_movie
>>>
>>> make_non_local_movie(
...     time_slice=slice(0, 2500),  # 5 seconds at 500 Hz
...     position_info=data['position_info'],
...     posterior=posterior_2d,
...     non_local_prob=non_local_prob,
...     MUA_firing_rate=mua_rate,
...     SWR_zscore=swr_zscore,
...     movie_name='replay_event.mp4'
... )

"""

# Figurl visualizations
# Animation
from .animation import make_non_local_movie
from .figurl import (
    create_2D_decoding_plot,
    create_interactive_1D_decoding_figurl,
    create_multiregion_figurl,
)

# Static plots
from .glm_plots import (
    plot_aics,
    plot_hist_relative_to_event,
    plot_raster_relative_to_event,
)

# Interactive components
from .interactive import (
    create_playback_speed_dropdown,
    create_player,
    create_scrolling_line_plot,
    create_visualization,
)
from .plots import (
    plot_acausal_probabilities,
    plot_conditional_non_local_posterior,
    plot_detector,
    plot_graph_as_1D,
    plot_head_direction,
    plot_multiunit_firing,
    plot_peri_event_probability,
    plot_ripple_consensus_trace_zscore,
    plot_speed,
    plot_spike_times,
    plot_theta_lfp,
    plot_theta_phase,
    plot_theta_power_zscore,
)

__all__ = [
    # Figurl visualizations
    "create_interactive_1D_decoding_figurl",
    "create_multiregion_figurl",
    "create_2D_decoding_plot",
    # Static plots
    "plot_peri_event_probability",
    "plot_theta_lfp",
    "plot_theta_power_zscore",
    "plot_theta_phase",
    "plot_multiunit_firing",
    "plot_ripple_consensus_trace_zscore",
    "plot_spike_times",
    "plot_acausal_probabilities",
    "plot_conditional_non_local_posterior",
    "plot_speed",
    "plot_head_direction",
    "plot_detector",
    "plot_graph_as_1D",
    # GLM plots
    "plot_raster_relative_to_event",
    "plot_hist_relative_to_event",
    "plot_aics",
    # Interactive components
    "create_scrolling_line_plot",
    "create_player",
    "create_playback_speed_dropdown",
    "create_visualization",
    # Animation
    "make_non_local_movie",
]
