"""An example event drawn as a column: ripple trace, spike raster, decode, states.

Figure 2 (panel b) and Figure 3's supplement draw example events the same way:
one column per event over a window around its center, with the ripple consensus
trace and the SWR threshold, the HPC tetrodes' spikes, the decoded position
(the posterior summed over the states that have one, ``bone_r``) with the
animal's position in magenta and the linearized track as the y key, and the
stacked state probabilities. :func:`draw_examples` draws a row of them from an
example session (:data:`SESSION_INPUTS`); the parts are used by Figure 2's
overview panel too.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import SubplotSpec
from non_local_detector.environment import Environment

from continuum_swr_replay.analysis.event_definition.nonlocal_event_rule import (
    nonlocal_state_probability,
)
from continuum_swr_replay.analysis.event_properties.class_or_continuum import (
    KAY_ZSCORE_THRESHOLD,
)
from continuum_swr_replay.visualization.palettes import COLORS
from continuum_swr_replay.visualization.plots import (
    plot_linear_track_graph_strip,
    plot_ripple_consensus_trace_zscore,
    plot_spike_times,
)
from continuum_swr_replay.visualization.publication_style import (
    LABEL_SIZE,
    LINE_WIDTH_LIMITS_PT,
)
from continuum_swr_replay.visualization.simulated_run_panels import ANIMAL_POS_COLOR

SESSION_INPUTS = ("data", "results", "detector", "env")
"""An example session's inputs: its data, and the decode's results, detector
and environment (``reported.inputs.read_example_decode``)."""

EXAMPLE_HALF_WINDOW_S = 0.2
"""Half the window drawn around each example's center (s)."""

TRACK_EDGE_LINEWIDTH = LINE_WIDTH_LIMITS_PT[1]
"""The track strip's line width, inside the journal's limits."""

C_LOCAL = COLORS["Local"]
C_NOSPIKE = COLORS["No-Spike"]
C_NL = COLORS["Non-Local"]
C_SWR = COLORS["SWR"]


def summed_posterior(window: xr.Dataset, env: Environment) -> np.ndarray:
    """Posterior over position summed over the states with one, (n_time, n_bins)."""
    post = window.acausal_posterior
    states = post.state.values
    total = sum(
        post.isel(state_bins=np.flatnonzero(states == s)).to_numpy()
        for s in ("Local", "Non-Local Continuous", "Non-Local Fragmented")
    )
    total[:, ~env.is_track_interior_] = np.nan
    return total


def draw_posterior(
    ax: Axes,
    data: dict,
    results: xr.Dataset,
    env: Environment,
    sl: slice,
    t_offset: float,
    max_pos: float,
) -> None:
    """The decoded position (bone_r) with the animal's position (magenta dots)."""
    window = results.sel(time=sl)
    cmap = plt.get_cmap("bone_r").copy()
    cmap.set_bad("#e6e6e6")
    centers = env.place_bin_centers_
    t = window.time.values - t_offset
    ax.imshow(
        summed_posterior(window, env).T,
        aspect="auto",
        origin="lower",
        extent=[t[0], t[-1], centers[0], centers[-1]],
        cmap=cmap,
        vmin=0,
        vmax=0.1,
        interpolation="none",
        rasterized=True,
    )
    position = data["position_info"].loc[sl]
    ax.scatter(
        position.index - t_offset,
        position.linear_position,
        color=ANIMAL_POS_COLOR,
        s=0.4,
        lw=0,
        rasterized=True,
    )
    ax.set_ylim(0, max_pos)


def draw_state_stack(ax: Axes, results: xr.Dataset, sl: slice, t_offset: float) -> None:
    """Stacked state probabilities: Non-local (both states), Local, No-Spike."""
    sp = results.acausal_state_probabilities.sel(time=sl)
    ax.stackplot(
        sp.time.values - t_offset,
        nonlocal_state_probability(sp).to_numpy(),
        sp.sel(states="Local").to_numpy(),
        sp.sel(states="No-Spike").to_numpy(),
        colors=[C_NL, C_LOCAL, C_NOSPIKE],
        lw=0,
        rasterized=True,
    )
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 1])


def add_track_strip(ax: Axes, data: dict, max_pos: float) -> Axes:
    """The linearized track as the posterior's y key, flush to its left."""
    strip = ax.inset_axes([-0.075, 0.0, 0.07, 1.0])
    plot_linear_track_graph_strip(
        data["track_graph"],
        data["linear_edge_order"],
        data["linear_edge_spacing"],
        ax=strip,
        max_position=max_pos,
        edge_linewidth=TRACK_EDGE_LINEWIDTH,
    )
    strip.set_ylabel("")
    ax.tick_params(labelleft=False, left=False)
    return strip


def draw_examples(
    fig: Figure, spec: SubplotSpec, session: dict, windows: dict, max_pos: float
) -> Axes:
    """One column per example (ripple, raster, decode, states); returns the
    first column's ripple axis. ``windows``: each example's title and its
    (start, end) in seconds."""
    data, results, detector, env = (session[name] for name in SESSION_INPUTS)
    n = len(windows)
    gs = spec.subgridspec(
        4, n, hspace=0.08, wspace=0.14, height_ratios=[0.5, 0.85, 1.1, 0.5]
    )
    first = None
    for j, (title, (start, end)) in enumerate(windows.items()):
        ax_rip, ax_raster, ax_post, ax_states = (
            fig.add_subplot(gs[r, j]) for r in range(4)
        )
        center = (start + end) / 2
        sl = slice(center - EXAMPLE_HALF_WINDOW_S, center + EXAMPLE_HALF_WINDOW_S)
        plot_ripple_consensus_trace_zscore(data, sl, ax_rip)
        # the SWR threshold as panel a draws it, under the trace
        ax_rip.axhline(KAY_ZSCORE_THRESHOLD, color=C_SWR, lw=0.5, ls="--", zorder=-1)
        plot_spike_times(data, sl, detector, env, results, ax_raster)
        draw_posterior(ax_post, data, results, env, sl, center, max_pos)
        draw_state_stack(ax_states, results, sl, center)
        for ax in (ax_rip, ax_raster):
            ax.set_xlim(sl.start, sl.stop)
            ax.tick_params(labelbottom=False)
        for ax in (ax_post, ax_states):
            ax.set_xlim(-EXAMPLE_HALF_WINDOW_S, EXAMPLE_HALF_WINDOW_S)
        ax_rip.set_title(title, fontsize=LABEL_SIZE, pad=3)
        ax_rip.set_ylim(-2, 8)
        ax_rip.set_yticks([0, 5])
        ax_post.tick_params(labelbottom=False, labelleft=False)
        ax_post.set_yticks([0, max_pos])
        ax_states.set_xticks([-0.1, 0, 0.1], ["-0.1", "0", "0.1"])
        ax_states.set_xlabel("Time from event center (s)" if j == n // 2 else "")
        if j == 0:
            first = (ax_rip, ax_raster, ax_post, ax_states)
            for ax, label in zip(
                first,
                ("Ripple\n(z)", "Tetrode", "Decoded\npos. (cm)", "State\nprob."),
                strict=True,
            ):
                ax.set_ylabel(label)
        else:
            for ax in (ax_rip, ax_raster, ax_states):
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)
    add_track_strip(first[2], data, max_pos)
    for ax in first:
        ax.yaxis.set_label_coords(-0.42, 0.5)
    return first[0]
