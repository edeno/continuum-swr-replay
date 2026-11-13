"""Sampled example figures of the detection comparison: one drawn event's figure.

What the inventory comparison's examples (``scripts/figures/generate_inventory_examples.py``)
and the event features' examples (``scripts/figures/generate_event_feature_examples.py``)
share: one drawn event's figure (:func:`draw_example`). The events are drawn by
``analysis.detector_comparison.selection.draw_examples``; the figure's
memoryless windows are recomputed for the plotted span and checked against the
stored ones by ``analysis.detector_comparison.example_windows``, on the device
the memoryless run used (``runs.device.use_memoryless_device``). Design:
``docs/design/replay_detection_comparison/03-inventory-comparison.md``
(Reporting, item 5).
"""

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from continuum_swr_replay.analysis.detector_comparison import inventory_run
from continuum_swr_replay.analysis.detector_comparison.example_windows import (
    SPREAD_RTOL,
    EpochInputs,
    recompute_windows,
)
from continuum_swr_replay.analysis.detector_comparison.selection import EXAMPLE_SEED
from continuum_swr_replay.analysis.event_definition.nonlocal_event_rule import (
    nonlocal_probability,
)
from continuum_swr_replay.intervals import contiguous_true_runs, sample_step
from continuum_swr_replay.parameters import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    DEFAULT_NON_LOCAL_THRESHOLD,
)
from continuum_swr_replay.visualization.figure_utilities import WONG
from continuum_swr_replay.visualization.palettes import COLORS
from continuum_swr_replay.visualization.plots import plot_linear_track_graph_strip

__all__ = [
    "PAD_S",
    "PANELS_CAPTION",
    "PENDING",
    "detector_color",
    "detector_label",
    "draw_example",
    "figure_name",
]

PAD_S = 0.5
"""Plotted time either side of the drawn event."""

PENDING = "pending the Phase 2 pilot checks"
PENDING_SENTENCE = (
    "Every figure shows the memoryless inventory and its window posteriors, so "
    f"every example is marked {PENDING}."
)

GATE_ROWS: Mapping[str, str] = MappingProxyType(
    {
        "speed_eligible": "ML speed",
        "spread_below_threshold": "ML spread",
        "jump_below_threshold": "ML jump",
        "kept": "ML kept",
    }
)
"""The stored memoryless gates drawn, with their row labels."""
POSITION_STATES = ("Local", "Non-Local Continuous", "Non-Local Fragmented")
"""The state-space states with a position; ``No-Spike`` has none."""

ANIMAL_POSITION_COLOR = "magenta"
"""The animal's position over a posterior, as in Figure 1 and the event gallery."""
MEMORYLESS_COLOR = WONG[5]
"""The memoryless inventories: Wong blue, which no ``COLORS`` entry uses."""
WINDOW_CENTER_COLOR = WONG[2]
"""Each memoryless window's center (sky blue), as in the memoryless comparison."""
TRACK_MARGIN = 0.015
"""The y margin of each posterior and its track strip, as a fraction of the
track's length on each side, so a position at either end of the track is drawn
inside the axes."""


def detector_color(detector: str) -> str:
    """The paper's color for a detector's events: ``COLORS`` by family.

    Every SWR definition takes ``COLORS["SWR"]``, ``mua`` ``COLORS["MUA"]`` and
    ``state_space`` ``COLORS["Non-Local"]``; both memoryless inventories take
    :data:`MEMORYLESS_COLOR`. The bars' rows name the detector.
    """
    if detector in inventory_run.SWR_DEFINITIONS:
        return str(COLORS["SWR"])
    if detector == "mua":
        return str(COLORS["MUA"])
    if detector == "state_space":
        return str(COLORS["Non-Local"])
    return str(MEMORYLESS_COLOR)


def _track_key(ax: Any, environment: Any, max_position: float, label: str) -> None:
    """The linearized track beside a posterior, as in the event gallery.

    A strip flush to the posterior's left (``plot_linear_track_graph_strip``,
    the decode's own ``edge_order`` and ``edge_spacing``), each arm labeled by
    its ``edge_id`` at its middle. The strip takes the posterior's y label.
    Both share one y range, the track with :data:`TRACK_MARGIN` either side,
    and the strip keeps its 0 and maximum ticks.
    """
    strip = ax.inset_axes([-0.075, 0.0, 0.06, 1.0])
    graph = environment.track_graph
    plot_linear_track_graph_strip(
        graph,
        environment.edge_order,
        environment.edge_spacing,
        ax=strip,
        max_position=max_position,
        node_size=6,
    )
    n_edges = len(environment.edge_order)
    spacing = np.atleast_1d(np.asarray(environment.edge_spacing, dtype=float))
    # one gap after each edge but the last, as plot_graph_as_1D lays them
    gaps = np.resize(spacing, n_edges - 1) if spacing.size == 1 else spacing
    # plot_graph_as_1D's per-edge colors
    colors = getattr(plt.get_cmap("tab10"), "colors")  # noqa: B009
    position = 0.0
    for index, edge in enumerate(environment.edge_order):
        length = float(graph.edges[edge]["distance"])
        strip.text(
            -0.8,
            position + length / 2,
            f"a{graph.edges[edge]['edge_id']}",
            fontsize=4.5,
            color=colors[index % len(colors)],
            ha="center",
            va="center",
        )
        position += length + (gaps[index] if index < n_edges - 1 else 0.0)
    margin = TRACK_MARGIN * max_position
    strip.set_ylim(-margin, max_position + margin)
    strip.set_ylabel(label, fontsize=7)
    strip.tick_params(labelsize=6)
    ax.set_ylabel("")
    ax.set_ylim(-margin, max_position + margin)
    ax.tick_params(labelleft=False)


def _spans(time: NDArray, mask: NDArray) -> list[tuple[float, float]]:
    """``(start, width)`` of each run of True samples, one step wide each."""
    if not mask.any():
        return []
    step = sample_step(time) if time.size > 1 else 0.0
    return [
        (time[a], time[b - 1] - time[a] + step) for a, b in contiguous_true_runs(mask)
    ]


def detector_label(detector: str) -> str:
    """A detector's name as the figures print it: a memoryless one marked †
    (pending)."""
    return f"{detector}†" if detector.startswith("memoryless") else detector


def draw_example(
    inputs: EpochInputs,
    drawn: pd.Series,
    *,
    against: list[str],
    path: Path,
    run_id: str,
    seed: int = EXAMPLE_SEED,
    note: str = "",
) -> dict[str, Any]:
    """One drawn event's figure, written to ``path``; returns its record.

    ``run_id`` is the inventory run's and ``seed`` the run seed the event was
    drawn by, both named in the title; ``note``, if given, is the title's last
    line.

    Raises
    ------
    ValueError
        As :func:`recompute_windows`, or the event is not in the epoch's
        inventory.
    """
    event = inputs.inventory[inputs.inventory.event_key == drawn.event_key]
    if len(event) != 1:
        raise ValueError(f"{drawn.event_key} is not one event of the inventory.")
    start, end = float(event.event_start_s.iloc[0]), float(event.event_end_s.iloc[0])
    t0, t1 = start - PAD_S, end + PAD_S
    candidates = inputs.candidates
    midpoints = candidates.midpoint_s
    first = int(np.searchsorted(midpoints, t0, side="left"))
    last = int(np.searchsorted(midpoints, t1, side="right")) - 1
    posterior, decisions, check = recompute_windows(inputs, first, last)
    window_time = midpoints[first : last + 1]

    def rel(values: NDArray) -> NDArray:
        return np.asarray(values, dtype=np.float64) - start

    in_plot = (inputs.time >= t0) & (inputs.time <= t1)
    time = inputs.time[in_plot]
    dot = inputs.dot.loc[in_plot]
    speed = dot["head_speed"].to_numpy(dtype=np.float64)
    non_local = nonlocal_probability(dot).to_numpy(dtype=np.float64)

    fig, axes = plt.subplots(
        7,
        1,
        figsize=(7.2, 10.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1.6, 1.6, 1.6, 1.0, 0.7, 1.8, 1.5]},
        layout="constrained",
    )
    raster, state_space, memoryless, gates, speed_ax, band_ax, bars = axes

    spikes = [rel(times[(times >= t0) & (times <= t1)]) for times in inputs.spike_times]
    raster.eventplot(spikes, colors="k", linelengths=0.8, linewidths=0.4)
    raster.set_ylabel("tetrode", fontsize=7)
    raster.set_ylim(-0.5, len(spikes) - 0.5)

    posterior_ss = inputs.results.acausal_posterior.sel(time=slice(t0, t1))
    ss_time = posterior_ss.time.to_numpy()
    values = posterior_ss.to_numpy()
    states = inputs.results.state.to_numpy()
    position = inputs.results.position.to_numpy()
    by_state = [states == name for name in POSITION_STATES]
    bin_position = position[by_state[0]]
    if not all(np.array_equal(position[s], bin_position) for s in by_state):
        raise ValueError("The states with a position do not share one place grid.")
    summed = sum(values[:, s] for s in by_state)
    # the bins off the track interior hold NaN
    on_track = ~np.isnan(summed).all(axis=0)
    summed = summed[:, on_track]
    state_space.pcolormesh(
        rel(ss_time),
        bin_position[on_track],
        summed.T,
        shading="nearest",
        cmap="bone_r",
        vmin=0.0,
        vmax=max(float(np.quantile(summed, 0.99)), 1e-12),
        rasterized=True,
    )
    animal_position = dot["linear_position"].to_numpy(dtype=np.float64)
    environment = inputs.encoding_model["environment"]
    max_position = float(np.ceil(np.nanmax(environment.place_bin_edges_)))
    state_space.plot(
        rel(time), animal_position, color=ANIMAL_POSITION_COLOR, lw=0.8, alpha=0.9
    )
    _track_key(state_space, environment, max_position, "state-space\nposterior (cm)")

    usable = np.isfinite(posterior).all(axis=1)
    shown = np.where(usable[:, None], posterior, 0.0)
    memoryless.pcolormesh(
        rel(window_time),
        inputs.bin_centers_cm,
        shown.T,
        shading="nearest",
        cmap="bone_r",
        vmin=0.0,
        vmax=max(float(np.quantile(shown, 0.99)), 1e-12),
        rasterized=True,
    )
    has_center = decisions.center_index >= 0
    memoryless.scatter(
        rel(window_time[has_center]),
        inputs.bin_centers_cm[decisions.center_index[has_center]],
        s=1.5,
        color=WINDOW_CENTER_COLOR,
        linewidths=0,
    )
    memoryless.plot(
        rel(time), animal_position, color=ANIMAL_POSITION_COLOR, lw=0.8, alpha=0.9
    )
    _track_key(memoryless, environment, max_position, "memoryless†\nwindows (cm)")

    stored = candidates.decisions.gates
    rows = {
        f"p(non-local) > {DEFAULT_NON_LOCAL_THRESHOLD}": (
            time,
            non_local > DEFAULT_NON_LOCAL_THRESHOLD,
        ),
        **{
            label: (window_time, getattr(stored, name)[first : last + 1])
            for name, label in GATE_ROWS.items()
        },
        f"speed < {DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC} cm/s": (
            time,
            speed < DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
        ),
    }
    row_colors = [COLORS["Non-Local"], *[MEMORYLESS_COLOR] * len(GATE_ROWS), "0.4"]
    for row, ((times, mask), color) in enumerate(
        zip(rows.values(), row_colors, strict=True)
    ):
        gates.broken_barh(
            [(s - start, w) for s, w in _spans(times, np.asarray(mask, bool))],
            (row - 0.4, 0.8),
            color=color,
            linewidth=0,
        )
    gates.set_yticks(range(len(rows)), list(rows), fontsize=6)
    gates.set_ylim(len(rows) - 0.5, -0.5)

    speed_ax.plot(rel(time), speed, color="k", lw=0.7)
    speed_ax.axhline(
        DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC, color="red", ls="--", lw=0.8
    )
    speed_ax.set_ylabel("speed\n(cm/s)", fontsize=7)

    in_band = (inputs.band_time >= t0) & (inputs.band_time <= t1)
    band = inputs.ripple_band[in_band]
    scale = float(np.nanquantile(np.abs(band), 0.99)) if band.size else 1.0
    offsets = -2.0 * scale * np.arange(band.shape[1])
    band_ax.plot(rel(inputs.band_time[in_band]), band + offsets, color="k", lw=0.3)
    band_ax.set_yticks([])
    band_ax.set_ylabel(f"ripple band\n({band.shape[1]} kept tetrodes)", fontsize=7)

    detectors = list(inventory_run.DETECTORS)
    for row, detector in enumerate(detectors):
        own = inputs.inventory[inputs.inventory.detector == detector]
        own = own[(own.event_end_s >= t0) & (own.event_start_s <= t1)]
        moving = ~own.whole_event_speed_eligible.to_numpy(dtype=bool)
        for subset, alpha in ((own[~moving], 0.9), (own[moving], 0.3)):
            bars.broken_barh(
                list(
                    zip(rel(subset.event_start_s), subset.interval_span_s, strict=True)
                ),
                (row - 0.4, 0.8),
                color=detector_color(detector),
                alpha=alpha,
                linewidth=0,
            )
        aside = inputs.set_aside[inputs.set_aside.detector == detector]
        aside = aside[(aside.interval_end_s >= t0) & (aside.interval_start_s <= t1)]
        bars.broken_barh(
            list(
                zip(
                    rel(aside.interval_start_s),
                    aside.interval_end_s - aside.interval_start_s,
                    strict=True,
                )
            ),
            (row - 0.4, 0.8),
            facecolor="none",
            edgecolor="0.3",
            linewidth=0.6,
        )
    bars.set_yticks(
        range(len(detectors)), [detector_label(d) for d in detectors], fontsize=6
    )
    bars.set_ylim(len(detectors) - 0.5, -0.5)
    bars.set_xlabel(
        f"time from the event's start (s); event lasts {end - start:.3f} s", fontsize=7
    )

    excluded = inputs.excluded[
        (inputs.excluded.end_time >= t0) & (inputs.excluded.start_time <= t1)
    ]
    for ax in axes:
        for _, interval in excluded.iterrows():
            ax.axvspan(
                interval.start_time - start,
                interval.end_time - start,
                color="0.6",
                alpha=0.35,
                linewidth=0,
            )
        for edge in (0.0, end - start):
            ax.axvline(edge, color="k", lw=0.6, ls=":")
        ax.tick_params(labelsize=6)
    raster.set_xlim(-PAD_S, end - start + PAD_S)
    fig.suptitle(
        f"{drawn.cell}: {detector_label(drawn.detector)} event, {drawn.role} "
        f"({drawn.membership} by {', '.join(detector_label(d) for d in against)})\n"
        f"{drawn.event_key} ({drawn.animal} {drawn.day} {drawn.epoch})\n"
        f"† {PENDING}; inventory run {run_id}; seed {seed}"
        + (f"\n{note}" if note else ""),
        fontsize=7,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return {
        **drawn.to_dict(),
        "against": against,
        "event_start_s": start,
        "event_end_s": end,
        "path": str(path),
        "memoryless_check": check,
    }


def figure_name(drawn: pd.Series) -> str:
    return f"{drawn.cell}_{drawn.detector}_{drawn.animal}_{drawn.role}{drawn.draw}.png"


PANELS_CAPTION = (
    f" Each figure shows one event on one time axis, {PAD_S} s either side of "
    "it (dotted lines mark the event): the spikes of each tetrode (the decode's "
    "marks); the state-space posterior summed over the Local and both "
    "Non-Local states, with the animal's linear position (magenta); the "
    "memoryless window posteriors, recomputed for the plotted windows on the "
    "decode's encoding model, with each window's center (sky-blue dots) and the "
    "animal's position. Beside each posterior, the decode's linearized track, "
    "its arms labeled a0-a8. The recomputed windows' spreads, "
    "jumps and candidate statistics equal the stored ones within "
    f"{SPREAD_RTOL} relative, and their centers and gates equal them except "
    "where a center tied between two bins or a statistic sat at its threshold "
    "(manifest.json, memoryless_check). Then the gate masks "
    f"(non-local probability over {DEFAULT_NON_LOCAL_THRESHOLD}; the memoryless "
    "per-window speed, spread, jump and kept gates as stored; speed below "
    f"{DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC} cm/s); the speed; the stored ripple "
    "band of the tetrodes the SWR source kept; and every inventory's events "
    "(compared events solid, moving events light, events set aside outlined), "
    "colored by family as in the paper: SWR definitions orange, MUA purple, "
    "state-space green, memoryless blue. "
    "Excluded time is shaded grey in every panel. " + PENDING_SENTENCE
)
"""What every figure's panels show, after the sampling rule in a caption."""
