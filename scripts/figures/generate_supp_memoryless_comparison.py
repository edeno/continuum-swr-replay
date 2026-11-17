#!/usr/bin/env python
"""Supplementary figure, the companion to Figure 1: a sliding-window sequence
detector on the same run.

Figure 1 shows how the switching state-space model works on a simulated run.
This figure runs the published sliding-window sequence detector (Widloski &
Foster; ``analysis.detection.memoryless_sequence``) beside it, on replays whose
shape is known:

    a: seven simulated events at a millisecond scale — raster, each detector's
       decoded position, and three rows of events: what was simulated, what the
       sliding-window detector reports, what the state-space model reports
    b: the share of injected replays each detector reports, by how far the
       replay travels and how long it lasts
    c: the share of replays whose onset or offset error is beyond each size
    d: each detector's position error against the replay's speed

The narrative is: the sliding-window detector asks whether the decoded position
moves far enough for long enough, so it declines a replay that is short in
path, held still or scattered, and it reports the ones it finds too long; the
state-space model asks where the spikes point, and reports each.

Panel a is one recording and its two decodes,
``memoryless_sequence.comparison_simulation.simulate_and_decode``: Figure 1's
run with four replays drawn in, its model and event rule, and the sliding-window
detector at its published
settings, reading the same place fields. Panels b-d read the aggregates of
``scripts/detection/sweep_memoryless_detection.py``, which wrote replays of
chosen path length, duration, firing rate and distance from the animal one at a
time into that run; re-run the sweep to change them. The timing panels read
them through ``memoryless_sequence.sweep_statistics``, which the detection
summary reports from.

The figure's numbers are reported values (``continuum_swr_replay.reported.detection``):
the sweep's from its CSV, the labeled recording's from the JSON this script
writes (:data:`RECORDING_NUMBERS_PATH`,
``memoryless_sequence.recording_statistics.recording_numbers``). What the
panels' titles say about an event is read from the data, and where the figure
commits to a verdict the script checks it and stops rather than draw a figure
that contradicts itself (:func:`check_recording_claims`,
:func:`check_summary_claims`).

Usage:
    python scripts/figures/generate_supp_memoryless_comparison.py
"""

import argparse
import json
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import SubplotSpec
from matplotlib.transforms import offset_copy

from continuum_swr_replay.analysis.detection.memoryless_sequence import (
    sweep_statistics,
    sweep_summaries,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.comparison_simulation import (
    CONFIG,
    FAR_REPLAY,
    FAR_REPLAY_EVENT,
    FAST_REPLAY,
    FAST_REPLAY_EVENT,
    FRAGMENTED_EVENT,
    N_NEURONS,
    REMOTE_STATIONARY_EVENT,
    REPLAY_EVENTS,
    SHORT_PATH,
    SHORT_PATH_EVENT,
    SHORT_REPLAY,
    SHORT_REPLAY_EVENT,
    SILENCE_EVENT,
    SWEEP_EVENT,
    simulate_and_decode,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.config import (
    MemorylessSequenceConfig,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.recording_statistics import (
    best_dispersion_cm,
    bound_error_ms,
    dominant_state,
    duration_ratio,
    duration_ratios,
    is_reported,
    recording_numbers,
    reported_overlapping,
    verdict,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.shuffle import (
    DEFAULT_N_SHUFFLES,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.sweep import overlaps
from continuum_swr_replay.runs.device import default_jax_platforms
from continuum_swr_replay.runs.provenance import described_commit, produced_by
from continuum_swr_replay.visualization.figure_utilities import (
    save_figure,
    set_figure_defaults,
)
from continuum_swr_replay.visualization.simulated_run_panels import (
    LABEL_SIZE,
    NON_LOCAL_COLOR,
    SMALL_SIZE,
    axes_added,
    check_journal_limits,
    drawn_extent,
    letter_panel,
    plot_posterior,
    plot_raster,
    posterior_color_limit,
    space_rows_evenly,
)

warnings.filterwarnings("ignore")

SUMMARY_PATH = Path("paper/reported/inputs/detection/detection_sweep_summary.csv")
"""Aggregates of the replay sweep, drawn in panels b-e."""

RECORDING_NUMBERS_PATH = Path(
    "paper/reported/inputs/detection/memoryless_comparison_recording.json"
)
"""The numbers reported about the one labeled recording (panel a)
(``recording_statistics.recording_numbers``), which this script writes; the
detection summary reads them."""

SLIDING = "Sliding window"
STATE_SPACE = "State space"
"""One name each. Neither says "our": a figure names a method, not a side."""

# The state-space model's events are Figure 1's non-local events, so they keep
# that colour. The sliding window's is no state's colour, nor the animal's.
SIMULATED_COLOR = "#222222"
SLIDING_COLOR = "#6a3d9a"
STATE_SPACE_COLOR = NON_LOCAL_COLOR
# Rows of the event axes, top to bottom.
EVENT_ROWS = (
    ("Simulated", SIMULATED_COLOR),
    (SLIDING, SLIDING_COLOR),
    (STATE_SPACE, STATE_SPACE_COLOR),
)

# Inches. Narrower than Figure 1's 6.9: the event rows' names reach the left edge,
# and the saved figure must stay within Cell Press's 17.4 cm.
FIGURE_WIDTH = 6.7
FIGURE_HEIGHT = 5.9
# The data axes of the panels that span the figure, in figure fractions: wider
# margins than Figure 1's, for the event rows' names.
DATA_LEFT, DATA_RIGHT = 0.12, 0.975
# The sweep's row needs less room on its left than the examples' row, whose
# first column carries the event rows' names.
SWEEP_LEFT = 0.085

CAPTION_SETTINGS = {
    "n_shuffles": 100,
    "window_ms": 80.0,
    "stride_ms": 5.0,
    "speed_threshold_cm_per_s": 4.0,
    "dispersion_threshold_cm": 12.0,
    "minimum_duration_ms": 100.0,
}
"""The sliding-window detector's settings that the legend states. They are
method, not results, so the legend types them; :func:`check_caption_settings`
stops if the code moves."""


def check_caption_settings(config: MemorylessSequenceConfig) -> None:
    """Stop when a setting the legend states is no longer the code's."""
    actual = {
        "n_shuffles": DEFAULT_N_SHUFFLES,
        "window_ms": 1000 * config.window_duration_s,
        "stride_ms": 1000 * config.stride_s,
        "speed_threshold_cm_per_s": config.speed_threshold_cm_per_s,
        "dispersion_threshold_cm": config.dispersion_threshold_cm,
        "minimum_duration_ms": 1000 * config.minimum_duration_s,
    }
    moved = {
        k: (v, actual[k])
        for k, v in CAPTION_SETTINGS.items()
        if not np.isclose(v, actual[k])
    }
    if moved:
        raise RuntimeError(
            f"The legend states settings the code no longer uses (legend, code): "
            f"{moved}. Rewrite the legend and CAPTION_SETTINGS."
        )


# ═══════════════════════════════════════════════════════════════════════════
# The examples of panel a
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Example:
    """One column of panel a."""

    event: int  # row of the recording's ``event_times``
    name: str
    simulated: str  # what the simulated spikes do
    expected: str  # the sliding-window detector's verdict the figure commits to
    group: str  # a key of EXAMPLE_GROUPS: the heading the column sits under


EXAMPLES = (
    Example(
        FAR_REPLAY_EVENT,
        "Distant",
        f"{FAR_REPLAY.path_cm:.0f} cm in {1000 * FAR_REPLAY.duration_s:.0f} ms",
        "found",
        "both",
    ),
    Example(SWEEP_EVENT, "At animal", "the whole track", "found", "both"),
    Example(
        SHORT_REPLAY_EVENT,
        "Brief",
        f"{1000 * SHORT_REPLAY.duration_s:.0f} ms",
        "found",
        "both",
    ),
    Example(
        SHORT_PATH_EVENT,
        "Short path",
        f"{SHORT_PATH.path_cm:.0f} cm",
        "declined",
        "state_space_only",
    ),
    Example(
        REMOTE_STATIONARY_EVENT,
        "Stationary",
        "one remote position",
        "declined",
        "state_space_only",
    ),
    Example(
        FAST_REPLAY_EVENT,
        "Fast",
        f"{FAST_REPLAY.path_cm:.0f} cm in {1000 * FAST_REPLAY.duration_s:.0f} ms",
        "declined",
        "state_space_only",
    ),
    Example(
        FRAGMENTED_EVENT,
        "Random order",
        "Fragmented state",
        "declined",
        "no_trajectory",
    ),
)
"""Panel a's columns, each a simulated event named for what sets it apart, under
the headings of :data:`EXAMPLE_GROUPS`. The sweep, the random-order event and the
place held still are Figure 1's examples; the others are added to its run. The
sparse replay is the far one at a lower peak firing rate."""

EXAMPLE_GROUPS = {
    "both": "Both detectors report",
    "state_space_only": "Only the state-space model reports",
    "no_trajectory": "Not a trajectory",
}
"""The heading over each run of columns. The first two hold replays: the ones the
sliding-window detector reports and the ones it does not. The last holds spikes
at scattered positions, which are not at the animal's position and are no
trajectory either: the state-space model reports them, in its Fragmented state,
which is how it tells them from a replay. :func:`check_recording_claims` stops
unless the detectors do what each heading says."""


def sliding_events(decoded: dict) -> list[tuple[float, float]]:
    """The sliding-window detector's events under the published definition.

    A candidate sequence that meets the duration and dispersion criteria and
    passes the shuffle, from its first window's midpoint to its last's.
    """
    midpoint_s, runs = decoded["midpoint_s"], decoded["candidates"].runs
    return [
        (float(midpoint_s[runs[i, 0]]), float(midpoint_s[runs[i, 1]]))
        for i in np.flatnonzero(is_reported(decoded))
    ]


def simulated_events(decoded: dict) -> list[tuple[float, float]]:
    """When simulated spiking represents a place other than the animal's: every
    labeled event but the silence, in time order."""
    return sorted(tuple(decoded["event_times"][event]) for event in REPLAY_EVENTS)


def check_recording_claims(decoded: dict, config: MemorylessSequenceConfig) -> None:
    """Stop rather than draw panels whose headers and framing the data contradict.

    The figure commits to the sliding-window detector's verdict on each example
    (:data:`EXAMPLES`), to the short path and the place held still being
    declined for too little dispersion, to the reports it does make being
    longer than the events, and to the state-space model reporting every
    simulated replay and nothing over the silence.
    """
    for example in EXAMPLES:
        got = verdict(decoded, example.event)
        if got != example.expected:
            raise RuntimeError(
                f"The figure says the sliding-window detector {example.expected} "
                f"the {example.name.lower()!r} example; the data say {got}."
            )
        reported = example.expected == "found"
        if reported != (example.group == "both"):
            raise RuntimeError(
                f"The {example.name.lower()!r} example sits under "
                f"{EXAMPLE_GROUPS[example.group]!r} but is expected {example.expected}."
            )
    if dominant_state(decoded, FRAGMENTED_EVENT) != "Non-Local Fragmented":
        raise RuntimeError(
            "The figure says the state-space model reports the random-order event in "
            "its Fragmented state; its most probable state there is "
            f"{dominant_state(decoded, FRAGMENTED_EVENT)!r}."
        )
    for event in (SHORT_PATH_EVENT, REMOTE_STATIONARY_EVENT):
        if not best_dispersion_cm(decoded, event) < config.dispersion_threshold_cm:
            raise RuntimeError(
                f"Event {event} is said to be declined for too little dispersion, "
                "but a candidate overlapping it clears the threshold."
            )
    for event in (SWEEP_EVENT, SHORT_REPLAY_EVENT, FAR_REPLAY_EVENT):
        if not duration_ratio(decoded, event) > 1.0:
            raise RuntimeError(
                f"The sliding window's report of event {event} is said to be too "
                "long; it is not."
            )
    if not np.isfinite(bound_error_ms(decoded, REPLAY_EVENTS)):
        raise RuntimeError(
            "The state-space model is said to report every simulated replay; it "
            "misses one."
        )
    silence = tuple(decoded["event_times"][SILENCE_EVENT])
    if any(
        overlaps(reported, silence)
        for reported in (*decoded["state_space_intervals"], *sliding_events(decoded))
    ):
        raise RuntimeError("A detector reports an event over the silence.")


# ═══════════════════════════════════════════════════════════════════════════
# Panel a — the labeled recording
# ═══════════════════════════════════════════════════════════════════════════


def plot_event_rows(
    ax: Axes,
    decoded: dict,
    *,
    scale: float,
    origin: float,
    labels: bool,
) -> None:
    """Three rows of bars: what was simulated and what each detector reports.

    ``scale`` and ``origin`` put the events, in seconds, on the axes' x
    coordinate: ``(time - origin) * scale``.
    """
    rows = (
        simulated_events(decoded),
        sliding_events(decoded),
        decoded["state_space_intervals"],
    )
    for row, (events, (_, color)) in enumerate(zip(rows, EVENT_ROWS, strict=True)):
        y = len(rows) - 1 - row
        ax.broken_barh(
            [
                ((start - origin) * scale, (end - start) * scale)
                for start, end in events
            ],
            (y - 0.3, 0.6),
            facecolors=color,
            # an outline, so that a brief event keeps a visible width
            edgecolors=color,
            linewidth=0.6,
        )
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_yticks([])
    if labels:
        ax.set_yticks(range(len(rows) - 1, -1, -1))
        ax.set_yticklabels([name for name, _ in EVENT_ROWS], fontsize=SMALL_SIZE)
        ax.tick_params(axis="y", length=0, pad=2)
        for label, (_, color) in zip(ax.get_yticklabels(), EVENT_ROWS, strict=True):
            label.set_color(color)
    ax.spines["left"].set_visible(False)


def plot_examples(
    fig: Figure,
    slot: SubplotSpec,
    decoded: dict,
    sliding_vmax: float,
    state_space_vmax: float,
) -> None:
    """Panel a: raster, each detector's decoded position, and the events.

    Each column is one example on its own millisecond axis, zero at the start
    of the simulated event. The sliding-window detector's decode is its
    posterior in each window, drawn at the window's midpoint; the state-space
    model's is its posterior in each decode bin.
    """
    gs = slot.subgridspec(
        4,
        len(EXAMPLES),
        height_ratios=[1.0, 1.0, 1.0, 0.5],
        hspace=0.16,
        wspace=0.14,
    )
    time, midpoint_s = decoded["time"], decoded["midpoint_s"]
    animal_at_midpoint = np.interp(midpoint_s, time, decoded["position"])
    rasters: list[Axes] = []
    for k, example in enumerate(EXAMPLES):
        event_start, event_end = decoded["event_times"][example.event]
        # room for the sliding window's reports of the event
        runs = decoded["candidates"].runs
        bounds = [
            (event_start, event_end),
            *[
                (midpoint_s[runs[i, 0]], midpoint_s[runs[i, 1]])
                for i in reported_overlapping(decoded, example.event)
            ],
        ]
        pad = 0.6 * (event_end - event_start) + 0.03
        start = min(a for a, _ in bounds) - pad
        end = max(b for _, b in bounds) + pad

        def to_ms(time_s, origin: float = event_start):
            """Seconds of the run to the column's milliseconds."""
            return (time_s - origin) * 1e3

        ax_raster = fig.add_subplot(gs[0, k])
        ax_sliding = fig.add_subplot(gs[1, k], sharex=ax_raster)
        ax_state_space = fig.add_subplot(gs[2, k], sharex=ax_raster)
        ax_events = fig.add_subplot(gs[3, k], sharex=ax_raster)
        axes = (ax_raster, ax_sliding, ax_state_space, ax_events)

        plot_raster(
            ax_raster,
            [
                to_ms(spikes[(spikes >= start) & (spikes <= end)])
                for spikes in decoded["spike_times"]
            ],
            linewidth=0.6,
        )
        # The title and what was simulated, a fixed distance above the raster
        # whatever its height.
        for above_pt, text, color, bold in (
            (16, example.name, SIMULATED_COLOR, True),
            (7, example.simulated, "#444444", False),
        ):
            ax_raster.text(
                0.5,
                1.0,
                text,
                transform=offset_copy(
                    ax_raster.transAxes, fig=fig, y=above_pt, units="points"
                ),
                fontsize=LABEL_SIZE if bold else SMALL_SIZE,
                fontweight="bold" if bold else "normal",
                color=color,
                ha="center",
                va="center",
            )
        rasters.append(ax_raster)

        windows = slice(*np.searchsorted(midpoint_s, [start, end]))
        plot_posterior(
            ax_sliding,
            decoded["posterior"][windows],
            decoded["place_bin_centers_cm"],
            to_ms(midpoint_s[windows]),
            animal_at_midpoint[windows],
            vmax=sliding_vmax,
            linewidth=0.8,
        )
        rows = slice(*np.searchsorted(time, [start, end]))
        plot_posterior(
            ax_state_space,
            decoded["state_space_posterior"][rows],
            decoded["state_space_position_bins"],
            to_ms(time[rows]),
            decoded["position"][rows],
            vmax=state_space_vmax,
            linewidth=0.8,
        )
        plot_event_rows(
            ax_events, decoded, scale=1e3, origin=event_start, labels=k == 0
        )
        # beside each report, how long it is against the simulated event
        reports = (sliding_events(decoded), decoded["state_space_intervals"])
        for ratio, events, y, color in zip(
            duration_ratios(decoded, example.event),
            reports,
            (1, 0),
            (SLIDING_COLOR, STATE_SPACE_COLOR),
            strict=True,
        ):
            if np.isfinite(ratio):
                last = max(
                    end
                    for start_, end in events
                    if overlaps((start_, end), (event_start, event_end))
                )
                ax_events.annotate(
                    f"{ratio:.1f}×",
                    (to_ms(last), y),
                    xytext=(2, 0),
                    textcoords="offset points",
                    fontsize=SMALL_SIZE,
                    color=color,
                    ha="left",
                    va="center",
                )
        ax_events.set_xlabel("Time (ms)", fontsize=LABEL_SIZE, labelpad=2)

        for ax in axes:
            ax.set_xlim(to_ms(start), to_ms(end))
            ax.tick_params(axis="x", labelsize=SMALL_SIZE, length=2)
        for ax in axes[:-1]:
            ax.tick_params(labelbottom=False, labelsize=SMALL_SIZE, length=2)
            if k > 0:
                ax.tick_params(labelleft=False)
        if k == 0:
            ax_raster.set_ylabel("Neuron", fontsize=LABEL_SIZE, labelpad=3)
            ax_sliding.set_ylabel(
                f"{SLIDING.replace(' ', chr(10))} (cm)",
                fontsize=LABEL_SIZE,
                labelpad=3,
                color=SLIDING_COLOR,
            )
            ax_state_space.set_ylabel(
                f"{STATE_SPACE.replace(' ', chr(10))} (cm)",
                fontsize=LABEL_SIZE,
                labelpad=3,
                color=STATE_SPACE_COLOR,
            )

    # A heading over each run of columns, with a rule as wide as the columns it
    # covers.
    groups = [example.group for example in EXAMPLES]
    for group, heading in EXAMPLE_GROUPS.items():
        columns = [k for k, name in enumerate(groups) if name == group]
        first, last = rasters[columns[0]], rasters[columns[-1]]
        left, right = first.get_position().x0, last.get_position().x1
        # the columns' span on the first one's own x axis, where 1 is its width
        span = (right - left) / first.get_position().width
        first.plot(
            [0, span],
            [1.0, 1.0],
            color=SIMULATED_COLOR,
            linewidth=0.6,
            transform=offset_copy(first.transAxes, fig=fig, y=27, units="points"),
            clip_on=False,
        )
        first.text(
            span / 2,
            1.0,
            heading,
            transform=offset_copy(first.transAxes, fig=fig, y=34, units="points"),
            fontsize=LABEL_SIZE,
            color=SIMULATED_COLOR,
            ha="center",
            va="center",
        )


# ═══════════════════════════════════════════════════════════════════════════
# Panels b-e — the sweep
# ═══════════════════════════════════════════════════════════════════════════


def detection_grid(summary: pd.DataFrame, measure: str) -> pd.DataFrame:
    """Detection rate as path length by duration."""
    rows = summary[
        (summary["summary"] == "detection_rate_by_path_and_duration")
        & (summary.measure == measure)
    ]
    return rows.pivot(index="path_cm", columns="duration_s", values="value")


def plot_detection(ax: Axes, grid: pd.DataFrame, title: str, color: str, show_y: bool):
    """Percent of the swept replays a detector reported, path by duration."""
    image = ax.imshow(
        100 * grid.to_numpy(),
        vmin=0,
        vmax=100,
        cmap="magma",
        aspect="auto",
        origin="lower",
        interpolation="nearest",
    )
    ax.set_xticks(range(grid.shape[1]))
    ax.set_xticklabels([f"{1000 * c:.0f}" for c in grid.columns], fontsize=SMALL_SIZE)
    ax.set_yticks(range(grid.shape[0]))
    ax.set_yticklabels([f"{r:.0f}" for r in grid.index], fontsize=SMALL_SIZE)
    ax.tick_params(length=2)
    ax.set_xlabel("Duration (ms)", fontsize=LABEL_SIZE, labelpad=2)
    if show_y:
        ax.set_ylabel("Path length (cm)", fontsize=LABEL_SIZE, labelpad=3)
    else:
        ax.tick_params(labelleft=False)
    ax.set_title(title, fontsize=LABEL_SIZE, fontweight="bold", color=color, pad=3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)
    for row in range(grid.shape[0]):
        for column in range(grid.shape[1]):
            percent = 100 * grid.iat[row, column]
            ax.text(
                column,
                row,
                f"{percent:.0f}",
                ha="center",
                va="center",
                fontsize=SMALL_SIZE,
                color="white" if percent < 55 else "black",
            )
    return image


def plot_edge_error(ax: Axes, summary: pd.DataFrame, edge: str) -> None:
    """How far off each detector's onset or offset is, for replays both report.

    The share of replays whose absolute error, against the replay's own first
    or last spike, is beyond each size: a curve's height at 50% is its median,
    and its right end is its tail.
    """
    shares = sweep_statistics.sweep_timing_exceedance(summary)
    # a log axis has no zero: the curves start at the smallest size above it
    shares = shares[shares.index > 0]
    ax.axvline(
        sweep_summaries.LARGE_TIMING_ERROR_MS,
        color="#999999",
        linewidth=0.6,
        linestyle=":",
    )
    for detector, color, label in (
        ("sliding", SLIDING_COLOR, SLIDING),
        ("state_space", STATE_SPACE_COLOR, STATE_SPACE),
    ):
        ax.step(
            shares.index.to_numpy(),
            100 * shares[f"{detector}_{edge}"].to_numpy(),
            where="post",
            color=color,
            linewidth=1.0,
            label=label,
        )
    ax.set_xscale("log")
    ax.set_xlim(shares.index[0], shares.index[-1])
    # 1000 ms is the axis's end; unlabeled, it leaves room for the next panel
    ax.set_xticks([1, 10, 100, 1000], ["1", "10", "100", ""])
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 50, 100])
    ax.tick_params(labelsize=SMALL_SIZE, length=2)
    ax.tick_params(which="minor", length=1)
    # on two lines: the two parts' labels would meet on one
    ax.set_xlabel(
        f"{'Onset' if edge == 'start' else 'Offset'}\nerror (ms)",
        fontsize=LABEL_SIZE,
        labelpad=2,
    )


def plot_position_error(ax: Axes, summary: pd.DataFrame) -> None:
    """How far each detector's decoded position is from the replay's true
    position, against the replay's speed, for replays both detectors report.

    Each point is the median, over the replays of a speed bin, of a report's
    own median distance.
    """
    by_speed = sweep_statistics.sweep_position_by_speed(summary)
    edges = [*by_speed.index, np.inf]
    labels = [
        f">{low:g}" if np.isinf(high) else f"{low:g}–{high:g}"
        for low, high in zip(edges[:-1], edges[1:], strict=True)
    ]
    x = np.arange(len(by_speed))
    for detector, color in (
        ("sliding", SLIDING_COLOR),
        ("state_space", STATE_SPACE_COLOR),
    ):
        ax.plot(
            x,
            by_speed[f"{detector}_position_error_cm"].to_numpy(),
            color=color,
            linewidth=1.0,
            marker="o",
            markersize=2.5,
        )
    ax.set_xticks(x, labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylim(0, None)
    ax.tick_params(labelsize=SMALL_SIZE, length=2)
    ax.set_xlabel("Replay speed (m/s)", fontsize=LABEL_SIZE, labelpad=2)
    ax.set_ylabel("Position\nerror (cm)", fontsize=LABEL_SIZE, labelpad=3)


def plot_sweep(fig: Figure, slot: SubplotSpec, summary: pd.DataFrame) -> None:
    """Panels b-d: detection by path and duration, how far off onset and offset
    are, and how far off the decoded position is."""
    gs = slot.subgridspec(
        1, 5, width_ratios=[0.86, 0.86, 0.035, 0.55, 1.9], wspace=0.12
    )
    ax_sliding = fig.add_subplot(gs[0])
    ax_state_space = fig.add_subplot(gs[1])
    image = plot_detection(
        ax_sliding,
        detection_grid(summary, "sliding_matches"),
        SLIDING,
        SLIDING_COLOR,
        show_y=True,
    )
    plot_detection(
        ax_state_space,
        detection_grid(summary, "state_space_matches"),
        STATE_SPACE,
        STATE_SPACE_COLOR,
        show_y=False,
    )
    colorbar = fig.colorbar(image, cax=fig.add_subplot(gs[2]))
    colorbar.set_ticks([0, 50, 100])
    colorbar.ax.tick_params(labelsize=SMALL_SIZE, length=2, pad=1)
    colorbar.set_label("Replays reported (%)", fontsize=SMALL_SIZE, labelpad=2)
    colorbar.outline.set_linewidth(0.5)

    # onset and offset are one panel in two parts: they share a y axis and a letter
    right = gs[4].subgridspec(1, 2, width_ratios=[2.0, 1.0], wspace=0.62)
    timing = right[0].subgridspec(1, 2, wspace=0.25)
    ax_onset = fig.add_subplot(timing[0])
    ax_offset = fig.add_subplot(timing[1])
    plot_position_error(fig.add_subplot(right[1]), summary)
    plot_edge_error(ax_onset, summary, "start")
    plot_edge_error(ax_offset, summary, "end")
    ax_onset.set_ylabel(
        "Replays with a\nlarger error (%)", fontsize=LABEL_SIZE, labelpad=3
    )
    ax_offset.tick_params(labelleft=False)
    # Each curve is named once, where the space beside it is empty, in the
    # colours of the two detection grids.
    for ax, label, color, x, y in (
        (ax_offset, SLIDING, SLIDING_COLOR, 130, 50),
        (ax_onset, STATE_SPACE, STATE_SPACE_COLOR, 1.25, 16),
    ):
        ax.text(
            x,
            y,
            label.replace(" ", "\n"),
            color=color,
            fontsize=SMALL_SIZE,
            ha="left",
            va="center",
            linespacing=0.95,
        )


def check_summary_claims(summary: pd.DataFrame) -> None:
    """The figure's framing of the sweep, checked against the summary.

    Companion to :func:`check_recording_claims`. The figure says the
    sliding-window detector reports no replay that is held still, that the
    state-space model reports most of them, and that the sliding window starts
    its events early and ends them late where the state-space model does not.
    """
    held_still = {
        measure: detection_grid(summary, measure).loc[0.0]
        for measure in ("sliding_matches", "state_space_matches")
    }
    if (held_still["sliding_matches"] > 0).any():
        raise RuntimeError(
            "The figure says the sliding-window detector reports no replay held "
            "still; the sweep has one."
        )
    if not (held_still["state_space_matches"] > 0.5).all():
        raise RuntimeError(
            "The figure says the state-space model reports most replays held "
            "still at every duration; the sweep disagrees."
        )
    pooled = sweep_statistics.sweep_pooled_timing_statistics(summary)
    early_and_late = (
        pooled["sliding_start_error_ms_median"]
        < pooled["state_space_start_error_ms_median"]
        < 0
        < pooled["state_space_end_error_ms_median"]
        < pooled["sliding_end_error_ms_median"]
    )
    longer_tail = all(
        pooled[f"sliding_{edge}_error_ms_far_share"]
        > pooled[f"state_space_{edge}_error_ms_far_share"]
        for edge in ("start", "end")
    )
    if not (early_and_late and longer_tail):
        raise RuntimeError(
            "The figure contrasts the sliding window's early starts, late ends and "
            "long tail of timing errors with the state-space model's; the sweep's "
            "pooled errors do not differ that way."
        )


# ═══════════════════════════════════════════════════════════════════════════
# Main Figure Assembly
# ═══════════════════════════════════════════════════════════════════════════


def make_figure(summary: pd.DataFrame) -> tuple[Figure, dict]:
    """Decode the labeled recording and draw the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    decoded : dict
        ``comparison_simulation.simulate_and_decode``'s.
    """
    decoded = simulate_and_decode(N_NEURONS, CONFIG)
    check_recording_claims(decoded, CONFIG)
    state_space_vmax = posterior_color_limit(
        decoded["state_space_posterior"],
        decoded["time"],
        decoded["state_space_intervals"],
    )
    sliding_vmax = posterior_color_limit(
        decoded["posterior"], decoded["midpoint_s"], simulated_events(decoded)
    )

    set_figure_defaults(context="paper")
    # The rows are placed by hand below, from what each one draws, so the
    # automatic layout the paper defaults switch on must be off.
    fig = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), dpi=450, layout="none")

    # Provisional vertical positions; the rows are re-spaced below.
    gs_examples = fig.add_gridspec(
        1, 1, left=DATA_LEFT, right=DATA_RIGHT, top=0.90, bottom=0.42
    )
    gs_sweep = fig.add_gridspec(
        1, 1, left=SWEEP_LEFT, right=DATA_RIGHT, top=0.30, bottom=0.07
    )
    axes_examples = axes_added(
        fig,
        lambda: plot_examples(
            fig, gs_examples[0], decoded, sliding_vmax, state_space_vmax
        ),
    )
    axes_sweep = axes_added(fig, lambda: plot_sweep(fig, gs_sweep[0], summary))
    n_axes = 4  # an example's axes
    fig.align_ylabels(axes_examples[:n_axes])
    fig.draw_without_rendering()

    extents = space_rows_evenly(fig, [axes_examples, axes_sweep])

    # Panel letters: a and b share a left edge; the letters of the sweep's row
    # share its top edge. The two grids are one panel, as are onset and offset.
    left = min(extent.x0 for extent in extents)
    _, _, _, ax_onset, _, ax_position = axes_sweep
    for letter, x, y in (
        ("a", left, extents[0].y1),
        ("b", left, extents[1].y1),
        ("c", drawn_extent(fig, [ax_onset]).x0, extents[1].y1),
        ("d", drawn_extent(fig, [ax_position]).x0, extents[1].y1),
    ):
        letter_panel(fig, letter, x, y)
    return fig, decoded


def write_recording_numbers(numbers: dict, n_neurons: int, path: Path) -> None:
    """Save ``recording_numbers`` with the provenance the sweeps' CSVs carry."""
    path.write_text(
        json.dumps(
            {
                "produced_by": produced_by(__file__),
                "produced_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "git_commit": described_commit(),
                "n_neurons": n_neurons,
                "numbers": numbers,
            },
            indent=1,
            allow_nan=False,
        )
        + "\n"
    )


def main(argv: Sequence[str] | None = None) -> None:
    # the CPU unless JAX_PLATFORMS names another
    default_jax_platforms("cpu")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/figures/supp_memoryless_comparison"),
    )
    parser.add_argument(
        "--recording-numbers", type=Path, default=RECORDING_NUMBERS_PATH
    )
    args = parser.parse_args(argv)

    if not args.summary.exists():
        raise FileNotFoundError(
            f"No summary at {args.summary}. Run "
            "`python scripts/detection/sweep_memoryless_detection.py` first."
        )
    summary = pd.read_csv(args.summary)
    check_caption_settings(CONFIG)
    check_summary_claims(summary)
    fig, decoded = make_figure(summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    check_journal_limits(fig)
    save_figure(fig, "supp_memoryless_comparison", base_dir=args.output_dir)
    plt.close(fig)
    write_recording_numbers(
        recording_numbers(decoded), N_NEURONS, args.recording_numbers
    )
    print(
        f"saved to {args.output_dir}; the recording's numbers to "
        f"{args.recording_numbers}"
    )


if __name__ == "__main__":
    main()
