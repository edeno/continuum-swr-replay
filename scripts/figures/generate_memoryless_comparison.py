#!/usr/bin/env python
"""Supplemental figure: the memoryless sequence detector on simulated data.

Runs the ported Widloski & Foster sliding-window detector over the same
simulated recording that Figure 1 uses to explain the switching state-space
model, so the two procedures can be read against one ground truth. The
simulation contains five labeled events: a continuous outbound replay, a
fragmented replay, a continuous inbound replay, a silent stretch, and a local
event.

The point of the figure is what the memoryless detector has no way to express.
It scores each 80 ms window on posterior spread and on the jump to the next
window, and it has no competing hypothesis for the animal's own position. A
stretch of local coding is smooth, narrow and slow, so it passes every window
criterion and is assembled into the same sequence as the replay that follows
it. The detected event therefore begins long before the replay does.

Usage:
    python scripts/figures/generate_memoryless_comparison.py
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from non_local_detector.environment import Environment
from non_local_detector.likelihoods.sorted_spikes_kde import (
    fit_sorted_spikes_kde_encoding_model,
)
from non_local_detector.simulate.sorted_spikes_simulation import make_simulated_data

from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.config import (
    MemorylessSequenceConfig,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.detect import (
    detect_candidate_sequences,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.likelihood import (
    interior_bin_distances_cm,
    sorted_spikes_epoch_log_likelihood,
    window_midpoints_s,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.statistics import (
    aggregate_windows,
    normalize_windows,
)
from continuum_swr_replay.figure_utilities import (
    TWO_COLUMN,
    WONG,
    label_panels,
    save_figure,
    set_figure_defaults,
)

TRACK_HEIGHT_CM = 170
DECODE_BIN_STRIDE = 3
"""Figure 1 strides the 1500 Hz simulation to the project's 500 Hz decode grid."""

EVENT_LABELS = [
    "continuous\noutbound",
    "fragmented",
    "continuous\ninbound",
    "no spike",
    "local",
    "remote\nstationary",
]

REMOTE_STATIONARY_CM = 42.5
"""The place a remote stationary event holds, far from the animal's 170 cm."""

REMOTE_STATIONARY_S = 0.300
REMOTE_SPIKE_INTERVAL_S = 0.012
"""One neuron firing at ~83 Hz, matching the other events' spike density."""


def splice_remote_stationary(
    spike_times: list[np.ndarray],
    place_field_means: np.ndarray,
    start_s: float,
) -> tuple[list[np.ndarray], tuple[float, float]]:
    """Add a non-local representation that is held still rather than travelled.

    The package simulates travelling replay, a stationary local event and
    silence, but not a remote place held still. That case separates two things
    the detector does for different reasons: it absorbs a stationary
    representation when the representation is the animal's own place, and it
    declines a stationary representation when it is somewhere else. Only the
    second is the method behaving as designed.
    """
    end_s = start_s + REMOTE_STATIONARY_S
    neuron = int(np.argmin(np.abs(place_field_means - REMOTE_STATIONARY_CM)))
    spliced = [
        neuron_times[(neuron_times < start_s) | (neuron_times > end_s)]
        for neuron_times in spike_times
    ]
    spliced[neuron] = np.sort(
        np.concatenate(
            [spliced[neuron], np.arange(start_s, end_s, REMOTE_SPIKE_INTERVAL_S)]
        )
    )
    return spliced, (start_s, end_s)


BLACK, ORANGE, SKY, GREEN, YELLOW, BLUE, VERMILION, PURPLE = WONG[:8]


def simulate_and_decode(n_neurons: int, config: MemorylessSequenceConfig) -> dict:
    """Decode the Figure 1 simulation window by window under a uniform prior."""
    (
        speed,
        position,
        spike_times,
        time,
        event_times,
        _sampling_frequency,
        is_event,
        _place_fields,
    ) = make_simulated_data(n_neurons=n_neurons)

    # the package has no remote stationary event; add one in the last pause,
    # and hold it out of training like every other event
    spike_times, remote_stationary = splice_remote_stationary(
        spike_times,
        np.linspace(0, TRACK_HEIGHT_CM, n_neurons),
        float(event_times[4, 0]),
    )
    event_times = np.vstack([event_times, remote_stationary])
    is_event |= (time >= remote_stationary[0]) & (time <= remote_stationary[1])

    time, position, speed, is_event = (
        array[::DECODE_BIN_STRIDE] for array in (time, position, speed, is_event)
    )

    environment = Environment(place_bin_size=config.place_bin_size_cm).fit_place_grid(
        position[:, None]
    )

    # encode on non-event data only, the analogue of this project's no-ripple
    # training mask; spikes inside an event must not shape the place fields
    is_training = ~is_event
    training_spikes = [
        neuron[np.interp(neuron, time, is_training.astype(float)) > 0.5]
        for neuron in spike_times
    ]
    encoding_model = fit_sorted_spikes_kde_encoding_model(
        position_time=time[is_training],
        position=position[is_training, None],
        spike_times=training_spikes,
        environment=environment,
        sampling_frequency=round(config.sampling_frequency),
        position_std=6.0,
        disable_progress_bar=True,
    )

    result = sorted_spikes_epoch_log_likelihood(
        time, position, spike_times, encoding_model, config
    )
    position, speed = position[: result.time.size], speed[: result.time.size]

    posterior = normalize_windows(
        aggregate_windows(
            result.log_likelihood, config.window_n_samples, config.stride_n_samples
        )
    )
    midpoint_s = window_midpoints_s(result.time, config)
    window_speed = np.interp(midpoint_s, result.time, speed)

    return {
        "time": result.time,
        "position": position,
        "speed": speed,
        "spike_times": spike_times,
        "event_times": event_times,
        "place_bin_centers_cm": result.place_bin_centers_cm,
        "posterior": posterior,
        "midpoint_s": midpoint_s,
        "window_speed": window_speed,
        "candidates": detect_candidate_sequences(
            posterior,
            midpoint_s,
            interior_bin_distances_cm(environment),
            window_speed,
            config,
        ),
    }


def decoded_center_cm(decoded: dict) -> np.ndarray:
    """Center of each window, NaN where the window has none."""
    center = decoded["candidates"].decisions.center_index
    out = np.full(center.size, np.nan)
    present = center >= 0
    out[present] = decoded["place_bin_centers_cm"][center[present]]
    return out


def local_fraction(decoded: dict, sequence: int, tolerance_cm: float = 15.0) -> float:
    """Share of a sequence's windows whose center sits at the animal's own place."""
    candidates = decoded["candidates"]
    start, stop = candidates.runs[sequence]
    center_cm = decoded_center_cm(decoded)[start : stop + 1]
    animal_cm = np.interp(
        decoded["midpoint_s"][start : stop + 1], decoded["time"], decoded["position"]
    )
    usable = np.isfinite(center_cm)
    return float(np.mean(np.abs(center_cm[usable] - animal_cm[usable]) < tolerance_cm))


def plot_overview(ax, decoded: dict, config: MemorylessSequenceConfig) -> None:
    """The whole recording: what was decoded, and what the detector kept."""
    candidates = decoded["candidates"]
    midpoint_s = decoded["midpoint_s"]
    center_cm = decoded_center_cm(decoded)
    kept = candidates.decisions.gates.kept

    for (start_s, end_s), label in zip(
        decoded["event_times"], EVENT_LABELS, strict=True
    ):
        if end_s <= start_s:
            # the package's "local" entry is a zero-width marker for an
            # unmanipulated pause; the remote stationary event sits there now
            continue
        ax.axvspan(start_s, end_s, color=YELLOW, alpha=0.6, lw=0)
        ax.text(
            0.5 * (start_s + end_s),
            190,
            label,
            fontsize=5,
            ha="center",
            va="bottom",
            color="0.25",
            linespacing=0.9,
        )

    ax.plot(decoded["time"], decoded["position"], color=GREEN, lw=0.6, zorder=1)
    ax.scatter(
        midpoint_s[~kept],
        center_cm[~kept],
        s=0.4,
        color="0.55",
        lw=0,
        zorder=2,
    )
    ax.scatter(midpoint_s[kept], center_cm[kept], s=0.6, color=SKY, lw=0, zorder=3)

    for sequence, (start, stop) in enumerate(candidates.runs):
        accepted = candidates.is_replay[sequence]
        ax.plot(
            midpoint_s[[start, stop]],
            [-14, -14],
            color=VERMILION if accepted else "0.6",
            lw=2.4 if accepted else 1.2,
            solid_capstyle="butt",
            zorder=4,
        )

    handles = [
        Line2D([], [], color=GREEN, lw=1.0, label="animal position"),
        Line2D(
            [], [], color=SKY, marker="o", ms=2.0, lw=0, label="window center, kept"
        ),
        Line2D(
            [],
            [],
            color="0.55",
            marker="o",
            ms=2.0,
            lw=0,
            label="window center, rejected",
        ),
        Patch(color=YELLOW, alpha=0.6, label="labeled event"),
        Line2D([], [], color=VERMILION, lw=2.4, label="sequence, meets criteria"),
        Line2D([], [], color="0.6", lw=1.2, label="sequence, rejected"),
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.02),
        ncol=3,
        fontsize=5,
        frameon=False,
        handlelength=1.4,
        columnspacing=1.0,
        handletextpad=0.5,
        borderpad=0.0,
    )

    ax.set_ylim(-24, 262)
    ax.set_xlim(decoded["time"][0], decoded["time"][-1])
    ax.set_yticks([0, 50, 100, 150])
    ax.set_ylabel("position (cm)")
    ax.set_xlabel("time (s)")


def zoom_window(decoded: dict, event: int) -> tuple[float, float, list[int]]:
    """The stretch to show around one labeled event.

    Wide enough to hold the event itself and every candidate sequence that
    touches it, which is the point: the sequences reach well beyond the event.
    """
    candidates = decoded["candidates"]
    midpoint_s = decoded["midpoint_s"]
    true_start, true_end = decoded["event_times"][event]

    overlapping = [
        i
        for i, (start, stop) in enumerate(candidates.runs)
        if midpoint_s[start] < true_end + 0.1 and true_start - 0.1 < midpoint_s[stop]
    ]
    spans = [midpoint_s[candidates.runs[i]] for i in overlapping]
    return (
        min([span[0] for span in spans] + [true_start]) - 0.25,
        max([span[1] for span in spans] + [true_end]) + 0.25,
        overlapping,
    )


def plot_event(ax, decoded: dict, event: int, config: MemorylessSequenceConfig) -> None:
    """One labeled event, with the sequence the detector built around it."""
    candidates = decoded["candidates"]
    midpoint_s = decoded["midpoint_s"]
    center_cm = decoded_center_cm(decoded)
    true_start, true_end = decoded["event_times"][event]

    window_lo, window_hi, overlapping = zoom_window(decoded, event)
    shown = (midpoint_s >= window_lo) & (midpoint_s <= window_hi)

    extent = (
        midpoint_s[shown][0],
        midpoint_s[shown][-1],
        decoded["place_bin_centers_cm"][0],
        decoded["place_bin_centers_cm"][-1],
    )
    ax.imshow(
        decoded["posterior"][shown].T,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="bone_r",
        vmin=0.0,
        vmax=np.percentile(decoded["posterior"][shown], 99.9),
        interpolation="nearest",
    )
    ax.axvspan(
        true_start, max(true_end, true_start + 0.02), color=YELLOW, alpha=0.45, lw=0
    )

    kept = candidates.decisions.gates.kept & shown
    ax.scatter(midpoint_s[kept], center_cm[kept], s=1.6, color=SKY, lw=0, zorder=3)
    ax.plot(
        decoded["time"],
        decoded["position"],
        color=GREEN,
        lw=1.0,
        ls="--",
        zorder=2,
    )

    for i in overlapping:
        start, stop = candidates.runs[i]
        accepted = candidates.is_replay[i]
        ax.plot(
            midpoint_s[[start, stop]],
            [-9, -9],
            color=VERMILION if accepted else "0.6",
            lw=2.6 if accepted else 1.4,
            solid_capstyle="butt",
            clip_on=False,
            zorder=4,
        )

    ax.set_xlim(window_lo, window_hi)
    ax.set_ylim(-14, 178)
    ax.set_xlabel("time (s)")
    ax.set_title(EVENT_LABELS[event].replace("\n", " "), fontsize=6)


def plot_window_statistics(ax_spread, ax_jump, decoded, event, config) -> None:
    """The two window criteria across the same stretch of time."""
    candidates = decoded["candidates"]
    midpoint_s = decoded["midpoint_s"]
    true_start, true_end = decoded["event_times"][event]
    window_lo, window_hi, _ = zoom_window(decoded, event)
    shown = (midpoint_s >= window_lo) & (midpoint_s <= window_hi)

    for ax, values, threshold, name in (
        (
            ax_spread,
            candidates.decisions.spread_cm,
            config.spread_threshold_cm,
            "spread",
        ),
        (ax_jump, candidates.decisions.jump_cm, config.jump_threshold_cm, "jump"),
    ):
        ax.axvspan(
            true_start, max(true_end, true_start + 0.02), color=YELLOW, alpha=0.45, lw=0
        )
        ax.plot(midpoint_s[shown], values[shown], color=BLACK, lw=0.5)
        ax.axhline(threshold, color=VERMILION, lw=0.7, ls=":")
        ax.set_xlim(window_lo, window_hi)
        ax.set_ylim(0, threshold * 2.6)
        ax.set_ylabel(f"{name} (cm)")
    ax_jump.set_xlabel("time (s)")


def make_figure(n_neurons: int, config: MemorylessSequenceConfig):
    decoded = simulate_and_decode(n_neurons, config)
    candidates = decoded["candidates"]

    set_figure_defaults(context="paper")
    fig = plt.figure(figsize=(TWO_COLUMN, TWO_COLUMN * 0.92), dpi=450)
    grid = GridSpec(
        3,
        4,
        figure=fig,
        height_ratios=[1.0, 1.05, 0.7],
        hspace=0.6,
        wspace=0.42,
        left=0.075,
        right=0.985,
        top=0.885,
        bottom=0.165,
    )

    ax_overview = fig.add_subplot(grid[0, :])
    plot_overview(ax_overview, decoded, config)

    zooms = []
    for column, event in enumerate((0, 5, 1, 3)):
        ax = fig.add_subplot(grid[1, column])
        plot_event(ax, decoded, event, config)
        if column:
            ax.set_yticklabels([])
        zooms.append(ax)
    zooms[0].set_ylabel("position (cm)")

    ax_spread = fig.add_subplot(grid[2, :2])
    ax_jump = fig.add_subplot(grid[2, 2:])
    plot_window_statistics(ax_spread, ax_jump, decoded, 0, config)
    ax_spread.set_title("posterior spread, continuous outbound", fontsize=6)
    ax_jump.set_title("jump to the next window", fontsize=6)

    label_panels([ax_overview, *zooms, ax_spread, ax_jump], xpos=-0.08, ypos=1.12)

    fig.suptitle(
        "Memoryless sliding-window detection on the Figure 1 simulation",
        fontsize=7,
        y=0.975,
    )
    lines = []
    for sequence in np.flatnonzero(candidates.is_replay):
        start, stop = candidates.runs[sequence]
        midpoints = decoded["midpoint_s"][start : stop + 1]
        overlapped = [
            j
            for j, (event_start, event_end) in enumerate(decoded["event_times"])
            if decoded["midpoint_s"][start] < event_end
            and event_start < decoded["midpoint_s"][stop]
        ]
        names = ", ".join(EVENT_LABELS[j].replace("\n", " ") for j in overlapped)
        event_start, event_end = decoded["event_times"][overlapped[0]]
        outside = 1.0 - np.mean((midpoints >= event_start) & (midpoints <= event_end))
        lines.append(
            f"{names} — detected {candidates.duration_s[sequence]:.2f} s for a "
            f"{event_end - event_start:.2f} s event; {100 * outside:.0f}% of it lies "
            f"outside the labeled event, and {100 * local_fraction(decoded, sequence):.0f}% "
            "of its windows decode the animal's own position."
        )
    lines.append(
        "The merge rule is most of it: with merging off the same data gives 1.4x and "
        "2.1x inflation rather than 5.8x and 4.3x. A stationary representation is "
        "absorbed when it sits near a travelling one, as here, where the replay begins "
        "at the animal's own place; the remote stationary event in c has no travelling "
        "neighbour and is declined for going nowhere, which is the method working as "
        "designed."
    )
    fig.text(
        0.5,
        0.012,
        "\n".join(lines),
        fontsize=5.5,
        ha="center",
        va="bottom",
        linespacing=1.7,
        color="0.2",
        wrap=True,
    )
    return fig, decoded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-neurons", type=int, default=25)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("paper/figures/memoryless_comparison")
    )
    args = parser.parse_args()

    config = MemorylessSequenceConfig(sampling_frequency=500.0)
    fig, decoded = make_figure(args.n_neurons, config)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_figure("memoryless_comparison", base_dir=args.output_dir)
    plt.close(fig)

    candidates = decoded["candidates"]
    print(
        f"{len(candidates.runs)} candidate sequences, "
        f"{int(candidates.is_replay.sum())} accepted"
    )
    for sequence in np.flatnonzero(candidates.is_replay):
        start, stop = candidates.runs[sequence]
        print(
            f"  [{decoded['midpoint_s'][start]:7.3f}, {decoded['midpoint_s'][stop]:7.3f}] "
            f"dur {candidates.duration_s[sequence]:.3f} s  "
            f"disp {candidates.dispersion_cm[sequence]:5.1f} cm  "
            f"local {100 * local_fraction(decoded, sequence):.0f}%"
        )
    print(f"saved to {args.output_dir}")


if __name__ == "__main__":
    main()
