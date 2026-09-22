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

Both detectors fit their own encoding model, from the same recording, the same
training mask and the same hyperparameters. They are separate fits, and the
state-space model's call re-estimates its place fields as well, so the two are
close but not one shared object.

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
from non_local_detector.simulate.sorted_spikes_simulation import (
    SAMPLING_FREQUENCY,
    make_simulated_data,
)

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
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.simulation import (
    inject_replay,
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
from continuum_swr_replay.parameters import LOCAL_POSITION_STD

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
    "short path\n30 cm",
    "short replay\n80 ms",
]

SHORT_REPLAY = (35.0, 100.0, 0.080, 120.0, 44.40)
"""start_cm, path_cm, duration_s, peak_rate_hz, time.

A replay that travels far but lasts only 80 ms, firing strongly enough that
spike count is not the issue. The detector finds it and reports it too long;
the panel title reads the ratio off the data rather than stating it here. The
window puts a floor under any duration the method can report.
"""

SHORT_PATH = (40.0, 30.0, 0.300, 40.0, 20.50)
"""start_cm, path_cm, duration_s, peak_rate_hz, time.

A replay that travels only 30 cm. The detector decodes it well, but the
dispersion criterion needs a path of about 42 cm, so it rejects the candidate.
"""

NON_LOCAL_PROB_THRESHOLD = 0.8
DISCRETE_TRANSITION_STICKINESS = np.array([30.0, 100000.0, 30.0, 200.0])
"""Settings Figure 1 uses for the switching state-space model."""

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

    # two replays the detector should find but does not
    place_field_means = np.linspace(0, TRACK_HEIGHT_CM, n_neurons)

    start_cm, path_cm, duration_s, rate_hz, at_s = SHORT_PATH
    spike_times, bounds = inject_replay(
        spike_times,
        place_field_means,
        start_cm,
        path_cm,
        duration_s,
        rate_hz,
        at_s,
        sampling_frequency=SAMPLING_FREQUENCY,
        seed=0,
    )
    event_times = np.vstack([event_times, bounds])
    is_event |= (time >= bounds[0]) & (time <= bounds[1])

    start_cm, path_cm, duration_s, rate_hz, at_s = SHORT_REPLAY
    spike_times, bounds = inject_replay(
        spike_times,
        place_field_means,
        start_cm,
        path_cm,
        duration_s,
        rate_hz,
        at_s,
        sampling_frequency=SAMPLING_FREQUENCY,
        seed=1,
    )
    event_times = np.vstack([event_times, bounds])
    is_event |= (time >= bounds[0]) & (time <= bounds[1])

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
        "is_event": is_event[: result.time.size],
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


def state_space_intervals(decoded: dict) -> list[tuple[float, float]]:
    """When the switching state-space model calls the representation non-local.

    The comparison needs this. Every event the sliding-window detector misses
    has to be an event some detector can find, or the figure shows the limits
    of the data rather than the limits of the method.
    """
    from non_local_detector import NonLocalSortedSpikesDetector

    time, is_event = decoded["time"], decoded["is_event"]
    detector = NonLocalSortedSpikesDetector(
        sorted_spikes_algorithm="sorted_spikes_kde",
        sorted_spikes_algorithm_params={"position_std": 6.0, "block_size": int(2**8)},
        local_position_std=LOCAL_POSITION_STD,
        discrete_transition_stickiness=DISCRETE_TRANSITION_STICKINESS,
    )
    results = detector.estimate_parameters(
        position_time=time,
        position=decoded["position"],
        spike_times=decoded["spike_times"],
        time=time,
        is_training=~is_event,
    )
    probability = np.asarray(
        results.acausal_state_probabilities.sel(
            states=["Non-Local Continuous", "Non-Local Fragmented"]
        )
        .sum("states")
        .values
    ).squeeze()

    above = probability > NON_LOCAL_PROB_THRESHOLD
    edges = np.diff(above.astype(np.int8), prepend=0, append=0)
    return [
        (time[start], time[stop])
        for start, stop in zip(
            np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1, strict=True
        )
    ]


def duration_ratio(decoded: dict, event: int) -> float | None:
    """How much longer the detector calls an event than the event really is.

    Read from the data, so a panel title cannot drift away from the panel.
    """
    candidates = decoded["candidates"]
    event_start, event_end = decoded["event_times"][event]
    for i in np.flatnonzero(candidates.is_replay):
        start, stop = candidates.runs[i]
        if (
            decoded["midpoint_s"][start] < event_end
            and event_start < decoded["midpoint_s"][stop]
        ):
            return float(candidates.duration_s[i] / (event_end - event_start))
    return None


def decoded_center_cm(decoded: dict) -> np.ndarray:
    """Center of each window, NaN where the window has none."""
    center = decoded["candidates"].decisions.center_index
    out = np.full(center.size, np.nan)
    present = center >= 0
    out[present] = decoded["place_bin_centers_cm"][center[present]]
    return out


def local_fraction_of_event(decoded: dict, event: int) -> float:
    """Share of the accepted event overlapping ``event`` that is local coding."""
    candidates = decoded["candidates"]
    event_start, event_end = decoded["event_times"][event]
    for i in np.flatnonzero(candidates.is_replay):
        start, stop = candidates.runs[i]
        if (
            decoded["midpoint_s"][start] < event_end
            and event_start < decoded["midpoint_s"][stop]
        ):
            return local_fraction(decoded, i)
    return float("nan")


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

    # the package's "local" entry is a zero-width marker for an unmanipulated
    # pause; the remote stationary event sits there now
    labelled = [
        (start_s, end_s, label)
        for (start_s, end_s), label in zip(
            decoded["event_times"], EVENT_LABELS, strict=True
        )
        if end_s > start_s
    ]
    # two rows, alternating in time order, so neighbours never share a row
    for row, (start_s, end_s, label) in enumerate(sorted(labelled)):
        ax.axvspan(start_s, end_s, color=YELLOW, alpha=0.6, lw=0)
        ax.text(
            0.5 * (start_s + end_s),
            190 if row % 2 else 234,
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

    for start_s, end_s in decoded["state_space_intervals"]:
        ax.plot(
            [start_s, end_s],
            [-22, -22],
            color=BLUE,
            lw=2.4,
            solid_capstyle="butt",
            zorder=4,
        )

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
        Line2D([], [], color=BLUE, lw=2.4, label="our state-space detector"),
    ]
    ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=4,
        fontsize=5,
        frameon=False,
        handlelength=1.4,
        columnspacing=1.0,
        handletextpad=0.5,
        borderpad=0.0,
    )

    ax.set_ylim(-30, 268)
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

    for start_s, end_s in decoded["state_space_intervals"]:
        if start_s > window_hi or end_s < window_lo:
            continue
        ax.plot(
            [start_s, end_s],
            [-16, -16],
            color=BLUE,
            lw=2.6,
            solid_capstyle="butt",
            clip_on=False,
            zorder=4,
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
    ax.set_ylim(-20, 178)
    ax.set_xlabel("time (s)")
    ax.set_title(EVENT_LABELS[event].replace("\n", " "), fontsize=6)


def plot_criterion_space(ax, decoded: dict, config: MemorylessSequenceConfig) -> None:
    """Every candidate sequence, against the two criteria that accept it.

    Panels b to d show three cases. This panel shows that they are not special:
    of every candidate the detector built, only the two continuous replays
    reach the box, and nothing else comes near the dispersion line.
    """
    candidates = decoded["candidates"]
    duration_s, dispersion_cm = candidates.duration_s, candidates.dispersion_cm

    ax.axhspan(config.dispersion_threshold_cm, 1e3, color=GREEN, alpha=0.06, lw=0)
    ax.axvline(config.minimum_duration_s, color="0.55", lw=0.7, ls=":")
    ax.axhline(config.dispersion_threshold_cm, color="0.55", lw=0.7, ls=":")
    ax.scatter(duration_s, dispersion_cm, s=7, color="0.72", lw=0, zorder=2)
    accepted = candidates.is_replay
    ax.scatter(
        duration_s[accepted],
        dispersion_cm[accepted],
        s=30,
        zorder=3,
        facecolor=VERMILION,
        edgecolor=VERMILION,
        lw=1.1,
    )

    # each case from panels b to d, marked and named where it lands
    annotated = (
        (0, "continuous replay\nfound", VERMILION, True, (10, 6)),
        (7, "short replay\nfound", VERMILION, True, (14, 24)),
        (6, "short path\nmissed", BLUE, False, (48, 22)),
        (5, "remote place\ndeclined", PURPLE, False, (108, 2)),
    )
    for event, name, color, filled, offset in annotated:
        event_start, event_end = decoded["event_times"][event]
        for i in range(len(candidates.runs)):
            start, stop = candidates.runs[i]
            if not (
                decoded["midpoint_s"][start] < event_end
                and event_start < decoded["midpoint_s"][stop]
            ):
                continue
            if bool(candidates.is_replay[i]) != filled:
                continue
            ax.scatter(
                duration_s[i],
                dispersion_cm[i],
                s=30,
                zorder=4,
                facecolor=color if filled else "none",
                edgecolor=color,
                lw=1.1,
            )
            ax.annotate(
                name,
                xy=(duration_s[i], dispersion_cm[i]),
                xytext=(duration_s[i] + offset[0] / 100, dispersion_cm[i] + offset[1]),
                fontsize=5.5,
                color=color,
                ha="center",
                va="bottom",
                linespacing=1.25,
                arrowprops={"arrowstyle": "-", "lw": 0.5, "color": color},
            )
            break

    ax.set_xlim(0, 1.85)
    ax.set_ylim(-3, 62)
    ax.set_xlabel("event duration (s)")
    ax.set_ylabel("dispersion (cm)")
    ax.set_title(
        f"all {len(candidates.runs)} candidate sequences. Only "
        f"{int(accepted.sum())} reach the shaded box,\nwhere both criteria are met.",
        fontsize=5.5,
        linespacing=1.4,
    )
    ax.text(
        config.minimum_duration_s + 0.03,
        59,
        "minimum duration",
        fontsize=5,
        color="0.45",
        ha="left",
        va="top",
    )
    ax.text(
        1.8,
        config.dispersion_threshold_cm + 1.5,
        "minimum dispersion",
        fontsize=5,
        color="0.45",
        ha="right",
        va="bottom",
    )


def make_figure(n_neurons: int, config: MemorylessSequenceConfig):
    decoded = simulate_and_decode(n_neurons, config)
    decoded["state_space_intervals"] = state_space_intervals(decoded)

    set_figure_defaults(context="paper")
    # this project's defaults turn constrained layout on, so the grid must not
    # also set its own margins: the two fight and the panels collapse
    fig = plt.figure(
        figsize=(TWO_COLUMN, TWO_COLUMN * 0.82), dpi=450, constrained_layout=True
    )
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.05, hspace=0.09)
    grid = GridSpec(3, 8, figure=fig, height_ratios=[1.0, 1.15, 0.95])

    ax_overview = fig.add_subplot(grid[0, :])
    plot_overview(ax_overview, decoded, config)

    outcomes = (
        (
            0,
            "continuous replay",
            f"found, but {duration_ratio(decoded, 0):.1f}x too long:\n"
            f"{100 * local_fraction_of_event(decoded, 0):.0f}% of it is the animal's own place",
        ),
        (
            7,
            "short replay, 80 ms",
            f"found, but {duration_ratio(decoded, 7):.1f}x too long:\n"
            "the window sets a floor under any duration",
        ),
        (6, "short path, 30 cm", "missed: the path is too short"),
        (5, "remote place, held still", "declined: it does not travel"),
    )
    zooms = []
    for column, (event, name, outcome) in enumerate(outcomes):
        ax = fig.add_subplot(grid[1, 2 * column : 2 * column + 2])
        plot_event(ax, decoded, event, config)
        ax.set_title(f"{name}\n{outcome}", fontsize=5.5, linespacing=1.4)
        if column:
            ax.set_yticklabels([])
        zooms.append(ax)
    zooms[0].set_ylabel("position (cm)")

    ax_criteria = fig.add_subplot(grid[2, 3:5])
    plot_criterion_space(ax_criteria, decoded, config)

    label_panels([ax_overview], labels=["a"], xpos=-0.05, ypos=1.14)
    label_panels(zooms, labels=["b", "c", "d", "e"], xpos=-0.10, ypos=1.34)
    label_panels([ax_criteria], labels=["f"], xpos=-0.20, ypos=1.02)

    fig.suptitle(
        "Memoryless sliding-window detection on the Figure 1 simulation", fontsize=7
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
