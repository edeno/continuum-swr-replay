#!/usr/bin/env python
"""Supplemental figure: two replay detectors on replays whose shape is known.

Upper panels run the ported Widloski & Foster sliding-window detector and this
project's switching state-space model over the Figure 1 simulation (25 neurons
here where Figure 1 uses 30, and a plain fit rather than EM), so the two
procedures can be read against a handful of labeled events.

Lower panels summarize the sweep in `scripts/qa/sweep_memoryless_detection.py`,
which wrote replays of chosen path length, duration, firing rate and distance
from the animal one at a time into a simulated recording and recorded what each
detector reported. Those panels read the aggregates the sweep saves, so the
figure and `docs/design/replay_detection_comparison/02-memoryless-sequence.md`
cannot disagree; re-run the sweep to change them.

Every outcome the titles and caption state is read from the data. Where the
prose commits to a verdict -- an event missed, declined, or found -- the script
checks it and stops rather than write a caption the panels contradict.

In both halves the two detectors read one encoding model, fitted once.

Usage:
    python scripts/figures/generate_memoryless_comparison.py
"""

import argparse
import os
import string
import textwrap
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from non_local_detector import NonLocalSortedSpikesDetector
from non_local_detector.environment import Environment
from non_local_detector.simulate.sorted_spikes_simulation import (
    SAMPLING_FREQUENCY,
    TRACK_HEIGHT,
    make_simulated_data,
)

from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.config import (
    MemorylessSequenceConfig,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.detect import (
    detect_candidate_sequences,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.likelihood import (
    ENCODING_KEY,
    interior_bin_distances_cm,
    sorted_spikes_epoch_log_likelihood,
    window_midpoints_s,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.simulation import (
    InjectedReplay,
    inject_replay,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.statistics import (
    aggregate_windows,
    normalize_windows,
)
from continuum_swr_replay.figure_utilities import (
    TWO_COLUMN,
    WONG,
    save_figure,
    set_figure_defaults,
)
from continuum_swr_replay.parameters import LOCAL_POSITION_STD

SUMMARY_PATH = Path(
    "docs/design/replay_detection_comparison/detection_sweep_summary.csv"
)
"""Aggregates of the parameter sweep, drawn in the lower panels."""

DISTANCE_ORDER = ("(0, 40]", "(40, 80]", "(80, 120]", "(120, 200]")
"""Every bin the sweep's `pd.cut` can produce, in order. Panel (i) draws the
ones the summary holds and refuses a summary holding any other."""

ROW_HEIGHTS = [0.80, 0.95, 1.05, 0.75]
"""Relative heights of the figure's four rows: overview, events, sweep, sweep."""

LETTER_RISE_PT = 9.0
"""Panel letters ride this far above the axes: the height of a panel title,
so letter and title share a line."""

LETTER_GAP_PT = 4.0
"""And this far left of whatever the panel draws furthest left."""

TRACK_HEIGHT_CM = float(TRACK_HEIGHT)
"""The simulation package's own track length, read rather than restated."""

DECODE_BIN_STRIDE = 3
"""Figure 1 strides the 1500 Hz simulation to the project's 500 Hz decode grid."""

EVENT_LABELS = [
    "continuous\noutbound",
    "fragmented",
    "continuous\ninbound",
    "no spike",
    "local",
    "remote\nstationary*",
    "short path\n30 cm*",
    "short replay\n80 ms*",
]
"""Labels for the simulation's events. A star marks one added here; the rest
come with the package's simulation."""

SHORT_REPLAY = InjectedReplay(
    start_cm=35.0,
    path_cm=100.0,
    duration_s=0.080,
    peak_rate_hz=120.0,
    at_s=44.40,
    seed=1,
)
"""A replay that travels far but lasts only 80 ms, firing strongly enough that
spike count is not the issue. The detector is expected to find it and report
it too long; the panel title reads the ratio off the data rather than stating
it here. The window puts a floor under any duration the method can report."""

SHORT_PATH = InjectedReplay(
    start_cm=40.0, path_cm=30.0, duration_s=0.300, peak_rate_hz=40.0, at_s=20.50, seed=0
)
"""A replay that travels only 30 cm. The detector decodes it well, but the
dispersion criterion needs a path of about 42 cm, so it is expected to reject
the candidate."""

SHORT_PATH_EVENT, SHORT_REPLAY_EVENT = 6, 7
CONTINUOUS_OUTBOUND_EVENT, REMOTE_STATIONARY_EVENT = 0, 5
"""Rows of ``event_times`` in :func:`simulate_and_decode`, in the order the
events are added."""

NON_LOCAL_PROB_THRESHOLD = 0.8
NON_LOCAL_MINIMUM_DURATION_S = 0.015
"""The project's non-local event definition, as in
`analysis/hippocampus/event_detection.get_confident_times`: non-local
probability above 0.8 for at least 15 ms.

The state-space model is otherwise left on the library's defaults. Figure 1
sets a discrete transition stickiness, but stickiness is a Dirichlet
concentration read only by the EM M-step, and this comparison calls `fit`
rather than `estimate_parameters`, so passing one would change nothing here
and would read as tuning.
"""

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

SLIDING = "sliding-window detector"
STATE_SPACE = "state-space detector"
"""One name each, used in both figures. Neither says "our": a figure should
name a method, not a side."""


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

    # two replays the detector is expected to mis-report, in the order
    # SHORT_PATH_EVENT, SHORT_REPLAY_EVENT
    place_field_means = np.linspace(0, TRACK_HEIGHT_CM, n_neurons)
    for replay in (SHORT_PATH, SHORT_REPLAY):
        spike_times, bounds = inject_replay(
            spike_times,
            place_field_means,
            replay,
            sampling_frequency=SAMPLING_FREQUENCY,
        )
        event_times = np.vstack([event_times, bounds])
        is_event |= (time >= bounds[0]) & (time <= bounds[1])

    time, position, speed, is_event = (
        array[::DECODE_BIN_STRIDE] for array in (time, position, speed, is_event)
    )

    # encode on non-event data only, the analogue of this project's no-ripple
    # training mask; spikes inside an event must not shape the place fields
    is_training = ~is_event

    # the grid both procedures decode on comes from the config, matching the sweep
    detector = NonLocalSortedSpikesDetector(
        environments=[Environment(place_bin_size=config.place_bin_size_cm)],
        sorted_spikes_algorithm="sorted_spikes_kde",
        sorted_spikes_algorithm_params={"position_std": 6.0, "block_size": int(2**8)},
        local_position_std=LOCAL_POSITION_STD,
    )
    # fit rather than estimate_parameters, matching the sweep: EM would
    # re-estimate the place fields and the transitions on the data it is then
    # read on, and the two scripts should not differ in that.
    detector.fit(
        position_time=time,
        position=position,
        spike_times=spike_times,
        is_training=is_training,
    )

    # one set of place fields for both procedures, read from the detector that
    # fitted them
    encoding_model = detector.encoding_model_[ENCODING_KEY]
    environment = encoding_model["environment"]

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
        "detector": detector,
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

    Reads the detector fitted in :func:`simulate_and_decode`, which is also the
    one whose place fields the sliding-window likelihood was given, so the two
    procedures differ in how they use the encoding model and not in what it is.
    """
    time = decoded["time"]
    results = decoded["detector"].predict(
        position_time=time,
        position=decoded["position"],
        spike_times=decoded["spike_times"],
        time=time,
    )
    probability = np.asarray(
        results.acausal_state_probabilities.sel(
            states=["Non-Local Continuous", "Non-Local Fragmented"]
        )
        .sum("states")
        .values
    ).squeeze()
    if not np.isfinite(probability).all() or probability.max() <= 0.0:
        raise RuntimeError(
            "The state-space model returned no probability mass; its state "
            "probabilities must sum to one at every time."
        )

    above = probability > NON_LOCAL_PROB_THRESHOLD
    edges = np.diff(above.astype(np.int8), prepend=0, append=0)
    return [
        (time[start], time[stop])
        for start, stop in zip(
            np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1, strict=True
        )
        if time[stop] - time[start] >= NON_LOCAL_MINIMUM_DURATION_S
    ]


def candidates_overlapping(decoded: dict, event: int) -> list[int]:
    """Every candidate sequence that overlaps a labeled event, accepted or not."""
    candidates = decoded["candidates"]
    event_start, event_end = decoded["event_times"][event]
    return [
        i
        for i, (start, stop) in enumerate(candidates.runs)
        if decoded["midpoint_s"][start] < event_end
        and event_start < decoded["midpoint_s"][stop]
    ]


def accepted_overlapping(decoded: dict, event: int) -> list[int]:
    """The subset the detector called replay."""
    is_replay = decoded["candidates"].is_replay
    return [i for i in candidates_overlapping(decoded, event) if is_replay[i]]


def verdict(decoded: dict, event: int) -> str:
    """What the sliding-window detector did with one labeled event.

    ``found`` when an accepted sequence overlaps it; ``declined`` when a
    candidate was assembled but none was accepted; ``missed`` when no
    candidate overlaps it at all. Read from the data, so a panel title cannot
    drift away from the panel.
    """
    if accepted_overlapping(decoded, event):
        return "found"
    if candidates_overlapping(decoded, event):
        return "declined"
    return "missed"


def duration_ratio(decoded: dict, event: int) -> float:
    """How much longer the detector calls an event than the event really is.

    Over the longest accepted sequence overlapping it. Raises when there is
    none, so a title cannot be formatted from a detection that did not happen.
    """
    accepted = accepted_overlapping(decoded, event)
    if not accepted:
        raise RuntimeError(
            f"No accepted sequence overlaps event {event} ({EVENT_LABELS[event]!r}); "
            "the figure's titles and caption assume one. Re-read the data before "
            "regenerating."
        )
    candidates = decoded["candidates"]
    event_start, event_end = decoded["event_times"][event]
    return float(
        max(candidates.duration_s[i] for i in accepted) / (event_end - event_start)
    )


def outcome_title(decoded: dict, event: int, label: str) -> str:
    """One-line panel title stating what the detector did, from the data."""
    what = verdict(decoded, event)
    if what == "found":
        return f"{label}: {duration_ratio(decoded, event):.1f}x too long"
    return f"{label}: {what}"


def decoded_center_cm(decoded: dict) -> np.ndarray:
    """Center of each window, NaN where the window has none."""
    center = decoded["candidates"].decisions.center_index
    out = np.full(center.size, np.nan)
    present = center >= 0
    out[present] = decoded["place_bin_centers_cm"][center[present]]
    return out


def local_fraction_of_event(decoded: dict, event: int) -> float:
    """Share of the accepted event overlapping ``event`` that is local coding.

    Over the longest accepted sequence, the one :func:`duration_ratio` reads.
    """
    accepted = accepted_overlapping(decoded, event)
    if not accepted:
        raise RuntimeError(f"No accepted sequence overlaps event {event}.")
    longest = max(accepted, key=lambda i: decoded["candidates"].duration_s[i])
    return local_fraction(decoded, longest)


def best_dispersion_cm(decoded: dict, event: int) -> float:
    """The largest dispersion any candidate overlapping ``event`` reached."""
    dispersion = decoded["candidates"].dispersion_cm
    return max(
        (float(dispersion[i]) for i in candidates_overlapping(decoded, event)),
        default=float("nan"),
    )


def check_caption_claims(decoded: dict, config: MemorylessSequenceConfig) -> None:
    """Stop rather than write prose the panels contradict.

    The caption commits to a verdict for each of the four zoomed events and to
    the state-space detector reporting all of them. Each is read from the
    data here; a threshold, seed or library change that moves one turns into
    an error naming it instead of a caption that says the opposite of its
    figure.
    """
    expected = {
        CONTINUOUS_OUTBOUND_EVENT: "found",
        SHORT_REPLAY_EVENT: "found",
        SHORT_PATH_EVENT: "declined",
        REMOTE_STATIONARY_EVENT: "declined",
    }
    for event, want in expected.items():
        got = verdict(decoded, event)
        if got != want:
            raise RuntimeError(
                f"The caption says the sliding-window detector {want} "
                f"{EVENT_LABELS[event]!r}; the data say {got}. Rewrite the caption."
            )
    for event in (SHORT_PATH_EVENT, REMOTE_STATIONARY_EVENT):
        if not best_dispersion_cm(decoded, event) < config.dispersion_threshold_cm:
            raise RuntimeError(
                f"The caption says {EVENT_LABELS[event]!r} was declined for too "
                "little dispersion, but a candidate overlapping it clears the "
                "threshold. Rewrite the caption."
            )
    if not np.isfinite(bound_error_ms(decoded, tuple(expected))):
        raise RuntimeError(
            "The caption says the state-space detector reports all four zoomed "
            "events; it does not. Rewrite the caption."
        )


def local_fraction(decoded: dict, sequence: int, tolerance_cm: float = 15.0) -> float:
    """Share of a sequence's kept windows whose center is the animal's own place.

    Kept windows only, as the sweep scores them: a window inside a merged gap
    was rejected and is not one of the event's decoded positions.
    """
    candidates = decoded["candidates"]
    start, stop = candidates.runs[sequence]
    center_cm = decoded_center_cm(decoded)[start : stop + 1]
    animal_cm = np.interp(
        decoded["midpoint_s"][start : stop + 1], decoded["time"], decoded["position"]
    )
    kept = candidates.assembly.kept[start : stop + 1]
    return float(np.mean(np.abs(center_cm[kept] - animal_cm[kept]) < tolerance_cm))


def bound_error_ms(decoded: dict, events: tuple[int, ...]) -> float:
    """Largest error in either bound of the state-space model's reports.

    The caption used to state a round number for this. It is read off the
    data now, over exactly the events the figure shows.
    """
    worst = 0.0
    for event in events:
        start_s, end_s = decoded["event_times"][event]
        overlapping = [
            (a, b)
            for a, b in decoded["state_space_intervals"]
            if a < end_s and start_s < b
        ]
        if not overlapping:
            return float("nan")
        worst = max(
            worst,
            abs(overlapping[0][0] - start_s),
            abs(overlapping[-1][1] - end_s),
        )
    return 1000 * worst


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
        Line2D([], [], color=GREEN, lw=1.0, label="where the animal is"),
        Line2D(
            [],
            [],
            color=SKY,
            marker="o",
            ms=2.0,
            lw=0,
            label="decoded place, window kept",
        ),
        Line2D(
            [],
            [],
            color="0.55",
            marker="o",
            ms=2.0,
            lw=0,
            label="decoded place, window rejected",
        ),
        Patch(color=YELLOW, alpha=0.6, label="labeled event"),
        Patch(color="0.75", label="posterior, panels b-e"),
        Line2D([], [], color=VERMILION, lw=2.4, label=f"event, {SLIDING}"),
        Line2D([], [], color="0.6", lw=1.2, label="candidate, rejected"),
        Line2D([], [], color=BLUE, lw=2.4, label=f"event, {STATE_SPACE}"),
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
    # the grey wash is the posterior itself; panel b's label says so once
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
        label="where the animal is",
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


def detection_grid(summary: pd.DataFrame, measure: str) -> pd.DataFrame:
    """Detection rate as path length by duration."""
    rows = summary[
        (summary["summary"] == "detection_rate_by_path_and_duration")
        & (summary.measure == measure)
    ]
    return rows.pivot(index="path_cm", columns="duration_s", values="value")


def plot_detection(ax, grid: pd.DataFrame, title: str, show_y: bool):
    """Percent of the swept replays each detector reported, path by duration."""
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
    ax.set_xticklabels([f"{1000 * c:.0f}" for c in grid.columns], fontsize=5)
    ax.set_yticks(range(grid.shape[0]))
    ax.set_yticklabels([f"{r:.0f}" for r in grid.index], fontsize=5)
    ax.set_xlabel("duration (ms)")
    if show_y:
        ax.set_ylabel("path length (cm)")
    else:
        ax.set_yticklabels([])
    ax.set_title(title, fontsize=6)

    for row in range(grid.shape[0]):
        for column in range(grid.shape[1]):
            percent = 100 * grid.iat[row, column]
            ax.text(
                column,
                row,
                f"{percent:.0f}",
                ha="center",
                va="center",
                fontsize=4.5,
                color="white" if percent < 55 else "black",
            )
    return image


def plot_duration_bias(ax, summary: pd.DataFrame):
    """Reported duration over true duration, for events both procedures found."""
    rows = summary[summary["summary"] == "median_duration_ratio_by_duration"]
    ax.axhline(1.0, color="0.6", lw=0.7, ls=":")
    for measure, color, label in (
        ("sliding_duration_ratio", VERMILION, SLIDING),
        ("state_space_duration_ratio", BLUE, STATE_SPACE),
    ):
        series = rows[rows.measure == measure].sort_values("duration_s")
        ax.plot(
            1000 * series.duration_s,
            series.value,
            color=color,
            lw=1.0,
            marker="o",
            ms=2.5,
            label=label,
        )
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 5, 10])
    ax.set_yticklabels(["1x", "2x", "5x", "10x"])
    ax.set_xlabel("duration (ms)")
    ax.set_ylabel("reported / actual duration")
    ax.set_title("reported duration", fontsize=6)
    ax.legend(fontsize=5, frameon=False, loc="upper right", handlelength=1.3)


def plot_absorption(ax, summary: pd.DataFrame):
    """The over-call against how far the replay sits from the animal's own place."""
    rows = summary[summary["summary"] == "median_by_distance_from_animal"]
    present = set(rows.distance_bin_cm)
    if not present <= set(DISTANCE_ORDER):
        raise ValueError(
            f"The sweep summary holds distance bins {sorted(present - set(DISTANCE_ORDER))} "
            "this panel does not know how to place; update DISTANCE_ORDER."
        )
    order = list(DISTANCE_ORDER)
    at = np.arange(len(order))

    def series(measure):
        indexed = rows[rows.measure == measure].set_index("distance_bin_cm").value
        return np.array([indexed.get(bin_name, np.nan) for bin_name in order])

    # an expected bin with no jointly reported replay is a result, not a gap
    for x, bin_name in zip(at, order, strict=True):
        if bin_name not in present:
            ax.text(
                x,
                1.0,
                "none\nreported\nby both",
                fontsize=4.5,
                color="0.45",
                ha="center",
                va="bottom",
                linespacing=0.9,
            )

    ax.axhline(1.0, color="0.6", lw=0.7, ls=":")
    ax.plot(
        at, series("sliding_duration_ratio"), color=VERMILION, lw=1.0, marker="o", ms=3
    )
    ax.plot(
        at, series("state_space_duration_ratio"), color=BLUE, lw=1.0, marker="o", ms=3
    )
    for x, local in zip(at, series("sliding_local_fraction"), strict=True):
        if not np.isfinite(local):
            continue
        ax.annotate(
            f"{100 * local:.0f}%",
            xy=(x, 1.0),
            xytext=(x, 1.28),
            fontsize=5,
            color="0.35",
            ha="center",
            va="center",
        )
    ax.set_xticks(at)
    ax.set_xticklabels([bin_name.replace(", ", "-").strip("(]") for bin_name in order])
    ax.set_ylim(0.85, 2.6)
    ax.set_xlabel("distance from the animal (cm)")
    ax.set_ylabel("reported / actual duration")
    ax.set_title("reported duration by distance", fontsize=6)


def place_panel_letters(fig, rows: list[list]) -> None:
    """Draw each panel's letter on its title line, flush with a common margin.

    The project's `label_panels` positions a letter in axes fractions, which
    is right for a grid of equal panels. Here the four rows differ in height
    and in how wide a y label each carries, so one fraction puts the letters
    at four different heights and four different depths. This measures the
    drawn figure instead: every letter sits on its row's title line, and the
    first letter of each row sits on one left margin.

    Constrained layout settles only when the figure is drawn, so this must
    draw first and must run after every panel is filled in.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_figure = fig.transFigure.inverted()
    rise = LETTER_RISE_PT / (72 * fig.get_figheight())
    gap = LETTER_GAP_PT / (72 * fig.get_figwidth())

    def drawn_left(ax) -> float:
        """Left edge of everything the panel draws, y label and ticks included."""
        return to_figure.transform_bbox(ax.get_tightbbox(renderer)).x0

    margin = min(drawn_left(row[0]) for row in rows) - gap
    letters = iter(string.ascii_lowercase)
    for row in rows:
        for column, ax in enumerate(row):
            fig.text(
                margin if column == 0 else drawn_left(ax) - gap,
                to_figure.transform_bbox(ax.bbox).y1 + rise,
                next(letters),
                fontsize=8,
                fontweight="bold",
                ha="left",
                va="center",
            )


def make_figure(
    n_neurons: int, config: MemorylessSequenceConfig, summary: pd.DataFrame
):
    decoded = simulate_and_decode(n_neurons, config)
    decoded["state_space_intervals"] = state_space_intervals(decoded)
    check_caption_claims(decoded, config)

    set_figure_defaults(context="paper")
    # this project's defaults turn constrained layout on, so the grid must not
    # also set its own margins: the two fight and the panels collapse
    fig = plt.figure(
        figsize=(TWO_COLUMN, TWO_COLUMN * 1.22), dpi=450, constrained_layout=True
    )
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.05, hspace=0.09)
    grid = GridSpec(4, 8, figure=fig, height_ratios=ROW_HEIGHTS)

    ax_overview = fig.add_subplot(grid[0, :])
    plot_overview(ax_overview, decoded, config)

    # one line each, so these titles sit at the same height as every other
    # row's and the panel letters can share one offset; the verdict in each
    # is read from the data
    outcomes = (
        (CONTINUOUS_OUTBOUND_EVENT, "continuous replay"),
        (SHORT_REPLAY_EVENT, "80 ms replay"),
        (SHORT_PATH_EVENT, "30 cm path"),
        (REMOTE_STATIONARY_EVENT, "remote place, held still"),
    )
    zooms = []
    for column, (event, label) in enumerate(outcomes):
        ax = fig.add_subplot(grid[1, 2 * column : 2 * column + 2])
        plot_event(ax, decoded, event, config)
        ax.set_title(outcome_title(decoded, event, label), fontsize=5.5)
        if column:
            ax.set_yticklabels([])
        zooms.append(ax)
    zooms[0].set_ylabel("decoded place (cm)")

    heatmaps = grid[2, :].subgridspec(1, 2, wspace=0.06)
    ax_sliding = fig.add_subplot(heatmaps[0])
    ax_state = fig.add_subplot(heatmaps[1])
    image = plot_detection(
        ax_sliding, detection_grid(summary, "sliding_matches"), SLIDING, show_y=True
    )
    plot_detection(
        ax_state,
        detection_grid(summary, "state_space_matches"),
        STATE_SPACE,
        show_y=False,
    )
    bar = fig.colorbar(image, ax=[ax_sliding, ax_state], fraction=0.03, pad=0.01)
    bar.set_label("replays reported (%)", fontsize=5.5)
    bar.ax.tick_params(labelsize=5)

    ax_duration = fig.add_subplot(grid[3, :4])
    plot_duration_bias(ax_duration, summary)
    ax_distance = fig.add_subplot(grid[3, 4:])
    plot_absorption(ax_distance, summary)

    place_panel_letters(
        fig,
        [
            [ax_overview],
            zooms,
            [ax_sliding, ax_state],
            [ax_duration, ax_distance],
        ],
    )

    return fig, decoded


def rewrap(text: str, width: int = 79) -> str:
    """Re-fill each paragraph, so interpolated numbers cannot leave ragged lines.

    The caption is prose with blank lines between paragraphs and nothing else,
    so filling paragraph by paragraph is safe.
    """
    return "\n\n".join(
        textwrap.fill(
            " ".join(paragraph.split()),
            width=width,
            break_on_hyphens=False,  # keeps "state-space" on one line
            break_long_words=False,
        )
        for paragraph in text.strip().split("\n\n")
    )


def check_summary_claims(summary: pd.DataFrame) -> None:
    """The caption's claims about the sweep, checked against the summary.

    Companion to :func:`check_caption_claims`, which covers the decoded
    recording. Anything the caption states about the sweep as a fact rather
    than interpolates as a number is asserted here.
    """
    by_path = summary[summary["summary"] == "by_path"]
    below = by_path[
        (by_path.measure == "state_space_replay_matches")
        & (by_path.path_cm < SHORT_PATH.path_cm + 15.0)
    ]
    if below.empty or (below.value > 0).any():
        raise RuntimeError(
            "The caption says the state-space detector, held to the published "
            "replay definition, reports nothing below 45 cm; the summary disagrees."
        )
    by_distance = summary[summary["summary"] == "median_by_distance_from_animal"]
    local = (
        by_distance[by_distance.measure == "sliding_local_fraction"]
        .set_index("distance_bin_cm")
        .value
    )
    ratio = (
        by_distance[by_distance.measure == "sliding_duration_ratio"]
        .set_index("distance_bin_cm")
        .value
    )
    near, far = "(40, 80]", ["(80, 120]", "(120, 200]"]
    if not (local.get(near, 0.0) > 0.0 and all(local.get(b, 0.0) == 0.0 for b in far)):
        raise RuntimeError(
            "The caption says absorbed local coding reaches about 80 cm and no "
            "further; the summary's local fractions disagree."
        )
    if not all(ratio.get(b, np.nan) > 1.0 for b in far):
        raise RuntimeError(
            "The caption says the window floor still leaves events too long "
            "beyond 80 cm; the summary's duration ratios disagree."
        )


def pct(value: float, digits: int = 0) -> str:
    """A percentage for the caption, refusing to print one that is not a number."""
    if not np.isfinite(value):
        raise RuntimeError(
            "A caption percentage is not finite; the sweep summary is incomplete."
        )
    return f"{100 * value:.{digits}f}%"


def write_caption(
    decoded: dict, summary: pd.DataFrame, config: MemorylessSequenceConfig, path: Path
) -> None:
    """Save the caption beside the figure, with its numbers read from the data.

    The figure itself carries almost no words, which is what a paper figure
    should look like. Everything a reader needs is here instead, written from
    the same arrays and the same summary the figure plots, so the two cannot
    drift apart.
    """
    overall = summary[summary["summary"] == "overall"].set_index("measure").value
    domain = (
        summary[summary["summary"] == "overall_in_domain"].set_index("measure").value
    )
    ratio = summary[
        (summary["summary"] == "median_duration_ratio_by_duration")
        & (summary.measure == "sliding_duration_ratio")
    ].sort_values("duration_s")
    shortest = ratio.iloc[0]
    check_summary_claims(summary)

    path.write_text(
        rewrap(
            f"""# {path.stem}

**Two replay detectors on replays whose shape is known.** The sliding-window
detector decodes each 80 ms window on its own, with no model of how one window
follows the next, and keeps or drops windows on the spread of their posterior
and on the jump to the next window. The state-space detector infers the whole
event at once and carries the animal's own place as a competing hypothesis.
Throughout, both detectors read one encoding model, fitted once.

**(a-e) One recording with labeled events.** The simulation carries a
continuous outbound replay, a fragmented replay, a continuous inbound replay
and a silent stretch; a remote stationary event, a 30 cm replay and an 80 ms
replay were written into the spike trains here and are marked with a star.

**(a)** The whole recording. Green is where the animal is; dots are the decoded
place of each window, kept or rejected; yellow marks a labeled event; bars below
the axis are the events each detector reports and the candidates the
sliding-window detector rejected.

**(b-e)** Four events, with the posterior in grey. The sliding-window
detector finds the continuous replay and gives it
{duration_ratio(decoded, CONTINUOUS_OUTBOUND_EVENT):.1f} times its true length,
because {100 * local_fraction_of_event(decoded, CONTINUOUS_OUTBOUND_EVENT):.0f}%
of the event it reports is the animal's own place, absorbed by the merge rule.
It also finds the 80 ms replay and reports it
{duration_ratio(decoded, SHORT_REPLAY_EVENT):.1f} times too long, with
{100 * local_fraction_of_event(decoded, SHORT_REPLAY_EVENT):.0f}% local coding
in it: the window alone puts a floor under any duration it can report. It
assembles a candidate from the 30 cm replay and another from the remote
stationary event and declines both for too little dispersion
({best_dispersion_cm(decoded, SHORT_PATH_EVENT):.0f} and
{best_dispersion_cm(decoded, REMOTE_STATIONARY_EVENT):.0f} cm against the
{config.dispersion_threshold_cm:.0f} cm it requires). The state-space detector
reports all four, with every bound within
{bound_error_ms(decoded, (CONTINUOUS_OUTBOUND_EVENT, SHORT_REPLAY_EVENT, SHORT_PATH_EVENT, REMOTE_STATIONARY_EVENT)):.0f} ms
of the truth.

**(f-i) {int(overall["n_cells"])} replays of chosen shape.** Made-up
replays were written into the spike trains of a simulated linear-track
recording, one at a time, each with a path length, duration, firing rate and
distance from the animal's own place that we chose. Because the right answer is known, what each
detector reports can be scored against it. A detector counts as reporting a
replay only when what it returns is that replay: on its path, and not covering
far more ground.

**(f, g)** Percent of those {int(overall["n_cells"])} replays reported, by the
path the replay travels and how long it lasts. Each detector runs as published,
with its own acceptance criteria, and both are scored the same way against the
same known answer. The sliding-window detector reports
{pct(overall["sliding_matches"])} of the replays and the state-space
detector {pct(overall["state_space_matches"])}.
{pct(overall["found_by_state_space_only"])} were reported by the
state-space detector and missed by the sliding-window detector;
{pct(overall["found_by_sliding_window_only"], 2)} went the other way.

The two detectors reach their limits on different quantities. The state-space
detector is limited by duration, which is spike count, which is the data. The
sliding-window detector is limited by path length, through its own dispersion
criterion: a replay must spread more than 12 cm about its own centre, and a
30 cm path cannot, however well it is decoded.
Held to that same criterion our state-space detector also reports nothing below
45 cm, so this is the published replay definition acting, not the window-by-window
inference. On the part of the grid that definition can accept at all — a path
over about 42 cm and a duration over 100 ms, {pct(domain["n_cells"] / overall["n_cells"])}
of the cells — the two read {pct(domain["sliding_matches"])} and
{pct(domain["state_space_matches"])}.

**(h)** Reported duration over actual duration, for replays both detectors
reported. A {1000 * shortest.duration_s:.0f} ms replay is reported as about
{1000 * shortest.duration_s * shortest.value:.0f} ms — the same floor that panel
(c) shows for one event.

**(i)** The same ratio against how far the replay sits from the animal's own
place. Numbers give the share of the reported event's 80 ms windows that
decode the animal's place rather than a replayed one. That absorption of local
coding reaches about 80 cm from the animal and is what drives the largest
over-reporting, as in panel (b); beyond it the window floor of panel (h) still
leaves the reported duration too long. Over the whole sweep,
{pct(overall["sliding_accepted_but_not_the_injection"])} of the
sliding-window detector's accepted events are not the replay that was put in.

**Caveats.** Sorted spikes, a linear track, no artifacts. In (f-i) the animal
rests at one end of the track and pauses nowhere else, so distance from its
place is fixed by where the replay is put and cannot be varied independently.
Replays run at constant speed with Poisson firing from the same place fields
the encoding model was fitted on, so there is no model mismatch and the picture
is optimistic. The significance test the published method requires is not
applied here.

Produced by `scripts/figures/generate_memoryless_comparison.py` from
`docs/design/replay_detection_comparison/detection_sweep_summary.csv`.
"""
        )
        + "\n"
    )


def read_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"No sweep summary at {path}. Run "
            "`python scripts/qa/sweep_memoryless_detection.py` first."
        )
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-neurons", type=int, default=25)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("paper/figures/memoryless_comparison")
    )
    args = parser.parse_args()

    config = MemorylessSequenceConfig(sampling_frequency=500.0)
    summary = read_summary(args.summary)
    fig, decoded = make_figure(args.n_neurons, config, summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_figure("memoryless_comparison", base_dir=args.output_dir)
    plt.close(fig)
    write_caption(
        decoded, summary, config, args.output_dir / "memoryless_comparison_caption.md"
    )

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
