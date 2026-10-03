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
figure and the numbers quoted from the sweep cannot disagree; re-run the sweep
to change them.

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
    with_reference_rate_floor,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.shuffle import (
    DEFAULT_N_SHUFFLES,
    SIGNIFICANCE_LEVEL,
    ShuffleSeed,
    shuffle_candidate_sorted,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.simulation import (
    InjectedReplay,
    immobile_intervals,
    inject_replay,
    stray_window_spikes,
    without,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.statistics import (
    NO_CENTER,
    aggregate_windows,
    dispersion_cm,
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
"""Every bin the sweep's `pd.cut` can produce, in order. Panel (k) draws the
ones the summary holds and refuses a summary holding any other."""

CONTROLS_SUMMARY_PATH = Path(
    "docs/design/replay_detection_comparison/detection_controls_summary.csv"
)
"""Aggregates of the controls sweep: no replay, bursts, jumps, chains."""

HELDOUT_SUMMARY_PATH = Path(
    "docs/design/replay_detection_comparison/detection_heldout_summary.csv"
)
"""Aggregates of the held-out sweep: support by fold, estimator, class and truth."""

ROW_HEIGHTS = [0.80, 0.95, 1.05, 0.75, 0.75, 0.70]
"""Relative heights of the figure's rows: overview, events, sweep, timing,
controls, held-out evidence."""

HELDOUT_GROUPS = (
    ("replay,\nours only", ("ours_only",), "continuous_replay"),
    ("replay,\nshared", ("shared",), "continuous_replay"),
    ("jumping\nburst", ("ours_only", "shared"), "non_local"),
    ("place-less\nburst", ("ours_only", "shared"), "no_content"),
)
"""The kinds of event the held-out panels compare: by which detectors found
them and what they truly are."""

FOLD_HATCH = {"A": None, "B": None, "C": "////"}
"""Fold C's evaluation cells cover only half the track."""

OUTPUTS = (
    ("sliding_accepted", "sliding, before test", 0.4),
    ("sliding_published", "sliding, published", 1.0),
    ("state_space_canonical", "ours, non-local", 1.0),
    ("state_space_continuous", "ours, continuous", 0.4),
)
"""The four scored outputs of the controls, with their bar opacity."""

MULTIPART_CONDITIONS = (
    "jump_120ms",
    "jump_200ms",
    "chain_continuing",
    "chain_elsewhere",
)
"""The controls conditions whose events have two parts."""

SEGMENTATION_STYLE = (
    ("spanning", "0.2", "one event over all parts"),
    ("one_per_part", "0.55", "one event per part"),
    ("other", "0.8", "other"),
    ("parts_missing", "1.0", "a part missed"),
)
"""How a multi-part event was carved, as the stacked bars shade it."""

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
    "one window\nfar away*",
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
CONTINUOUS_OUTBOUND_EVENT, CONTINUOUS_INBOUND_EVENT = 0, 2
REMOTE_STATIONARY_EVENT = 5
STRAY_WINDOW_EVENT = 8
"""Rows of ``event_times`` in :func:`simulate_and_decode`, in the order the
events are added."""

SHUFFLE_RUN_SEED = 20260922
"""Seeds the place-cell identity shuffle, as in the sweep."""

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

STRAY_WINDOW_AT_S = 31.40
"""Where the hand-placed stray-window event begins: in the stillness before the
inbound replay, among candidates of local coding that no caption claim reads.
The stillness before the outbound replay is no good: that replay's candidate
absorbs the local coding before it, so an event there changes its verdict."""

STRAY_WINDOW_PLACEMENT_MARGIN_S = 0.4
"""How far the event's cleared span must sit from every other event."""


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

    place_field_means = np.linspace(0, TRACK_HEIGHT_CM, n_neurons)

    # the package has no remote stationary event; add one in the last pause,
    # and hold it out of training like every other event
    spike_times, remote_stationary = splice_remote_stationary(
        spike_times, place_field_means, float(event_times[4, 0])
    )
    event_times = np.vstack([event_times, remote_stationary])
    is_event |= (time >= remote_stationary[0]) & (time <= remote_stationary[1])

    # two replays the detector is expected to mis-report, in the order
    # SHORT_PATH_EVENT, SHORT_REPLAY_EVENT
    for replay in (SHORT_PATH, SHORT_REPLAY):
        spike_times, bounds = inject_replay(
            spike_times,
            place_field_means,
            replay,
            sampling_frequency=SAMPLING_FREQUENCY,
        )
        event_times = np.vstack([event_times, bounds])
        is_event |= (time >= bounds[0]) & (time <= bounds[1])

    # a still event at the animal's place with one window decoded far away,
    # hand-placed in the first stillness clear of every other event. It is
    # written into the spikes that are decoded and nowhere else: the encoding
    # model is fitted on the recording as it was before, so adding this event
    # leaves every other panel exactly as it was
    training_spike_times, is_training_event = spike_times, is_event.copy()
    clear = without(
        immobile_intervals(
            time, speed, config.speed_threshold_cm_per_s, minimum_duration_s=1.0
        ),
        event_times,
        margin_s=STRAY_WINDOW_PLACEMENT_MARGIN_S,
    )

    stray_local_cm = float(np.interp(STRAY_WINDOW_AT_S, time, position))
    spike_times, stray = stray_window_spikes(
        spike_times,
        place_field_means,
        STRAY_WINDOW_AT_S,
        local_cm=stray_local_cm,
        window_s=config.window_duration_s,
        stride_s=config.stride_s,
    )
    if not (
        (clear[:, 0] <= stray.cleared[0]) & (stray.cleared[1] <= clear[:, 1])
    ).any():
        raise RuntimeError(
            f"The stray-window event at {STRAY_WINDOW_AT_S} s does not sit in "
            "stillness clear of every other event; move STRAY_WINDOW_AT_S."
        )
    event_times = np.vstack([event_times, stray.bounds])
    is_event |= (time >= stray.bounds[0]) & (time <= stray.bounds[1])

    time, position, speed, is_event, is_training_event = (
        array[::DECODE_BIN_STRIDE]
        for array in (time, position, speed, is_event, is_training_event)
    )

    # encode on non-event data only, the analogue of this project's no-ripple
    # training mask; spikes inside an event must not shape the place fields
    is_training = ~is_training_event

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
        spike_times=training_spike_times,
        is_training=is_training,
    )

    # one set of place fields for both procedures, read from the detector that
    # fitted them; the sliding window decodes them as the reference does, with
    # its floor under every field
    encoding_model = with_reference_rate_floor(
        detector.encoding_model_[ENCODING_KEY], config
    )
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
    bin_distance_cm = interior_bin_distances_cm(environment)
    candidates = detect_candidate_sequences(
        posterior, midpoint_s, bin_distance_cm, window_speed, config
    )
    # the reference's own significance test, on the full decode grid: the
    # likelihood's grid has lost the final row, which spikes are binned against
    shuffles = {
        int(i): shuffle_candidate_sorted(
            int(candidates.runs[i, 0]),
            int(candidates.runs[i, 1]),
            time,
            spike_times,
            encoding_model,
            bin_distance_cm,
            config,
            seed=ShuffleSeed(run=SHUFFLE_RUN_SEED, session="figure-one-simulation"),
            n_shuffles=DEFAULT_N_SHUFFLES,
        )
        for i in np.flatnonzero(candidates.is_replay)
    }

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
        "candidates": candidates,
        "shuffles": shuffles,
        "bin_distance_cm": bin_distance_cm,
        "stray_window": stray,
        "stray_window_local_cm": stray_local_cm,
        "stray_window_remote_cm": float(place_field_means[stray.remote_cell]),
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
    """The subset that met the duration and dispersion criteria."""
    is_replay = decoded["candidates"].is_replay
    return [i for i in candidates_overlapping(decoded, event) if is_replay[i]]


def is_reported(decoded: dict) -> np.ndarray:
    """Per candidate: replay under the whole published definition."""
    reported = np.zeros(len(decoded["candidates"].runs), dtype=bool)
    for i, result in decoded["shuffles"].items():
        reported[i] = result.passes
    return reported


def reported_overlapping(decoded: dict, event: int) -> list[int]:
    """The subset that also beat the place-cell identity shuffle: replay under
    the whole published definition."""
    return [
        i for i in accepted_overlapping(decoded, event) if decoded["shuffles"][i].passes
    ]


def verdict(decoded: dict, event: int) -> str:
    """What the sliding-window detector did with one labeled event.

    ``found`` when a sequence overlapping it meets the whole published
    definition; ``failed the shuffle`` when one met the duration and
    dispersion criteria but not the significance test; ``declined`` when a
    candidate was assembled but none was accepted; ``missed`` when no
    candidate overlaps it at all. Read from the data, so a panel title cannot
    drift away from the panel.
    """
    if reported_overlapping(decoded, event):
        return "found"
    if accepted_overlapping(decoded, event):
        return "failed the shuffle"
    if candidates_overlapping(decoded, event):
        return "declined"
    return "missed"


def duration_ratio(decoded: dict, event: int) -> float:
    """How much longer the detector calls an event than the event really is.

    Over the longest accepted sequence overlapping it, whether or not it then
    passes the shuffle, which never moves a sequence's bounds. Raises when
    there is none, so a title cannot be formatted from a detection that did
    not happen.
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
        return f"{label}: {duration_ratio(decoded, event):.1f}x long"
    if what == "failed the shuffle":
        return f"{label}: {duration_ratio(decoded, event):.1f}x long, fails shuffle"
    return f"{label}: {what}"


def shuffle_of(decoded: dict, event: int):
    """The shuffle result of the longest accepted sequence overlapping ``event``."""
    accepted = accepted_overlapping(decoded, event)
    longest = max(accepted, key=lambda i: decoded["candidates"].duration_s[i])
    return decoded["shuffles"][longest]


def draws_from_passing(result) -> int:
    """How many more draws below the observed value than a pass allows.

    As the reference tests it, a half passes when ``(1 + k) / (n + 1) <= 0.05``
    with ``k`` the draws strictly below the observed value: at most 4 of 100.
    Zero means the candidate passes; one means a single draw decided it.
    """
    allowed = int(np.floor(SIGNIFICANCE_LEVEL * (result.n_shuffles + 1) + 1e-9)) - 1
    worst = max(
        int(np.count_nonzero(result.null_spread_cm < result.observed_spread_cm)),
        int(np.count_nonzero(result.null_jump_cm < result.observed_jump_cm)),
    )
    return max(0, worst - allowed)


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


def outside_pieces(decoded: dict, sequence: int) -> np.ndarray:
    """Windows of a sequence that were kept but belong to none of its pieces.

    A single kept window between two runs is dropped before the merge, as the
    reference drops it, yet still counts toward the merged sequence's
    dispersion. Returns window indices.
    """
    candidates = decoded["candidates"]
    start, stop = candidates.runs[sequence]
    in_piece = np.zeros(stop - start + 1, dtype=bool)
    for piece in candidates.pieces[sequence]:
        first, last = candidates.pre_merge_runs[piece]
        in_piece[first - start : last - start + 1] = True
    kept = candidates.decisions.gates.kept[start : stop + 1]
    return start + np.flatnonzero(kept & ~in_piece)


def stray_window_dispersions(decoded: dict) -> tuple[float, float, int]:
    """The stray-window event's dispersion with and without the windows
    outside its pieces, and the sequence they come from.

    Over the longest accepted sequence overlapping it, the one the panel shows.
    """
    accepted = accepted_overlapping(decoded, STRAY_WINDOW_EVENT)
    if not accepted:
        raise RuntimeError(
            "The caption says the stray-window event is called replay; no "
            "accepted sequence overlaps it. Rewrite the caption."
        )
    candidates = decoded["candidates"]
    sequence = max(accepted, key=lambda i: candidates.duration_s[i])
    start, stop = candidates.runs[sequence]
    center = candidates.decisions.center_index[start : stop + 1].copy()
    center[np.isin(np.arange(start, stop + 1), outside_pieces(decoded, sequence))] = (
        NO_CENTER
    )
    center[~candidates.decisions.gates.kept[start : stop + 1]] = NO_CENTER
    return (
        float(candidates.dispersion_cm[sequence]),
        dispersion_cm(center, decoded["bin_distance_cm"]),
        sequence,
    )


def stray_window_shuffle_sentence(decoded: dict) -> str:
    """What the significance test made of the stray-window event, from the data."""
    _, _, sequence = stray_window_dispersions(decoded)
    result = decoded["shuffles"][sequence]
    halves = (
        "passes both halves"
        if result.passes
        else "fails it"
        if not (
            result.p_spread <= SIGNIFICANCE_LEVEL or result.p_jump <= SIGNIFICANCE_LEVEL
        )
        else "passes the spread half and fails the jump half"
        if result.p_spread <= SIGNIFICANCE_LEVEL
        else "passes the jump half and fails the spread half"
    )
    return (
        f"Put through the significance test it {halves} (p = "
        f"{result.p_spread:.2f} on spread and {result.p_jump:.2f} on jump)"
    )


def stray_window_state_space_sentence(decoded: dict) -> str:
    """What the state-space detector reported over the stray-window event."""
    start_s, end_s = decoded["event_times"][STRAY_WINDOW_EVENT]
    overlapping = [
        (max(a, start_s), min(b, end_s))
        for a, b in decoded["state_space_intervals"]
        if a < end_s and start_s < b
    ]
    if not overlapping:
        return "The state-space detector reports no non-local event over it."
    total_ms = 1000 * sum(b - a for a, b in overlapping)
    return (
        f"The state-space detector reports {len(overlapping)} short non-local "
        f"event{'s' if len(overlapping) > 1 else ''} over it, {total_ms:.0f} ms "
        f"of its {1000 * (end_s - start_s):.0f} ms in all."
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
        CONTINUOUS_OUTBOUND_EVENT: "failed the shuffle",
        CONTINUOUS_INBOUND_EVENT: "found",
        SHORT_PATH_EVENT: "declined",
        REMOTE_STATIONARY_EVENT: "declined",
    }
    if not accepted_overlapping(decoded, SHORT_REPLAY_EVENT):
        raise RuntimeError(
            "The caption says the 80 ms replay meets the duration and dispersion "
            "criteria; no accepted sequence overlaps it. Rewrite the caption."
        )
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
    zoomed = (
        CONTINUOUS_OUTBOUND_EVENT,
        SHORT_REPLAY_EVENT,
        SHORT_PATH_EVENT,
        REMOTE_STATIONARY_EVENT,
    )
    full, pieces, sequence = stray_window_dispersions(decoded)
    stray_start = decoded["stray_window"].stray_window_start_s
    stray_index = int(
        np.argmin(
            np.abs(decoded["midpoint_s"] - stray_start - config.window_duration_s / 2)
        )
    )
    if not (
        full > config.dispersion_threshold_cm > pieces
        and outside_pieces(decoded, sequence).tolist() == [stray_index]
    ):
        raise RuntimeError(
            "The caption says the stray-window event is called replay only "
            "because of its one window outside every piece; the data say "
            f"dispersion {full:.1f} cm with it, {pieces:.1f} cm without, and "
            f"windows {outside_pieces(decoded, sequence).tolist()} outside the "
            f"pieces against the placed one, {stray_index}. Rewrite the caption."
        )
    if not np.isfinite(bound_error_ms(decoded, zoomed)):
        raise RuntimeError(
            "The caption says the state-space detector reports all four zoomed "
            "events; it does not. Rewrite the caption."
        )
    for event in (CONTINUOUS_OUTBOUND_EVENT, CONTINUOUS_INBOUND_EVENT):
        if not local_fraction_of_event(decoded, event) > 0.5:
            raise RuntimeError(
                "The caption says most of each reported continuous replay is the "
                f"animal's own place; {EVENT_LABELS[event]!r} is not. Rewrite it."
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


def plot_overview(ax, decoded: dict) -> None:
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

    reported = is_reported(decoded)
    for sequence, (start, stop) in enumerate(candidates.runs):
        accepted = reported[sequence]
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
        Patch(color="0.75", label="posterior, panels b-f"),
        Line2D(
            [],
            [],
            color=VERMILION,
            marker="*",
            ms=4.0,
            lw=0,
            label="kept window outside every piece",
        ),
        Line2D([], [], color=VERMILION, lw=2.4, label=f"event, {SLIDING}"),
        Line2D([], [], color="0.6", lw=1.2, label="candidate, rejected"),
        Line2D([], [], color=BLUE, lw=2.4, label=f"event, {STATE_SPACE}"),
    ]
    ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=5,
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


def plot_event(ax, decoded: dict, event: int) -> None:
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
    # a kept window outside every piece still counts toward dispersion
    stray = np.concatenate(
        [outside_pieces(decoded, i) for i in overlapping] or [np.array([], int)]
    )
    ax.scatter(
        midpoint_s[stray],
        center_cm[stray],
        s=22,
        marker="*",
        color=VERMILION,
        lw=0,
        zorder=5,
    )
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

    reported = is_reported(decoded)
    for i in overlapping:
        start, stop = candidates.runs[i]
        accepted = reported[i]
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


def timing_series(
    summary: pd.DataFrame, detector: str, edge: str, statistic: str
) -> tuple[np.ndarray, np.ndarray]:
    """One timing curve: true duration in ms, and the error statistic in ms."""
    rows = summary[
        (summary["summary"] == "timing_by_duration")
        & (summary.measure == f"{detector}_{edge}_error_ms_{statistic}")
    ].sort_values("duration_s")
    return 1000 * rows.duration_s.to_numpy(), rows.value.to_numpy()


def plot_edge_error(ax, summary: pd.DataFrame, edge: str) -> None:
    """Reported minus true start or end, for replays both procedures found."""
    ax.axhline(0.0, color="0.6", lw=0.7, ls=":")
    for detector, color, label in (
        ("sliding", VERMILION, SLIDING),
        ("state_space", BLUE, STATE_SPACE),
    ):
        duration_ms, median = timing_series(summary, detector, edge, "median")
        _, low = timing_series(summary, detector, edge, "q25")
        _, high = timing_series(summary, detector, edge, "q75")
        ax.fill_between(duration_ms, low, high, color=color, alpha=0.2, lw=0)
        ax.plot(
            duration_ms, median, color=color, lw=1.0, marker="o", ms=2.5, label=label
        )
    ax.set_xlabel("duration (ms)")
    ax.set_ylabel(f"{edge} error (ms)")
    ax.set_title("onset" if edge == "start" else "offset", fontsize=6)


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
    # every bin gets a full slot, so an empty one is not drawn on the spine
    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.set_ylim(0.85, 2.6)
    ax.set_xlabel("distance from the animal (cm)")
    ax.set_ylabel("reported / actual duration")
    ax.set_title("reported duration by distance", fontsize=6)


def output_color(detector: str) -> str:
    return VERMILION if detector.startswith("sliding") else BLUE


def paired_part_changes(controls: pd.DataFrame, detector: str) -> tuple[int, int, int]:
    """Trials, over every jump and chain, where a detector recovers fewer and more
    parts inside the event than alone, and all trials."""
    return tuple(
        int(
            sum(
                controls_value(controls, "parts", measure, condition, None, detector)
                for condition in MULTIPART_CONDITIONS
            )
        )
        for measure in (
            "n_trials_fewer_in_context",
            "n_trials_more_in_context",
            "n_trials",
        )
    )


def controls_value(
    controls: pd.DataFrame,
    summary: str,
    measure: str,
    condition: str,
    level: float | None,
    detector: str,
) -> float:
    """One value of the controls summary, refusing to guess between several."""
    rows = controls[
        (controls["summary"] == summary)
        & (controls.measure == measure)
        & (controls.condition == condition)
        & (controls.detector == detector)
    ]
    rows = (
        rows[rows.level.isna()]
        if level is None
        else rows[np.isclose(rows.level, level)]
    )
    if len(rows) != 1:
        raise RuntimeError(
            f"The controls summary holds {len(rows)} rows for {summary}, {measure}, "
            f"{condition}, {level}, {detector}; expected one."
        )
    return float(rows.value.iloc[0])


def no_replay_minutes(controls: pd.DataFrame) -> float:
    rows = controls[
        (controls["summary"] == "no_replay") & (controls.measure == "immobile_minutes")
    ]
    return float(rows.value.iloc[0])


def controls_levels(controls: pd.DataFrame, condition: str) -> list[float]:
    rows = controls[
        (controls["summary"] == "parts") & (controls.condition == condition)
    ]
    return sorted(rows.level.dropna().unique())


def plot_bursts(ax, controls: pd.DataFrame) -> None:
    """Share of bursts with an event reported on them, for each output."""
    width = 0.2
    for group, condition in enumerate(("placeless_burst", "jumping_burst")):
        for offset, (detector, label, alpha) in enumerate(OUTPUTS):
            share = controls_value(
                controls, "bursts", "called_rate", condition, None, detector
            )
            ax.bar(
                group + (offset - 1.5) * width,
                100 * share,
                width=width,
                color=output_color(detector),
                alpha=alpha,
                label=label if group == 0 else None,
            )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["place-less\nburst", "jumping\nburst"])
    # headroom: the no-replay note above the bars, the legend above it
    ax.set_ylim(0, 165)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("bursts called (%)")
    ax.set_title("bursts", fontsize=6)
    minutes = no_replay_minutes(controls)
    ax.text(
        0.5,
        118,
        f"no replay: no event from any\noutput in {minutes:.1f} min of rest",
        ha="center",
        va="center",
        fontsize=4.5,
        color="0.3",
    )
    ax.legend(fontsize=4.5, frameon=False, loc="upper left", handlelength=1.0, ncol=2)


def plot_carving(
    ax,
    controls: pd.DataFrame,
    conditions: tuple[str, ...],
    xlabel: str,
    title: str,
    tick,
    legend: bool,
    show_y: bool = True,
) -> None:
    """How each detector carved multi-part events, pooled over ``conditions``."""
    levels = controls_levels(controls, conditions[0])
    width = 0.36
    for group, level in enumerate(levels):
        for offset, detector in enumerate(
            ("sliding_published", "state_space_canonical")
        ):
            x = group + (offset - 0.5) * width
            bottom = 0.0
            for label, shade, name in SEGMENTATION_STYLE:
                share = np.mean(
                    [
                        controls_value(
                            controls,
                            "parts",
                            f"segmentation_{label}",
                            c,
                            level,
                            detector,
                        )
                        for c in conditions
                    ]
                )
                ax.bar(
                    x,
                    100 * share,
                    bottom=100 * bottom,
                    width=width,
                    color=shade,
                    edgecolor=output_color(detector),
                    linewidth=0.8,
                    label=name if legend and group == 0 and offset == 0 else None,
                )
                bottom += share
            recovery = np.mean(
                [
                    controls_value(
                        controls, "parts", "part_recovery", c, level, detector
                    )
                    for c in conditions
                ]
            )
            # the two detectors' numbers at two heights, so neighbours do not run
            # together
            ax.text(
                x,
                102 + 9 * offset,
                f"{100 * recovery:.0f}",
                ha="center",
                va="bottom",
                fontsize=4,
                color=output_color(detector),
            )
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([tick(level) for level in levels])
    # headroom for the recovery numbers and, above them, the legend
    ax.set_ylim(0, 162)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel(xlabel)
    if show_y:
        ax.set_ylabel("trials (%)")
    else:
        ax.set_yticklabels([])
    ax.set_title(title, fontsize=6)
    if legend:
        ax.legend(
            fontsize=4.5, frameon=False, loc="upper left", handlelength=1.0, ncol=2
        )


def heldout_rate(
    heldout: pd.DataFrame,
    fold: str,
    estimator: str,
    classes: tuple[str, ...],
    truth: str,
    measure: str = "support_rate",
    suffix: str = "",
) -> tuple[float, int]:
    """A held-out rate over our detector's events, pooled over event classes.

    Each event is counted once, as our detector bounds it: a shared event is
    also scored as the sliding window bounds it, and pooling both would count
    it twice. A rate over informative events is weighted by how many are
    informative, a share of all events by how many there are.

    Returns
    -------
    rate : float
        NaN when no event counts.
    n : int
        The events the rate rests on.
    """
    rows = heldout[
        (heldout["summary"] == "by_class")
        & (heldout.fold == fold)
        & (heldout.estimator == estimator)
        & (heldout.detector == "state_space")
        & heldout.event_class.isin(classes)
        & (heldout.truth == truth)
    ].pivot_table(index="event_class", columns="measure", values="value")
    weight = "n_events_all" if measure == "informative_share" else f"n_events{suffix}"
    if rows.empty or weight not in rows or rows[weight].sum() == 0:
        return float("nan"), 0
    n = rows[weight].fillna(0)
    return float((rows[f"{measure}{suffix}"].fillna(0) * n).sum() / n.sum()), int(
        n.sum()
    )


def plot_heldout_support(ax, heldout: pd.DataFrame) -> None:
    """Share of informative events with held-out support, by kind and fold."""
    width = 0.26
    for group, (_, classes, truth) in enumerate(HELDOUT_GROUPS):
        for offset, (fold, hatch) in enumerate(FOLD_HATCH.items()):
            x = group + (offset - 1) * width
            rate, n = heldout_rate(heldout, fold, "state_space", classes, truth)
            other, _ = heldout_rate(heldout, fold, "sliding_window", classes, truth)
            if not n:
                ax.text(x, 3, "-", ha="center", va="bottom", fontsize=4.5, color="0.5")
                continue
            ax.bar(
                x,
                100 * rate,
                width=width,
                color=BLUE,
                alpha=0.35 + 0.25 * offset,
                hatch=hatch,
                edgecolor="white" if hatch else None,
                linewidth=0,
                label=f"fold {fold}" if group == 0 else None,
            )
            ax.plot(
                x,
                100 * other,
                "o",
                color=VERMILION,
                ms=2.2,
                label="sliding window's content"
                if group == 0 and offset == 0
                else None,
            )
            ax.text(x, 100 * rate + 3, f"{n}", ha="center", va="bottom", fontsize=3.8)
    ax.set_xticks(range(len(HELDOUT_GROUPS)))
    ax.set_xticklabels([label for label, _, _ in HELDOUT_GROUPS])
    ax.set_ylim(0, 150)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("held-out support (%)")
    ax.set_title("held-out support, informative events", fontsize=6)
    ax.legend(fontsize=4.5, frameon=False, loc="upper right", handlelength=1.0, ncol=2)


def plot_heldout_coverage(ax, heldout: pd.DataFrame) -> None:
    """Share of each kind of replay the evaluation cells can judge at all."""
    width = 0.26
    for group, (_, classes, truth) in enumerate(HELDOUT_GROUPS[:2]):
        for offset, (fold, hatch) in enumerate(FOLD_HATCH.items()):
            share, _ = heldout_rate(
                heldout, fold, "state_space", classes, truth, "informative_share"
            )
            ax.bar(
                group + (offset - 1) * width,
                100 * share,
                width=width,
                color=BLUE,
                alpha=0.35 + 0.25 * offset,
                hatch=hatch,
                edgecolor="white" if hatch else None,
                linewidth=0,
            )
    ax.set_xticks(range(2))
    ax.set_xticklabels([label for label, _, _ in HELDOUT_GROUPS[:2]])
    ax.set_ylim(0, 100)
    ax.set_ylabel("informative (%)")
    ax.set_title("events the evaluation cells can judge", fontsize=6)


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
    n_neurons: int,
    config: MemorylessSequenceConfig,
    summary: pd.DataFrame,
    controls: pd.DataFrame,
    heldout: pd.DataFrame,
):
    decoded = simulate_and_decode(n_neurons, config)
    decoded["state_space_intervals"] = state_space_intervals(decoded)
    check_caption_claims(decoded, config)

    set_figure_defaults(context="paper")
    # this project's defaults turn constrained layout on, so the grid must not
    # also set its own margins: the two fight and the panels collapse
    fig = plt.figure(
        figsize=(TWO_COLUMN, TWO_COLUMN * 1.72), dpi=450, constrained_layout=True
    )
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.05, hspace=0.09)
    # one column of rows, each row with its own columns, so the layout solves
    # each row's widths on its own; one shared grid couples every row's
    # decorations and collapses
    grid = GridSpec(len(ROW_HEIGHTS), 1, figure=fig, height_ratios=ROW_HEIGHTS)

    ax_overview = fig.add_subplot(grid[0])
    plot_overview(ax_overview, decoded)

    # one line each, so these titles sit at the same height as every other
    # row's and the panel letters can share one offset; the verdict in each
    # is read from the data
    outcomes = (
        (CONTINUOUS_OUTBOUND_EVENT, "continuous"),
        (SHORT_REPLAY_EVENT, "80 ms replay"),
        (SHORT_PATH_EVENT, "30 cm path"),
        (REMOTE_STATIONARY_EVENT, "remote, held still"),
        (STRAY_WINDOW_EVENT, "one far window"),
    )
    zooms = []
    events = grid[1].subgridspec(1, len(outcomes))
    for column, (event, label) in enumerate(outcomes):
        ax = fig.add_subplot(events[column])
        plot_event(ax, decoded, event)
        title = (
            f"{label}: "
            + {"found": "found", "failed the shuffle": "fails shuffle"}.get(
                verdict(decoded, event), verdict(decoded, event)
            )
            if event == STRAY_WINDOW_EVENT
            else outcome_title(decoded, event, label)
        )
        ax.set_title(title, fontsize=5)
        if column:
            ax.set_yticklabels([])
        zooms.append(ax)
    zooms[0].set_ylabel("decoded place (cm)")

    heatmaps = grid[2].subgridspec(1, 2, wspace=0.06)
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

    timing = grid[3].subgridspec(1, 3)
    ax_onset = fig.add_subplot(timing[0])
    plot_edge_error(ax_onset, summary, "start")
    ax_onset.legend(fontsize=5, frameon=False, loc="center right", handlelength=1.3)
    ax_offset = fig.add_subplot(timing[1])
    plot_edge_error(ax_offset, summary, "end")
    ax_distance = fig.add_subplot(timing[2])
    plot_absorption(ax_distance, summary)

    carving = grid[4].subgridspec(1, 4)
    ax_bursts = fig.add_subplot(carving[0])
    plot_bursts(ax_bursts, controls)
    ax_jumps = fig.add_subplot(carving[1])
    plot_carving(
        ax_jumps,
        controls,
        ("jump_120ms", "jump_200ms"),
        "jump (cm)",
        "mid-event jump",
        lambda level: f"{level:.0f}",
        legend=True,
    )
    ax_continuing = fig.add_subplot(carving[2])
    plot_carving(
        ax_continuing,
        controls,
        ("chain_continuing",),
        "gap (ms)",
        "chain, continuing",
        lambda level: f"{1000 * level:.0f}",
        legend=False,
        show_y=False,
    )
    evidence = grid[5].subgridspec(1, 2, width_ratios=[2.2, 1])
    ax_support = fig.add_subplot(evidence[0])
    plot_heldout_support(ax_support, heldout)
    ax_coverage = fig.add_subplot(evidence[1])
    plot_heldout_coverage(ax_coverage, heldout)
    ax_elsewhere = fig.add_subplot(carving[3])
    plot_carving(
        ax_elsewhere,
        controls,
        ("chain_elsewhere",),
        "gap (ms)",
        "chain, elsewhere",
        lambda level: f"{1000 * level:.0f}",
        legend=False,
        show_y=False,
    )

    place_panel_letters(
        fig,
        [
            [ax_overview],
            zooms,
            [ax_sliding, ax_state],
            [ax_onset, ax_offset, ax_distance],
            [ax_bursts, ax_jumps, ax_continuing, ax_elsewhere],
            [ax_support, ax_coverage],
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
    passes = (
        summary[summary["summary"] == "shuffle_pass_rate_by_duration"]
        .pivot(index="duration_s", columns="measure", values="value")
        .sort_index()
    )
    if not passes.sliding_jump_pass.iloc[0] < passes.sliding_jump_pass.iloc[-1]:
        raise RuntimeError(
            "The caption says the jump half passes fewer of the shortest replays "
            "than of the longest; the summary's pass rates by duration disagree."
        )
    _, sliding_onset = timing_series(summary, "sliding", "start", "median")
    _, state_space_onset = timing_series(summary, "state_space", "start", "median")
    _, sliding_offset = timing_series(summary, "sliding", "end", "median")
    _, state_space_offset = timing_series(summary, "state_space", "end", "median")
    if not ((sliding_onset < 0).all() and (sliding_offset > 0).all()):
        raise RuntimeError(
            "The caption says the sliding window starts early and ends late at "
            "every duration; the summary's median timing errors disagree."
        )
    _, sliding_offset_q25 = timing_series(summary, "sliding", "end", "q25")
    _, sliding_offset_q75 = timing_series(summary, "sliding", "end", "q75")
    if not (
        sliding_offset_q75 - sliding_offset > sliding_offset - sliding_offset_q25
    ).all():
        raise RuntimeError(
            "The caption says the sliding window's late ends carry a long tail; "
            "its offset errors are not skewed late at every duration."
        )
    if not (
        (sliding_onset < state_space_onset).all()
        and (sliding_offset > state_space_offset).all()
    ):
        raise RuntimeError(
            "The caption contrasts the sliding window's early starts and late ends "
            "with the state-space detector's; at some duration they do not differ "
            "that way."
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


def pct_range(values) -> str:
    """A range of percentages for the caption, one value when both ends agree."""
    low, high = pct(min(values)), pct(max(values))
    return low if low == high else f"{low} to {high}"


def continuous_shuffle_sentence(decoded: dict) -> str:
    """What the significance test did with the two continuous replays."""
    parts = []
    for event, name in (
        (CONTINUOUS_OUTBOUND_EVENT, "outbound"),
        (CONTINUOUS_INBOUND_EVENT, "inbound"),
    ):
        result = shuffle_of(decoded, event)
        p = f"p = {result.p_spread:.3f} on spread and {result.p_jump:.3f} on jump"
        if result.passes:
            parts.append(f"the {name} continuous replay passes it ({p})")
        else:
            margin = draws_from_passing(result)
            parts.append(
                f"the {name} continuous replay fails it ({p}), "
                f"{margin} draw{'s' if margin != 1 else ''} from passing"
            )
    return parts[0][0].upper() + parts[0][1:] + ", and " + parts[1] + "."


def short_replay_shuffle_sentence(decoded: dict) -> str:
    """What the significance test did with the 80 ms replay, from the data."""
    result = shuffle_of(decoded, SHORT_REPLAY_EVENT)
    p = f"p = {result.p_spread:.3f} on spread and {result.p_jump:.3f} on jump"
    if result.passes:
        return f"It passes the significance test ({p})."
    margin = draws_from_passing(result)
    closeness = (
        "one draw from passing, so another seed could pass it"
        if margin == 1
        else f"{margin} draws from passing"
    )
    return (
        f"The significance test rejects it ({p}, against 0.05), {closeness}, "
        "and the published method would not report it."
    )


def check_controls_claims(controls: pd.DataFrame) -> None:
    """The caption's claims about the controls, checked against their summary."""
    no_replay = controls[
        (controls["summary"] == "no_replay") & (controls.measure == "n_events")
    ]
    if len(no_replay) != len(OUTPUTS) or (no_replay.value != 0).any():
        raise RuntimeError(
            "The caption says no output reported an event where there was no "
            "replay; the controls summary disagrees."
        )
    for condition in ("jump_120ms", "jump_200ms"):
        if controls_value(
            controls,
            "parts",
            "segmentation_spanning",
            condition,
            80.0,
            "sliding_published",
        ):
            raise RuntimeError(
                "The caption says the sliding window never reports one event across "
                "an 80 cm jump; the controls summary disagrees."
            )
        for level in controls_levels(controls, condition):
            if (
                controls_value(
                    controls,
                    "parts",
                    "segmentation_spanning",
                    condition,
                    level,
                    "state_space_canonical",
                )
                < 0.5
            ):
                raise RuntimeError(
                    "The caption says the state-space detector reports one event over "
                    "both parts of most jumps at every size; the summary disagrees."
                )
    shortest = controls_levels(controls, "chain_continuing")[0]
    for detector in ("sliding_published", "state_space_canonical"):
        if (
            controls_value(
                controls,
                "parts",
                "segmentation_spanning",
                "chain_continuing",
                shortest,
                detector,
            )
            < 0.5
        ):
            raise RuntimeError(
                "The caption says both detectors report one event over a chain at "
                "the shortest gap; the controls summary disagrees."
            )
    for detector in ("sliding_published", "state_space_canonical"):
        within, alone = (
            np.mean(
                [
                    controls_value(controls, "parts", measure, c, level, detector)
                    for c in MULTIPART_CONDITIONS
                    for level in controls_levels(controls, c)
                ]
            )
            for measure in ("part_recovery", "part_recovery_alone")
        )
        if within <= alone:
            raise RuntimeError(
                "The caption says pooled recovery is higher inside the event than "
                "alone; the controls summary disagrees."
            )
        fewer, more, _ = paired_part_changes(controls, detector)
        if more <= fewer:
            raise RuntimeError(
                "The caption says more trials gain parts inside the event than "
                "lose them; the controls summary disagrees."
            )
    longest = controls_levels(controls, "chain_continuing")[-1]
    for detector in ("sliding_published", "state_space_canonical"):
        if controls_value(
            controls,
            "parts",
            "segmentation_spanning",
            "chain_continuing",
            longest,
            detector,
        ):
            raise RuntimeError(
                "The caption says neither detector reports one event over a chain "
                "at the longest gap; the controls summary disagrees."
            )
    for level in controls_levels(controls, "chain_elsewhere"):
        if controls_value(
            controls,
            "parts",
            "segmentation_spanning",
            "chain_elsewhere",
            level,
            "sliding_published",
        ):
            raise RuntimeError(
                "The caption says the sliding window never reports one event over a "
                "chain whose parts are far apart; the controls summary disagrees."
            )


def check_heldout_claims(heldout: pd.DataFrame) -> None:
    """The caption's claims about held-out evidence, checked against its summary."""
    by_class = heldout[heldout["summary"] == "by_class"]
    if (by_class.event_class == "comparator_only").any():
        raise RuntimeError(
            "The caption says the sliding window reported no event our detector "
            "missed; the held-out summary holds comparator-only events."
        )
    for fold in FOLD_HATCH:
        for estimator in ("state_space", "sliding_window"):
            rate, n = heldout_rate(
                heldout, fold, estimator, ("ours_only", "shared"), "no_content"
            )
            if n and rate > 0:
                raise RuntimeError(
                    "The caption says no informative place-less burst event has "
                    "held-out support; the held-out summary disagrees."
                )
    local = by_class[(by_class.truth == "local") & (by_class.measure == "n_events_all")]
    if local.value.sum() > 0:
        raise RuntimeError(
            "The caption says no event of pure local coding reached the held-out "
            "test; the held-out summary holds some."
        )
    if not sum(
        heldout_rate(
            heldout, fold, "state_space", ("ours_only", "shared"), "no_content"
        )[1]
        for fold in FOLD_HATCH
    ):
        raise RuntimeError(
            "The caption states a place-less burst result; no place-less burst "
            "event is informative, so there is none to state."
        )
    for fold in FOLD_HATCH:
        for estimator in ("state_space", "sliding_window"):
            rate, n = heldout_rate(
                heldout, fold, estimator, ("ours_only",), "continuous_replay"
            )
            if n and rate < 0.8:
                raise RuntimeError(
                    "The caption says most informative replays only our detector "
                    "reported have held-out support in every fold; the summary "
                    "disagrees."
                )


def write_caption(
    decoded: dict,
    summary: pd.DataFrame,
    controls: pd.DataFrame,
    heldout: pd.DataFrame,
    config: MemorylessSequenceConfig,
    path: Path,
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
    durations_ms, sliding_onset = timing_series(summary, "sliding", "start", "median")
    _, sliding_offset = timing_series(summary, "sliding", "end", "median")
    _, sliding_offset_q75 = timing_series(summary, "sliding", "end", "q75")
    state_space_edges = np.concatenate(
        [
            timing_series(summary, "state_space", edge, "median")[1]
            for edge in ("start", "end")
        ]
    )
    passes = summary[summary["summary"] == "shuffle_pass_rate_by_duration"]
    jump_pass = (
        passes[passes.measure == "sliding_jump_pass"]
        .set_index("duration_s")
        .value.sort_index()
    )
    check_summary_claims(summary)
    check_controls_claims(controls)
    check_heldout_claims(heldout)

    def support(fold, classes, truth, estimator="state_space"):
        return heldout_rate(heldout, fold, estimator, classes, truth)[0]

    ours_support = [support(f, ("ours_only",), "continuous_replay") for f in "AB"]
    shared_support = [support(f, ("shared",), "continuous_replay") for f in "ABC"]
    jumping_support = [support(f, ("ours_only", "shared"), "non_local") for f in "AB"]
    informative = {
        name: [
            heldout_rate(
                heldout,
                f,
                "state_space",
                classes,
                "continuous_replay",
                "informative_share",
            )[0]
            for f in "AB"
        ]
        for name, classes in (("ours", ("ours_only",)), ("shared", ("shared",)))
    }
    agreement = heldout[
        (heldout["summary"] == "agreement") & (heldout.measure == "estimator_agreement")
    ].value
    placeless = [
        heldout_rate(heldout, f, "state_space", ("ours_only", "shared"), "no_content")
        for f in FOLD_HATCH
    ]
    placeless_loose = [
        heldout_rate(
            heldout,
            f,
            "state_space",
            ("ours_only", "shared"),
            "no_content",
            suffix="_floor2",
        )
        for f in FOLD_HATCH
    ]
    loose_supported = sum(round(rate * n) for rate, n in placeless_loose if n)

    def called(condition, detector):
        return controls_value(
            controls, "bursts", "called_rate", condition, None, detector
        )

    def burst_share(share):
        rows = controls[
            (controls["summary"] == "bursts")
            & (controls.measure == share)
            & (controls.condition == "placeless_burst")
        ]
        return float(rows.value.min()), float(rows.value.max())

    def carving(conditions, label, detector):
        """Share of trials carved this way at each level, pooled over conditions."""
        return np.array(
            [
                np.mean(
                    [
                        controls_value(
                            controls,
                            "parts",
                            f"segmentation_{label}",
                            c,
                            level,
                            detector,
                        )
                        for c in conditions
                    ]
                )
                for level in controls_levels(controls, conditions[0])
            ]
        )

    def recovery(condition, detector):
        return np.array(
            [
                controls_value(
                    controls, "parts", "part_recovery", condition, level, detector
                )
                for level in controls_levels(controls, condition)
            ]
        )

    minutes = no_replay_minutes(controls)
    n_recordings = int(
        controls[
            (controls["summary"] == "provenance") & (controls.measure == "n_recordings")
        ].value.iloc[0]
    )

    def mean_recovery(detector, measure):
        return np.mean(
            [
                controls_value(controls, "parts", measure, c, level, detector)
                for c in MULTIPART_CONDITIONS
                for level in controls_levels(controls, c)
            ]
        )

    sliding_fewer, sliding_more, sliding_trials = paired_part_changes(
        controls, "sliding_published"
    )
    ours_fewer, ours_more, _ = paired_part_changes(controls, "state_space_canonical")
    fragmented_low, fragmented_high = burst_share("fragmented_share")
    jumps = ("jump_120ms", "jump_200ms")
    spanning_jumps = carving(jumps, "spanning", "state_space_canonical")
    sliding_spanning = carving(jumps, "spanning", "sliding_published")
    elsewhere_missing = carving(
        ("chain_elsewhere",), "parts_missing", "sliding_published"
    )
    elsewhere_split = carving(
        ("chain_elsewhere",), "one_per_part", "state_space_canonical"
    )
    continuing = {
        (label, detector): carving(("chain_continuing",), label, detector)
        for label in ("spanning", "one_per_part")
        for detector in ("sliding_published", "state_space_canonical")
    }
    gaps_ms = 1000 * np.array(controls_levels(controls, "chain_continuing"))

    path.write_text(
        rewrap(
            f"""# {path.stem}

**Two replay detectors on replays whose shape is known.** The sliding-window
detector decodes each 80 ms window on its own, with no model of how one window
follows the next, and keeps or drops windows on the spread of their posterior
and on the jump to the next window. The state-space detector infers the whole
event at once and carries the animal's own place as a competing hypothesis.
Throughout, both detectors read one encoding model, fitted once; the
sliding-window detector decodes it as the reference does, with 1e-3 Hz added to
every place field, and is held to the published definition as adapted here,
its significance test included: each accepted sequence is decoded again
{DEFAULT_N_SHUFFLES} times with the place fields swapped between cells, and
must beat those draws on both its mean posterior spread and its mean jump. The
adaptations: a window's spread is a root-mean-square distance along the track
about a place bin, cut at {config.spread_threshold_cm:.0f} cm, where the
paper's open-field statistic is a two-dimensional image moment that a 10 cm
cutoff does not carry over to exactly; centers are place bins, so the pairs of windows whose
jump the original sets aside as zero are read from the spikes instead;
windows step {1000 * config.stride_s:.0f} ms rather than 5 ms; and the
immobility cutoff is {config.speed_threshold_cm_per_s:.0f} rather than
5 cm/s. The figure compares the events two complete pipelines report, each with
its own acceptance criteria, so a difference between them cannot be put down
to how either finds candidates alone.

**(a-f) One recording with labeled events.** The simulation carries a
continuous outbound replay, a fragmented replay, a continuous inbound replay
and a silent stretch; a remote stationary event, a 30 cm replay, an 80 ms
replay and a still event with one window decoded far away were written into
the spike trains here and are marked with a star.

**(a)** The whole recording. Green is where the animal is; dots are the decoded
place of each window, kept or rejected; yellow marks a labeled event; bars below
the axis are the events each detector reports and the candidates the
sliding-window detector rejected.

**(b-f)** Five events, with the posterior in grey; bars below the axis are
the events each detector reports, and a sequence that fails the significance
test is drawn as rejected. The sliding-window detector finds the continuous
replay and gives it
{duration_ratio(decoded, CONTINUOUS_OUTBOUND_EVENT):.1f} times its true length,
because {100 * local_fraction_of_event(decoded, CONTINUOUS_OUTBOUND_EVENT):.0f}%
of the event it reports is the animal's own place, absorbed by the merge rule.
It also accepts the 80 ms replay, at
{duration_ratio(decoded, SHORT_REPLAY_EVENT):.1f} times its length with
{100 * local_fraction_of_event(decoded, SHORT_REPLAY_EVENT):.0f}% local coding
in it: the window alone puts a floor under any duration it can report.
{short_replay_shuffle_sentence(decoded)} It
assembles a candidate from the 30 cm replay and another from the remote
stationary event and declines both for too little dispersion
({best_dispersion_cm(decoded, SHORT_PATH_EVENT):.0f} and
{best_dispersion_cm(decoded, REMOTE_STATIONARY_EVENT):.0f} cm against the
{config.dispersion_threshold_cm:.0f} cm it requires). The state-space detector
reports all four, with every bound within
{bound_error_ms(decoded, (CONTINUOUS_OUTBOUND_EVENT, SHORT_REPLAY_EVENT, SHORT_PATH_EVENT, REMOTE_STATIONARY_EVENT)):.0f} ms
of the truth. {continuous_shuffle_sentence(decoded)} Most of each reported
continuous replay is the animal's own place either way: the test asks whether
a sequence is spatially coherent, not whether it is somewhere other than the
animal. **(f)** A still event at the animal's own place, with spikes placed by
hand so that one 80 ms window holds only two spikes of a cell
{abs(decoded["stray_window_remote_cm"] - decoded["stray_window_local_cm"]):.0f} cm
away (star). The windows either side of it are too broad to keep, but the one
after it centres near it, so it passes the jump rule; as a run of one window it
is dropped before the merge, and the still parts either side merge across it.
Its dispersion still counts toward the merged event's, as in the reference:
{stray_window_dispersions(decoded)[0]:.1f} cm with it and
{stray_window_dispersions(decoded)[1]:.1f} cm without, against the
{config.dispersion_threshold_cm:.0f} cm required, so a still event is called
replay on the strength of one window. {stray_window_shuffle_sentence(decoded)}.
{stray_window_state_space_sentence(decoded)}
On the first recorded pilot epoch this pattern decided 2 of its 4 replay calls,
both of which passed the test; the event table now carries the dispersion
without such windows beside the published one.

**(g-k) {int(overall["n_cells"])} replays of chosen shape.** Made-up
replays were written into the spike trains of a simulated linear-track
recording, one at a time, each with a path length, duration, firing rate and
distance from the animal's own place that we chose. Because the right answer is known, what each
detector reports can be scored against it. A detector counts as reporting a
replay only when what it returns is that replay: on its path throughout, and
not covering far more ground than the path does.

**(g, h)** Percent of those {int(overall["n_cells"])} replays reported, by the
path the replay travels and how long it lasts; both detectors are scored the
same way against the same known answer. The comparison that isolates detection
uses a shared target: the replays the published definition can accept at all,
a path over about 42 cm and a duration over 100 ms,
{pct(domain["n_cells"] / overall["n_cells"])} of the cells, with both detectors
held to that definition's duration and dispersion criteria. There the
sliding-window detector reports {pct(domain["sliding_matches_before_shuffle"])}
before its significance test and the state-space detector
{pct(domain["state_space_replay_matches"])}. The significance test, which only
the sliding-window detector is put through, brings it to
{pct(domain["sliding_matches"])}; the state-space detector's own acceptance
criteria, in place of the published ones, bring it to
{pct(domain["state_space_matches"])}. Over the whole grid, each under its own
criteria, the two read {pct(overall["sliding_matches"])} and
{pct(overall["state_space_matches"])}; that gap combines differences in
detection with differences in what each counts as a replay. Over the whole
grid the sliding-window detector reports
{pct(overall["sliding_matches_before_shuffle"])} before its significance test,
and {pct(overall["sliding_matches_spread_only"])} with the spread half alone,
the only half the detection script's own filter could fail; the paper's
analyses read both halves, from a corrected script. Of the replays it reported
before the test,
{pct(overall["sliding_shuffle_spread_pass_rate"])} pass the spread half and
{pct(overall["sliding_shuffle_jump_pass_rate"])} the jump half; a short replay
covers its path in few windows, so its own jumps are large, and the jump half
passes {pct(jump_pass.iloc[0])} of {1000 * jump_pass.index[0]:.0f} ms replays
against {pct(jump_pass.iloc[-1])} of {1000 * jump_pass.index[-1]:.0f} ms ones.
The test keeps the reference's conventions where they carry over, counting only
draws strictly below the observed value and passing each half at p ≤ 0.05, and
translates its zero-jump rule to binned centers.
{pct(overall["found_by_state_space_only"])} were reported by the
state-space detector and not by the sliding-window detector, counting the
{pct(overall["missed_failed_shuffle"])} it reported and then failed on the
significance test; {pct(overall["found_by_sliding_window_only"], 2)} went the
other way ({pct(overall["found_by_sliding_window_only_before_shuffle"], 2)}
before the significance test).

The two detectors reach their limits on different quantities. The state-space
detector is limited by duration, which is spike count, which is the data. The
sliding-window detector is limited by path length, through its own dispersion
criterion: a replay must spread more than 12 cm about its own centre, and a
30 cm path cannot, however well it is decoded.
Held to that same criterion our state-space detector also reports nothing below
45 cm, so this is the published replay definition acting, not the window-by-window
inference.

**(i, j)** Where each detector puts a replay's start (i) and end (j): reported
minus true, in ms, median line and interquartile band by true duration, over
the replays both detectors reported, the sliding window's before its
significance test, which never moves a sequence's bounds. The sliding window's
bounds are its first and last window midpoints, and its 80 ms windows reach
40 ms beyond each. The state-space detector's median start and end sit within
{np.abs(state_space_edges).max():.0f} ms of the truth at every duration. The
sliding window starts a median {-sliding_onset.max():.0f} to
{-sliding_onset.min():.0f} ms early and ends {sliding_offset.min():.0f} to
{sliding_offset.max():.0f} ms late, and its late ends carry a long tail, the
upper quartile reaching {sliding_offset_q75.max():.0f} ms; panel (k) shows
how much of a reported event is the animal's own place. A {durations_ms[0]:.0f} ms
replay's reported start is a median {-sliding_onset[0]:.0f} ms early and its end a
median {sliding_offset[0]:.0f} ms late, and its median reported duration is about
{1000 * shortest.duration_s * shortest.value:.0f} ms, the floor panel (c) shows for
one event.

**(k)** Reported duration over actual duration against how far the replay sits
from the animal's own place. Numbers give the share of the reported event's
80 ms windows that decode the animal's place rather than a replayed one. That
absorption of local coding reaches about 80 cm from the animal and is what
drives the largest over-reporting, as in panel (b); beyond it the window
floor of panels (i, j) still leaves the reported duration too long. Over the whole sweep,
{pct(overall["sliding_accepted_but_not_the_injection"])} of the
sliding-window detector's accepted events are not the replay that was put in.

**(l-o) Activity that is not one clean replay**, written into the same recording
and scored by the same rule; our detector is read both ways, its non-local
events (continuous and fragmented states together) and the continuous state
alone. **(l)** Share of bursts with an event reported on them. A place-less
burst, every cell firing at one rate unrelated to its field, carries no
spatial content, so any event on it is false; a jumping burst represents a
random place every 15 ms, so calling it non-local is right and calling it a
sequence is not. Our detector's non-local events cover
{pct(called("placeless_burst", "state_space_canonical"))} of place-less
bursts, which spend {pct(fragmented_low)} to {pct(fragmented_high)} of their
time in the fragmented state; the continuous state alone covers
{pct(called("placeless_burst", "state_space_continuous"))}, and the sliding
window {pct(called("placeless_burst", "sliding_published"))}
({pct(called("placeless_burst", "sliding_accepted"))} before its significance
test). These pool the three peak rates, 20, 60 and 160 Hz; at 20 Hz a
place-less burst fires about as fast as the local coding it replaces, and our
non-local events still cover
{pct(controls_value(controls, "bursts", "called_rate_at_20hz", "placeless_burst", None, "state_space_canonical"))}
of them. On jumping bursts our non-local events cover
{pct(called("jumping_burst", "state_space_canonical"))}, which is right; the
continuous state alone covers {pct(called("jumping_burst", "state_space_continuous"))}
and the sliding window {pct(called("jumping_burst", "sliding_published"))}
({pct(called("jumping_burst", "sliding_accepted"))} before its test), each
calling a sequence that is not there. Over {minutes:.1f} minutes of immobility
with no replay, from {n_recordings} simulated recordings that pause in the same
places at the track's end, no output reported any event: fewer than
{3 / minutes:.1f} per minute, at 95% confidence, and only there.
**(m)** Two 60 cm parts, the second starting the given distance behind where
the first ended, with no gap in time, pooled over 120 and 200 ms parts, their
far end held 45 cm from where the animal rests whatever the jump; bars
give how each detector carved the event (vermilion outline the sliding window
under the published definition, blue our non-local events), and the numbers
above them the share of parts recovered. Our detector reports one event over
both parts in {pct(spanning_jumps.min())} to {pct(spanning_jumps.max())} of
trials at every jump; the sliding window does so in up to
{pct(sliding_spanning.max())} of trials for jumps up to 40 cm and never across
80 cm. **(n, o)** Two 150 ms, 60 cm replays with a silent gap, the second
continuing from the first's end (n) or starting 140 cm away from it (o), both
reaching no nearer the animal than 20 cm.
Continuing, at a {gaps_ms[0]:.0f} ms gap both detectors report one event over
both parts ({pct(continuing["spanning", "sliding_published"][0])} of trials
for the sliding window, {pct(continuing["spanning", "state_space_canonical"][0])}
for ours); at {gaps_ms[-1]:.0f} ms neither does, and they report one event per
part in {pct(continuing["one_per_part", "sliding_published"][-1])} and
{pct(continuing["one_per_part", "state_space_canonical"][-1])}. Far apart, the
sliding window never reports one event over both, and misses a part in
{pct(elsewhere_missing.min())} to {pct(elsewhere_missing.max())} of trials; our
detector reports one event per part in {pct(elsewhere_split[0])} of trials at a
{gaps_ms[0]:.0f} ms gap and {pct(elsewhere_split[-1])} at {gaps_ms[-1]:.0f} ms.
Each part is also decoded alone, with exactly the spikes it had in its event
and the recording's own spikes around it; a part counts as recovered only where
the event's positions over that part sit on its path. Across jumps and chains,
pooled recovery is higher inside the event than alone: the sliding window
recovers {pct(mean_recovery("sliding_published", "part_recovery"))} of parts
inside the event against
{pct(mean_recovery("sliding_published", "part_recovery_alone"))} alone, and ours
{pct(mean_recovery("state_space_canonical", "part_recovery"))} against
{pct(mean_recovery("state_space_canonical", "part_recovery_alone"))}. Paired by
trial, the gain is not uniform: the sliding window recovers fewer parts inside
the event than alone in {sliding_fewer} of {sliding_trials} trials and more in
{sliding_more}, ours fewer in {ours_fewer} and more in {ours_more}.

**(p, q) Held-out evidence.** The simulated cells are split into discovery and
evaluation cells three ways: two interleaved folds with their roles swapped (A,
B), each half tiling the track, and one whose evaluation cells cover only its
lower half (C, hatched). Both detectors are fitted and run on the discovery
cells alone, over every injected replay, controls trial and no-replay stretch.
Events are counted once each, as our detector bounds them. An event's content,
estimated from the discovery cells by our decoder (bars) or by the sliding
window (dots), is scored on the evaluation cells against the animal's own
position and against 100 permutations of the evaluation fields, spikes and
silence both counting, and has support when it beats both (p ≤ 0.05). **(p)**
Share of events with support, among those along whose content the evaluation
cells are expected to fire at least 5 spikes; numbers give how many. True
replays that only our detector reported have support in
{pct_range(ours_support)} of cases in folds A and B
and {pct(support("C", ("ours_only",), "continuous_replay"))} in fold C; shared
ones in {pct_range(shared_support)}. No
informative place-less burst event has support
({sum(n for _, n in placeless)} of them), and at a floor of 2 expected spikes
{loose_supported} of {sum(n for _, n in placeless_loose)} do; jumping bursts,
which do represent places, have support in {pct_range(jumping_support)}. Neither
detector reported an event from discovery cells on the no-replay stretches, so
no event of pure local coding reached the test. The two content estimates agree
on {pct(agreement.min())} of the events informative under both, or more. The sliding window
reported no event our detector missed, so there is no comparator-only class to
test. **(q)** Share of each kind of replay the evaluation cells can judge at
all: {pct_range(informative["ours"])} of those only our detector reported,
{pct_range(informative["shared"])} of shared ones, in folds A and B. The rest are too short or too
sparse for a dozen evaluation cells, so an event without support is mostly one
that cannot be judged, not one shown false. Held-out support tests spatial
information, not sequence, and is read with these error rates, not alone.

**Caveats.** Sorted spikes, a linear track, no artifacts. In (g-k) the animal
rests at one end of the track and pauses nowhere else, so distance from its
place is fixed by where the replay is put and cannot be varied independently.
Replays run at constant speed with Poisson firing from the same place fields
the encoding model was fitted on, so there is no model mismatch and the picture
is optimistic. Every cell of (g-k) holds a replay, so those panels measure how
often each detector reports a replay that is there; (l) is where they are tested
on activity that holds none. The significance
test is the reference's place-cell identity shuffle, which needs sorted cells;
recorded data, decoded clusterless, are tested with an encoding-pair shuffle
instead.

Produced by `scripts/figures/generate_memoryless_comparison.py` from
`docs/design/replay_detection_comparison/detection_sweep_summary.csv`.
"""
        )
        + "\n"
    )


def read_summary(path: Path, producer: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No summary at {path}. Run `python {producer}` first.")
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-neurons", type=int, default=25)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--controls-summary", type=Path, default=CONTROLS_SUMMARY_PATH)
    parser.add_argument("--heldout-summary", type=Path, default=HELDOUT_SUMMARY_PATH)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("paper/figures/memoryless_comparison")
    )
    args = parser.parse_args()

    config = MemorylessSequenceConfig(sampling_frequency=500.0)
    summary = read_summary(args.summary, "scripts/qa/sweep_memoryless_detection.py")
    controls = read_summary(
        args.controls_summary, "scripts/qa/sweep_memoryless_controls.py"
    )
    heldout = read_summary(
        args.heldout_summary, "scripts/qa/sweep_memoryless_heldout.py"
    )
    fig, decoded = make_figure(args.n_neurons, config, summary, controls, heldout)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_figure("memoryless_comparison", base_dir=args.output_dir)
    plt.close(fig)
    write_caption(
        decoded,
        summary,
        controls,
        heldout,
        config,
        args.output_dir / "memoryless_comparison_caption.md",
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
