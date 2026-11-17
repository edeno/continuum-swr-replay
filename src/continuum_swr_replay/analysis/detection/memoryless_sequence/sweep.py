"""The simulated recording the memoryless sweeps share, and each detector's run on it.

The injected-replay sweep (``sweep_replays``), its controls (``sweep_controls``),
the held-out check (``detector_comparison.heldout_sweep``) and the reference
check (``scripts/checks/compare_memoryless_to_reference.py``) read one simulated
recording (:func:`build_recording`, sorted spikes on a linear track, both
detectors fitted on its non-event data), place events in its immobile stretches
clear of its own events (:func:`usable_intervals`), read one grid of injected
replays (:data:`PATH_CM`, :data:`DURATION_S`, :data:`PEAK_RATE_HZ`,
:data:`MIDPOINT_CM`; the combinations inside the track are :func:`replay_grid`),
and run the two detectors over a stretch of it (:func:`decode_window`) the same
way: the memoryless sliding window (:func:`run_sliding_window`, whose events
under the published definition are :func:`published_events`) and the switching
state-space model (:func:`run_state_space`), each reporting its events as
:class:`ReportedEvent`.
"""

from dataclasses import dataclass

import numpy as np
import xarray as xr
from non_local_detector.simulate.sorted_spikes_simulation import TRACK_HEIGHT

from continuum_swr_replay.analysis.detection.memoryless_sequence.comparison_simulation import (
    CONFIG,
    N_NEURONS,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.detect import (
    CandidateSequences,
    detect_candidate_sequences,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.likelihood import (
    sorted_spikes_epoch_log_likelihood,
    window_midpoints_s,
    with_reference_rate_floor,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.shuffle import (
    DEFAULT_N_SHUFFLES,
    ShuffleResult,
    ShuffleSeed,
    shuffle_candidate_sorted,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.simulation import (
    immobile_intervals,
    without,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.window_statistics import (
    aggregate_windows,
    normalize_windows,
)
from continuum_swr_replay.analysis.detection.simulated_run import (
    estimate_state_space_model,
    simulate_run,
)
from continuum_swr_replay.analysis.event_definition.nonlocal_event_rule import (
    NON_LOCAL_STATES,
    detect_nonlocal_events,
)
from continuum_swr_replay.decoding.encoding_model import interior_bin_distances_cm
from continuum_swr_replay.decoding.results import ENCODING_KEY

__all__ = [
    "DECODE_MARGIN_S",
    "DURATION_S",
    "MATCH_TOLERANCE_CM",
    "MIDPOINT_CM",
    "PATH_CM",
    "PEAK_RATE_HZ",
    "PLACEMENT_MARGIN_S",
    "SHUFFLE_RUN_SEED",
    "ReportedEvent",
    "SlidingWindowRun",
    "StateSpaceRun",
    "build_recording",
    "decode_window",
    "nearest_bin",
    "overlaps",
    "published_events",
    "replay_grid",
    "run_sliding_window",
    "run_state_space",
    "usable_intervals",
]

PATH_CM = (0.0, 10.0, 20.0, 30.0, 45.0, 60.0, 90.0, 130.0)
"""How far the represented position travels; 0 is a place held still."""
DURATION_S = (0.04, 0.06, 0.08, 0.12, 0.20, 0.30, 0.50)
PEAK_RATE_HZ = (20.0, 60.0, 160.0)
MIDPOINT_CM = (30.0, 60.0, 90.0, 120.0, 150.0)
"""Where the represented path is centered, which sets its distance from the animal."""

DECODE_MARGIN_S = 1.0
"""How much time each side of an injected event is decoded.

Wide, so that the window does not cap how long an over-called event can be
reported as. It may reach past the immobile stretch; the speed gate rejects
those windows, which is what it is for."""

PLACEMENT_MARGIN_S = 0.25
"""How much immobile room the injection needs on each side.

Enough for a window of full support plus context, and small enough that the
pieces left after cutting out the recording's own events can still hold an
event."""

SHUFFLE_RUN_SEED = 20260922
"""Seeds the place-cell identity shuffle; each cell's draws are keyed on the cell too."""

MATCH_TOLERANCE_CM = 20.0
"""How far a detector's decoded position may sit from the injected one.

Generous against a 2 cm grid and the place fields' smoothing kernel, so it separates
"reported the injected path" from "reported somewhere else entirely" rather
than grading accuracy."""


def build_recording(seed: int = 0, cells: np.ndarray | None = None):
    """One simulated recording, with both detectors fitted on its non-event data.

    The recording is Figure 1's simulated run (``simulated_run.simulate_run``)
    and the state-space model is Figure 1's, fitted as a real decode is
    (``simulated_run.estimate_state_space_model``): the real decodes' settings,
    the place fields and the transitions estimated by EM.

    ``seed`` is the simulator's; 0 is the recording every sweep reads. With
    ``cells``, the detectors are fitted on those cells alone, in that order,
    and the recording's ``spike_times`` are theirs; ``all_spike_times`` keeps
    every cell's.
    """
    run = simulate_run(n_neurons=N_NEURONS, seed=seed)
    all_spike_times = run.spike_times
    spike_times = all_spike_times
    if cells is not None:
        spike_times = [all_spike_times[cell] for cell in cells]

    detector, _ = estimate_state_space_model(
        run.time, run.position, spike_times, ~run.is_pattern
    )

    # One set of place fields, not two. The detector's own encoding model holds
    # every entry the sliding-window likelihood needs, so handing it over makes
    # the two procedures read the same object rather than two fits of the same
    # data, which differed by about 4% of peak firing rate.
    encoding_model = detector.encoding_model_[ENCODING_KEY]
    environment = encoding_model["environment"]

    return {
        "time": run.time,
        "position": run.position,
        "speed": run.speed,
        "spike_times": spike_times,
        "all_spike_times": all_spike_times,
        # the run's patterns, in time order: (n_patterns, 2), s
        "event_times": np.array(list(run.patterns.values())),
        "encoding_model": encoding_model,
        # the same fields, decoded as the reference decodes them, with its
        # floor under every one; the state-space detector reads them as fitted
        "sliding_encoding_model": with_reference_rate_floor(encoding_model, CONFIG),
        "bin_distance_cm": interior_bin_distances_cm(environment),
        "detector": detector,
        "place_bin_centers_cm": np.asarray(environment.place_bin_centers_).ravel()[
            np.asarray(environment.is_track_interior_).ravel()
        ],
    }


def replay_grid() -> list[tuple[float, float, float, float]]:
    """Every injected replay of the grid whose path lies on the track.

    Returns
    -------
    list of (path_cm, midpoint_cm, duration_s, peak_rate_hz)
        In the order :data:`PATH_CM`, :data:`MIDPOINT_CM`, :data:`DURATION_S`,
        :data:`PEAK_RATE_HZ`, the last varying fastest; a combination whose
        path, centered on its midpoint, would run off either end of the track
        is left out.
    """
    return [
        (path_cm, midpoint_cm, duration_s, rate_hz)
        for path_cm in PATH_CM
        for midpoint_cm in MIDPOINT_CM
        for duration_s in DURATION_S
        for rate_hz in PEAK_RATE_HZ
        if 0.0 <= midpoint_cm - path_cm / 2
        and midpoint_cm + path_cm / 2 <= TRACK_HEIGHT
    ]


def usable_intervals(recording: dict) -> np.ndarray:
    """The recording's immobile stretches with its own labeled events cut out.

    Stretches still for at least 1 s, less each labeled event and
    :data:`PLACEMENT_MARGIN_S` either side of it. The events are cut out rather
    than their stretches discarded: every stretch holds one, so discarding them
    would leave nothing.

    Returns
    -------
    ndarray, shape (n_intervals, 2)
        Start and end times in seconds.
    """
    return without(
        immobile_intervals(
            recording["time"],
            recording["speed"],
            CONFIG.speed_threshold_cm_per_s,
            minimum_duration_s=1.0,
        ),
        recording["event_times"],
        margin_s=PLACEMENT_MARGIN_S,
    )


def decode_window(
    recording: dict,
    bounds: tuple[float, float],
    margin_s: float = DECODE_MARGIN_S,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The recording's samples from ``margin_s`` before to ``margin_s`` after ``bounds``.

    Both ends are inclusive.

    Parameters
    ----------
    recording : dict
        As :func:`build_recording` returns it.
    bounds : (float, float)
        Start and end of the stretch, in seconds.
    margin_s : float, optional
        Decoded on each side of it; :data:`DECODE_MARGIN_S` around an
        injection, 0 for a stretch decoded as it is.

    Returns
    -------
    time, position, speed : ndarray, shape (n_window,)
    """
    start_s, end_s = bounds
    mask = (recording["time"] >= start_s - margin_s) & (
        recording["time"] <= end_s + margin_s
    )
    return (
        recording["time"][mask],
        recording["position"][mask],
        recording["speed"][mask],
    )


def overlaps(reported: tuple[float, float], bounds: tuple[float, float]) -> bool:
    """Strict, positive-duration overlap. The same rule for both detectors."""
    return reported[0] < bounds[1] and bounds[0] < reported[1]


@dataclass(frozen=True)
class ReportedEvent:
    """One event a detector reports, with the decoded positions it is scored on.

    Attributes
    ----------
    start_s, end_s : float
        Its bounds: first and last window midpoint for the sliding window,
        first and last sample for the state-space detector.
    decoded_time_s, decoded_cm : ndarray, shape (n_decoded,)
        Where the event puts the replay, and when: the kept windows' centers
        at their midpoints, or the non-local posterior's most probable place
        at each sample.
    """

    start_s: float
    end_s: float
    decoded_time_s: np.ndarray
    decoded_cm: np.ndarray


@dataclass(frozen=True)
class SlidingWindowRun:
    """What the sliding-window detector reports over one decoded stretch.

    Every candidate meeting the duration and dispersion criteria is
    ``accepted`` and tested against the place-cell identity shuffle; a
    candidate's draws are seeded by the stretch's key and its first window,
    so which other candidates are tested does not change its result.
    ``time`` is the decoded stretch's samples, on the decode grid; the windows
    were decoded on the detector's grid derived from it.
    """

    time: np.ndarray
    midpoint_s: np.ndarray
    posterior: np.ndarray
    candidates: CandidateSequences
    accepted: list[int]
    events: dict[int, ReportedEvent]
    shuffles: dict[int, ShuffleResult]


def run_sliding_window(recording, injected, window, cell_key) -> SlidingWindowRun:
    """Decode a stretch window by window, assemble, and test every accepted sequence."""
    time, position, speed = window
    result = sorted_spikes_epoch_log_likelihood(
        time, position, injected, recording["sliding_encoding_model"], CONFIG
    )
    posterior = normalize_windows(
        aggregate_windows(
            result.log_likelihood, CONFIG.window_n_samples, CONFIG.stride_n_samples
        )
    )
    midpoint_s = window_midpoints_s(result.time, CONFIG)
    window_speed = np.interp(midpoint_s, time, speed)
    candidates = detect_candidate_sequences(
        posterior, midpoint_s, recording["bin_distance_cm"], window_speed, CONFIG
    )
    accepted = [int(i) for i in np.flatnonzero(candidates.is_replay)]

    # The event's decoded positions are its kept windows, as the detector's own
    # dispersion reads them: a window inside a merged gap was rejected and is
    # not a decoded position, whatever center its wide posterior happens to
    # have.
    events = {}
    for i in accepted:
        run_start, run_stop = candidates.runs[i]
        centers = candidates.decisions.center_index[run_start : run_stop + 1]
        times = midpoint_s[run_start : run_stop + 1]
        present = candidates.assembly.kept[run_start : run_stop + 1]
        events[i] = ReportedEvent(
            start_s=float(midpoint_s[run_start]),
            end_s=float(midpoint_s[run_stop]),
            decoded_time_s=times[present],
            decoded_cm=recording["place_bin_centers_cm"][centers[present]],
        )

    # the reference's own shuffle, for every accepted candidate
    shuffles = {
        i: shuffle_candidate_sorted(
            int(candidates.runs[i, 0]),
            int(candidates.runs[i, 1]),
            time,
            injected,
            recording["sliding_encoding_model"],
            recording["bin_distance_cm"],
            CONFIG,
            seed=ShuffleSeed(run=SHUFFLE_RUN_SEED, session=cell_key),
            n_shuffles=DEFAULT_N_SHUFFLES,
        )
        for i in accepted
    }
    return SlidingWindowRun(
        time=time,
        midpoint_s=midpoint_s,
        posterior=posterior,
        candidates=candidates,
        accepted=accepted,
        events=events,
        shuffles=shuffles,
    )


def published_events(run: SlidingWindowRun) -> list[ReportedEvent]:
    """The sliding window's events under the published definition.

    The accepted candidates (duration and dispersion criteria met) that also
    pass the place-cell identity shuffle, in candidate order.
    """
    return [run.events[i] for i in run.accepted if run.shuffles[i].passes]


def nearest_bin(recording, position_cm: np.ndarray) -> np.ndarray:
    """Interior place bin nearest each decoded position.

    The state-space posterior carries one position axis per state, so its
    argmax is an index into a concatenation. Mapping back to interior bins
    lets the state-space trajectory be measured with the same
    `dispersion_cm` over the same track-graph distances as the sliding
    window's.
    """
    centers = recording["place_bin_centers_cm"]
    return np.abs(centers[None, :] - np.asarray(position_cm)[:, None]).argmin(axis=1)


@dataclass(frozen=True)
class StateSpaceRun:
    """What the switching state-space model reports over one decoded stretch.

    Attributes
    ----------
    time : ndarray, shape (n_time,)
    p_continuous, p_fragmented : ndarray, shape (n_time,)
        Probability of each non-local state.
    canonical : list of ReportedEvent
        The project's non-local events
        (``nonlocal_event_rule.detect_nonlocal_events``): continuous plus
        fragmented above threshold for the minimum duration, the animal
        immobile throughout.
    continuous_only : list of ReportedEvent
        The same rule on the continuous state alone.
    non_local_posterior : ndarray, shape (n_time, n_bins)
        Where the non-local states put the replay: the two states summed at
        each interior place bin, each row renormalized to one.
    """

    time: np.ndarray
    p_continuous: np.ndarray
    p_fragmented: np.ndarray
    canonical: list[ReportedEvent]
    continuous_only: list[ReportedEvent]
    non_local_posterior: np.ndarray


def run_state_space(recording, injected, window) -> StateSpaceRun:
    """Decode a stretch with the state-space model and read off its events."""
    time, position, speed = window
    results = recording["detector"].predict(
        position_time=time, position=position, spike_times=injected, time=time
    )
    everything = np.asarray(results.acausal_state_probabilities.values)
    if not np.isfinite(everything).all() or everything.max() <= 0.0:
        raise RuntimeError(
            "The state-space model returned no probability mass. Its state "
            "probabilities must sum to one at every time. A detector fitted with "
            "estimate_parameters does this silently when predict is given another "
            "time grid."
        )
    continuous, fragmented = (
        results.acausal_state_probabilities.sel(states=state).squeeze()
        for state in NON_LOCAL_STATES
    )
    p_continuous, p_fragmented = np.asarray(continuous), np.asarray(fragmented)

    # where the non-local states put the animal, to score this the same way
    posterior = results["acausal_posterior"]
    is_non_local = np.isin(
        np.asarray(posterior.coords["state"].values),
        NON_LOCAL_STATES,
    )
    bin_position_cm = np.asarray(posterior.coords["position"].values)[is_non_local]
    non_local = np.asarray(posterior.values)[:, is_non_local]

    def events(probability: xr.DataArray) -> list[ReportedEvent]:
        detected, _ = detect_nonlocal_events(probability, speed, time)
        reported = []
        for start_s, end_s in zip(
            detected["start_time"], detected["end_time"], strict=True
        ):
            a, b = np.searchsorted(time, [start_s, end_s])
            reported.append(
                ReportedEvent(
                    start_s=float(time[a]),
                    end_s=float(time[b]),
                    decoded_time_s=time[a : b + 1],
                    decoded_cm=bin_position_cm[np.argmax(non_local[a : b + 1], axis=1)],
                )
            )
        return reported

    # each non-local state's columns onto the interior bins they stand for
    to_bin = np.zeros((bin_position_cm.size, recording["place_bin_centers_cm"].size))
    to_bin[np.arange(bin_position_cm.size), nearest_bin(recording, bin_position_cm)] = 1
    content = non_local @ to_bin
    return StateSpaceRun(
        time=time,
        p_continuous=p_continuous,
        p_fragmented=p_fragmented,
        canonical=events(continuous + fragmented),
        continuous_only=events(continuous),
        non_local_posterior=content / content.sum(axis=1, keepdims=True),
    )
