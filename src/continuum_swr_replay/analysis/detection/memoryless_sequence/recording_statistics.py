"""What the two replay detectors made of the labeled events of one simulated recording.

The supplementary comparison figure's examples come from one decoded recording
(:func:`~continuum_swr_replay.analysis.detection.memoryless_sequence.comparison_simulation.simulate_and_decode`)
holding events whose bounds and content are known. The functions here read that
decode and return what each detector did with an event: the sliding-window
detector's verdict, how much longer it reports an event than it is, how much of
its report is the animal's own place, how far its significance test is from a
pass, and how far the state-space detector's bounds are from the truth.
:func:`recording_numbers` gathers the ones the detection summary reports
(``continuum_swr_replay.reported.detection`` names and describes them), and the
figure (``scripts/figures/generate_supp_memoryless_comparison.py``) draws and checks
the same functions' values.

Every function but :func:`draws_from_passing` (which takes the result of
:func:`shuffle_of`) takes ``decoded``, :func:`simulate_and_decode`'s dict; the
state-space detector's events are its ``"state_space_intervals"``. An event is a row of
``decoded["event_times"]``; a sequence is an index into
``decoded["candidates"]``. Times are in seconds unless a name says ms, distances
in cm, shares in [0, 1].
"""

import numpy as np

from continuum_swr_replay.analysis.detection.memoryless_sequence.comparison_simulation import (
    REMOTE_STATIONARY_EVENT,
    REPLAY_EVENTS,
    SHORT_PATH_EVENT,
    SHORT_REPLAY_EVENT,
    SWEEP_EVENT,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.shuffle import (
    SIGNIFICANCE_LEVEL,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.sweep import overlaps
from continuum_swr_replay.analysis.detection.memoryless_sequence.sweep_replays import (
    SAME_PLACE_CM,
)

__all__ = [
    "accepted_overlapping",
    "best_dispersion_cm",
    "bound_error_ms",
    "candidates_overlapping",
    "decoded_center_cm",
    "dominant_state",
    "draws_from_passing",
    "duration_ratio",
    "duration_ratios",
    "is_reported",
    "local_fraction",
    "local_fraction_of_event",
    "recording_numbers",
    "reported_overlapping",
    "shuffle_of",
    "state_space_overlapping",
    "state_space_duration_ratio",
    "verdict",
]


def candidates_overlapping(decoded: dict, event: int) -> list[int]:
    """Every candidate sequence that overlaps a labeled event, accepted or not."""
    midpoint_s = decoded["midpoint_s"]
    bounds = tuple(decoded["event_times"][event])
    return [
        i
        for i, (start, stop) in enumerate(decoded["candidates"].runs)
        if overlaps((midpoint_s[start], midpoint_s[stop]), bounds)
    ]


def state_space_overlapping(decoded: dict, event: int) -> list[tuple[float, float]]:
    """The state-space model's events that overlap a labeled event, in time order."""
    bounds = tuple(decoded["event_times"][event])
    return [
        interval
        for interval in decoded["state_space_intervals"]
        if overlaps(interval, bounds)
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


def _longest_accepted(decoded: dict, event: int) -> int:
    """The longest accepted sequence overlapping ``event``.

    Raises
    ------
    RuntimeError
        If no accepted sequence overlaps it.
    """
    accepted = accepted_overlapping(decoded, event)
    if not accepted:
        raise RuntimeError(f"No accepted sequence overlaps event {event}.")
    return max(accepted, key=lambda i: decoded["candidates"].duration_s[i])


def duration_ratio(decoded: dict, event: int) -> float:
    """How much longer the detector calls an event than the event really is.

    Over the longest accepted sequence overlapping it, whether or not it then
    passes the shuffle, which never moves a sequence's bounds.

    Raises
    ------
    RuntimeError
        If no accepted sequence overlaps the event, so a number is never
        formatted from a detection that did not happen.
    """
    event_start, event_end = decoded["event_times"][event]
    longest = _longest_accepted(decoded, event)
    return float(decoded["candidates"].duration_s[longest] / (event_end - event_start))


def shuffle_of(decoded: dict, event: int):
    """The shuffle result of the longest accepted sequence overlapping ``event``."""
    return decoded["shuffles"][_longest_accepted(decoded, event)]


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


def local_fraction(decoded: dict, sequence: int) -> float:
    """Share of a sequence's kept windows whose center is the animal's own place:
    closer than :data:`~.sweep_replays.SAME_PLACE_CM` to where the animal is at the
    window's midpoint.

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
    return float(np.mean(np.abs(center_cm[kept] - animal_cm[kept]) < SAME_PLACE_CM))


def local_fraction_of_event(decoded: dict, event: int) -> float:
    """Share of the accepted event overlapping ``event`` that is local coding.

    Over the longest accepted sequence, the one :func:`duration_ratio` reads.
    """
    return local_fraction(decoded, _longest_accepted(decoded, event))


def best_dispersion_cm(decoded: dict, event: int) -> float:
    """The largest dispersion any candidate overlapping ``event`` reached; NaN
    when none overlaps it."""
    dispersion = decoded["candidates"].dispersion_cm
    return max(
        (float(dispersion[i]) for i in candidates_overlapping(decoded, event)),
        default=float("nan"),
    )


def dominant_state(decoded: dict, event: int) -> str:
    """The state the state-space model gives most probability over an event.

    The mean of each state's probability over the decode samples inside the
    event, largest first; the name is the detector's own (``"Local"``,
    ``"Non-Local Fragmented"``, ...).
    """
    start_s, end_s = decoded["event_times"][event]
    inside = (decoded["time"] >= start_s) & (decoded["time"] <= end_s)
    mean = decoded["state_probabilities"][inside].mean(axis=0)
    return decoded["state_names"][int(np.argmax(mean))]


def bound_error_ms(decoded: dict, events: tuple[int, ...]) -> float:
    """Largest error in either bound of the state-space model's reports, in ms.

    For each event, the start of the first state-space interval overlapping
    it against the event's start, and the end of the last against its end
    (:func:`state_space_overlapping`). NaN when the state-space detector
    reports nothing over one of ``events``.
    """
    worst = 0.0
    for event in events:
        start_s, end_s = decoded["event_times"][event]
        overlapping = state_space_overlapping(decoded, event)
        if not overlapping:
            return float("nan")
        worst = max(
            worst,
            abs(overlapping[0][0] - start_s),
            abs(overlapping[-1][1] - end_s),
        )
    return 1000 * worst


def state_space_duration_ratio(decoded: dict, event: int) -> float:
    """How long the state-space model's report of an event is, over the event's
    own duration.

    From the start of the first state-space interval overlapping the event to
    the end of the last (:func:`state_space_overlapping`). NaN when the model
    reports nothing over the event.
    """
    start_s, end_s = decoded["event_times"][event]
    overlapping = state_space_overlapping(decoded, event)
    if not overlapping:
        return float("nan")
    return float((overlapping[-1][1] - overlapping[0][0]) / (end_s - start_s))


def duration_ratios(decoded: dict, event: int) -> tuple[float, float]:
    """How long each detector's report of an event is, over the event's own
    duration: the sliding window's (:func:`duration_ratio`), then the
    state-space model's (:func:`state_space_duration_ratio`). NaN for a
    detector that does not report the event."""
    sliding = (
        duration_ratio(decoded, event)
        if verdict(decoded, event) == "found"
        else float("nan")
    )
    return sliding, state_space_duration_ratio(decoded, event)


def recording_numbers(decoded: dict) -> dict[str, float | int]:
    """The numbers the detection summary reports about the labeled recording.

    Parameters
    ----------
    decoded : dict
        :func:`simulate_and_decode`'s dict.

    Returns
    -------
    numbers : dict
        Durations in ms, distances in cm, shares in [0, 1]. The prefixes
        ``sweep`` and ``short_replay`` name Figure 1's sweep and the 80 ms
        replay; each value is over the longest accepted sequence overlapping
        the event.

        - ``sweep_``, ``short_replay_duration_ratio``: its reported over its
          true duration (:func:`duration_ratio`).
        - ``sweep_``, ``short_replay_local_fraction``: the share of its kept
          windows decoding the animal's own place
          (:func:`local_fraction_of_event`).
        - ``*_p_spread``, ``*_p_jump``, for both: the two halves of its
          significance test.
        - ``sweep_draws_from_passing``: :func:`draws_from_passing`.
        - ``short_path_dispersion_cm``, ``remote_stationary_dispersion_cm``: the
          largest dispersion of a candidate over each (:func:`best_dispersion_cm`).
        - ``state_space_bound_error_ms``: :func:`bound_error_ms` over
          :data:`~.comparison_simulation.REPLAY_EVENTS`.

    Raises
    ------
    RuntimeError
        If no accepted sequence overlaps one of the two events read.
    """
    sweep = shuffle_of(decoded, SWEEP_EVENT)
    short = shuffle_of(decoded, SHORT_REPLAY_EVENT)
    return {
        "sweep_duration_ratio": duration_ratio(decoded, SWEEP_EVENT),
        "sweep_local_fraction": local_fraction_of_event(decoded, SWEEP_EVENT),
        "sweep_p_spread": float(sweep.p_spread),
        "sweep_p_jump": float(sweep.p_jump),
        "sweep_draws_from_passing": draws_from_passing(sweep),
        "short_replay_duration_ratio": duration_ratio(decoded, SHORT_REPLAY_EVENT),
        "short_replay_local_fraction": local_fraction_of_event(
            decoded, SHORT_REPLAY_EVENT
        ),
        "short_replay_p_spread": float(short.p_spread),
        "short_replay_p_jump": float(short.p_jump),
        "short_path_dispersion_cm": best_dispersion_cm(decoded, SHORT_PATH_EVENT),
        "remote_stationary_dispersion_cm": best_dispersion_cm(
            decoded, REMOTE_STATIONARY_EVENT
        ),
        "state_space_bound_error_ms": bound_error_ms(decoded, REPLAY_EVENTS),
    }
