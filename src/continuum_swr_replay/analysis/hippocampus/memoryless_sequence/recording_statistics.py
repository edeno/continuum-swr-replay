"""What the two replay detectors made of the labeled events of one simulated recording.

The memoryless comparison figure's upper panels decode one recording
(:func:`~continuum_swr_replay.analysis.hippocampus.memoryless_sequence.comparison_simulation.simulate_and_decode`)
holding events whose bounds and content are known. The functions here read that
decode and return what each detector did with an event: the sliding-window
detector's verdict, how much longer it reports an event than it is, how much of
its report is the animal's own place, how far its significance test is from a
pass, and how far the state-space detector's bounds are from the truth.
:func:`recording_numbers` gathers the ones the detection summary reports
(``continuum_swr_replay.reported.detection`` names and describes them), and the
figure (``scripts/figures/generate_memoryless_comparison.py``) draws and checks
the same functions' values.

Every function but :func:`draws_from_passing` (which takes the result of
:func:`shuffle_of`) takes ``decoded``, :func:`simulate_and_decode`'s dict; those
reading the state-space detector's events (:func:`bound_error_ms`,
:func:`recording_numbers`) also need its ``"state_space_intervals"``, which
:func:`state_space_intervals` gives. An event is a row of
``decoded["event_times"]``; a sequence is an index into
``decoded["candidates"]``. Times are in seconds unless a name says ms, distances
in cm, shares in [0, 1].
"""

import numpy as np

from continuum_swr_replay.analysis.constants import DEFAULT_NON_LOCAL_THRESHOLD
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.comparison_simulation import (
    CONTINUOUS_INBOUND_EVENT,
    CONTINUOUS_OUTBOUND_EVENT,
    REMOTE_STATIONARY_EVENT,
    SHORT_PATH_EVENT,
    SHORT_REPLAY_EVENT,
    STRAY_WINDOW_EVENT,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.shuffle import (
    SIGNIFICANCE_LEVEL,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.window_statistics import (
    NO_CENTER,
    dispersion_cm,
)

__all__ = [
    "LOCAL_TOLERANCE_CM",
    "NON_LOCAL_MINIMUM_DURATION_S",
    "NON_LOCAL_PROB_THRESHOLD",
    "ZOOMED_EVENTS",
    "accepted_overlapping",
    "best_dispersion_cm",
    "bound_error_ms",
    "candidates_overlapping",
    "decoded_center_cm",
    "draws_from_passing",
    "duration_ratio",
    "is_reported",
    "local_fraction",
    "local_fraction_of_event",
    "outside_pieces",
    "recording_numbers",
    "reported_overlapping",
    "shuffle_of",
    "state_space_intervals",
    "stray_window_dispersions",
    "verdict",
]

NON_LOCAL_PROB_THRESHOLD = DEFAULT_NON_LOCAL_THRESHOLD
NON_LOCAL_MINIMUM_DURATION_S = 0.015
"""The project's non-local event threshold and nominal minimum, applied with a
timestamp-span test (:func:`state_space_intervals`): non-local probability
above 0.8 over a first-to-last span of at least 15 ms, i.e. 9 samples on the
simulation's 2 ms grid. ``get_confident_times`` counts samples instead (7 on the
real decode grid; ``docs/known_issues/nonlocal_minimum_duration_rounding.md``).
The span test is kept so the published comparison does not change."""

LOCAL_TOLERANCE_CM = 15.0
"""A window decodes the animal's own place when its center is closer than this."""

ZOOMED_EVENTS = (
    CONTINUOUS_OUTBOUND_EVENT,
    SHORT_REPLAY_EVENT,
    SHORT_PATH_EVENT,
    REMOTE_STATIONARY_EVENT,
)
"""The zoomed events whose bounds the state-space detector is scored on
(:func:`bound_error_ms`): every one the figure zooms on but the stray-window
event, over which it reports nothing."""


def state_space_intervals(decoded: dict) -> list[tuple[float, float]]:
    """When the switching state-space model calls the representation non-local.

    The comparison needs this. Every event the sliding-window detector misses
    has to be an event some detector can find, or the figure shows the limits
    of the data rather than the limits of the method.

    Reads the detector fitted in :func:`simulate_and_decode`, which is also the
    one whose place fields the sliding-window likelihood was given, so the two
    procedures differ in how they use the encoding model and not in what it is.

    The state-space model is otherwise left on the library's defaults. Figure 1
    sets a discrete transition stickiness, but stickiness is a Dirichlet
    concentration read only by the EM M-step, and this comparison calls `fit`
    rather than `estimate_parameters`, so passing one would change nothing here
    and would read as tuning.
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
    closer than :data:`LOCAL_TOLERANCE_CM` to where the animal is at the
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
    return float(
        np.mean(np.abs(center_cm[kept] - animal_cm[kept]) < LOCAL_TOLERANCE_CM)
    )


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

    Raises
    ------
    RuntimeError
        If no accepted sequence overlaps the stray-window event.
    """
    candidates = decoded["candidates"]
    sequence = _longest_accepted(decoded, STRAY_WINDOW_EVENT)
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


def bound_error_ms(decoded: dict, events: tuple[int, ...]) -> float:
    """Largest error in either bound of the state-space model's reports, in ms.

    For each event, the start of the first state-space interval overlapping
    it against the event's start, and the end of the last against its end.
    NaN when the state-space detector reports nothing over one of ``events``.
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


def recording_numbers(decoded: dict) -> dict[str, float | int]:
    """The numbers the detection summary reports about the labeled recording.

    Parameters
    ----------
    decoded : dict
        :func:`simulate_and_decode`'s dict with ``"state_space_intervals"``
        (:func:`state_space_intervals`) added.

    Returns
    -------
    numbers : dict
        Durations in ms, distances in cm, shares in [0, 1]. The prefixes
        ``outbound``, ``inbound``, ``short_replay`` and ``stray_window`` name
        the continuous outbound and inbound replays, the 80 ms replay and the
        stray-window event; each value is over the longest accepted sequence
        overlapping the event.

        - ``outbound_``, ``short_replay_duration_ratio``: its reported over its
          true duration (:func:`duration_ratio`).
        - ``outbound_``, ``short_replay_``, ``stray_window_local_fraction``:
          the share of its kept windows decoding the animal's own place
          (:func:`local_fraction_of_event`).
        - ``*_p_spread``, ``*_p_jump``, for all four: the two halves of its
          significance test.
        - ``outbound_draws_from_passing``: :func:`draws_from_passing`.
        - ``short_path_dispersion_cm``, ``remote_stationary_dispersion_cm``: the
          largest dispersion of a candidate over each (:func:`best_dispersion_cm`).
        - ``state_space_bound_error_ms``: :func:`bound_error_ms` over
          :data:`ZOOMED_EVENTS`.
        - ``stray_window_distance_cm``: how far the stray window's far cell is
          from the animal.
        - ``stray_window_duration_ms``: the stray-window sequence's duration.
        - ``stray_window_dispersion_cm``, ``stray_window_pieces_dispersion_cm``:
          its dispersion with and without the windows outside its pieces
          (:func:`stray_window_dispersions`).

    Raises
    ------
    RuntimeError
        If no accepted sequence overlaps one of the four events read.
    """
    outbound = shuffle_of(decoded, CONTINUOUS_OUTBOUND_EVENT)
    inbound = shuffle_of(decoded, CONTINUOUS_INBOUND_EVENT)
    short = shuffle_of(decoded, SHORT_REPLAY_EVENT)
    stray_full, stray_pieces, stray_sequence = stray_window_dispersions(decoded)
    stray = decoded["shuffles"][stray_sequence]
    return {
        "outbound_duration_ratio": duration_ratio(decoded, CONTINUOUS_OUTBOUND_EVENT),
        "outbound_local_fraction": local_fraction_of_event(
            decoded, CONTINUOUS_OUTBOUND_EVENT
        ),
        "outbound_p_spread": float(outbound.p_spread),
        "outbound_p_jump": float(outbound.p_jump),
        "outbound_draws_from_passing": draws_from_passing(outbound),
        "inbound_p_spread": float(inbound.p_spread),
        "inbound_p_jump": float(inbound.p_jump),
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
        "state_space_bound_error_ms": bound_error_ms(decoded, ZOOMED_EVENTS),
        "stray_window_distance_cm": abs(
            decoded["stray_window_remote_cm"] - decoded["stray_window_local_cm"]
        ),
        "stray_window_local_fraction": local_fraction_of_event(
            decoded, STRAY_WINDOW_EVENT
        ),
        "stray_window_duration_ms": 1000
        * float(decoded["candidates"].duration_s[stray_sequence]),
        "stray_window_dispersion_cm": stray_full,
        "stray_window_pieces_dispersion_cm": stray_pieces,
        "stray_window_p_spread": float(stray.p_spread),
        "stray_window_p_jump": float(stray.p_jump),
    }
