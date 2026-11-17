"""The injected-replay sweep: what each detector does to a replay of known shape.

Each cell injects one replay of a given path length, duration, peak rate and
midpoint (``sweep.replay_grid``) into the sweep's recording, decodes a window
around it with both detectors, and scores what each reports
(:func:`run_cell`). A detector finds the event only when what it reports is the
injected event (``scoring.score_against_injection``); every reported event
that overlaps the injection is scored, and the best answers for the cell
(:func:`best_entry`). :func:`replay_sweep` runs every cell of the grid, in
each direction (:func:`planned_cells`).
"""

import hashlib
import multiprocessing
import pickle
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from non_local_detector.simulate.sorted_spikes_simulation import (
    SAMPLING_FREQUENCY,
    TRACK_HEIGHT,
)

from continuum_swr_replay.analysis.detection.memoryless_sequence.comparison_simulation import (
    CONFIG,
    N_NEURONS,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.scoring import (
    best_scored,
    score_against_injection,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.shuffle import (
    SIGNIFICANCE_LEVEL,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.simulation import (
    InjectedReplay,
    added_spike_span,
    inject_replay,
    place_field_means,
    placements,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.sweep import (
    MATCH_TOLERANCE_CM,
    PLACEMENT_MARGIN_S,
    SlidingWindowRun,
    StateSpaceRun,
    decode_window,
    nearest_bin,
    overlaps,
    replay_grid,
    run_sliding_window,
    run_state_space,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.window_statistics import (
    dispersion_cm,
)

__all__ = [
    "SAME_PLACE_CM",
    "best_entry",
    "DIRECTIONS",
    "SWEEP_SEED",
    "cell_key",
    "memoryless_outcome",
    "planned_cells",
    "replay_sweep",
    "run_cell",
    "score_sliding_window",
    "score_state_space",
    "state_space_outcome",
]

SAME_PLACE_CM = 15.0
"""How near the animal's own place a sliding-window event's decoded position
must be to count toward its ``sliding_local_fraction``."""


def best_entry(scored: list[tuple]) -> tuple:
    """The scored event that answers for the injection.

    Both detectors score every reported event that overlaps the injection and
    keep whatever else they need beside it; each entry carries its score
    second.
    """
    return scored[best_scored([entry[1] for entry in scored])]


def score_sliding_window(
    run: SlidingWindowRun, bounds, span, true_position_cm, path_cm, position
) -> dict:
    """What the sliding-window detector reports for one injected event.

    ``bounds`` is the window the event was drawn in, which decides whether a
    report overlaps it; ``span`` is its own first and last spike, which a
    report's start and end are held to (:func:`run_cell`).

    Scored three ways, because the published definition has three parts and
    the reference's detection script applied only some of them.
    ``sliding_matches`` is the whole published definition: a candidate that
    meets the duration and dispersion criteria, is the injected replay, and
    beats the place-cell identity shuffle on both mean spread and mean jump.
    That is what the paper's analyses read, from the corrected
    ``load_replayEvents_cellIDshuffle_sameEvents.m``. The detection script
    itself computed its jump p-values from an all-NaN vector, so every event
    passed that half; ``sliding_matches_spread_only`` is what its own filter
    applied.
    ``sliding_matches_before_shuffle`` leaves the shuffle out.
    """
    candidates, midpoint_s = run.candidates, run.midpoint_s
    start_s, end_s = bounds
    inside = (midpoint_s >= start_s) & (midpoint_s <= end_s)
    gates = candidates.decisions.gates
    overlapping = [
        i
        for i, (run_start, run_stop) in enumerate(candidates.runs)
        if overlaps((midpoint_s[run_start], midpoint_s[run_stop]), bounds)
    ]
    accepted = [i for i in overlapping if candidates.is_replay[i]]

    def inside_rate(flag):
        return float(flag[inside].mean()) if inside.any() else np.nan

    out = {
        "sliding_found": bool(accepted),
        "sliding_any_candidate": bool(overlapping),
        "sliding_n_candidates": len(overlapping),
        "sliding_n_accepted": len(accepted),
        "sliding_has_center": inside_rate(gates.has_center),
        "sliding_posterior_usable": inside_rate(gates.posterior_usable),
        "sliding_speed_eligible": inside_rate(gates.speed_eligible),
        "sliding_spread_ok": inside_rate(gates.spread_below_threshold),
        "sliding_jump_ok": inside_rate(gates.jump_below_threshold),
        "sliding_best_dispersion_cm": max(
            (candidates.dispersion_cm[i] for i in overlapping), default=np.nan
        ),
        "sliding_best_duration_s": max(
            (candidates.duration_s[i] for i in overlapping), default=np.nan
        ),
        "sliding_matches": False,
        "sliding_matches_before_shuffle": False,
        "sliding_matches_spread_only": False,
        "sliding_p_spread": np.nan,
        "sliding_p_jump": np.nan,
        "sliding_position_error_cm": np.nan,
        "sliding_span_cm": np.nan,
    }
    if not accepted:
        return out

    # Every accepted sequence that overlaps the injection is scored, and the
    # best answers for the cell. Two things have to hold over the whole
    # sequence for a match: its decoded positions sit on the path, and it does
    # not cover far more ground than the path does. An absorbed sequence
    # tracks the injection during the injection and then sits at the animal's
    # own place, which is where its dispersion comes from.
    scored = [
        (
            i,
            score_against_injection(
                run.events[i].decoded_time_s,
                run.events[i].decoded_cm,
                true_position_cm,
                bounds,
                path_cm,
                MATCH_TOLERANCE_CM,
            ),
        )
        for i in accepted
    ]
    shuffles = run.shuffles

    def any_match(keep) -> bool:
        return any(entry[1].matches for entry in scored if keep(entry[0]))

    i, score = best_entry(scored)
    event = run.events[i]
    animal_cm = np.interp(event.decoded_time_s, run.time, position[: run.time.size])
    out |= {
        "sliding_matches_before_shuffle": score.matches,
        "sliding_matches": any_match(lambda j: shuffles[j].passes),
        "sliding_matches_spread_only": any_match(
            lambda j: shuffles[j].p_spread <= SIGNIFICANCE_LEVEL
        ),
        "sliding_p_spread": shuffles[i].p_spread,
        "sliding_p_jump": shuffles[i].p_jump,
        "sliding_position_error_cm": score.position_error_cm,
        "sliding_span_cm": score.span_cm,
        "sliding_duration_s": float(candidates.duration_s[i]),
        "sliding_dispersion_cm": float(candidates.dispersion_cm[i]),
        "sliding_start_error_s": float(event.start_s - span[0]),
        "sliding_end_error_s": float(event.end_s - span[1]),
        "sliding_n_pieces": int(candidates.pieces[i].size),
        "sliding_local_fraction": float(
            np.mean(np.abs(event.decoded_cm - animal_cm) < SAME_PLACE_CM)
        ),
    }
    return out


def memoryless_outcome(
    recording, injected, window, bounds, span, true_position_cm, path_cm, key
):
    """The sliding-window detector run over the window and scored against the event."""
    run = run_sliding_window(recording, injected, window, key)
    return score_sliding_window(run, bounds, span, true_position_cm, path_cm, window[1])


def score_state_space(
    recording, run: StateSpaceRun, bounds, span, true_position_cm, path_cm
) -> dict:
    """What the switching state-space model reports for the same event.

    Scored by the same rule as the sliding window: every reported event that
    overlaps the injection is scored on its own, over its whole extent, and
    the best answers for the cell.
    """
    time = run.time
    probability = run.p_continuous + run.p_fragmented
    start_s, end_s = bounds
    hits = [
        event
        for event in run.canonical
        if overlaps((event.start_s, event.end_s), bounds)
    ]
    inside = (time >= start_s) & (time <= end_s)

    out = {
        "state_space_found": bool(hits),
        "state_space_n_intervals": len(hits),
        "state_space_peak_probability": (
            float(probability[inside].max()) if inside.any() else np.nan
        ),
        "state_space_matches": False,
        "state_space_replay_matches": False,
        "state_space_position_error_cm": np.nan,
        "state_space_span_cm": np.nan,
        "state_space_dispersion_cm": np.nan,
    }
    if not hits:
        return out

    scored = []
    for event in hits:
        duration_s = event.end_s - event.start_s
        dispersion = dispersion_cm(
            nearest_bin(recording, event.decoded_cm), recording["bin_distance_cm"]
        )
        score = score_against_injection(
            event.decoded_time_s,
            event.decoded_cm,
            true_position_cm,
            bounds,
            path_cm,
            MATCH_TOLERANCE_CM,
        )
        # and again under the published replay definition the sliding window
        # is held to, so the two rates can be read against each other; judged
        # per event, as the sliding window's accepted set is, before the best
        # is chosen
        meets_replay_definition = bool(
            score.matches
            and duration_s > CONFIG.minimum_duration_s
            and dispersion > CONFIG.dispersion_threshold_cm
        )
        scored.append((event, score, duration_s, dispersion, meets_replay_definition))
    event, score, duration_s, dispersion, _ = best_entry(scored)

    out |= {
        "state_space_matches": score.matches,
        "state_space_replay_matches": any(entry[4] for entry in scored),
        "state_space_position_error_cm": score.position_error_cm,
        "state_space_span_cm": score.span_cm,
        "state_space_duration_s": duration_s,
        "state_space_start_error_s": float(event.start_s - span[0]),
        "state_space_end_error_s": float(event.end_s - span[1]),
        "state_space_dispersion_cm": dispersion,
    }
    return out


def state_space_outcome(
    recording, injected, window, bounds, span, true_position_cm, path_cm
):
    """The state-space model run over the window and scored against the event."""
    run = run_state_space(recording, injected, window)
    return score_state_space(recording, run, bounds, span, true_position_cm, path_cm)


def cell_key(
    path_cm: float,
    midpoint_cm: float,
    duration_s: float,
    rate_hz: float,
    direction: int,
    replicate: int,
) -> str:
    """A cell's name: its combination, its direction and which replicate it is.

    Everything random about a cell is seeded from this name
    (:func:`planned_cells`, and the shuffle of ``sweep.run_sliding_window``),
    so no two cells share a draw and a cell's draws do not depend on which
    other cells the sweep holds.
    """
    return (
        f"path{path_cm}-mid{midpoint_cm}-dur{duration_s}-rate{rate_hz}"
        f"-dir{direction:+d}-rep{replicate}"
    )


def run_cell(
    recording,
    means,
    *,
    path_cm,
    midpoint_cm,
    duration_s,
    rate_hz,
    direction,
    seed,
    at_s,
    injection_seed,
):
    """One cell: inject, decode with both detectors, summarize.

    ``direction`` is 1 for a replay that travels up the track, towards larger
    positions, and -1 for one that travels down it over the same path.
    ``seed`` is the replicate's index; ``at_s`` and ``injection_seed`` are the
    cell's own placement and firing noise (:func:`planned_cells`).
    """
    injected, bounds = inject_replay(
        recording["spike_times"],
        means,
        InjectedReplay(
            start_cm=midpoint_cm - direction * path_cm / 2,
            path_cm=direction * path_cm,
            duration_s=duration_s,
            peak_rate_hz=rate_hz,
            at_s=at_s,
            seed=injection_seed,
        ),
        sampling_frequency=SAMPLING_FREQUENCY,
    )
    start_s, end_s = bounds
    window = decode_window(recording, bounds)

    original = np.concatenate(recording["spike_times"])
    added = np.concatenate(injected)
    n_spikes = int(
        ((added >= start_s) & (added <= end_s)).sum()
        - ((original >= start_s) & (original <= end_s) & np.isin(original, added)).sum()
    )
    # A replay's timing is held to its own first and last spike, not to the
    # window it was drawn in: a sparse replay's first spike can come well
    # after the window opens, and no detector can report what has not fired.
    span = added_spike_span(recording["spike_times"], injected)
    row = {
        "path_cm": path_cm,
        "midpoint_cm": midpoint_cm,
        "duration_s": duration_s,
        "peak_rate_hz": rate_hz,
        "direction": direction,
        "seed": seed,
        "injection_seed": injection_seed,
        "at_s": at_s,
        "speed_cm_per_s": path_cm / duration_s,
        "distance_from_animal_cm": abs(
            midpoint_cm
            - float(np.interp(at_s, recording["time"], recording["position"]))
        ),
        "n_spikes": n_spikes,
        "first_spike_s": span[0],
        "last_spike_s": span[1],
        "spikes_per_window": n_spikes * CONFIG.window_duration_s / duration_s,
    }

    def true_position_cm(at):
        """Where the injected replay says the animal is, at these times."""
        fraction = np.clip((np.asarray(at) - start_s) / (end_s - start_s), 0, 1)
        return midpoint_cm - direction * path_cm / 2 + direction * path_cm * fraction

    key = cell_key(path_cm, midpoint_cm, duration_s, rate_hz, direction, seed)
    row |= memoryless_outcome(
        recording, injected, window, bounds, span, true_position_cm, path_cm, key
    )
    row |= state_space_outcome(
        recording, injected, window, bounds, span, true_position_cm, path_cm
    )
    return row


DIRECTIONS = (1, -1)
"""Up the track and down it. The animal rests at the top of the track wherever a
replay is placed, so an upward replay ends at its edge nearest the animal and a
downward one starts there."""

SWEEP_SEED = 20261117
"""Seeds every cell's placement and firing noise, with the cell's own name
(:func:`planned_cells`)."""


def planned_cells(intervals: np.ndarray, n_seeds: int) -> list[dict]:
    """Every cell of the sweep: each combination of ``sweep.replay_grid`` in each
    direction, ``n_seeds`` times.

    A cell's start time and firing noise are drawn from generators seeded by
    :data:`SWEEP_SEED` and the cell's own name (:func:`cell_key`), one stream
    each. So every cell has its own noise, and neither draw depends on the
    grid's order or on which other cells are planned. A path of 0 cm has no
    direction and is planned once.

    Parameters
    ----------
    intervals : ndarray, shape (n_intervals, 2)
        Where events may be placed: ``sweep.usable_intervals(recording)``.
    n_seeds : int
        Replicates of each combination and direction.

    Returns
    -------
    list of dict
        :func:`run_cell`'s keyword arguments after the recording and the
        fields, one dict per cell: its combination, ``direction``, ``seed``
        (the replicate's index), ``at_s`` and ``injection_seed``.
    """
    cells = []
    for path_cm, midpoint_cm, duration_s, rate_hz in replay_grid():
        for direction in DIRECTIONS if path_cm > 0 else DIRECTIONS[:1]:
            for replicate in range(n_seeds):
                key = cell_key(
                    path_cm, midpoint_cm, duration_s, rate_hz, direction, replicate
                )
                named = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
                placement, injection = np.random.SeedSequence(
                    [SWEEP_SEED, named]
                ).spawn(2)
                at_s = placements(
                    intervals,
                    duration_s,
                    PLACEMENT_MARGIN_S,
                    1,
                    np.random.default_rng(placement),
                )
                cells.append(
                    {
                        "path_cm": path_cm,
                        "midpoint_cm": midpoint_cm,
                        "duration_s": duration_s,
                        "rate_hz": rate_hz,
                        "direction": direction,
                        "seed": replicate,
                        "at_s": float(at_s[0]),
                        "injection_seed": int(injection.generate_state(1)[0]),
                    }
                )
    return cells


_WORKER: dict = {}
"""A worker process's recording and fields (:func:`_start_worker`)."""


def _start_worker(recording_bytes: bytes) -> None:
    # JAX runs where the parent's JAX_PLATFORMS says: the variable is inherited
    _WORKER["recording"] = pickle.loads(recording_bytes)
    _WORKER["means"] = place_field_means(N_NEURONS, TRACK_HEIGHT)


def _run_planned_cell(cell: dict) -> dict:
    return run_cell(_WORKER["recording"], _WORKER["means"], **cell)


def replay_sweep(
    recording: dict, intervals: np.ndarray, n_seeds: int, n_workers: int = 1
) -> Iterator[dict]:
    """Run every cell of :func:`planned_cells`, in its order.

    Most of a cell's time is the state-space decode compiling for that cell's
    window, which no cell shares with the next; the cells are independent, so
    they are run in ``n_workers`` processes. Each process is given the one
    fitted recording, and a cell's row does not depend on which runs it.

    Parameters
    ----------
    recording : dict
        As ``sweep.build_recording`` returns it.
    intervals : ndarray, shape (n_intervals, 2)
        Where events may be placed: ``sweep.usable_intervals(recording)``.
    n_seeds : int
        Injections per combination.
    n_workers : int, optional
        Processes to run the cells in; 1 runs them in this one.

    Yields
    ------
    dict
        A cell's :func:`run_cell` row.
    """
    cells = planned_cells(intervals, n_seeds)
    if n_workers == 1:
        means = place_field_means(N_NEURONS, TRACK_HEIGHT)
        for cell in cells:
            yield run_cell(recording, means, **cell)
        return
    # spawned, not forked: JAX is multithreaded
    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_start_worker,
        initargs=(pickle.dumps(recording),),
    ) as pool:
        yield from pool.map(_run_planned_cell, cells, chunksize=8)
