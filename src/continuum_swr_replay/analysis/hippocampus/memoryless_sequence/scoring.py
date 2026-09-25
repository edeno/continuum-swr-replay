"""Score what a detector reports against a replay of known shape.

One rule, applied to every detector the same way. A reported event counts as
the injected replay only when it overlaps the injection in time and two
things hold over the *whole* reported event: its decoded positions sit on the
injected path, and it does not cover far more ground than the path does.
Outside the injection the reference position is the path's nearer end, where
the replay began or ended, so an event that merely runs on in time while
staying where the replay stopped still counts, and one that runs off to the
animal's own place does not. That second case is what separates a detection
from an over-call: a sequence built out of the local coding beside an
injection tracks the injection during the injection and then sits somewhere
else, and only the whole event shows it.

Both tests read the whole reported event, whichever detector produced it.
Reading only the part that overlaps the injection would hide what the event
does outside it, which is the very thing the tests are for.

An injected event may have several parts, with a jump or a silent gap between
them. Its true path is then NaN in the gaps: decoded positions there have no
truth to be compared with, so they take no part in the position error, but
they still count toward the event's span. Each part is scored as a target of
its own (``recovered_parts``): an event recovers a part only when it overlaps
at least half of it *and* its decoded positions over that part sit on that
part's path (``part_agreement``), since a whole-event median can hide one
part decoded somewhere else. How a detector carved the whole is one label
(``segmentation``); neither answer is declared the correct one.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "SEGMENTATION_LABELS",
    "InjectionScore",
    "best_scored",
    "bound_errors",
    "part_agreement",
    "recovered_parts",
    "score_against_injection",
    "segmentation",
]

SEGMENTATION_LABELS = ("spanning", "one_per_part", "parts_missing", "other")
"""How a detector carved a multi-part event: one event over every part; one
event per part; some part not recovered; anything else, such as a part split
across two events."""


@dataclass(frozen=True)
class InjectionScore:
    """How one reported event compares with the replay that was put in.

    Attributes
    ----------
    matches : bool
        The event is the injected replay under the rule in the module
        docstring.
    position_error_cm : float
        Median distance between the decoded position and the injected path
        over the whole reported event, the path's nearer end standing in for
        the path outside the injection. NaN when the event has no decoded
        position inside the injection.
    span_cm : float
        Range of the decoded position over the whole reported event. NaN
        when the event has no decoded position at all.

    """

    matches: bool
    position_error_cm: float
    span_cm: float


def score_against_injection(
    decoded_time_s: NDArray,
    decoded_cm: NDArray,
    true_position_cm: Callable[[NDArray], NDArray],
    bounds: tuple[float, float],
    path_cm: float,
    tolerance_cm: float,
) -> InjectionScore:
    """Score one reported event against the injected replay.

    Parameters
    ----------
    decoded_time_s : ndarray, shape (n_decoded,)
        When each decoded position of the reported event was read. Every
        decoded position of the event, not only those inside the injection.
    decoded_cm : ndarray, shape (n_decoded,)
        The decoded position at each of those times.
    true_position_cm : callable
        Where the injected replay says the animal is, at the times it is
        given; before the replay begins its start, after it ends its end;
        NaN in a silent gap between parts.
    bounds : tuple of float
        When the injected replay begins and ends.
    path_cm : float
        How far the injected replay travels; for several parts, the ground
        they cover together.
    tolerance_cm : float
        How far the decoded position may sit from the injected one. The span
        may exceed the path by twice this, one tolerance at each end.

    Returns
    -------
    score : InjectionScore

    """
    times = np.asarray(decoded_time_s, dtype=np.float64)
    decoded = np.asarray(decoded_cm, dtype=np.float64)
    if times.shape != decoded.shape:
        raise ValueError(
            f"{times.shape} decoded times carry {decoded.shape} decoded positions."
        )
    start_s, end_s = bounds
    span_cm = float(np.ptp(decoded)) if decoded.size else float("nan")

    truth = np.asarray(true_position_cm(times), dtype=np.float64)
    has_truth = np.isfinite(truth)
    during = (times >= start_s) & (times <= end_s) & has_truth
    if not during.any():
        return InjectionScore(False, float("nan"), span_cm)

    error_cm = float(np.median(np.abs(decoded[has_truth] - truth[has_truth])))
    return InjectionScore(
        matches=bool(error_cm < tolerance_cm and span_cm <= path_cm + 2 * tolerance_cm),
        position_error_cm=error_cm,
        span_cm=span_cm,
    )


def best_scored(scores: list[InjectionScore]) -> int:
    """Index of the reported event that best answers for the injection.

    A matching event beats a non-matching one; among equals the smaller
    position error wins, and a NaN error loses to any finite one. Scoring
    only the first overlapping event would let an absorbed sequence that
    happens to start earlier mask the true detection behind it.
    """
    if not scores:
        raise ValueError("No scored events to choose from.")
    return min(
        range(len(scores)),
        key=lambda i: (
            not scores[i].matches,
            np.isnan(scores[i].position_error_cm),
            scores[i].position_error_cm,
        ),
    )


def part_agreement(
    decoded_time_s: NDArray,
    decoded_cm: NDArray,
    true_position_cm: Callable[[NDArray], NDArray],
    part_bounds: list[tuple[float, float]],
    tolerance_cm: float,
) -> NDArray:
    """Whether an event's decoded positions over each part sit on that part's path.

    Parameters
    ----------
    decoded_time_s, decoded_cm : ndarray, shape (n_decoded,)
        The event's decoded positions and when each was read.
    true_position_cm : callable
        The injected path, NaN in silent gaps.
    part_bounds : list of (float, float)
        When each part begins and ends.
    tolerance_cm : float
        How far the median decoded position may sit from the part's path.

    Returns
    -------
    agrees : ndarray of bool, shape (n_parts,)
        False for a part the event has no decoded position in.

    """
    times = np.asarray(decoded_time_s, dtype=np.float64)
    decoded = np.asarray(decoded_cm, dtype=np.float64)
    truth = np.asarray(true_position_cm(times), dtype=np.float64)
    agrees = []
    for start, end in part_bounds:
        here = (times >= start) & (times <= end) & np.isfinite(truth)
        agrees.append(
            bool(here.any())
            and float(np.median(np.abs(decoded[here] - truth[here]))) < tolerance_cm
        )
    return np.array(agrees, dtype=bool)


def _covered_parts(
    event_bounds: list[tuple[float, float]],
    agreement: NDArray,
    part_bounds: list[tuple[float, float]],
) -> list[set[int]]:
    """For each event right about any part, the parts it recovers."""
    covered = []
    for (start, end), agrees in zip(event_bounds, agreement, strict=True):
        if not np.any(agrees):
            continue
        covered.append(
            {
                part
                for part, (part_start, part_end) in enumerate(part_bounds)
                if agrees[part]
                and min(end, part_end) - max(start, part_start)
                >= 0.5 * (part_end - part_start)
            }
        )
    return covered


def recovered_parts(
    event_bounds: list[tuple[float, float]],
    agreement: NDArray,
    part_bounds: list[tuple[float, float]],
) -> NDArray:
    """Which parts some event recovers: overlaps half of it and is right about it.

    Parameters
    ----------
    event_bounds : list of (float, float)
        Every event the detector reported that overlaps the injection.
    agreement : array-like of bool, shape (n_events, n_parts)
        Whether each event is right about each part: it is the injected
        replay under ``score_against_injection`` against the whole piecewise
        path, and ``part_agreement`` holds for that part.
    part_bounds : list of (float, float)
        When each part of the injected event begins and ends.

    Returns
    -------
    recovered : ndarray of bool, shape (n_parts,)

    """
    covered = _covered_parts(event_bounds, agreement, part_bounds)
    recovered = set().union(*covered) if covered else set()
    return np.array([part in recovered for part in range(len(part_bounds))])


def segmentation(
    event_bounds: list[tuple[float, float]],
    agreement: NDArray,
    part_bounds: list[tuple[float, float]],
) -> str:
    """How the detector carved the event, as one of ``SEGMENTATION_LABELS``.

    Parameters are as for :func:`recovered_parts`.
    """
    covered = _covered_parts(event_bounds, agreement, part_bounds)
    n_parts = len(part_bounds)
    if any(len(parts) == n_parts for parts in covered):
        return "spanning"
    recovered = set().union(*covered) if covered else set()
    if len(recovered) < n_parts:
        return "parts_missing"
    covering = [parts for parts in covered if parts]
    if len(covering) == n_parts and all(len(parts) == 1 for parts in covering):
        return "one_per_part"
    return "other"


def bound_errors(
    event_bounds: list[tuple[float, float]],
    agreement: NDArray,
    part_bounds: list[tuple[float, float]],
) -> NDArray:
    """Reported minus true start and end, where the carving makes them meaningful.

    One row per part when the detector reported one event per part, against
    that part; one row when it reported one spanning event, against the first
    part's start and the last part's end; no rows otherwise.

    Returns
    -------
    errors_s : ndarray, shape (n_rows, 2)
        Start error and end error, in seconds.

    """
    label = segmentation(event_bounds, agreement, part_bounds)
    matched = [
        bounds
        for bounds, agrees in zip(event_bounds, agreement, strict=True)
        if np.any(agrees)
    ]
    covered = _covered_parts(event_bounds, agreement, part_bounds)
    if label == "spanning":
        start, end = next(
            bounds
            for bounds, parts in zip(matched, covered, strict=True)
            if len(parts) == len(part_bounds)
        )
        return np.array([[start - part_bounds[0][0], end - part_bounds[-1][1]]])
    if label == "one_per_part":
        rows = []
        for part, (part_start, part_end) in enumerate(part_bounds):
            start, end = next(
                bounds
                for bounds, parts in zip(matched, covered, strict=True)
                if parts == {part}
            )
            rows.append([start - part_start, end - part_end])
        return np.array(rows)
    return np.zeros((0, 2))
