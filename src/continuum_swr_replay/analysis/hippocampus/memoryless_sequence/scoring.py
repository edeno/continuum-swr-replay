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
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["InjectionScore", "best_scored", "score_against_injection"]


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
        given; before the replay begins its start, after it ends its end.
    bounds : tuple of float
        When the injected replay begins and ends.
    path_cm : float
        How far the injected replay travels.
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

    during = (times >= start_s) & (times <= end_s)
    if not during.any():
        return InjectionScore(False, float("nan"), span_cm)

    error_cm = float(np.median(np.abs(decoded - true_position_cm(times))))
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
