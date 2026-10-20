"""Interval sets: unions, intersections and containment of time intervals.

An interval is a closed ``[start, end]`` in seconds; a set is an array of
shape ``(n, 2)``. Every function that returns intervals returns a normalized
set: sorted by start, disjoint, with overlapping or touching intervals joined
into one, and with any interval of no duration, or with a missing or inverted
bound, dropped. A shared endpoint therefore intersects in nothing, which is
what a positive-duration overlap rule needs: a single sample observes no time.
Dropping an inverted or non-finite bound is right for the outputs of interval
arithmetic, where an empty intersection appears as an inverted pair; for an
input a caller means to be well formed, ``normalize(..., strict=True)`` raises
instead.
"""

import numpy as np
from numpy.typing import NDArray

__all__ = ["contains", "intersect", "normalize", "subtract", "total_duration", "union"]


def _pairs(intervals: NDArray) -> NDArray:
    array = np.asarray(intervals, dtype=np.float64)
    if array.size == 0:
        return np.empty((0, 2))
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(
            f"Intervals must be (start, end) pairs; got shape {array.shape}."
        )
    return array


def normalize(intervals: NDArray, *, strict: bool = False) -> NDArray:
    """Sort, join overlapping or touching intervals, and drop empty ones.

    Parameters
    ----------
    intervals : ndarray, shape (n, 2)
    strict : bool, default False
        Raise on an inverted or non-finite bound instead of dropping it, for
        an input that is supposed to be well formed.

    Returns
    -------
    normalized : ndarray, shape (m, 2)

    """
    array = _pairs(intervals)
    if strict and array.size:
        finite = np.isfinite(array).all(axis=1)
        if not finite.all() or (array[finite, 1] < array[finite, 0]).any():
            raise ValueError("An interval has a non-finite or inverted bound.")
    array = array[np.isfinite(array).all(axis=1) & (array[:, 1] > array[:, 0])]
    if array.size == 0:
        return np.empty((0, 2))
    array = array[np.argsort(array[:, 0], kind="stable")]
    # a new interval starts wherever a start lies beyond every earlier end
    reach = np.maximum.accumulate(array[:, 1])
    begins = np.r_[True, array[1:, 0] > reach[:-1]]
    starts = array[begins, 0]
    ends = np.maximum.reduceat(array[:, 1], np.flatnonzero(begins))
    return np.column_stack([starts, ends])


def union(*sets: NDArray) -> NDArray:
    """Every time covered by any of the sets."""
    if not sets:
        return np.empty((0, 2))
    return normalize(np.vstack([_pairs(each) for each in sets]))


def intersect(a: NDArray, b: NDArray) -> NDArray:
    """Every time covered by both sets."""
    a, b = normalize(a), normalize(b)
    if a.size == 0 or b.size == 0:
        return np.empty((0, 2))
    starts = np.maximum(a[:, None, 0], b[None, :, 0])
    ends = np.minimum(a[:, None, 1], b[None, :, 1])
    return normalize(np.column_stack([starts.ravel(), ends.ravel()]))


def subtract(a: NDArray, b: NDArray) -> NDArray:
    """Every time covered by ``a`` and not by ``b``."""
    a, b = normalize(a), normalize(b)
    if a.size == 0 or b.size == 0:
        return a
    # the complement of b, bounded by a's own span so no bound is infinite
    low, high = a[0, 0], a[-1, 1]
    gaps = np.column_stack([np.r_[low, b[:, 1]], np.r_[b[:, 0], high]])
    return intersect(a, gaps)


def total_duration(intervals: NDArray) -> float:
    """Seconds covered, overlap counted once."""
    normalized = normalize(intervals)
    return float((normalized[:, 1] - normalized[:, 0]).sum())


def contains(intervals: NDArray, starts: NDArray, ends: NDArray) -> NDArray:
    """Whether each ``[start, end]`` lies wholly inside the set.

    Parameters
    ----------
    intervals : ndarray, shape (n, 2)
    starts, ends : ndarray, shape (n_events,)

    Returns
    -------
    within : ndarray of bool, shape (n_events,)

    """
    normalized = normalize(intervals)
    starts = np.asarray(starts, dtype=np.float64)
    ends = np.asarray(ends, dtype=np.float64)
    if starts.shape != ends.shape:
        raise ValueError("starts and ends must be the same shape.")
    if (ends < starts).any():
        raise ValueError("An event has end < start; nothing contains it.")
    if normalized.size == 0:
        return np.zeros(starts.shape, dtype=bool)
    index = np.searchsorted(normalized[:, 0], starts, side="right") - 1
    found = index >= 0
    safe = np.clip(index, 0, None)
    within: NDArray = (
        found & (normalized[safe, 0] <= starts) & (ends <= normalized[safe, 1])
    )
    return within
