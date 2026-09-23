"""What each detector observed, apart from what it detected.

A detector that looked and found nothing observed that time; a detector whose
inputs were missing did not, and an empty event table alone cannot tell the
two apart. Coverage is built from the samples a detector needed, split at
every non-finite sample and every gap in the timestamps, never interpolated
across either; a detector that reads a window of samples observes a moment
only when the whole window is there.

Exposure is observed time during which the animal was known to be still:
finite speed below the canonical threshold. Comparisons between detectors use
the intersection of their exposures.
"""

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from continuum_swr_replay.analysis.constants import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
)
from continuum_swr_replay.analysis.inventory_comparison.intervals import (
    intersect,
    normalize,
    subtract,
)
from continuum_swr_replay.analysis.inventory_comparison.overlap import Coverage
from continuum_swr_replay.analysis.inventory_comparison.schema import SessionIdentity
from continuum_swr_replay.session_keys import SESSION_COLUMNS_DAY

__all__ = [
    "OBSERVATION_DTYPES",
    "coverage_from_observations",
    "exposure",
    "missing_run_table",
    "observation_table",
    "sample_runs",
    "still_intervals",
    "window_supported",
]

OBSERVATION_DTYPES: Mapping[str, object] = MappingProxyType(
    {
        "animal": object,
        "day": np.int64,
        "epoch": object,
        "brain_area": object,
        "detector": object,
        "run_id": object,
        "start_s": np.float64,
        "end_s": np.float64,
        "available": np.bool_,
        "reason": object,
    }
)

GAP_FACTOR = 1.5
"""A spacing more than this many median sample intervals is a gap."""


def _joined(time: NDArray, valid: NDArray, max_gap_s: float | None) -> NDArray:
    """Whether each sample continues the run of the one before it.

    Returns
    -------
    joined : ndarray of bool, shape (n_samples - 1,)
        False at an invalid sample, a non-finite timestamp, or a spacing more
        than ``max_gap_s`` (1.5 median sample intervals by default).
    """
    if time.shape != valid.shape or time.ndim != 1:
        raise ValueError("time and valid must be 1-D and the same length.")
    finite = time[np.isfinite(time)]
    if np.any(np.diff(finite) <= 0):
        raise ValueError("time must be strictly increasing.")
    if finite.size < 2:
        # nothing can continue a run of one sample
        return np.zeros(max(time.size - 1, 0), dtype=bool)
    if max_gap_s is None:
        max_gap_s = GAP_FACTOR * float(np.median(np.diff(finite)))
    with np.errstate(invalid="ignore"):
        close = np.diff(time) <= max_gap_s
    # a non-finite timestamp fails the spacing test on both sides
    joined: NDArray = valid[1:] & valid[:-1] & close
    return joined


def sample_runs(
    time: NDArray, valid: NDArray, max_gap_s: float | None = None
) -> NDArray:
    """Stretches of consecutive valid samples, from first to last sample.

    A run ends at an invalid sample, a non-finite timestamp, and any gap in
    the timestamps longer than ``max_gap_s`` (1.5 median sample intervals by
    default). A single isolated sample covers no time and is dropped.

    Parameters
    ----------
    time : ndarray, shape (n_samples,)
        Strictly increasing where finite.
    valid : ndarray of bool, shape (n_samples,)
    max_gap_s : float, optional

    Returns
    -------
    runs : ndarray, shape (n_runs, 2)

    """
    time = np.asarray(time, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if time.size < 2:
        return np.empty((0, 2))
    joined = _joined(time, valid, max_gap_s)
    valid = valid & np.isfinite(time)
    # a new run begins at a valid sample after an invalid one or after a gap
    begins = valid & np.r_[True, ~joined]
    ends = valid & np.r_[~joined, True]
    return normalize(np.column_stack([time[begins], time[ends]]))


def window_supported(
    time: NDArray,
    valid: NDArray,
    window_n_samples: int,
    max_gap_s: float | None = None,
) -> NDArray:
    """Samples at which a window stamped there has its whole support.

    A window over samples ``[k, k + window_n_samples)`` is stamped at sample
    ``k + window_n_samples // 2``, as the memoryless detector stamps it, so
    a sample is covered when the window around it is valid without a break.
    Counted in samples, not seconds, so the coverage edges are the first and
    last stamps on any clock.

    Parameters
    ----------
    time : ndarray, shape (n_samples,)
    valid : ndarray of bool, shape (n_samples,)
    window_n_samples : int
    max_gap_s : float, optional
        As for ``sample_runs``.

    Returns
    -------
    supported : ndarray of bool, shape (n_samples,)
        Pass to ``sample_runs`` with ``time`` for the observed intervals.

    """
    time = np.asarray(time, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    supported = np.zeros(time.size, dtype=bool)
    if time.size < max(window_n_samples, 2):
        return supported
    # samples share a run id exactly when no break lies between them
    run = np.r_[0, np.cumsum(~_joined(time, valid, max_gap_s))]
    first = np.arange(time.size - window_n_samples + 1)
    last = first + window_n_samples - 1
    stamp = first + window_n_samples // 2
    supported[stamp] = valid[stamp] & (run[first] == run[last])
    return supported


def still_intervals(
    time: NDArray,
    speed_cm_per_s: NDArray,
    threshold_cm_per_s: float = DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    max_gap_s: float | None = None,
) -> NDArray:
    """Stretches of finite speed below the threshold; an unknown speed is not still."""
    speed = np.asarray(speed_cm_per_s, dtype=np.float64)
    return sample_runs(
        time, np.isfinite(speed) & (speed < threshold_cm_per_s), max_gap_s
    )


def exposure(observed: NDArray, still: NDArray) -> NDArray:
    """Observed time during which the animal was still."""
    return intersect(observed, still)


def _epoch_bounds(epoch_bounds: tuple[float, float]) -> tuple[float, float]:
    start, end = float(epoch_bounds[0]), float(epoch_bounds[1])
    if not (np.isfinite(start) and np.isfinite(end) and start < end):
        raise ValueError(
            f"epoch_bounds must be finite with start < end, got {epoch_bounds}."
        )
    return start, end


def _table(
    session: SessionIdentity, detector: str, run_id: str, rows: list[tuple]
) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["start_s", "end_s", "available", "reason"])
    for name, value in (
        ("animal", session.animal),
        ("day", session.day),
        ("epoch", session.epoch),
        ("brain_area", session.brain_area),
        ("detector", detector),
        ("run_id", run_id),
    ):
        frame[name] = value
    return frame[list(OBSERVATION_DTYPES)].astype(dict(OBSERVATION_DTYPES))


def observation_table(
    session: SessionIdentity,
    *,
    detector: str,
    run_id: str,
    epoch_bounds: tuple[float, float],
    observed: NDArray,
    unavailable_reason: str,
) -> pd.DataFrame:
    """The epoch, tiled into observed and unavailable stretches.

    Parameters
    ----------
    session : SessionIdentity
    detector, run_id : str
    epoch_bounds : (float, float)
        First and last time of the epoch.
    observed : ndarray, shape (n, 2)
        What the detector observed; must lie within the epoch.
    unavailable_reason : str
        Why the rest was not observed.

    Returns
    -------
    table : DataFrame
        Columns and dtypes as ``OBSERVATION_DTYPES``, in time order, the
        stretches meeting end to end.

    """
    start, end = _epoch_bounds(epoch_bounds)
    observed = normalize(observed, strict=True)
    if observed.size and (observed[0, 0] < start or observed[-1, 1] > end):
        raise ValueError("Observed time must lie within the epoch.")
    unavailable = subtract(np.array([[start, end]]), observed)
    rows = [(a, b, True, "") for a, b in observed]
    rows += [(a, b, False, unavailable_reason) for a, b in unavailable]
    rows.sort(key=lambda row: row[0])
    return _table(session, detector, run_id, rows)


def missing_run_table(
    session: SessionIdentity,
    *,
    detector: str,
    run_id: str,
    epoch_bounds: tuple[float, float],
    reason: str,
) -> pd.DataFrame:
    """A detector that produced nothing for the epoch observed none of it."""
    start, end = _epoch_bounds(epoch_bounds)
    return _table(session, detector, run_id, [(start, end, False, reason)])


def coverage_from_observations(
    observations: pd.DataFrame, still: Mapping[tuple, NDArray]
) -> dict[tuple, Coverage]:
    """The coverage map the counts read, from observation tables.

    One origin for both representations of what a detector observed: the
    ``available`` stretches become ``observed``, and their intersection with
    the epoch's still intervals becomes ``exposure``. Every (epoch, detector)
    in the tables gets an entry, so a detector whose run was missing, entered
    through ``missing_run_table``, stays on the roster with empty coverage
    and reads as unavailable rather than vanishing.

    Parameters
    ----------
    observations : DataFrame
        ``observation_table`` and ``missing_run_table`` rows, concatenated.
    still : mapping
        ``(animal, day, epoch) -> ndarray (n, 2)``, the epoch's still
        intervals from ``still_intervals``; an epoch absent here is taken as
        never still.

    Returns
    -------
    coverage : dict
        ``(animal, day, epoch, detector) -> Coverage``.

    """
    coverage = {}
    for (session, detector), group in observations.groupby(
        [observations[list(SESSION_COLUMNS_DAY)].apply(tuple, axis=1), "detector"],
        sort=False,
    ):
        available = group[group.available]
        observed = (
            available[["start_s", "end_s"]].to_numpy(dtype=np.float64)
            if len(available)
            else np.empty((0, 2))
        )
        coverage[(*session, detector)] = Coverage(
            observed=observed,
            exposure=exposure(observed, still.get(tuple(session), np.empty((0, 2)))),
        )
    return coverage
