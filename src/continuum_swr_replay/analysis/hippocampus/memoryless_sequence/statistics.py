"""Per-window statistics on a linearized track graph.

Pure functions. Each takes a posterior over real place bins and a matrix of
track-graph distances between those bins, and returns the quantities the
detector gates on: the center of each window, its spread, the jump to the next
window, and the dispersion of a whole sequence.

Distances are measured on the track graph, not as differences of the linear
coordinate, because two bins that are adjacent in linear position can sit on
different arms (spec 02, V6). The center is the discrete Fréchet mean over real
bins under that distance, and the spread is the square root of the minimized
expected squared distance (D4). On a straight segment with a fine grid both
approach the ordinary center of mass and standard deviation.

A missing center is carried as the sentinel ``NO_CENTER`` rather than NaN, so
centers stay an integer index into the bins.

Deviation numbers (V6, V8, V10 ...), decision numbers (D1 ...) and the
data-flow steps cited below are in ``docs/design/replay_detection_comparison/02-memoryless-sequence.md``.
"""

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "NO_CENTER",
    "aggregate_windows",
    "dispersion_cm",
    "forward_jumps_cm",
    "normalize_windows",
    "window_center_and_spread",
]

NO_CENTER = -1
"""Center index of a window with no usable posterior, or one the speed gate rejected."""


def aggregate_windows(
    log_likelihood: NDArray,
    window_n_samples: int,
    stride_n_samples: int,
) -> NDArray:
    """Sum a log-likelihood over sliding windows.

    The windows overlap, so a row contributes to several of them. Summation is
    in float64 whatever the input dtype, because a float32 cumulative sum over a
    long epoch loses precision that the later subtraction then exposes.

    Parameters
    ----------
    log_likelihood : ndarray, shape (n_time, n_bins)
        Prior-excluded log-likelihood on the native time grid.
    window_n_samples : int
        Rows in one window.
    stride_n_samples : int
        Rows between the starts of consecutive windows.

    Returns
    -------
    window_log_likelihood : ndarray, shape (n_windows, n_bins)
        One row per window whose full support is present. A window that would
        run past the end of the data is not returned.

    """
    values = np.asarray(log_likelihood, dtype=np.float64)
    n_time, n_bins = values.shape
    if window_n_samples < 1 or stride_n_samples < 1:
        raise ValueError(
            f"window_n_samples ({window_n_samples}) and stride_n_samples "
            f"({stride_n_samples}) must both be at least 1."
        )
    n_windows = (
        1 + (n_time - window_n_samples) // stride_n_samples
        if n_time >= window_n_samples
        else 0
    )
    if n_windows <= 0:
        return np.empty((0, n_bins), dtype=np.float64)

    cumulative = np.zeros((n_time + 1, n_bins), dtype=np.float64)
    np.cumsum(values, axis=0, out=cumulative[1:])
    starts = np.arange(n_windows) * stride_n_samples
    return cumulative[starts + window_n_samples] - cumulative[starts]


def normalize_windows(window_log_likelihood: NDArray) -> NDArray:
    """Turn window log-likelihoods into posteriors under a uniform prior.

    Parameters
    ----------
    window_log_likelihood : ndarray, shape (n_windows, n_bins)

    Returns
    -------
    posterior : ndarray, shape (n_windows, n_bins)
        Each row sums to one. A row whose values are all non-finite stays
        non-finite rather than becoming a uniform posterior.

    """
    values = np.asarray(window_log_likelihood, dtype=np.float64)
    # subtract the row maximum before exponentiating, so a large log-likelihood
    # does not overflow; the prior is uniform, so this cancels in the ratio
    shifted = values - np.nanmax(values, axis=1, keepdims=True)
    weights = np.exp(shifted)
    total = weights.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return weights / total


def window_center_and_spread(
    posterior: NDArray, bin_distance_cm: NDArray
) -> tuple[NDArray, NDArray]:
    """Discrete Fréchet mean of each window and its spread.

    The center is the real bin minimizing the posterior-weighted squared track
    distance to every other bin. The spread is the square root of that minimum.
    Ties take the lowest bin index, so the result does not depend on the order
    the windows arrive in.

    Parameters
    ----------
    posterior : ndarray, shape (n_windows, n_bins)
        Rows sum to one. Junction placeholder bins must already be excluded
        (spec 02, V8).
    bin_distance_cm : ndarray, shape (n_bins, n_bins)
        Track-graph distance between bin centers.

    Returns
    -------
    center_index : ndarray of int, shape (n_windows,)
        Index into the bins, or ``NO_CENTER`` where the posterior is unusable.
    spread_cm : ndarray, shape (n_windows,)
        ``nan`` where the center is ``NO_CENTER``.

    """
    posterior = np.asarray(posterior, dtype=np.float64)
    distance = np.asarray(bin_distance_cm, dtype=np.float64)
    if posterior.shape[1] != distance.shape[0]:
        raise ValueError(
            f"posterior has {posterior.shape[1]} bins but bin_distance_cm has "
            f"{distance.shape[0]}."
        )
    n_windows = posterior.shape[0]
    center = np.full(n_windows, NO_CENTER, dtype=int)
    spread = np.full(n_windows, np.nan, dtype=np.float64)
    if n_windows == 0:
        return center, spread

    usable = np.isfinite(posterior).all(axis=1) & (posterior.sum(axis=1) > 0)
    if not usable.any():
        return center, spread

    # A track graph in pieces has inf between bins with no path between them.
    # Zeroing those before the sum keeps 0 * inf out of the cost, which would
    # otherwise be nan and send argmin to the first nan rather than to a center.
    squared = distance**2
    reaches = np.isfinite(squared)
    cost = posterior[usable] @ np.where(reaches, squared, 0.0)

    # a bin can only be the center if every bin holding mass can reach it
    holds_mass = posterior[usable] > 0.0
    reachable = (holds_mass.astype(np.float64) @ (~reaches).astype(np.float64)) == 0.0
    has_center = reachable.any(axis=1)

    best = np.argmin(np.where(reachable, cost, np.inf), axis=1)
    found = np.flatnonzero(usable)[has_center]
    center[found] = best[has_center]
    spread[found] = np.sqrt(
        np.take_along_axis(cost[has_center], best[has_center, None], axis=1).ravel()
    )
    return center, spread


def forward_jumps_cm(center_index: NDArray, bin_distance_cm: NDArray) -> NDArray:
    """Track distance from each window's center to the next window's center.

    Window ``k`` carries the jump to window ``k + 1``, and the last window
    repeats its predecessor's, matching the reference driver's
    ``jumps = [jumps; jumps(end)]`` (spec 02, step 3). A window whose successor
    has no center therefore has no jump, which is why the speed gate must blank
    centers before this is called: a sequence ends one window before the first
    speed-rejected window.

    Parameters
    ----------
    center_index : ndarray of int, shape (n_windows,)
        Bin index per window, ``NO_CENTER`` where absent.
    bin_distance_cm : ndarray, shape (n_bins, n_bins)

    Returns
    -------
    jump_cm : ndarray, shape (n_windows,)
        ``nan`` where the jump is undefined.

    """
    center = np.asarray(center_index, dtype=int)
    distance = np.asarray(bin_distance_cm, dtype=np.float64)
    n_windows = center.size
    jumps = np.full(n_windows, np.nan, dtype=np.float64)
    if n_windows == 0:
        return jumps
    if n_windows > 1:
        here, there = center[:-1], center[1:]
        defined = (here != NO_CENTER) & (there != NO_CENTER)
        jumps[:-1][defined] = distance[here[defined], there[defined]]
        jumps[-1] = jumps[-2]
    return jumps


def dispersion_cm(center_index: NDArray, bin_distance_cm: NDArray) -> float:
    """How far a sequence's centers spread around their own center.

    The same minimization as :func:`window_center_and_spread`, with the
    sequence's retained centers weighted equally. Windows with no center, such
    as those inside a merged gap, take no part.

    Parameters
    ----------
    center_index : ndarray of int, shape (n_windows,)
    bin_distance_cm : ndarray, shape (n_bins, n_bins)

    Returns
    -------
    dispersion_cm : float
        ``nan`` when no window has a center.

    """
    center = np.asarray(center_index, dtype=int)
    distance = np.asarray(bin_distance_cm, dtype=np.float64)
    present = center[center != NO_CENTER]
    if present.size == 0:
        return float("nan")
    cost = (distance[present] ** 2).mean(axis=0)
    return float(np.sqrt(cost.min()))
