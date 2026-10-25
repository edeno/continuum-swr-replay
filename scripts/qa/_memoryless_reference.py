"""The Widloski & Foster detector as its MATLAB computes it, for a sorted 1-D track.

An independent re-implementation of the reference's arithmetic, kept apart
from the port in ``continuum_swr_replay.analysis.hippocampus.memoryless_sequence``
so the two can be run on the same input and compared step by step. It differs
from the port exactly where the port adapts the original:

- the center of each window is the continuous center of mass, in bins
  (``compute_centerOfMass_1D.m``), not a place bin;
- the spread is the reference's 1-D branch of ``compute_imageSpread.m``,
  ``sqrt(sum |x - com|^2 p)`` about that continuous center, in bins;
- jumps are ``|com[k + 1] - com[k]|``, padded forward
  (``load_replayEvents_cellIDshuffle.m``, line 82), and in the significance
  test a jump that is exactly zero is set aside (``jumps(jumps==0) = nan``,
  ``load_replayEvents_cellIDshuffle_sameEvents.m``, line 73);
- dispersion is the RMS distance of the retained centers of mass from their
  mean.

Thresholds are taken in cm and divided by the bin size, as the driver divides
``sequence_posteriorSpreadThr / binSize``. Rate maps get the reference's
``+1e-3`` before decoding (``compute_BayesianDecoding_fast.m``). Window
spike counts are supplied by the caller, so both procedures can be handed the
same counts. The helpers the reference does not ship are ``seqle`` and
``compute_sequenceDispersion``; run lengths here are index differences, a run
of one window has length 0 and is dropped, and dispersion follows the paper's
Methods.

Written from the vendored reference and the paper's Methods; nothing here is
imported by the analysis package. Not a script: ``compare_memoryless_to_reference.py``
imports it as ``scripts.qa._memoryless_reference``.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

RATE_MAP_BASELINE_HZ = 1e-3
"""Added to every tuning curve before decoding, as the reference does."""

ZERO_JUMP_BINS = 1e-9
"""A jump no larger than this, in bins, is the reference's ``jumps == 0``.

MATLAB's exact test also turns on its own rounding, which no other
implementation reproduces; centers equal in exact arithmetic differ by about
1e-14 bins after rounding, so this reads the test as exact arithmetic would."""


@dataclass(frozen=True)
class ReferenceThresholds:
    """The reference's detection thresholds, in cm and seconds."""

    spread_cm: float
    jump_cm: float
    speed_cm_per_s: float
    merge_gap_cm: float
    merge_gap_s: float
    minimum_duration_s: float
    dispersion_cm: float
    stride_s: float
    bin_size_cm: float


def decode_windows(
    window_counts: NDArray,
    rate_map_hz: NDArray,
    window_s: float,
    baseline_hz: float = RATE_MAP_BASELINE_HZ,
) -> NDArray:
    """Posterior of each window under a Poisson model and a uniform prior.

    Parameters
    ----------
    window_counts : ndarray, shape (n_windows, n_cells)
    rate_map_hz : ndarray, shape (n_cells, n_bins)
    window_s : float
    baseline_hz : float
        Added to every rate map; 0 decodes the fields as they are.

    Returns
    -------
    posterior : ndarray, shape (n_windows, n_bins)

    """
    rate = np.asarray(rate_map_hz, dtype=np.float64) + baseline_hz
    # the reference decodes window by window, so the same counts give the same
    # posterior to the last bit; decode each distinct window once to match
    counts, inverse = np.unique(
        np.asarray(window_counts, dtype=np.float64), axis=0, return_inverse=True
    )
    log_likelihood = counts @ np.log(rate) - window_s * rate.sum(axis=0)
    posterior = np.exp(
        log_likelihood - logsumexp(log_likelihood, axis=1, keepdims=True)
    )
    return posterior[inverse.ravel()]


def center_and_spread(posterior: NDArray) -> tuple[NDArray, NDArray]:
    """Continuous center of mass and the spread about it, both in bins.

    The center is ``nansum((1:n) .* X) / nansum(X)``, as
    ``compute_centerOfMass_1D.m`` computes it, divided by the posterior's own
    sum rather than assuming it is exactly one.

    Returns
    -------
    center_bins, spread_bins : ndarray, shape (n_windows,)

    """
    posterior = np.asarray(posterior, dtype=np.float64)
    bins = np.arange(posterior.shape[1])
    center = (posterior @ bins) / posterior.sum(axis=1)
    spread = np.sqrt(((bins[None, :] - center[:, None]) ** 2 * posterior).sum(axis=1))
    return center, spread


def forward_jumps(center_bins: NDArray) -> NDArray:
    """``|com[k + 1] - com[k]|``, the last window repeating its predecessor's."""
    center = np.asarray(center_bins, dtype=np.float64)
    if center.size < 2:
        return np.full(center.size, np.nan)
    jumps = np.abs(np.diff(center))
    return np.append(jumps, jumps[-1])


def runs_of_kept(kept: NDArray) -> NDArray:
    """Inclusive bounds of each run of kept windows, runs of one window dropped."""
    kept = np.asarray(kept, dtype=bool)
    edges = np.diff(np.concatenate([[0], kept.astype(int), [0]]))
    runs = np.column_stack(
        [np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1]
    )
    return runs[runs[:, 1] > runs[:, 0]]


def merge_runs(
    runs: NDArray,
    center_bins: NDArray,
    time_s: NDArray,
    thresholds: ReferenceThresholds,
) -> NDArray:
    """Merge neighbours closer than both gaps, as ``compute_allSequences_NaNseparated_merge.m``.

    The gap is measured from the last window of one run to the first of the
    next, in bins and in seconds.
    """
    runs = np.asarray(runs, dtype=int).reshape(-1, 2)
    if len(runs) < 2:
        return runs
    end, start = runs[:-1, 1], runs[1:, 0]
    gap_bins = np.abs(center_bins[start] - center_bins[end])
    gap_s = time_s[start] - time_s[end]
    joins = (gap_bins < thresholds.merge_gap_cm / thresholds.bin_size_cm) & (
        gap_s < thresholds.merge_gap_s
    )
    # a run starts a merged sequence unless it joins the one before
    begins = np.concatenate([[True], ~joins])
    group = np.cumsum(begins) - 1
    merged = np.zeros((group[-1] + 1, 2), dtype=int)
    merged[:, 0] = runs[begins, 0]
    np.maximum.at(merged[:, 1], group, runs[:, 1])
    return merged


def dispersion_bins(center_bins: NDArray) -> float:
    """RMS distance of the retained centers from their mean; NaN windows skipped."""
    center = np.asarray(center_bins, dtype=np.float64)
    center = center[~np.isnan(center)]
    if center.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((center - center.mean()) ** 2)))


@dataclass(frozen=True)
class ReferenceDetection:
    """What the reference computes for one epoch.

    Attributes
    ----------
    center_bins, spread_bins, jump_bins : ndarray, shape (n_windows,)
        NaN where the speed gate removed the window.
    kept : ndarray of bool, shape (n_windows,)
    runs : ndarray of int, shape (n_sequences, 2)
        Inclusive window bounds after merging and the duration filter.
    duration_s, dispersion_bins : ndarray, shape (n_sequences,)
    is_replay : ndarray of bool, shape (n_sequences,)
    """

    center_bins: NDArray
    spread_bins: NDArray
    jump_bins: NDArray
    kept: NDArray
    runs: NDArray
    duration_s: NDArray
    dispersion_bins: NDArray
    is_replay: NDArray


def detect(
    posterior: NDArray,
    time_s: NDArray,
    speed_cm_per_s: NDArray,
    thresholds: ReferenceThresholds,
) -> ReferenceDetection:
    """The reference driver, ``load_replayEvents_cellIDshuffle.m`` lines 72-110.

    Parameters
    ----------
    posterior : ndarray, shape (n_windows, n_bins)
    time_s : ndarray, shape (n_windows,)
        Window midpoints.
    speed_cm_per_s : ndarray, shape (n_windows,)
    thresholds : ReferenceThresholds

    Returns
    -------
    detection : ReferenceDetection

    """
    center, spread = center_and_spread(posterior)
    # windows the speed gate removes are never decoded, so they carry no center
    moving = ~(np.asarray(speed_cm_per_s) < thresholds.speed_cm_per_s)
    center[moving] = np.nan
    spread[moving] = np.nan
    jumps = forward_jumps(center)
    kept = (spread < thresholds.spread_cm / thresholds.bin_size_cm) & (
        jumps < thresholds.jump_cm / thresholds.bin_size_cm
    )
    runs = merge_runs(runs_of_kept(kept), center, time_s, thresholds)
    runs = runs[
        (runs[:, 1] - runs[:, 0])
        >= thresholds.minimum_duration_s / thresholds.stride_s - 1e-9
    ]
    duration = time_s[runs[:, 1]] - time_s[runs[:, 0]]
    dispersion = np.array(
        [
            dispersion_bins(np.where(kept[a : b + 1], center[a : b + 1], np.nan))
            for a, b in runs
        ],
        dtype=np.float64,
    )
    is_replay = (duration > thresholds.minimum_duration_s) & (
        dispersion > thresholds.dispersion_cm / thresholds.bin_size_cm
    )
    return ReferenceDetection(
        center_bins=center,
        spread_bins=spread,
        jump_bins=jumps,
        kept=kept,
        runs=runs,
        duration_s=duration,
        dispersion_bins=dispersion,
        is_replay=is_replay,
    )


def set_aside_pairs(
    window_counts: NDArray,
    rate_map_hz: NDArray,
    window_s: float,
    baseline_hz: float = RATE_MAP_BASELINE_HZ,
) -> NDArray:
    """Which jumps ``jumps(jumps==0) = nan`` sets aside, after the forward pad.

    Returns
    -------
    set_aside : ndarray of bool, shape (n_windows,)

    """
    center, _ = center_and_spread(
        decode_windows(window_counts, rate_map_hz, window_s, baseline_hz)
    )
    set_aside: NDArray = forward_jumps(center) <= ZERO_JUMP_BINS
    return set_aside


def shuffle_statistics(
    window_counts: NDArray,
    rate_map_hz: NDArray,
    window_s: float,
    baseline_hz: float = RATE_MAP_BASELINE_HZ,
) -> tuple[float, float]:
    """Mean spread and mean jump of a candidate's windows, zero jumps set aside.

    ``load_replayEvents_cellIDshuffle_sameEvents.m`` lines 72-75, on every
    window of the candidate.
    """
    center, spread = center_and_spread(
        decode_windows(window_counts, rate_map_hz, window_s, baseline_hz)
    )
    jumps = forward_jumps(center)
    jumps[jumps <= ZERO_JUMP_BINS] = np.nan
    return float(np.nanmean(spread)), float(np.nanmean(jumps))


def shuffle_p_values(
    window_counts: NDArray,
    rate_map_hz: NDArray,
    window_s: float,
    permutations: list[NDArray],
    baseline_hz: float = RATE_MAP_BASELINE_HZ,
) -> tuple[float, float]:
    """The reference's p-values, each draw's rate map permuted between cells.

    ``(1 + nansum(null < observed)) / (1 + n_shuffles)``, lines 78-79.
    """
    rate_map_hz = np.asarray(rate_map_hz)
    observed = shuffle_statistics(window_counts, rate_map_hz, window_s, baseline_hz)
    null = np.array(
        [
            shuffle_statistics(
                window_counts, rate_map_hz[permutation], window_s, baseline_hz
            )
            for permutation in permutations
        ]
    )
    n = len(permutations)
    return (
        (1 + np.sum(null[:, 0] < observed[0])) / (1 + n),
        (1 + np.sum(null[:, 1] < observed[1])) / (1 + n),
    )
