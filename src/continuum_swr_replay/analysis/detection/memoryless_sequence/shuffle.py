"""The encoding-pair shuffle: is a candidate sequence more than chance?

Widloski & Foster require a replay to pass a significance test as well as the
duration and dispersion criteria: each event is re-decoded 100 times with
place-cell identities permuted, and both its mean posterior spread and its
mean jump must be smaller than under the shuffles, p < 0.05 each
(``reference/widloski_foster_2025/zenodo_15199609/load_replayEvents_cellIDshuffle.m``,
lines 119-148). The hippocampus here is decoded clusterless, so there are no
cells to permute. The adaptation permutes encoding pairs instead: within each
electrode, the positions at which the encoding spikes were fired are shuffled
against their waveform marks, which breaks the association between a mark and
a place while keeping every marginal distribution, and therefore the occupancy,
the mean rates and the ground-process intensity, exactly as they were. The
event's own spikes and marks are never touched.

Two implementation facts keep this exact and cheap. Pairing marks ``M`` with
positions ``X[pi]`` produces the same set of pairs as pairing ``M[pi^-1]`` with
``X``, and the joint mark intensity is a sum over pairs, so the code permutes
the small mark array rather than the ``(n_encoding, n_bins)`` position kernel.
And the per-spike joint mark intensity is the library's own
``block_estimate_log_joint_mark_intensity``, with spikes assigned to rows by
the library's own ``get_spike_time_bin_ind`` on the full epoch grid, the
detector's (``likelihood.likelihood_grid``), with the ground-process term
scaled to its rows as the epoch decode scales it, so the unshuffled re-decode
is the epoch decode restricted to the candidate's rows.

The candidate's windows are re-decoded as they stand, merged gaps included,
without redetecting anything or reapplying the spread and jump gates. The
statistics and the test follow the corrected reference script
(``load_replayEvents_cellIDshuffle_sameEvents.m``, lines 72-79): both means
skip NaN (``nanmean``), a draw counts against the candidate only when its
statistic is strictly below the observed one (``nansum(shuffle < observed)``,
so a tie or a NaN draw does not count), and a half passes at ``p <= 0.05``.

One line of the reference, ``jumps(jumps==0) = nan``, sets aside every pair
of windows whose continuous center of mass did not move, and it does so in
each draw, since whether a center moves can depend on the fields a draw
assigns. This port's centers are bins, so a literal zero test on them would
also set aside every pair whose center stayed in its 2 cm bin, most pairs on a
slow replay, turning the mean jump into the mean size of a bin crossing. The
set-aside pairs therefore come from a continuous center, separately from the
binned centers whose jumps are averaged.

For sorted cells the rule is the reference's exactly
(:func:`unmoved_center_pairs`): each window is decoded from its own whole-number
spike counts, as the reference decodes window by window, so windows holding
the same spikes get the same center to the last bit, and a pair is set aside
in a draw when that draw's continuous center of mass, ``nansum((1:n) .* X) /
nansum(X)`` in bin units, is unchanged up to rounding (:data:`CENTER_TIE_BINS`). That includes a change of
spikes that leaves the center where it was, such as a posterior saturated in
one bin gaining a spike.

For clusterless marks, on a track graph, there is no continuous center, and
the rule is an approximation read from the spikes (:func:`unchanged_window_pairs`):
a pair is set aside, in every draw alike, when no spike enters or leaves.
Marks are continuous, so no spike can stand in for another, but a change of
spikes that leaves the center exactly unmoved is kept where the reference
would set it aside.

A candidate whose observed statistic is NaN passes that half trivially under
the reference's arithmetic; that cannot happen to a candidate that meets the
dispersion criterion, and ``ShuffleResult.testable`` records it when it does.

For sorted cells the reference's own shuffle is used instead: whole place
fields are swapped between cells (``rateMap(ind_rand, :)``), so each cell's
spikes are decoded with another cell's field. The fields summed over cells,
which is the term for the spikes that did not arrive, does not change.
``shuffle_candidate_sorted`` runs it; the simulations use it, and recorded
data, decoded clusterless, uses the encoding-pair shuffle.
"""

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from continuum_swr_replay.analysis.detection.memoryless_sequence.config import (
    MemorylessSequenceConfig,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.likelihood import (
    encoding_bins_per_row,
    likelihood_grid,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.window_statistics import (
    aggregate_windows,
    forward_jumps_cm,
    normalize_windows,
    window_center_and_spread,
)

__all__ = [
    "CENTER_TIE_BINS",
    "DEFAULT_N_SHUFFLES",
    "MINIMUM_N_SHUFFLES",
    "SIGNIFICANCE_LEVEL",
    "ShuffleResult",
    "ShuffleSeed",
    "candidate_window_log_likelihood",
    "cell_identity_permutation",
    "encoding_permutation",
    "shuffle_candidate",
    "shuffle_candidate_sorted",
    "shuffle_p_value",
    "shuffle_statistics",
    "sorted_candidate_window_log_likelihood",
    "sorted_candidate_set_aside_pairs",
    "unchanged_window_pairs",
    "unmoved_center_pairs",
]

DEFAULT_N_SHUFFLES = 100
"""Shuffled re-decodes per candidate, as in the reference (``numShuffles_replay``)."""

SIGNIFICANCE_LEVEL = 0.05
"""Both p-values must be at or below this, as the reference compares.

The paper says ``p < 0.05``; the script tests ``<= 0.05``. At 100 draws the
p-values are multiples of 1/101, none of which is 0.05, so the two agree.
"""

MINIMUM_N_SHUFFLES = 19
"""The fewest draws at which a p-value at or below 0.05 is possible.

With ``n`` draws the smallest p is ``1 / (n + 1)``, which is 0.05 at 19.
Below that no candidate could pass, and a run in which nothing passed would
still look like a result.
"""

_CELL_IDENTITY_STREAM = 2**32 - 1
"""Seeds the cell-identity shuffle apart from any electrode's stream."""

_LIBRARY_BLOCK_SIZE = 100
"""The library's default decoding block; the result does not depend on it."""


@dataclass(frozen=True)
class ShuffleSeed:
    """What a run's shuffles are seeded from.

    Every draw's permutation is fixed by these two together with the
    candidate, the draw and the electrode, so any single draw can be
    regenerated on its own.

    Attributes
    ----------
    run : int
        The run's seed, recorded in its manifest. Non-negative.
    session : str
        Identifies the epoch, e.g. ``"chimi20200212_02_r1_HPC"``.

    """

    run: int
    session: str

    def __post_init__(self) -> None:
        if isinstance(self.run, bool) or not isinstance(self.run, int | np.integer):
            raise TypeError(f"run must be an integer, got {self.run!r}.")
        if self.run < 0:
            raise ValueError(f"run must be non-negative, got {self.run}.")
        if not isinstance(self.session, str) or not self.session:
            raise ValueError(
                f"session must be a non-empty string, got {self.session!r}."
            )


def encoding_permutation(
    seed: ShuffleSeed, *, candidate: int, draw: int, electrode: int, n: int
) -> NDArray:
    """The permutation of one electrode's encoding positions for one draw.

    Parameters
    ----------
    seed : ShuffleSeed
    candidate : int
        Identifies the candidate within the epoch; its first window index,
        which does not change when other candidates are added or removed.
    draw : int
        1 to ``n_shuffles``. Draw 0 is the observed, unshuffled decode.
    electrode : int
    n : int
        Encoding spikes on the electrode.

    Returns
    -------
    permutation : ndarray of int, shape (n,)
        Encoding spike ``i`` is given the position of encoding spike
        ``permutation[i]``.

    """
    return _seeded_permutation(seed, candidate, draw, electrode, n)


def cell_identity_permutation(
    seed: ShuffleSeed, *, candidate: int, draw: int, n: int
) -> NDArray:
    """The permutation of sorted cells' place fields for one draw.

    Returns
    -------
    permutation : ndarray of int, shape (n,)
        Cell ``i``'s spikes are decoded with the field of cell
        ``permutation[i]``.

    """
    return _seeded_permutation(seed, candidate, draw, _CELL_IDENTITY_STREAM, n)


def _seeded_permutation(
    seed: ShuffleSeed, candidate: int, draw: int, stream: int, n: int
) -> NDArray:
    if draw < 1:
        raise ValueError(
            f"draw must be 1 or more; draw 0 is the observed decode, got {draw}."
        )
    for name, value in (("candidate", candidate), ("stream", stream), ("n", n)):
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}.")
    # a keyed digest rather than hash(): hash() of a str changes between
    # interpreters. Two 32-bit words carry 64 bits of the session's identity.
    digest = hashlib.blake2b(seed.session.encode("utf-8"), digest_size=8).digest()
    session_code = (
        int.from_bytes(digest[:4], "little"),
        int.from_bytes(digest[4:], "little"),
    )
    sequence = np.random.SeedSequence(
        entropy=seed.run, spawn_key=(*session_code, candidate, draw, stream)
    )
    return np.random.default_rng(sequence).permutation(n)


def shuffle_p_value(observed: float, null: NDArray) -> float:
    """The reference's p-value: ``(1 + nansum(null < observed)) / (n + 1)``.

    Small statistics are the extreme ones and the observed decode counts as
    one of the draws. As in the reference, a draw counts against the candidate
    only when it is strictly below the observed value: a tie does not, and a
    NaN draw does not (``nansum`` skips it without shrinking the denominator).
    A NaN observed value is below nothing, so its p is ``1 / (n + 1)``, again
    as the reference's arithmetic has it.

    Parameters
    ----------
    observed : float
    null : ndarray, shape (n_shuffles,)

    Returns
    -------
    p : float

    """
    null = np.asarray(null, dtype=np.float64).ravel()
    if null.size == 0:
        raise ValueError("A p-value needs at least one shuffled draw.")
    with np.errstate(invalid="ignore"):
        below = np.count_nonzero(null < observed)
    return float((1 + below) / (null.size + 1))


@dataclass(frozen=True, eq=False)
class ShuffleResult:
    """One candidate's observed statistics against its shuffled draws.

    The p-values and the verdict are derived rather than stored, so they
    cannot disagree with the draws.

    Attributes
    ----------
    observed_spread_cm, observed_jump_cm : float
        Mean posterior spread and mean non-zero forward jump of the
        unshuffled decode, NaN skipped, as the reference computes them.
    null_spread_cm, null_jump_cm : ndarray, shape (n_shuffles,)
        The same statistics for each shuffled draw, in draw order. At least
        ``MINIMUM_N_SHUFFLES`` of them, so that a pass is possible.

    """

    observed_spread_cm: float
    observed_jump_cm: float
    null_spread_cm: NDArray
    null_jump_cm: NDArray

    def __post_init__(self) -> None:
        spread = np.array(self.null_spread_cm, dtype=np.float64, copy=True)
        jump = np.array(self.null_jump_cm, dtype=np.float64, copy=True)
        if spread.ndim != 1 or spread.shape != jump.shape:
            raise ValueError(
                f"The spread and jump nulls must hold the same number of draws; "
                f"got {spread.shape} and {jump.shape}."
            )
        _check_n_shuffles(spread.size)
        spread.flags.writeable = False
        jump.flags.writeable = False
        object.__setattr__(self, "observed_spread_cm", float(self.observed_spread_cm))
        object.__setattr__(self, "observed_jump_cm", float(self.observed_jump_cm))
        object.__setattr__(self, "null_spread_cm", spread)
        object.__setattr__(self, "null_jump_cm", jump)

    @property
    def n_shuffles(self) -> int:
        return int(self.null_spread_cm.size)

    @property
    def p_spread(self) -> float:
        return shuffle_p_value(self.observed_spread_cm, self.null_spread_cm)

    @property
    def p_jump(self) -> float:
        return shuffle_p_value(self.observed_jump_cm, self.null_jump_cm)

    @property
    def testable(self) -> bool:
        """Both observed statistics are finite.

        Informational: the reference's arithmetic lets a NaN observed
        statistic pass its half, and ``passes`` follows the reference. A
        candidate that meets the dispersion criterion always has a finite
        jump mean, so this is False only for a candidate no run would test.
        """
        return bool(
            np.isfinite(self.observed_spread_cm) and np.isfinite(self.observed_jump_cm)
        )

    @property
    def passes(self) -> bool:
        """Both p-values at or below ``SIGNIFICANCE_LEVEL``, as the reference tests."""
        return bool(
            self.p_spread <= SIGNIFICANCE_LEVEL and self.p_jump <= SIGNIFICANCE_LEVEL
        )


def shuffle_statistics(
    posterior: NDArray,
    bin_distance_cm: NDArray,
    unchanged_pairs: NDArray | None = None,
) -> tuple[float, float]:
    """Mean spread and mean forward jump over a candidate's windows.

    The reference's statistics, computed as ``load_replayEvents_cellIDshuffle_sameEvents.m``
    lines 72-75 compute them: every window of the candidate counts, including
    those inside merged gaps; the final window repeats the jump before it
    (``jumps = [jumps; jumps(end)]``); the jump of a window pair whose
    posterior did not change is set aside (``jumps(jumps==0) = nan``, which is
    what that line does to a continuous center; see the module docstring);
    and both means skip NaN (``nanmean``), so a window with no usable
    posterior drops out rather than spoiling the mean.

    Parameters
    ----------
    posterior : ndarray, shape (n_windows, n_bins)
    bin_distance_cm : ndarray, shape (n_bins, n_bins)
    unchanged_pairs : ndarray of bool, shape (n_windows,), optional
        ``True`` at window ``k`` when windows ``k`` and ``k + 1`` hold the
        same spikes; the last entry repeats the one before it, as the padded
        jump does. Omitted, no pair is set aside.

    Returns
    -------
    mean_spread_cm, mean_jump_cm : float
        NaN when no window contributes: no usable posterior at all, or no
        jump between windows that differ.

    """
    center, spread = window_center_and_spread(posterior, bin_distance_cm)
    jumps = forward_jumps_cm(center, bin_distance_cm)
    if unchanged_pairs is not None:
        unchanged = np.asarray(unchanged_pairs, dtype=bool)
        if unchanged.shape != jumps.shape:
            raise ValueError(
                f"unchanged_pairs {unchanged.shape} must have one entry per window "
                f"{jumps.shape}."
            )
        jumps = np.where(unchanged, np.nan, jumps)
    return _nanmean(spread), _nanmean(jumps)


CENTER_TIE_BINS = 1e-9
"""How far a continuous center may move, in bins, and still count as unmoved.

The reference tests ``jumps == 0`` exactly, which in floating point also
depends on rounding: two windows whose centers agree in exact arithmetic can
differ by about 1e-14 bins, and two implementations of the same arithmetic
disagree there. Far above that rounding and far below any movement (1e-9 bins
is about 2e-9 cm), this reads the reference's rule as exact arithmetic would."""


def unmoved_center_pairs(window_log_likelihood: NDArray) -> NDArray:
    """Which jumps the reference sets aside, on a continuous center of mass.

    ``jumps(jumps==0) = nan`` in ``load_replayEvents_cellIDshuffle_sameEvents.m``,
    with the center as ``compute_centerOfMass_1D.m`` computes it,
    ``nansum((1:n) .* X) / nansum(X)`` in bin units. A pair is set aside when
    the center of window ``k + 1`` is exactly that of window ``k``; the last
    window repeats the entry before it, as its padded jump does.

    Each distinct window is normalized once and its center computed once, so
    windows holding the same spikes, whose log-likelihood rows are identical,
    get the same center to the last bit wherever they sit; the reference
    decodes window by window and gets that for free. "Exactly" means to within
    :data:`CENTER_TIE_BINS`, rounding and nothing more.

    Parameters
    ----------
    window_log_likelihood : ndarray, shape (n_windows, n_bins)
        Bins in track order along a straight track.

    Returns
    -------
    set_aside : ndarray of bool, shape (n_windows,)

    """
    rows, inverse = np.unique(
        np.asarray(window_log_likelihood, dtype=np.float64),
        axis=0,
        return_inverse=True,
    )
    posterior = normalize_windows(rows)
    center = (posterior @ np.arange(posterior.shape[1])) / posterior.sum(axis=1)
    center = center[inverse.ravel()]
    set_aside = np.zeros(center.size, dtype=bool)
    if center.size > 1:
        set_aside[:-1] = np.abs(center[1:] - center[:-1]) <= CENTER_TIE_BINS
        set_aside[-1] = set_aside[-2]
    return set_aside


def _nanmean(values: NDArray) -> float:
    """MATLAB's ``nanmean``: the mean of the non-NaN values, NaN if there are none."""
    kept = values[~np.isnan(values)]
    return float(kept.mean()) if kept.size else float("nan")


def _unchanged_pairs(
    spike_rows: NDArray, n_windows: int, config: MemorylessSequenceConfig
) -> NDArray:
    """Which consecutive window pairs no spike entered or left.

    Window ``k`` covers rows ``[k S, k S + W)``; window ``k + 1`` drops the
    first ``S`` of those rows and adds the ``S`` rows after them. The pair is
    unchanged when neither slice holds a spike. The last window repeats the
    entry before it, as its padded jump repeats the previous jump.

    Parameters
    ----------
    spike_rows : ndarray of int, shape (n_spikes,)
        Each spike's row, counted from the candidate's first row.
    n_windows : int
    config : MemorylessSequenceConfig

    Returns
    -------
    unchanged_pairs : ndarray of bool, shape (n_windows,)

    """
    stride, width = config.stride_n_samples, config.window_n_samples
    rows = np.sort(np.asarray(spike_rows))
    first_row = stride * np.arange(max(n_windows - 1, 0))
    leaving = np.searchsorted(rows, first_row + stride) - np.searchsorted(
        rows, first_row
    )
    entering = np.searchsorted(rows, first_row + width + stride) - np.searchsorted(
        rows, first_row + width
    )
    unchanged = np.zeros(n_windows, dtype=bool)
    unchanged[: n_windows - 1] = (leaving == 0) & (entering == 0)
    if n_windows > 1:
        unchanged[-1] = unchanged[-2]
    return unchanged


def unchanged_window_pairs(
    first_window: int,
    last_window: int,
    time: NDArray,
    spike_times: list[NDArray],
    config: MemorylessSequenceConfig,
) -> NDArray:
    """Which of a candidate's consecutive window pairs hold the same marks.

    The pairs whose jump the reference's ``jumps(jumps==0) = nan`` sets aside
    (see the module docstring) for a clusterless decode: those no spike of
    any electrode entered or left. The same mask is used for the observed
    decode and every shuffled draw, as it is in the reference.

    Parameters
    ----------
    first_window, last_window : int
        Inclusive window indices.
    time : ndarray, shape (n_time,)
        The full decode grid.
    spike_times : list of ndarray
        Per electrode.
    config : MemorylessSequenceConfig

    Returns
    -------
    unchanged_pairs : ndarray of bool, shape (n_windows,)
        ``True`` at window ``k`` when windows ``k`` and ``k + 1`` hold the
        same spikes; the last entry repeats the one before it.

    """
    from non_local_detector.likelihoods.common import get_spike_time_bin_ind

    time, row_start, row_stop = _candidate_rows(first_window, last_window, time, config)
    spike_rows = []
    for times in spike_times:
        times = np.asarray(times, dtype=np.float64)
        times = times[(times >= time[0]) & (times <= time[-1])]
        rows = get_spike_time_bin_ind(times, time)
        spike_rows.append(rows[(rows >= row_start) & (rows < row_stop)] - row_start)
    rows = np.concatenate([np.zeros(0, dtype=int), *spike_rows])
    return _unchanged_pairs(rows, last_window - first_window + 1, config)


def _candidate_rows(
    first_window: int,
    last_window: int,
    time: NDArray,
    config: MemorylessSequenceConfig,
) -> tuple[NDArray, int, int]:
    """The detector's grid, and the rows of it a candidate's windows cover.

    ``time`` is the decode's grid; the detector's is derived from it as the
    epoch decode derives it (``likelihood_grid``).
    """
    time = likelihood_grid(time, config)
    # the final sample is a row no spike can reach, and the epoch decode drops it
    n_decodable = (
        1 + (time.size - 1 - config.window_n_samples) // config.stride_n_samples
    )
    if not 0 <= first_window <= last_window < n_decodable:
        raise ValueError(
            f"Windows {first_window} to {last_window} are not within the "
            f"{n_decodable} windows this grid decodes."
        )
    row_start = first_window * config.stride_n_samples
    row_stop = last_window * config.stride_n_samples + config.window_n_samples
    return time, row_start, row_stop


def _check_n_shuffles(n_shuffles: int) -> None:
    if n_shuffles < MINIMUM_N_SHUFFLES:
        raise ValueError(
            f"n_shuffles must be at least {MINIMUM_N_SHUFFLES}, so that a p-value "
            f"below {SIGNIFICANCE_LEVEL} is possible; got {n_shuffles}."
        )


def _result_from_draws(
    windows: NDArray, bin_distance_cm: NDArray, set_aside: NDArray
) -> ShuffleResult:
    """Draw 0 is the observed decode; the rest are the null.

    ``set_aside`` is one mask for every draw, shape ``(n_windows,)``, or one
    per draw, shape ``(n_draws, n_windows)``.
    """
    masks = np.broadcast_to(np.asarray(set_aside, dtype=bool), windows.shape[:2])
    statistics = np.array(
        [
            shuffle_statistics(normalize_windows(draw), bin_distance_cm, mask)
            for draw, mask in zip(windows, masks, strict=True)
        ]
    )
    return ShuffleResult(
        observed_spread_cm=statistics[0, 0],
        observed_jump_cm=statistics[0, 1],
        null_spread_cm=statistics[1:, 0],
        null_jump_cm=statistics[1:, 1],
    )


def _electrode_waveform_std(waveform_std: object, n_features: int) -> NDArray:
    """Expand a scalar bandwidth to one per feature, as the library does."""
    std = np.asarray(waveform_std, dtype=np.float64)
    return np.full(n_features, float(std)) if std.ndim == 0 else std


def _decode_draws(
    first_window: int,
    last_window: int,
    time: NDArray,
    spike_times: list[NDArray],
    spike_waveform_features: list[NDArray],
    encoding_model: dict,
    config: MemorylessSequenceConfig,
    n_draws: int,
    permutation_for: Callable[[int, int, int], NDArray | None],
) -> tuple[NDArray, NDArray]:
    """Window log-likelihoods of one candidate under each draw's encoding.

    ``permutation_for(draw, electrode, n_encoding)`` returns the permutation
    of that electrode's encoding positions for that draw, or None for the
    observed pairing. Electrodes are the outer loop so that only one
    electrode's position kernel is held at a time.

    Returns
    -------
    window_log_likelihood : ndarray, shape (n_draws, n_windows, n_bins)
    unchanged_pairs : ndarray of bool, shape (n_windows,)
        Which consecutive windows hold the same spikes; see :func:`_unchanged_pairs`.
    """
    from non_local_detector.likelihoods.clusterless_kde import (
        block_estimate_log_joint_mark_intensity,
        kde_distance,
    )
    from non_local_detector.likelihoods.common import get_spike_time_bin_ind

    time, row_start, row_stop = _candidate_rows(first_window, last_window, time, config)

    encoding_positions = encoding_model["encoding_positions"]
    n_electrodes = len(encoding_positions)
    if not len(spike_times) == len(spike_waveform_features) == n_electrodes:
        raise ValueError(
            f"{len(spike_times)} spike trains and {len(spike_waveform_features)} "
            f"feature arrays for {n_electrodes} encoding electrodes."
        )

    environment = encoding_model["environment"]
    interior = np.asarray(environment.is_track_interior_).ravel()
    bin_centers = np.asarray(environment.place_bin_centers_)[interior]
    # per row of the detector's grid, as the epoch decode subtracts it
    ground = np.asarray(
        encoding_model["summed_ground_process_intensity"], dtype=np.float64
    ) * encoding_bins_per_row(config)
    rows = np.broadcast_to(-ground, (n_draws, row_stop - row_start, ground.size)).copy()
    candidate_rows = []

    for electrode in range(n_electrodes):
        times = np.asarray(spike_times[electrode], dtype=np.float64)
        # the library keeps spikes on the closed grid, then bins them
        in_bounds = (times >= time[0]) & (times <= time[-1])
        spike_rows = get_spike_time_bin_ind(times[in_bounds], time)
        in_candidate = (spike_rows >= row_start) & (spike_rows < row_stop)
        if not in_candidate.any():
            continue
        spike_rows = spike_rows[in_candidate] - row_start
        candidate_rows.append(spike_rows)
        decoding_marks = np.asarray(spike_waveform_features[electrode])[in_bounds][
            in_candidate
        ]
        encoding_marks = np.asarray(
            encoding_model["encoding_spike_waveform_features"][electrode]
        )
        n_encoding = encoding_marks.shape[0]
        position_distance = kde_distance(
            bin_centers,
            encoding_positions[electrode],
            std=encoding_model["position_std"],
        )
        waveform_std = _electrode_waveform_std(
            encoding_model["waveform_std"], encoding_marks.shape[1]
        )
        for draw in range(n_draws):
            permutation = permutation_for(draw, electrode, n_encoding)
            if permutation is None:
                marks = encoding_marks
            else:
                # positions X[pi] against marks M are the pairs M[pi^-1] against X
                marks = encoding_marks[np.argsort(permutation)]
            per_spike = np.asarray(
                block_estimate_log_joint_mark_intensity(
                    decoding_marks,
                    cast(Any, marks),
                    cast(Any, waveform_std),
                    encoding_model["occupancy"],
                    encoding_model["mean_rates"][electrode],
                    position_distance,
                    encoding_model.get("block_size", _LIBRARY_BLOCK_SIZE),
                ),
                dtype=np.float64,
            )
            np.add.at(rows[draw], spike_rows, per_spike)

    windows = np.stack(
        [
            aggregate_windows(
                draw_rows, config.window_n_samples, config.stride_n_samples
            )
            for draw_rows in rows
        ]
    )
    rows = np.concatenate([np.zeros(0, dtype=int), *candidate_rows])
    return windows, _unchanged_pairs(rows, windows.shape[1], config)


def _validated_permutation(permutation: NDArray, n: int, electrode: int) -> NDArray:
    permutation = np.asarray(permutation)
    if permutation.shape != (n,) or not np.array_equal(
        np.sort(permutation), np.arange(n)
    ):
        raise ValueError(
            f"The permutation for electrode {electrode} is not a permutation of "
            f"its {n} encoding spikes."
        )
    return permutation


def candidate_window_log_likelihood(
    first_window: int,
    last_window: int,
    time: NDArray,
    spike_times: list[NDArray],
    spike_waveform_features: list[NDArray],
    encoding_model: dict,
    config: MemorylessSequenceConfig,
    permutations: list[NDArray] | None = None,
) -> NDArray:
    """Re-decode a candidate's windows, optionally with encoding pairs permuted.

    With no permutations this is the epoch decode's window log-likelihood over
    windows ``first_window`` to ``last_window``.

    Parameters
    ----------
    first_window, last_window : int
        Inclusive window indices, as in ``CandidateSequences.runs``.
    time : ndarray, shape (n_time,)
        The full decode grid, as given to ``clusterless_epoch_log_likelihood``.
    spike_times, spike_waveform_features : list of ndarray
        Per electrode, in the encoding model's electrode order.
    encoding_model : dict
        From ``load_clusterless_encoding_model``. Not modified.
    config : MemorylessSequenceConfig
    permutations : list of ndarray, optional
        One per electrode: the permutation of that electrode's encoding
        positions, as :func:`encoding_permutation` returns.

    Returns
    -------
    window_log_likelihood : ndarray, shape (n_windows, n_bins)

    """
    if permutations is not None:
        n_electrodes = len(encoding_model["encoding_positions"])
        if len(permutations) != n_electrodes:
            raise ValueError(
                f"{len(permutations)} permutations for {n_electrodes} electrodes; "
                f"one per electrode is required."
            )

    def permutation_for(draw: int, electrode: int, n: int) -> NDArray | None:
        if permutations is None:
            return None
        return _validated_permutation(permutations[electrode], n, electrode)

    windows, _ = _decode_draws(
        first_window,
        last_window,
        time,
        spike_times,
        spike_waveform_features,
        encoding_model,
        config,
        n_draws=1,
        permutation_for=permutation_for,
    )
    return np.asarray(windows[0])


def shuffle_candidate(
    first_window: int,
    last_window: int,
    time: NDArray,
    spike_times: list[NDArray],
    spike_waveform_features: list[NDArray],
    encoding_model: dict,
    bin_distance_cm: NDArray,
    config: MemorylessSequenceConfig,
    *,
    seed: ShuffleSeed,
    n_shuffles: int = DEFAULT_N_SHUFFLES,
) -> ShuffleResult:
    """Test one candidate against ``n_shuffles`` encoding-pair shuffles.

    Parameters
    ----------
    first_window, last_window : int
        Inclusive window indices of the candidate. The first also keys its
        seed, so a candidate's draws do not depend on which others are tested.
    time, spike_times, spike_waveform_features, encoding_model, config
        As for :func:`candidate_window_log_likelihood`.
    bin_distance_cm : ndarray, shape (n_bins, n_bins)
        Track-graph distances between the encoding model's interior bins.
    seed : ShuffleSeed
    n_shuffles : int, default 100

    Returns
    -------
    result : ShuffleResult

    """
    _check_n_shuffles(n_shuffles)

    def permutation_for(draw: int, electrode: int, n: int) -> NDArray | None:
        if draw == 0:
            return None
        return encoding_permutation(
            seed, candidate=first_window, draw=draw, electrode=electrode, n=n
        )

    windows, unchanged = _decode_draws(
        first_window,
        last_window,
        time,
        spike_times,
        spike_waveform_features,
        encoding_model,
        config,
        n_draws=n_shuffles + 1,
        permutation_for=permutation_for,
    )
    return _result_from_draws(windows, bin_distance_cm, unchanged)


def _decode_draws_sorted(
    first_window: int,
    last_window: int,
    time: NDArray,
    spike_times: list[NDArray],
    encoding_model: dict,
    config: MemorylessSequenceConfig,
    permutations: list[NDArray | None],
) -> tuple[NDArray, NDArray]:
    """Window log-likelihoods of one candidate under each draw's cell pairing.

    The row log-likelihood is ``sum_n count_n log field_n(x) - sum_n field_n(x)``,
    as ``predict_sorted_spikes_kde_log_likelihood`` computes it; a draw
    permutes which field each cell's counts meet, which is the first term only.
    As the reference does, each window is decoded from its own spike counts,
    which are whole numbers, so windows holding the same spikes get the same
    log-likelihood to the last bit and :func:`unmoved_center_pairs` finds their
    centers equal; a running sum over rows would not.

    Returns
    -------
    window_log_likelihood : ndarray, shape (n_draws, n_windows, n_bins)
    set_aside : ndarray of bool, shape (n_draws, n_windows)
        Each draw's pairs whose continuous center did not move.
    """
    from non_local_detector.likelihoods.common import get_spike_time_bin_ind

    time, row_start, row_stop = _candidate_rows(first_window, last_window, time, config)
    environment = encoding_model["environment"]
    interior = np.asarray(environment.is_track_interior_).ravel()
    fields = np.asarray(encoding_model["place_fields"], dtype=np.float64)[:, interior]
    if len(spike_times) != fields.shape[0]:
        raise ValueError(
            f"{len(spike_times)} spike trains for {fields.shape[0]} cells; they "
            f"must match one for one."
        )
    counts = np.zeros((row_stop - row_start, fields.shape[0]), dtype=np.int64)
    for cell, cell_times in enumerate(spike_times):
        cell_times = np.asarray(cell_times, dtype=np.float64)
        # the library keeps spikes on the closed grid, then bins them
        cell_times = cell_times[(cell_times >= time[0]) & (cell_times <= time[-1])]
        rows = get_spike_time_bin_ind(cell_times, time)
        rows = rows[(rows >= row_start) & (rows < row_stop)] - row_start
        np.add.at(counts[:, cell], rows, 1)
    # the fit clips every field away from zero, so the log is finite
    log_fields = np.log(fields)
    # per row of the detector's grid, as the epoch decode subtracts it
    absent = np.asarray(
        encoding_model["no_spike_part_log_likelihood"], dtype=np.float64
    )[interior] * encoding_bins_per_row(config)

    width, stride = config.window_n_samples, config.stride_n_samples
    cumulative = np.concatenate(
        [np.zeros((1, counts.shape[1]), dtype=np.int64), np.cumsum(counts, axis=0)]
    )
    start = stride * np.arange((counts.shape[0] - width) // stride + 1)
    window_counts = cumulative[start + width] - cumulative[start]
    distinct, inverse = np.unique(window_counts, axis=0, return_inverse=True)
    distinct = distinct.astype(np.float64)
    windows = np.stack(
        [
            (
                distinct
                @ (log_fields if permutation is None else log_fields[permutation])
                - width * absent
            )[inverse.ravel()]
            for permutation in permutations
        ]
    )
    set_aside = np.stack([unmoved_center_pairs(draw) for draw in windows])
    return windows, set_aside


def sorted_candidate_window_log_likelihood(
    first_window: int,
    last_window: int,
    time: NDArray,
    spike_times: list[NDArray],
    encoding_model: dict,
    config: MemorylessSequenceConfig,
    permutation: NDArray | None = None,
) -> NDArray:
    """Re-decode a candidate's windows from sorted cells, fields optionally swapped.

    With no permutation this is the epoch decode's window log-likelihood over
    windows ``first_window`` to ``last_window``.

    Parameters
    ----------
    first_window, last_window : int
        Inclusive window indices.
    time : ndarray, shape (n_time,)
        The full decode grid.
    spike_times : list of ndarray
        One per cell, in the encoding model's cell order.
    encoding_model : dict
        A sorted-spikes KDE encoding model. Not modified.
    config : MemorylessSequenceConfig
    permutation : ndarray of int, shape (n_cells,), optional
        Cell ``i`` is decoded with the field of cell ``permutation[i]``.

    Returns
    -------
    window_log_likelihood : ndarray, shape (n_windows, n_bins)

    """
    if permutation is not None:
        n_cells = np.asarray(encoding_model["place_fields"]).shape[0]
        permutation = _validated_permutation(permutation, n_cells, 0)
    windows, _ = _decode_draws_sorted(
        first_window,
        last_window,
        time,
        spike_times,
        encoding_model,
        config,
        [permutation],
    )
    return np.asarray(windows[0])


def sorted_candidate_set_aside_pairs(
    first_window: int,
    last_window: int,
    time: NDArray,
    spike_times: list[NDArray],
    encoding_model: dict,
    config: MemorylessSequenceConfig,
    permutations: list[NDArray | None],
) -> NDArray:
    """The jumps each draw of the cell-identity shuffle sets aside.

    What :func:`shuffle_candidate_sorted` sets aside in each of its draws,
    for checking against the reference.

    Parameters
    ----------
    first_window, last_window : int
        Inclusive window indices.
    time : ndarray, shape (n_time,)
        The full decode grid.
    spike_times : list of ndarray
        One per cell, in the encoding model's cell order.
    encoding_model : dict
        A sorted-spikes KDE encoding model. Not modified.
    config : MemorylessSequenceConfig
    permutations : list
        Per draw, None for the observed pairing or a permutation of the cells.

    Returns
    -------
    set_aside : ndarray of bool, shape (n_draws, n_windows)

    """
    n_cells = np.asarray(encoding_model["place_fields"]).shape[0]
    _, set_aside = _decode_draws_sorted(
        first_window,
        last_window,
        time,
        spike_times,
        encoding_model,
        config,
        [
            None
            if permutation is None
            else _validated_permutation(permutation, n_cells, 0)
            for permutation in permutations
        ],
    )
    return set_aside


def shuffle_candidate_sorted(
    first_window: int,
    last_window: int,
    time: NDArray,
    spike_times: list[NDArray],
    encoding_model: dict,
    bin_distance_cm: NDArray,
    config: MemorylessSequenceConfig,
    *,
    seed: ShuffleSeed,
    n_shuffles: int = DEFAULT_N_SHUFFLES,
) -> ShuffleResult:
    """Test one candidate against the reference's place-cell identity shuffle.

    Parameters are as for :func:`shuffle_candidate`, with one spike train per
    sorted cell and a sorted-spikes encoding model.
    """
    _check_n_shuffles(n_shuffles)
    n_cells = np.asarray(encoding_model["place_fields"]).shape[0]
    permutations: list[NDArray | None] = [None] + [
        cell_identity_permutation(seed, candidate=first_window, draw=draw, n=n_cells)
        for draw in range(1, n_shuffles + 1)
    ]
    windows, unchanged = _decode_draws_sorted(
        first_window,
        last_window,
        time,
        spike_times,
        encoding_model,
        config,
        permutations,
    )
    return _result_from_draws(windows, bin_distance_cm, unchanged)
