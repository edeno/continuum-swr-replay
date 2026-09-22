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
the library's own ``get_spike_time_bin_ind`` on the full epoch grid, so the
unshuffled re-decode is the epoch decode restricted to the candidate's rows.

The candidate's windows are re-decoded as they stand, merged gaps included,
without redetecting anything or reapplying the spread and jump gates. Two
choices differ from the reference script, both deliberately: zero jumps are
kept rather than dropped, and a shuffled draw that ties the observed value
counts against the candidate, because centers on a discrete track graph can
repeat exactly. A candidate whose observed or shuffled statistic is not finite
is untestable, and an untestable candidate does not pass.

The design is in ``docs/design/replay_detection_comparison/02-memoryless-sequence.md``,
section "Clusterless shuffle: permute encoding pairs".
"""

import zlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.config import (
    MemorylessSequenceConfig,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.likelihood import (
    validate_time_grid,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.statistics import (
    aggregate_windows,
    forward_jumps_cm,
    normalize_windows,
    window_center_and_spread,
)

__all__ = [
    "DEFAULT_N_SHUFFLES",
    "SIGNIFICANCE_LEVEL",
    "ShuffleResult",
    "ShuffleSeed",
    "candidate_window_log_likelihood",
    "encoding_permutation",
    "shuffle_candidate",
    "shuffle_p_value",
    "shuffle_statistics",
]

DEFAULT_N_SHUFFLES = 100
"""Shuffled re-decodes per candidate, as in the reference (``numShuffles_replay``)."""

SIGNIFICANCE_LEVEL = 0.05
"""Both p-values must fall below this.

The reference compares with ``<=``, the paper says ``p < 0.05``. At 100 draws
the p-values are multiples of 1/101, none of which is 0.05, so the two agree.
"""


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
        if not self.session:
            raise ValueError("session must name the epoch.")


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
    if draw < 1:
        raise ValueError(
            f"draw must be 1 or more; draw 0 is the observed decode, got {draw}."
        )
    for name, value in (("candidate", candidate), ("electrode", electrode), ("n", n)):
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}.")
    # crc32 rather than hash(): hash() of a str changes between interpreters
    session_code = zlib.crc32(seed.session.encode("utf-8"))
    sequence = np.random.SeedSequence(
        entropy=seed.run, spawn_key=(session_code, candidate, draw, electrode)
    )
    return np.random.default_rng(sequence).permutation(n)


def shuffle_p_value(observed: float, null: NDArray) -> float:
    """Share of draws at least as extreme as the observed value, conservatively.

    ``(1 + count(null <= observed)) / (n + 1)``: small statistics are the
    extreme ones, a tie counts against the candidate, and the observed decode
    counts as one of the draws.

    Parameters
    ----------
    observed : float
    null : ndarray, shape (n_shuffles,)

    Returns
    -------
    p : float
        NaN when the observed value or any draw is not finite. A non-finite
        draw is not dropped: dropping it would shrink the denominator.

    """
    null = np.asarray(null, dtype=np.float64).ravel()
    if null.size == 0:
        raise ValueError("A p-value needs at least one shuffled draw.")
    if not (np.isfinite(observed) and np.isfinite(null).all()):
        return float("nan")
    return float((1 + np.count_nonzero(null <= observed)) / (null.size + 1))


@dataclass(frozen=True)
class ShuffleResult:
    """One candidate's observed statistics against its shuffled draws.

    The p-values and the verdict are derived rather than stored, so they
    cannot disagree with the draws.

    Attributes
    ----------
    observed_spread_cm, observed_jump_cm : float
        Mean posterior spread and mean forward jump of the unshuffled decode.
    null_spread_cm, null_jump_cm : ndarray, shape (n_shuffles,)
        The same statistics for each shuffled draw, in draw order.

    """

    observed_spread_cm: float
    observed_jump_cm: float
    null_spread_cm: NDArray
    null_jump_cm: NDArray

    def __post_init__(self) -> None:
        spread = np.asarray(self.null_spread_cm, dtype=np.float64)
        jump = np.asarray(self.null_jump_cm, dtype=np.float64)
        if spread.ndim != 1 or spread.shape != jump.shape or spread.size == 0:
            raise ValueError(
                f"The spread and jump nulls must hold the same number of draws, "
                f"at least one; got {spread.shape} and {jump.shape}."
            )
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
        return bool(np.isfinite(self.p_spread) and np.isfinite(self.p_jump))

    @property
    def passes(self) -> bool:
        return bool(
            self.testable
            and self.p_spread < SIGNIFICANCE_LEVEL
            and self.p_jump < SIGNIFICANCE_LEVEL
        )


def shuffle_statistics(
    posterior: NDArray, bin_distance_cm: NDArray
) -> tuple[float, float]:
    """Mean spread and mean forward jump over every window of a candidate.

    Unlike candidate assembly, nothing is masked: every window counts,
    including those inside merged gaps, and a jump of zero is a jump. The
    final window repeats the jump before it, matching the reference's
    length-N jump vector.

    Parameters
    ----------
    posterior : ndarray, shape (n_windows, n_bins)
    bin_distance_cm : ndarray, shape (n_bins, n_bins)

    Returns
    -------
    mean_spread_cm, mean_jump_cm : float
        Not finite when any window has no usable posterior.

    """
    center, spread = window_center_and_spread(posterior, bin_distance_cm)
    jumps = forward_jumps_cm(center, bin_distance_cm)
    return float(np.mean(spread)), float(np.mean(jumps))


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
) -> NDArray:
    """Window log-likelihoods of one candidate under each draw's encoding.

    ``permutation_for(draw, electrode, n_encoding)`` returns the permutation
    of that electrode's encoding positions for that draw, or None for the
    observed pairing. Electrodes are the outer loop so that only one
    electrode's position kernel is held at a time.

    Returns
    -------
    window_log_likelihood : ndarray, shape (n_draws, n_windows, n_bins)
    """
    from non_local_detector.likelihoods.clusterless_kde import (
        block_estimate_log_joint_mark_intensity,
        kde_distance,
    )
    from non_local_detector.likelihoods.common import get_spike_time_bin_ind

    time = np.asarray(time, dtype=np.float64)
    validate_time_grid(time, config)
    # the final sample is a row no spike can reach, and the epoch decode drops it
    n_decodable = (
        1 + (time.size - 1 - config.window_n_samples) // config.stride_n_samples
    )
    if not 0 <= first_window <= last_window < n_decodable:
        raise ValueError(
            f"Windows {first_window} to {last_window} are not within the "
            f"{n_decodable} windows this grid decodes."
        )

    encoding_positions = encoding_model["encoding_positions"]
    n_electrodes = len(encoding_positions)
    if not len(spike_times) == len(spike_waveform_features) == n_electrodes:
        raise ValueError(
            f"{len(spike_times)} spike trains and {len(spike_waveform_features)} "
            f"feature arrays for {n_electrodes} encoding electrodes."
        )

    row_start = first_window * config.stride_n_samples
    row_stop = last_window * config.stride_n_samples + config.window_n_samples

    environment = encoding_model["environment"]
    interior = np.asarray(environment.is_track_interior_).ravel()
    bin_centers = np.asarray(environment.place_bin_centers_)[interior]
    ground = np.asarray(
        encoding_model["summed_ground_process_intensity"], dtype=np.float64
    )
    rows = np.broadcast_to(-ground, (n_draws, row_stop - row_start, ground.size)).copy()

    for electrode in range(n_electrodes):
        times = np.asarray(spike_times[electrode], dtype=np.float64)
        # the library keeps spikes on the closed grid, then bins them
        in_bounds = (times >= time[0]) & (times <= time[-1])
        spike_rows = get_spike_time_bin_ind(times[in_bounds], time)
        in_candidate = (spike_rows >= row_start) & (spike_rows < row_stop)
        if not in_candidate.any():
            continue
        spike_rows = spike_rows[in_candidate] - row_start
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
                    encoding_model["block_size"],
                ),
                dtype=np.float64,
            )
            np.add.at(rows[draw], spike_rows, per_spike)

    return np.stack(
        [
            aggregate_windows(
                draw_rows, config.window_n_samples, config.stride_n_samples
            )
            for draw_rows in rows
        ]
    )


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

    windows = _decode_draws(
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
    if n_shuffles < 1:
        raise ValueError(f"n_shuffles must be at least 1, got {n_shuffles}.")

    def permutation_for(draw: int, electrode: int, n: int) -> NDArray | None:
        if draw == 0:
            return None
        return encoding_permutation(
            seed, candidate=first_window, draw=draw, electrode=electrode, n=n
        )

    windows = _decode_draws(
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
    statistics = np.array(
        [
            shuffle_statistics(normalize_windows(draw), bin_distance_cm)
            for draw in windows
        ]
    )
    return ShuffleResult(
        observed_spread_cm=statistics[0, 0],
        observed_jump_cm=statistics[0, 1],
        null_spread_cm=statistics[1:, 0],
        null_jump_cm=statistics[1:, 1],
    )
