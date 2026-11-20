"""Held-out tetrode score of each non-local event: does the rest of the population's
position estimate predict a tetrode it never saw better than local coding does?

The events are fixed by the all-tetrode decode (the canonical event table). For
each epoch:

1. **Time folds.** The epoch is cut into :data:`N_FOLDS` contiguous blocks of
   equal duration. Every tetrode's encoding is refit, once per block, on the
   decode's own training bins outside that block (and :data:`MARGIN_BINS` from
   its edges, which the library's padding of training runs would otherwise
   reach). A bin is scored with the encodings of its own block, so no scored
   spike fit its own tuning.
2. **Per-tetrode terms.** For tetrode j, bin t and real place bin x, the
   decoder's clusterless spatial log-likelihood ``l_j(t, x)`` (expected firing
   and each spike's mark intensity, ``state_evidence.terms.per_tetrode_terms``)
   and its No-Spike term. The decoder's Local, Non-Local Continuous and Non-Local
   Fragmented states all use the same spatial term; Local adds the
   position kernel around the animal (``local_position_std``) once.
3. **Leave tetrode k out.** The four-state log-likelihood of the other
   tetrodes is smoothed over the whole epoch with the decoder's own transitions
   (``_predict``). At each bin, the position distribution ``pi_{-k}(t, x)`` is
   the posterior summed over the spatial states (Local, Non-Local Continuous,
   Non-Local Fragmented) and renormalized: No-Spike is left out, so a silent
   tetrode does not score for its silence.
4. **Score tetrode k** at each bin of each window, under the same held-out
   encoding:

   - population: ``log sum_x exp(l_k(t, x)) pi_{-k}(t, x)``;
   - local: ``log sum_x exp(l_k(t, x)) K(x | animal(t))``, ``K`` the decoder's
     Local kernel normalized over the place bins;
   - local at the animal's bin: ``l_k(t, animal bin)``;
   - uniform: ``log mean_x exp(l_k(t, x))``, no place preferred.

   A window's gain for tetrode k is the sum over its bins of population minus
   local; its agreement, population minus uniform. Its spikes and its silence
   both count. The gain is positive whenever tetrode k's activity is not local
   and the population's distribution covers it, coherent or not (a burst whose
   tetrodes each follow their own random place scores positive); the agreement
   is positive only when the population's distribution predicts where tetrode k
   points better than no preference does.

Bins are scored one at a time and summed; the joint score of a whole window
would need the posterior over paths.
"""

from __future__ import annotations

import time as clock
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.special import logsumexp

from continuum_swr_replay.analysis.event_evidence.likelihood_advantage import (
    likelihood_advantage_statistics,
)
from continuum_swr_replay.analysis.event_evidence.state_evidence.terms import (
    interior_bin_mask,
    nearest_interior_bins,
    per_tetrode_terms,
    refit_encoding,
    spikes_in_bins,
    training_spike_bins,
)
from continuum_swr_replay.decoding.results import ENCODING_KEY

N_FOLDS = 5
"""Contiguous time blocks per epoch; each block's encodings are fit on the others."""

MARGIN_BINS = 2
"""Training bins this close to a block are left out of its encodings' fit: the
library pads each training run by one bin width (``training_spike_bins``), and
two bins cannot leave a sliver of the block in the fit."""

TRIAL_MARGIN_S = 1.0
"""The margin of :func:`trial_folds`' folds, s: training bins this close to any
bin of a fold are left out of its encodings' fit. A trial fold is many separate
segments, so its scored bins neighbor other folds' training bins about a hundred
times an epoch; 1 s is longer than the longest non-local event, so no event is
scored with an encoding fit on its own edges."""

MAX_SEGMENT_S = 30.0
"""The longest piece of a segment :func:`trial_folds` keeps whole, s: a longer
segment is cut into equal pieces, each its own group, so a long stay is spread
over the folds."""

BEFORE_FIRST_TRIAL = -1
"""The class of the segment before an epoch's first trial."""

AFTER_LAST_TRIAL = -2
"""The class of the segment after an epoch's last trial."""

MIN_TRIAL_BIN_SHARE = 0.8
"""The smallest share of an epoch's bins that its trials must cover
(:func:`task_segment_starts`). Over the 242 analyzed epochs the trials cover
93 % to 99.97 % (median 98.7 %; the time before the first trial is at most 65 s
and after the last at most 120 s), so a smaller share means the trials are not
on the decode's clock, where every bin would fall outside them and the cap would
cut the epoch into blocks of one class."""

NO_SPIKE = "No-Spike"
LOCAL = "Local"


def time_blocks(time: NDArray, n_folds: int = N_FOLDS) -> NDArray[np.int64]:
    """Each bin's block: ``n_folds`` contiguous blocks of equal duration.

    Parameters
    ----------
    time : ndarray, shape (n_time,)
        The decode's time axis, s, increasing.
    n_folds : int

    Returns
    -------
    ndarray of int, shape (n_time,)
        0 .. n_folds - 1.
    """
    time = np.asarray(time, dtype=float)
    span = time[-1] - time[0]
    block = np.floor((time - time[0]) / span * n_folds).astype(np.int64)
    return np.minimum(block, n_folds - 1)


def trial_segments(time: NDArray, segment_starts: NDArray) -> NDArray[np.int64]:
    """Each bin's segment: segment i runs from ``segment_starts[i]`` up to the
    next start. Bins before the first start belong to segment 0, bins after the
    last start to the last segment.

    Parameters
    ----------
    time : ndarray, shape (n_time,)
        The decode's time axis, s, increasing.
    segment_starts : ndarray, shape (n_segments,)
        Each segment's first time, s, strictly increasing.

    Returns
    -------
    ndarray of int, shape (n_time,)
        0 .. n_segments - 1.

    Raises
    ------
    ValueError
        When there is no start, or the starts are not finite and strictly
        increasing.
    """
    starts = np.asarray(segment_starts, dtype=float)
    if starts.size == 0:
        raise ValueError("no segment starts")
    if not np.isfinite(starts).all() or (np.diff(starts) <= 0).any():
        raise ValueError("segment starts must be finite and strictly increasing")
    segment = np.searchsorted(starts, np.asarray(time, dtype=float), side="right") - 1
    return np.maximum(segment, 0).astype(np.int64)


def trial_bin_share(time: NDArray, departures: NDArray, last_trial_end: float) -> float:
    """The share of the bins from the first departure to the last trial's end.

    Parameters
    ----------
    time : ndarray, shape (n_time,)
    departures : ndarray, shape (n_trials,)
        s, increasing.
    last_trial_end : float
        s.
    """
    time = np.asarray(time, dtype=float)
    first = float(np.asarray(departures, dtype=float)[0])
    return float(np.mean((time >= first) & (time < last_trial_end)))


def task_segment_starts(
    time: NDArray,
    departures: NDArray,
    destinations: NDArray,
    last_trial_end: float,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """The segments of an epoch of the bandit task and their classes.

    One segment per trial, from its departure from a well (``movement_start_time``,
    the final poke-out at its origin) to the next departure, so the run to a well
    and the whole stay there are one segment, its class the destination well
    (``to_well``). The time before the first departure is a segment of class
    :data:`BEFORE_FIRST_TRIAL` and the time from the last trial's end (its final
    poke-out at its destination, ``end_time``) a segment of class
    :data:`AFTER_LAST_TRIAL`, when the epoch has bins there.

    Parameters
    ----------
    time : ndarray, shape (n_time,)
        The decode's time axis, s, increasing.
    departures : ndarray, shape (n_trials,)
        Each trial's departure, s, strictly increasing.
    destinations : ndarray of int, shape (n_trials,)
        Each trial's destination well.
    last_trial_end : float
        The last trial's end, s, after its departure.

    Returns
    -------
    starts : ndarray, shape (n_segments,)
        Each segment's first time, s (:func:`trial_segments`).
    classes : ndarray of int, shape (n_segments,)

    Raises
    ------
    ValueError
        When there is no trial, a destination is missing, the departures and
        destinations differ in length, the last trial ends before it departs, a
        departure lies outside the time axis, or the trials cover less than
        :data:`MIN_TRIAL_BIN_SHARE` of the bins (:func:`trial_bin_share`): the
        trials are then not on the decode's clock.
    """
    time = np.asarray(time, dtype=float)
    departures = np.asarray(departures, dtype=float)
    destinations = np.asarray(destinations)
    if departures.size == 0:
        raise ValueError("no trials")
    if destinations.shape != departures.shape:
        raise ValueError(
            f"{destinations.size} destinations for {departures.size} departures"
        )
    if destinations.dtype.kind == "f" and not np.isfinite(destinations).all():
        raise ValueError("every trial needs a destination well")
    if not last_trial_end > departures[-1]:
        raise ValueError("the last trial ends before its departure")
    if departures[0] < time[0] or departures[-1] > time[-1]:
        raise ValueError(
            f"departures from {departures[0]} to {departures[-1]} s lie outside "
            f"the time axis ({time[0]} to {time[-1]} s): are the trials on the "
            "decode's clock?"
        )
    share = trial_bin_share(time, departures, last_trial_end)
    if share < MIN_TRIAL_BIN_SHARE:
        raise ValueError(
            f"the trials cover {share:.1%} of the bins, less than "
            f"{MIN_TRIAL_BIN_SHARE:.0%}: are the trials on the decode's clock?"
        )
    starts = [departures]
    classes = [destinations.astype(np.int64)]
    if departures[0] > time[0]:
        starts.insert(0, time[:1])
        classes.insert(0, np.array([BEFORE_FIRST_TRIAL]))
    if last_trial_end <= time[-1]:
        starts.append(np.array([last_trial_end]))
        classes.append(np.array([AFTER_LAST_TRIAL]))
    return np.concatenate(starts), np.concatenate(classes)


def capped_groups(
    time: NDArray, segment: NDArray, max_segment_s: float | None = MAX_SEGMENT_S
) -> NDArray[np.int64]:
    """Each bin's group: its segment, cut into equal pieces where the segment
    is longer than ``max_segment_s``.

    A segment of ``n`` bins longer than the cap (``m`` bins, the cap over the
    median bin width, rounded down) is cut into ``ceil(n / m)`` pieces of
    consecutive bins whose lengths differ by at most one bin, each at most
    ``m`` bins; a segment of at most ``m`` bins is one group.

    Parameters
    ----------
    time : ndarray, shape (n_time,)
    segment : ndarray of int, shape (n_time,)
        Each bin's segment (:func:`trial_segments`), nondecreasing.
    max_segment_s : float or None
        The cap, s; None leaves every segment whole.

    Returns
    -------
    ndarray of int, shape (n_time,)
        0 .. n_groups - 1, nondecreasing.
    """
    segment = np.asarray(segment, dtype=np.int64)
    if (np.diff(segment) < 0).any():
        raise ValueError("segments must be runs of consecutive bins")
    first = np.r_[0, np.flatnonzero(np.diff(segment)) + 1]
    lengths = np.diff(np.r_[first, segment.size])
    if max_segment_s is None:
        pieces = np.ones(lengths.size, dtype=np.int64)
    else:
        bin_s = float(np.median(np.diff(np.asarray(time, dtype=float))))
        cap_bins = max(int(np.floor(max_segment_s / bin_s + 1e-6)), 1)
        pieces = -(-lengths // cap_bins)  # ceil
    run = np.repeat(np.arange(lengths.size), lengths)
    rank = np.arange(segment.size) - first[run]
    piece = rank * pieces[run] // lengths[run]
    offset = np.r_[0, np.cumsum(pieces)[:-1]]
    return (offset[run] + piece).astype(np.int64)


def trial_folds(
    time: NDArray,
    segment_starts: NDArray,
    segment_classes: NDArray,
    n_folds: int = N_FOLDS,
    max_segment_s: float | None = MAX_SEGMENT_S,
) -> NDArray[np.int64]:
    """Each bin's fold: whole groups of bins, the folds balanced by class.

    scikit-learn's ``StratifiedGroupKFold(n_splits=n_folds, shuffle=False)``
    with one sample per bin, its class the class of the bin's segment
    (:func:`trial_segments`) and its group the bin's piece of that segment
    (:func:`capped_groups`: a segment longer than ``max_segment_s`` is cut into
    equal pieces, so a long stay is spread over the folds). Every bin of a group
    is in one fold, and each class's time is spread over the folds as evenly as
    whole groups allow. Deterministic: without shuffling the splitter places the
    groups in a fixed order (the most bins first).

    With the segments of :func:`task_segment_starts` the folds sample every well
    with at least ``n_folds`` groups in every fold, where contiguous blocks of
    time (:func:`time_blocks`) leave a patch the animal visits in one stretch of
    the epoch to a single block. A well with fewer groups is not: its stays are
    scored with encodings that may have no other stay there, which the per-bin
    ``place_sampling_s`` (:func:`place_sampling`) and the runner's
    ``immobile_share_place_sampling_0s`` show.

    Parameters
    ----------
    time : ndarray, shape (n_time,)
        The decode's time axis, s, increasing.
    segment_starts : ndarray, shape (n_segments,)
        Each segment's first time, s, strictly increasing.
    segment_classes : ndarray of int, shape (n_segments,)
        Each segment's class.
    n_folds : int
    max_segment_s : float or None
        The longest group, s; None leaves every segment whole.

    Returns
    -------
    ndarray of int, shape (n_time,)
        0 .. n_folds - 1; every fold has bins.

    Raises
    ------
    ValueError
        When the classes are not one per start or not all finite, when fewer
        groups than ``n_folds`` hold a bin, or when a fold is left empty.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    classes = np.asarray(segment_classes)
    if classes.shape != np.shape(segment_starts):
        raise ValueError(
            f"{classes.size} segment classes for {np.size(segment_starts)} starts"
        )
    if classes.dtype.kind == "f" and not np.isfinite(classes).all():
        raise ValueError("every segment needs a class (a destination well)")
    segment = trial_segments(time, segment_starts)
    group = capped_groups(time, segment, max_segment_s)
    n_groups = int(group[-1]) + 1
    if n_groups < n_folds:
        raise ValueError(
            f"{n_groups} segments (trials) hold bins, fewer than the {n_folds} folds"
        )
    folds = np.full(segment.size, -1, dtype=np.int64)
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=False)
    for fold, (_, scored) in enumerate(
        splitter.split(np.zeros(segment.size), classes[segment], group)
    ):
        folds[scored] = fold
    empty = np.setdiff1d(np.arange(n_folds), folds)
    if empty.size:
        raise ValueError(f"folds {empty.tolist()} hold no segment")
    return folds


def margin_in_bins(margin_s: float, time: NDArray) -> int:
    """``margin_s`` in bins of the time axis (its median bin width), rounded up.

    Parameters
    ----------
    margin_s : float
        s.
    time : ndarray, shape (n_time,)

    Returns
    -------
    int
    """
    bin_s = float(np.median(np.diff(np.asarray(time, dtype=float))))
    # the tolerance keeps 1.0 / 0.002 at 500 bins, whatever its last float bit
    return int(np.ceil(margin_s / bin_s - 1e-6))


def fold_training_masks(
    is_training: NDArray, blocks: NDArray, n_folds: int, margin_bins: int = MARGIN_BINS
) -> list[NDArray[np.bool_]]:
    """The training bins of each fold's encodings: the decode's, outside the
    fold and more than ``margin_bins`` from every bin of it.

    A fold may be one contiguous block (:func:`time_blocks`) or many separate
    segments (:func:`trial_folds`): a training bin is left out when any bin of
    the fold lies within ``margin_bins`` of it.

    Parameters
    ----------
    is_training : ndarray of bool, shape (n_time,)
        The decode's training bins.
    blocks : ndarray of int, shape (n_time,)
        Each bin's fold.
    n_folds : int
    margin_bins : int

    Returns
    -------
    list of ndarray of bool, length n_folds, each shape (n_time,)
    """
    is_training = np.asarray(is_training, dtype=bool)
    blocks = np.asarray(blocks)
    n_time = blocks.size
    index = np.arange(n_time)
    # the fold's bins in [t - margin, t + margin], from a running count
    first = np.maximum(index - margin_bins, 0)
    last = np.minimum(index + margin_bins + 1, n_time)
    masks = []
    for b in range(n_folds):
        count = np.r_[0, np.cumsum(blocks == b, dtype=np.int64)]
        near = count[last] - count[first] > 0
        masks.append(is_training & ~near)
    return masks


def check_no_fold_leakage(
    time: NDArray,
    position: NDArray,
    masks: Sequence[NDArray],
    blocks: NDArray,
) -> None:
    """Raise if a fold's encoding would take spikes from a bin of the fold.

    The library trains on runs of training bins, padded by a bin width and
    bridged over bins without a position
    (:func:`~.state_evidence.terms.training_spike_bins`), so the bins it takes
    spikes from can reach past the training mask.

    Parameters
    ----------
    time : ndarray, shape (n_time,)
    position : ndarray, shape (n_time,) or (n_time, n_dims)
        The position the encodings are fit with.
    masks : sequence of ndarray of bool, each shape (n_time,)
        Each fold's training bins (:func:`fold_training_masks`).
    blocks : ndarray of int, shape (n_time,)
        Each bin's fold.

    Raises
    ------
    ValueError
        Naming the fold and how many of its bins its encoding would train on.
    """
    blocks = np.asarray(blocks)
    for b, mask in enumerate(masks):
        leaked = training_spike_bins(time, position, mask) & (blocks == b)
        if leaked.any():
            raise ValueError(
                f"fold {b}'s encoding would train on {int(leaked.sum())} of its own "
                f"bins (first at bin {int(np.argmax(leaked))}): widen the margin"
            )


def animal_place_bins(environment: Any, linear_position: NDArray) -> NDArray[np.int64]:
    """The decoder's real place bin of the animal at each bin
    (:func:`~.state_evidence.terms.nearest_interior_bins`); -1 where the
    position is not finite.

    Parameters
    ----------
    environment : non_local_detector.environment.Environment
        The decoder's, fitted.
    linear_position : ndarray, shape (n_time,)
        cm.

    Returns
    -------
    ndarray of int, shape (n_time,)
        Index into the interior bins, or -1.
    """
    linear_position = np.asarray(linear_position, dtype=float)
    out = np.full(linear_position.size, -1, dtype=np.int64)
    finite = np.isfinite(linear_position)
    out[finite] = nearest_interior_bins(environment, linear_position[finite])
    return out


def place_sampling(
    place_bin: NDArray,
    is_training: NDArray,
    blocks: NDArray,
    n_folds: int,
    margin_bins: int,
    bin_s: float,
) -> NDArray[np.float32]:
    """At each bin, the seconds its fold's encodings were fit on at the
    animal's place.

    The training bins of a fold's encodings are :func:`fold_training_masks`'
    (the decode's training bins outside the fold and its margin). A bin's value
    is ``bin_s`` times the number of them whose place bin is the bin's own: 0
    where the encoding saw no data at the animal's place. Computed from the
    positions and folds alone, not from any score.

    Parameters
    ----------
    place_bin : ndarray of int, shape (n_time,)
        The animal's place bin at each bin (:func:`animal_place_bins`), -1
        where unknown.
    is_training : ndarray of bool, shape (n_time,)
    blocks : ndarray of int, shape (n_time,)
        Each bin's fold.
    n_folds, margin_bins : int
        As :func:`fold_training_masks`.
    bin_s : float
        The bin width, s.

    Returns
    -------
    ndarray of float32, shape (n_time,)
        s; NaN where the place is unknown.
    """
    place_bin = np.asarray(place_bin, dtype=np.int64)
    blocks = np.asarray(blocks, dtype=np.int64)
    known = place_bin >= 0
    n_places = int(place_bin.max()) + 1 if known.any() else 1
    occupancy = np.stack(
        [
            np.bincount(place_bin[mask & known], minlength=n_places)
            for mask in fold_training_masks(is_training, blocks, n_folds, margin_bins)
        ]
    )  # (n_folds, n_places), training bins
    out = np.full(place_bin.size, np.nan, dtype=np.float32)
    out[known] = occupancy[blocks[known], place_bin[known]] * bin_s
    return out


def _add_rows(out: NDArray, index: NDArray, values: NDArray) -> None:
    """``np.add.at(out, index, values)`` in passes of distinct rows.

    Pass r adds every row's r-th value (in the order given), so each row of
    ``out`` receives its values in the same order as ``np.add.at`` and the sums
    are the same to the bit; plain fancy-index addition per pass replaces the
    slow unbuffered ``add.at``.
    """
    if index.size == 0:
        return
    order = np.argsort(index, kind="stable")
    sorted_index = index[order]
    first = np.r_[0, np.flatnonzero(np.diff(sorted_index)) + 1]
    rank = np.empty(index.size, dtype=np.int64)
    rank[order] = np.arange(index.size) - np.repeat(
        first, np.diff(np.r_[first, index.size])
    )
    for r in range(int(rank.max()) + 1):
        mine = rank == r
        out[index[mine]] += values[mine]


@dataclass(frozen=True)
class FoldTerms:
    """Per-tetrode spatial log-likelihood terms, each bin under its block's
    encodings.

    ``l_j(t, x) = -ground[block[t], j, x] + sum_{spikes i of j in bin t}
    log_intensity[i, x]``.

    Attributes
    ----------
    ground : ndarray, shape (n_folds, n_tetrodes, n_bins)
        Each block's expected spikes per bin of each tetrode at each place bin.
    log_intensity : ndarray, shape (n_spikes, n_bins)
    time_bin, tetrode : ndarray of int, shape (n_spikes,)
    block : ndarray of int, shape (n_time,)
    """

    ground: NDArray
    log_intensity: NDArray
    time_bin: NDArray
    tetrode: NDArray
    block: NDArray

    @property
    def n_tetrodes(self) -> int:
        return int(self.ground.shape[1])

    def tetrode_terms(self, k: int) -> NDArray:
        """``l_k(t, x)`` at every bin, shape (n_time, n_bins)."""
        out = self.ground[self.block, k].astype(np.float64, copy=False)
        np.negative(out, out=out)
        mine = self.tetrode == k
        _add_rows(out, self.time_bin[mine], self.log_intensity[mine])
        return out

    def total(self) -> NDArray:
        """``sum_j l_j(t, x)`` at every bin, shape (n_time, n_bins)."""
        out = self.ground.sum(axis=1)[self.block].astype(np.float64, copy=False)
        np.negative(out, out=out)
        _add_rows(out, self.time_bin, self.log_intensity)
        return out

    def spike_counts(self) -> NDArray[np.int64]:
        """Each tetrode's spikes in each bin, shape (n_time, n_tetrodes)."""
        counts = np.zeros((self.block.size, self.n_tetrodes), dtype=np.int64)
        np.add.at(counts, (self.time_bin, self.tetrode), 1)
        return counts


def fold_terms(
    encodings: Sequence[dict],
    spike_times: Sequence[NDArray],
    marks: Sequence[NDArray],
    time: NDArray,
    centers: NDArray,
    blocks: NDArray,
) -> FoldTerms:
    """Every spike's terms under the encodings of its bin's block.

    Spikes are binned as the decoder bins them, ``np.digitize(t, time[1:-1])``,
    and those outside ``[time[0], time[-1]]`` are dropped.
    """
    time = np.asarray(time, dtype=float)
    grounds, intensities, time_bins, tetrodes = [], [], [], []
    for b, encoding in enumerate(encodings):
        kept_times, kept_marks = spikes_in_bins(
            list(spike_times), list(marks), time, keep=np.asarray(blocks) == b
        )
        terms = per_tetrode_terms(encoding, kept_times, kept_marks, time, centers)
        grounds.append(terms.ground)
        intensities.append(terms.log_intensity)
        time_bins.append(terms.time_bin)
        tetrodes.append(terms.tetrode)
    return FoldTerms(
        ground=np.stack(grounds),
        log_intensity=np.concatenate(intensities),
        time_bin=np.concatenate(time_bins),
        tetrode=np.concatenate(tetrodes),
        block=np.asarray(blocks, dtype=np.int64),
    )


def state_log_likelihood(
    spatial: NDArray,
    no_spike: NDArray,
    kernel: NDArray,
    state_ind: NDArray,
    state_names: Sequence[str],
    is_missing: NDArray,
) -> NDArray[np.float32]:
    """The decoder's four-state log-likelihood from its pieces.

    Parameters
    ----------
    spatial : ndarray, shape (n_time, n_bins)
        The summed spatial terms of the tetrodes in the model.
    no_spike : ndarray, shape (n_time,)
        Their summed No-Spike terms.
    kernel : ndarray, shape (n_time, n_bins)
        The decoder's log Local kernel (``_compute_local_position_kernel``).
    state_ind : ndarray of int, shape (n_state_bins,)
        Each interior state bin's state.
    state_names : sequence of str
    is_missing : ndarray of bool, shape (n_time,)
        Bins whose likelihood is set to 0, as the decoder sets them.

    Returns
    -------
    ndarray of float32, shape (n_time, n_state_bins)
    """
    out = np.zeros((spatial.shape[0], state_ind.size), dtype=np.float32)
    for s, name in enumerate(state_names):
        # the library lays each state's bins out together (``state_ind_`` is
        # built by concatenating the states), so a slice writes them in place
        where = np.flatnonzero(state_ind == s)
        columns = slice(where[0], where[-1] + 1)
        if where.size != columns.stop - columns.start:
            raise ValueError(f"the bins of state {name!r} are not contiguous")
        if name == NO_SPIKE:
            out[:, columns] = np.asarray(no_spike)[:, None]
        elif name == LOCAL:
            np.add(spatial, kernel, out=out[:, columns], casting="unsafe")
        else:
            out[:, columns] = spatial
    out[np.asarray(is_missing, dtype=bool)] = 0.0
    return out


def spatial_distribution(
    posterior: NDArray, state_ind: NDArray, state_names: Sequence[str]
) -> NDArray:
    """The position distribution over the spatial states, renormalized.

    Parameters
    ----------
    posterior : ndarray, shape (n_rows, n_state_bins)
    state_ind : ndarray of int, shape (n_state_bins,)
    state_names : sequence of str

    Returns
    -------
    ndarray, shape (n_rows, n_bins)
        Rows sum to 1; NaN where the spatial states hold no mass.
    """
    posterior = np.asarray(posterior, dtype=np.float64)
    spatial = [s for s, name in enumerate(state_names) if name != NO_SPIKE]
    total = sum(posterior[:, state_ind == s] for s in spatial)
    mass = total.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(mass > 0, total / mass, np.nan)


def predictive_log_score(terms: NDArray, weights: NDArray) -> NDArray:
    """``log sum_x exp(terms[t, x]) weights[t, x]`` per row; NaN where a row's
    weights are NaN."""
    with np.errstate(divide="ignore"):
        log_weights = np.log(weights)
    return logsumexp(terms + log_weights, axis=1)


def window_rows(starts: NDArray, ends: NDArray) -> NDArray[np.int64]:
    """The distinct bins of windows ``[s, e]`` (inclusive), sorted."""
    lengths = np.asarray(ends) - np.asarray(starts) + 1
    rows = np.repeat(np.asarray(starts), lengths) + (
        np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    )
    return np.unique(rows)


def window_sums(
    values: NDArray, rows: NDArray, starts: NDArray, ends: NDArray
) -> NDArray:
    """Each window's sum of ``values`` over its bins; ``values`` aligned with
    ``rows`` (sorted, containing every bin of every window).

    Parameters
    ----------
    values : ndarray, shape (n_rows,) or (n_rows, n_columns)

    Returns
    -------
    ndarray, shape (n_windows,) or (n_windows, n_columns)
    """
    first = np.searchsorted(rows, starts)
    last = np.searchsorted(rows, ends, side="right")
    cumulative = np.concatenate(
        [np.zeros((1, *np.shape(values)[1:])), np.cumsum(values, axis=0)]
    )
    return cumulative[last] - cumulative[first]


def heldout_terms(
    detector: Any,
    time: NDArray,
    head_position: NDArray,
    spike_times: Sequence[NDArray],
    marks: Sequence[NDArray],
    is_training: NDArray,
    *,
    n_folds: int = N_FOLDS,
    folds: NDArray | None = None,
    margin_bins: int = MARGIN_BINS,
    refit: Callable[..., dict] = refit_encoding,
    log: Callable[[str], None] = print,
) -> tuple[FoldTerms, NDArray, NDArray, NDArray[np.int64], dict]:
    """The held-out stage of the decoder's likelihood: each time block's
    tetrode terms under encodings refit on the other blocks, the per-tetrode
    No-Spike likelihood and the Local kernel.

    Shared by :func:`holdout_scores` and
    ``nonlocal_restriction.heldout_log_likelihood``.

    Parameters
    ----------
    detector : NonLocalClusterlessDetector
        The epoch's fitted decoder (its environment and No-Spike rate).
    time : ndarray, shape (n_time,)
        Its time axis, s.
    head_position : ndarray, shape (n_time, 2)
        The position it was decoded with.
    spike_times, marks : sequence of ndarray, one per tetrode
    is_training : ndarray of bool, shape (n_time,)
        The decode's training bins.
    n_folds : int
    folds : ndarray of int, shape (n_time,), optional
        Each bin's fold, 0 .. n_folds - 1 (for example :func:`trial_folds`);
        by default ``n_folds`` contiguous blocks (:func:`time_blocks`).
    margin_bins : int
        Training bins within this many bins of a fold are left out of its
        encodings' fit (:func:`fold_training_masks`).
    refit : callable
        ``refit(detector, time, position, spike_times, marks, is_training)``,
        an encoding model.
    log : callable

    Returns
    -------
    terms : FoldTerms
        Every spike's terms under the encodings of its bin's block.
    no_spike : ndarray, shape (n_time, n_tetrodes)
        Each tetrode's No-Spike log-likelihood.
    kernel : ndarray, shape (n_time, n_place_bins)
        The decoder's Local position kernel.
    blocks : ndarray of int, shape (n_time,)
        Each bin's fold: its time block, or ``folds``.
    record : dict
        ``refit_s``, ``terms_s`` (the terms, No-Spike and kernel), ``n_spikes``.

    Raises
    ------
    ValueError
        When ``folds`` is not one fold per bin in 0 .. n_folds - 1, each with
        bins, or when a fold's encoding would take spikes from one of its own
        bins (:func:`check_no_fold_leakage`).
    """
    from non_local_detector.likelihoods.no_spike import (
        predict_no_spike_log_likelihood,
    )

    record: dict[str, Any] = {}
    started = clock.time()
    time = np.asarray(time, dtype=float)
    environment = detector.encoding_model_[ENCODING_KEY]["environment"]
    centers = np.asarray(environment.place_bin_centers_)[interior_bin_mask(environment)]

    if folds is None:
        blocks = time_blocks(time, n_folds)
    else:
        blocks = np.asarray(folds, dtype=np.int64)
        if blocks.shape != time.shape or not np.array_equal(
            np.unique(blocks), np.arange(n_folds)
        ):
            raise ValueError(
                f"folds must give each of the {time.size} bins a fold in "
                f"0 .. {n_folds - 1}, each fold with bins"
            )
    masks = fold_training_masks(is_training, blocks, n_folds, margin_bins)
    check_no_fold_leakage(time, head_position, masks, blocks)
    encodings = [
        refit(detector, time, head_position, spike_times, marks, mask) for mask in masks
    ]
    record["refit_s"] = round(clock.time() - started, 1)
    log(f"refit {n_folds} encodings: {record['refit_s']} s")

    t0 = clock.time()
    terms = fold_terms(encodings, spike_times, marks, time, centers, blocks)
    del encodings  # their terms are all the scoring needs
    no_spike = np.column_stack(
        [
            np.asarray(
                predict_no_spike_log_likelihood(time, [st], detector.no_spike_rate)
            )[:, 0]
            for st in spike_times
        ]
    )
    kernel = np.asarray(
        detector._compute_local_position_kernel(time, time, head_position, environment)
    )
    record["terms_s"] = round(clock.time() - t0, 1)
    record["n_spikes"] = int(terms.time_bin.size)
    log(f"terms: {terms.time_bin.size} spikes, {record['terms_s']} s")
    return terms, no_spike, kernel, blocks, record


def holdout_scores(
    detector: Any,
    time: NDArray,
    head_position: NDArray,
    linear_position: NDArray,
    spike_times: Sequence[NDArray],
    marks: Sequence[NDArray],
    is_training: NDArray,
    is_missing: NDArray,
    windows: pd.DataFrame,
    *,
    n_folds: int = N_FOLDS,
    refit: Callable[..., dict] = refit_encoding,
    log: Callable[[str], None] = print,
) -> tuple[pd.DataFrame, dict]:
    """The held-out score of every window, one row per window and tetrode.

    Parameters
    ----------
    detector : NonLocalClusterlessDetector
        The epoch's fitted decoder (its states, transitions and environment).
    time : ndarray, shape (n_time,)
        Its time axis, s.
    head_position : ndarray, shape (n_time, 2)
        The position it was decoded with (``head_position_x``, ``_y``).
    linear_position : ndarray, shape (n_time,)
        The animal's linear position, cm, for its place bin.
    spike_times, marks : sequence of ndarray, one per tetrode
    is_training, is_missing : ndarray of bool, shape (n_time,)
        The decode's masks.
    windows : pd.DataFrame
        ``window_id``, ``s`` and ``e`` (first and last bin, inclusive).
    n_folds : int
    refit : callable
        ``refit(detector, time, position, spike_times, marks, is_training)``,
        an encoding model.
    log : callable

    Returns
    -------
    scores : pd.DataFrame
        One row per window and tetrode: ``window_id``, ``tetrode``, ``n_bins``,
        ``n_scored_bins``, ``n_spikes`` (the tetrode's), ``population``,
        ``local``, ``local_bin``, ``uniform``, ``gain`` (population - local),
        ``gain_bin`` (population - local_bin), ``agreement`` (population -
        uniform), ``nonlocal_prob`` (the mean over the window of the reduced
        decode's non-local probability), ``gain_silent`` and
        ``agreement_silent`` (the part of each from bins where the tetrode did
        not spike) and ``spread_cm`` (the mean over scored bins of the
        population distribution's effective width, ``exp(entropy)`` place bins
        times the place-bin size).
    record : dict
        Timings and checks.
    """
    started = clock.time()
    time = np.asarray(time, dtype=float)
    is_missing = np.asarray(is_missing, dtype=bool)
    environment = detector.encoding_model_[ENCODING_KEY]["environment"]
    interior = interior_bin_mask(environment)
    centers = np.asarray(environment.place_bin_centers_)[interior]
    state_ind = np.asarray(detector.state_ind_)[detector.is_track_interior_state_bins_]
    state_names = list(detector.state_names)
    nonlocal_states = [
        s for s, name in enumerate(state_names) if name not in (LOCAL, NO_SPIKE)
    ]

    terms, no_spike, kernel, _, record = heldout_terms(
        detector,
        time,
        head_position,
        spike_times,
        marks,
        is_training,
        n_folds=n_folds,
        refit=refit,
        log=log,
    )
    total = terms.total()

    s = windows["s"].to_numpy(dtype=np.int64)
    e = windows["e"].to_numpy(dtype=np.int64)
    rows = window_rows(s, e)
    # the decoder's Local kernel at each scored bin, normalized over the bins
    log_local_weights = kernel[rows] - logsumexp(kernel[rows], axis=1, keepdims=True)
    animal_bin = nearest_interior_bins(environment, np.asarray(linear_position)[rows])
    counts = terms.spike_counts()[rows]
    bin_cm = float(np.median(np.diff(np.sort(centers.ravel()))))
    missing = is_missing[rows]
    n_bins = e - s + 1

    parts = []
    record["smooth_s"] = []
    for k in range(terms.n_tetrodes):
        t1 = clock.time()
        tetrode = terms.tetrode_terms(k)
        mine = tetrode[rows]
        reduced = np.subtract(total, tetrode, out=tetrode)  # the others' terms
        likelihood = state_log_likelihood(
            reduced,
            no_spike.sum(axis=1) - no_spike[:, k],
            kernel,
            state_ind,
            state_names,
            is_missing,
        )
        smoothed = detector._predict(
            time=time, log_likelihoods=likelihood, is_missing=is_missing
        )
        posterior = np.asarray(smoothed[0][rows])
        state_probability = np.asarray(smoothed[1][rows])
        del smoothed, likelihood, reduced, tetrode
        distribution = spatial_distribution(posterior, state_ind, state_names)
        population = predictive_log_score(mine, distribution)
        local = logsumexp(mine + log_local_weights, axis=1)
        local_bin = mine[np.arange(rows.size), animal_bin]
        uniform = logsumexp(mine, axis=1) - np.log(mine.shape[1])
        scored = np.isfinite(population) & ~missing
        silent = scored & (counts[:, k] == 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            entropy = -np.nansum(
                np.where(distribution > 0, distribution * np.log(distribution), 0.0),
                axis=1,
            )
        per_bin = {
            "population": np.where(scored, population, 0.0),
            "local": np.where(scored, local, 0.0),
            "local_bin": np.where(scored, local_bin, 0.0),
            "n_scored_bins": scored,
            "uniform": np.where(scored, uniform, 0.0),
            "n_spikes": counts[:, k],
            "nonlocal_prob": state_probability[:, nonlocal_states].sum(axis=1),
            "gain_silent": np.where(silent, population - local, 0.0),
            "agreement_silent": np.where(silent, population - uniform, 0.0),
            "spread_cm": np.where(scored, np.exp(entropy) * bin_cm, 0.0),
        }
        sums = dict(
            zip(
                per_bin,
                window_sums(np.column_stack(list(per_bin.values())), rows, s, e).T,
                strict=True,
            )
        )
        parts.append(
            pd.DataFrame(
                {
                    "window_id": windows["window_id"].to_numpy(),
                    "tetrode": k,
                    "n_bins": n_bins,
                    "n_scored_bins": sums["n_scored_bins"].astype(np.int64),
                    "n_spikes": sums["n_spikes"].astype(np.int64),
                    "population": sums["population"],
                    "local": sums["local"],
                    "local_bin": sums["local_bin"],
                    "uniform": sums["uniform"],
                    "nonlocal_prob": sums["nonlocal_prob"] / n_bins,
                    "gain_silent": sums["gain_silent"],
                    "agreement_silent": sums["agreement_silent"],
                    "spread_cm": sums["spread_cm"]
                    / np.maximum(sums["n_scored_bins"], 1),
                }
            )
        )
        record["smooth_s"].append(round(clock.time() - t1, 1))
        log(f"tetrode {k}: {record['smooth_s'][-1]} s")
    scores = pd.concat(parts, ignore_index=True)
    scores["gain"] = scores["population"] - scores["local"]
    scores["gain_bin"] = scores["population"] - scores["local_bin"]
    scores["agreement"] = scores["population"] - scores["uniform"]
    record["total_s"] = round(clock.time() - started, 1)
    return scores, record


#: The per-tetrode sums :func:`window_scores` adds over a window's tetrodes.
_SUMMED = ("gain", "gain_silent", "agreement", "agreement_silent", "n_spikes")


def window_scores(table: pd.DataFrame) -> pd.DataFrame:
    """Each window's score over its tetrodes.

    Parameters
    ----------
    table : pd.DataFrame
        The run's rows, one per window and tetrode (:func:`holdout_scores` with
        the runner's ``epoch_id``, ``animal``, ``event_id``, ``kind`` and
        ``group``).

    Returns
    -------
    pd.DataFrame
        One row per window: the keys, the sums over tetrodes of
        :data:`_SUMMED`, the mean over tetrodes of ``spread_cm`` and
        ``nonlocal_prob``, and per spike of the window (NaN without spikes):
        ``gain_per_spike``, ``spiking_gain_per_spike`` (the gain of the bins
        where the scored tetrode spiked) and ``agreement_per_spike``.
    """
    keys = ["epoch_id", "animal", "window_id", "event_id", "kind", "group"]
    grouped = table.groupby(keys, sort=False)
    windows = grouped[list(_SUMMED)].sum()
    windows[["spread_cm", "nonlocal_prob"]] = grouped[
        ["spread_cm", "nonlocal_prob"]
    ].mean()
    windows = windows.reset_index()
    spikes = windows["n_spikes"].where(windows["n_spikes"] > 0)
    windows["gain_per_spike"] = windows["gain"] / spikes
    windows["spiking_gain_per_spike"] = (
        windows["gain"] - windows["gain_silent"]
    ) / spikes
    windows["agreement_per_spike"] = windows["agreement"] / spikes
    return windows


def holdout_statistics(
    windows: pd.DataFrame, kind: str, column: str = "gain_per_spike"
) -> pd.DataFrame:
    """Each group's ``column`` over the windows of ``kind`` (``"event"`` or
    ``"control"``): one value per animal, their mean and its interval
    (``likelihood_advantage_statistics``)."""
    return likelihood_advantage_statistics(
        windows[windows["kind"] == kind], column=column
    )
