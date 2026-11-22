"""Held-out tetrode score of each non-local event: does the rest of the population's
position estimate predict a tetrode left out of the decoder better than local
coding does?

The events are fixed by the all-tetrode decode (the canonical event table). For
each epoch:

1. **Time folds.** The epoch's bins are split into the folds the caller gives
   with their margin; the runner
   (``scripts/event_evidence/score_tetrode_holdout.py``) gives the project's
   fold rule (:func:`epoch_folds`: ``analysis/regression/folds.py``'s
   ``N_FOLDS`` folds of whole trials balanced by destination well, with a
   margin of :data:`TRIAL_MARGIN_S`). The folds are whole trials because the animal
   stays in one patch for long stretches: a place visited in only one stretch
   of time would have no data in that stretch's held-out encoding.
   Every tetrode's encoding is refit, once per fold, on the decode's own
   training bins outside that fold and more than the margin from each of its
   bins (the library's padding of training runs would otherwise reach the
   fold). A bin is scored with the encodings of its own fold, so no scored
   spike fit its own tuning.
2. **Per-tetrode terms.** For tetrode j, bin t and real place bin x, the
   decoder's clusterless spatial log-likelihood ``l_j(t, x)`` (expected firing
   and each spike's mark intensity, ``state_evidence.terms.per_tetrode_terms``)
   and its No-Spike term. The decoder's Local, Non-Local Continuous and Non-Local
   Fragmented states all use the same spatial term; Local adds the
   position kernel around the animal (``local_position_std``) once.
3. **Leave tetrode k out.** The four-state log-likelihood of the other
   tetrodes is smoothed over the whole epoch with the canonical decoder's state
   transitions and initial conditions (``_predict``). At each bin, the position distribution ``pi_{-k}(t, x)`` is
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

What is held out: at a scored bin, the scored tetrode's term and the other
tetrodes' terms come from encodings fit without the bin's fold or its margin,
and the scored tetrode is absent from the reduced likelihood at every bin. What
is not: the state transitions and initial conditions are the canonical
decoder's, fit by EM on the whole epoch with every tetrode; the other folds'
encodings, used at the other bins of the smoothed posterior, were fit on this
fold's spikes from the other tetrodes; and the windows come from the
all-tetrode decode.
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
from continuum_swr_replay.analysis.regression.folds import (
    MAX_SEGMENT_S,
    MIN_TRIAL_BIN_SHARE,
    N_FOLDS,
    task_segment_starts,
    trial_bin_share,
    trial_folds_and_groups,
)
from continuum_swr_replay.decoding.results import ENCODING_KEY
from continuum_swr_replay.intervals import sample_step

TRIAL_MARGIN_S = 1.0
"""The margin of :func:`epoch_folds`' folds, s: training bins this close to any
bin of a fold are left out of its encodings' fit. A trial fold is many separate
segments, so its scored bins neighbor other folds' training bins many times an
epoch. The margin guarantees that no bin is scored under an encoding fit on its
own fold or on training bins within the margin of any bin of its fold. It does
not keep every event within one fold: an event that spans a change of fold has
its parts scored under different folds' encodings, and a part that lies farther
than the margin from the change may be among the training bins of the encoding
that scores the other part."""

NO_SPIKE = "No-Spike"
LOCAL = "Local"


def fold_blocks(time: NDArray, n_folds: int, folds: NDArray) -> NDArray[np.int64]:
    """Each bin's fold: ``folds``, checked.

    Parameters
    ----------
    time : ndarray, shape (n_time,)
        The decode's time axis, s, increasing.
    n_folds : int
    folds : ndarray of int, shape (n_time,)
        Each bin's fold, 0 .. n_folds - 1 (for example :func:`epoch_folds`).

    Returns
    -------
    ndarray of int, shape (n_time,)

    Raises
    ------
    ValueError
        When the folds are not one fold per bin in 0 .. n_folds - 1, each with
        bins.
    """
    time = np.asarray(time, dtype=float)
    blocks = np.asarray(folds, dtype=np.int64)
    if blocks.shape != time.shape or not np.array_equal(
        np.unique(blocks), np.arange(n_folds)
    ):
        raise ValueError(
            f"folds must give each of the {time.size} bins a fold in "
            f"0 .. {n_folds - 1}, each fold with bins"
        )
    return blocks


TRIAL_DESIGN = "trials"
"""The fold design of whole trials balanced by destination well
(:func:`epoch_folds`, with a margin of :data:`TRIAL_MARGIN_S`)."""


def fold_settings() -> dict:
    """The header's record of the fold design: its name, the folds' classes,
    the margin in seconds, the segment cap, the trials' smallest share of the
    bins and the scikit-learn version, whose ``StratifiedGroupKFold`` heuristic
    assigns the folds."""
    import sklearn

    return {
        "fold_design": TRIAL_DESIGN,
        "fold_classes": "to_well",
        "margin_s": TRIAL_MARGIN_S,
        "max_segment_s": MAX_SEGMENT_S,
        "min_trial_bin_share": MIN_TRIAL_BIN_SHARE,
        "sklearn_version": sklearn.__version__,
    }


def epoch_folds(
    time: NDArray, trials: pd.DataFrame
) -> tuple[NDArray[np.int64], int, dict]:
    """Each bin's fold, the margin in bins and the folds' counts.

    The segments are :func:`task_segment_starts`' (each trial from its
    ``movement_start_time`` to the next, of class ``to_well``; the time before
    the first trial and after the last trial's ``end_time``, each its own
    class), cut into pieces of at most ``MAX_SEGMENT_S`` and dealt into
    ``N_FOLDS`` folds balanced by class (``analysis/regression/folds.py``,
    ``trial_folds_and_groups``, whose segments and groups give the counts);
    the margin is :data:`TRIAL_MARGIN_S`.

    Every well with at least ``N_FOLDS`` groups is then sampled in every fold.
    A well with fewer groups is not: its stays are scored with encodings that
    may have no other stay there, which the per-bin ``place_sampling_s``
    (:func:`place_sampling`) and the runner's
    ``immobile_share_place_sampling_0s`` show.

    Parameters
    ----------
    time : ndarray, shape (n_time,)
        The decode's time axis, s.
    trials : pd.DataFrame
        The epoch's trials (``movement_start_time``, ``end_time``,
        ``to_well``). The runners give ``load_data``'s trials (parsed from the
        DIO events), not the session cache's, and each epoch's record keeps
        their digest (``scripts/event_evidence/_per_event_run.trials_input``).

    Returns
    -------
    folds : ndarray of int, shape (n_time,)
    margin_bins : int
    counts : dict
        ``n_segments`` (the segments holding a bin), ``n_groups`` (their
        pieces after the cap) and ``trial_bin_share`` (the share of the bins
        from the first departure to the last trial's end).
    """
    trials = trials.sort_values("movement_start_time")
    departures = trials["movement_start_time"].to_numpy(dtype=float)
    last_end = float(trials["end_time"].iloc[-1])
    starts, classes = task_segment_starts(
        time, departures, trials["to_well"].to_numpy(), last_end
    )
    folds, segment, group = trial_folds_and_groups(
        time, starts, classes, N_FOLDS, MAX_SEGMENT_S
    )
    counts = {
        "n_segments": int(np.unique(segment).size),
        "n_groups": int(group[-1]) + 1,
        "trial_bin_share": trial_bin_share(time, departures, last_end),
    }
    return folds, margin_in_bins(TRIAL_MARGIN_S, time), counts


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
    bin_s = sample_step(np.asarray(time, dtype=float))
    # the tolerance keeps 1.0 / 0.002 at 500 bins, whatever its last float bit
    return int(np.ceil(margin_s / bin_s - 1e-6))


def fold_training_masks(
    is_training: NDArray, blocks: NDArray, n_folds: int, margin_bins: int
) -> list[NDArray[np.bool_]]:
    """The training bins of each fold's encodings: the decode's, outside the
    fold and more than ``margin_bins`` from every bin of it.

    A fold may be many separate segments (:func:`epoch_folds`): a training bin
    is left out when any bin of the fold lies within ``margin_bins`` of it.

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
    masks: Sequence[NDArray],
    blocks: NDArray,
    bin_s: float,
) -> NDArray[np.float32]:
    """At each bin, the seconds its fold's encodings were fit on at the
    animal's place.

    The training bins of a fold's encodings are ``masks``
    (:func:`fold_training_masks`: the decode's training bins outside the fold
    and its margin). A bin's value is ``bin_s`` times the number of them whose
    place bin is the bin's own: 0 where the encoding saw no data at the
    animal's place. Computed from the positions and folds alone, not from any
    score.

    Parameters
    ----------
    place_bin : ndarray of int, shape (n_time,)
        The animal's place bin at each bin (:func:`animal_place_bins`), -1
        where unknown.
    masks : sequence of ndarray of bool, one per fold, each shape (n_time,)
        Each fold's training bins (:func:`fold_training_masks`).
    blocks : ndarray of int, shape (n_time,)
        Each bin's fold.
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
        [np.bincount(place_bin[mask & known], minlength=n_places) for mask in masks]
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
    folds: NDArray,
    margin_bins: int,
    refit: Callable[..., dict] = refit_encoding,
    log: Callable[[str], None] = print,
) -> tuple[FoldTerms, NDArray, NDArray, dict]:
    """The held-out stage of the decoder's likelihood: each fold's
    tetrode terms under encodings refit on the other folds, the per-tetrode
    No-Spike likelihood and the Local kernel.

    The held-out stage of :func:`holdout_scores`.

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
    folds : ndarray of int, shape (n_time,)
        Each bin's fold, 0 .. n_folds - 1 (for example :func:`epoch_folds`).
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
    record : dict
        ``refit_s``, ``terms_s`` (the terms, No-Spike and kernel), ``n_spikes``.

    Raises
    ------
    ValueError
        When the folds are not one fold per bin in 0 .. n_folds - 1, each with
        bins (:func:`fold_blocks`), or when a fold's encoding would take spikes
        from one of its own bins (:func:`check_no_fold_leakage`).
    """
    from non_local_detector.likelihoods.no_spike import (
        predict_no_spike_log_likelihood,
    )

    record: dict[str, Any] = {}
    started = clock.time()
    time = np.asarray(time, dtype=float)
    environment = detector.encoding_model_[ENCODING_KEY]["environment"]
    centers = np.asarray(environment.place_bin_centers_)[interior_bin_mask(environment)]

    blocks = fold_blocks(time, n_folds, folds)
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
    return terms, no_spike, kernel, record


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
    folds: NDArray,
    margin_bins: int,
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
    folds : ndarray of int, shape (n_time,)
        Each bin's fold, 0 .. n_folds - 1 (for example :func:`epoch_folds`).
    margin_bins : int
        Training bins within this many bins of a fold are left out of its
        encodings' fit (:func:`fold_training_masks`).
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

    terms, no_spike, kernel, record = heldout_terms(
        detector,
        time,
        head_position,
        spike_times,
        marks,
        is_training,
        n_folds=n_folds,
        folds=folds,
        margin_bins=margin_bins,
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
        where the scored tetrode spiked), ``silent_gain_per_spike`` (that of
        the bins where it was silent; the two add to ``gain_per_spike``) and
        ``agreement_per_spike``.
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
    windows["silent_gain_per_spike"] = windows["gain_silent"] / spikes
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
