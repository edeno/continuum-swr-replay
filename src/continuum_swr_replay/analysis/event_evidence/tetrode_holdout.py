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


HPD_COVERAGE = 0.95
"""The coverage of :func:`hpd_lengths`, the event table's (``hpd``)."""

HPD_CHUNK_ROWS = 20_000
"""Rows per batch of :func:`hpd_lengths` (it builds a full-grid copy)."""


def hpd_lengths(
    distribution: NDArray, environment: Any, *, chunk_rows: int = HPD_CHUNK_ROWS
) -> NDArray[np.float64]:
    """Each row's 95 % highest-posterior-density length, cm, as the event
    table's ``hpd`` column is made.

    The decode frame's ``hpd`` (``decode_quality.posterior_analysis.
    get_posterior_stats``) is the library's
    ``get_HPD_spatial_coverage(posterior, get_highest_posterior_threshold(
    posterior, coverage=0.95))`` on the decode's posterior over every place bin
    of the environment, where the junction placeholders (the bins that are not
    track interior, ``edge_id = -1``) hold NaN: the threshold step drops those
    columns, the coverage step counts the bins at or above the threshold and
    multiplies by the first spacing of the place-bin centers. The event table's
    ``hpd_coverage_cm`` is its mean over the event's bins
    (``event_tables.trajectory._hpd_coverage_mean``). Here the same two
    library functions are applied to ``distribution`` laid out the same way.

    Parameters
    ----------
    distribution : ndarray, shape (n_rows, n_interior_bins)
        A distribution over the real place bins (``interior_bin_mask`` order),
        rows summing to 1; a row with a NaN gets NaN. Kept in its own
        precision (the decode's posterior is float32, the held-out
        distribution float64).
    environment : non_local_detector.environment.Environment
        Fitted, one position dimension (a linearized track).
    chunk_rows : int

    Returns
    -------
    ndarray of float64, shape (n_rows,)
    """
    import xarray as xr
    from non_local_detector.model_checking.highest_posterior_density import (
        get_highest_posterior_threshold,
        get_HPD_spatial_coverage,
    )

    # in the distribution's own precision: the threshold step sorts and sums
    # it, and a cast moves a bin at the threshold in or out
    distribution = np.asarray(distribution)
    interior = interior_bin_mask(environment)
    centers = np.asarray(environment.place_bin_centers_, dtype=float).reshape(-1)
    if centers.size != interior.size:
        raise ValueError("hpd_lengths needs a one-dimensional place grid")
    out = np.full(distribution.shape[0], np.nan)
    finite = np.flatnonzero(np.isfinite(distribution).all(axis=1))
    for start in range(0, finite.size, chunk_rows):
        mine = finite[start : start + chunk_rows]
        full = np.full((mine.size, centers.size), np.nan, dtype=distribution.dtype)
        full[:, interior] = distribution[mine]
        posterior = xr.DataArray(
            full, dims=("time", "position"), coords={"position": centers}
        )
        out[mine] = get_HPD_spatial_coverage(
            posterior, get_highest_posterior_threshold(posterior, HPD_COVERAGE)
        )
    return out


def rate_terms(ground: NDArray, counts: NDArray) -> NDArray[np.float64]:
    """The rate part of a tetrode's bin term,
    ``r(t, x) = -ground(t, x) + n_t log ground(t, x)``.

    ``l_k(t, x) = r(t, x) + m(t, x)``, ``m(t, x) = sum_i (log_intensity[i, x] -
    log ground(t, x))`` over the bin's spikes: the Poisson log-likelihood of
    the spike count at place x (less ``log n!``, the same at every x), and the
    log density of the marks given a spike at x. In a silent bin ``r = l_k``.

    The library floors both terms at ``EPS = 1e-15``: ``ground`` is the mean
    rate times the spike density over the occupancy, clipped at ``EPS``
    (``state_evidence.terms.per_tetrode_terms``), and each spike's log joint
    mark intensity is clipped at ``log EPS`` (``clusterless_kde.
    block_estimate_log_joint_mark_intensity``). So ``log ground`` is finite. At
    a place where both are at the floor (no training spikes of the tetrode
    near it), a spike's mark term is ``log EPS - log EPS = 0`` and the whole
    term is the rate part: the floor, not the marks, says that the place is
    unlikely.

    Parameters
    ----------
    ground : ndarray, shape (n_rows, n_place_bins)
        The tetrode's expected spikes per bin at each place, under each row's
        fold's encoding.
    counts : ndarray of int, shape (n_rows,)
        Its spikes in each row's bin.

    Returns
    -------
    ndarray of float64, shape (n_rows, n_place_bins)
    """
    ground = np.asarray(ground, dtype=np.float64)
    return np.asarray(counts, dtype=np.float64)[:, None] * np.log(ground) - ground


GROUND_FLOOR_FACTOR = 1.001
"""A place is at the floor of a tetrode's ground intensity where its expected
spikes are at most this many times the library's ``EPS`` (``EPS`` is stored in
the encoding's float32, 1.0000000036e-15, so an exact comparison would miss
it)."""

INFORMATION_CHUNK_SPIKES = 20_000
"""Spikes per batch of :func:`spike_information`."""


def _information(log_weights: NDArray, n_places: NDArray) -> NDArray:
    """``log(n_places) - H(q)``, ``q`` proportional to ``exp(log_weights)``
    over each row's finite entries, nats."""
    # a row without a finite entry is NaN (its spike is not counted)
    with np.errstate(invalid="ignore", divide="ignore"):
        top = log_weights.max(axis=1, keepdims=True)
        weights = np.exp(log_weights - top)
        total = weights.sum(axis=1, keepdims=True)
        q = weights / total
        cross = np.where(q > 0, q * (log_weights - top), 0.0).sum(axis=1)
        entropy = np.log(total[:, 0]) - cross
        return np.log(n_places) - entropy


def spike_information(
    terms: FoldTerms, *, chunk_spikes: int = INFORMATION_CHUNK_SPIKES
) -> dict[str, NDArray]:
    """Each spike's place information, from its own terms under its bin's
    fold's encoding (nothing of the other tetrodes or of the decode).

    For spike i of tetrode k:

    - ``mark``: ``q_i(x)`` proportional to ``exp(log_intensity[i, x] - log
      ground[k, x])``, the conditional density of the mark given a spike at x,
      normalized over place with equal weight per place bin;
    - ``spike``: ``p_i(x)`` proportional to ``exp(log_intensity[i, x])``, the
      whole per-spike term (rate map and mark together);

    each as ``log(n) - H``, nats: 0 when the spike is equally likely at every
    place (a mark of a cell with no place field), ``log(n)`` when it points to
    one place bin.

    The place bins are the scores' (the real place bins). Floors: where
    tetrode k's ground intensity is at its floor (:data:`GROUND_FLOOR_FACTOR`
    times ``EPS``: no training spike of the tetrode near the place), the
    conditional mark density is 0 over 0 and its value there is the floors'
    ratio, an arbitrary 1; such places are left out of ``q_i`` and of its
    ``n``, which is the number of the tetrode's places above the floor in the
    spike's fold. A spike whose tetrode has fewer than 2 such places has no
    ``mark`` information and is not counted (``counted`` false; the window
    sums keep the count of those counted). A log intensity at its floor
    (``log EPS``) at a place above the ground floor gives that place a weight
    of about 0, which is what it is. ``p_i`` is over every real place bin, the
    log intensity's floor giving a weight of about 0.

    Parameters
    ----------
    terms : FoldTerms
    chunk_spikes : int

    Returns
    -------
    dict
        ``mark`` and ``spike`` (float64, shape (n_spikes,); NaN where not
        counted), ``counted`` (bool, shape (n_spikes,)) and ``n_places``
        (int, shape (n_folds, n_tetrodes): the places above the ground floor).
    """
    from non_local_detector.likelihoods.common import EPS

    ground = np.asarray(terms.ground, dtype=np.float64)
    above = ground > GROUND_FLOOR_FACTOR * EPS
    n_places = above.sum(axis=2)
    log_ground = np.log(ground)
    fold = terms.block[terms.time_bin]
    n_spikes, n_bins = terms.log_intensity.shape
    mark = np.full(n_spikes, np.nan)
    spike = np.full(n_spikes, np.nan)
    counted = n_places[fold, terms.tetrode] >= 2
    for start in range(0, n_spikes, chunk_spikes):
        batch = slice(start, min(start + chunk_spikes, n_spikes))
        f, k = fold[batch], terms.tetrode[batch]
        intensity = terms.log_intensity[batch].astype(np.float64)
        conditional = np.where(above[f, k], intensity - log_ground[f, k], -np.inf)
        mark[batch] = _information(conditional, n_places[f, k])
        spike[batch] = _information(intensity, np.full(intensity.shape[0], n_bins))
    mark[~counted] = np.nan
    spike[~counted] = np.nan
    return {"mark": mark, "spike": spike, "counted": counted, "n_places": n_places}


NULL_DONOR_RULE = (
    "each event window whose bins lie in one fold takes a donor event window of "
    "the same group and fold: the windows of a group and fold, in a random order "
    "of numpy.random.default_rng([seed, fold, crc32(group)]), each take the next "
    "(the last the first); a donor's bins are matched by offset from its first "
    "bin, wrapping when the donor is shorter; a group of one window in a fold, a "
    "window that spans folds and a control window have no donor"
)
"""The shuffled-window reference's donor rule (:func:`null_donors`), as the
run's header records it."""


def null_donors(windows: pd.DataFrame, folds: NDArray, seed: int) -> NDArray[np.int64]:
    """Each window's donor window for the shuffled-window reference
    (:data:`NULL_DONOR_RULE`).

    Parameters
    ----------
    windows : pd.DataFrame
        ``window_id``, ``kind``, ``group``, ``s`` and ``e``.
    folds : ndarray of int, shape (n_time,)
        Each bin's fold.
    seed : int

    Returns
    -------
    ndarray of int, shape (n_windows,)
        The position (in ``windows``) of each window's donor, -1 without one.
    """
    import zlib

    folds = np.asarray(folds)
    s = windows["s"].to_numpy(dtype=np.int64)
    e = windows["e"].to_numpy(dtype=np.int64)
    first = folds[s]
    one_fold = np.array(
        [(folds[a : b + 1] == f).all() for a, b, f in zip(s, e, first, strict=True)],
        dtype=bool,
    )
    eligible = (windows["kind"].to_numpy() == "event") & one_fold
    groups = windows["group"].astype(str).to_numpy()
    donor = np.full(len(windows), -1, dtype=np.int64)
    for group in sorted(set(groups[eligible])):
        for fold in sorted(set(first[eligible & (groups == group)])):
            members = np.flatnonzero(eligible & (groups == group) & (first == fold))
            if members.size < 2:
                continue
            rng = np.random.default_rng([seed, int(fold), zlib.crc32(group.encode())])
            order = members[rng.permutation(members.size)]
            donor[order] = np.roll(order, -1)
    return donor


def null_pairs(
    windows: pd.DataFrame, donors: NDArray, rows: NDArray
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """The scored bins of the shuffled-window reference: each bin of a window
    with a donor and its donor bin (matched by offset from the first bin,
    wrapping), as positions in ``rows``.

    Returns
    -------
    receiver, donor, window : ndarray of int, shape (n_pairs,)
        Positions in ``rows`` of each receiving bin and its donor bin, and the
        receiving window's position in ``windows``.
    """
    s = windows["s"].to_numpy(dtype=np.int64)
    e = windows["e"].to_numpy(dtype=np.int64)
    receivers, donated, owner = [], [], []
    for w in np.flatnonzero(np.asarray(donors) >= 0):
        d = donors[w]
        offset = np.arange(e[w] - s[w] + 1)
        receivers.append(s[w] + offset)
        donated.append(s[d] + offset % (e[d] - s[d] + 1))
        owner.append(np.full(offset.size, w))
    if not receivers:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty
    rows = np.asarray(rows)
    return (
        np.searchsorted(rows, np.concatenate(receivers)),
        np.searchsorted(rows, np.concatenate(donated)),
        np.concatenate(owner),
    )


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
    extras: bool = False,
    null_seed: int = 0,
    log: Callable[[str], None] = print,
) -> tuple[pd.DataFrame, dict]:
    """The held-out score of every window, one row per window and tetrode.

    With ``extras``, three more things per window and tetrode
    (:data:`EXTRA_COLUMNS`):

    - the width of the other tetrodes' distribution as the event table
      measures a posterior's: ``hpd_cm``, the mean over the scored bins of its
      95 % HPD length (:func:`hpd_lengths`);
    - the rate part of each score (:func:`rate_terms`): with ``r(t, x)``, the
      bin term of the tetrode's spike count alone, a score's rate part is the
      same score of ``r``, ``log sum_x exp(r(t, x)) w(t, x)`` for its weights
      ``w`` (the population's ``pi``, the Local kernel, uniform). Its mark part
      is the score less its rate part. That is the predictive score of the
      marks given the count: ``log sum_x exp(l) w - log sum_x exp(r) w = log
      sum_x exp(m) w'`` with ``w'(x)`` proportional to ``exp(r(x)) w(x)``, the
      weights updated by the count. In a silent bin ``r = l``, so the mark
      part is exactly 0 there. Stored: ``gain_rate`` and ``agreement_rate``
      (summed over the scored bins). A reader gets the mark parts as ``gain -
      gain_rate`` and ``agreement - agreement_rate``, and the parts over the
      bins with a spike by subtracting the silent bins' (``gain_silent``,
      ``agreement_silent``, all rate);
    - a shuffled-window reference for the agreement (event windows only): each
      scored bin of the window scored against the other tetrodes'
      distribution at a bin of a donor window (:func:`null_donors`, seeded by
      ``null_seed``: another event window of the same group and fold, so the
      same encodings and the same decode without the tetrode),
      ``population_null = log sum_x exp(l_k(t, x)) pi(t', x)``. The agreement
      is negative in expectation for any distribution unrelated to the scored
      spikes (by Jensen's inequality, when the distribution's mean over such
      bins is near uniform), so 0 is not its level for no shared content; the
      reference is. Stored: ``agreement_null`` (``population_null - uniform``
      summed over the bins scored both at the window and at its donor),
      ``agreement_null_silent``, ``agreement_null_rate``, ``n_null_bins``,
      ``n_null_spikes``, the donor's ``spread_null_cm`` and ``hpd_null_cm``
      (means over the donor bins used) and ``null_window_id``; NaN (and -1)
      for a window without a donor;
    - the place information of the scored tetrode's spikes
      (:func:`spike_information`), event and control windows alike: summed
      over its spikes in the window's scored bins, ``mark_information`` (of
      the conditional mark density given a spike at each place) and
      ``spike_information`` (of the whole per-spike term), with
      ``n_information_spikes``, the spikes counted (the reader divides by
      it). It is a property of each spike and its fold's encoding. The
      record keeps ``information_places``, each fold's and tetrode's places
      above the ground floor (the bound of ``mark_information`` per spike is
      their log).

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
        times the place-bin size); with ``extras``, :data:`EXTRA_COLUMNS`.
    record : dict
        Timings and checks; with ``extras``, ``n_null_windows`` (event windows
        with a donor) and ``n_event_windows_spanning_folds``.
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
    if extras:
        donors = null_donors(windows, folds, null_seed)
        receiver, donor_row, owner = null_pairs(windows, donors, rows)
        record["n_null_windows"] = int((donors >= 0).sum())
        record["n_event_windows_spanning_folds"] = int(
            sum(
                (windows["kind"].to_numpy()[w] == "event")
                and not (np.asarray(folds)[s[w] : e[w] + 1] == folds[s[w]]).all()
                for w in range(len(windows))
            )
        )
        has_donor = donors >= 0
        null_window_id = np.where(
            has_donor, windows["window_id"].to_numpy()[np.maximum(donors, 0)], -1
        )
        # each spike's place information, summed per scored row and tetrode
        information = spike_information(terms)
        record["information_places"] = information["n_places"].tolist()
        position = np.searchsorted(rows, terms.time_bin)
        in_rows = (position < rows.size) & (
            rows[np.minimum(position, rows.size - 1)] == terms.time_bin
        )
        kept = in_rows & information["counted"]
        row_information = {
            name: np.zeros((rows.size, terms.n_tetrodes))
            for name in ("mark", "spike", "counted")
        }
        for name in ("mark", "spike"):
            np.add.at(
                row_information[name],
                (position[kept], terms.tetrode[kept]),
                information[name][kept],
            )
        np.add.at(row_information["counted"], (position[kept], terms.tetrode[kept]), 1)
        del information

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
        if extras:
            rate = rate_terms(terms.ground[terms.block[rows], k], counts[:, k])
            population_rate = predictive_log_score(rate, distribution)
            uniform_rate = logsumexp(rate, axis=1) - np.log(rate.shape[1])
            local_rate = logsumexp(rate + log_local_weights, axis=1)
            width = np.exp(entropy) * bin_cm
            hpd = hpd_lengths(distribution, environment)
            per_bin |= {
                "gain_rate": np.where(scored, population_rate - local_rate, 0.0),
                "agreement_rate": np.where(scored, population_rate - uniform_rate, 0.0),
                "hpd_cm": np.where(scored, hpd, 0.0),
                "mark_information": np.where(
                    scored, row_information["mark"][:, k], 0.0
                ),
                "spike_information": np.where(
                    scored, row_information["spike"][:, k], 0.0
                ),
                "n_information_spikes": np.where(
                    scored, row_information["counted"][:, k], 0.0
                ),
            }
            null = _null_sums(
                mine,
                rate,
                distribution,
                uniform,
                uniform_rate,
                scored & np.isfinite(hpd),
                missing,
                counts[:, k],
                width,
                hpd,
                receiver,
                donor_row,
                owner,
                len(windows),
            )
            del rate
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
                    **(
                        {
                            "gain_rate": sums["gain_rate"],
                            "agreement_rate": sums["agreement_rate"],
                            "hpd_cm": sums["hpd_cm"]
                            / np.maximum(sums["n_scored_bins"], 1),
                            "mark_information": sums["mark_information"],
                            "spike_information": sums["spike_information"],
                            "n_information_spikes": np.rint(
                                sums["n_information_spikes"]
                            ).astype(np.int64),
                            **{
                                name: np.where(has_donor, values, np.nan)
                                for name, values in null.items()
                            },
                            "n_null_bins": null["n_null_bins"].astype(np.int64),
                            "n_null_spikes": null["n_null_spikes"].astype(np.int64),
                            "null_window_id": null_window_id,
                        }
                        if extras
                        else {}
                    ),
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


def _null_sums(
    terms: NDArray,
    rate: NDArray,
    distribution: NDArray,
    uniform: NDArray,
    uniform_rate: NDArray,
    scored: NDArray,
    missing: NDArray,
    counts: NDArray,
    spread: NDArray,
    hpd: NDArray,
    receiver: NDArray,
    donor: NDArray,
    owner: NDArray,
    n_windows: int,
) -> dict[str, NDArray]:
    """One tetrode's shuffled-window reference, summed per window
    (:func:`holdout_scores` with ``extras``): each receiving bin's terms scored
    under its donor bin's distribution. Arrays are over the scored rows;
    ``receiver``, ``donor`` and ``owner`` are :func:`null_pairs`'."""
    weights = distribution[donor]
    population = predictive_log_score(terms[receiver], weights)
    population_rate = predictive_log_score(rate[receiver], weights)
    kept = scored[receiver] & scored[donor] & ~missing[donor] & np.isfinite(population)
    silent = kept & (counts[receiver] == 0)
    agreement = population - uniform[receiver]

    def summed(values: NDArray, where: NDArray) -> NDArray:
        return np.bincount(owner[where], weights=values[where], minlength=n_windows)

    n_bins = np.bincount(owner[kept], minlength=n_windows).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "agreement_null": summed(agreement, kept),
            "agreement_null_silent": summed(agreement, silent),
            "agreement_null_rate": summed(
                population_rate - uniform_rate[receiver], kept
            ),
            "n_null_bins": n_bins,
            "n_null_spikes": summed(counts[receiver].astype(float), kept),
            "spread_null_cm": summed(spread[donor], kept) / n_bins,
            "hpd_null_cm": summed(hpd[donor], kept) / n_bins,
        }


EXTRA_COLUMNS = (
    "gain_rate",
    "agreement_rate",
    "hpd_cm",
    "agreement_null",
    "agreement_null_silent",
    "agreement_null_rate",
    "n_null_bins",
    "n_null_spikes",
    "spread_null_cm",
    "hpd_null_cm",
    "null_window_id",
    "mark_information",
    "spike_information",
    "n_information_spikes",
)
"""The columns :func:`holdout_scores` adds with ``extras`` (a run made with
``--extra-scores``); the canonical run's tables have none of them."""

#: The per-tetrode sums :func:`window_scores` adds over a window's tetrodes.
_SUMMED = ("gain", "gain_silent", "agreement", "agreement_silent", "n_spikes")

_SUMMED_EXTRA = (
    "gain_rate",
    "agreement_rate",
    "agreement_null",
    "agreement_null_silent",
    "agreement_null_rate",
    "n_null_bins",
    "n_null_spikes",
    "mark_information",
    "spike_information",
    "n_information_spikes",
)
"""The per-tetrode sums of :data:`EXTRA_COLUMNS` :func:`window_scores` adds
when the table has them."""

_MEANS_EXTRA = ("hpd_cm", "spread_null_cm", "hpd_null_cm")
"""The per-tetrode means of :data:`EXTRA_COLUMNS` :func:`window_scores` takes
when the table has them."""


def require_extra_columns(table: pd.DataFrame, what: str) -> None:
    """Refuse a table of the canonical held-out tetrode run where ``what``
    needs the columns a run with ``--extra-scores`` adds.

    Raises
    ------
    ValueError
        Naming the missing columns and the run that makes them.
    """
    if missing := [c for c in EXTRA_COLUMNS if c not in table.columns]:
        raise ValueError(
            f"{what} needs {missing}, which only a held-out tetrode run made with "
            "--extra-scores has (scripts/event_evidence/score_tetrode_holdout.py "
            "--extra-scores --output-dir <new folder>); the canonical run has not"
        )


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
        ``agreement_per_spike``. For a table with :data:`EXTRA_COLUMNS` (a run
        with ``--extra-scores``), also the sums over tetrodes of
        :data:`_SUMMED_EXTRA` (NaN in the reference's for a window without a
        donor), the means over tetrodes of :data:`_MEANS_EXTRA`,
        ``null_window_id``, and per spike: ``gain_rate_per_spike``,
        ``gain_mark_per_spike``, ``agreement_rate_per_spike`` and
        ``agreement_mark_per_spike`` (each pair adds to its score per spike),
        ``mark_information_per_spike`` and ``spike_information_per_spike``
        (over ``n_information_spikes``),
        ``agreement_null_per_spike`` (over the spikes of the bins the
        reference scored, ``n_null_spikes``) and
        ``agreement_minus_null_per_spike`` (the agreement per spike minus the
        reference's, NaN without a donor). A canonical table gives the
        canonical columns only (:func:`require_extra_columns` refuses it where
        they are needed).
    """
    keys = ["epoch_id", "animal", "window_id", "event_id", "kind", "group"]
    grouped = table.groupby(keys, sort=False)
    extra = all(c in table.columns for c in EXTRA_COLUMNS)
    means = ["spread_cm", "nonlocal_prob", *(_MEANS_EXTRA if extra else ())]
    windows = grouped[list(_SUMMED)].sum()
    if extra:
        windows[list(_SUMMED_EXTRA)] = grouped[list(_SUMMED_EXTRA)].sum()
        # a window without a donor keeps NaN in the reference's sums
        null = [c for c in _SUMMED_EXTRA if c.startswith("agreement_null")]
        windows[null] = grouped[null].sum(min_count=1)
        windows["null_window_id"] = grouped["null_window_id"].first()
    windows[means] = grouped[means].mean()
    windows = windows.reset_index()
    spikes = windows["n_spikes"].where(windows["n_spikes"] > 0)
    windows["gain_per_spike"] = windows["gain"] / spikes
    windows["spiking_gain_per_spike"] = (
        windows["gain"] - windows["gain_silent"]
    ) / spikes
    windows["silent_gain_per_spike"] = windows["gain_silent"] / spikes
    windows["agreement_per_spike"] = windows["agreement"] / spikes
    if extra:
        windows["gain_rate_per_spike"] = windows["gain_rate"] / spikes
        windows["gain_mark_per_spike"] = (
            windows["gain"] - windows["gain_rate"]
        ) / spikes
        windows["agreement_rate_per_spike"] = windows["agreement_rate"] / spikes
        windows["agreement_mark_per_spike"] = (
            windows["agreement"] - windows["agreement_rate"]
        ) / spikes
        informative = windows["n_information_spikes"].where(
            windows["n_information_spikes"] > 0
        )
        windows["mark_information_per_spike"] = (
            windows["mark_information"] / informative
        )
        windows["spike_information_per_spike"] = (
            windows["spike_information"] / informative
        )
        null_spikes = windows["n_null_spikes"].where(windows["n_null_spikes"] > 0)
        windows["agreement_null_per_spike"] = windows["agreement_null"] / null_spikes
        windows["agreement_minus_null_per_spike"] = (
            windows["agreement_per_spike"] - windows["agreement_null_per_spike"]
        )
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
