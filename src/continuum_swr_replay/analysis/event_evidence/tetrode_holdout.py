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
)
from continuum_swr_replay.decoding.results import ENCODING_KEY

N_FOLDS = 5
"""Contiguous time blocks per epoch; each block's encodings are fit on the others."""

MARGIN_BINS = 2
"""Training bins this close to a block are left out of its encodings' fit: the
library pads each training run by one bin width (``training_spike_bins``), and
two bins cannot leave a sliver of the block in the fit."""

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


def fold_training_masks(
    is_training: NDArray, blocks: NDArray, n_folds: int, margin_bins: int = MARGIN_BINS
) -> list[NDArray[np.bool_]]:
    """The training bins of each block's encodings: the decode's, outside the
    block and ``margin_bins`` from it.

    Returns
    -------
    list of ndarray of bool, length n_folds, each shape (n_time,)
    """
    is_training = np.asarray(is_training, dtype=bool)
    width = 2 * margin_bins + 1
    masks = []
    for b in range(n_folds):
        near = (
            np.convolve(
                (blocks == b).astype(np.int64), np.ones(width, np.int64), "same"
            )
            > 0
        )
        masks.append(is_training & ~near)
    return masks


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
    from non_local_detector.likelihoods.no_spike import (
        predict_no_spike_log_likelihood,
    )

    record: dict[str, Any] = {}
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

    blocks = time_blocks(time, n_folds)
    masks = fold_training_masks(is_training, blocks, n_folds)
    encodings = [
        refit(detector, time, head_position, spike_times, marks, mask) for mask in masks
    ]
    record["refit_s"] = round(clock.time() - started, 1)
    log(f"refit {n_folds} encodings: {record['refit_s']} s")

    t0 = clock.time()
    terms = fold_terms(encodings, spike_times, marks, time, centers, blocks)
    del encodings  # their terms are all the scoring needs
    record["terms_s"] = round(clock.time() - t0, 1)
    record["n_spikes"] = int(terms.time_bin.size)
    log(f"terms: {terms.time_bin.size} spikes, {record['terms_s']} s")

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
