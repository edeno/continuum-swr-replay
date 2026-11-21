"""Is the non-local activity of rest the activity of sharp-wave ripples? Three
decoders that differ only in when the non-local states may be occupied, scored
by how well each predicts each bin's spikes.

The models share one decoder (in each fold, the decoder fit on the other folds,
below): its encoding, its continuous transitions ``C`` and its fitted discrete
transitions ``D`` (no per-model refit). They differ only in the discrete states
allowed at each time bin, ``a_t`` in {0, 1}^S:

- ``none``: the local states (Local and No-Spike) only, at every bin;
- ``swr``: an SWR is non-local activity. Inside an SWR bin (not padded) only
  the non-local states, outside SWRs only the local states: the SWR times set
  the state;
- ``full``: every state at every bin (the paper's decoder).

**The restricted model.** State bins s = 1..N (the track-interior bins of every
state), with discrete state ``d(s)``. The library's joint transition is
``T[s', s] = C[s', s] D[d(s'), d(s)]`` (row: the bin at t - 1). Where a state is
forbidden at t, the transitions into it are removed and the rows of the
*discrete* matrix renormalized over the allowed destinations:

- ``D~_t[i, j] = D[i, j] a_t[j] / sum_k D[i, k] a_t[k]``;
- ``T~_t[s', s] = C[s', s] D~_t[d(s'), d(s)]``, ``a_t`` the allowed set at the
  destination time t.

So at an SWR's first bin ``swr`` enters a non-local state with probability 1,
split between Continuous and Fragmented as the row's renormalized entries say,
and at the first bin after it returns to Local or No-Spike the same way. A row
with no allowed destination is an error. With every state allowed, ``D~ = D``
and the model is the library's decoder. Each restricted model is a hidden
Markov model whose transitions depend on a known input (the SWR indicator). It
is a proper one (rows of ``T~_t`` sum to 1) because each block of ``C``, the
columns of one destination state, has rows that sum to 1
(:func:`continuous_block_deviation` reports how far they are from it).

**Reading the contrasts.** ``swr - none`` is what calling every SWR non-local
buys. ``full`` and ``swr`` differ in two places: inside SWRs ``swr`` must be
non-local where ``full`` may stay local, and outside SWRs ``swr`` cannot be
non-local where ``full`` may be. ``full - swr`` over a set of bins mixes the
two, so it is read per category of bins (:func:`restriction_sums`): over bins
inside SWRs it is the cost of forcing the state, over bins outside them what
the non-local states buy there. The share of the non-local gain that lies
outside SWRs (:func:`share_outside_swr`) is taken from ``full - none`` alone.
Allowing a state is not free: a model that may leave Local puts predictive
mass on the non-local states, which costs it wherever activity stays local.

**Two scores**, both read from the library's filter and smoother
(``chunked_filter_smoother_covariate_dependent``, started from
``pi a_0[d(.)]`` renormalized; this module has no recursion of its own). With
``pred_t`` the predictive posterior, ``gamma_t`` the smoothed one and ``L_t =
exp(ll[t])``:

- the filter score, ``log p(y_t | y_1..y_{t-1}) = log sum_s pred_t[s] L_t[s]``.
  The per-bin scores add up to the model's log marginal likelihood of the
  epoch, so a sum over any set of bins is that set's share of it;
- the two-sided score, ``log p(y_t | y_{-t})``: the bin's spikes predicted from
  every other bin, earlier and later. With ``r_t[s] = sum_{s'} T~_{t+1}[s, s']
  gamma_{t+1}[s'] / pred_{t+1}[s']`` (the factor of the smoother's step),
  ``gamma_t = pred_t L_t r_t / c_t`` with ``c_t`` the filter score's
  probability, and ``p(x_t | y_{-t})`` is proportional to ``pred_t r_t``, so
  ``p(y_t | y_{-t}) = c_t / sum_s pred_t[s] r_t[s]`` and ``sum_s pred_t[s]
  r_t[s] = sum_s q_{t+1}[s] gamma_{t+1}[s] / pred_{t+1}[s]``, ``q_{t+1} = pred_t
  T~_{t+1}``. The term of a state bin with ``pred_{t+1}[s] = 0`` is dropped.
  That is exact: ``gamma_{t+1}`` is proportional to ``pred_{t+1}``, so it is 0
  there too, and ``q_{t+1}[s]`` is 0 wherever the bin is forbidden at t + 1;
  only a predicted mass that underflows to 0 in float32 where the exact one is
  positive (and its smoothed mass with it) is lost. The last bin's two-sided
  score is its filter score. The two-sided scores do not add up to a
  likelihood.

**Holding out the whole decoder** (:func:`fold_fit_scores`). The epoch's bins
are split into :data:`~.tetrode_holdout.N_FOLDS` folds; the runner's folds
are whole trials balanced by destination well
(:func:`.tetrode_holdout.trial_folds`), with a margin of
:data:`~.tetrode_holdout.TRIAL_MARGIN_S` (the default without folds is
contiguous blocks of time). The time before the first trial and after the last
are segments of their own two classes, and a segment longer than
:data:`~.tetrode_holdout.MAX_SEGMENT_S` is cut into equal pieces, so a long stay
is spread over the folds. For each fold k, a detector configured as the
canonical one (:func:`fresh_detector`, never a copy of the fitted one) is fit
by the library's own EM with the canonical decode's call and EM settings
(:func:`fit_fold_detector`), its encoding trained on the decode's training bins
outside fold k and its margin (:func:`.tetrode_holdout.fold_training_masks`)
and every bin of fold k missing, so that no spike of fold k reaches its
encoding, ``D`` or ``pi``. Every bin's likelihood ``ll`` is then the library's
own under that fit, with every tetrode in the model (:func:`fold_log_likelihood`),
the three models are run as above with its ``D`` and ``pi``, and fold k's bins
keep that pass's scores, with the ``full`` model's smoothed state probabilities
and most likely non-local place bin. ``C`` has nothing EM fits (its blocks are
discrete, uniform and random-walk kernels on the track), so it is the same in
every fit. The other folds' fits do see fold k (it is their training data), and
each pass's smoother runs over the whole epoch with fold k observed: what is
held out of fold k's scores is fold k's spikes from the model that scores them.
The library's M-step refit of the encoding is skipped in the fold fits
(:data:`FOLD_FIT_ESTIMATE_ENCODING_MODEL`): in the pinned library it refits the
same encoding, so it changes nothing but the time. An earlier version held out
only the encodings and ran the models with the canonical decoder's ``D`` and
``pi``, fit on the whole epoch, so its transitions had seen the bins they
scored.

In this task the animal stays in one patch for long stretches, so place and
time are confounded: a place visited in only one fold has no data in that
fold's encoding, the Local state cannot explain its spikes there, and ``full``
gains by explaining them with place fields from elsewhere. Each bin's
``place_sampling_s`` records how much training data its fold's encoding had at
the animal's place. A missing bin's likelihood is 0, so it scores 0 on both, to
float32 rounding of the posterior's mass.

What the score does not show: that the activity it credits is coherent spatial
content. Any activity the Local state predicts poorly and a non-local state
predicts less poorly is credited: in simulation
(``tests/integration/test_nonlocal_restriction_simulated.py``) a burst whose
tetrodes each follow their own random place gains over ``none`` about as much
as one whose tetrodes share a random place, because at 2 ms a bin holds about
one spike and the Non-Local Fragmented state can follow each spike's place.
SWRs are not padded, so non-local activity that begins before its SWR or ends
after it is counted outside SWRs. Nor does it show anything about a transition
matrix other than the fitted one, which each fold's EM fits on the other folds
with every state allowed. Holding out costs data: with the trial folds' margin
each fold's encoding is fit on about 75 % of the decode's training bins (about
80 % with contiguous blocks). And the comparison is biased against the
non-local states outside SWRs: the decoder trains on every bin outside SWRs, so
non-local activity outside SWRs at a well, from the other stays there, is in the
data the Local state's encoding at that well is fit on, while spikes inside SWRs
never are; the Local state explains part of that activity, so ``full - swr``
outside SWRs and the share outside SWRs are understated, not overstated.
"""

from __future__ import annotations

import time as clock
from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from continuum_swr_replay.analysis.event_definition.analyzed_time import (
    analyzed_bins,
    immobile_analyzed_bins,
)
from continuum_swr_replay.analysis.event_evidence.state_evidence.terms import (
    interior_bin_mask,
)
from continuum_swr_replay.analysis.event_evidence.tetrode_holdout import (
    LOCAL,
    MARGIN_BINS,
    N_FOLDS,
    NO_SPIKE,
    animal_place_bins,
    check_no_fold_leakage,
    fold_blocks,
    fold_training_masks,
    place_sampling,
)
from continuum_swr_replay.decoding.results import ENCODING_KEY
from continuum_swr_replay.intervals import sample_step
from continuum_swr_replay.parameters import DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC

MODELS = ("none", "swr", "full")
"""The three models: never non-local, non-local exactly in SWRs, unrestricted."""

LOCAL_STATES = (LOCAL, NO_SPIKE)
"""The local states; every other state is a non-local one."""

FAMILY_SUFFIX = {"filter": "", "two_sided": "_two_sided"}
"""The two scores, ``log p(y_t | y_1..y_{t-1})`` and ``log p(y_t | y_{-t})``,
and the suffix of their columns: a model's filter score is column ``<model>``,
its two-sided score ``<model>_two_sided``."""

SCORE_COLUMNS = tuple(
    f"{model}{suffix}" for suffix in FAMILY_SUFFIX.values() for model in MODELS
)
"""The six score columns, the filter scores first."""

CONTRASTS = {
    "full_minus_swr": ("full", "swr"),
    "swr_minus_none": ("swr", "none"),
    "full_minus_none": ("full", "none"),
}
"""Each contrast's name and its two models, ``a - b``."""

IMMOBILE_IN_SWR = "immobile_in_swr"
IMMOBILE_OUTSIDE_SWR = "immobile_outside_swr"
CATEGORIES = ("excluded", IMMOBILE_IN_SWR, IMMOBILE_OUTSIDE_SWR, "moving")
"""Each bin's category (:func:`bin_categories`)."""

CAUSAL_MASS_TOLERANCE = 1e-3
"""Largest allowed distance of a model's filtered posterior mass from 1."""

PREDICTED_MASS_TOLERANCE = 1e-4
"""Largest allowed distance of a model's predictive posterior mass,
``sum_s pred_t[s]``, from 1 at any bin.

A deviation d biases that bin's scores by about d nats, and differently for
each model, so it would enter the contrasts. Observed: at most 6e-7 on the GPU
and 1.4e-5 on the CPU (``full``, 100,000 bins of ``wilbur20210329_02_r1_HPC``).
The case the check is for, the library's filter run at reduced matrix-product
precision on the GPU, would be about 4e-4. The filtered posterior's mass
(:data:`CAUSAL_MASS_TOLERANCE`) does not show it: the filter renormalizes."""

CONTINUOUS_BLOCK_TOLERANCE = 1e-6
"""Largest allowed :func:`continuous_block_deviation`: rows of a destination
state's block of ``C`` that do not sum to 1 would make the restricted model
improper (5.8e-15 on ``wilbur20210329_02_r1_HPC``)."""


def bin_categories(
    frame: pd.DataFrame,
    in_swr: NDArray[np.bool_],
    speed_threshold: float = DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
) -> pd.Categorical:
    """Each bin's category: excluded, immobile in or outside an SWR, or moving.

    Parameters
    ----------
    frame : pd.DataFrame
        A decode frame read through ``analyzed_time.read_dot_with_swr_source``
        (``in_excluded_time`` and ``head_speed``), one row per bin.
    in_swr : ndarray of bool, shape (n_time,)
        Whether each bin lies inside an SWR.
    speed_threshold : float
        cm/s; a bin is immobile when its speed is strictly below it
        (``analyzed_time.immobile_analyzed_bins``).

    Returns
    -------
    pd.Categorical, length n_time
        Categories :data:`CATEGORIES`: ``excluded`` where the bin is not
        analyzed; else :data:`IMMOBILE_IN_SWR` / :data:`IMMOBILE_OUTSIDE_SWR`
        where it is immobile; else ``moving`` (a bin without a speed included).
    """
    in_swr = np.asarray(in_swr, dtype=bool)
    if in_swr.shape != (len(frame),):
        raise ValueError(f"in_swr has shape {in_swr.shape}, not ({len(frame)},)")
    analyzed = analyzed_bins(frame)
    immobile = immobile_analyzed_bins(frame, speed_threshold)
    labels = np.select(
        [~analyzed, immobile & in_swr, immobile],
        ["excluded", IMMOBILE_IN_SWR, IMMOBILE_OUTSIDE_SWR],
        default="moving",
    )
    return pd.Categorical(labels, categories=CATEGORIES)


def allowed_states(
    state_names: Sequence[str], in_swr: NDArray
) -> dict[str, NDArray[np.bool_]]:
    """Each model's allowed discrete states at each bin.

    Parameters
    ----------
    state_names : sequence of str, length n_states
    in_swr : ndarray of bool, shape (n_time,)

    Returns
    -------
    dict of str to ndarray of bool, shape (n_time, n_states)
        ``none`` (the local states at every bin), ``swr`` (the non-local
        states inside SWRs, the local states outside them) and ``full`` (every
        state at every bin) (:data:`MODELS`).
    """
    in_swr = np.asarray(in_swr, dtype=bool)
    local = np.isin(np.asarray(state_names), LOCAL_STATES)
    every = np.ones(local.size, dtype=bool)
    return {
        "none": np.broadcast_to(local, (in_swr.size, local.size)),
        "swr": np.where(in_swr[:, None], ~local, local),
        "full": np.broadcast_to(every, (in_swr.size, local.size)),
    }


def _row_normalizers(discrete: NDArray, patterns: NDArray) -> NDArray:
    """``sum_k D[i, k] a[k]`` for every pattern and row, shape (n_patterns,
    n_states); raises where a row has no allowed destination."""
    normalizers = np.asarray(patterns, dtype=float) @ np.asarray(discrete).T
    empty = np.argwhere(normalizers <= 0)
    if empty.size:
        p, i = empty[0]
        raise ValueError(
            f"allowed pattern {np.asarray(patterns, dtype=int)[p].tolist()} leaves "
            f"discrete state {int(i)} with no allowed destination"
        )
    return normalizers


def restricted_discrete_transitions(discrete: NDArray, patterns: NDArray) -> NDArray:
    """The renormalized discrete transitions of each allowed pattern,
    ``D~[p, i, j] = D[i, j] a_p[j] / sum_k D[i, k] a_p[k]``.

    Parameters
    ----------
    discrete : ndarray, shape (n_states, n_states)
        ``D[i, j]``, from state i to state j.
    patterns : ndarray of bool, shape (n_patterns, n_states)
        Each pattern's allowed destination states.

    Returns
    -------
    ndarray, shape (n_patterns, n_states, n_states)

    Raises
    ------
    ValueError
        When a pattern leaves a row of ``D`` with no allowed destination.
    """
    discrete = np.asarray(discrete, dtype=float)
    patterns = np.asarray(patterns, dtype=bool)
    normalizers = _row_normalizers(discrete, patterns)
    return discrete[None] * patterns[:, None, :] / normalizers[:, :, None]


def continuous_block_deviation(continuous: NDArray, state_ind: NDArray) -> float:
    """How far the rows of each destination state's block of ``C`` are from
    summing to 1: ``max |sum_{s: d(s) = j} C[s', s] - 1|`` over rows s' and
    states j.

    Parameters
    ----------
    continuous : ndarray, shape (n_state_bins, n_state_bins)
    state_ind : ndarray of int, shape (n_state_bins,)
    """
    continuous = np.asarray(continuous, dtype=float)
    state_ind = np.asarray(state_ind)
    sums = continuous @ (state_ind[:, None] == np.unique(state_ind)[None, :])
    return float(np.abs(sums - 1.0).max())


#: Bins per batch of the two-sided step.
CHUNK_BINS = 20_000

IDENTITY_SAMPLE_BINS = 2_000
"""Bins, evenly spaced over the epoch and the first and last among them, on which
:func:`filter_identity_deviation` checks the library's filter."""

FILTER_IDENTITY_TOLERANCE = 1e-4
"""Largest allowed :func:`filter_identity_deviation`: float32 gives about 1e-6,
a swapped or misaligned array a difference of order 1."""


def filter_identity_deviation(
    causal: NDArray, predicted: NDArray, log_likelihood: NDArray
) -> float:
    """How far the library's filtered posterior is from its own update of the
    predictive posterior by the likelihood, ``causal_t = pred_t exp(ll_t)``
    renormalized, at the given bins: the largest absolute difference.

    It checks the wiring per bin, where float32 does not accumulate: the order
    of the arrays the library returns and the alignment of the likelihood with
    the posteriors. Computed in float64.

    Parameters
    ----------
    causal, predicted : ndarray, shape (n_bins, n_state_bins)
        The library's filtered and predictive posteriors at the bins.
    log_likelihood : ndarray, shape (n_bins, n_state_bins)
        The likelihood the library was given at the bins (forbidden state bins
        at -inf). A missing bin, a row of zeros, gives ``causal = pred``.

    Returns
    -------
    float
        NaN when a bin has no predicted mass where its likelihood is finite.
    """
    ll = np.asarray(log_likelihood, dtype=np.float64)
    pred = np.asarray(predicted, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        top = np.where(pred > 0, ll, -np.inf).max(axis=1, keepdims=True)
        weights = pred * np.exp(ll - top)
        expected = weights / weights.sum(axis=1, keepdims=True)
    difference = np.abs(np.asarray(causal, dtype=np.float64) - expected)
    return float(difference.max())


@jax.jit
def _bin_scores(
    predicted: jnp.ndarray,
    log_likelihood: jnp.ndarray,
    predicted_next: jnp.ndarray,
    smoothed_next: jnp.ndarray,
    next_pattern: jnp.ndarray,
    joint: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """A batch of bins' filter and two-sided scores from the library's
    posteriors.

    Parameters
    ----------
    predicted, log_likelihood : jnp.ndarray, shape (n_batch, n_state_bins)
        ``pred[t]`` and ``ll[t]``.
    predicted_next, smoothed_next : jnp.ndarray, shape (n_batch, n_state_bins)
        ``pred[t + 1]`` and ``gamma[t + 1]``.
    next_pattern : jnp.ndarray of int, shape (n_batch,)
        The allowed pattern of bin t + 1.
    joint : jnp.ndarray, shape (n_patterns, n_state_bins, n_state_bins)
        ``T~`` of each pattern.

    Returns
    -------
    filter_score, two_sided : jnp.ndarray, shape (n_batch,)
    """
    has_mass = predicted > 0
    terms = jnp.where(
        has_mass,
        jnp.log(jnp.where(has_mass, predicted, 1.0)) + log_likelihood,
        -jnp.inf,
    )
    top = terms.max(axis=1, keepdims=True)
    filter_score = jnp.log(jnp.exp(terms - top).sum(axis=1)) + top[:, 0]
    # q[t + 1] = pred[t] @ T~_{t+1}: the prediction of bin t + 1 that skips
    # bin t's spikes
    # at full float32 precision: at the GPU's default (reduced) precision this
    # product shifted the two-sided scores by about 5e-4 nats a bin, by a
    # different amount for each model, which moved a two-sided model contrast
    # of a real epoch by about 1 % (the filter score was unaffected)
    skipped = jnp.zeros_like(predicted)
    for p in range(joint.shape[0]):
        skipped = skipped + (next_pattern == p)[:, None] * jnp.matmul(
            predicted, joint[p], precision=jax.lax.Precision.HIGHEST
        )
    next_mass = predicted_next > 0
    ratio = jnp.where(
        next_mass, smoothed_next / jnp.where(next_mass, predicted_next, 1.0), 0.0
    )
    return filter_score, filter_score - jnp.log((skipped * ratio).sum(axis=1))


@jax.jit
def _summary_batch(
    smoothed: jnp.ndarray, membership: jnp.ndarray, place_columns: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """A batch of bins' smoothed state probabilities and the place bin with the
    most summed smoothed mass over the states of ``place_columns``.

    Parameters
    ----------
    smoothed : jnp.ndarray, shape (n_batch, n_state_bins)
    membership : jnp.ndarray, shape (n_state_bins, n_states)
        1 where a state bin is of a state.
    place_columns : jnp.ndarray of int, shape (n_summed_states, n_place_bins)
        Each summed state's state bins, in place-bin order.
    """
    # at full float32 precision, as in _bin_scores
    probability = jnp.matmul(smoothed, membership, precision=jax.lax.Precision.HIGHEST)
    return probability, jnp.argmax(smoothed[:, place_columns].sum(axis=1), axis=1)


def smoothed_summary(
    smoothed: Any,
    state_ind: NDArray,
    summed_states: Sequence[int],
    rows: NDArray,
    *,
    chunk_bins: int = CHUNK_BINS,
) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
    """At the bins ``rows``, each discrete state's smoothed probability and the
    most likely place bin of the states ``summed_states`` together.

    Parameters
    ----------
    smoothed : array, shape (n_time, n_state_bins)
        The smoothed (acausal) posterior over the track-interior state bins.
    state_ind : ndarray of int, shape (n_state_bins,)
        Each state bin's discrete state, 0 .. n_states - 1.
    summed_states : sequence of int
        The states whose smoothed mass is summed place bin by place bin (the
        two non-local states); each has the same place bins, in the same order.
    rows : ndarray of int, shape (n_rows,)
    chunk_bins : int

    Returns
    -------
    probability : ndarray of float32, shape (n_rows, n_states)
        ``sum_{s: d(s) = j} gamma_t[s]``.
    place_bin : ndarray of int32, shape (n_rows,)
        ``argmax_x sum_{j in summed_states} gamma_t[j, x]``, an index into the
        summed states' place bins (the track-interior bins).

    Raises
    ------
    ValueError
        When the summed states differ in their number of place bins.
    """
    state_ind = np.asarray(state_ind)
    n_states = int(state_ind.max()) + 1
    columns = [np.flatnonzero(state_ind == s) for s in summed_states]
    if len({c.size for c in columns}) != 1:
        raise ValueError(
            f"states {list(summed_states)} have {[c.size for c in columns]} "
            "place bins, not one number"
        )
    membership = jnp.asarray(
        state_ind[:, None] == np.arange(n_states)[None, :], dtype=jnp.float32
    )
    place_columns = jnp.asarray(np.stack(columns))
    rows = np.asarray(rows, dtype=np.int64)
    probability = np.empty((rows.size, n_states), dtype=np.float32)
    place_bin = np.empty(rows.size, dtype=np.int32)
    for start in range(0, rows.size, chunk_bins):
        stop = min(start + chunk_bins, rows.size)
        # index where the posterior is (the host, from the library), so that
        # only the batch is copied to the device
        batch = jnp.asarray(smoothed[rows[start:stop]])
        p, x = _summary_batch(batch, membership, place_columns)
        probability[start:stop] = np.asarray(p)
        place_bin[start:stop] = np.asarray(x)
    return probability, place_bin


def predictive_log_scores(
    log_likelihood: NDArray,
    initial: NDArray,
    continuous: NDArray,
    discrete: NDArray,
    state_ind: NDArray,
    allowed: NDArray,
    *,
    chunk_bins: int = CHUNK_BINS,
    summary_rows: NDArray | None = None,
    summed_states: Sequence[int] = (),
) -> tuple[NDArray[np.float32], NDArray[np.float32], dict]:
    """Each bin's filter and two-sided scores under the model that allows the
    states ``allowed`` at each bin, from the library's filter and smoother.

    Parameters
    ----------
    log_likelihood : ndarray, shape (n_time, n_state_bins)
        ``ll[t, s]``; a row of zeros (a missing bin) scores 0 on both, to
        float32 rounding.
    initial : ndarray, shape (n_state_bins,)
        ``pi``, restricted to the states allowed at the first bin and
        renormalized.
    continuous : ndarray, shape (n_state_bins, n_state_bins)
        ``C[s', s]``, from bin s' to bin s.
    discrete : ndarray, shape (n_states, n_states)
        ``D[i, j]``.
    state_ind : ndarray of int, shape (n_state_bins,)
        Each state bin's discrete state.
    allowed : ndarray of bool, shape (n_time, n_states)
        ``a_t``: the states that may be occupied at bin t.
    chunk_bins : int
        Bins per batch of the two-sided step.
    summary_rows : ndarray of int, shape (n_rows,), optional
        Given, the record also has the smoothed posterior's summary at these
        bins (:func:`smoothed_summary`): ``state_probability`` (n_rows,
        n_states) and ``place_bin`` (n_rows,). It changes no score.
    summed_states : sequence of int
        The states whose place bins ``place_bin`` sums (the non-local ones).

    Returns
    -------
    filter_score : ndarray of float32, shape (n_time,)
        ``log p(y_t | y_1..y_{t-1})``.
    two_sided : ndarray of float32, shape (n_time,)
        ``log p(y_t | y_{-t})``.
    record : dict
        ``library_s``, ``two_sided_s``; ``marginal`` (the library's log
        marginal likelihood), ``filter_score_sum`` (the float64 sum of the
        filter scores) and ``marginal_difference`` (``marginal`` minus
        ``filter_score_sum``), recorded and not checked: the pinned library
        (non_local_detector 2d730c3a in ``uv.lock``) adds its per-bin terms into
        a float32 total whose error grows with the number of bins, about 8e-4
        of the total at 654,000 bins (``wilbur20210329_02_r1_HPC``) and about
        6e-3 at 1.94 million (``chimi20200213_06_r3_HPC``), by a different
        amount for each model, so no contrast uses it (upstream main fixed the
        total, 23369aa3, 2026-11-11); ``filter_identity_deviation``
        (:func:`filter_identity_deviation` on :data:`IDENTITY_SAMPLE_BINS`
        bins), ``causal_mass_deviation`` (the largest distance of a filtered
        posterior's sum from 1) and ``predicted_mass_deviation`` (the largest
        distance of a predictive posterior's sum from 1, summed in float64).

    Raises
    ------
    ValueError
        When ``allowed`` and ``log_likelihood`` differ in their number of bins,
        ``discrete`` is not 2-D, a bin's allowed states leave a row of ``D``
        with no allowed destination, or none of the first bin's allowed states
        has initial mass.
    """
    from non_local_detector.core import chunked_filter_smoother_covariate_dependent

    if len(allowed) != len(log_likelihood):
        raise ValueError(
            f"allowed has {len(allowed)} bins, log_likelihood {len(log_likelihood)}"
        )
    if np.ndim(discrete) != 2:
        raise ValueError(f"discrete must be 2-D, not {np.ndim(discrete)}-D")
    record: dict[str, Any] = {}
    state_ind = np.asarray(state_ind)
    patterns, pattern = np.unique(
        np.asarray(allowed, dtype=bool), axis=0, return_inverse=True
    )
    pattern = pattern.ravel()
    n_time = pattern.size
    restricted = restricted_discrete_transitions(discrete, patterns)
    destination = patterns[:, state_ind]
    first = np.asarray(initial, dtype=float) * destination[pattern[0]]
    if first.sum() <= 0:
        raise ValueError("no allowed state of the first bin has initial mass")
    # the library pairs transition t with the move from bin t to bin t + 1, so
    # index t holds D~ of the allowed set at t + 1; the last is unused
    next_pattern = np.r_[pattern[1:], pattern[-1]]
    log_likelihood = jnp.asarray(log_likelihood, dtype=jnp.float32)
    # A forbidden bin holds no predicted mass, so its likelihood is not part of
    # the model; it is set to -inf so that the library subtracts the largest
    # likelihood of a bin that can hold mass before exponentiating. The pinned
    # library (non_local_detector 2d730c3a) adds 1e-15 in its normalization,
    # which loses mass when every bin with mass is far below the subtracted
    # maximum; upstream main fixed that (23369aa3, 2026-11-11), and the mask is
    # harmless after an upgrade.
    if not patterns.all():
        masked = jnp.where(
            jnp.asarray(destination)[jnp.asarray(pattern)], log_likelihood, -jnp.inf
        )
    else:
        masked = log_likelihood
    # The first bin's predicted mass is the restricted initial distribution, so
    # a state bin it gives no mass is not part of the model there either: its
    # likelihood is set to -inf for the same reason. Otherwise, when the animal
    # starts where the Local likelihood is far below a non-local one that has
    # no initial mass, the pinned library's 1e-15 leaves that bin's filtered
    # mass far below 1 and its scores off by its log (16 nats at the first bin
    # of chimi20200223_04_r2_HPC).
    masked = masked.at[0].set(jnp.where(jnp.asarray(first > 0), masked[0], -jnp.inf))

    started = clock.time()
    (smoothed, _, marginal, _, _, _, causal, predicted) = (
        chunked_filter_smoother_covariate_dependent(
            time=np.arange(n_time),
            state_ind=state_ind,
            initial_distribution=first / first.sum(),
            discrete_transition_matrix=restricted[next_pattern],
            continuous_transition_matrix=np.asarray(continuous),
            log_likelihood_func=None,
            log_likelihood_args=(),
            log_likelihoods=masked,
        )
    )
    record["library_s"] = round(clock.time() - started, 2)
    sample = np.unique(np.linspace(0, n_time - 1, IDENTITY_SAMPLE_BINS).round())
    sample = sample.astype(np.int64)
    record["filter_identity_deviation"] = filter_identity_deviation(
        causal[sample], predicted[sample], masked[sample]
    )
    del masked
    record["causal_mass_deviation"] = float(np.abs(causal.sum(axis=1) - 1.0).max())
    del causal

    started = clock.time()
    joint = jnp.asarray(
        np.asarray(continuous)[None] * restricted[:, state_ind][:, :, state_ind],
        dtype=jnp.float32,
    )
    filter_score = np.empty(n_time, dtype=np.float32)
    two_sided = np.empty(n_time, dtype=np.float32)
    predicted_mass_deviation = 0.0
    for start in range(0, n_time, chunk_bins):
        stop = min(start + chunk_bins, n_time)
        chunk = np.asarray(predicted[start:stop])
        predicted_mass_deviation = max(
            predicted_mass_deviation,
            float(np.abs(chunk.astype(np.float64).sum(axis=1) - 1.0).max()),
        )
        # the next bin of the last bin is a stand-in; its two-sided score is set
        # below
        following = np.minimum(np.arange(start, stop) + 1, n_time - 1)
        scores = _bin_scores(
            jnp.asarray(chunk),
            log_likelihood[start:stop],
            jnp.asarray(predicted[following]),
            jnp.asarray(smoothed[following]),
            jnp.asarray(pattern[following]),
            joint,
        )
        filter_score[start:stop], two_sided[start:stop] = (
            np.asarray(s) for s in scores
        )
    two_sided[-1] = filter_score[-1]
    record["two_sided_s"] = round(clock.time() - started, 2)
    record["predicted_mass_deviation"] = predicted_mass_deviation
    if summary_rows is not None:
        record["state_probability"], record["place_bin"] = smoothed_summary(
            smoothed, state_ind, summed_states, summary_rows, chunk_bins=chunk_bins
        )
    del smoothed, predicted

    total = float(filter_score.astype(np.float64).sum())
    record["marginal"] = float(marginal)
    record["filter_score_sum"] = total
    record["marginal_difference"] = float(marginal) - total
    return filter_score, two_sided, record


#: Each per-model key of :func:`predictive_log_scores`' record and its name in
#: :func:`model_scores`' record.
_MODEL_RECORD_KEYS = {
    "library_s": "library_{}_s",
    "two_sided_s": "two_sided_{}_s",
    "marginal": "marginal_log_likelihood_{}",
    "filter_score_sum": "filter_score_sum_{}",
    "marginal_difference": "marginal_difference_{}",
    "filter_identity_deviation": "filter_identity_deviation_{}",
    "causal_mass_deviation": "causal_mass_deviation_{}",
    "predicted_mass_deviation": "predicted_mass_deviation_{}",
}


STATE_PROBABILITY_COLUMNS = {
    LOCAL: "full_prob_local",
    NO_SPIKE: "full_prob_no_spike",
    "Non-Local Continuous": "full_prob_nonlocal_continuous",
    "Non-Local Fragmented": "full_prob_nonlocal_fragmented",
}
"""Each discrete state's column of the ``full`` model's smoothed (acausal)
probability of the state, in the fold-fit table."""

NONLOCAL_PLACE_BIN_COLUMN = "full_nonlocal_place_bin"
"""The fold-fit table's column of the ``full`` model's most likely non-local
place bin: the argmax over the track-interior place bins (junction placeholders
excluded) of the summed smoothed mass of the two non-local states, an index
into the interior bins (``is_track_interior_`` order)."""


def model_scores(
    detector: Any,
    log_likelihood: NDArray,
    in_swr: NDArray,
    *,
    models: Sequence[str] = MODELS,
    summary_bins: NDArray | None = None,
    log: Callable[[str], None] = print,
) -> tuple[dict[str, NDArray], dict]:
    """Every bin's scores under the three models (or ``models``), on a given
    likelihood.

    The decoder's ``C``, ``D``, ``pi`` and state bins are read from
    ``detector``, restricted to its track-interior state bins.

    Parameters
    ----------
    detector : NonLocalClusterlessDetector
        The epoch's fitted decoder (its states and transitions).
    log_likelihood : ndarray, shape (n_time, n_state_bins)
        ``ll[t, s]`` (:func:`fold_log_likelihood`).
    in_swr : ndarray of bool, shape (n_time,)
        The bins inside an SWR, where ``swr`` allows only the non-local states.
    models : sequence of str
        The models to score, of :data:`MODELS`.
    summary_bins : ndarray of bool, shape (n_time,), optional
        Given (with ``full`` among ``models``), ``scores`` also has the ``full``
        model's smoothed state probabilities (:data:`STATE_PROBABILITY_COLUMNS`,
        float32, NaN outside these bins) and its most likely non-local place bin
        (:data:`NONLOCAL_PLACE_BIN_COLUMN`, int32, -1 outside them).
    log : callable

    Returns
    -------
    scores : dict of str to ndarray, shape (n_time,)
        Each model's columns of :data:`SCORE_COLUMNS` (float32), and the
        summary's columns with ``summary_bins``.
    record : dict
        ``continuous_block_deviation``; per model the library's
        ``marginal_log_likelihood_<model>`` as it returned it, the
        ``filter_score_sum_<model>``, their ``marginal_difference_<model>``,
        ``filter_identity_deviation_<model>``,
        ``causal_mass_deviation_<model>``, ``predicted_mass_deviation_<model>``,
        ``library_<model>_s`` and ``two_sided_<model>_s``.
    """
    likelihood = jnp.asarray(log_likelihood)  # on the device once for the models
    interior = np.asarray(detector.is_track_interior_state_bins_)
    state_ind = np.asarray(detector.state_ind_)[interior]
    continuous = np.asarray(detector.continuous_state_transitions_)[
        np.ix_(interior, interior)
    ]
    discrete = np.asarray(detector.discrete_state_transitions_)
    initial = np.asarray(detector.initial_conditions_)[interior]
    record: dict[str, Any] = {
        "continuous_block_deviation": continuous_block_deviation(continuous, state_ind)
    }
    scores: dict[str, NDArray] = {}
    allowed = allowed_states(detector.state_names, in_swr)
    state_names = list(detector.state_names)
    if summary_bins is not None:
        if "full" not in models:
            raise ValueError("the summary is the full model's; full is not scored")
        unknown = set(state_names) - set(STATE_PROBABILITY_COLUMNS)
        if unknown:
            raise ValueError(f"no state probability column for states {unknown}")
        summary_bins = np.asarray(summary_bins, dtype=bool)
        if summary_bins.shape != (len(log_likelihood),):
            raise ValueError(
                f"summary_bins has shape {summary_bins.shape}, not "
                f"({len(log_likelihood)},)"
            )
        nonlocal_states = [
            s for s, name in enumerate(state_names) if name not in LOCAL_STATES
        ]
        # the state bins are the track-interior bins, which must be the real
        # place bins (no junction placeholder)
        environment = detector.encoding_model_[ENCODING_KEY]["environment"]
        n_real = int(interior_bin_mask(environment).sum())
        for s in nonlocal_states:
            if int((state_ind == s).sum()) != n_real:
                raise ValueError(
                    f"state {state_names[s]!r} has {int((state_ind == s).sum())} "
                    f"state bins, not the {n_real} real place bins"
                )
    for model in models:
        summarized = summary_bins is not None and model == "full"
        filter_score, two_sided, part = predictive_log_scores(
            likelihood,
            initial,
            continuous,
            discrete,
            state_ind,
            allowed[model],
            **(
                {
                    "summary_rows": np.flatnonzero(summary_bins),
                    "summed_states": nonlocal_states,
                }
                if summarized
                else {}
            ),
        )
        scores[f"{model}{FAMILY_SUFFIX['filter']}"] = filter_score
        scores[f"{model}{FAMILY_SUFFIX['two_sided']}"] = two_sided
        if summarized:
            for s, name in enumerate(state_names):
                column = np.full(summary_bins.size, np.nan, dtype=np.float32)
                column[summary_bins] = part["state_probability"][:, s]
                scores[STATE_PROBABILITY_COLUMNS[name]] = column
            place = np.full(summary_bins.size, -1, dtype=np.int32)
            place[summary_bins] = part["place_bin"]
            scores[NONLOCAL_PLACE_BIN_COLUMN] = place
        for key, name in _MODEL_RECORD_KEYS.items():
            record[name.format(model)] = part[key]
        log(
            f"{model}: library {part['library_s']} s, two-sided {part['two_sided_s']} s"
        )
    return scores, record


def check_failures(record: dict) -> list[str]:
    """The model checks a record of :func:`model_scores` (one fold's pass of
    :func:`fold_fit_scores`) fails, as messages.

    A model's filtered posterior must be the library's own update of its
    predictive posterior (``filter_identity_deviation_<model>`` within
    :data:`FILTER_IDENTITY_TOLERANCE`), its filtered posterior mass must be within
    :data:`CAUSAL_MASS_TOLERANCE` of 1 and its predictive posterior mass within
    :data:`PREDICTED_MASS_TOLERANCE`; the blocks of ``C`` must have rows that sum
    to 1 within :data:`CONTINUOUS_BLOCK_TOLERANCE`.
    """
    failures = []
    deviation = record["continuous_block_deviation"]
    if not deviation <= CONTINUOUS_BLOCK_TOLERANCE:
        failures.append(
            f"rows of the continuous transitions' blocks are {deviation} from "
            f"summing to 1 (tolerance {CONTINUOUS_BLOCK_TOLERANCE})"
        )
    for model in MODELS:
        deviation = record[f"filter_identity_deviation_{model}"]
        if not deviation <= FILTER_IDENTITY_TOLERANCE:
            failures.append(
                f"{model}: the filtered posterior is {deviation} from the update "
                f"of the predictive posterior (tolerance {FILTER_IDENTITY_TOLERANCE})"
            )
        deviation = record[f"causal_mass_deviation_{model}"]
        if not deviation <= CAUSAL_MASS_TOLERANCE:
            failures.append(
                f"{model}: filtered posterior mass is {deviation} from 1 "
                f"(tolerance {CAUSAL_MASS_TOLERANCE})"
            )
        deviation = record[f"predicted_mass_deviation_{model}"]
        if not deviation <= PREDICTED_MASS_TOLERANCE:
            failures.append(
                f"{model}: predictive posterior mass is {deviation} from 1 "
                f"(tolerance {PREDICTED_MASS_TOLERANCE})"
            )
    return failures


EM_SETTING_NAMES = (
    "estimate_initial_conditions",
    "estimate_discrete_transition",
    "max_iter",
    "tolerance",
    "cache_likelihood",
    "n_chunks",
    "min_encoding_local_mass",
    "min_encoding_local_ess",
    "encoding_update_damping",
)
"""The arguments of the library's ``estimate_parameters`` that the canonical
decode (``decoding.fit.perform_decoding``) leaves at their defaults, apart from
``estimate_encoding_model``; :func:`em_settings` reads their values from the
pinned library."""


FOLD_FIT_ESTIMATE_ENCODING_MODEL = False
"""The fold fits' ``estimate_encoding_model``; the canonical decode leaves it at
the library's default, True. The pinned library's clusterless KDE
(``fit_clusterless_kde_encoding_model``, non_local_detector 2d730c3a) takes the
Local-state weights that step passes and never reads them, so the step refits
the same encoding from the same training spikes. Measured on fold 0 of
``wilbur20210329_02_r1_HPC`` (GPU): with True and with False the encodings
(spike positions, marks, rates, occupancy, ground intensity) and initial
conditions are identical, the marginal traces identical, and the discrete
transitions differ by at most 2.9e-7 (GPU run-to-run level); the fit took 305 s
and 249 s."""

FIT = "folds"
"""The fit a restriction run's header settings and each record name
(``"fit"``): the whole decoder fit on each fold's training folds
(:func:`fold_fit_scores`). A run of the earlier design, which held out only the
encodings, records none."""


def em_settings() -> dict[str, Any]:
    """The EM settings of a fold fit: the library's defaults of
    :data:`EM_SETTING_NAMES`, as the canonical decode takes them, and
    ``estimate_encoding_model`` (:data:`FOLD_FIT_ESTIMATE_ENCODING_MODEL`)."""
    import inspect

    from non_local_detector import NonLocalClusterlessDetector

    parameters = inspect.signature(
        NonLocalClusterlessDetector.estimate_parameters
    ).parameters
    return {
        **{name: parameters[name].default for name in EM_SETTING_NAMES},
        "estimate_encoding_model": bool(FOLD_FIT_ESTIMATE_ENCODING_MODEL),
    }


def _same_setting(a: Any, b: Any) -> bool:
    """Whether two detector settings are equal: arrays by value, settings
    objects by type and their constructor fields (a dataclass) or attributes,
    sequences and mappings item by item.

    A fitted transition object also holds the environment it was made on, an
    attribute that is not a constructor field, so it is not compared."""
    import dataclasses

    if dataclasses.is_dataclass(a) and not isinstance(a, type):
        return type(a) is type(b) and all(
            _same_setting(getattr(a, f.name), getattr(b, f.name))
            for f in dataclasses.fields(a)
            if f.init
        )
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.shape(a) == np.shape(b) and np.array_equal(
            np.asarray(a), np.asarray(b)
        )
    if isinstance(a, list | tuple) and isinstance(b, list | tuple):
        return len(a) == len(b) and all(
            _same_setting(x, y) for x, y in zip(a, b, strict=True)
        )
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same_setting(a[k], b[k]) for k in a)
    if hasattr(a, "__dict__") and not isinstance(a, type):
        return type(a) is type(b) and _same_setting(vars(a), vars(b))
    return bool(a == b)


def fresh_detector(canonical: Any) -> Any:
    """An unfitted detector configured as the canonical decode's.

    Built as ``decoding.fit.perform_decoding`` builds it (the NonLocal
    clusterless detector, ``NonLocalDecodingConfig``'s algorithm parameters,
    ``decoding.transitions.nonlocal_detector_kwargs`` and an ``Environment`` of
    ``PLACE_BIN_SIZE_1D`` on the decode's track graph, edge order and spacing),
    never by copying ``canonical``: the library's EM writes its fitted discrete
    initial distribution into the detector's ``discrete_initial_conditions``
    setting, so a fit of a copy would start from the canonical fit. Every
    other setting must equal ``canonical``'s, so a configuration changed since
    the decode raises.

    Parameters
    ----------
    canonical : NonLocalClusterlessDetector
        The epoch's saved decoder.

    Raises
    ------
    ValueError
        When a setting other than ``discrete_initial_conditions`` differs from
        ``canonical``'s.
    """
    import dataclasses

    from non_local_detector import NonLocalClusterlessDetector
    from non_local_detector.environment import Environment

    from continuum_swr_replay.decoding.config import NonLocalDecodingConfig
    from continuum_swr_replay.decoding.fit import algorithm_params
    from continuum_swr_replay.decoding.transitions import nonlocal_detector_kwargs
    from continuum_swr_replay.parameters import PLACE_BIN_SIZE_1D

    (decoded,) = canonical.environments
    config = NonLocalDecodingConfig()
    environment = Environment(
        place_bin_size=PLACE_BIN_SIZE_1D,
        track_graph=decoded.track_graph,
        edge_order=decoded.edge_order,
        edge_spacing=decoded.edge_spacing,
    )
    detector = NonLocalClusterlessDetector(
        environments=environment,
        **algorithm_params(config, "clusterless"),
        **nonlocal_detector_kwargs(config),
    )
    # the environment's settings; its fitted grid (the names ending in an
    # underscore, and track_graphDD, the graph a 1-D grid without a track graph
    # is given when it is fitted) is refit by the library from the position
    differ = [
        f.name
        for f in dataclasses.fields(Environment)
        if f.init
        and not f.name.startswith("_")
        and not f.name.endswith("_")
        and f.name not in ("track_graph", "track_graphDD")
        and not _same_setting(getattr(environment, f.name), getattr(decoded, f.name))
    ]
    expected = canonical.get_params(deep=False)
    actual = detector.get_params(deep=False)
    differ += [
        name
        for name in sorted(expected.keys() | actual.keys())
        if name not in ("environments", "discrete_initial_conditions")
        and not _same_setting(expected.get(name), actual.get(name))
    ]
    if differ:
        raise ValueError(
            f"the configured detector differs from the decode's in {differ}: has "
            "the configuration changed since the decode?"
        )
    return detector


def em_record(marginal_log_likelihoods: NDArray, tolerance: float) -> dict[str, Any]:
    """The EM iterations and convergence of a fit, from its trace.

    Parameters
    ----------
    marginal_log_likelihoods : ndarray, shape (n_iterations + 1,)
        The library's trace (``results.attrs["marginal_log_likelihoods"]``):
        each iteration's E-step marginal, then the final E-step's.
    tolerance : float
        The fit's.

    Returns
    -------
    dict
        ``em_iterations``; ``converged``, as the library's loop decides it
        (``core.check_converged`` on the last two iterations' marginals;
        False when the loop stopped at ``max_iter``, which
        :func:`fold_check_failures` fails); ``marginal_log_likelihoods`` (the
        trace).
    """
    from non_local_detector.core import check_converged

    trace = np.asarray(marginal_log_likelihoods, dtype=float)
    n_iter = trace.size - 1
    converged = n_iter > 1 and bool(
        check_converged(trace[n_iter - 1], trace[n_iter - 2], tolerance)[0]
    )
    return {
        "em_iterations": int(n_iter),
        "converged": converged,
        "marginal_log_likelihoods": trace.tolist(),
    }


def fit_fold_detector(
    canonical: Any,
    time: NDArray,
    position: NDArray,
    spike_times: Sequence[NDArray],
    marks: Sequence[NDArray],
    is_training: NDArray,
    is_missing: NDArray,
) -> tuple[Any, dict]:
    """The whole decoder fit by the library's EM on some bins, as the canonical
    decode is fit.

    A :func:`fresh_detector` and the canonical decode's call
    (``decoding.fit.perform_decoding``): ``estimate_parameters`` with the
    decode's time axis as ``time`` and ``position_time``, its 2-D position, the
    spikes and marks, no covariates, and the library's defaults of
    :data:`EM_SETTING_NAMES`. The library re-initializes the discrete
    transitions from the configured ``discrete_transition_type`` and the
    initial conditions from the configured ``discrete_initial_conditions``, so
    nothing is warm-started. Only ``is_training``, ``is_missing`` and
    ``estimate_encoding_model`` (:data:`FOLD_FIT_ESTIMATE_ENCODING_MODEL`) differ
    from the decode's. The decode also
    asks for its log-likelihood back (``return_outputs``), which changes no
    parameter and is not asked for here.

    Parameters
    ----------
    canonical : NonLocalClusterlessDetector
        The epoch's saved decoder (its configuration only).
    time : ndarray, shape (n_time,)
    position : ndarray, shape (n_time, 2)
    spike_times, marks : sequence of ndarray, one per tetrode
    is_training : ndarray of bool, shape (n_time,)
        The bins the encoding is fit on.
    is_missing : ndarray of bool, shape (n_time,)
        The bins whose likelihood EM sets to 0, so that their spikes reach no
        parameter.

    Returns
    -------
    detector : NonLocalClusterlessDetector
        Fitted.
    record : dict
        ``discrete_transition`` (the fitted matrix), ``initial_discrete`` (the
        fitted initial distribution over the states), :func:`em_record`'s and
        ``fit_s``.
    """
    started = clock.time()
    settings = em_settings()
    detector = fresh_detector(canonical)
    results = detector.estimate_parameters(
        position_time=np.asarray(time),
        position=np.asarray(position),
        spike_times=list(spike_times),
        spike_waveform_features=list(marks),
        time=np.asarray(time),
        discrete_transition_covariate_data=None,
        is_training=np.asarray(is_training, dtype=bool),
        is_missing=np.asarray(is_missing, dtype=bool),
        estimate_encoding_model=FOLD_FIT_ESTIMATE_ENCODING_MODEL,
    )
    trace = np.asarray(results.attrs["marginal_log_likelihoods"])
    del results
    return detector, {
        "discrete_transition": np.asarray(detector.discrete_state_transitions_),
        "initial_discrete": np.asarray(detector.discrete_initial_conditions),
        **em_record(trace, settings["tolerance"]),
        "fit_s": round(clock.time() - started, 1),
    }


def fold_log_likelihood(
    detector: Any,
    time: NDArray,
    position: NDArray,
    spike_times: Sequence[NDArray],
    marks: Sequence[NDArray],
    is_missing: NDArray,
) -> Any:
    """Every bin's four-state log-likelihood under a fitted detector: the
    library's own (``compute_log_likelihood``), with the bins ``is_missing``
    and the bins without a position at 0, as the library's ``predict`` sets them.

    Returns
    -------
    jax array of float32, shape (n_time, n_state_bins)
    """
    position = np.asarray(position, dtype=float)
    missing = np.asarray(is_missing, dtype=bool) | np.isnan(
        position.reshape(len(position), -1)
    ).any(axis=1)
    return detector.compute_log_likelihood(
        np.asarray(time),
        np.asarray(time),
        position,
        list(spike_times),
        list(marks),
        is_missing=missing,
    )


def check_place_grid(detector: Any, canonical: Any) -> None:
    """Raise unless a fold's fitted place grid is the canonical decoder's: the
    same place-bin centers and track-interior bins.

    ``full_nonlocal_place_bin`` indexes the fold detector's interior bins, and
    ``place_sampling_s`` and every reader of the table use the canonical
    environment's.

    Raises
    ------
    ValueError
        Naming what differs.
    """
    fitted = detector.encoding_model_[ENCODING_KEY]["environment"]
    decoded = canonical.encoding_model_[ENCODING_KEY]["environment"]
    for name in ("place_bin_centers_", "is_track_interior_"):
        a, b = np.asarray(getattr(fitted, name)), np.asarray(getattr(decoded, name))
        if a.shape != b.shape or not np.array_equal(a, b):
            raise ValueError(
                f"the fold's fitted {name} is not the canonical decoder's: its "
                "place bins would not index the canonical environment's"
            )


def _device_peak_bytes() -> int | None:
    """The default device's peak memory in use since the process started, or
    None where the device does not report it (the CPU)."""
    stats = jax.local_devices()[0].memory_stats()
    return None if not stats else int(stats.get("peak_bytes_in_use", 0))


def fold_fit_scores(
    canonical: Any,
    time: NDArray,
    head_position: NDArray,
    spike_times: Sequence[NDArray],
    marks: Sequence[NDArray],
    is_training: NDArray,
    is_missing: NDArray,
    in_swr: NDArray,
    *,
    n_folds: int = N_FOLDS,
    folds: NDArray | None = None,
    margin_bins: int = MARGIN_BINS,
    linear_position: NDArray | None = None,
    fit: Callable[..., tuple[Any, dict]] = fit_fold_detector,
    log: Callable[[str], None] = print,
) -> tuple[pd.DataFrame, dict]:
    """Every bin's score under the three models, each fold's bins under a whole
    decoder fit on the other folds.

    For each fold k: the decoder is fit by EM (``fit``, by default
    :func:`fit_fold_detector`) with the encoding's training bins those of fold
    k's encoding (:func:`.tetrode_holdout.fold_training_masks`: the decode's
    training bins outside fold k and its margin) and with every bin of fold k missing, so
    that no spike of fold k reaches its encoding, transitions or initial
    conditions. Every bin's likelihood is then computed under that fit
    (:func:`fold_log_likelihood`, only the decode's missing bins missing) and
    the three models are run over the epoch with its transitions and initial
    conditions (:func:`model_scores`); fold k's bins keep that pass's scores,
    and the ``full`` model's smoothed state probabilities and most likely
    non-local place bin.

    Parameters
    ----------
    canonical : NonLocalClusterlessDetector
        The epoch's saved decoder: its configuration, and its transitions for
        the record's comparison.
    time : ndarray, shape (n_time,)
    head_position : ndarray, shape (n_time, 2)
    spike_times, marks : sequence of ndarray, one per tetrode
    is_training, is_missing : ndarray of bool, shape (n_time,)
        The decode's masks.
    in_swr : ndarray of bool, shape (n_time,)
    n_folds : int
    folds : ndarray of int, shape (n_time,), optional
        Each bin's fold (``tetrode_holdout.trial_folds``); by default
        ``n_folds`` contiguous time blocks.
    margin_bins : int
    linear_position : ndarray, shape (n_time,), optional
        Given, the table also has ``place_sampling_s``.
    fit : callable
        ``fit(canonical, time, position, spike_times, marks, is_training,
        is_missing)``, returning the fitted detector and its record.
    log : callable

    Returns
    -------
    scores : pd.DataFrame
        One row per bin: ``time``; ``none``, ``swr``, ``full`` (float32
        filter scores, ``log p(y_t | y_1..y_{t-1})``) and ``none_two_sided``,
        ``swr_two_sided``, ``full_two_sided`` (``log p(y_t | y_{-t})``), each
        bin from its own fold's pass; ``in_swr`` and ``block`` (the bin's
        fold); with ``linear_position``, ``place_sampling_s`` (float32): the
        seconds the training bins of the bin's fold's encoding spend at the
        animal's place bin (``tetrode_holdout.place_sampling``), NaN where the
        position is not finite; and the ``full`` model's :data:`STATE_PROBABILITY_COLUMNS`
        (float32) and :data:`NONLOCAL_PLACE_BIN_COLUMN` (int32).
    record : dict
        ``folds``: one record per fold (``fold``, ``n_bins``,
        ``n_training_bins``, ``fit``'s record, the largest absolute difference
        of its discrete transitions and initial distribution from the
        canonical's, ``likelihood_s``, ``scoring_s``, :func:`model_scores`'
        record and ``device_peak_bytes``, the device's peak so far, None on
        the CPU); ``fit_s``, ``likelihood_s`` and ``scoring_s`` summed over the
        folds; ``n_in_swr``, ``n_missing``, ``total_s``.

    Raises
    ------
    ValueError
        When ``in_swr`` and ``time`` differ in length, as
        :func:`predictive_log_scores`, when the folds are not one fold per bin
        in 0 .. n_folds - 1, each with bins
        (:func:`.tetrode_holdout.fold_blocks`), or when a fold's encoding would
        take spikes from one of its own bins
        (:func:`.tetrode_holdout.check_no_fold_leakage`), or when a fold's
        fitted place grid is not the canonical one (:func:`check_place_grid`).
    """
    started = clock.time()
    time = np.asarray(time, dtype=float)
    in_swr = np.asarray(in_swr, dtype=bool)
    is_missing = np.asarray(is_missing, dtype=bool)
    if in_swr.shape != time.shape:
        raise ValueError(f"in_swr has shape {in_swr.shape}, time {time.shape}")
    blocks = fold_blocks(time, n_folds, folds)
    masks = fold_training_masks(is_training, blocks, n_folds, margin_bins)
    check_no_fold_leakage(time, head_position, masks, blocks)
    canonical_discrete = np.asarray(canonical.discrete_state_transitions_)
    canonical_initial = np.asarray(canonical.discrete_initial_conditions)

    columns: dict[str, NDArray] = {}
    fold_records = []
    for k in range(n_folds):
        mine = blocks == k
        detector, fit_record = fit(
            canonical,
            time,
            head_position,
            spike_times,
            marks,
            masks[k],
            is_missing | mine,
        )
        log(
            f"fold {k}: fit {fit_record['fit_s']} s, "
            f"{fit_record['em_iterations']} EM iterations"
        )
        check_place_grid(detector, canonical)
        t0 = clock.time()
        likelihood = fold_log_likelihood(
            detector, time, head_position, spike_times, marks, is_missing
        )
        likelihood_s = round(clock.time() - t0, 1)
        t0 = clock.time()
        scores, part = model_scores(
            detector, likelihood, in_swr, summary_bins=mine, log=log
        )
        scoring_s = round(clock.time() - t0, 1)
        del likelihood
        for name, values in scores.items():
            if name not in columns:
                columns[name] = np.empty(time.size, dtype=values.dtype)
            columns[name][mine] = values[mine]
        fold_records.append(
            {
                "fold": k,
                "n_bins": int(mine.sum()),
                "n_training_bins": int(masks[k].sum()),
                **fit_record,
                "discrete_transition_difference": float(
                    np.abs(fit_record["discrete_transition"] - canonical_discrete).max()
                ),
                "initial_discrete_difference": float(
                    np.abs(fit_record["initial_discrete"] - canonical_initial).max()
                ),
                "likelihood_s": likelihood_s,
                "scoring_s": scoring_s,
                **part,
                "device_peak_bytes": _device_peak_bytes(),
            }
        )
        del detector, scores
        log(f"fold {k}: likelihood {likelihood_s} s, scoring {scoring_s} s")

    record: dict[str, Any] = {
        "folds": fold_records,
        **{
            key: round(sum(f[key] for f in fold_records), 1)
            for key in ("fit_s", "likelihood_s", "scoring_s")
        },
        "n_in_swr": int(in_swr.sum()),
        "n_missing": int(is_missing.sum()),
        "total_s": round(clock.time() - started, 1),
    }
    table = pd.DataFrame(
        {
            "time": time,
            **{column: columns[column] for column in SCORE_COLUMNS},
            "in_swr": in_swr,
            "block": blocks,
        }
    )
    if linear_position is not None:
        environment = canonical.encoding_model_[ENCODING_KEY]["environment"]
        table["place_sampling_s"] = place_sampling(
            animal_place_bins(environment, linear_position),
            masks,
            blocks,
            sample_step(time),
        )
    for column in (*STATE_PROBABILITY_COLUMNS.values(), NONLOCAL_PLACE_BIN_COLUMN):
        table[column] = columns[column]
    return table, record


def fold_check_failures(record: dict) -> list[str]:
    """The checks a :func:`fold_fit_scores` record fails, as messages, each
    named by its fold: a fold whose EM did not converge (it stopped at the
    library's ``max_iter``), and :func:`check_failures` of every fold's pass."""
    failures = []
    for fold in record["folds"]:
        if not fold["converged"]:
            failures.append(
                f"fold {fold['fold']}: EM did not converge in "
                f"{fold['em_iterations']} iterations"
            )
        failures += [f"fold {fold['fold']}: {m}" for m in check_failures(fold)]
    return failures


def restriction_sums(scores: pd.DataFrame, labels: Sequence) -> pd.DataFrame:
    """The epoch's bins and summed scores in each category of bins.

    Parameters
    ----------
    scores : pd.DataFrame
        :func:`fold_fit_scores`' table, one row per bin.
    labels : sequence, length n_time
        Each bin's category (for example ``immobile_in_swr``,
        ``immobile_outside_swr``, ``moving``, ``excluded``); a missing label
        (None or NaN) leaves the bin out of every category.

    Returns
    -------
    pd.DataFrame
        One row per category (index ``category``): ``n_bins`` and every model's
        summed score (nats) of both families, ``<model>`` (filter) and
        ``<model>_two_sided``.

    Raises
    ------
    ValueError
        When a labeled bin has a score that is not finite.
    """
    labels = pd.Series(np.asarray(labels, dtype=object), index=scores.index)
    columns = list(SCORE_COLUMNS)
    kept = labels.notna()
    not_finite = ~np.isfinite(scores.loc[kept, columns].to_numpy(np.float64))
    if not_finite.any():
        counts = {
            column: n
            for column, n in zip(columns, not_finite.sum(axis=0).tolist(), strict=True)
            if n
        }
        raise ValueError(f"labeled bins with a score that is not finite: {counts}")
    grouped = (
        scores.loc[kept, columns]
        .astype(np.float64)
        .groupby(labels[kept].rename("category"))
    )
    sums = grouped.sum()
    sums.insert(0, "n_bins", grouped.size())
    return sums


def share_outside_swr(
    sums: pd.DataFrame,
    inside: str = IMMOBILE_IN_SWR,
    outside: str = IMMOBILE_OUTSIDE_SWR,
) -> dict[str, float]:
    """Of the nats the non-local states buy over the bins of two categories,
    the part in the bins outside SWRs, both score families.

    ``(full - none)`` summed over ``outside`` divided by ``(full - none)``
    summed over ``inside`` and ``outside`` together. Only ``full`` and ``none``
    enter, so the share is a split of the non-local gain by where it lies, not
    a model contrast.

    Parameters
    ----------
    sums : pd.DataFrame
        :func:`restriction_sums`' output.
    inside, outside : str
        The categories of the bins inside and outside SWRs (by default
        :data:`IMMOBILE_IN_SWR` and :data:`IMMOBILE_OUTSIDE_SWR`); a category
        with no bins adds nothing.

    Returns
    -------
    dict
        ``share_outside_swr`` (filter) and ``share_outside_swr_two_sided``,
        NaN when the gain over both categories is not positive.
    """
    rows = sums.reindex([inside, outside]).fillna(0.0)
    out: dict[str, float] = {}
    for suffix in FAMILY_SUFFIX.values():
        gain = rows[f"full{suffix}"] - rows[f"none{suffix}"]
        out[f"share_outside_swr{suffix}"] = ratio_or_nan(
            float(gain[outside]), float(gain.sum())
        )
    return out


def ratio_or_nan(value: float, divisor: float) -> float:
    """``value / divisor``, NaN when ``divisor`` is not positive."""
    return value / divisor if divisor > 0 else np.nan
